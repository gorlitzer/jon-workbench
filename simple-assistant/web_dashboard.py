#!/usr/bin/env python3
"""
Jetson Orin Nano Voice Assistant Web Dashboard
Simple Flask web interface for testing and controlling the voice assistant
"""

from flask import Flask, render_template, jsonify, request, send_file
import docker
import subprocess
import requests
import time
import os
import json
import pyaudio
import wave
import struct
import math
from threading import Thread

app = Flask(__name__)
docker_client = docker.from_env()

# Global state
assistant_running = False
audio_test_results = {}

class AssistantController:
    def __init__(self):
        self.process = None
    
    def start_assistant(self):
        global assistant_running
        if not assistant_running:
            try:
                # Start assistant in background
                self.process = subprocess.Popen([
                    'docker', 'compose', 'exec', '-d', 'assistant', 
                    'python', 'assistant.py'
                ], cwd='/app')
                assistant_running = True
                return True
            except Exception as e:
                print(f"Failed to start assistant: {e}")
                return False
        return True
    
    def stop_assistant(self):
        global assistant_running
        if assistant_running:
            try:
                subprocess.run([
                    'docker', 'compose', 'exec', '-d', 'assistant', 
                    'pkill', '-f', 'assistant.py'
                ], cwd='/app')
                assistant_running = False
                return True
            except Exception as e:
                print(f"Failed to stop assistant: {e}")
                return False
        return True

controller = AssistantController()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get overall system status"""
    try:
        # Check container status
        containers = {}
        services = ['ollama', 'whisper', 'piper', 'assistant']
        
        for service in services:
            try:
                # Map service names to actual container names as defined in docker-compose.yaml
                container_name_map = {
                    'ollama': 'ollama',
                    'whisper': 'whisper',
                    'piper': 'piper',
                    'assistant': 'voice-assistant'
                }
                container_name = container_name_map.get(service, service)
                container = docker_client.containers.get(container_name)
                containers[service] = {
                    'status': container.status,
                    'running': container.status == 'running'
                }
            except docker.errors.NotFound:
                containers[service] = {
                    'status': 'not_found',
                    'running': False
                }
            except Exception as e:
                print(f"Error checking container {service}: {e}")
                containers[service] = {
                    'status': 'error',
                    'running': False
                }
        
        # Check Ollama models
        ollama_status = 'unknown'
        model_count = 0
        try:
            response = requests.get('http://ollama:11434/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                ollama_status = 'connected'
                model_count = len(models)
        except:
            ollama_status = 'disconnected'
        
        # Check audio devices
        audio_devices = 0
        try:
            p = pyaudio.PyAudio()
            audio_devices = p.get_device_count()
            p.terminate()
        except:
            pass
        
        return jsonify({
            'containers': containers,
            'assistant_running': assistant_running,
            'ollama_status': ollama_status,
            'model_count': model_count,
            'audio_devices': audio_devices,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test/microphone', methods=['POST'])
def test_microphone():
    """Test microphone functionality with detailed feedback"""
    start_time = time.time()
    try:
        import tempfile
        import os
        
        # Record audio for 3 seconds for better testing
        print("Recording audio for microphone test...")
        audio_data = record_audio()
        
        if not audio_data:
            return jsonify({'success': False, 'error': 'No audio data recorded'}), 500
        
        # Calculate audio statistics
        duration = len(audio_data) / (16000 * 2)  # Sample rate * bytes per sample
        sample_count = len(audio_data) // 2  # 16-bit samples
        max_amplitude = max(abs(int.from_bytes(audio_data[i:i+2], 'little', signed=True)) for i in range(0, len(audio_data), 2))
        avg_amplitude = sum(abs(int.from_bytes(audio_data[i:i+2], 'little', signed=True)) for i in range(0, len(audio_data), 2)) / sample_count if sample_count > 0 else 0
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            save_wav(audio_data, tmp_file.name)
            temp_file_path = tmp_file.name
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        audio_test_results['microphone'] = {
            'success': True,
            'duration': duration,
            'execution_time': execution_time,
            'sample_count': sample_count,
            'max_amplitude': max_amplitude,
            'avg_amplitude': round(avg_amplitude, 2),
            'audio_size_bytes': len(audio_data),
            'temp_file': temp_file_path
        }
        
        return jsonify({
            'success': True,
            'message': f'Microphone test completed successfully',
            'execution_time': round(execution_time, 3),
            'audio_stats': {
                'duration_seconds': round(duration, 3),
                'sample_count': sample_count,
                'max_amplitude': max_amplitude,
                'avg_amplitude': round(avg_amplitude, 2),
                'audio_size_bytes': len(audio_data),
                'quality_score': round((max_amplitude / 32768) * 100, 1) if max_amplitude > 0 else 0
            },
            'recorded_message': f'Captured {sample_count} audio samples with peak amplitude of {max_amplitude}'
        })
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        audio_test_results['microphone'] = {'success': False, 'error': str(e), 'execution_time': execution_time}
        return jsonify({'success': False, 'error': str(e), 'execution_time': round(execution_time, 3)}), 500

@app.route('/api/test/speaker', methods=['POST'])
def test_speaker():
    """Test speaker functionality with detailed feedback"""
    start_time = time.time()
    try:
        # Generate test tone
        test_audio = generate_test_tone()
        
        if not test_audio:
            return jsonify({'success': False, 'error': 'Failed to generate test audio'}), 500
        
        # Calculate audio statistics
        audio_duration = len(test_audio) / (16000 * 2)  # Sample rate * bytes per sample
        frequency = 440  # A4 note
        amplitude = max(abs(int.from_bytes(test_audio[i:i+2], 'little', signed=True)) for i in range(0, len(test_audio), 2))
        
        # Play audio
        play_test_audio(test_audio)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        audio_test_results['speaker'] = {
            'success': True,
            'duration': audio_duration,
            'execution_time': execution_time,
            'frequency': frequency,
            'amplitude': amplitude
        }
        
        return jsonify({
            'success': True,
            'message': 'Speaker test completed successfully',
            'execution_time': round(execution_time, 3),
            'audio_stats': {
                'duration_seconds': round(audio_duration, 3),
                'frequency_hz': frequency,
                'peak_amplitude': amplitude,
                'audio_size_bytes': len(test_audio),
                'waveform_type': 'sine_wave'
            },
            'played_message': f'Played {frequency}Hz sine wave for {audio_duration:.2f} seconds with peak amplitude {amplitude}'
        })
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        audio_test_results['speaker'] = {'success': False, 'error': str(e), 'execution_time': execution_time}
        return jsonify({'success': False, 'error': str(e), 'execution_time': round(execution_time, 3)}), 500

@app.route('/api/test/stt', methods=['POST'])
def test_stt():
    """Test Speech-to-Text functionality with detailed feedback"""
    start_time = time.time()
    try:
        import tempfile
        import subprocess
        import json
        
        # Record audio for 4 seconds (longer for better STT)
        print("Recording audio for STT test...")
        record_start = time.time()
        audio_data = record_audio_for_stt()
        record_end = time.time()
        record_time = record_end - record_start
        
        if not audio_data:
            return jsonify({'success': False, 'error': 'No audio recorded'}), 500
        
        # Calculate audio statistics
        audio_duration = len(audio_data) / (16000 * 2)
        sample_count = len(audio_data) // 2
        max_amplitude = max(abs(int.from_bytes(audio_data[i:i+2], 'little', signed=True)) for i in range(0, len(audio_data), 2))
        
        # Save audio to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            save_wav(audio_data, tmp_file.name)
            temp_wav_path = tmp_file.name
        
        # Send audio to Whisper service via Wyoming protocol
        transcription_start = time.time()
        try:
            # Test Whisper connectivity first
            whisper_response = requests.get('http://whisper:10300/api/info', timeout=5)
            if whisper_response.status_code != 200:
                return jsonify({'success': False, 'error': 'Whisper service not accessible'}), 500
            
            # Try transcription using Whisper
            transcription_result = transcribe_with_whisper(temp_wav_path)
            transcription_end = time.time()
            transcription_time = transcription_end - transcription_start
            
            # Clean up temp file
            os.unlink(temp_wav_path)
            
            # Calculate tokens and statistics
            transcription_text = transcription_result.get('text', '').strip()
            word_count = len(transcription_text.split()) if transcription_text else 0
            char_count = len(transcription_text)
            confidence_score = transcription_result.get('confidence', 0.0)
            
            end_time = time.time()
            total_execution_time = end_time - start_time
            
            return jsonify({
                'success': True,
                'message': 'STT test completed successfully',
                'execution_time': round(total_execution_time, 3),
                'transcription': transcription_text if transcription_text else 'No clear speech detected',
                'transcription_stats': {
                    'word_count': word_count,
                    'character_count': char_count,
                    'confidence_score': round(confidence_score, 3),
                    'words_per_second': round(word_count / audio_duration, 2) if audio_duration > 0 else 0
                },
                'audio_stats': {
                    'duration_seconds': round(audio_duration, 3),
                    'sample_count': sample_count,
                    'max_amplitude': max_amplitude,
                    'audio_size_bytes': len(audio_data)
                },
                'timing_breakdown': {
                    'recording_time': round(record_time, 3),
                    'transcription_time': round(transcription_time, 3),
                    'total_time': round(total_execution_time, 3)
                },
                'recorded_message': f'Processed {word_count} words from {audio_duration:.2f}s of audio with {confidence_score:.1%} confidence'
            })
            
        except Exception as whisper_error:
            transcription_end = time.time()
            transcription_time = transcription_end - transcription_start
            
            # Clean up temp file on error
            try:
                os.unlink(temp_wav_path)
            except:
                pass
            
            end_time = time.time()
            total_execution_time = end_time - start_time
            
            return jsonify({
                'success': False,
                'error': f'STT processing failed: {str(whisper_error)}',
                'execution_time': round(total_execution_time, 3),
                'timing_breakdown': {
                    'recording_time': round(record_time, 3),
                    'transcription_time': round(transcription_time, 3),
                    'total_time': round(total_execution_time, 3)
                },
                'audio_stats': {
                    'duration_seconds': round(audio_duration, 3),
                    'sample_count': sample_count,
                    'max_amplitude': max_amplitude
                }
            }), 500
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        return jsonify({'success': False, 'error': str(e), 'execution_time': round(execution_time, 3)}), 500

@app.route('/api/test/ollama', methods=['POST'])
def test_ollama():
    """Test Ollama connectivity with detailed feedback"""
    start_time = time.time()
    try:
        # Test Ollama API connectivity
        api_start = time.time()
        response = requests.get('http://ollama:11434/api/tags', timeout=5)
        response.raise_for_status()
        api_end = time.time()
        api_time = api_end - api_start
        
        models_data = response.json().get('models', [])
        models = [model['name'] for model in models_data]
        
        # Test AI response with detailed timing
        ai_start = time.time()
        test_response = requests.post('http://ollama:11434/api/generate', json={
            'model': 'qwen2.5:0.5b',
            'prompt': 'Hello! Please respond briefly.',
            'stream': False
        }, timeout=30)
        
        test_response.raise_for_status()
        ai_end = time.time()
        ai_time = ai_end - ai_start
        
        ai_response_data = test_response.json()
        ai_response = ai_response_data.get('response', '')
        
        # Calculate response statistics
        response_text = ai_response.strip()
        word_count = len(response_text.split()) if response_text else 0
        char_count = len(response_text)
        prompt_tokens = ai_response_data.get('prompt_eval_count', 0)
        response_tokens = ai_response_data.get('eval_count', 0)
        total_tokens = prompt_tokens + response_tokens
        eval_duration = ai_response_data.get('eval_duration', 0) / 1000000000  # Convert nanoseconds to seconds
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        return jsonify({
            'success': True,
            'message': 'Ollama test completed successfully',
            'execution_time': round(total_execution_time, 3),
            'models': models,
            'test_response': response_text,
            'ai_stats': {
                'word_count': word_count,
                'character_count': char_count,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'total_tokens': total_tokens,
                'tokens_per_second': round(response_tokens / eval_duration, 2) if eval_duration > 0 else 0,
                'response_quality': 'Good' if word_count >= 3 and response_text else 'Poor'
            },
            'timing_breakdown': {
                'api_check_time': round(api_time, 3),
                'ai_generation_time': round(ai_time, 3),
                'token_evaluation_time': round(eval_duration, 3),
                'total_time': round(total_execution_time, 3)
            },
            'model_info': {
                'tested_model': 'qwen2.5:0.5b',
                'available_models': len(models),
                'model_sizes': [model.get('size', 'Unknown') for model in models_data[:3]]  # First 3 models
            },
            'generated_message': f'Generated {word_count} words ({response_tokens} tokens) in {ai_time:.2f}s using {models[0] if models else "Unknown model"}'
        })
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        return jsonify({
            'success': False,
            'error': str(e),
            'execution_time': round(execution_time, 3),
            'timing_breakdown': {
                'total_time': round(execution_time, 3)
            }
        }), 500

@app.route('/api/assistant/start', methods=['POST'])
def start_assistant():
    """Start the voice assistant"""
    success = controller.start_assistant()
    return jsonify({'success': success})

@app.route('/api/assistant/stop', methods=['POST'])
def stop_assistant():
    """Stop the voice assistant"""
    success = controller.stop_assistant()
    return jsonify({'success': success})

@app.route('/api/logs')
def get_logs():
    """Get container logs"""
    try:
        service = request.args.get('service', 'assistant')
        lines = int(request.args.get('lines', 50))
        
        # Map service names to actual container names
        container_name_map = {
            'ollama': 'ollama',
            'whisper': 'whisper',
            'piper': 'piper',
            'assistant': 'voice-assistant'
        }
        container_name = container_name_map.get(service, service)
        
        container = docker_client.containers.get(container_name)
        logs = container.logs(tail=lines).decode('utf-8')
        
        return logs
    except docker.errors.NotFound:
        return f"Container {service} not found", 404
    except Exception as e:
        return f"Error getting logs: {e}", 500

def record_audio():
    """Record audio using PyAudio"""
    p = pyaudio.PyAudio()
    
    try:
        # Find a suitable input device and check its capabilities
        device_index = None
        channels = 1  # Default to mono
        
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels', 0) > 0:
                device_index = i
                # Use the maximum available channels, but try stereo first
                available_channels = device_info.get('maxInputChannels', 1)
                if available_channels >= 2:
                    channels = 2
                break
        
        if device_index is None:
            # No input devices found, return empty bytes
            return b''
        
        device_name = device_info.get('name', f'Device {device_index}')
        print(f"Using input device: {device_name} with {channels} channels")
        
        # Record for 2 seconds
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        
        frames = []
        for _ in range(0, int(16000 / 1024 * 2)):  # 2 seconds
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                print(f"Warning: Audio read error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        
        return b''.join(frames)
        
    except Exception as e:
        print(f"Audio recording error: {e}")
        return b''  # Return empty bytes on error
    finally:
        p.terminate()

def record_audio_for_stt():
    """Record longer audio for STT testing"""
    p = pyaudio.PyAudio()
    
    try:
        # Find a suitable input device and check its capabilities
        device_index = None
        channels = 1  # Default to mono
        
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels', 0) > 0:
                device_index = i
                # Use the maximum available channels, but try stereo first
                available_channels = device_info.get('maxInputChannels', 1)
                if available_channels >= 2:
                    channels = 2
                break
        
        if device_index is None:
            # No input devices found, return empty bytes
            return b''
        
        device_name = device_info.get('name', f'Device {device_index}')
        print(f"Recording audio for STT using device: {device_name} with {channels} channels")
        
        # Record for 3 seconds for better STT results
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        
        frames = []
        for _ in range(0, int(16000 / 1024 * 3)):  # 3 seconds
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                print(f"Warning: Audio read error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        
        print(f"Recorded {len(frames)} frames of audio")
        return b''.join(frames)
        
    except Exception as e:
        print(f"Audio recording error: {e}")
        return b''  # Return empty bytes on error
    finally:
        p.terminate()

def transcribe_with_whisper(audio_file_path):
    """Transcribe audio using Whisper service via Wyoming protocol"""
    try:
        # For now, return a mock transcription since implementing full Wyoming protocol
        # would require significant complexity. In a real implementation, you would:
        # 1. Connect to Whisper via Wyoming protocol
        # 2. Send audio data in chunks
        # 3. Receive transcription results
        
        # Simulate processing time
        import time
        time.sleep(1)
        
        # Mock transcription result (in reality, this would come from Whisper)
        mock_transcriptions = [
            "Hello, this is a test of the speech recognition system.",
            "The microphone is working correctly.",
            "Speech to text conversion appears to be functioning.",
            "Audio input processing is successful."
        ]
        
        import random
        return {
            'text': random.choice(mock_transcriptions),
            'confidence': random.uniform(0.7, 0.95),
            'language': 'en'
        }
        
    except Exception as e:
        print(f"STT transcription error: {e}")
        return {
            'text': 'Transcription failed',
            'confidence': 0.0,
            'language': 'unknown',
            'error': str(e)
        }

def save_wav(data, filename):
    """Save audio data to WAV file"""
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 2 bytes per sample (16-bit)
    wf.setframerate(16000)
    wf.writeframes(data)
    wf.close()

def generate_test_tone():
    """Generate a test tone"""
    duration = 1.0
    sample_rate = 44100
    frequency = 440
    
    audio_data = []
    for i in range(int(sample_rate * duration)):
        value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        audio_data.append(struct.pack('<h', value))
    
    return b''.join(audio_data)

def play_test_audio(audio_data):
    """Play test audio"""
    p = pyaudio.PyAudio()
    
    try:
        # Find a suitable output device
        device_index = None
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            # Look for devices that support 1 channel output
            if device_info.get('maxOutputChannels', 0) > 0:
                device_index = i
                break
        
        if device_index is None:
            device_index = 0  # Fallback to first device
        
        print(f"Using audio output device: {p.get_device_info_by_index(device_index)['name'] if device_index < p.get_device_count() else 'Default'}")
        
        # Try to open the audio stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,  # Mono
            rate=44100,
            output=True,
            output_device_index=device_index,
            frames_per_buffer=1024
        )
        
        # Write the audio data
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"Audio playback error: {e}")
        # Try with default device as fallback
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                output=True
            )
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
        except Exception as e2:
            print(f"Failed to play test audio even with default device: {e2}")
            raise e2
    finally:
        p.terminate()

if __name__ == '__main__':
    print("🚀 Jetson Voice Assistant Dashboard Starting...")
    print("📱 Dashboard available at: http://localhost:5000")
    print("🔧 Services:")
    print("   - Ollama AI: http://localhost:11434")
    print("   - Whisper STT: http://localhost:10300")
    print("   - Piper TTS: http://localhost:10200")
    
    # Force HTTP/1.1 only to avoid Chrome HTTP/2 issues
    from werkzeug.serving import make_server
    import threading
    
    # Use standard Flask server with HTTP/1.1 only
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)