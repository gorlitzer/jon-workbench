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
import tempfile
from threading import Thread

app = Flask(__name__, static_folder='static')
docker_client = docker.from_env()

# Global state
assistant_running = False
audio_test_results = {}

def communicate_with_wyoming_piper(text, host='piper', port=10200, timeout=10):
    """
    Communicate with Piper TTS service using Wyoming protocol
    Returns audio data as bytes, or None if failed
    """
    import socket
    
    try:
        # Create Wyoming protocol message for SPEAK command
        # Format: "SPEAK <text_length>\n\n<text>"
        message_data = text.encode('utf-8')
        message_header = f"SPEAK {len(message_data)}\n\n".encode('utf-8')
        full_message = message_header + message_data
        
        # Connect to Piper via Wyoming protocol
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        
        # Send the message
        sock.sendall(full_message)
        
        # Receive audio data
        audio_data = b''
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                audio_data += chunk
            except socket.timeout:
                break
        
        sock.close()
        
        # Check if we got audio data
        if len(audio_data) < 44:  # Minimum WAV header size
            raise Exception(f"Insufficient audio data received: {len(audio_data)} bytes")
            
        return audio_data
        
    except Exception as e:
        print(f"Wyoming protocol error: {e}")
        return None

class WhisperServiceController:
    def __init__(self):
        self.process = None
        self.service_enabled = False
        self.last_health_check = None
        self.memory_threshold = 0.8  # 80% GPU memory usage threshold
        
    def start_whisper_service(self, force=False):
        """Start Whisper service with memory optimization"""
        try:
            # Check memory usage if not forced
            if not force:
                if self.get_gpu_memory_usage() > self.memory_threshold:
                    print("GPU memory usage too high, skipping Whisper startup")
                    return False
            
            # Stop existing service if running
            self.stop_whisper_service()
            
            # Start Whisper container with optimized settings
            self.process = subprocess.Popen([
                'docker', 'run', '-d',
                '--name', 'whisper-dynamic',
                '--rm',
                '--runtime', 'nvidia',
                '-p', '10300:10300',
                '-e', 'NVIDIA_VISIBLE_DEVICES=all',
                '-e', f'WHISPER_MODEL={os.getenv("WHISPER_MODEL", "base")}',
                '-e', f'WHISPER_LANGUAGE={os.getenv("WHISPER_LANGUAGE", "en")}',
                'rhasspy/wyoming-whisper'
            ], cwd='/app')
            
            self.service_enabled = True
            print("Whisper service started dynamically")
            return True
            
        except Exception as e:
            print(f"Failed to start Whisper service: {e}")
            self.service_enabled = False
            return False
    
    def stop_whisper_service(self):
        """Stop Whisper service and free GPU memory"""
        try:
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=10)
                self.process = None
            
            # Stop and remove dynamic container
            subprocess.run(['docker', 'stop', 'whisper-dynamic'],
                         capture_output=True, timeout=10)
            subprocess.run(['docker', 'rm', 'whisper-dynamic'],
                         capture_output=True, timeout=10)
            
            self.service_enabled = False
            print("Whisper service stopped and memory freed")
            return True
            
        except Exception as e:
            print(f"Error stopping Whisper service: {e}")
            return False
    
    def is_whisper_running(self):
        """Check if Whisper service is running"""
        try:
            # Check if dynamic container is running
            result = subprocess.run([
                'docker', 'ps', '--filter', 'name=whisper-dynamic', '--format', '{{.Names}}'
            ], capture_output=True, text=True)
            
            return 'whisper-dynamic' in result.stdout.strip()
            
        except Exception:
            return False
    
    def get_whisper_status(self):
        """Get detailed Whisper service status"""
        try:
            # Check running status
            running = self.is_whisper_running()
            
            # Check health endpoint if running
            health_status = "unknown"
            if running:
                try:
                    response = requests.get('http://whisper-dynamic:10300/api/info', timeout=3)
                    if response.status_code == 200:
                        health_status = "healthy"
                    else:
                        health_status = "unhealthy"
                except:
                    health_status = "unreachable"
            
            # Get model info
            model = os.getenv('WHISPER_MODEL', 'unknown')
            language = os.getenv('WHISPER_LANGUAGE', 'unknown')
            
            return {
                'running': running,
                'enabled': self.service_enabled,
                'health': health_status,
                'model': model,
                'language': language,
                'last_check': self.last_health_check,
                'memory_threshold': self.memory_threshold
            }
            
        except Exception as e:
            return {
                'running': False,
                'enabled': self.service_enabled,
                'health': 'error',
                'error': str(e),
                'model': 'unknown',
                'language': 'unknown'
            }
    
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage percentage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                memory_used = int(lines[0].split(',')[0].strip())
                memory_total = int(lines[0].split(',')[1].strip())
                return memory_used / memory_total if memory_total > 0 else 0
            return 0
        except:
            return 0
    
    def health_check(self):
        """Perform health check and update last_health_check"""
        try:
            self.last_health_check = time.time()
            status = self.get_whisper_status()
            
            # Auto-start if enabled but not running
            if self.service_enabled and not status['running']:
                print("Whisper service enabled but not running, attempting restart...")
                return self.start_whisper_service()
            
            return status
            
        except Exception as e:
            print(f"Whisper health check failed: {e}")
            return {'running': False, 'health': 'error', 'error': str(e)}

# Initialize controllers
whisper_controller = WhisperServiceController()

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

@app.route('/api/test/tts', methods=['POST'])
def test_tts():
    """Test TTS (Text-to-Speech) functionality with detailed feedback"""
    start_time = time.time()
    try:
        # Test Piper TTS service connectivity first
        tts_start = time.time()
        test_text = "This is a test of the text to speech system. The microphone is working correctly."
        
        # Try Wyoming protocol (Piper TTS standard)
        try:
            audio_data = communicate_with_wyoming_piper(test_text)
            if not audio_data:
                raise Exception("No audio data received from Wyoming protocol")
        except Exception as wyoming_error:
            print(f"Wyoming protocol failed: {wyoming_error}")
            
            # Try HTTP API as fallback (for older Piper versions)
            try:
                response = requests.post('http://piper:10200/speak',
                                        json={"text": test_text},
                                        timeout=5)
                response.raise_for_status()
                audio_data = response.content
            except Exception as http_error:
                end_time = time.time()
                execution_time = end_time - start_time
                
                return jsonify({
                    'success': False,
                    'error': f'TTS service unavailable: {str(wyoming_error)}, HTTP fallback also failed: {str(http_error)}',
                    'execution_time': round(execution_time, 3)
                }), 500
        
        tts_end = time.time()
        tts_time = tts_end - tts_start
        
        # Save audio data to temporary file and play it
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_data)
            temp_audio_path = tmp_file.name
        
        # Play the generated audio
        play_test_tts_audio(temp_audio_path)
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return jsonify({
            'success': True,
            'message': 'TTS test completed successfully',
            'execution_time': round(execution_time, 3),
            'test_response': f'Generated speech for: "{test_text}"',
            'timing_breakdown': {
                'tts_generation_time': round(tts_time, 3),
                'total_time': round(execution_time, 3)
            },
            'played_message': f'Successfully generated and played TTS audio in {tts_time:.2f} seconds'
        })
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        return jsonify({
            'success': False,
            'error': f'TTS test failed: {str(e)}',
            'execution_time': round(execution_time, 3)
        }), 500

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

# Whisper Service Management Endpoints
@app.route('/api/whisper/start', methods=['POST'])
def start_whisper():
    """Start Whisper service dynamically"""
    try:
        force = request.json.get('force', False) if request.json else False
        success = whisper_controller.start_whisper_service(force=force)
        return jsonify({'success': success, 'message': 'Whisper service started' if success else 'Failed to start Whisper service'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/whisper/stop', methods=['POST'])
def stop_whisper():
    """Stop Whisper service and free memory"""
    try:
        success = whisper_controller.stop_whisper_service()
        return jsonify({'success': success, 'message': 'Whisper service stopped' if success else 'Failed to stop Whisper service'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/whisper/enable', methods=['POST'])
def enable_whisper():
    """Enable Whisper service for auto-start"""
    try:
        whisper_controller.service_enabled = True
        return jsonify({'success': True, 'message': 'Whisper service enabled for auto-start'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/whisper/disable', methods=['POST'])
def disable_whisper():
    """Disable Whisper service for auto-start"""
    try:
        whisper_controller.service_enabled = False
        whisper_controller.stop_whisper_service()
        return jsonify({'success': True, 'message': 'Whisper service disabled'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/whisper/status', methods=['GET'])
def get_whisper_status():
    """Get Whisper service status"""
    try:
        status = whisper_controller.get_whisper_status()
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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

def play_test_tts_audio(audio_file_path):
    """Play TTS generated audio file"""
    try:
        wf = wave.open(audio_file_path, 'rb')
        
        # TTS audio player settings
        player_params = {
            'format': p.get_format_from_width(wf.getsampwidth()),
            'channels': wf.getnchannels(),
            'rate': wf.getframerate(),
            'output': True
        }
        
        # Try to open audio stream
        try:
            player = p.open(**player_params)
        except Exception as e:
            print(f"Primary audio output failed: {e}")
            print("🎵 Attempting alternative audio output...")
            
            # Fallback: try with default device
            player_params.pop('output_device_index', None)
            player = p.open(**player_params)
        
        print("🔊 Playing TTS audio...")
        data = wf.readframes(1024)
        while data:
            player.write(data)
            data = wf.readframes(1024)
        
        player.stop_stream()
        player.close()
        wf.close()
        
    except Exception as e:
        print(f"❌ TTS audio playback failed: {e}")
        # Don't raise exception - continue with the test
        if 'wf' in locals():
            try:
                wf.close()
            except:
                pass

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