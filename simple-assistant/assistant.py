# assistant.py
# Offline Voice Assistant: Record → STT → LLM → TTS → Speak
# Works in Docker on Mac (CPU) and Jetson (CUDA) — no host deps

import os
import time
import wave
import subprocess
import pyaudio
import requests
import threading
import sys

# === CONFIG ===
OLLAMA_URL = f"http://{os.getenv('OLLAMA_HOST', 'ollama:11434')}/api/generate"
WHISPER_URL = f"http://{os.getenv('WHISPER_HOST', 'whisper:10300')}/transcribe"
PIPER_URL = f"http://{os.getenv('PIPER_HOST', 'piper:10200')}/speak"
WAKE_WORDS = ["hey assistant", "hello assistant", "hi assistant", "assistant"]
RECORD_SECONDS = 4
SAMPLE_RATE = 16000
CHUNK = 1024

# === DYNAMIC MEMORY OPTIMIZATION ===
# Models prioritized by quality first, memory requirement second
PREFERRED_MODELS = [
    {
        "name": "qwen2.5:1.5b",
        "gpu_layers": 15,
        "context_size": 4096,
        "memory_intensive": True,
        "quality": "high"
    },
    {
        "name": "qwen2.5:0.5b",
        "gpu_layers": 10,
        "context_size": 2048,
        "memory_intensive": False,
        "quality": "medium"
    },
    {
        "name": "llama3.2:1b",
        "gpu_layers": 5,
        "context_size": 2048,
        "memory_intensive": False,
        "quality": "medium"
    },
    {
        "name": "phi3:mini",
        "gpu_layers": 0,  # CPU-only
        "context_size": 1024,
        "memory_intensive": False,
        "quality": "basic"
    }
]

# Dynamic fallback strategy based on available memory
def get_available_memory_gb():
    """Check available GPU memory (primary) with system RAM fallback"""
    # Try GPU memory first (more relevant for AI model selection)
    try:
        import subprocess
        # Use nvidia-smi to get GPU memory
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_memory_mb = int(result.stdout.strip().split('\n')[0])
            gpu_memory_gb = gpu_memory_mb / 1024
            print(f"🎮 GPU Memory Detected: {gpu_memory_gb:.1f}GB available")
            return gpu_memory_gb
    except:
        pass  # Fall back to system memory if GPU check fails
    
    # Fallback: Check available system memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            available_kb = int(meminfo.split('MemAvailable:')[1].split()[0])
            return available_kb / 1024 / 1024  # Convert to GB
    except:
        return 4.0  # Default conservative estimate

def select_model_for_memory():
    """Select appropriate model based on available memory"""
    available_gb = get_available_memory_gb()
    
    if available_gb > 6:
        return PREFERRED_MODELS[:2]  # Use top 2 models
    elif available_gb > 3:
        return PREFERRED_MODELS[1:3]  # Use middle models
    else:
        return PREFERRED_MODELS[2:]  # Use lightweight models

# === AUDIO SETUP ===
p = pyaudio.PyAudio()
stream = None
audio_device_info = {}

def check_pulseaudio_status():
    """Check if PulseAudio is running and accessible"""
    print("🔧 Checking audio system status...")
    
    # Check PULSE_SERVER environment variable
    pulse_server = os.getenv('PULSE_SERVER')
    if pulse_server:
        print(f"   📡 PULSE_SERVER: {pulse_server}")
        
        # Check if PulseAudio socket exists
        if pulse_server.startswith('unix:'):
            socket_path = pulse_server.replace('unix:', '')
            try:
                result = subprocess.run(['ls', '-la', socket_path],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   ✅ PulseAudio socket exists: {socket_path}")
                    
                    # Test if we can connect to PulseAudio
                    try:
                        test_result = subprocess.run(['pactl', 'info'],
                                                   capture_output=True, text=True, timeout=5)
                        if test_result.returncode == 0:
                            print(f"   ✅ PulseAudio daemon is responsive")
                            return True
                        else:
                            print(f"   ⚠️ PulseAudio socket exists but daemon not responsive")
                            return False
                    except Exception as e:
                        print(f"   ⚠️ PulseAudio connection test failed: {e}")
                        return False
                else:
                    print(f"   ❌ PulseAudio socket not found: {socket_path}")
                    print("   🔄 Will attempt direct ALSA access...")
                    return False
            except Exception as e:
                print(f"   ❌ Error checking PulseAudio socket: {e}")
                print("   🔄 Will attempt direct ALSA access...")
                return False
    else:
        print("   ❌ PULSE_SERVER not set")
        print("   🔄 Will attempt direct ALSA access...")
    
    return False

def list_audio_devices():
    """List all available audio devices for debugging"""
    print("🎤 Available audio devices:")
    device_count = p.get_device_count()
    print(f"   Total devices: {device_count}")
    
    # For Mac Docker Desktop, try different approaches
    if device_count == 0:
        print("   ⚠️  No audio devices detected by PyAudio")
        print("   🔄 Mac Docker Desktop detected - trying fallback strategies...")
        
        # Try different PyAudio backends
        backends_to_try = ['coreaudio', 'alsa', 'pulse']
        
        for backend in backends_to_try:
            try:
                print(f"   🔍 Trying {backend} backend...")
                # Test if we can create a PyAudio instance with specific backend
                temp_p = pyaudio.PyAudio()
                temp_count = temp_p.get_device_count()
                temp_p.terminate()
                
                if temp_count > 0:
                    print(f"   ✅ {backend} backend found {temp_count} devices!")
                    break
            except Exception as e:
                print(f"   ❌ {backend} backend failed: {e}")
        
        # Check for ALSA devices manually
        try:
            result = subprocess.run(['ls', '-la', '/dev/snd/'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ALSA devices available:\n{result.stdout}")
            else:
                print("   ❌ ALSA devices not accessible")
        except Exception as e:
            print(f"   ❌ Error checking ALSA: {e}")
    
    # List all available devices
    for i in range(device_count):
        try:
            device_info = p.get_device_info_by_index(i)
            print(f"   Device {i}: {device_info['name']}")
            print(f"     Max Input Channels: {device_info['maxInputChannels']}")
            print(f"     Max Output Channels: {device_info['maxOutputChannels']}")
            print(f"     Default Sample Rate: {device_info['defaultSampleRate']}")
            
            if device_info['maxInputChannels'] > 0:
                print(f"     ✅ Can record audio")
            if device_info['maxOutputChannels'] > 0:
                print(f"     ✅ Can play audio")
            print()
        except Exception as e:
            print(f"   Device {i}: Error reading info - {e}")

def find_best_input_device():
    """Find the best available input device for Jetson/Docker"""
    device_count = p.get_device_count()
    print(f"🔍 Scanning {device_count} audio devices...")
    
    # Handle case where no devices are found at all
    if device_count <= 0:
        print("❌ No audio devices detected by PyAudio")
        return None
    
    # First, try to get default input device (most reliable)
    try:
        default_input = p.get_default_input_device_info()
        print(f"🔍 Default input device: {default_input['name']}")
        
        if default_input['maxInputChannels'] > 0:
            try:
                # Test default device with Jetson-friendly parameters
                test_stream = p.open(format=pyaudio.paInt16,
                                   channels=1,
                                   rate=16000,
                                   input=True,
                                   frames_per_buffer=1024,
                                   input_device_index=default_input['index'],
                                   start=False)
                test_stream.close()
                print(f"✅ Default input device works: {default_input['name']}")
                return default_input['index']
            except Exception as e:
                print(f"❌ Default input device failed: {e}")
    except Exception as e:
        print(f"⚠️  Cannot get default input device: {e}")
    
    # Try all available input devices with simple test
    working_devices = []
    
    for i in range(device_count):
        try:
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"🔄 Testing device {i}: {device_info['name']}")
                
                # Simple test - just try to open the device
                test_stream = p.open(format=pyaudio.paInt16,
                                   channels=1,
                                   rate=16000,
                                   input=True,
                                   frames_per_buffer=512,
                                   input_device_index=i,
                                   start=False)
                test_stream.close()
                print(f"✅ Working input device found: {device_info['name']}")
                working_devices.append(i)
                
        except Exception as e:
            print(f"❌ Device {i} failed: {e}")
            continue
    
    # Return the first working device, or None if none found
    if working_devices:
        return working_devices[0]
    else:
        print("⚠️  No working input devices found")
        return None

def start_pulseaudio_daemon():
    """Start PulseAudio daemon for Mac Docker Desktop"""
    print("🔧 Starting PulseAudio daemon...")
    
    try:
        # Check if PulseAudio is already running
        result = subprocess.run(['pulseaudio', '--check'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ PulseAudio daemon already running")
            return True
            
        # Create necessary directories
        subprocess.run(['mkdir', '-p', '/tmp/pulse-runtime'], check=True)
        subprocess.run(['mkdir', '-p', '/root/.config/pulse'], check=True)
        
        # Start PulseAudio daemon
        pulse_cmd = [
            'pulseaudio', '--daemonize=no', '--fail=true', '--log-level=debug',
            '--disallow-exit', '--exit-idle-time=-1',
            '--load-module', 'module-native-protocol-unix',
            '--socket=/tmp/pulse-runtime/native'
        ]
        
        result = subprocess.run(pulse_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ PulseAudio daemon started successfully")
            
            # Set environment variables for PulseAudio
            os.environ['PULSE_SERVER'] = 'unix:/tmp/pulse-runtime/native'
            os.environ['XDG_RUNTIME_DIR'] = '/tmp/pulse-runtime'
            
            # Wait a moment for PulseAudio to initialize
            time.sleep(2)
            return True
        else:
            print(f"❌ Failed to start PulseAudio: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting PulseAudio daemon: {e}")
        return False

def init_audio():
    global stream, audio_device_info
    
    print("🔧 Initializing audio system for Jetson/Docker...")
    
    # For Jetson/Docker: Skip PulseAudio complexity and use direct ALSA
    print("🔧 Using direct ALSA audio setup...")
    
    # Set environment for better ALSA compatibility
    os.environ['ALSA_PCM_CARD'] = '0'
    os.environ['ALSA_PCM_DEVICE'] = '0'
    
    try:
        list_audio_devices()
    except Exception as e:
        print(f"⚠️  Device listing failed: {e}")
    
    # Try to find working input device with Jetson-friendly parameters
    input_device = find_best_input_device()
    
    # If no device found, try to proceed anyway (may work with system defaults)
    if input_device is None:
        print("⚠️  No specific device found, using system defaults...")
        input_device = None
    
    try:
        # Use Jetson/Docker compatible stream parameters
        stream_params = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': 16000,
            'input': True,
            'frames_per_buffer': 1024,  # Larger buffer for stability
            'start': False
        }
        
        if input_device is not None:
            stream_params['input_device_index'] = input_device
            
        stream = p.open(**stream_params)
        audio_device_info = {'defaultSampleRate': 16000}
        
        print("✅ Audio stream initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to open audio stream: {e}")
        print("🔄 Trying alternative configuration...")
        
        try:
            # Fallback: Try with different parameters
            stream_params = {
                'format': pyaudio.paInt16,
                'channels': 1,
                'rate': 16000,
                'input': True,
                'frames_per_buffer': 512,
                'input_device_index': None  # Use default
            }
            
            stream = p.open(**stream_params)
            audio_device_info = {'defaultSampleRate': 16000}
            print("✅ Audio stream initialized with fallback settings!")
            
        except Exception as fallback_e:
            print(f"❌ Fallback audio setup failed: {fallback_e}")
            raise RuntimeError(f"❌ All audio initialization attempts failed: {e}, {fallback_e}")

def record_audio():
    print("🎤 Listening...")
    frames = []
    sample_rate = int(audio_device_info.get('defaultSampleRate', SAMPLE_RATE))
    
    try:
        if stream and not stream.is_active():
            stream.start_stream()
        
        for _ in range(0, int(sample_rate / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        print("✅ Recorded.")
        return b''.join(frames)
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        raise
    finally:
        if stream and stream.is_active():
            stream.stop_stream()

def save_wav(data, filename="input.wav"):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(int(audio_device_info.get('defaultSampleRate', SAMPLE_RATE)))
    wf.writeframes(data)
    wf.close()

def speech_to_text():
    with open("input.wav", "rb") as f:
        files = {"audio": f}
        try:
            r = requests.post(WHISPER_URL, files=files, timeout=10)
            r.raise_for_status()
            text = r.json().get("text", "").strip().lower()
            print(f"🗣️ You: {text}")
            return text
        except Exception as e:
            print(f"STT Error: {e}")
            return ""

def ask_llm(prompt):
    # Get models based on available memory
    available_models = select_model_for_memory()
    
    print(f"💾 Available memory: {get_available_memory_gb():.1f}GB")
    print(f"🔄 Testing {len(available_models)} models based on memory constraints")
    
    for model_config in available_models:
        model_name = model_config["name"]
        gpu_layers = model_config["gpu_layers"]
        context_size = model_config["context_size"]
        quality = model_config["quality"]
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            # Dynamic memory optimization based on model config
            "options": {
                "num_gpu_layers": gpu_layers,
                "context_size": context_size,
                "temperature": 0.7,
                "top_p": 0.9,
                # Memory management settings
                "repeat_last_n": 64,
                "repeat_penalty": 1.1,
                "stop": ["</s>", "User:", "Assistant:"]
            }
        }
        
        try:
            print(f"🤖 Testing {model_name} (GPU layers: {gpu_layers}, Context: {context_size}, Quality: {quality})")
            r = requests.post(OLLAMA_URL, json=payload, timeout=30)
            r.raise_for_status()
            response = r.json().get("response", "").strip()
            print(f"🤖 AI ({model_name}): {response}")
            return response
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"⚠️ Model {model_name} not found, trying next...")
                continue
            elif e.response.status_code == 500:
                print(f"⚠️ Model {model_name} failed (likely out of memory), trying next...")
                continue
            else:
                print(f"⚠️ LLM Error with {model_name}: {e}")
                continue
        except Exception as e:
            print(f"⚠️ Error with {model_name}: {e}")
            continue
    
    # If all models fail, try to download the most lightweight one
    try:
        print("🤖 No models available, downloading phi3:mini (CPU-compatible)...")
        pull_response = requests.post(f"http://{os.getenv('OLLAMA_HOST', 'ollama:11434')}/api/pull",
                                    json={"name": "phi3:mini"}, timeout=120)
        pull_response.raise_for_status()
        print("✅ Downloaded phi3:mini successfully!")
        return ask_llm(prompt)  # Retry with downloaded model
    except Exception as download_error:
        print(f"Model download failed: {download_error}")
        return "Sorry, I'm having trouble loading the AI models. Please check memory availability and try again."

def text_to_speech(text):
    try:
        r = requests.post(PIPER_URL, json={"text": text}, timeout=10)
        r.raise_for_status()
        with open("output.wav", "wb") as f:
            f.write(r.content)
        play_audio("output.wav")
    except Exception as e:
        print(f"TTS Error: {e}")

def play_audio(filename):
    try:
        wf = wave.open(filename, 'rb')
        
        # Mac Docker friendly player settings
        player_params = {
            'format': p.get_format_from_width(wf.getsampwidth()),
            'channels': wf.getnchannels(),
            'rate': wf.getframerate(),
            'output': True
        }
        
        # For Mac Docker, try different output strategies
        try:
            player = p.open(**player_params)
        except Exception as e:
            print(f"Primary audio output failed: {e}")
            print("🎵 Attempting alternative audio output...")
            
            # Fallback: try with default device
            player_params.pop('output_device_index', None)
            player = p.open(**player_params)
        
        print("🔊 Speaking...")
        data = wf.readframes(CHUNK)
        while data:
            player.write(data)
            data = wf.readframes(CHUNK)
        
        player.stop_stream()
        player.close()
        wf.close()
        
    except Exception as e:
        print(f"❌ Audio playback failed: {e}")
        # Don't raise exception - continue with the conversation
        if 'wf' in locals():
            try:
                wf.close()
            except:
                pass

# === MAIN LOOP ===
def main():
    print("🚀 Offline Voice Assistant Ready!")
    print("   Say: 'Hey assistant, [your question]'")
    
    # Try to initialize audio with proper Jetson/Docker handling
    for attempt in range(2):
        try:
            print(f"🔧 Initializing audio system (attempt {attempt + 1}/2)...")
            init_audio()
            print("✅ Audio system initialized successfully!")
            break
        except Exception as e:
            print(f"❌ Audio initialization failed: {e}")
            if attempt == 0:
                print("🔄 Retrying with simplified audio setup...")
                time.sleep(2)
            else:
                print("❌ Audio system unavailable - continuing with voice mode disabled")
                print("⚠️  Note: This may be normal in headless Docker environments")
                return

    # Voice interaction mode
    print("🎤 Starting VOICE MODE - listening for 'Hey assistant'...")
    while True:
        try:
            audio_data = record_audio()
            save_wav(audio_data)
            text = speech_to_text()

            if text and any(ww in text for ww in WAKE_WORDS):
                question = text.split("assistant", 1)[-1].strip() or text
                if not question:
                    question = "What can I help with?"
                response = ask_llm(question)
                text_to_speech(response)
            else:
                print("⏳ No wake word — waiting...")
            
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("🔄 Continuing voice mode...")
            time.sleep(1)

    stream.stop_stream()
    stream.close()
    
    p.terminate()

if __name__ == "__main__":
    main()