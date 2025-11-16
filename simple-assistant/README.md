================================================================
  OFFLINE AI VOICE ASSISTANT – 100% CONTAINERIZED
  "Hey Assistant" → Speaks back | Mac (CPU) ↔ Jetson (CUDA)
================================================================

QUICK START (2 MINUTES)
-----------------------
1. Install Docker:
   • Mac: https://www.docker.com/products/docker-desktop/
   • Jetson: Comes with JetPack (Docker pre-installed)

2. Clone & run:
   git clone https://github.com/yourname/offline-voice-assistant.git
   cd offline-voice-assistant
   docker compose up --build

3. Speak: "Hey assistant, what time is it?"
   → Hears → Thinks → Speaks answer

   (First run: downloads ~3GB model – one-time only)

FEATURES
--------
• 100% offline after setup
• Zero host packages (no pip, no apt, no Python)
• Same code runs on Mac & Jetson
• CUDA auto-enabled on Jetson
• CPU fallback on Mac
• Delete folder = system 100% clean

TECH STACK (ALL IN DOCKER)
--------------------------
• LLM:      ollama/ollama:0.5.7 + qwen2.5:1.5b
• STT:      rhasspy/wyoming-whisper (base model)
• TTS:      rhasspy/wyoming-piper (en_US-lessac-medium)
• Logic:    Python in container (assistant.py)

FILES
-----
docker-compose.yml  ← Starts all services
Dockerfile          ← Builds assistant container
assistant.py        ← Voice loop (record → STT → LLM → TTS)
ollama_data/        ← Stores model (git ignored)

CLEANUP
-------
docker compose down -v
rm -rf offline-voice-assistant

→ Your system is pristine.

TROUBLESHOOTING
---------------
• Mac mic access: Docker Desktop → Settings → Resources → Allow /dev/snd
• Jetson GPU not used? Run: nvidia-smi → should show ollama
• Slow first response? Model loading into RAM (~1.5GB)

NEXT STEPS
----------
• Add wake word: "Hey Jarv" → wyoming-porcupine
• Web GUI: Add ovos-gui container
• K3s + Rancher: Scale to cluster (later)

Enjoy your private, edge-native voice assistant!