# Dockerfile for Voice Assistant
FROM python:3.11-slim

# Install build tools + PortAudio headers + runtime audio libs + Docker CLI
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-dev \
    gcc \
    libasound2 \
    libasound2-dev \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    libpulse-dev \
    libportaudio2 \
    libportaudiocpp0 \
    sox \
    libsox-fmt-all \
    libjack-jackd2-dev \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY assistant.py .

# Install Python deps (pyaudio now compiles cleanly)
RUN pip install --no-cache-dir requests pyaudio websocket-client flask docker

COPY assistant.py web_dashboard.py ./
COPY templates/ ./templates/

# Default command (can be overridden in docker-compose)
CMD ["python", "assistant.py"]