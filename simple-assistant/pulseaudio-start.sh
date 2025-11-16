# Start PulseAudio daemon for Mac Docker Desktop audio
#!/bin/sh

# Create necessary directories
mkdir -p /tmp/pulse-runtime
mkdir -p /root/.config/pulse

# Start PulseAudio as a system daemon for audio forwarding
exec pulseaudio \
  --daemonize=no \
  --fail=true \
  --log-level=debug \
  --disallow-exit \
  --load-module=module-native-protocol-unix socket=/tmp/pulse-runtime/native \
  --exit-idle-time=-1 \
  --no-cork \
  --load-module=module-alsa-sink device=default \
  --load-module=module-alsa-source device=default \
  --local

echo "PulseAudio daemon started for Mac Docker Desktop audio"