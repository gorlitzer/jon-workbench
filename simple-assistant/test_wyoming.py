#!/usr/bin/env python3
"""
Test script to verify Wyoming protocol communication with Piper TTS
"""

import socket
import time

def test_wyoming_piper():
    """Test communication with Piper via Wyoming protocol"""
    host = 'localhost'  # External access
    port = 10200
    timeout = 5
    
    test_text = "Hello, this is a test of the text to speech system."
    
    try:
        print(f"🔧 Testing Wyoming protocol connection to {host}:{port}")
        print(f"📝 Test text: '{test_text}'")
        
        # Create Wyoming protocol message for SPEAK command
        message_data = test_text.encode('utf-8')
        message_header = f"SPEAK {len(message_data)}\n\n".encode('utf-8')
        full_message = message_header + message_data
        
        print(f"📦 Sending message: {len(full_message)} bytes")
        
        # Connect to Piper via Wyoming protocol
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        
        print("✅ Connected successfully!")
        
        # Send the message
        sock.sendall(full_message)
        print("📤 Message sent")
        
        # Receive audio data
        audio_data = b''
        chunk_count = 0
        
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    print("📥 Connection closed by server")
                    break
                
                chunk_count += 1
                audio_data += chunk
                print(f"📥 Received chunk {chunk_count}: {len(chunk)} bytes (total: {len(audio_data)} bytes)")
                
                # Stop after reasonable amount of data
                if len(audio_data) > 100000:  # 100KB limit
                    print("🛑 Stopping reception after 100KB")
                    break
                    
            except socket.timeout:
                print("⏰ Socket timeout")
                break
        
        sock.close()
        
        print(f"✅ Total received: {len(audio_data)} bytes")
        
        if len(audio_data) > 44:  # Minimum WAV header
            # Check for WAV file signature
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                print("🎵 Valid WAV audio data received!")
                
                # Save to file for verification
                with open('test_output.wav', 'wb') as f:
                    f.write(audio_data)
                print("💾 Audio saved to test_output.wav")
                
                return True
            else:
                print("⚠️ Data received but not a valid WAV file")
                print(f"Header: {audio_data[:20]}")
                return False
        else:
            print("❌ Insufficient audio data received")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Wyoming Protocol with Piper TTS")
    success = test_wyoming_piper()
    print(f"🏁 Test {'PASSED' if success else 'FAILED'}")