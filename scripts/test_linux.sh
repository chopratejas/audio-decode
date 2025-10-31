#!/bin/bash
# Script to test AudioDecode on Linux and measure subprocess overhead

set -e

echo "=================================="
echo "AudioDecode Linux Test Suite"
echo "=================================="

# 1. Check platform
echo -e "\n1. Platform Information:"
echo "OS: $(uname -s)"
echo "Kernel: $(uname -r)"
python3 -c "import platform; print(f'Python: {platform.python_version()}')"

# 2. Check audioread backend (should use ffmpeg subprocess on Linux)
echo -e "\n2. Checking audioread backend:"
python3 << 'EOF'
import audioread
print(f"Available backends: {audioread.available_backends()}")
EOF

# 3. Test librosa with subprocess monitoring
echo -e "\n3. Testing librosa (should show subprocess spawning):"
python3 << 'EOF'
import subprocess
import time
from pathlib import Path

# Monkey-patch to detect subprocess
original_popen = subprocess.Popen
spawn_count = [0]

def tracked_popen(*args, **kwargs):
    spawn_count[0] += 1
    print(f"   🚨 SUBPROCESS SPAWNED #{spawn_count[0]}: {args[0] if args else 'unknown'}")
    return original_popen(*args, **kwargs)

subprocess.Popen = tracked_popen

# Test librosa
import librosa
test_file = "fixtures/audio/wav_1s_mono_16000.mp3"

if Path(test_file).exists():
    print(f"   Decoding: {test_file}")
    start = time.perf_counter()
    audio, sr = librosa.load(test_file, sr=None)
    elapsed = time.perf_counter() - start

    print(f"   ✓ Decoded {len(audio)} samples in {elapsed*1000:.2f}ms")
    print(f"   Total subprocesses: {spawn_count[0]}")
else:
    print(f"   ⚠ MP3 file not found: {test_file}")

subprocess.Popen = original_popen
EOF

# 4. Run benchmarks
echo -e "\n4. Running benchmarks:"
python3 -m pytest benchmarks/test_performance.py -k "mp3" -v --tb=short || echo "Benchmarks may fail until PyAV backend is implemented"

echo -e "\n=================================="
echo "Test complete!"
echo "=================================="
