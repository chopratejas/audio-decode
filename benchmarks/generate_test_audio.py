#!/usr/bin/env python3
"""
Generate test audio files for benchmarking.

This script creates sample audio files in various formats for testing:
- WAV (uncompressed, multiple sample rates)
- FLAC (lossless compression)
- MP3 (lossy compression, multiple bitrates)

Usage:
    python benchmarks/generate_test_audio.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_sine_wave(duration=5.0, sample_rate=44100, frequency=440.0, channels=2):
    """
    Generate a sine wave audio signal.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        frequency: Frequency in Hz
        channels: Number of channels (1=mono, 2=stereo)

    Returns:
        numpy array of shape (samples, channels) or (samples,) for mono
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t)

    if channels == 2:
        # Create stereo by adding slightly different frequencies
        left = audio
        right = np.sin(2 * np.pi * (frequency * 1.01) * t)
        audio = np.column_stack([left, right])

    return audio.astype(np.float32)


def main():
    """Generate test audio files."""
    output_dir = Path(__file__).parent.parent / "fixtures" / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test audio files in: {output_dir}")

    # 1. WAV files (uncompressed)
    print("\n1. Generating WAV files...")

    # 16kHz mono (common for speech)
    audio = generate_sine_wave(duration=5.0, sample_rate=16000, channels=1)
    sf.write(output_dir / "test_16khz_mono.wav", audio, 16000)
    print(f"   ✓ test_16khz_mono.wav ({audio.shape[0]} samples, 16kHz, mono)")

    # 44.1kHz stereo (CD quality)
    audio = generate_sine_wave(duration=5.0, sample_rate=44100, channels=2)
    sf.write(output_dir / "test_44khz_stereo.wav", audio, 44100)
    print(f"   ✓ test_44khz_stereo.wav ({audio.shape[0]} samples, 44.1kHz, stereo)")

    # 48kHz stereo (professional audio)
    audio = generate_sine_wave(duration=5.0, sample_rate=48000, channels=2)
    sf.write(output_dir / "test_48khz_stereo.wav", audio, 48000)
    print(f"   ✓ test_48khz_stereo.wav ({audio.shape[0]} samples, 48kHz, stereo)")

    # 2. FLAC files (lossless compression)
    print("\n2. Generating FLAC files...")

    audio = generate_sine_wave(duration=5.0, sample_rate=44100, channels=2)
    sf.write(output_dir / "test_44khz_stereo.flac", audio, 44100)
    print(f"   ✓ test_44khz_stereo.flac ({audio.shape[0]} samples, 44.1kHz, stereo)")

    # 3. MP3 files (lossy compression)
    # Note: soundfile doesn't support MP3 writing, but we can document
    # what files should be added manually
    print("\n3. MP3 files (requires manual creation or ffmpeg):")
    print("   To create MP3 test files, use ffmpeg:")
    print(f"   cd {output_dir}")
    print("   ffmpeg -i test_44khz_stereo.wav -b:a 128k test_128kbps.mp3")
    print("   ffmpeg -i test_44khz_stereo.wav -b:a 320k test_320kbps.mp3")

    # Try to create MP3 files if av is available
    try:
        import av

        print("\n   Attempting to create MP3 files with PyAV...")

        # 128kbps MP3
        audio = generate_sine_wave(duration=5.0, sample_rate=44100, channels=2)
        output_path = str(output_dir / "test_128kbps.mp3")

        container = av.open(output_path, mode="w")
        stream = container.add_stream("mp3", rate=44100)
        stream.channels = 2
        stream.bit_rate = 128000

        # Convert float32 to int16 for encoding
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create audio frame
        frame = av.AudioFrame.from_ndarray(audio_int16, format="s16", layout="stereo")
        frame.sample_rate = 44100

        for packet in stream.encode(frame):
            container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

        container.close()
        print(f"   ✓ test_128kbps.mp3 ({audio.shape[0]} samples, 44.1kHz, stereo, 128kbps)")

        # 320kbps MP3
        audio = generate_sine_wave(duration=5.0, sample_rate=44100, channels=2)
        output_path = str(output_dir / "test_320kbps.mp3")

        container = av.open(output_path, mode="w")
        stream = container.add_stream("mp3", rate=44100)
        stream.channels = 2
        stream.bit_rate = 320000

        audio_int16 = (audio * 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(audio_int16, format="s16", layout="stereo")
        frame.sample_rate = 44100

        for packet in stream.encode(frame):
            container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()
        print(f"   ✓ test_320kbps.mp3 ({audio.shape[0]} samples, 44.1kHz, stereo, 320kbps)")

    except Exception as e:
        print(f"   ✗ Could not create MP3 files with PyAV: {e}")
        print("   Please create MP3 files manually using ffmpeg (see commands above)")

    print(f"\n✓ Test audio files generated in: {output_dir}")
    print("\nFiles created:")
    for file in sorted(output_dir.glob("*")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
