#!/usr/bin/env python3
"""
Generate audio test fixtures for AudioDecode project.

Creates WAV files with various configurations (sample rates, channels, durations)
and attempts to create MP3/FLAC files using ffmpeg if available.
"""

import numpy as np
import soundfile as sf
import subprocess
import shutil
from pathlib import Path


def generate_sine_wave(frequency: float, duration: float, sample_rate: int, num_channels: int = 1) -> np.ndarray:
    """
    Generate a sine wave audio signal.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        num_channels: Number of channels (1=mono, 2=stereo)

    Returns:
        numpy array of shape (samples,) for mono or (samples, channels) for stereo
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    if num_channels == 1:
        # Mono: simple sine wave
        signal = np.sin(2 * np.pi * frequency * t, dtype=np.float32)
    else:
        # Stereo: left channel at frequency, right channel at frequency * 1.5
        left = np.sin(2 * np.pi * frequency * t, dtype=np.float32)
        right = np.sin(2 * np.pi * (frequency * 1.5) * t, dtype=np.float32)
        signal = np.column_stack((left, right))

    # Normalize to -0.8 to 0.8 range (avoid clipping)
    signal = signal * 0.8

    return signal


def create_wav_fixtures(output_dir: Path):
    """Create WAV file fixtures with various configurations."""
    fixtures = [
        # (name, duration, sample_rate, num_channels, frequency)
        ("wav_1s_mono_8000.wav", 1.0, 8000, 1, 440.0),
        ("wav_1s_mono_16000.wav", 1.0, 16000, 1, 440.0),
        ("wav_1s_stereo_44100.wav", 1.0, 44100, 2, 440.0),
        ("wav_10s_mono_16000.wav", 10.0, 16000, 1, 880.0),
        ("wav_60s_mono_16000.wav", 60.0, 16000, 1, 440.0),
    ]

    created_files = []

    for name, duration, sample_rate, num_channels, frequency in fixtures:
        filepath = output_dir / name
        print(f"Generating {name}...")

        # Generate audio signal
        audio = generate_sine_wave(frequency, duration, sample_rate, num_channels)

        # Write WAV file
        sf.write(filepath, audio, sample_rate, subtype='FLOAT')

        # Verify file
        info = sf.info(filepath)
        file_size = filepath.stat().st_size

        created_files.append({
            'name': name,
            'path': filepath,
            'size': file_size,
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype
        })

        print(f"  ✓ Created: {file_size:,} bytes, {info.duration:.2f}s, "
              f"{info.samplerate}Hz, {info.channels}ch")

    return created_files


def convert_to_mp3(wav_file: Path, output_dir: Path, bitrate: str = "128k") -> dict | None:
    """Convert WAV to MP3 using ffmpeg if available."""
    if not shutil.which('ffmpeg'):
        return None

    mp3_file = output_dir / wav_file.name.replace('.wav', '.mp3')

    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(wav_file),
            '-codec:a', 'libmp3lame',
            '-b:a', bitrate,
            '-ar', '16000',  # Downsample to 16kHz for MP3
            str(mp3_file)
        ], check=True, capture_output=True, text=True)

        # Get info about created MP3
        file_size = mp3_file.stat().st_size

        return {
            'name': mp3_file.name,
            'path': mp3_file,
            'size': file_size,
            'format': 'MP3',
            'bitrate': bitrate
        }
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to create MP3: {e}")
        return None


def convert_to_flac(wav_file: Path, output_dir: Path) -> dict | None:
    """Convert WAV to FLAC using ffmpeg if available."""
    if not shutil.which('ffmpeg'):
        return None

    flac_file = output_dir / wav_file.name.replace('.wav', '.flac')

    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(wav_file),
            '-codec:a', 'flac',
            str(flac_file)
        ], check=True, capture_output=True, text=True)

        # Get info about created FLAC
        info = sf.info(flac_file)
        file_size = flac_file.stat().st_size

        return {
            'name': flac_file.name,
            'path': flac_file,
            'size': file_size,
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': 'FLAC'
        }
    except (subprocess.CalledProcessError, Exception) as e:
        print(f"  ✗ Failed to create FLAC: {e}")
        return None


def main():
    """Main function to generate all audio fixtures."""
    # Set up output directory
    script_dir = Path(__file__).parent
    audio_dir = script_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Audio Fixture Generator for AudioDecode")
    print("=" * 60)
    print()

    # Create WAV fixtures
    print("Creating WAV fixtures...")
    print("-" * 60)
    wav_files = create_wav_fixtures(audio_dir)
    print()

    # Try to create MP3 and FLAC fixtures
    ffmpeg_available = shutil.which('ffmpeg') is not None
    print(f"ffmpeg available: {ffmpeg_available}")
    print()

    mp3_files = []
    flac_files = []

    if ffmpeg_available:
        print("Creating MP3 fixtures...")
        print("-" * 60)
        # Convert a few WAV files to MP3
        for wav_info in wav_files[:3]:  # Convert first 3 WAV files
            print(f"Converting {wav_info['name']} to MP3...")
            mp3_info = convert_to_mp3(wav_info['path'], audio_dir)
            if mp3_info:
                mp3_files.append(mp3_info)
                print(f"  ✓ Created: {mp3_info['size']:,} bytes")
        print()

        print("Creating FLAC fixtures...")
        print("-" * 60)
        # Convert a few WAV files to FLAC
        for wav_info in wav_files[:2]:  # Convert first 2 WAV files
            print(f"Converting {wav_info['name']} to FLAC...")
            flac_info = convert_to_flac(wav_info['path'], audio_dir)
            if flac_info:
                flac_files.append(flac_info)
                print(f"  ✓ Created: {flac_info['size']:,} bytes, "
                      f"{flac_info['duration']:.2f}s, {flac_info['sample_rate']}Hz")
        print()
    else:
        print("Skipping MP3/FLAC creation (ffmpeg not found)")
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"WAV files created: {len(wav_files)}")
    print(f"MP3 files created: {len(mp3_files)}")
    print(f"FLAC files created: {len(flac_files)}")
    print()

    # Calculate total size
    all_files = wav_files + mp3_files + flac_files
    total_size = sum(f['size'] for f in all_files)
    print(f"Total fixtures: {len(all_files)}")
    print(f"Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    print()

    # Verify all files can be read
    print("Verifying all files...")
    print("-" * 60)
    for file_info in all_files:
        try:
            if file_info['name'].endswith('.mp3'):
                # MP3 files might not be readable by soundfile
                if file_info['path'].exists():
                    print(f"  ✓ {file_info['name']} exists")
            else:
                info = sf.info(file_info['path'])
                print(f"  ✓ {file_info['name']} is valid")
        except Exception as e:
            print(f"  ✗ {file_info['name']} verification failed: {e}")

    print()
    print("=" * 60)
    print("Fixture generation complete!")
    print(f"All files saved to: {audio_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
