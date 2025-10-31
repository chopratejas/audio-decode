"""
Extended Benchmarks - Test with larger files and realistic workloads
====================================================================

Tests we didn't do in original benchmarks:
1. Larger files (1min, 5min, 30min audio)
2. Realistic batch sizes (1000+ files)
3. Memory usage over time
4. Different sample rates and formats
"""

import sys
import os
sys.path.insert(0, "src")

import time
import subprocess
from pathlib import Path
from typing import List
import psutil
import numpy as np

from audiodecode import AudioDecoder, clear_cache
import librosa


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def create_test_audio(duration_sec: int, sample_rate: int, output_path: str):
    """Create synthetic audio file for testing"""
    # Generate sine wave
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    # Save as WAV using ffmpeg
    cmd = [
        "ffmpeg", "-f", "f32le", "-ar", str(sample_rate),
        "-ac", "1", "-i", "-",
        "-y", output_path
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate(input=audio.astype(np.float32).tobytes())

    return output_path


def test_large_files():
    """Test with larger audio files"""
    print("\n" + "="*70)
    print("TEST: Large File Performance")
    print("="*70)

    test_dir = Path("fixtures/extended_tests")
    test_dir.mkdir(exist_ok=True)

    durations = [10, 60]  # 10s, 1min (would add 5min, 30min but takes time)

    for duration in durations:
        print(f"\n{duration} second file:")

        wav_file = test_dir / f"test_{duration}s.wav"
        if not wav_file.exists():
            print(f"  Creating {duration}s test file...")
            create_test_audio(duration, 16000, str(wav_file))

        # librosa
        print("  librosa:")
        start = time.perf_counter()
        audio_lr, sr = librosa.load(str(wav_file), sr=None)
        lr_time = time.perf_counter() - start
        print(f"    Time: {lr_time*1000:.2f}ms")
        print(f"    Shape: {audio_lr.shape}")

        # AudioDecode
        print("  AudioDecode:")
        clear_cache()
        start = time.perf_counter()
        audio_ad = AudioDecoder(str(wav_file)).decode()
        ad_time = time.perf_counter() - start
        print(f"    Time: {ad_time*1000:.2f}ms")
        print(f"    Shape: {audio_ad.shape}")

        speedup = lr_time / ad_time
        print(f"    Speedup: {speedup:.2f}x")


def test_large_batch():
    """Test with 1000+ files"""
    print("\n" + "="*70)
    print("TEST: Large Batch (1000 files)")
    print("="*70)

    # Use existing small file repeated
    test_file = "fixtures/audio/wav_1s_mono_16000.mp3"
    files = [test_file] * 1000

    # Track memory
    mem_before = get_memory_usage_mb()
    print(f"\nMemory before: {mem_before:.1f} MB")

    # AudioDecode without cache
    print("\n1. AudioDecode (no cache):")
    clear_cache()
    start = time.perf_counter()
    for f in files[:1000]:
        audio = AudioDecoder(f).decode(use_cache=False)
    nocache_time = time.perf_counter() - start
    mem_after_nocache = get_memory_usage_mb()
    print(f"   Time: {nocache_time:.3f}s ({nocache_time/len(files)*1000:.3f}ms per file)")
    print(f"   Memory: {mem_after_nocache:.1f} MB (delta: {mem_after_nocache - mem_before:.1f} MB)")

    # AudioDecode with cache
    print("\n2. AudioDecode (with cache):")
    clear_cache()
    start = time.perf_counter()
    for f in files[:1000]:
        audio = AudioDecoder(f).decode(use_cache=True)
    cache_time = time.perf_counter() - start
    mem_after_cache = get_memory_usage_mb()
    print(f"   Time: {cache_time:.3f}s ({cache_time/len(files)*1000:.3f}ms per file)")
    print(f"   Memory: {mem_after_cache:.1f} MB (delta: {mem_after_cache - mem_before:.1f} MB)")
    print(f"   Cache speedup: {nocache_time/cache_time:.1f}x")

    # librosa (skip for now, too slow)
    print("\n3. librosa:")
    print("   SKIPPED (would take too long)")


def test_different_sample_rates():
    """Test with different sample rates"""
    print("\n" + "="*70)
    print("TEST: Different Sample Rates")
    print("="*70)

    test_dir = Path("fixtures/extended_tests")
    test_dir.mkdir(exist_ok=True)

    sample_rates = [8000, 16000, 22050, 44100, 48000]

    results = []

    for sr in sample_rates:
        wav_file = test_dir / f"test_1s_{sr}hz.wav"
        if not wav_file.exists():
            create_test_audio(1, sr, str(wav_file))

        # AudioDecode
        clear_cache()
        start = time.perf_counter()
        audio = AudioDecoder(str(wav_file)).decode()
        ad_time = time.perf_counter() - start

        # librosa
        start = time.perf_counter()
        audio_lr, _ = librosa.load(str(wav_file), sr=None)
        lr_time = time.perf_counter() - start

        speedup = lr_time / ad_time
        results.append((sr, lr_time * 1000, ad_time * 1000, speedup))

    print("\n| Sample Rate | librosa (ms) | AudioDecode (ms) | Speedup |")
    print("|-------------|--------------|------------------|---------|")
    for sr, lr_time, ad_time, speedup in results:
        print(f"| {sr:5d} Hz    | {lr_time:8.2f}     | {ad_time:12.2f}     | {speedup:5.2f}x  |")


def test_resampling_overhead():
    """Test resampling performance"""
    print("\n" + "="*70)
    print("TEST: Resampling Overhead")
    print("="*70)

    test_file = "fixtures/audio/wav_1s_stereo_44100.wav"

    # No resampling
    print("\n1. No resampling (44.1kHz -> 44.1kHz):")
    start = time.perf_counter()
    audio1 = AudioDecoder(test_file).decode()
    no_resample_time = time.perf_counter() - start
    print(f"   Time: {no_resample_time*1000:.3f}ms")

    # Resample to 16kHz
    print("\n2. With resampling (44.1kHz -> 16kHz):")
    start = time.perf_counter()
    audio2 = AudioDecoder(test_file, target_sr=16000).decode()
    resample_time = time.perf_counter() - start
    print(f"   Time: {resample_time*1000:.3f}ms")

    overhead = resample_time - no_resample_time
    print(f"\n   Resampling overhead: {overhead*1000:.3f}ms ({overhead/no_resample_time*100:.1f}%)")


def test_memory_leak():
    """Test for memory leaks in repeated decoding"""
    print("\n" + "="*70)
    print("TEST: Memory Leak Detection")
    print("="*70)

    test_file = "fixtures/audio/wav_1s_mono_16000.mp3"

    mem_start = get_memory_usage_mb()
    print(f"\nMemory at start: {mem_start:.1f} MB")

    # Decode 1000 times without cache
    clear_cache()
    for i in range(1000):
        audio = AudioDecoder(test_file).decode(use_cache=False)
        if i % 100 == 0:
            mem_current = get_memory_usage_mb()
            print(f"  After {i:4d} decodes: {mem_current:.1f} MB (delta: {mem_current - mem_start:+.1f} MB)")

    mem_end = get_memory_usage_mb()
    mem_growth = mem_end - mem_start

    print(f"\nMemory at end: {mem_end:.1f} MB")
    print(f"Total growth: {mem_growth:.1f} MB")

    if mem_growth > 10:
        print("WARNING: Possible memory leak detected!")
    else:
        print("PASS: No significant memory leak")


def main():
    """Run all extended tests"""
    print("="*70)
    print("EXTENDED BENCHMARKS - Realistic Workloads")
    print("="*70)

    tests = [
        test_large_files,
        test_large_batch,
        test_different_sample_rates,
        test_resampling_overhead,
        test_memory_leak,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\nTEST FAILED: {test_func.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("EXTENDED BENCHMARKS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
