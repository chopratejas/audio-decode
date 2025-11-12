"""
Comprehensive profiling script to identify optimization opportunities in AudioDecode.

This script profiles:
1. Audio loading (decode backends)
2. Inference (Whisper transcription)
3. Resampling operations
4. Array operations (mono conversion, normalization)
5. Cache operations
6. Model loading

Usage:
    python profile_hotspots.py
"""

import time
import cProfile
import pstats
import io
from pathlib import Path
import numpy as np
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"[TIMER] {self.name}: {self.elapsed*1000:.2f}ms")


def profile_audio_loading():
    """Profile audio loading backends."""
    print("\n" + "="*80)
    print("PROFILING: Audio Loading Backends")
    print("="*80)

    from audiodecode import load

    audio_file = Path("audio.mp3")
    if not audio_file.exists():
        print("⚠️  audio.mp3 not found, skipping audio loading profile")
        return

    # Cold load (no cache)
    with Timer("Audio load (cold, full)"):
        audio1, sr1 = load(str(audio_file), sr=16000, mono=True)

    print(f"  → Loaded {len(audio1)/sr1:.1f}s audio, shape={audio1.shape}")

    # Warm load (with cache)
    with Timer("Audio load (warm, cached)"):
        audio2, sr2 = load(str(audio_file), sr=16000, mono=True)

    # Load without resampling
    with Timer("Audio load (no resample)"):
        audio3, sr3 = load(str(audio_file), mono=True)

    print(f"  → Original SR: {sr3}Hz")

    # Load stereo (no mono conversion)
    with Timer("Audio load (stereo)"):
        audio4, sr4 = load(str(audio_file), sr=16000, mono=False)

    if audio4.ndim > 1:
        print(f"  → Stereo shape: {audio4.shape}")


def profile_inference():
    """Profile Whisper inference with detailed breakdown."""
    print("\n" + "="*80)
    print("PROFILING: Whisper Inference (Detailed)")
    print("="*80)

    from audiodecode import load, WhisperInference
    import torch

    audio_file = Path("audio.mp3")
    if not audio_file.exists():
        print("⚠️  audio.mp3 not found, skipping inference profile")
        return

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model loading
    with Timer("Model loading"):
        whisper = WhisperInference(
            model_size="base",
            device=device,
            compute_type="auto",
            batch_size=16
        )

    # Audio loading
    with Timer("Audio loading (for inference)"):
        audio, sr = load(str(audio_file), sr=16000, mono=True)

    print(f"  → Audio: {len(audio)/sr:.1f}s, {len(audio):,} samples")

    # Transcription (first run)
    with Timer("Transcription (first run)"):
        result1 = whisper.transcribe_audio(audio, sr, word_timestamps=False)

    print(f"  → Transcribed: {len(result1.segments)} segments, {len(result1.text.split())} words")

    # Transcription (second run, warmed up)
    with Timer("Transcription (warm run)"):
        result2 = whisper.transcribe_audio(audio, sr, word_timestamps=False)

    # Transcription with word timestamps
    with Timer("Transcription (with word timestamps)"):
        result3 = whisper.transcribe_audio(audio, sr, word_timestamps=True)

    # Count word timestamps
    word_count = sum(len(seg.words or []) for seg in result3.segments)
    print(f"  → Word timestamps: {word_count} words")


def profile_resampling():
    """Profile resampling operations."""
    print("\n" + "="*80)
    print("PROFILING: Resampling Operations")
    print("="*80)

    # Create test signal
    duration = 10.0  # 10 seconds
    orig_sr = 44100
    target_sr = 16000

    print(f"Test signal: {duration}s @ {orig_sr}Hz → {target_sr}Hz")

    # Generate test audio
    samples = int(duration * orig_sr)
    audio = np.random.randn(samples).astype(np.float32)

    print(f"  → Input: {audio.shape}, {audio.nbytes/1024/1024:.2f}MB")

    # Test with scipy
    try:
        from scipy import signal as scipy_signal

        num_samples = int(len(audio) * target_sr / orig_sr)

        with Timer("Resample (scipy)"):
            resampled_scipy = scipy_signal.resample(audio, num_samples)

        print(f"  → Output: {resampled_scipy.shape}")
    except ImportError:
        print("  ⚠️  scipy not available")

    # Test with soxr if available
    try:
        import soxr

        with Timer("Resample (soxr HQ)"):
            resampled_soxr = soxr.resample(audio, orig_sr, target_sr, quality="HQ")

        print(f"  → Output: {resampled_soxr.shape}")
    except ImportError:
        print("  ⚠️  soxr not available")


def profile_array_operations():
    """Profile numpy array operations."""
    print("\n" + "="*80)
    print("PROFILING: Array Operations")
    print("="*80)

    # Create test stereo audio
    duration = 60.0  # 1 minute
    sr = 44100
    samples = int(duration * sr)

    stereo_audio = np.random.randn(samples, 2).astype(np.float32)
    print(f"Test stereo audio: {stereo_audio.shape}, {stereo_audio.nbytes/1024/1024:.2f}MB")

    # Stereo to mono conversion (mean)
    with Timer("Stereo → Mono (mean)"):
        mono1 = stereo_audio.mean(axis=1).astype(np.float32)

    # Stereo to mono conversion (average, keepdims=False)
    with Timer("Stereo → Mono (mean, no keepdims)"):
        mono2 = stereo_audio.mean(axis=1, keepdims=False).astype(np.float32)

    # Test int16 → float32 normalization
    int16_audio = (np.random.randn(samples) * 32767).astype(np.int16)

    with Timer("Int16 → Float32 (normalize)"):
        float_audio = int16_audio.astype(np.float32) / 32768.0

    # Test array copy vs view
    large_audio = np.random.randn(samples).astype(np.float32)

    with Timer("Array copy"):
        copy = large_audio.copy()

    with Timer("Array view (reshape)"):
        view = large_audio.reshape(-1, 1)


def profile_cache():
    """Profile cache operations."""
    print("\n" + "="*80)
    print("PROFILING: Cache Operations")
    print("="*80)

    from audiodecode.cache import AudioCache

    cache = AudioCache(maxsize=10)

    # Create test audio
    audio = np.random.randn(16000 * 10).astype(np.float32)  # 10s @ 16kHz
    filepath = Path("test.mp3")

    print(f"Test audio: {audio.nbytes/1024/1024:.2f}MB")

    # Cache put
    with Timer("Cache PUT"):
        for i in range(20):
            cache.put(
                filepath,
                target_sr=16000,
                mono=True,
                offset=float(i),
                duration=None,
                audio=audio
            )

    print(f"  → Cache size: {cache.info()['entries']} entries, {cache.info()['total_memory_mb']:.2f}MB")

    # Cache get (hit)
    with Timer("Cache GET (hit)"):
        for i in range(10, 20):  # Last 10 should be in cache
            result = cache.get(
                filepath,
                target_sr=16000,
                mono=True,
                offset=float(i),
                duration=None
            )

    # Cache get (miss)
    with Timer("Cache GET (miss)"):
        for i in range(100, 110):  # Not in cache
            result = cache.get(
                filepath,
                target_sr=16000,
                mono=True,
                offset=float(i),
                duration=None
            )


def run_cprofile():
    """Run comprehensive cProfile on full transcription pipeline."""
    print("\n" + "="*80)
    print("PROFILING: Full Pipeline (cProfile)")
    print("="*80)

    audio_file = Path("audio.mp3")
    if not audio_file.exists():
        print("⚠️  audio.mp3 not found, skipping cProfile")
        return

    from audiodecode import WhisperInference
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Profile full transcription
    profiler = cProfile.Profile()
    profiler.enable()

    whisper = WhisperInference(model_size="base", device=device, batch_size=16)
    result = whisper.transcribe_file(str(audio_file), word_timestamps=True)

    profiler.disable()

    # Print top 30 functions by cumulative time
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)

    print("\nTop 30 functions by cumulative time:")
    print(s.getvalue())

    # Print top 30 functions by total time
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    stats.print_stats(30)

    print("\nTop 30 functions by total time:")
    print(s.getvalue())


def profile_batch_size_memory():
    """Profile memory usage across different batch sizes."""
    print("\n" + "="*80)
    print("PROFILING: Batch Size Memory Usage")
    print("="*80)

    import torch
    from audiodecode import WhisperInference, load

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU memory profiling")
        return

    audio_file = Path("audio.mp3")
    if not audio_file.exists():
        print("⚠️  audio.mp3 not found, skipping batch size profile")
        return

    # Load audio once
    audio, sr = load(str(audio_file), sr=16000, mono=True)

    batch_sizes = [8, 16, 24, 32]

    for batch_size in batch_sizes:
        # Clear GPU cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create model
        whisper = WhisperInference(
            model_size="base",
            device="cuda",
            compute_type="float16",
            batch_size=batch_size
        )

        # Transcribe
        with Timer(f"Transcribe (batch_size={batch_size})"):
            result = whisper.transcribe_audio(audio, sr)

        # Check GPU memory
        allocated = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2

        print(f"  → GPU Memory: {allocated:.1f}MB allocated, {peak:.1f}MB peak")

        # Cleanup
        del whisper
        torch.cuda.empty_cache()


def main():
    """Run all profiling benchmarks."""
    print("\n" + "="*80)
    print("AudioDecode Optimization Profiler")
    print("="*80)
    print("This will identify hot spots and optimization opportunities")
    print()

    # Run profiles
    profile_array_operations()
    profile_cache()
    profile_resampling()
    profile_audio_loading()
    profile_inference()
    profile_batch_size_memory()

    # Run comprehensive cProfile (most detailed)
    run_cprofile()

    print("\n" + "="*80)
    print("Profiling Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
