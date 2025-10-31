"""
Performance regression tests using pytest-benchmark.

These tests FAIL if AudioDecode is slower than librosa baseline.
Run with: pytest benchmarks/test_performance.py --benchmark-only

Key features:
- Statistical rigor via pytest-benchmark (min, max, mean, stddev)
- Automatic baseline comparison and regression detection
- Separate tests for different backends (soundfile for WAV/FLAC, PyAV for MP3)
- Memory regression tests using memory_profiler
"""

import json
from pathlib import Path

import librosa
import numpy as np
import pytest

# Import AudioDecode (will fail gracefully if not implemented yet)
try:
    from audiodecode import AudioDecoder
    AUDIODECODE_AVAILABLE = True
except (ImportError, NotImplementedError):
    AUDIODECODE_AVAILABLE = False


# ============================================================================
# Test Configuration
# ============================================================================

# Path configuration
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "audio"
BASELINE_DIR = Path(__file__).parent / "baseline"
BASELINE_FILE = BASELINE_DIR / "baseline.json"

# Performance thresholds (fail if we're slower than these multipliers of librosa)
SPEEDUP_THRESHOLD = 1.0  # Must be at least as fast as librosa
MEMORY_THRESHOLD_MB = 50.0  # Must not use >50MB more memory than librosa


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def audio_files():
    """Discover all audio files in fixtures directory."""
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Fixtures directory not found: {FIXTURES_DIR}")

    files = {
        "wav": list(FIXTURES_DIR.glob("*.wav")),
        "flac": list(FIXTURES_DIR.glob("*.flac")),
        "mp3": list(FIXTURES_DIR.glob("*.mp3")),
        "m4a": list(FIXTURES_DIR.glob("*.m4a")),
    }

    # Filter out empty lists
    files = {k: v for k, v in files.items() if v}

    if not any(files.values()):
        pytest.skip(f"No audio files found in {FIXTURES_DIR}")

    return files


@pytest.fixture(scope="session")
def baseline_data():
    """Load baseline performance data if available."""
    if not BASELINE_FILE.exists():
        return {}

    with open(BASELINE_FILE) as f:
        return json.load(f)


# ============================================================================
# WAV/FLAC Tests (soundfile backend)
# ============================================================================

@pytest.mark.benchmark(group="wav-decode")
@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_decode_wav_audiodecode(benchmark, audio_files):
    """Benchmark AudioDecode WAV decoding (soundfile backend)."""
    if "wav" not in audio_files or not audio_files["wav"]:
        pytest.skip("No WAV files in fixtures")

    file_path = audio_files["wav"][0]

    def decode():
        decoder = AudioDecoder(file_path)
        return decoder.decode()

    result = benchmark(decode)
    assert result is not None
    assert result.dtype == np.float32


@pytest.mark.benchmark(group="wav-decode")
def test_decode_wav_librosa(benchmark, audio_files):
    """Benchmark librosa WAV decoding (baseline)."""
    if "wav" not in audio_files or not audio_files["wav"]:
        pytest.skip("No WAV files in fixtures")

    file_path = audio_files["wav"][0]

    def decode():
        audio, sr = librosa.load(str(file_path), sr=None)
        return audio

    result = benchmark(decode)
    assert result is not None


@pytest.mark.benchmark(group="wav-resample")
@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_decode_wav_resample_audiodecode(benchmark, audio_files):
    """Benchmark AudioDecode WAV decoding with resampling to 16kHz."""
    if "wav" not in audio_files or not audio_files["wav"]:
        pytest.skip("No WAV files in fixtures")

    file_path = audio_files["wav"][0]

    def decode():
        decoder = AudioDecoder(file_path, target_sr=16000)
        return decoder.decode()

    result = benchmark(decode)
    assert result is not None


@pytest.mark.benchmark(group="wav-resample")
def test_decode_wav_resample_librosa(benchmark, audio_files):
    """Benchmark librosa WAV decoding with resampling to 16kHz (baseline)."""
    if "wav" not in audio_files or not audio_files["wav"]:
        pytest.skip("No WAV files in fixtures")

    file_path = audio_files["wav"][0]

    def decode():
        audio, sr = librosa.load(str(file_path), sr=16000)
        return audio

    result = benchmark(decode)
    assert result is not None


# ============================================================================
# MP3 Tests (PyAV backend)
# ============================================================================

@pytest.mark.benchmark(group="mp3-decode")
@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_decode_mp3_audiodecode(benchmark, audio_files):
    """Benchmark AudioDecode MP3 decoding (PyAV backend)."""
    if "mp3" not in audio_files or not audio_files["mp3"]:
        pytest.skip("No MP3 files in fixtures")

    file_path = audio_files["mp3"][0]

    def decode():
        decoder = AudioDecoder(file_path)
        return decoder.decode()

    result = benchmark(decode)
    assert result is not None
    assert result.dtype == np.float32


@pytest.mark.benchmark(group="mp3-decode")
def test_decode_mp3_librosa(benchmark, audio_files):
    """Benchmark librosa MP3 decoding (baseline)."""
    if "mp3" not in audio_files or not audio_files["mp3"]:
        pytest.skip("No MP3 files in fixtures")

    file_path = audio_files["mp3"][0]

    def decode():
        audio, sr = librosa.load(str(file_path), sr=None)
        return audio

    result = benchmark(decode)
    assert result is not None


@pytest.mark.benchmark(group="mp3-resample")
@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_decode_mp3_resample_audiodecode(benchmark, audio_files):
    """Benchmark AudioDecode MP3 decoding with resampling to 16kHz."""
    if "mp3" not in audio_files or not audio_files["mp3"]:
        pytest.skip("No MP3 files in fixtures")

    file_path = audio_files["mp3"][0]

    def decode():
        decoder = AudioDecoder(file_path, target_sr=16000)
        return decoder.decode()

    result = benchmark(decode)
    assert result is not None


@pytest.mark.benchmark(group="mp3-resample")
def test_decode_mp3_resample_librosa(benchmark, audio_files):
    """Benchmark librosa MP3 decoding with resampling to 16kHz (baseline)."""
    if "mp3" not in audio_files or not audio_files["mp3"]:
        pytest.skip("No MP3 files in fixtures")

    file_path = audio_files["mp3"][0]

    def decode():
        audio, sr = librosa.load(str(file_path), sr=16000)
        return audio

    result = benchmark(decode)
    assert result is not None


# ============================================================================
# Regression Tests (FAIL if slower than baseline)
# ============================================================================

@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_wav_faster_than_librosa(audio_files):
    """
    REGRESSION TEST: Ensure AudioDecode WAV decoding is at least as fast as librosa.

    This test FAILS if AudioDecode is slower than librosa, preventing performance regressions.
    """
    if "wav" not in audio_files or not audio_files["wav"]:
        pytest.skip("No WAV files in fixtures")

    file_path = audio_files["wav"][0]

    # Warm up
    decoder = AudioDecoder(file_path)
    _ = decoder.decode()

    # Time AudioDecode (3 runs, take best)
    import time
    audiodecode_times = []
    for _ in range(3):
        start = time.perf_counter()
        decoder = AudioDecoder(file_path)
        audio = decoder.decode()
        audiodecode_time = time.perf_counter() - start
        audiodecode_times.append(audiodecode_time)

    audiodecode_time = min(audiodecode_times)

    # Time librosa (3 runs, take best)
    librosa_times = []
    for _ in range(3):
        start = time.perf_counter()
        audio, sr = librosa.load(str(file_path), sr=None)
        librosa_time = time.perf_counter() - start
        librosa_times.append(librosa_time)

    librosa_time = min(librosa_times)

    # Calculate speedup
    speedup = librosa_time / audiodecode_time

    # FAIL if slower than threshold
    assert speedup >= SPEEDUP_THRESHOLD, (
        f"AudioDecode is SLOWER than librosa for WAV files!\n"
        f"  AudioDecode: {audiodecode_time:.3f}s\n"
        f"  Librosa: {librosa_time:.3f}s\n"
        f"  Speedup: {speedup:.2f}x (threshold: {SPEEDUP_THRESHOLD:.2f}x)\n"
        f"  File: {file_path.name}"
    )

    print(f"\n✓ WAV decode speedup: {speedup:.2f}x (AudioDecode: {audiodecode_time:.3f}s, librosa: {librosa_time:.3f}s)")


@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_mp3_faster_than_librosa(audio_files):
    """
    REGRESSION TEST: Ensure AudioDecode MP3 decoding is at least as fast as librosa.

    This test FAILS if AudioDecode is slower than librosa, preventing performance regressions.
    """
    if "mp3" not in audio_files or not audio_files["mp3"]:
        pytest.skip("No MP3 files in fixtures")

    file_path = audio_files["mp3"][0]

    # Warm up
    decoder = AudioDecoder(file_path)
    _ = decoder.decode()

    # Time AudioDecode (3 runs, take best)
    import time
    audiodecode_times = []
    for _ in range(3):
        start = time.perf_counter()
        decoder = AudioDecoder(file_path)
        audio = decoder.decode()
        audiodecode_time = time.perf_counter() - start
        audiodecode_times.append(audiodecode_time)

    audiodecode_time = min(audiodecode_times)

    # Time librosa (3 runs, take best)
    librosa_times = []
    for _ in range(3):
        start = time.perf_counter()
        audio, sr = librosa.load(str(file_path), sr=None)
        librosa_time = time.perf_counter() - start
        librosa_times.append(librosa_time)

    librosa_time = min(librosa_times)

    # Calculate speedup
    speedup = librosa_time / audiodecode_time

    # FAIL if slower than threshold
    assert speedup >= SPEEDUP_THRESHOLD, (
        f"AudioDecode is SLOWER than librosa for MP3 files!\n"
        f"  AudioDecode: {audiodecode_time:.3f}s\n"
        f"  Librosa: {librosa_time:.3f}s\n"
        f"  Speedup: {speedup:.2f}x (threshold: {SPEEDUP_THRESHOLD:.2f}x)\n"
        f"  File: {file_path.name}"
    )

    print(f"\n✓ MP3 decode speedup: {speedup:.2f}x (AudioDecode: {audiodecode_time:.3f}s, librosa: {librosa_time:.3f}s)")


# ============================================================================
# Memory Regression Tests
# ============================================================================

@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_memory_usage_wav(audio_files):
    """
    REGRESSION TEST: Ensure AudioDecode doesn't use excessive memory.

    This test FAILS if AudioDecode uses significantly more memory than librosa.
    """
    if "wav" not in audio_files or not audio_files["wav"]:
        pytest.skip("No WAV files in fixtures")

    import psutil
    import gc

    file_path = audio_files["wav"][0]
    process = psutil.Process()

    # Measure librosa memory
    gc.collect()
    mem_before = process.memory_info().rss / (1024 * 1024)
    audio, sr = librosa.load(str(file_path), sr=None)
    mem_after = process.memory_info().rss / (1024 * 1024)
    librosa_memory = mem_after - mem_before
    del audio

    # Measure AudioDecode memory
    gc.collect()
    mem_before = process.memory_info().rss / (1024 * 1024)
    decoder = AudioDecoder(file_path)
    audio = decoder.decode()
    mem_after = process.memory_info().rss / (1024 * 1024)
    audiodecode_memory = mem_after - mem_before
    del audio, decoder

    # Calculate difference
    memory_diff = audiodecode_memory - librosa_memory

    # FAIL if using significantly more memory
    assert memory_diff <= MEMORY_THRESHOLD_MB, (
        f"AudioDecode uses TOO MUCH memory compared to librosa!\n"
        f"  AudioDecode: {audiodecode_memory:.1f} MB\n"
        f"  Librosa: {librosa_memory:.1f} MB\n"
        f"  Difference: {memory_diff:.1f} MB (threshold: {MEMORY_THRESHOLD_MB:.1f} MB)\n"
        f"  File: {file_path.name}"
    )

    print(f"\n✓ Memory usage: AudioDecode={audiodecode_memory:.1f}MB, librosa={librosa_memory:.1f}MB (diff={memory_diff:.1f}MB)")


# ============================================================================
# Baseline Comparison Tests
# ============================================================================

@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_compare_to_baseline_wav(audio_files, baseline_data):
    """
    REGRESSION TEST: Compare current performance to saved baseline.

    This test FAILS if current performance is worse than the saved baseline,
    preventing performance regressions between releases.
    """
    if "wav" not in audio_files or not audio_files["wav"]:
        pytest.skip("No WAV files in fixtures")

    if not baseline_data:
        pytest.skip("No baseline data available. Run with --benchmark-save to create baseline.")

    file_path = audio_files["wav"][0]
    baseline_key = f"{file_path.name}_None_False"  # format: filename_sr_mono

    if baseline_key not in baseline_data:
        pytest.skip(f"No baseline for {baseline_key}")

    baseline = baseline_data[baseline_key]

    # Time current implementation
    import time
    decoder = AudioDecoder(file_path)
    start = time.perf_counter()
    audio = decoder.decode()
    current_time = time.perf_counter() - start

    # Compare to baseline
    baseline_time = baseline["decode_time_seconds"]
    regression = (current_time - baseline_time) / baseline_time * 100

    # Allow 10% regression tolerance (to account for system variance)
    REGRESSION_TOLERANCE = 10.0

    assert regression <= REGRESSION_TOLERANCE, (
        f"Performance REGRESSION detected!\n"
        f"  Current: {current_time:.3f}s\n"
        f"  Baseline: {baseline_time:.3f}s\n"
        f"  Regression: {regression:.1f}% (threshold: {REGRESSION_TOLERANCE:.1f}%)\n"
        f"  File: {file_path.name}"
    )

    print(f"\n✓ Baseline comparison: {regression:+.1f}% change (current: {current_time:.3f}s, baseline: {baseline_time:.3f}s)")


# ============================================================================
# Quality Validation Tests
# ============================================================================

@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_output_matches_librosa_wav(audio_files):
    """
    Validate that AudioDecode produces bit-accurate (or nearly identical) output to librosa.

    This ensures performance improvements don't sacrifice decode quality.
    """
    if "wav" not in audio_files or not audio_files["wav"]:
        pytest.skip("No WAV files in fixtures")

    file_path = audio_files["wav"][0]

    # Decode with both libraries
    decoder = AudioDecoder(file_path)
    audiodecode_audio = decoder.decode()

    librosa_audio, sr = librosa.load(str(file_path), sr=None)

    # Compare shapes
    assert audiodecode_audio.shape == librosa_audio.shape, (
        f"Output shape mismatch!\n"
        f"  AudioDecode: {audiodecode_audio.shape}\n"
        f"  Librosa: {librosa_audio.shape}"
    )

    # Compare values (allow small floating-point differences)
    max_diff = np.max(np.abs(audiodecode_audio - librosa_audio))
    mean_diff = np.mean(np.abs(audiodecode_audio - librosa_audio))

    # Allow very small differences due to floating-point precision
    assert max_diff < 1e-5, (
        f"Audio output differs significantly from librosa!\n"
        f"  Max difference: {max_diff}\n"
        f"  Mean difference: {mean_diff}"
    )

    print(f"\n✓ Output quality: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")


@pytest.mark.skipif(not AUDIODECODE_AVAILABLE, reason="AudioDecode not yet implemented")
def test_output_matches_librosa_mp3(audio_files):
    """
    Validate that AudioDecode MP3 output matches librosa.

    Note: MP3 decoding may have minor differences between decoders due to codec variations,
    so we use a more lenient threshold than for lossless formats.
    """
    if "mp3" not in audio_files or not audio_files["mp3"]:
        pytest.skip("No MP3 files in fixtures")

    file_path = audio_files["mp3"][0]

    # Decode with both libraries
    decoder = AudioDecoder(file_path)
    audiodecode_audio = decoder.decode()

    librosa_audio, sr = librosa.load(str(file_path), sr=None)

    # Compare shapes
    assert audiodecode_audio.shape == librosa_audio.shape, (
        f"Output shape mismatch!\n"
        f"  AudioDecode: {audiodecode_audio.shape}\n"
        f"  Librosa: {librosa_audio.shape}"
    )

    # Compare values (more lenient for MP3)
    max_diff = np.max(np.abs(audiodecode_audio - librosa_audio))
    mean_diff = np.mean(np.abs(audiodecode_audio - librosa_audio))

    # Allow larger differences for MP3 (different decoders may produce slightly different output)
    assert max_diff < 0.01, (
        f"Audio output differs too much from librosa!\n"
        f"  Max difference: {max_diff}\n"
        f"  Mean difference: {mean_diff}"
    )

    print(f"\n✓ MP3 output quality: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
