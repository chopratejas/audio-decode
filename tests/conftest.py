"""
Pytest configuration and shared fixtures.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return Path(__file__).parent.parent / "fixtures" / "audio"


@pytest.fixture(scope="session")
def ensure_fixtures(fixtures_dir: Path) -> Path:
    """
    Ensure test audio fixtures exist.

    If fixtures don't exist, we'll generate them using soundfile
    (which should be installed as a dependency).
    """
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Check if we need to generate fixtures
    expected_files = [
        "test_1s_mono_8000.wav",
        "test_1s_mono_16000.wav",
        "test_1s_stereo_44100.wav",
        "test_10s_mono_16000.wav",
    ]

    if all((fixtures_dir / f).exists() for f in expected_files):
        return fixtures_dir

    # Generate fixtures
    import soundfile as sf

    # Generate test audio signals
    def generate_sine_wave(
        duration: float, sample_rate: int, frequency: float = 440.0, channels: int = 1
    ) -> np.ndarray:
        """Generate a sine wave."""
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t)

        if channels == 2:
            # Create stereo with slightly different frequencies
            audio_stereo = np.stack([audio, np.sin(2 * np.pi * (frequency * 1.01) * t)], axis=1)
            return audio_stereo.astype(np.float32)

        return audio.astype(np.float32)

    # Generate WAV files
    fixtures = [
        ("test_1s_mono_8000.wav", 1.0, 8000, 1),
        ("test_1s_mono_16000.wav", 1.0, 16000, 1),
        ("test_1s_stereo_44100.wav", 1.0, 44100, 2),
        ("test_10s_mono_16000.wav", 10.0, 16000, 1),
    ]

    for filename, duration, sample_rate, channels in fixtures:
        audio = generate_sine_wave(duration, sample_rate, channels=channels)
        sf.write(fixtures_dir / filename, audio, sample_rate)

    return fixtures_dir


@pytest.fixture
def audio_1s_mono_16k(ensure_fixtures: Path) -> Path:
    """1 second mono audio at 16kHz."""
    return ensure_fixtures / "test_1s_mono_16000.wav"


@pytest.fixture
def audio_1s_stereo_44k(ensure_fixtures: Path) -> Path:
    """1 second stereo audio at 44.1kHz."""
    return ensure_fixtures / "test_1s_stereo_44100.wav"


@pytest.fixture
def audio_10s_mono_16k(ensure_fixtures: Path) -> Path:
    """10 second mono audio at 16kHz."""
    return ensure_fixtures / "test_10s_mono_16000.wav"


@pytest.fixture(scope="session")
def has_librosa() -> bool:
    """Check if librosa is installed (for benchmarks)."""
    try:
        import librosa
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_torch() -> bool:
    """Check if torch is installed."""
    try:
        import torch
        return True
    except ImportError:
        return False
