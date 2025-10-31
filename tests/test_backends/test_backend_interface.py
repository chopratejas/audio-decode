"""
Test backend interface contracts.

These tests define what ALL backends must do. Any backend implementation
must pass these tests to be considered compliant with the AudioBackend protocol.
"""

from pathlib import Path

import numpy as np
import pytest

from audiodecode.backends.base import AudioBackend, AudioData, AudioInfo


class MockBackend(AudioBackend):
    """
    Mock backend for testing the interface.

    This minimal implementation helps us verify that the abstract
    interface is correctly defined.
    """

    def __init__(self, supported_formats: set[str] | None = None):
        self._supported_formats = supported_formats or {".wav", ".flac"}

    def supports_format(self, filepath: Path) -> bool:
        return filepath.suffix.lower() in self._supported_formats

    def get_info(self, filepath: Path) -> AudioInfo:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Mock implementation
        return AudioInfo(
            sample_rate=16000,
            channels=1,
            duration=1.0,
            samples=16000,
            format=filepath.suffix[1:],  # Remove leading dot
        )

    def decode(
        self,
        filepath: Path,
        target_sr: int | None = None,
        mono: bool = False,
        offset: float = 0.0,
        duration: float | None = None,
    ) -> AudioData:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Mock implementation - return synthetic audio
        sr = target_sr if target_sr else 16000
        num_samples = int(sr * 1.0)  # 1 second

        # Generate simple sine wave
        t = np.linspace(0, 1.0, num_samples, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        return AudioData(
            data=audio,
            sample_rate=sr,
            channels=1,
        )

    @property
    def name(self) -> str:
        return "mock"


class TestBackendInterface:
    """Test the AudioBackend interface contract."""

    def test_backend_has_required_methods(self):
        """All backends must implement the required methods."""
        backend = MockBackend()

        assert hasattr(backend, "supports_format")
        assert hasattr(backend, "get_info")
        assert hasattr(backend, "decode")
        assert hasattr(backend, "name")

    def test_supports_format_returns_bool(self, tmp_path):
        """supports_format must return a boolean."""
        backend = MockBackend(supported_formats={".wav"})

        wav_file = tmp_path / "test.wav"
        wav_file.touch()

        mp3_file = tmp_path / "test.mp3"
        mp3_file.touch()

        assert backend.supports_format(wav_file) is True
        assert backend.supports_format(mp3_file) is False

    def test_get_info_returns_audio_info(self, audio_1s_mono_16k):
        """get_info must return AudioInfo dataclass."""
        backend = MockBackend()

        info = backend.get_info(audio_1s_mono_16k)

        assert isinstance(info, AudioInfo)
        assert hasattr(info, "sample_rate")
        assert hasattr(info, "channels")
        assert hasattr(info, "duration")
        assert hasattr(info, "samples")
        assert hasattr(info, "format")

    def test_get_info_validates_types(self, audio_1s_mono_16k):
        """get_info must return correct types."""
        backend = MockBackend()

        info = backend.get_info(audio_1s_mono_16k)

        assert isinstance(info.sample_rate, int)
        assert isinstance(info.channels, int)
        assert isinstance(info.duration, float)
        assert isinstance(info.samples, int)
        assert isinstance(info.format, str)

    def test_get_info_raises_on_missing_file(self, tmp_path):
        """get_info must raise FileNotFoundError for missing files."""
        backend = MockBackend()
        missing_file = tmp_path / "does_not_exist.wav"

        with pytest.raises(FileNotFoundError):
            backend.get_info(missing_file)

    def test_decode_returns_audio_data(self, audio_1s_mono_16k):
        """decode must return AudioData dataclass."""
        backend = MockBackend()

        audio_data = backend.decode(audio_1s_mono_16k)

        assert isinstance(audio_data, AudioData)
        assert hasattr(audio_data, "data")
        assert hasattr(audio_data, "sample_rate")
        assert hasattr(audio_data, "channels")

    def test_decode_returns_numpy_array(self, audio_1s_mono_16k):
        """decode must return float32 numpy array."""
        backend = MockBackend()

        audio_data = backend.decode(audio_1s_mono_16k)

        assert isinstance(audio_data.data, np.ndarray)
        assert audio_data.data.dtype == np.float32

    def test_decode_mono_shape(self, audio_1s_mono_16k):
        """Mono audio must have shape (samples,)."""
        backend = MockBackend()

        audio_data = backend.decode(audio_1s_mono_16k, mono=True)

        assert audio_data.data.ndim == 1
        assert audio_data.channels == 1

    def test_decode_respects_target_sr(self, audio_1s_mono_16k):
        """decode must respect target_sr parameter."""
        backend = MockBackend()

        audio_data_8k = backend.decode(audio_1s_mono_16k, target_sr=8000)
        audio_data_16k = backend.decode(audio_1s_mono_16k, target_sr=16000)

        assert audio_data_8k.sample_rate == 8000
        assert audio_data_16k.sample_rate == 16000

    def test_decode_raises_on_missing_file(self, tmp_path):
        """decode must raise FileNotFoundError for missing files."""
        backend = MockBackend()
        missing_file = tmp_path / "does_not_exist.wav"

        with pytest.raises(FileNotFoundError):
            backend.decode(missing_file)

    def test_backend_name_property(self):
        """Backend must have a name property."""
        backend = MockBackend()

        assert isinstance(backend.name, str)
        assert len(backend.name) > 0

    def test_backend_repr(self):
        """Backend must have useful repr."""
        backend = MockBackend()

        repr_str = repr(backend)
        assert "Mock" in repr_str or "mock" in repr_str


class TestAudioDataProperties:
    """Test AudioData dataclass properties."""

    def test_audio_data_duration_property(self):
        """AudioData.duration should compute correctly."""
        data = np.zeros(16000, dtype=np.float32)
        audio_data = AudioData(data=data, sample_rate=16000, channels=1)

        assert audio_data.duration == 1.0

    def test_audio_data_samples_property(self):
        """AudioData.samples should return number of samples."""
        data = np.zeros(16000, dtype=np.float32)
        audio_data = AudioData(data=data, sample_rate=16000, channels=1)

        assert audio_data.samples == 16000

    def test_audio_data_stereo_shape(self):
        """Stereo AudioData should have shape (samples, 2)."""
        data = np.zeros((16000, 2), dtype=np.float32)
        audio_data = AudioData(data=data, sample_rate=16000, channels=2)

        assert audio_data.data.shape == (16000, 2)
        assert audio_data.channels == 2


class TestAudioInfo:
    """Test AudioInfo dataclass."""

    def test_audio_info_creation(self):
        """AudioInfo should be creatable with all fields."""
        info = AudioInfo(
            sample_rate=44100,
            channels=2,
            duration=3.5,
            samples=154350,
            format="mp3",
        )

        assert info.sample_rate == 44100
        assert info.channels == 2
        assert info.duration == 3.5
        assert info.samples == 154350
        assert info.format == "mp3"
