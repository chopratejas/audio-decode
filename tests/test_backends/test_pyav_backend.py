"""
Comprehensive tests for the PyAV backend.

Tests follow TDD principles - these tests define the expected behavior
of the PyAVBackend before implementation.
"""

from pathlib import Path

import numpy as np
import pytest

from audiodecode.backends.base import AudioData, AudioInfo
from audiodecode.backends.pyav_backend import PyAVBackend


class TestFormatSupport:
    """Test format detection and support."""

    def test_supports_mp3_format(self, tmp_path):
        """Should support .mp3 files."""
        backend = PyAVBackend()
        mp3_file = tmp_path / "test.mp3"
        mp3_file.touch()

        assert backend.supports_format(mp3_file) is True

    def test_supports_aac_format(self, tmp_path):
        """Should support .aac files."""
        backend = PyAVBackend()
        aac_file = tmp_path / "test.aac"
        aac_file.touch()

        assert backend.supports_format(aac_file) is True

    def test_supports_m4a_format(self, tmp_path):
        """Should support .m4a files."""
        backend = PyAVBackend()
        m4a_file = tmp_path / "test.m4a"
        m4a_file.touch()

        assert backend.supports_format(m4a_file) is True

    def test_supports_ogg_format(self, tmp_path):
        """Should support .ogg files (Opus)."""
        backend = PyAVBackend()
        ogg_file = tmp_path / "test.ogg"
        ogg_file.touch()

        assert backend.supports_format(ogg_file) is True

    def test_supports_opus_format(self, tmp_path):
        """Should support .opus files."""
        backend = PyAVBackend()
        opus_file = tmp_path / "test.opus"
        opus_file.touch()

        assert backend.supports_format(opus_file) is True

    def test_does_not_support_wav_format(self, tmp_path):
        """Should not support .wav files (use soundfile backend)."""
        backend = PyAVBackend()
        wav_file = tmp_path / "test.wav"
        wav_file.touch()

        assert backend.supports_format(wav_file) is False

    def test_does_not_support_flac_format(self, tmp_path):
        """Should not support .flac files (use soundfile backend)."""
        backend = PyAVBackend()
        flac_file = tmp_path / "test.flac"
        flac_file.touch()

        assert backend.supports_format(flac_file) is False

    def test_format_detection_is_case_insensitive(self, tmp_path):
        """Format detection should be case insensitive."""
        backend = PyAVBackend()

        mp3_upper = tmp_path / "test.MP3"
        mp3_upper.touch()
        assert backend.supports_format(mp3_upper) is True

        m4a_mixed = tmp_path / "test.M4a"
        m4a_mixed.touch()
        assert backend.supports_format(m4a_mixed) is True


class TestFileInfo:
    """Test get_info() method for metadata extraction."""

    def test_get_info_returns_audio_info(self, mp3_1s_mono_16k):
        """Should return AudioInfo dataclass."""
        backend = PyAVBackend()

        info = backend.get_info(mp3_1s_mono_16k)

        assert isinstance(info, AudioInfo)

    def test_get_info_extracts_correct_sample_rate_16k(self, mp3_1s_mono_16k):
        """Should extract correct sample rate from 16kHz file."""
        backend = PyAVBackend()

        info = backend.get_info(mp3_1s_mono_16k)

        assert info.sample_rate == 16000

    def test_get_info_detects_mono_audio(self, mp3_1s_mono_16k):
        """Should correctly detect mono audio (channels=1)."""
        backend = PyAVBackend()

        info = backend.get_info(mp3_1s_mono_16k)

        assert info.channels == 1

    def test_get_info_detects_stereo_audio(self, mp3_1s_stereo_44k):
        """Should correctly detect stereo audio (channels=2)."""
        backend = PyAVBackend()

        info = backend.get_info(mp3_1s_stereo_44k)

        assert info.channels == 2

    def test_get_info_calculates_duration_1_second(self, mp3_1s_mono_16k):
        """Should calculate correct duration for 1-second file."""
        backend = PyAVBackend()

        info = backend.get_info(mp3_1s_mono_16k)

        # Allow small floating point tolerance
        assert abs(info.duration - 1.0) < 0.1

    def test_get_info_calculates_duration_10_seconds(self, mp3_10s_mono_16k):
        """Should calculate correct duration for 10-second file."""
        backend = PyAVBackend()

        info = backend.get_info(mp3_10s_mono_16k)

        # Allow small floating point tolerance
        assert abs(info.duration - 10.0) < 0.1

    def test_get_info_returns_correct_number_of_samples(self, mp3_1s_mono_16k):
        """Should return correct total number of samples."""
        backend = PyAVBackend()

        info = backend.get_info(mp3_1s_mono_16k)

        # 1 second at 16kHz = 16000 samples (allow for MP3 encoder padding)
        assert abs(info.samples - 16000) < 1000

    def test_get_info_extracts_correct_format_mp3(self, mp3_1s_mono_16k):
        """Should identify format as 'mp3'."""
        backend = PyAVBackend()

        info = backend.get_info(mp3_1s_mono_16k)

        assert info.format.lower() == "mp3"

    def test_get_info_raises_file_not_found_error(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        backend = PyAVBackend()
        missing_file = tmp_path / "does_not_exist.mp3"

        with pytest.raises(FileNotFoundError):
            backend.get_info(missing_file)

    def test_get_info_raises_error_for_unsupported_format(self, fixtures_dir):
        """Should raise ValueError for unsupported formats."""
        backend = PyAVBackend()
        wav_file = fixtures_dir / "test_1s_mono_16000.wav"

        with pytest.raises(ValueError):
            backend.get_info(wav_file)


class TestDecoding:
    """Test decode() method for audio decoding."""

    def test_decode_returns_audio_data(self, mp3_1s_mono_16k):
        """Should return AudioData dataclass."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_mono_16k)

        assert isinstance(audio_data, AudioData)

    def test_decode_returns_numpy_array(self, mp3_1s_mono_16k):
        """Should return numpy array in data field."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_mono_16k)

        assert isinstance(audio_data.data, np.ndarray)

    def test_decode_returns_float32_dtype(self, mp3_1s_mono_16k):
        """Should return float32 dtype for audio data."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_mono_16k)

        assert audio_data.data.dtype == np.float32

    def test_decode_mono_returns_1d_array(self, mp3_1s_mono_16k):
        """Mono audio should have shape (samples,)."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_mono_16k)

        assert audio_data.data.ndim == 1
        assert audio_data.channels == 1

    def test_decode_stereo_returns_2d_array(self, mp3_1s_stereo_44k):
        """Stereo audio should have shape (samples, 2)."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_stereo_44k)

        assert audio_data.data.ndim == 2
        assert audio_data.data.shape[1] == 2
        assert audio_data.channels == 2

    def test_decode_mono_correct_number_of_samples(self, mp3_1s_mono_16k):
        """Decoded mono audio should have correct sample count."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_mono_16k)

        # 1 second at 16kHz = 16000 samples (allow for MP3 encoder padding)
        assert abs(audio_data.data.shape[0] - 16000) < 1000

    def test_decode_stereo_correct_number_of_samples(self, mp3_1s_stereo_44k):
        """Decoded stereo audio should have correct sample count."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_stereo_44k)

        # 1 second at the file's sample rate (allow for MP3 encoder padding)
        # The fixture might be 16kHz or 44.1kHz depending on what's available
        expected_samples = audio_data.sample_rate * 1.0
        assert abs(audio_data.data.shape[0] - expected_samples) < 1000

    def test_decode_preserves_original_sample_rate(self, mp3_1s_mono_16k):
        """Should preserve original sample rate when target_sr=None."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_mono_16k, target_sr=None)

        assert audio_data.sample_rate == 16000

    def test_decode_with_target_sr_resamples_audio(self, mp3_1s_mono_16k):
        """Should resample audio when target_sr is specified."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_mono_16k, target_sr=8000)

        assert audio_data.sample_rate == 8000
        # Should have approximately 8000 samples for 1 second at 8kHz
        assert abs(audio_data.data.shape[0] - 8000) < 100

    def test_decode_upsampling(self, mp3_1s_mono_16k):
        """Should handle upsampling correctly."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_mono_16k, target_sr=48000)

        assert audio_data.sample_rate == 48000
        # Should have approximately 48000 samples for 1 second at 48kHz
        assert abs(audio_data.data.shape[0] - 48000) < 1000

    def test_decode_stereo_to_mono_conversion(self, mp3_1s_stereo_44k):
        """Should convert stereo to mono when mono=True."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_stereo_44k, mono=True)

        assert audio_data.data.ndim == 1
        assert audio_data.channels == 1

    def test_decode_stereo_preserves_stereo_when_mono_false(self, mp3_1s_stereo_44k):
        """Should keep stereo when mono=False."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_stereo_44k, mono=False)

        assert audio_data.data.ndim == 2
        assert audio_data.channels == 2

    def test_decode_mono_stays_mono(self, mp3_1s_mono_16k):
        """Mono audio should stay mono regardless of mono parameter."""
        backend = PyAVBackend()

        audio_data_false = backend.decode(mp3_1s_mono_16k, mono=False)
        audio_data_true = backend.decode(mp3_1s_mono_16k, mono=True)

        assert audio_data_false.channels == 1
        assert audio_data_true.channels == 1

    def test_decode_values_are_normalized(self, mp3_1s_mono_16k):
        """Decoded values should be normalized to [-1.0, 1.0]."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_mono_16k)

        assert np.all(audio_data.data >= -1.0)
        assert np.all(audio_data.data <= 1.0)

    def test_decode_with_offset_starts_at_correct_position(self, mp3_10s_mono_16k):
        """Should start decoding from specified offset."""
        backend = PyAVBackend()

        # Decode from 5 seconds in
        audio_data = backend.decode(mp3_10s_mono_16k, offset=5.0)

        # Should have approximately 5 seconds of audio (10s - 5s offset)
        # MP3 seeking is not frame-accurate, so allow larger tolerance
        expected_samples = 5 * 16000
        assert abs(audio_data.data.shape[0] - expected_samples) < 5000

    def test_decode_with_duration_limits_length(self, mp3_10s_mono_16k):
        """Should decode only specified duration."""
        backend = PyAVBackend()

        # Decode only 2 seconds
        audio_data = backend.decode(mp3_10s_mono_16k, duration=2.0)

        # Should have approximately 2 seconds of audio
        expected_samples = 2 * 16000
        assert abs(audio_data.data.shape[0] - expected_samples) < 1000

    def test_decode_with_offset_and_duration(self, mp3_10s_mono_16k):
        """Should handle both offset and duration correctly."""
        backend = PyAVBackend()

        # Decode 3 seconds starting from 2 seconds in
        audio_data = backend.decode(mp3_10s_mono_16k, offset=2.0, duration=3.0)

        # Should have approximately 3 seconds of audio
        # MP3 seeking is not frame-accurate, so allow larger tolerance
        expected_samples = 3 * 16000
        assert abs(audio_data.data.shape[0] - expected_samples) < 5000

    def test_decode_raises_file_not_found_error(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        backend = PyAVBackend()
        missing_file = tmp_path / "does_not_exist.mp3"

        with pytest.raises(FileNotFoundError):
            backend.decode(missing_file)

    def test_decode_raises_error_for_unsupported_format(self, fixtures_dir):
        """Should raise ValueError for unsupported formats."""
        backend = PyAVBackend()
        wav_file = fixtures_dir / "test_1s_mono_16000.wav"

        with pytest.raises(ValueError):
            backend.decode(wav_file)


class TestQuality:
    """Test decoding quality and accuracy."""

    @pytest.mark.skipif(not pytest.importorskip("librosa", reason="librosa not installed"), reason="librosa not installed")
    def test_decoded_output_matches_librosa_mp3(self, mp3_1s_mono_16k, has_librosa):
        """Decoded MP3 should match librosa output within tolerance."""
        if not has_librosa:
            pytest.skip("librosa not installed")

        import librosa

        backend = PyAVBackend()

        # Decode with our backend
        audio_data = backend.decode(mp3_1s_mono_16k)

        # Decode with librosa
        librosa_audio, librosa_sr = librosa.load(mp3_1s_mono_16k, sr=None, mono=False)

        # Compare sample rates
        assert audio_data.sample_rate == librosa_sr

        # For MP3, we can't expect bit-perfect accuracy due to encoder/decoder differences
        # but the decoded audio should be very similar
        # Account for potential padding differences
        min_len = min(len(audio_data.data), len(librosa_audio))

        np.testing.assert_allclose(
            audio_data.data[:min_len],
            librosa_audio[:min_len],
            rtol=0.1,
            atol=0.01,
            err_msg="PyAV output should closely match librosa output for MP3"
        )

    def test_decode_mp3_consistency(self, mp3_1s_mono_16k):
        """Multiple MP3 decodes should be consistent."""
        backend = PyAVBackend()

        # Decode twice
        audio_data_1 = backend.decode(mp3_1s_mono_16k)
        audio_data_2 = backend.decode(mp3_1s_mono_16k)

        # Should be exactly identical
        np.testing.assert_array_equal(
            audio_data_1.data,
            audio_data_2.data,
            err_msg="Multiple MP3 decodes should be identical"
        )

    def test_resampling_preserves_audio_content(self, mp3_1s_mono_16k):
        """Resampling should preserve audio content (within tolerance)."""
        backend = PyAVBackend()

        # Decode at different sample rates
        audio_16k = backend.decode(mp3_1s_mono_16k, target_sr=16000)
        audio_8k = backend.decode(mp3_1s_mono_16k, target_sr=8000)

        # Duration should be approximately the same
        assert abs(audio_16k.duration - audio_8k.duration) < 0.1

        # Both should have valid audio data
        assert not np.all(audio_16k.data == 0)
        assert not np.all(audio_8k.data == 0)

    def test_stereo_to_mono_averages_channels(self, mp3_1s_stereo_44k):
        """Stereo to mono conversion should average channels."""
        backend = PyAVBackend()

        # Decode as stereo
        stereo = backend.decode(mp3_1s_stereo_44k, mono=False)

        # Decode as mono
        mono = backend.decode(mp3_1s_stereo_44k, mono=True)

        # Manual average of stereo channels
        expected_mono = stereo.data.mean(axis=1).astype(np.float32)

        # Should be close to manual average
        np.testing.assert_allclose(
            mono.data,
            expected_mono,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Mono conversion should average stereo channels"
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_decode_with_zero_offset(self, mp3_1s_mono_16k):
        """Offset of 0.0 should decode from beginning."""
        backend = PyAVBackend()

        audio_with_offset = backend.decode(mp3_1s_mono_16k, offset=0.0)
        audio_without_offset = backend.decode(mp3_1s_mono_16k)

        np.testing.assert_array_equal(
            audio_with_offset.data,
            audio_without_offset.data
        )

    def test_decode_with_offset_beyond_file_length(self, mp3_1s_mono_16k):
        """Offset beyond file length should return empty or small array."""
        backend = PyAVBackend()

        # Try to decode starting from 10 seconds into a 1-second file
        audio_data = backend.decode(mp3_1s_mono_16k, offset=10.0)

        # Should return empty array or very small array
        assert audio_data.data.shape[0] < 1000

    def test_decode_with_very_large_duration(self, mp3_1s_mono_16k):
        """Duration longer than file should decode entire file."""
        backend = PyAVBackend()

        audio_large_duration = backend.decode(mp3_1s_mono_16k, duration=1000.0)
        audio_normal = backend.decode(mp3_1s_mono_16k)

        # Should be same length (entire file)
        assert abs(audio_large_duration.data.shape[0] - audio_normal.data.shape[0]) < 100

    def test_decode_with_very_small_duration(self, mp3_1s_mono_16k):
        """Very small duration should return minimal samples."""
        backend = PyAVBackend()

        # Decode only 0.01 seconds (10ms)
        audio_data = backend.decode(mp3_1s_mono_16k, duration=0.01)

        # Should have approximately 160 samples (16000 Hz * 0.01s)
        expected_samples = int(16000 * 0.01)
        assert abs(audio_data.data.shape[0] - expected_samples) < 100

    def test_decode_combines_resampling_and_mono_conversion(self, mp3_1s_stereo_44k):
        """Should handle both resampling and mono conversion together."""
        backend = PyAVBackend()

        audio_data = backend.decode(mp3_1s_stereo_44k, target_sr=16000, mono=True)

        assert audio_data.sample_rate == 16000
        assert audio_data.channels == 1
        assert audio_data.data.ndim == 1
        # Should have approximately 16000 samples
        assert abs(audio_data.data.shape[0] - 16000) < 1000

    def test_decode_combines_all_parameters(self, mp3_10s_mono_16k):
        """Should handle offset, duration, resampling, and mono together."""
        backend = PyAVBackend()

        audio_data = backend.decode(
            mp3_10s_mono_16k,
            offset=2.0,
            duration=3.0,
            target_sr=8000,
            mono=True
        )

        assert audio_data.sample_rate == 8000
        assert audio_data.channels == 1
        # Should have approximately 24000 samples (3 seconds at 8kHz)
        # MP3 seeking is not frame-accurate, so allow larger tolerance
        expected_samples = 3 * 8000
        assert abs(audio_data.data.shape[0] - expected_samples) < 5000

    def test_backend_name_property(self):
        """Backend should have correct name."""
        backend = PyAVBackend()

        assert backend.name == "pyav"

    def test_backend_repr(self):
        """Backend should have useful repr."""
        backend = PyAVBackend()

        repr_str = repr(backend)
        assert "PyAV" in repr_str or "pyav" in repr_str

    def test_decode_empty_file_handling(self, tmp_path):
        """Should handle empty or corrupted files gracefully."""
        backend = PyAVBackend()

        # Create an empty file
        empty_file = tmp_path / "empty.mp3"
        empty_file.write_bytes(b"")

        with pytest.raises((ValueError, RuntimeError, Exception)):
            backend.decode(empty_file)

    def test_decode_corrupted_file_handling(self, tmp_path):
        """Should raise appropriate error for corrupted files."""
        backend = PyAVBackend()

        # Create a file with invalid MP3 data
        corrupted_file = tmp_path / "corrupted.mp3"
        corrupted_file.write_bytes(b"ID3" + b"\x00" * 100)

        with pytest.raises((ValueError, RuntimeError, Exception)):
            backend.decode(corrupted_file)

    def test_get_info_empty_file_handling(self, tmp_path):
        """Should handle empty files in get_info gracefully."""
        backend = PyAVBackend()

        empty_file = tmp_path / "empty.mp3"
        empty_file.write_bytes(b"")

        with pytest.raises((ValueError, RuntimeError, Exception)):
            backend.get_info(empty_file)


class TestBackendProperties:
    """Test backend properties and metadata."""

    def test_backend_name_is_pyav(self):
        """Backend name should be 'pyav'."""
        backend = PyAVBackend()

        assert backend.name == "pyav"
        assert isinstance(backend.name, str)

    def test_backend_repr_contains_name(self):
        """Backend repr should contain the backend name."""
        backend = PyAVBackend()

        repr_str = repr(backend)
        assert "pyav" in repr_str.lower()

    def test_backend_is_reusable(self, mp3_1s_mono_16k):
        """Same backend instance should be reusable for multiple operations."""
        backend = PyAVBackend()

        # Use backend multiple times
        info1 = backend.get_info(mp3_1s_mono_16k)
        audio1 = backend.decode(mp3_1s_mono_16k)
        info2 = backend.get_info(mp3_1s_mono_16k)
        audio2 = backend.decode(mp3_1s_mono_16k)

        # Results should be consistent
        assert info1.sample_rate == info2.sample_rate
        np.testing.assert_array_equal(audio1.data, audio2.data)


# Additional fixtures for MP3 files
@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures" / "audio"


@pytest.fixture
def mp3_1s_mono_16k(fixtures_dir: Path) -> Path:
    """1 second mono MP3 at 16kHz."""
    # Try both naming patterns
    mp3_file = fixtures_dir / "wav_1s_mono_16000.mp3"
    if not mp3_file.exists():
        mp3_file = fixtures_dir / "test_1s_mono_16000.mp3"
    return mp3_file


@pytest.fixture
def mp3_1s_stereo_44k(fixtures_dir: Path) -> Path:
    """1 second stereo MP3 at 44.1kHz."""
    mp3_file = fixtures_dir / "wav_1s_stereo_44100.mp3"
    if not mp3_file.exists():
        mp3_file = fixtures_dir / "test_1s_stereo_44100.mp3"
    return mp3_file


@pytest.fixture
def mp3_10s_mono_16k(fixtures_dir: Path) -> Path:
    """10 second mono MP3 at 16kHz."""
    mp3_file = fixtures_dir / "test_10s_mono_16000.mp3"
    if not mp3_file.exists():
        pytest.skip("10-second MP3 fixture not available")
    return mp3_file
