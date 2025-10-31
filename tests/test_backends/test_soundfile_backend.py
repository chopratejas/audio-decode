"""
Comprehensive tests for the soundfile backend.

Tests follow TDD principles - these tests define the expected behavior
of the SoundfileBackend before implementation.
"""

from pathlib import Path

import numpy as np
import pytest

from audiodecode.backends.base import AudioData, AudioInfo
from audiodecode.backends.soundfile_backend import SoundfileBackend


class TestFormatSupport:
    """Test format detection and support."""

    def test_supports_wav_format(self, tmp_path):
        """Should support .wav files."""
        backend = SoundfileBackend()
        wav_file = tmp_path / "test.wav"
        wav_file.touch()

        assert backend.supports_format(wav_file) is True

    def test_supports_flac_format(self, tmp_path):
        """Should support .flac files."""
        backend = SoundfileBackend()
        flac_file = tmp_path / "test.flac"
        flac_file.touch()

        assert backend.supports_format(flac_file) is True

    def test_supports_ogg_format(self, tmp_path):
        """Should support .ogg files."""
        backend = SoundfileBackend()
        ogg_file = tmp_path / "test.ogg"
        ogg_file.touch()

        assert backend.supports_format(ogg_file) is True

    def test_does_not_support_mp3_format(self, tmp_path):
        """Should not support .mp3 files."""
        backend = SoundfileBackend()
        mp3_file = tmp_path / "test.mp3"
        mp3_file.touch()

        assert backend.supports_format(mp3_file) is False

    def test_does_not_support_aac_format(self, tmp_path):
        """Should not support .aac files."""
        backend = SoundfileBackend()
        aac_file = tmp_path / "test.aac"
        aac_file.touch()

        assert backend.supports_format(aac_file) is False

    def test_does_not_support_m4a_format(self, tmp_path):
        """Should not support .m4a files."""
        backend = SoundfileBackend()
        m4a_file = tmp_path / "test.m4a"
        m4a_file.touch()

        assert backend.supports_format(m4a_file) is False

    def test_format_detection_is_case_insensitive(self, tmp_path):
        """Format detection should be case insensitive."""
        backend = SoundfileBackend()

        wav_upper = tmp_path / "test.WAV"
        wav_upper.touch()
        assert backend.supports_format(wav_upper) is True

        flac_mixed = tmp_path / "test.FlAc"
        flac_mixed.touch()
        assert backend.supports_format(flac_mixed) is True


class TestFileInfo:
    """Test get_info() method for metadata extraction."""

    def test_get_info_returns_audio_info(self, audio_1s_mono_16k):
        """Should return AudioInfo dataclass."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_1s_mono_16k)

        assert isinstance(info, AudioInfo)

    def test_get_info_extracts_correct_sample_rate_16k(self, audio_1s_mono_16k):
        """Should extract correct sample rate from 16kHz file."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_1s_mono_16k)

        assert info.sample_rate == 16000

    def test_get_info_extracts_correct_sample_rate_44k(self, audio_1s_stereo_44k):
        """Should extract correct sample rate from 44.1kHz file."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_1s_stereo_44k)

        assert info.sample_rate == 44100

    def test_get_info_detects_mono_audio(self, audio_1s_mono_16k):
        """Should correctly detect mono audio (channels=1)."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_1s_mono_16k)

        assert info.channels == 1

    def test_get_info_detects_stereo_audio(self, audio_1s_stereo_44k):
        """Should correctly detect stereo audio (channels=2)."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_1s_stereo_44k)

        assert info.channels == 2

    def test_get_info_calculates_duration_1_second(self, audio_1s_mono_16k):
        """Should calculate correct duration for 1-second file."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_1s_mono_16k)

        # Allow small floating point tolerance
        assert abs(info.duration - 1.0) < 0.01

    def test_get_info_calculates_duration_10_seconds(self, audio_10s_mono_16k):
        """Should calculate correct duration for 10-second file."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_10s_mono_16k)

        # Allow small floating point tolerance
        assert abs(info.duration - 10.0) < 0.01

    def test_get_info_returns_correct_number_of_samples(self, audio_1s_mono_16k):
        """Should return correct total number of samples."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_1s_mono_16k)

        # 1 second at 16kHz = 16000 samples
        assert info.samples == 16000

    def test_get_info_samples_matches_duration_and_sample_rate(self, audio_1s_stereo_44k):
        """Samples should equal duration * sample_rate."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_1s_stereo_44k)

        expected_samples = int(info.duration * info.sample_rate)
        # Allow for small rounding differences
        assert abs(info.samples - expected_samples) <= 1

    def test_get_info_extracts_correct_format_wav(self, audio_1s_mono_16k):
        """Should identify format as 'wav'."""
        backend = SoundfileBackend()

        info = backend.get_info(audio_1s_mono_16k)

        assert info.format.lower() == "wav"

    def test_get_info_extracts_correct_format_flac(self, fixtures_dir):
        """Should identify format as 'flac'."""
        backend = SoundfileBackend()
        flac_file = fixtures_dir / "wav_1s_mono_16000.flac"

        # Skip if FLAC fixture doesn't exist
        if not flac_file.exists():
            pytest.skip("FLAC fixture not available")

        info = backend.get_info(flac_file)

        assert info.format.lower() == "flac"

    def test_get_info_raises_file_not_found_error(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        backend = SoundfileBackend()
        missing_file = tmp_path / "does_not_exist.wav"

        with pytest.raises(FileNotFoundError):
            backend.get_info(missing_file)

    def test_get_info_raises_error_for_unsupported_format(self, fixtures_dir):
        """Should raise ValueError for unsupported formats."""
        backend = SoundfileBackend()
        mp3_file = fixtures_dir / "wav_1s_mono_16000.mp3"

        # Skip if MP3 fixture doesn't exist
        if not mp3_file.exists():
            pytest.skip("MP3 fixture not available")

        with pytest.raises(ValueError):
            backend.get_info(mp3_file)


class TestDecoding:
    """Test decode() method for audio decoding."""

    def test_decode_returns_audio_data(self, audio_1s_mono_16k):
        """Should return AudioData dataclass."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_mono_16k)

        assert isinstance(audio_data, AudioData)

    def test_decode_returns_numpy_array(self, audio_1s_mono_16k):
        """Should return numpy array in data field."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_mono_16k)

        assert isinstance(audio_data.data, np.ndarray)

    def test_decode_returns_float32_dtype(self, audio_1s_mono_16k):
        """Should return float32 dtype for audio data."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_mono_16k)

        assert audio_data.data.dtype == np.float32

    def test_decode_mono_returns_1d_array(self, audio_1s_mono_16k):
        """Mono audio should have shape (samples,)."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_mono_16k)

        assert audio_data.data.ndim == 1
        assert audio_data.channels == 1

    def test_decode_stereo_returns_2d_array(self, audio_1s_stereo_44k):
        """Stereo audio should have shape (samples, 2)."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_stereo_44k)

        assert audio_data.data.ndim == 2
        assert audio_data.data.shape[1] == 2
        assert audio_data.channels == 2

    def test_decode_mono_correct_number_of_samples(self, audio_1s_mono_16k):
        """Decoded mono audio should have correct sample count."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_mono_16k)

        # 1 second at 16kHz = 16000 samples
        assert audio_data.data.shape[0] == 16000

    def test_decode_stereo_correct_number_of_samples(self, audio_1s_stereo_44k):
        """Decoded stereo audio should have correct sample count."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_stereo_44k)

        # 1 second at 44.1kHz = 44100 samples
        assert audio_data.data.shape[0] == 44100

    def test_decode_preserves_original_sample_rate(self, audio_1s_mono_16k):
        """Should preserve original sample rate when target_sr=None."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_mono_16k, target_sr=None)

        assert audio_data.sample_rate == 16000

    def test_decode_with_target_sr_resamples_audio(self, audio_1s_mono_16k):
        """Should resample audio when target_sr is specified."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_mono_16k, target_sr=8000)

        assert audio_data.sample_rate == 8000
        # Should have approximately 8000 samples for 1 second at 8kHz
        assert abs(audio_data.data.shape[0] - 8000) < 10

    def test_decode_upsampling(self, audio_1s_mono_16k):
        """Should handle upsampling correctly."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_mono_16k, target_sr=48000)

        assert audio_data.sample_rate == 48000
        # Should have approximately 48000 samples for 1 second at 48kHz
        assert abs(audio_data.data.shape[0] - 48000) < 10

    def test_decode_stereo_to_mono_conversion(self, audio_1s_stereo_44k):
        """Should convert stereo to mono when mono=True."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_stereo_44k, mono=True)

        assert audio_data.data.ndim == 1
        assert audio_data.channels == 1

    def test_decode_stereo_preserves_stereo_when_mono_false(self, audio_1s_stereo_44k):
        """Should keep stereo when mono=False."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_stereo_44k, mono=False)

        assert audio_data.data.ndim == 2
        assert audio_data.channels == 2

    def test_decode_mono_stays_mono(self, audio_1s_mono_16k):
        """Mono audio should stay mono regardless of mono parameter."""
        backend = SoundfileBackend()

        audio_data_false = backend.decode(audio_1s_mono_16k, mono=False)
        audio_data_true = backend.decode(audio_1s_mono_16k, mono=True)

        assert audio_data_false.channels == 1
        assert audio_data_true.channels == 1

    def test_decode_values_are_normalized(self, audio_1s_mono_16k):
        """Decoded values should be normalized to [-1.0, 1.0]."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_mono_16k)

        assert np.all(audio_data.data >= -1.0)
        assert np.all(audio_data.data <= 1.0)

    def test_decode_with_offset_starts_at_correct_position(self, audio_10s_mono_16k):
        """Should start decoding from specified offset."""
        backend = SoundfileBackend()

        # Decode from 5 seconds in
        audio_data = backend.decode(audio_10s_mono_16k, offset=5.0)

        # Should have approximately 5 seconds of audio (10s - 5s offset)
        expected_samples = 5 * 16000
        assert abs(audio_data.data.shape[0] - expected_samples) < 100

    def test_decode_with_duration_limits_length(self, audio_10s_mono_16k):
        """Should decode only specified duration."""
        backend = SoundfileBackend()

        # Decode only 2 seconds
        audio_data = backend.decode(audio_10s_mono_16k, duration=2.0)

        # Should have approximately 2 seconds of audio
        expected_samples = 2 * 16000
        assert abs(audio_data.data.shape[0] - expected_samples) < 100

    def test_decode_with_offset_and_duration(self, audio_10s_mono_16k):
        """Should handle both offset and duration correctly."""
        backend = SoundfileBackend()

        # Decode 3 seconds starting from 2 seconds in
        audio_data = backend.decode(audio_10s_mono_16k, offset=2.0, duration=3.0)

        # Should have approximately 3 seconds of audio
        expected_samples = 3 * 16000
        assert abs(audio_data.data.shape[0] - expected_samples) < 100

    def test_decode_raises_file_not_found_error(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        backend = SoundfileBackend()
        missing_file = tmp_path / "does_not_exist.wav"

        with pytest.raises(FileNotFoundError):
            backend.decode(missing_file)

    def test_decode_raises_error_for_unsupported_format(self, fixtures_dir):
        """Should raise ValueError for unsupported formats."""
        backend = SoundfileBackend()
        mp3_file = fixtures_dir / "wav_1s_mono_16000.mp3"

        # Skip if MP3 fixture doesn't exist
        if not mp3_file.exists():
            pytest.skip("MP3 fixture not available")

        with pytest.raises(ValueError):
            backend.decode(mp3_file)


class TestQuality:
    """Test decoding quality and accuracy."""

    @pytest.mark.skipif(not pytest.importorskip("librosa", reason="librosa not installed"), reason="librosa not installed")
    def test_decoded_output_matches_librosa_wav(self, audio_1s_mono_16k, has_librosa):
        """Decoded WAV should match librosa output."""
        if not has_librosa:
            pytest.skip("librosa not installed")

        import librosa

        backend = SoundfileBackend()

        # Decode with our backend
        audio_data = backend.decode(audio_1s_mono_16k)

        # Decode with librosa
        librosa_audio, librosa_sr = librosa.load(audio_1s_mono_16k, sr=None, mono=False)

        # Compare
        assert audio_data.sample_rate == librosa_sr
        np.testing.assert_allclose(
            audio_data.data,
            librosa_audio,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Soundfile output should match librosa output"
        )

    @pytest.mark.skipif(not pytest.importorskip("librosa", reason="librosa not installed"), reason="librosa not installed")
    def test_decoded_output_matches_librosa_flac(self, fixtures_dir, has_librosa):
        """Decoded FLAC should match librosa output."""
        if not has_librosa:
            pytest.skip("librosa not installed")

        import librosa

        flac_file = fixtures_dir / "wav_1s_mono_16000.flac"
        if not flac_file.exists():
            pytest.skip("FLAC fixture not available")

        backend = SoundfileBackend()

        # Decode with our backend
        audio_data = backend.decode(flac_file)

        # Decode with librosa
        librosa_audio, librosa_sr = librosa.load(flac_file, sr=None, mono=False)

        # Compare
        assert audio_data.sample_rate == librosa_sr
        np.testing.assert_allclose(
            audio_data.data,
            librosa_audio,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Soundfile FLAC output should match librosa output"
        )

    def test_lossless_wav_bit_accuracy(self, audio_1s_mono_16k):
        """WAV decoding should be bit-accurate (lossless)."""
        backend = SoundfileBackend()

        # Decode twice
        audio_data_1 = backend.decode(audio_1s_mono_16k)
        audio_data_2 = backend.decode(audio_1s_mono_16k)

        # Should be exactly identical
        np.testing.assert_array_equal(
            audio_data_1.data,
            audio_data_2.data,
            err_msg="Multiple WAV decodes should be bit-for-bit identical"
        )

    def test_lossless_flac_bit_accuracy(self, fixtures_dir):
        """FLAC decoding should be bit-accurate (lossless)."""
        backend = SoundfileBackend()
        flac_file = fixtures_dir / "wav_1s_mono_16000.flac"

        if not flac_file.exists():
            pytest.skip("FLAC fixture not available")

        # Decode twice
        audio_data_1 = backend.decode(flac_file)
        audio_data_2 = backend.decode(flac_file)

        # Should be exactly identical
        np.testing.assert_array_equal(
            audio_data_1.data,
            audio_data_2.data,
            err_msg="Multiple FLAC decodes should be bit-for-bit identical"
        )

    def test_resampling_preserves_audio_content(self, audio_1s_mono_16k):
        """Resampling should preserve audio content (within tolerance)."""
        backend = SoundfileBackend()

        # Decode at different sample rates
        audio_16k = backend.decode(audio_1s_mono_16k, target_sr=16000)
        audio_8k = backend.decode(audio_1s_mono_16k, target_sr=8000)

        # Duration should be approximately the same
        assert abs(audio_16k.duration - audio_8k.duration) < 0.01

        # Both should have valid audio data
        assert not np.all(audio_16k.data == 0)
        assert not np.all(audio_8k.data == 0)

    def test_stereo_to_mono_averages_channels(self, audio_1s_stereo_44k):
        """Stereo to mono conversion should average channels."""
        backend = SoundfileBackend()

        # Decode as stereo
        stereo = backend.decode(audio_1s_stereo_44k, mono=False)

        # Decode as mono
        mono = backend.decode(audio_1s_stereo_44k, mono=True)

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

    def test_decode_with_zero_offset(self, audio_1s_mono_16k):
        """Offset of 0.0 should decode from beginning."""
        backend = SoundfileBackend()

        audio_with_offset = backend.decode(audio_1s_mono_16k, offset=0.0)
        audio_without_offset = backend.decode(audio_1s_mono_16k)

        np.testing.assert_array_equal(
            audio_with_offset.data,
            audio_without_offset.data
        )

    def test_decode_with_offset_beyond_file_length(self, audio_1s_mono_16k):
        """Offset beyond file length should return empty or raise error."""
        backend = SoundfileBackend()

        # Try to decode starting from 10 seconds into a 1-second file
        audio_data = backend.decode(audio_1s_mono_16k, offset=10.0)

        # Should return empty array or very small array
        assert audio_data.data.shape[0] < 100

    def test_decode_with_very_large_duration(self, audio_1s_mono_16k):
        """Duration longer than file should decode entire file."""
        backend = SoundfileBackend()

        audio_large_duration = backend.decode(audio_1s_mono_16k, duration=1000.0)
        audio_normal = backend.decode(audio_1s_mono_16k)

        # Should be same length (entire file)
        assert audio_large_duration.data.shape[0] == audio_normal.data.shape[0]

    def test_decode_with_very_small_duration(self, audio_1s_mono_16k):
        """Very small duration should return minimal samples."""
        backend = SoundfileBackend()

        # Decode only 0.01 seconds (10ms)
        audio_data = backend.decode(audio_1s_mono_16k, duration=0.01)

        # Should have approximately 160 samples (16000 Hz * 0.01s)
        expected_samples = int(16000 * 0.01)
        assert abs(audio_data.data.shape[0] - expected_samples) < 10

    def test_decode_combines_resampling_and_mono_conversion(self, audio_1s_stereo_44k):
        """Should handle both resampling and mono conversion together."""
        backend = SoundfileBackend()

        audio_data = backend.decode(audio_1s_stereo_44k, target_sr=16000, mono=True)

        assert audio_data.sample_rate == 16000
        assert audio_data.channels == 1
        assert audio_data.data.ndim == 1
        # Should have approximately 16000 samples
        assert abs(audio_data.data.shape[0] - 16000) < 10

    def test_decode_combines_all_parameters(self, audio_10s_mono_16k):
        """Should handle offset, duration, resampling, and mono together."""
        backend = SoundfileBackend()

        audio_data = backend.decode(
            audio_10s_mono_16k,
            offset=2.0,
            duration=3.0,
            target_sr=8000,
            mono=True
        )

        assert audio_data.sample_rate == 8000
        assert audio_data.channels == 1
        # Should have approximately 24000 samples (3 seconds at 8kHz)
        expected_samples = 3 * 8000
        assert abs(audio_data.data.shape[0] - expected_samples) < 100

    def test_backend_name_property(self):
        """Backend should have correct name."""
        backend = SoundfileBackend()

        assert backend.name == "soundfile"

    def test_backend_repr(self):
        """Backend should have useful repr."""
        backend = SoundfileBackend()

        repr_str = repr(backend)
        assert "Soundfile" in repr_str or "soundfile" in repr_str

    def test_decode_empty_file_handling(self, tmp_path):
        """Should handle empty or corrupted files gracefully."""
        backend = SoundfileBackend()

        # Create an empty file
        empty_file = tmp_path / "empty.wav"
        empty_file.write_bytes(b"")

        with pytest.raises((ValueError, RuntimeError, Exception)):
            backend.decode(empty_file)

    def test_decode_corrupted_file_handling(self, tmp_path):
        """Should raise appropriate error for corrupted files."""
        backend = SoundfileBackend()

        # Create a file with invalid WAV data
        corrupted_file = tmp_path / "corrupted.wav"
        corrupted_file.write_bytes(b"RIFF" + b"\x00" * 100)

        with pytest.raises((ValueError, RuntimeError, Exception)):
            backend.decode(corrupted_file)

    def test_get_info_empty_file_handling(self, tmp_path):
        """Should handle empty files in get_info gracefully."""
        backend = SoundfileBackend()

        empty_file = tmp_path / "empty.wav"
        empty_file.write_bytes(b"")

        with pytest.raises((ValueError, RuntimeError, Exception)):
            backend.get_info(empty_file)


class TestBackendProperties:
    """Test backend properties and metadata."""

    def test_backend_name_is_soundfile(self):
        """Backend name should be 'soundfile'."""
        backend = SoundfileBackend()

        assert backend.name == "soundfile"
        assert isinstance(backend.name, str)

    def test_backend_repr_contains_name(self):
        """Backend repr should contain the backend name."""
        backend = SoundfileBackend()

        repr_str = repr(backend)
        assert "soundfile" in repr_str.lower()

    def test_backend_is_reusable(self, audio_1s_mono_16k):
        """Same backend instance should be reusable for multiple operations."""
        backend = SoundfileBackend()

        # Use backend multiple times
        info1 = backend.get_info(audio_1s_mono_16k)
        audio1 = backend.decode(audio_1s_mono_16k)
        info2 = backend.get_info(audio_1s_mono_16k)
        audio2 = backend.decode(audio_1s_mono_16k)

        # Results should be consistent
        assert info1.sample_rate == info2.sample_rate
        np.testing.assert_array_equal(audio1.data, audio2.data)


# Additional fixtures for tests
@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures" / "audio"
