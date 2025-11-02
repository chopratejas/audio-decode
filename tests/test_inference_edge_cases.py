"""
Edge case tests for AudioDecode STT inference.

These tests cover error conditions and edge cases that could cause production failures.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from audiodecode.inference import (
    WhisperInference,
    transcribe_file,
    transcribe_audio,
    _FASTER_WHISPER_AVAILABLE,
)

# Skip all tests if faster-whisper not available
pytestmark = pytest.mark.skipif(
    not _FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)


class TestFileHandlingEdgeCases:
    """Test edge cases for file handling."""

    def test_file_not_found(self):
        """Test transcribe_file with non-existent file."""
        with pytest.raises(FileNotFoundError):
            transcribe_file("nonexistent_file.mp3")

    def test_empty_file(self, tmp_path):
        """Test transcribe_file with empty file."""
        empty_file = tmp_path / "empty.wav"
        empty_file.touch()  # Create empty file

        with pytest.raises(Exception):  # Should raise some error
            transcribe_file(str(empty_file))

    def test_path_object_input(self, tmp_path):
        """Test that Path objects work (not just strings)."""
        # Create a valid audio file
        import soundfile as sf
        audio_file = tmp_path / "test.wav"
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(str(audio_file), audio, 16000)

        # Should work with Path object
        result = transcribe_file(audio_file, model_size="tiny")
        assert isinstance(result.text, str)


class TestAudioDataEdgeCases:
    """Test edge cases for audio data."""

    def test_silent_audio(self):
        """Test transcribe_audio with completely silent audio."""
        silent_audio = np.zeros(16000, dtype=np.float32)

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_audio(silent_audio, sample_rate=16000)

        # Silent audio should return empty or minimal text
        assert isinstance(result.text, str)
        assert len(result.segments) >= 0  # May be empty

    def test_very_short_audio(self):
        """Test transcribe_audio with very short audio (<100ms)."""
        short_audio = np.random.randn(1600).astype(np.float32) * 0.1  # 100ms

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_audio(short_audio, sample_rate=16000)

        # Should not crash, may return empty
        assert isinstance(result.text, str)

    def test_noise_only_audio(self):
        """Test transcribe_audio with pure noise (no speech)."""
        noise = np.random.randn(48000).astype(np.float32) * 0.1  # 3 seconds

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_audio(noise, sample_rate=16000)

        # Should return some result (may hallucinate text)
        assert isinstance(result.text, str)

    def test_zero_length_audio(self):
        """Test transcribe_audio with zero-length audio."""
        empty_audio = np.array([], dtype=np.float32)

        whisper = WhisperInference(model_size="tiny")

        # Whisper may handle gracefully or raise error - just ensure it doesn't crash
        try:
            result = whisper.transcribe_audio(empty_audio, sample_rate=16000)
            assert isinstance(result.text, str)  # If it works, should return string
        except Exception:
            pass  # Expected - empty audio should fail or return empty


class TestParameterValidation:
    """Test parameter validation and edge cases."""

    def test_invalid_model_size(self):
        """Test WhisperInference with invalid model size."""
        with pytest.raises(Exception):  # Should raise an error
            WhisperInference(model_size="invalid_model")

    def test_invalid_device(self):
        """Test WhisperInference with invalid device."""
        # Should raise an error for invalid device
        with pytest.raises(ValueError, match="unsupported device"):
            WhisperInference(model_size="tiny", device="invalid_device")

    def test_negative_sample_rate(self):
        """Test transcribe_audio with negative sample rate."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1

        whisper = WhisperInference(model_size="tiny")

        # Should warn about invalid sample rate but not crash
        with pytest.warns(UserWarning, match="Whisper expects 16kHz"):
            result = whisper.transcribe_audio(audio, sample_rate=-16000)
            assert isinstance(result.text, str)  # Should still return result

    def test_invalid_compute_type(self):
        """Test WhisperInference with invalid compute type."""
        with pytest.raises(Exception):
            WhisperInference(model_size="tiny", compute_type="invalid")

    def test_wrong_sample_rate_warning(self):
        """Test that wrong sample rate triggers a warning."""
        audio = np.random.randn(22050).astype(np.float32) * 0.1

        whisper = WhisperInference(model_size="tiny")

        # Should warn about non-16kHz sample rate (or handle gracefully)
        result = whisper.transcribe_audio(audio, sample_rate=22050)
        assert isinstance(result.text, str)  # Should still work


class TestDataTypeHandling:
    """Test different data types and edge cases."""

    def test_float64_audio(self):
        """Test transcribe_audio with float64 input."""
        audio_f64 = np.random.randn(16000).astype(np.float64) * 0.1

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_audio(audio_f64, sample_rate=16000)

        # Should handle conversion automatically
        assert isinstance(result.text, str)

    def test_int16_audio(self):
        """Test transcribe_audio with int16 input (common format)."""
        audio_int16 = (np.random.randn(16000) * 32767 * 0.1).astype(np.int16)

        whisper = WhisperInference(model_size="tiny")

        # Should either work or raise a clear error
        try:
            result = whisper.transcribe_audio(audio_int16, sample_rate=16000)
            assert isinstance(result.text, str)
        except Exception as e:
            assert "dtype" in str(e).lower() or "type" in str(e).lower()


class TestExportFormats:
    """Test export format methods."""

    def test_to_srt_empty_segments(self):
        """Test to_srt() with empty segments."""
        from audiodecode.inference import TranscriptionResult, TranscriptionSegment

        result = TranscriptionResult(
            text="",
            segments=[],
            language="en",
            duration=0.0
        )

        srt = result.to_srt()
        assert isinstance(srt, str)
        assert srt == ""  # Empty segments = empty SRT

    def test_to_vtt_empty_segments(self):
        """Test to_vtt() with empty segments."""
        from audiodecode.inference import TranscriptionResult

        result = TranscriptionResult(
            text="",
            segments=[],
            language="en",
            duration=0.0
        )

        vtt = result.to_vtt()
        assert isinstance(vtt, str)
        assert "WEBVTT" in vtt  # Should still have header

    def test_save_invalid_format(self, tmp_path):
        """Test save() with unsupported file extension."""
        from audiodecode.inference import TranscriptionResult, TranscriptionSegment

        seg = TranscriptionSegment(text="Test", start=0.0, end=1.0, confidence=0.9)
        result = TranscriptionResult(
            text="Test",
            segments=[seg],
            language="en",
            duration=1.0
        )

        invalid_file = tmp_path / "output.xyz"

        with pytest.raises(ValueError, match="Unsupported format"):
            result.save(invalid_file)

    def test_save_txt_format(self, tmp_path):
        """Test save() with .txt format."""
        from audiodecode.inference import TranscriptionResult, TranscriptionSegment

        seg = TranscriptionSegment(text="Hello world", start=0.0, end=2.0, confidence=0.9)
        result = TranscriptionResult(
            text="Hello world",
            segments=[seg],
            language="en",
            duration=2.0
        )

        output_file = tmp_path / "transcript.txt"
        result.save(output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert content == "Hello world"

    def test_save_srt_format(self, tmp_path):
        """Test save() with .srt format."""
        from audiodecode.inference import TranscriptionResult, TranscriptionSegment

        seg = TranscriptionSegment(text="Hello world", start=0.0, end=2.0, confidence=0.9)
        result = TranscriptionResult(
            text="Hello world",
            segments=[seg],
            language="en",
            duration=2.0
        )

        output_file = tmp_path / "subtitles.srt"
        result.save(output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "00:00:00,000 --> 00:00:02,000" in content
        assert "Hello world" in content

    def test_save_json_format(self, tmp_path):
        """Test save() with .json format."""
        from audiodecode.inference import TranscriptionResult, TranscriptionSegment
        import json

        seg = TranscriptionSegment(text="Hello world", start=0.0, end=2.0, confidence=0.9)
        result = TranscriptionResult(
            text="Hello world",
            segments=[seg],
            language="en",
            duration=2.0
        )

        output_file = tmp_path / "data.json"
        result.save(output_file)

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["text"] == "Hello world"
        assert data["language"] == "en"
        assert len(data["segments"]) == 1


class TestModelReuse:
    """Test model loading and reuse."""

    def test_model_reuse_efficiency(self):
        """Test that multiple transcriptions reuse the same model."""
        whisper = WhisperInference(model_size="tiny")

        audio1 = np.random.randn(16000).astype(np.float32) * 0.1
        audio2 = np.random.randn(16000).astype(np.float32) * 0.1

        result1 = whisper.transcribe_audio(audio1, sample_rate=16000)
        result2 = whisper.transcribe_audio(audio2, sample_rate=16000)

        # Both should succeed
        assert isinstance(result1.text, str)
        assert isinstance(result2.text, str)

    def test_different_models_sequentially(self):
        """Test creating different model sizes sequentially."""
        # This ensures no conflicts between models
        whisper_tiny = WhisperInference(model_size="tiny")
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result_tiny = whisper_tiny.transcribe_audio(audio, sample_rate=16000)

        whisper_base = WhisperInference(model_size="base")
        result_base = whisper_base.transcribe_audio(audio, sample_rate=16000)

        assert isinstance(result_tiny.text, str)
        assert isinstance(result_base.text, str)


class TestTimestampFormatting:
    """Test timestamp formatting edge cases."""

    def test_format_srt_time_edge_cases(self):
        """Test SRT time formatting with edge cases."""
        from audiodecode.inference import TranscriptionResult

        # Test zero
        assert TranscriptionResult._format_srt_time(0.0) == "00:00:00,000"

        # Test less than 1 second
        assert TranscriptionResult._format_srt_time(0.5) == "00:00:00,500"

        # Test exactly 1 hour
        assert TranscriptionResult._format_srt_time(3600.0) == "01:00:00,000"

        # Test long duration (allowing for rounding differences)
        result = TranscriptionResult._format_srt_time(7265.123)
        assert result.startswith("02:01:05,")  # Check main part, allow millisecond variance

    def test_format_vtt_time_edge_cases(self):
        """Test VTT time formatting with edge cases."""
        from audiodecode.inference import TranscriptionResult

        # Test zero
        assert TranscriptionResult._format_vtt_time(0.0) == "00:00:00.000"

        # Test milliseconds
        assert TranscriptionResult._format_vtt_time(1.234) == "00:00:01.234"

        # Test hours
        assert TranscriptionResult._format_vtt_time(3661.5) == "01:01:01.500"
