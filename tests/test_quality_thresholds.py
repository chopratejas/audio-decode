"""
Test quality threshold features for AudioDecode.

Tests cover:
- compression_ratio_threshold parameter
- logprob_threshold parameter
- no_speech_threshold parameter
- Type validation for all thresholds
- Range validation where applicable
- Integration of all thresholds
- Pass-through to faster-whisper
- Edge cases and error handling
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import audiodecode.inference as inference_module


@pytest.fixture
def mock_whisper_model():
    """Create a mock WhisperModel for testing without downloading models."""
    mock_model = MagicMock()

    # Mock transcription info
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 1.0

    # Mock transcription segment
    mock_segment = MagicMock()
    mock_segment.text = " Hello world"
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.avg_logprob = -0.5

    # Mock transcribe method to return iterator and info
    mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

    return mock_model


@pytest.fixture
def mock_whisper_model_class(mock_whisper_model):
    """Mock the WhisperModel class constructor."""
    with patch('audiodecode.inference.WhisperModel', return_value=mock_whisper_model) as mock_class:
        yield mock_class


@pytest.fixture
def sample_audio_16k() -> tuple[np.ndarray, int]:
    """Generate 1 second of sample audio at 16kHz."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestCompressionRatioThreshold:
    """Test compression_ratio_threshold parameter."""

    def test_compression_ratio_threshold_default_value(self, mock_whisper_model_class, sample_audio_16k):
        """Default value should be None."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr)

        # Verify default value was passed to model.transcribe
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert "compression_ratio_threshold" in call_kwargs
        assert call_kwargs["compression_ratio_threshold"] is None

    def test_compression_ratio_threshold_with_custom_value(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept custom float value."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, compression_ratio_threshold=2.4)

        # Verify custom value was passed to model.transcribe
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2.4

    def test_compression_ratio_threshold_with_typical_value(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept typical value 2.4."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, compression_ratio_threshold=2.4)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2.4

    def test_compression_ratio_threshold_type_validation_invalid_string(self, mock_whisper_model_class, sample_audio_16k):
        """Must reject string type."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        with pytest.raises(TypeError, match="compression_ratio_threshold must be float or None"):
            whisper.transcribe_audio(audio, sample_rate=sr, compression_ratio_threshold="2.4")

    def test_compression_ratio_threshold_type_validation_invalid_list(self, mock_whisper_model_class, sample_audio_16k):
        """Must reject list type."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        with pytest.raises(TypeError, match="compression_ratio_threshold must be float or None"):
            whisper.transcribe_audio(audio, sample_rate=sr, compression_ratio_threshold=[2.4])

    def test_compression_ratio_threshold_accepts_none(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept None explicitly."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, compression_ratio_threshold=None)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] is None

    def test_compression_ratio_threshold_accepts_int(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept int and convert to float."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, compression_ratio_threshold=2)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2

    def test_compression_ratio_threshold_passed_in_transcribe_file(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Should be passed through in transcribe_file method."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(audio_1s_mono_16k, compression_ratio_threshold=2.4)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2.4

    def test_compression_ratio_threshold_in_convenience_function(self, mock_whisper_model_class, sample_audio_16k):
        """Should work with module-level transcribe_audio()."""
        from audiodecode.inference import transcribe_audio

        audio, sr = sample_audio_16k
        result = transcribe_audio(audio, sample_rate=sr, model_size="tiny", compression_ratio_threshold=2.4)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2.4

    def test_compression_ratio_threshold_in_convenience_file_function(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Should work with module-level transcribe_file()."""
        from audiodecode.inference import transcribe_file

        result = transcribe_file(audio_1s_mono_16k, model_size="tiny", compression_ratio_threshold=2.4)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2.4


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestLogprobThreshold:
    """Test logprob_threshold parameter."""

    def test_logprob_threshold_default_value(self, mock_whisper_model_class, sample_audio_16k):
        """Default value should be None."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr)

        # Verify default value was passed to model.transcribe
        # Note: faster-whisper uses log_prob_threshold (with underscore)
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert "log_prob_threshold" in call_kwargs
        assert call_kwargs["log_prob_threshold"] is None

    def test_logprob_threshold_with_custom_value(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept custom float value."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, logprob_threshold=-1.0)

        # Verify custom value was passed to model.transcribe
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["log_prob_threshold"] == -1.0

    def test_logprob_threshold_with_typical_value(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept typical value -1.0."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, logprob_threshold=-1.0)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["log_prob_threshold"] == -1.0

    def test_logprob_threshold_type_validation_invalid_string(self, mock_whisper_model_class, sample_audio_16k):
        """Must reject string type."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        with pytest.raises(TypeError, match="logprob_threshold must be float or None"):
            whisper.transcribe_audio(audio, sample_rate=sr, logprob_threshold="-1.0")

    def test_logprob_threshold_type_validation_invalid_list(self, mock_whisper_model_class, sample_audio_16k):
        """Must reject list type."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        with pytest.raises(TypeError, match="logprob_threshold must be float or None"):
            whisper.transcribe_audio(audio, sample_rate=sr, logprob_threshold=[-1.0])

    def test_logprob_threshold_accepts_none(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept None explicitly."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, logprob_threshold=None)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["log_prob_threshold"] is None

    def test_logprob_threshold_accepts_negative_values(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept negative float values (common for log probabilities)."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, logprob_threshold=-2.5)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["log_prob_threshold"] == -2.5

    def test_logprob_threshold_accepts_int(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept int and convert to float."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, logprob_threshold=-1)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["log_prob_threshold"] == -1

    def test_logprob_threshold_passed_in_transcribe_file(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Should be passed through in transcribe_file method."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(audio_1s_mono_16k, logprob_threshold=-1.0)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["log_prob_threshold"] == -1.0

    def test_logprob_threshold_in_convenience_function(self, mock_whisper_model_class, sample_audio_16k):
        """Should work with module-level transcribe_audio()."""
        from audiodecode.inference import transcribe_audio

        audio, sr = sample_audio_16k
        result = transcribe_audio(audio, sample_rate=sr, model_size="tiny", logprob_threshold=-1.0)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["log_prob_threshold"] == -1.0

    def test_logprob_threshold_in_convenience_file_function(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Should work with module-level transcribe_file()."""
        from audiodecode.inference import transcribe_file

        result = transcribe_file(audio_1s_mono_16k, model_size="tiny", logprob_threshold=-1.0)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["log_prob_threshold"] == -1.0


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestNoSpeechThreshold:
    """Test no_speech_threshold parameter."""

    def test_no_speech_threshold_default_value(self, mock_whisper_model_class, sample_audio_16k):
        """Default value should be None."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr)

        # Verify default value was passed to model.transcribe
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert "no_speech_threshold" in call_kwargs
        assert call_kwargs["no_speech_threshold"] is None

    def test_no_speech_threshold_with_custom_value(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept custom float value."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold=0.6)

        # Verify custom value was passed to model.transcribe
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] == 0.6

    def test_no_speech_threshold_with_typical_value(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept typical value 0.6."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold=0.6)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] == 0.6

    def test_no_speech_threshold_type_validation_invalid_string(self, mock_whisper_model_class, sample_audio_16k):
        """Must reject string type."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        with pytest.raises(TypeError, match="no_speech_threshold must be float or None"):
            whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold="0.6")

    def test_no_speech_threshold_type_validation_invalid_list(self, mock_whisper_model_class, sample_audio_16k):
        """Must reject list type."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        with pytest.raises(TypeError, match="no_speech_threshold must be float or None"):
            whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold=[0.6])

    def test_no_speech_threshold_accepts_none(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept None explicitly."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold=None)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] is None

    def test_no_speech_threshold_range_validation_negative(self, mock_whisper_model_class, sample_audio_16k):
        """Must reject negative values."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        with pytest.raises(ValueError, match="no_speech_threshold must be between 0 and 1"):
            whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold=-0.1)

    def test_no_speech_threshold_range_validation_too_large(self, mock_whisper_model_class, sample_audio_16k):
        """Must reject values greater than 1."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        with pytest.raises(ValueError, match="no_speech_threshold must be between 0 and 1"):
            whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold=1.5)

    def test_no_speech_threshold_accepts_boundary_values(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept 0.0 and 1.0 as boundary values."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        # Test 0.0
        result = whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold=0.0)
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] == 0.0

        # Test 1.0
        result = whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold=1.0)
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] == 1.0

    def test_no_speech_threshold_accepts_int(self, mock_whisper_model_class, sample_audio_16k):
        """Should accept int and convert to float."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, no_speech_threshold=1)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] == 1

    def test_no_speech_threshold_passed_in_transcribe_file(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Should be passed through in transcribe_file method."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(audio_1s_mono_16k, no_speech_threshold=0.6)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] == 0.6

    def test_no_speech_threshold_in_convenience_function(self, mock_whisper_model_class, sample_audio_16k):
        """Should work with module-level transcribe_audio()."""
        from audiodecode.inference import transcribe_audio

        audio, sr = sample_audio_16k
        result = transcribe_audio(audio, sample_rate=sr, model_size="tiny", no_speech_threshold=0.6)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] == 0.6

    def test_no_speech_threshold_in_convenience_file_function(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Should work with module-level transcribe_file()."""
        from audiodecode.inference import transcribe_file

        result = transcribe_file(audio_1s_mono_16k, model_size="tiny", no_speech_threshold=0.6)

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] == 0.6


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestQualityThresholdsIntegration:
    """Test all quality thresholds working together."""

    def test_all_thresholds_together(self, mock_whisper_model_class, sample_audio_16k):
        """Test using all three thresholds simultaneously."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(
            audio,
            sample_rate=sr,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        # Verify all parameters were passed correctly
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2.4
        assert call_kwargs["log_prob_threshold"] == -1.0
        assert call_kwargs["no_speech_threshold"] == 0.6

    def test_all_thresholds_with_other_parameters(self, mock_whisper_model_class, sample_audio_16k):
        """Test quality thresholds work alongside other parameters."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(
            audio,
            sample_rate=sr,
            language="en",
            beam_size=5,
            temperature=0.0,
            vad_filter=True,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        # Verify all parameters coexist correctly
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"
        assert call_kwargs["beam_size"] == 5
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["vad_filter"] is True
        assert call_kwargs["compression_ratio_threshold"] == 2.4
        assert call_kwargs["log_prob_threshold"] == -1.0
        assert call_kwargs["no_speech_threshold"] == 0.6

    def test_partial_thresholds(self, mock_whisper_model_class, sample_audio_16k):
        """Test using only some thresholds."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        # Only compression_ratio_threshold
        result = whisper.transcribe_audio(audio, sample_rate=sr, compression_ratio_threshold=2.4)
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2.4
        assert call_kwargs["log_prob_threshold"] is None
        assert call_kwargs["no_speech_threshold"] is None

    def test_all_thresholds_in_transcribe_file(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test all thresholds in file-based transcription."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(
            audio_1s_mono_16k,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2.4
        assert call_kwargs["log_prob_threshold"] == -1.0
        assert call_kwargs["no_speech_threshold"] == 0.6

    def test_all_thresholds_in_convenience_functions(self, mock_whisper_model_class, sample_audio_16k):
        """Test all thresholds work in convenience functions."""
        from audiodecode.inference import transcribe_audio

        audio, sr = sample_audio_16k
        result = transcribe_audio(
            audio,
            sample_rate=sr,
            model_size="tiny",
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] == 2.4
        assert call_kwargs["log_prob_threshold"] == -1.0
        assert call_kwargs["no_speech_threshold"] == 0.6

    def test_thresholds_with_word_timestamps(self, mock_whisper_model_class, sample_audio_16k):
        """Test thresholds work with word_timestamps feature."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(
            audio,
            sample_rate=sr,
            word_timestamps=True,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["word_timestamps"] is True
        assert call_kwargs["compression_ratio_threshold"] == 2.4
        assert call_kwargs["log_prob_threshold"] == -1.0
        assert call_kwargs["no_speech_threshold"] == 0.6

    def test_mixed_threshold_types(self, mock_whisper_model_class, sample_audio_16k):
        """Test mixing None and numeric values for thresholds."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(
            audio,
            sample_rate=sr,
            compression_ratio_threshold=None,
            logprob_threshold=-1.0,
            no_speech_threshold=None
        )

        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["compression_ratio_threshold"] is None
        assert call_kwargs["log_prob_threshold"] == -1.0
        assert call_kwargs["no_speech_threshold"] is None

    def test_error_handling_with_multiple_invalid_thresholds(self, mock_whisper_model_class, sample_audio_16k):
        """Test that first validation error is caught even with multiple invalid values."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        # Should catch the first type error
        with pytest.raises(TypeError):
            whisper.transcribe_audio(
                audio,
                sample_rate=sr,
                compression_ratio_threshold="invalid",
                logprob_threshold="also_invalid",
                no_speech_threshold="still_invalid"
            )
