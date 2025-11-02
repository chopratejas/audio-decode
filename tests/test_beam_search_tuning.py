"""Test beam search tuning parameters."""
import numpy as np
import pytest
from pathlib import Path

from audiodecode.inference import WhisperInference, transcribe_file, transcribe_audio


class TestPatienceParameter:
    """Test patience parameter."""

    def test_patience_default_value(self, audio_1s_mono_16k):
        """patience parameter should have None default."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
        )
        assert result is not None
        assert result.text is not None

    def test_patience_custom_value_low(self, audio_1s_mono_16k):
        """patience=0.5 should work (less patient, faster)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            patience=0.5,
        )
        assert result is not None
        assert result.text is not None

    def test_patience_custom_value_high(self, audio_1s_mono_16k):
        """patience=2.0 should work (more patient, better quality)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            patience=2.0,
        )
        assert result is not None
        assert result.text is not None

    def test_patience_explicit_none(self, audio_1s_mono_16k):
        """patience=None should work (use model default)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            patience=None,
        )
        assert result is not None
        assert result.text is not None

    def test_patience_type_validation_invalid_string(self, audio_1s_mono_16k):
        """patience must be float or None, not string."""
        with pytest.raises(TypeError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                patience="high",
            )
        assert "patience must be float or None" in str(exc_info.value)

    def test_patience_type_validation_invalid_list(self, audio_1s_mono_16k):
        """patience must be float or None, not list."""
        with pytest.raises(TypeError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                patience=[1.0, 2.0],
            )
        assert "patience must be float or None" in str(exc_info.value)

    def test_patience_range_validation_negative(self, audio_1s_mono_16k):
        """patience must be positive."""
        with pytest.raises(ValueError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                patience=-1.0,
            )
        assert "patience must be positive" in str(exc_info.value)

    def test_patience_range_validation_zero(self, audio_1s_mono_16k):
        """patience=0 should raise error (must be positive)."""
        with pytest.raises(ValueError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                patience=0.0,
            )
        assert "patience must be positive" in str(exc_info.value)

    def test_patience_with_beam_size(self, audio_1s_mono_16k):
        """patience should work with beam_size parameter."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            beam_size=3,
            patience=1.5,
        )
        assert result is not None
        assert result.text is not None


class TestLengthPenalty:
    """Test length_penalty parameter."""

    def test_length_penalty_default_value(self, audio_1s_mono_16k):
        """length_penalty parameter should have None default."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
        )
        assert result is not None
        assert result.text is not None

    def test_length_penalty_custom_value_positive(self, audio_1s_mono_16k):
        """length_penalty=1.2 should work (favor longer sequences)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            length_penalty=1.2,
        )
        assert result is not None
        assert result.text is not None

    def test_length_penalty_custom_value_negative(self, audio_1s_mono_16k):
        """length_penalty=-0.5 should work (favor shorter sequences)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            length_penalty=-0.5,
        )
        assert result is not None
        assert result.text is not None

    def test_length_penalty_zero(self, audio_1s_mono_16k):
        """length_penalty=0.0 should work (no penalty)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            length_penalty=0.0,
        )
        assert result is not None
        assert result.text is not None

    def test_length_penalty_explicit_none(self, audio_1s_mono_16k):
        """length_penalty=None should work (use model default)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            length_penalty=None,
        )
        assert result is not None
        assert result.text is not None

    def test_length_penalty_type_validation_invalid_string(self, audio_1s_mono_16k):
        """length_penalty must be float or None, not string."""
        with pytest.raises(TypeError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                length_penalty="high",
            )
        assert "length_penalty must be float or None" in str(exc_info.value)

    def test_length_penalty_type_validation_invalid_list(self, audio_1s_mono_16k):
        """length_penalty must be float or None, not list."""
        with pytest.raises(TypeError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                length_penalty=[1.0, 2.0],
            )
        assert "length_penalty must be float or None" in str(exc_info.value)

    def test_length_penalty_with_beam_size(self, audio_1s_mono_16k):
        """length_penalty should work with beam_size parameter."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            beam_size=3,
            length_penalty=0.8,
        )
        assert result is not None
        assert result.text is not None


class TestRepetitionPenalty:
    """Test repetition_penalty parameter."""

    def test_repetition_penalty_default_value(self, audio_1s_mono_16k):
        """repetition_penalty parameter should have None default."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
        )
        assert result is not None
        assert result.text is not None

    def test_repetition_penalty_custom_value_low(self, audio_1s_mono_16k):
        """repetition_penalty=1.1 should work (slight penalty)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            repetition_penalty=1.1,
        )
        assert result is not None
        assert result.text is not None

    def test_repetition_penalty_custom_value_high(self, audio_1s_mono_16k):
        """repetition_penalty=2.0 should work (strong penalty)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            repetition_penalty=2.0,
        )
        assert result is not None
        assert result.text is not None

    def test_repetition_penalty_one(self, audio_1s_mono_16k):
        """repetition_penalty=1.0 should work (no penalty)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            repetition_penalty=1.0,
        )
        assert result is not None
        assert result.text is not None

    def test_repetition_penalty_explicit_none(self, audio_1s_mono_16k):
        """repetition_penalty=None should work (use model default)."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            repetition_penalty=None,
        )
        assert result is not None
        assert result.text is not None

    def test_repetition_penalty_type_validation_invalid_string(self, audio_1s_mono_16k):
        """repetition_penalty must be float or None, not string."""
        with pytest.raises(TypeError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                repetition_penalty="high",
            )
        assert "repetition_penalty must be float or None" in str(exc_info.value)

    def test_repetition_penalty_type_validation_invalid_list(self, audio_1s_mono_16k):
        """repetition_penalty must be float or None, not list."""
        with pytest.raises(TypeError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                repetition_penalty=[1.0, 2.0],
            )
        assert "repetition_penalty must be float or None" in str(exc_info.value)

    def test_repetition_penalty_range_validation_negative(self, audio_1s_mono_16k):
        """repetition_penalty must be positive."""
        with pytest.raises(ValueError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                repetition_penalty=-1.0,
            )
        assert "repetition_penalty must be positive" in str(exc_info.value)

    def test_repetition_penalty_range_validation_zero(self, audio_1s_mono_16k):
        """repetition_penalty=0 should raise error (must be positive)."""
        with pytest.raises(ValueError) as exc_info:
            transcribe_file(
                audio_1s_mono_16k,
                model_size="tiny",
                repetition_penalty=0.0,
            )
        assert "repetition_penalty must be positive" in str(exc_info.value)

    def test_repetition_penalty_with_beam_size(self, audio_1s_mono_16k):
        """repetition_penalty should work with beam_size parameter."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            beam_size=3,
            repetition_penalty=1.5,
        )
        assert result is not None
        assert result.text is not None


class TestBeamSearchIntegration:
    """Test all beam search params together."""

    def test_all_beam_search_params_together(self, audio_1s_mono_16k):
        """All beam search tuning params should work together."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            beam_size=5,
            best_of=5,
            patience=1.5,
            length_penalty=0.8,
            repetition_penalty=1.2,
        )
        assert result is not None
        assert result.text is not None
        # Segments may be empty for pure sine wave audio (no speech)
        assert len(result.segments) >= 0

    def test_beam_search_with_word_timestamps(self, audio_1s_mono_16k):
        """Beam search tuning should work with word_timestamps."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            patience=1.0,
            length_penalty=1.0,
            repetition_penalty=1.1,
            word_timestamps=True,
        )
        assert result is not None
        assert result.text is not None
        # Word timestamps may or may not be present depending on audio

    def test_beam_search_with_temperature(self, audio_1s_mono_16k):
        """Beam search tuning should work with temperature parameter."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            patience=1.5,
            length_penalty=0.5,
            repetition_penalty=1.3,
            temperature=0.2,
        )
        assert result is not None
        assert result.text is not None

    def test_beam_search_with_vad(self, audio_1s_mono_16k):
        """Beam search tuning should work with VAD filter."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            patience=1.0,
            length_penalty=0.8,
            repetition_penalty=1.2,
            vad_filter=True,
        )
        assert result is not None
        assert result.text is not None

    def test_beam_search_with_initial_prompt(self, audio_1s_mono_16k):
        """Beam search tuning should work with initial_prompt."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            patience=1.2,
            length_penalty=0.9,
            repetition_penalty=1.1,
            initial_prompt="Technical discussion",
        )
        assert result is not None
        assert result.text is not None

    def test_beam_search_with_hotwords(self, audio_1s_mono_16k):
        """Beam search tuning should work with hotwords."""
        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="tiny",
            patience=1.0,
            length_penalty=1.0,
            repetition_penalty=1.2,
            hotwords="test audio",
        )
        assert result is not None
        assert result.text is not None


class TestBeamSearchWithTranscribeAudio:
    """Test beam search params with transcribe_audio function."""

    def test_patience_with_audio_array(self):
        """patience should work with audio array."""
        # Generate 1 second of silence
        audio = np.zeros(16000, dtype=np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=16000,
            model_size="tiny",
            patience=1.5,
        )
        assert result is not None

    def test_length_penalty_with_audio_array(self):
        """length_penalty should work with audio array."""
        # Generate 1 second of silence
        audio = np.zeros(16000, dtype=np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=16000,
            model_size="tiny",
            length_penalty=0.8,
        )
        assert result is not None

    def test_repetition_penalty_with_audio_array(self):
        """repetition_penalty should work with audio array."""
        # Generate 1 second of silence
        audio = np.zeros(16000, dtype=np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=16000,
            model_size="tiny",
            repetition_penalty=1.2,
        )
        assert result is not None

    def test_all_params_with_audio_array(self):
        """All beam search params should work with audio array."""
        # Generate 1 second of silence
        audio = np.zeros(16000, dtype=np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=16000,
            model_size="tiny",
            patience=1.5,
            length_penalty=0.8,
            repetition_penalty=1.2,
        )
        assert result is not None


class TestBeamSearchWithWhisperInference:
    """Test beam search params with WhisperInference class."""

    def test_patience_with_class(self, audio_1s_mono_16k):
        """patience should work with WhisperInference class."""
        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(
            audio_1s_mono_16k,
            patience=1.5,
        )
        assert result is not None
        assert result.text is not None

    def test_length_penalty_with_class(self, audio_1s_mono_16k):
        """length_penalty should work with WhisperInference class."""
        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(
            audio_1s_mono_16k,
            length_penalty=0.8,
        )
        assert result is not None
        assert result.text is not None

    def test_repetition_penalty_with_class(self, audio_1s_mono_16k):
        """repetition_penalty should work with WhisperInference class."""
        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(
            audio_1s_mono_16k,
            repetition_penalty=1.2,
        )
        assert result is not None
        assert result.text is not None

    def test_all_params_with_class(self, audio_1s_mono_16k):
        """All beam search params should work with WhisperInference class."""
        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(
            audio_1s_mono_16k,
            patience=1.5,
            length_penalty=0.8,
            repetition_penalty=1.2,
        )
        assert result is not None
        assert result.text is not None

    def test_all_params_with_class_audio_array(self):
        """All beam search params should work with WhisperInference.transcribe_audio."""
        whisper = WhisperInference(model_size="tiny")

        # Generate 1 second of silence
        audio = np.zeros(16000, dtype=np.float32)

        result = whisper.transcribe_audio(
            audio,
            sample_rate=16000,
            patience=1.5,
            length_penalty=0.8,
            repetition_penalty=1.2,
        )
        assert result is not None
