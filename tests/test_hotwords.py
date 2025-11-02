"""
Test hotwords and advanced prompt features for AudioDecode.

Tests two critical parameters from faster-whisper:
1. hotwords - Boost recognition of specific words/phrases
2. prompt_reset_on_temperature - Reset prompt when temperature changes

These features are essential for:
- Domain-specific vocabulary and terminology
- Product names, acronyms, and technical terms
- Speaker names and unique identifiers
- Controlling prompt behavior with temperature fallbacks
"""

import pytest
import numpy as np
from pathlib import Path

from audiodecode.inference import (
    transcribe_file,
    transcribe_audio,
    WhisperInference,
    TranscriptionResult,
)


class TestHotwords:
    """Test hotwords parameter for boosting specific word recognition."""

    def test_hotwords_string_accepted(self, tmp_path):
        """hotwords as string should be accepted without error."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Should not raise an error
        result = transcribe_file(
            str(audio_path),
            hotwords="OpenAI, Whisper, AudioDecode",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_hotwords_with_technical_terms(self, tmp_path):
        """hotwords should help with technical terminology."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        # With hotwords for technical terms
        result = transcribe_file(
            str(audio_path),
            hotwords="PyTorch, TensorFlow, CUDA, GPU",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_hotwords_none_default(self, tmp_path):
        """hotwords=None should be default behavior."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Explicit None
        result1 = transcribe_file(
            str(audio_path),
            hotwords=None,
            model_size="tiny"
        )
        assert isinstance(result1, TranscriptionResult)

        # Default (omitted)
        result2 = transcribe_file(str(audio_path), model_size="tiny")
        assert isinstance(result2, TranscriptionResult)

    def test_hotwords_type_validation(self, tmp_path):
        """hotwords must be str or None."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        # Valid: string
        result1 = transcribe_file(str(audio_path), hotwords="test", model_size="tiny")
        assert isinstance(result1, TranscriptionResult)

        # Valid: None
        result2 = transcribe_file(str(audio_path), hotwords=None, model_size="tiny")
        assert isinstance(result2, TranscriptionResult)

        # Invalid: number should raise TypeError
        with pytest.raises(TypeError, match="hotwords must be str or None"):
            transcribe_file(str(audio_path), hotwords=123, model_size="tiny")

        # Invalid: list should raise TypeError
        with pytest.raises(TypeError, match="hotwords must be str or None"):
            transcribe_file(str(audio_path), hotwords=["test", "words"], model_size="tiny")

    def test_hotwords_with_transcribe_audio(self):
        """hotwords should work with transcribe_audio (numpy input)."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=sample_rate,
            hotwords="Speaker names: Alice, Bob, Charlie",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_hotwords_empty_string(self, tmp_path):
        """Empty string for hotwords should be accepted."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        result = transcribe_file(str(audio_path), hotwords="", model_size="tiny")
        assert isinstance(result, TranscriptionResult)

    def test_hotwords_with_special_characters(self, tmp_path):
        """hotwords with special characters and unicode should work."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            hotwords="François, José, 北京, München, API v2.0",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_hotwords_long_phrase(self, tmp_path):
        """hotwords with long phrases should work."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            hotwords="Natural Language Processing, Machine Learning, Artificial Intelligence",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    # Helper method
    def _create_test_audio(self, path: Path, duration: float):
        """Create a simple WAV file for testing."""
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestPromptResetOnTemperature:
    """Test prompt_reset_on_temperature parameter for controlling prompt behavior."""

    def test_prompt_reset_parameter_accepted(self, tmp_path):
        """prompt_reset_on_temperature parameter should be accepted."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Should not raise an error
        result = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=0.5,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_prompt_reset_default_value(self, tmp_path):
        """prompt_reset_on_temperature should have default value 0.5."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Default (omitted) - should use 0.5
        result = transcribe_file(str(audio_path), model_size="tiny")
        assert isinstance(result, TranscriptionResult)

    def test_prompt_reset_custom_values(self, tmp_path):
        """prompt_reset_on_temperature should accept custom thresholds."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Low threshold
        result1 = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=0.2,
            model_size="tiny"
        )
        assert isinstance(result1, TranscriptionResult)

        # High threshold
        result2 = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=0.9,
            model_size="tiny"
        )
        assert isinstance(result2, TranscriptionResult)

    def test_prompt_reset_type_validation(self, tmp_path):
        """prompt_reset_on_temperature must be float."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        # Valid: float
        result1 = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=0.5,
            model_size="tiny"
        )
        assert isinstance(result1, TranscriptionResult)

        # Valid: int (should be auto-converted to float)
        result2 = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=1,
            model_size="tiny"
        )
        assert isinstance(result2, TranscriptionResult)

        # Invalid: string should raise TypeError
        with pytest.raises(TypeError, match="prompt_reset_on_temperature must be float"):
            transcribe_file(str(audio_path), prompt_reset_on_temperature="0.5", model_size="tiny")

        # Invalid: None should raise TypeError (has default value)
        with pytest.raises(TypeError, match="prompt_reset_on_temperature must be float"):
            transcribe_file(str(audio_path), prompt_reset_on_temperature=None, model_size="tiny")

    def test_prompt_reset_range_validation(self, tmp_path):
        """prompt_reset_on_temperature must be between 0 and 1."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        # Valid: 0.0
        result1 = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=0.0,
            model_size="tiny"
        )
        assert isinstance(result1, TranscriptionResult)

        # Valid: 1.0
        result2 = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=1.0,
            model_size="tiny"
        )
        assert isinstance(result2, TranscriptionResult)

        # Invalid: negative value
        with pytest.raises(ValueError, match="prompt_reset_on_temperature must be between 0 and 1"):
            transcribe_file(str(audio_path), prompt_reset_on_temperature=-0.1, model_size="tiny")

        # Invalid: greater than 1
        with pytest.raises(ValueError, match="prompt_reset_on_temperature must be between 0 and 1"):
            transcribe_file(str(audio_path), prompt_reset_on_temperature=1.5, model_size="tiny")

    def test_prompt_reset_with_transcribe_audio(self):
        """prompt_reset_on_temperature should work with numpy input."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=sample_rate,
            prompt_reset_on_temperature=0.7,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_prompt_reset_with_temperature_fallback(self, tmp_path):
        """prompt_reset_on_temperature should work with temperature fallback."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        # With temperature fallback tuple
        result = transcribe_file(
            str(audio_path),
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            prompt_reset_on_temperature=0.5,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_prompt_reset_with_condition_on_previous_text(self, tmp_path):
        """prompt_reset_on_temperature only has effect with condition_on_previous_text=True."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        # With condition_on_previous_text=True (should use prompt_reset_on_temperature)
        result1 = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=0.5,
            condition_on_previous_text=True,
            model_size="tiny"
        )
        assert isinstance(result1, TranscriptionResult)

        # With condition_on_previous_text=False (prompt_reset_on_temperature has no effect)
        result2 = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=0.5,
            condition_on_previous_text=False,
            model_size="tiny"
        )
        assert isinstance(result2, TranscriptionResult)

    # Helper method
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestHotwordsIntegration:
    """Test hotwords with other features."""

    def test_hotwords_with_initial_prompt(self, tmp_path):
        """hotwords should work with initial_prompt."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            hotwords="PyTorch, TensorFlow",
            initial_prompt="Technical discussion about deep learning frameworks",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_hotwords_with_condition_on_previous_text(self, tmp_path):
        """hotwords should work with condition_on_previous_text."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            hotwords="AudioDecode, Whisper",
            condition_on_previous_text=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_hotwords_with_prefix_interaction(self, tmp_path):
        """hotwords should have no effect when prefix is specified (per faster-whisper docs)."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        # Both specified - prefix takes precedence, hotwords ignored
        result = transcribe_file(
            str(audio_path),
            hotwords="OpenAI",
            prefix="Speaker 1:",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_all_advanced_features_together(self, tmp_path):
        """Test hotwords and prompt_reset_on_temperature with other features."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=5.0)

        result = transcribe_file(
            str(audio_path),
            hotwords="AudioDecode, PyTorch, CUDA",
            initial_prompt="Technical podcast about audio processing",
            condition_on_previous_text=True,
            prompt_reset_on_temperature=0.6,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
            word_timestamps=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)
        # Verify word timestamps work with hotwords
        for segment in result.segments:
            assert hasattr(segment, 'words')

    def test_hotwords_with_beam_search(self, tmp_path):
        """hotwords should work with beam search parameters."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            hotwords="machine learning, neural networks",
            beam_size=10,
            best_of=5,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_prompt_reset_with_vad_filter(self, tmp_path):
        """prompt_reset_on_temperature should work with VAD filtering."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=4.0)

        result = transcribe_file(
            str(audio_path),
            prompt_reset_on_temperature=0.5,
            vad_filter=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_hotwords_with_whisper_inference_class(self, tmp_path):
        """hotwords and prompt_reset_on_temperature should work with WhisperInference OOP API."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(
            str(audio_path),
            hotwords="AudioDecode, faster-whisper",
            prompt_reset_on_temperature=0.5,
        )

        assert isinstance(result, TranscriptionResult)

    def test_hotwords_with_transcribe_audio_method(self):
        """hotwords should work with WhisperInference.transcribe_audio."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_audio(
            audio,
            sample_rate=sample_rate,
            hotwords="test, audio, transcription",
            prompt_reset_on_temperature=0.7,
        )

        assert isinstance(result, TranscriptionResult)

    # Helper method
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestHotwordsDocumentation:
    """Test that hotwords features are well-documented."""

    def test_parameters_in_function_signature(self):
        """Ensure parameters are in function signatures."""
        import inspect

        # Check transcribe_file
        sig = inspect.signature(transcribe_file)
        assert 'hotwords' in sig.parameters
        assert 'prompt_reset_on_temperature' in sig.parameters

        # Check transcribe_audio
        sig = inspect.signature(transcribe_audio)
        assert 'hotwords' in sig.parameters
        assert 'prompt_reset_on_temperature' in sig.parameters

    def test_whisper_inference_has_parameters(self):
        """WhisperInference methods should have hotwords parameters."""
        import inspect

        whisper = WhisperInference(model_size="tiny")

        # Check transcribe_file method
        sig = inspect.signature(whisper.transcribe_file)
        assert 'hotwords' in sig.parameters
        assert 'prompt_reset_on_temperature' in sig.parameters

        # Check transcribe_audio method
        sig = inspect.signature(whisper.transcribe_audio)
        assert 'hotwords' in sig.parameters
        assert 'prompt_reset_on_temperature' in sig.parameters

    def test_default_values_documented(self):
        """Ensure default values are properly set."""
        import inspect

        # Check transcribe_file defaults
        sig = inspect.signature(transcribe_file)
        assert sig.parameters['hotwords'].default is None
        assert sig.parameters['prompt_reset_on_temperature'].default == 0.5

        # Check transcribe_audio defaults
        sig = inspect.signature(transcribe_audio)
        assert sig.parameters['hotwords'].default is None
        assert sig.parameters['prompt_reset_on_temperature'].default == 0.5
