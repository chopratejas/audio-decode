"""
Test prompt engineering features for AudioDecode.

Tests three critical parameters from openai-whisper:
1. initial_prompt - Guide model with context/terminology
2. condition_on_previous_text - Use previous segments as context
3. prefix - Force first segment to start with specific text

These features are essential for:
- Domain-specific terminology (medical, legal, technical)
- Speaker names and consistent formatting
- Maintaining context across long transcriptions
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


class TestInitialPrompt:
    """Test initial_prompt parameter for guiding transcription."""

    def test_initial_prompt_parameter_accepted(self, tmp_path):
        """initial_prompt parameter should be accepted without error."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Should not raise an error
        result = transcribe_file(
            str(audio_path),
            initial_prompt="This is a technical podcast about machine learning.",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_initial_prompt_with_terminology(self, tmp_path):
        """initial_prompt should help with domain-specific terminology."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        # With prompt guiding terminology
        result = transcribe_file(
            str(audio_path),
            initial_prompt="Medical terms: electrocardiogram, myocardial infarction, CT scan",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)
        # Result should be valid (specific terminology matching would require real audio)

    def test_initial_prompt_none_default(self, tmp_path):
        """initial_prompt=None should work (default behavior)."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            initial_prompt=None,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_initial_prompt_type_validation(self, tmp_path):
        """initial_prompt should accept string or None only."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        # Valid: string
        result1 = transcribe_file(str(audio_path), initial_prompt="Context here", model_size="tiny")
        assert isinstance(result1, TranscriptionResult)

        # Valid: None
        result2 = transcribe_file(str(audio_path), initial_prompt=None, model_size="tiny")
        assert isinstance(result2, TranscriptionResult)

        # Invalid: number should raise TypeError
        with pytest.raises(TypeError):
            transcribe_file(str(audio_path), initial_prompt=123, model_size="tiny")

        # Invalid: list should raise TypeError
        with pytest.raises(TypeError):
            transcribe_file(str(audio_path), initial_prompt=["test"], model_size="tiny")

    def test_initial_prompt_with_transcribe_audio(self):
        """initial_prompt should work with transcribe_audio (numpy input)."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=sample_rate,
            initial_prompt="Speaker names: Alice, Bob",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_initial_prompt_empty_string(self, tmp_path):
        """Empty string for initial_prompt should be accepted."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        result = transcribe_file(str(audio_path), initial_prompt="", model_size="tiny")
        assert isinstance(result, TranscriptionResult)

    def test_initial_prompt_with_special_characters(self, tmp_path):
        """initial_prompt with special characters should work."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            initial_prompt="Names: François, José, 北京",
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


class TestConditionOnPreviousText:
    """Test condition_on_previous_text parameter for context maintenance."""

    def test_condition_on_previous_text_parameter_accepted(self, tmp_path):
        """condition_on_previous_text parameter should be accepted."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Should not raise an error
        result = transcribe_file(
            str(audio_path),
            condition_on_previous_text=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_condition_on_previous_text_false(self, tmp_path):
        """condition_on_previous_text=False should disable context."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            condition_on_previous_text=False,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_condition_on_previous_text_default_true(self, tmp_path):
        """condition_on_previous_text should default to True (openai-whisper behavior)."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Don't specify - should use default
        result = transcribe_file(str(audio_path), model_size="tiny")

        assert isinstance(result, TranscriptionResult)

    def test_condition_on_previous_text_type_validation(self, tmp_path):
        """condition_on_previous_text should accept bool only."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        # Valid: bool
        result1 = transcribe_file(str(audio_path), condition_on_previous_text=True, model_size="tiny")
        assert isinstance(result1, TranscriptionResult)

        result2 = transcribe_file(str(audio_path), condition_on_previous_text=False, model_size="tiny")
        assert isinstance(result2, TranscriptionResult)

        # Invalid: string should raise TypeError
        with pytest.raises(TypeError):
            transcribe_file(str(audio_path), condition_on_previous_text="true", model_size="tiny")

    def test_condition_on_previous_text_with_transcribe_audio(self):
        """condition_on_previous_text should work with numpy input."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=sample_rate,
            condition_on_previous_text=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_condition_on_previous_text_with_initial_prompt(self, tmp_path):
        """Both condition_on_previous_text and initial_prompt should work together."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            initial_prompt="Technical discussion",
            condition_on_previous_text=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    # Helper method
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestPrefix:
    """Test prefix parameter for forcing first segment text."""

    def test_prefix_parameter_accepted(self, tmp_path):
        """prefix parameter should be accepted without error."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Should not raise an error
        result = transcribe_file(
            str(audio_path),
            prefix="Speaker 1:",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_prefix_none_default(self, tmp_path):
        """prefix=None should work (default behavior)."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            prefix=None,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_prefix_type_validation(self, tmp_path):
        """prefix should accept string or None only."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        # Valid: string
        result1 = transcribe_file(str(audio_path), prefix="[Music]", model_size="tiny")
        assert isinstance(result1, TranscriptionResult)

        # Valid: None
        result2 = transcribe_file(str(audio_path), prefix=None, model_size="tiny")
        assert isinstance(result2, TranscriptionResult)

        # Invalid: number should raise TypeError
        with pytest.raises(TypeError):
            transcribe_file(str(audio_path), prefix=123, model_size="tiny")

    def test_prefix_with_transcribe_audio(self):
        """prefix should work with transcribe_audio."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=sample_rate,
            prefix="Alice:",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_prefix_empty_string(self, tmp_path):
        """Empty string prefix should be accepted."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        result = transcribe_file(str(audio_path), prefix="", model_size="tiny")
        assert isinstance(result, TranscriptionResult)

    def test_prefix_with_special_characters(self, tmp_path):
        """prefix with special characters should work."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            prefix="♪ [Music] ♪",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    # Helper method
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestPromptEngineeringIntegration:
    """Test all prompt engineering features working together."""

    def test_all_prompt_features_together(self, tmp_path):
        """All three prompt engineering features should work together."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=5.0)

        result = transcribe_file(
            str(audio_path),
            initial_prompt="Technical podcast with speakers Alice and Bob",
            condition_on_previous_text=True,
            prefix="Alice:",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_prompt_features_with_word_timestamps(self, tmp_path):
        """Prompt features should work with word_timestamps."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            initial_prompt="Medical discussion",
            word_timestamps=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)
        # If segments exist, they should have words attribute
        for segment in result.segments:
            assert hasattr(segment, 'words')

    def test_prompt_features_with_other_parameters(self, tmp_path):
        """Prompt features should work with beam search parameters."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            initial_prompt="Technical talk",
            condition_on_previous_text=True,
            beam_size=10,
            best_of=3,
            temperature=0.2,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_prompt_features_with_whisper_inference_class(self, tmp_path):
        """Prompt features should work with WhisperInference OOP API."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(
            str(audio_path),
            initial_prompt="Context",
            condition_on_previous_text=True,
            prefix="Speaker:",
        )

        assert isinstance(result, TranscriptionResult)

    # Helper method
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestPromptEngineeringDocumentation:
    """Test that prompt engineering features are well-documented."""

    def test_parameters_in_function_signature(self):
        """Ensure parameters are in function signatures."""
        import inspect

        # Check transcribe_file
        sig = inspect.signature(transcribe_file)
        assert 'initial_prompt' in sig.parameters
        assert 'condition_on_previous_text' in sig.parameters
        assert 'prefix' in sig.parameters

        # Check transcribe_audio
        sig = inspect.signature(transcribe_audio)
        assert 'initial_prompt' in sig.parameters
        assert 'condition_on_previous_text' in sig.parameters
        assert 'prefix' in sig.parameters

    def test_whisper_inference_has_parameters(self):
        """WhisperInference methods should have prompt parameters."""
        import inspect

        whisper = WhisperInference(model_size="tiny")

        # Check transcribe_file method
        sig = inspect.signature(whisper.transcribe_file)
        assert 'initial_prompt' in sig.parameters
        assert 'condition_on_previous_text' in sig.parameters
        assert 'prefix' in sig.parameters

        # Check transcribe_audio method
        sig = inspect.signature(whisper.transcribe_audio)
        assert 'initial_prompt' in sig.parameters
        assert 'condition_on_previous_text' in sig.parameters
        assert 'prefix' in sig.parameters
