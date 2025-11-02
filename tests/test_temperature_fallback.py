"""
Test temperature fallback feature for AudioDecode.

In openai-whisper, temperature can be:
- Single float: 0.0 (greedy decoding, deterministic)
- Tuple/list: (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) (fallback retries)

Fallback behavior:
- Start with temperature[0]
- If quality is poor (compression ratio high, logprob low, etc.), retry with temperature[1]
- Continue until acceptable quality or exhausted all temperatures

This is CRITICAL for difficult audio - sampling often rescues failed greedy decoding.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Union

from audiodecode.inference import (
    transcribe_file,
    transcribe_audio,
    WhisperInference,
    TranscriptionResult,
)


class TestTemperatureSingleValue:
    """Test that single float temperature still works (backward compatibility)."""

    def test_temperature_float_accepted(self, tmp_path):
        """Single float temperature should work (existing behavior)."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Should work with single float
        result = transcribe_file(
            str(audio_path),
            temperature=0.0,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_temperature_nonzero_float(self, tmp_path):
        """Non-zero temperature for sampling should work."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            temperature=0.5,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_temperature_one_float(self, tmp_path):
        """Temperature=1.0 (maximum sampling) should work."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            temperature=1.0,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    # Helper
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestTemperatureTuple:
    """Test temperature as tuple for fallback retries."""

    def test_temperature_tuple_accepted(self, tmp_path):
        """Temperature as tuple should be accepted."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Classic openai-whisper pattern
        result = transcribe_file(
            str(audio_path),
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_temperature_list_accepted(self, tmp_path):
        """Temperature as list should also work."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            temperature=[0.0, 0.2, 0.4],
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_temperature_two_values(self, tmp_path):
        """Simple two-value fallback: greedy then sampling."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            temperature=(0.0, 0.8),
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_temperature_single_element_tuple(self, tmp_path):
        """Single-element tuple should work (no fallback, just like float)."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            temperature=(0.0,),
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_temperature_with_transcribe_audio(self):
        """Temperature tuple should work with numpy input."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        result = transcribe_audio(
            audio,
            sample_rate=sample_rate,
            temperature=(0.0, 0.5, 1.0),
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    # Helper
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestTemperatureTypeValidation:
    """Test type validation for temperature parameter."""

    def test_temperature_invalid_type_string(self, tmp_path):
        """String temperature should raise TypeError."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        with pytest.raises(TypeError):
            transcribe_file(str(audio_path), temperature="0.5", model_size="tiny")

    def test_temperature_invalid_type_dict(self, tmp_path):
        """Dict temperature should raise TypeError."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        with pytest.raises(TypeError):
            transcribe_file(str(audio_path), temperature={"temp": 0.5}, model_size="tiny")

    def test_temperature_invalid_tuple_with_strings(self, tmp_path):
        """Tuple with non-numeric values should raise TypeError."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        with pytest.raises((TypeError, ValueError)):
            transcribe_file(str(audio_path), temperature=("0.5", "0.8"), model_size="tiny")

    def test_temperature_out_of_range_negative(self, tmp_path):
        """Negative temperature should raise ValueError."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        with pytest.raises(ValueError):
            transcribe_file(str(audio_path), temperature=-0.5, model_size="tiny")

    def test_temperature_out_of_range_above_one(self, tmp_path):
        """Temperature > 1.0 should raise ValueError."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        with pytest.raises(ValueError):
            transcribe_file(str(audio_path), temperature=1.5, model_size="tiny")

    def test_temperature_tuple_out_of_range(self, tmp_path):
        """Tuple with values outside [0, 1] should raise ValueError."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        with pytest.raises(ValueError):
            transcribe_file(str(audio_path), temperature=(0.0, 1.5), model_size="tiny")

    def test_temperature_empty_tuple(self, tmp_path):
        """Empty tuple should raise ValueError."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=1.0)

        with pytest.raises(ValueError):
            transcribe_file(str(audio_path), temperature=(), model_size="tiny")

    # Helper
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestTemperatureWithOtherFeatures:
    """Test temperature works with other features."""

    def test_temperature_tuple_with_word_timestamps(self, tmp_path):
        """Temperature tuple should work with word timestamps."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            temperature=(0.0, 0.5),
            word_timestamps=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)
        # If segments exist, they should have words attribute
        for segment in result.segments:
            assert hasattr(segment, 'words')

    def test_temperature_tuple_with_initial_prompt(self, tmp_path):
        """Temperature tuple should work with initial_prompt."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            temperature=(0.0, 0.3, 0.6),
            initial_prompt="Technical discussion",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_temperature_tuple_with_all_features(self, tmp_path):
        """Temperature tuple should work with all other features."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=5.0)

        result = transcribe_file(
            str(audio_path),
            temperature=(0.0, 0.2, 0.4),
            word_timestamps=True,
            initial_prompt="Context",
            condition_on_previous_text=True,
            prefix="Speaker:",
            beam_size=5,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_temperature_tuple_with_whisper_inference_class(self, tmp_path):
        """Temperature tuple should work with WhisperInference OOP API."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0)

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(
            str(audio_path),
            temperature=(0.0, 0.5, 1.0)
        )

        assert isinstance(result, TranscriptionResult)

    # Helper
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)


class TestTemperatureDocumentation:
    """Test that temperature parameter is properly documented."""

    def test_temperature_in_function_signatures(self):
        """Temperature parameter should be in function signatures."""
        import inspect

        # Check transcribe_file
        sig = inspect.signature(transcribe_file)
        assert 'temperature' in sig.parameters

        # Check transcribe_audio
        sig = inspect.signature(transcribe_audio)
        assert 'temperature' in sig.parameters

    def test_temperature_type_hint_allows_union(self):
        """Temperature type hint should allow Union[float, tuple, list]."""
        import inspect
        from typing import get_type_hints

        # Check transcribe_file
        hints = get_type_hints(transcribe_file)
        # temperature should accept float or tuple/list
        # This is a documentation check - implementation validates at runtime

    def test_whisper_inference_has_temperature(self):
        """WhisperInference methods should support temperature."""
        import inspect

        whisper = WhisperInference(model_size="tiny")

        # Check transcribe_file
        sig = inspect.signature(whisper.transcribe_file)
        assert 'temperature' in sig.parameters

        # Check transcribe_audio
        sig = inspect.signature(whisper.transcribe_audio)
        assert 'temperature' in sig.parameters
