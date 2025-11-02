"""
Test word-level timestamp support for AudioDecode.

Tests the critical feature that users migrating from openai-whisper expect:
word-level timestamps for karaoke-style subtitles and fine-grained alignment.
"""

import pytest
import numpy as np
from pathlib import Path

# These imports will fail initially - that's the point of TDD!
from audiodecode.inference import (
    transcribe_file,
    transcribe_audio,
    WhisperInference,
    Word,  # NEW
    TranscriptionSegment,
    TranscriptionResult,
)


class TestWordTimestamps:
    """Test word-level timestamp functionality."""

    def test_word_timestamps_enabled_returns_words(self, tmp_path):
        """
        When word_timestamps=True, segments should contain a words array.

        This is the #1 feature users expect when migrating from openai-whisper.
        """
        # Create a simple test audio file (sine wave)
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=2.0, text="Hello world test")

        # Transcribe with word timestamps (disable VAD for test audio)
        result = transcribe_file(str(audio_path), word_timestamps=True, model_size="tiny", vad_filter=False)

        # Verify structure
        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) > 0

        # Check first segment has words
        first_segment = result.segments[0]
        assert hasattr(first_segment, 'words'), "Segment should have 'words' attribute"
        assert first_segment.words is not None, "Words should not be None when word_timestamps=True"
        assert len(first_segment.words) > 0, "Words array should not be empty"

        # Check word structure
        first_word = first_segment.words[0]
        assert isinstance(first_word, Word)
        assert hasattr(first_word, 'word')
        assert hasattr(first_word, 'start')
        assert hasattr(first_word, 'end')
        assert hasattr(first_word, 'probability')

        # Validate types
        assert isinstance(first_word.word, str)
        assert isinstance(first_word.start, float)
        assert isinstance(first_word.end, float)
        assert isinstance(first_word.probability, float)

        # Validate ranges
        assert first_word.start >= 0.0
        assert first_word.end > first_word.start
        assert 0.0 <= first_word.probability <= 1.0

    def test_word_timestamps_disabled_no_words(self, tmp_path):
        """When word_timestamps=False, segments should not have words."""
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=2.0, text="Test audio")

        # Transcribe without word timestamps
        result = transcribe_file(str(audio_path), word_timestamps=False, model_size="tiny")

        # Verify no words
        for segment in result.segments:
            assert segment.words is None or len(segment.words) == 0

    def test_word_level_timing_accuracy(self, tmp_path):
        """Word timestamps should be within segment boundaries and reasonably accurate."""
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=5.0, text="This is a test of word timing")

        result = transcribe_file(str(audio_path), word_timestamps=True, model_size="tiny")

        for segment in result.segments:
            if segment.words:
                # All words should be within segment boundaries
                for word in segment.words:
                    assert word.start >= segment.start - 0.1, f"Word start {word.start} before segment start {segment.start}"
                    assert word.end <= segment.end + 0.1, f"Word end {word.end} after segment end {segment.end}"
                    assert word.end > word.start, f"Word end {word.end} not after start {word.start}"

                # Words should be in chronological order
                for i in range(len(segment.words) - 1):
                    curr_word = segment.words[i]
                    next_word = segment.words[i + 1]
                    assert curr_word.end <= next_word.start + 0.2, \
                        f"Words not in order: {curr_word.word} ends at {curr_word.end}, {next_word.word} starts at {next_word.start}"

    def test_word_confidence_scores(self, tmp_path):
        """Word probability scores should be reasonable."""
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=3.0, text="Clear speech test")

        result = transcribe_file(str(audio_path), word_timestamps=True, model_size="tiny")

        for segment in result.segments:
            if segment.words:
                for word in segment.words:
                    # Probabilities should be between 0 and 1
                    assert 0.0 <= word.probability <= 1.0

                    # For clear speech with "tiny" model, expect reasonable confidence
                    # (This is a sanity check, not a strict requirement)
                    assert word.probability > 0.01, f"Suspiciously low probability for word: {word.word}"

    def test_words_match_segment_text(self, tmp_path):
        """Words concatenated should approximately match segment text."""
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=3.0, text="Hello world test")

        result = transcribe_file(str(audio_path), word_timestamps=True, model_size="tiny")

        for segment in result.segments:
            if segment.words and len(segment.words) > 0:
                # Concatenate words
                words_text = " ".join(w.word.strip() for w in segment.words)
                segment_text = segment.text.strip()

                # They should be very similar (allowing for punctuation differences)
                words_text_clean = words_text.replace(",", "").replace(".", "").lower()
                segment_text_clean = segment_text.replace(",", "").replace(".", "").lower()

                # Check substantial overlap
                words_set = set(words_text_clean.split())
                segment_set = set(segment_text_clean.split())

                if words_set and segment_set:
                    overlap = len(words_set & segment_set) / len(segment_set)
                    assert overlap > 0.5, f"Word/segment mismatch: '{words_text}' vs '{segment_text}'"

    def test_word_timestamps_with_transcribe_audio(self):
        """word_timestamps should work with transcribe_audio (numpy input)."""
        # Create synthetic audio (2 seconds of 440Hz tone)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        # Transcribe with word timestamps
        result = transcribe_audio(audio, sample_rate=sample_rate, word_timestamps=True, model_size="tiny")

        # Should return result (even if it's just "[BLANK_AUDIO]" or similar)
        assert isinstance(result, TranscriptionResult)

        # If there are segments, they should have words structure
        for segment in result.segments:
            assert hasattr(segment, 'words')

    def test_word_timestamps_parameter_types(self, tmp_path):
        """word_timestamps parameter should accept bool only."""
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=1.0, text="Test")

        # Valid: bool
        result1 = transcribe_file(str(audio_path), word_timestamps=True, model_size="tiny")
        assert isinstance(result1, TranscriptionResult)

        result2 = transcribe_file(str(audio_path), word_timestamps=False, model_size="tiny")
        assert isinstance(result2, TranscriptionResult)

        # Invalid: string should raise TypeError
        with pytest.raises(TypeError):
            transcribe_file(str(audio_path), word_timestamps="true", model_size="tiny")

    def test_word_timestamps_with_other_features(self, tmp_path):
        """word_timestamps should work alongside other features."""
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=3.0, text="Test with options")

        # Combine with other parameters
        result = transcribe_file(
            str(audio_path),
            word_timestamps=True,
            language="en",
            beam_size=3,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"

        # Should still have words
        if len(result.segments) > 0:
            assert hasattr(result.segments[0], 'words')

    def test_word_dataclass_structure(self):
        """Test the Word dataclass structure."""
        # Create a Word instance
        word = Word(
            word="hello",
            start=0.5,
            end=0.8,
            probability=0.95
        )

        # Validate structure
        assert word.word == "hello"
        assert word.start == 0.5
        assert word.end == 0.8
        assert word.probability == 0.95

        # Test immutability (dataclass with frozen=True)
        with pytest.raises(AttributeError):
            word.word = "world"

    def test_empty_audio_with_word_timestamps(self, tmp_path):
        """word_timestamps on silent audio should not crash."""
        audio_path = tmp_path / "silent.wav"
        self._create_silent_audio(audio_path, duration=2.0)

        # Should not crash
        result = transcribe_file(str(audio_path), word_timestamps=True, model_size="tiny")

        assert isinstance(result, TranscriptionResult)
        # May have no segments or segments with no words - that's fine

    # Helper methods

    def _create_test_audio(self, path: Path, duration: float, text: str):
        """Create a WAV file with synthesized speech (or silence for testing)."""
        import soundfile as sf

        sample_rate = 16000
        samples = int(sample_rate * duration)

        # Create silence (real speech would be better, but this tests the pipeline)
        # In practice, faster-whisper will transcribe silence as "[BLANK_AUDIO]" or similar
        audio = np.zeros(samples, dtype=np.float32)

        # Add a tiny bit of noise so it's not perfectly silent
        audio += np.random.normal(0, 0.01, samples).astype(np.float32)

        sf.write(path, audio, sample_rate)

    def _create_silent_audio(self, path: Path, duration: float):
        """Create a completely silent WAV file."""
        import soundfile as sf

        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.zeros(samples, dtype=np.float32)

        sf.write(path, audio, sample_rate)


class TestWordTimestampsWithRealAudio:
    """Tests with actual audio fixtures (if available)."""

    @pytest.mark.skipif(not Path("fixtures/test_audio.wav").exists(), reason="No fixtures available")
    def test_word_timestamps_real_audio(self):
        """Test word timestamps on real audio fixture."""
        result = transcribe_file("fixtures/test_audio.wav", word_timestamps=True, model_size="tiny")

        # Should have words
        assert len(result.segments) > 0
        assert result.segments[0].words is not None
        assert len(result.segments[0].words) > 0

        # Words should have reasonable timing
        for segment in result.segments:
            if segment.words:
                total_word_duration = sum(w.end - w.start for w in segment.words)
                segment_duration = segment.end - segment.start

                # Words should cover a reasonable portion of the segment
                # (not exact because of silence/pauses)
                assert total_word_duration <= segment_duration * 1.1


class TestWordTimestampEdgeCases:
    """Edge cases for word timestamp feature."""

    def test_very_short_audio(self, tmp_path):
        """word_timestamps on very short audio (< 0.5s)."""
        audio_path = tmp_path / "short.wav"
        self._create_test_audio(audio_path, duration=0.3, text="Hi")

        result = transcribe_file(str(audio_path), word_timestamps=True, model_size="tiny")
        assert isinstance(result, TranscriptionResult)

    def test_very_long_audio(self, tmp_path):
        """word_timestamps should work on long audio without memory issues."""
        audio_path = tmp_path / "long.wav"
        self._create_test_audio(audio_path, duration=60.0, text="Long test")

        # Should not crash or run out of memory
        result = transcribe_file(str(audio_path), word_timestamps=True, model_size="tiny")
        assert isinstance(result, TranscriptionResult)

    def test_word_timestamps_different_languages(self, tmp_path):
        """word_timestamps should work across languages."""
        audio_path = tmp_path / "test.wav"
        self._create_test_audio(audio_path, duration=2.0, text="Test")

        # English
        result_en = transcribe_file(str(audio_path), word_timestamps=True, language="en", model_size="tiny")
        assert isinstance(result_en, TranscriptionResult)

        # Spanish (even if audio isn't Spanish, should not crash)
        result_es = transcribe_file(str(audio_path), word_timestamps=True, language="es", model_size="tiny")
        assert isinstance(result_es, TranscriptionResult)

    # Helper method (same as above)
    def _create_test_audio(self, path: Path, duration: float, text: str):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)
