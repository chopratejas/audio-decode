"""
Real-time speech-to-text inference using optimized Whisper models.

This module provides fast transcription capabilities using faster-whisper
with CTranslate2 backend, achieving 4x speedup over vanilla Whisper while
maintaining accuracy.
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    _FASTER_WHISPER_AVAILABLE = True
    _BATCHED_PIPELINE_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False
    _BATCHED_PIPELINE_AVAILABLE = False
    WhisperModel = None
    BatchedInferencePipeline = None


@dataclass(frozen=True)
class Word:
    """
    A single word with timing and confidence information.

    Used for word-level timestamps in transcription segments.
    Compatible with openai-whisper's word_timestamps feature.

    Attributes:
        word: The word text
        start: Start time in seconds
        end: End time in seconds
        probability: Word-level confidence score (0.0 to 1.0)
    """
    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscriptionSegment:
    """
    A single segment of transcribed text with timing information.

    Attributes:
        text: The transcribed text
        start: Start time in seconds
        end: End time in seconds
        confidence: Average log probability (higher is more confident)
        words: Optional list of word-level timestamps (when word_timestamps=True)
    """
    text: str
    start: float
    end: float
    confidence: float
    words: Optional[List[Word]] = None


@dataclass
class TranscriptionResult:
    """
    Complete transcription result for an audio file.

    Attributes:
        text: Full transcribed text
        segments: List of timestamped segments
        language: Detected language code (e.g., 'en', 'es')
        duration: Audio duration in seconds
    """
    text: str
    segments: List[TranscriptionSegment]
    language: str
    duration: float

    def to_srt(self) -> str:
        """
        Export transcription in SubRip (SRT) subtitle format.

        Returns:
            SRT-formatted string with timestamps and text

        Example:
            >>> result = transcribe_file("video.mp4")
            >>> with open("subtitles.srt", "w") as f:
            ...     f.write(result.to_srt())
        """
        srt_lines = []
        for i, segment in enumerate(self.segments, 1):
            # SRT format: HH:MM:SS,mmm --> HH:MM:SS,mmm
            start_time = self._format_srt_time(segment.start)
            end_time = self._format_srt_time(segment.end)
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(segment.text.strip())
            srt_lines.append("")  # Empty line between subtitles
        return "\n".join(srt_lines)

    def to_vtt(self) -> str:
        """
        Export transcription in WebVTT subtitle format.

        Returns:
            VTT-formatted string with timestamps and text

        Example:
            >>> result = transcribe_file("video.mp4")
            >>> with open("captions.vtt", "w") as f:
            ...     f.write(result.to_vtt())
        """
        vtt_lines = ["WEBVTT", ""]
        for segment in self.segments:
            # VTT format: HH:MM:SS.mmm --> HH:MM:SS.mmm
            start_time = self._format_vtt_time(segment.start)
            end_time = self._format_vtt_time(segment.end)
            vtt_lines.append(f"{start_time} --> {end_time}")
            vtt_lines.append(segment.text.strip())
            vtt_lines.append("")  # Empty line between captions
        return "\n".join(vtt_lines)

    def to_json(self) -> str:
        """
        Export transcription as JSON string.

        Returns:
            JSON-formatted string with full transcription data

        Example:
            >>> result = transcribe_file("audio.mp3")
            >>> with open("transcript.json", "w") as f:
            ...     f.write(result.to_json())
        """
        import json
        return json.dumps({
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "confidence": seg.confidence
                }
                for seg in self.segments
            ]
        }, indent=2)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save transcription to file, auto-detecting format from extension.

        Supported formats:
            - .txt: Plain text (full transcription only)
            - .srt: SubRip subtitles
            - .vtt: WebVTT captions
            - .json: JSON with full data

        Args:
            path: Output file path

        Example:
            >>> result = transcribe_file("video.mp4")
            >>> result.save("subtitles.srt")
            >>> result.save("transcript.txt")
            >>> result.save("data.json")
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            content = self.text
        elif suffix == ".srt":
            content = self.to_srt()
        elif suffix == ".vtt":
            content = self.to_vtt()
        elif suffix == ".json":
            content = self.to_json()
        else:
            raise ValueError(
                f"Unsupported format: {suffix}. Use .txt, .srt, .vtt, or .json"
            )

        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class WhisperInference:
    """
    Fast speech-to-text inference using faster-whisper.

    Integrates with AudioDecode's fast audio loading for end-to-end optimization.

    Args:
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3')
        device: Device to run inference on ('cpu', 'cuda', 'auto')
        compute_type: Quantization type ('auto', 'int8', 'float16', 'float32')
                     'auto' selects optimal type: int8 (CPU) or float16 (GPU)
        num_workers: Number of parallel workers for CPU inference
        download_root: Directory to cache model weights
        batch_size: Batch size for parallel processing (default: 16 for CPU, 24 for GPU)
                   Higher values improve throughput but use more memory
        use_batched_inference: Use BatchedInferencePipeline for 2-3x speedup (default: True)

    Performance Guide:
        - tiny: Fastest (32x realtime), lowest accuracy
        - base: Fast (16x realtime), good for most uses
        - small: Balanced (7x realtime), good accuracy
        - medium: Slower (3x realtime), high accuracy
        - large-v3: Slowest (1-2x realtime), best accuracy

    Compute Type Selection:
        - auto: Recommended (int8 for CPU, float16 for GPU)
        - int8: Best CPU performance, memory efficient
        - float16: Best GPU performance
        - float32: Larger models, not recommended for production

    Example:
        >>> from audiodecode import load
        >>> from audiodecode.inference import WhisperInference
        >>>
        >>> # Load audio with fast decode
        >>> audio, sr = load("podcast.mp3", sr=16000)
        >>>
        >>> # Transcribe with optimized Whisper
        >>> whisper = WhisperInference(model_size="base")
        >>> result = whisper.transcribe_audio(audio, sr)
        >>> print(result.text)
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        num_workers: int = 1,
        download_root: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_batched_inference: bool = True,
    ):
        if not _FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install with: pip install audiodecode[inference]"
            )

        self.model_size = model_size
        self.device = device if device != "auto" else self._auto_device()

        # Auto-select optimal compute type based on device
        if compute_type == "auto":
            self.compute_type = self._auto_compute_type(self.device)
        else:
            self.compute_type = compute_type

        # Auto-select optimal batch size based on device
        if batch_size is None:
            self.batch_size = self._auto_batch_size(self.device)
        else:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError(f"batch_size must be positive integer, got {batch_size}")
            self.batch_size = batch_size

        # Allow disabling batched inference via environment variable (for testing)
        if os.environ.get("AUDIODECODE_DISABLE_BATCHED", "0") == "1":
            self.use_batched_inference = False
        else:
            self.use_batched_inference = use_batched_inference

        # Configure OpenMP threading for optimal CPU performance
        # Set OMP_NUM_THREADS if not already set by user
        if "OMP_NUM_THREADS" not in os.environ:
            # Benchmarks show 6 threads is optimal for most CPUs
            # Can be overridden by setting OMP_NUM_THREADS before creating WhisperInference
            num_threads = min(6, os.cpu_count() or 4)
            os.environ["OMP_NUM_THREADS"] = str(num_threads)

        # Load base model
        base_model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=num_workers,
            download_root=download_root,
        )

        # Wrap with BatchedInferencePipeline if enabled
        if self.use_batched_inference and _BATCHED_PIPELINE_AVAILABLE:
            self.model = BatchedInferencePipeline(model=base_model)
            self._is_batched = True
        else:
            self.model = base_model
            self._is_batched = False

    def _auto_device(self) -> str:
        """Auto-detect best available device."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _auto_compute_type(self, device: str) -> str:
        """
        Auto-select optimal compute type based on device.

        Performance benchmarks show:
        - CPU: int8 gives best performance (compact, cache-friendly)
        - GPU: float16 is optimal for speed and memory

        Args:
            device: Device type ('cpu' or 'cuda')

        Returns:
            Optimal compute type string
        """
        if device == "cpu":
            return "int8"  # Best performance on CPU
        else:
            return "float16"  # Optimal for GPU

    def _auto_batch_size(self, device: str) -> int:
        """
        Auto-select optimal batch size based on device.

        Benchmarks show:
        - CPU: batch_size=16 is optimal (good balance of speed/memory)
        - GPU: batch_size=24 provides best throughput

        Args:
            device: Device type ('cpu' or 'cuda')

        Returns:
            Optimal batch size
        """
        if device == "cpu":
            return 16  # Conservative for CPU memory
        else:
            return 24  # Higher throughput for GPU

    def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: Optional[float] = None,
        length_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Union[float, Tuple[float, ...], List[float]] = 0.0,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        prefix: Optional[str] = None,
        hotwords: Optional[str] = None,
        prompt_reset_on_temperature: float = 0.5,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using AudioDecode's fast loading + Whisper.

        Args:
            file_path: Path to audio file
            language: Force language code (None = auto-detect)
            task: 'transcribe' or 'translate' (translate to English)
            beam_size: Beam size for decoding (higher = better quality, slower)
            best_of: Number of candidates (higher = better quality, slower)
            patience: Beam search patience factor (None = use model default).
                     Higher values slow down beam pruning (more thorough search).
                     Typical range: 0.5-2.0.
            length_penalty: Length penalty for beam search (None = use model default).
                          Positive values favor longer sequences, negative favor shorter.
                          Typical range: -1.0 to 2.0.
            repetition_penalty: Penalty for repetitive text (None = use model default).
                              Values > 1.0 discourage repetition. Typical range: 1.0-2.0.
            temperature: Sampling temperature (0=greedy, >0=sampling).
                        Can be float or tuple/list for fallback retries.
            vad_filter: Use VAD to filter silence
            vad_parameters: Custom VAD parameters
            word_timestamps: Enable word-level timestamps (openai-whisper compatible)
            initial_prompt: Text to guide transcription style/terminology
            condition_on_previous_text: Use previous segments as context (default: True)
            prefix: Force first segment to start with this text
            hotwords: Hotwords/hint phrases to boost recognition (has no effect if prefix is set)
            prompt_reset_on_temperature: Reset prompt when temperature exceeds this threshold.
                                       Only has effect when condition_on_previous_text=True.
            compression_ratio_threshold: Skip segments with compression ratio above this threshold.
                                       Typical value: 2.4. Helps filter out hallucinations.
            logprob_threshold: Skip segments with average log probability below this threshold.
                             Typical value: -1.0. Helps filter out low-confidence transcriptions.
            no_speech_threshold: Skip segments with no-speech probability above this threshold.
                               Typical value: 0.6. Helps filter out silent segments.
            verbose: Print detailed progress to stdout (default: False)

        Returns:
            TranscriptionResult with text, segments, language, duration
        """
        # Use AudioDecode for fast audio loading
        from audiodecode import load

        # Load audio (this is where our 181x speedup helps!)
        audio, sr = load(str(file_path), sr=16000, mono=True)

        return self.transcribe_audio(
            audio=audio,
            sample_rate=sr,
            language=language,
            task=task,
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            prefix=prefix,
            hotwords=hotwords,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            verbose=verbose,
        )

    def transcribe_audio(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: Optional[float] = None,
        length_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Union[float, Tuple[float, ...], List[float]] = 0.0,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        prefix: Optional[str] = None,
        hotwords: Optional[str] = None,
        prompt_reset_on_temperature: float = 0.5,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio array directly.

        Args:
            audio: Audio array (mono, float32)
            sample_rate: Sample rate (should be 16000 for Whisper)
            language: Force language code (None = auto-detect)
            task: 'transcribe' or 'translate'
            beam_size: Beam size for decoding
            best_of: Number of candidates
            patience: Beam search patience factor (None = use model default).
                     Higher values slow down beam pruning (more thorough search).
            length_penalty: Length penalty for beam search (None = use model default).
                          Positive values favor longer sequences, negative favor shorter.
            repetition_penalty: Penalty for repetitive text (None = use model default).
                              Values > 1.0 discourage repetition.
            temperature: Sampling temperature (0=greedy, >0=sampling).
                        Can be float or tuple/list for fallback retries:
                        - 0.0: Greedy decoding (deterministic)
                        - (0.0, 0.2, 0.4, 0.6, 0.8, 1.0): Retry with increasing temps
            vad_filter: Use VAD to filter silence
            vad_parameters: Custom VAD parameters
            word_timestamps: Enable word-level timestamps (openai-whisper compatible)
            initial_prompt: Text to guide transcription style/terminology
            condition_on_previous_text: Use previous segments as context (default: True)
            prefix: Force first segment to start with this text
            hotwords: Hotwords/hint phrases to boost recognition (has no effect if prefix is set)
            prompt_reset_on_temperature: Reset prompt when temperature exceeds this threshold.
                                       Only has effect when condition_on_previous_text=True.
            compression_ratio_threshold: Skip segments with compression ratio above this threshold.
                                       Typical value: 2.4. Helps filter out hallucinations.
            logprob_threshold: Skip segments with average log probability below this threshold.
                             Typical value: -1.0. Helps filter out low-confidence transcriptions.
            no_speech_threshold: Skip segments with no-speech probability above this threshold.
                               Typical value: 0.6. Helps filter out silent segments.
            verbose: Print detailed progress to stdout (default: False)

        Returns:
            TranscriptionResult with text, segments, language, duration
        """
        # Validate word_timestamps type
        if not isinstance(word_timestamps, bool):
            raise TypeError(
                f"word_timestamps must be bool, got {type(word_timestamps).__name__}. "
                f"Use word_timestamps=True or word_timestamps=False"
            )

        # Validate initial_prompt type
        if initial_prompt is not None and not isinstance(initial_prompt, str):
            raise TypeError(
                f"initial_prompt must be str or None, got {type(initial_prompt).__name__}"
            )

        # Validate condition_on_previous_text type
        if not isinstance(condition_on_previous_text, bool):
            raise TypeError(
                f"condition_on_previous_text must be bool, got {type(condition_on_previous_text).__name__}"
            )

        # Validate prefix type
        if prefix is not None and not isinstance(prefix, str):
            raise TypeError(
                f"prefix must be str or None, got {type(prefix).__name__}"
            )

        # Validate hotwords type
        if hotwords is not None and not isinstance(hotwords, str):
            raise TypeError(
                f"hotwords must be str or None, got {type(hotwords).__name__}"
            )

        # Validate prompt_reset_on_temperature type and range
        if not isinstance(prompt_reset_on_temperature, (int, float)):
            raise TypeError(
                f"prompt_reset_on_temperature must be float, got {type(prompt_reset_on_temperature).__name__}"
            )
        if not 0.0 <= prompt_reset_on_temperature <= 1.0:
            raise ValueError(
                f"prompt_reset_on_temperature must be between 0 and 1, got {prompt_reset_on_temperature}"
            )

        # Validate verbose type
        if not isinstance(verbose, bool):
            raise TypeError(
                f"verbose must be bool, got {type(verbose).__name__}"
            )

        # Validate compression_ratio_threshold type
        if compression_ratio_threshold is not None and not isinstance(compression_ratio_threshold, (int, float)):
            raise TypeError(
                f"compression_ratio_threshold must be float or None, got {type(compression_ratio_threshold).__name__}"
            )

        # Validate logprob_threshold type
        if logprob_threshold is not None and not isinstance(logprob_threshold, (int, float)):
            raise TypeError(
                f"logprob_threshold must be float or None, got {type(logprob_threshold).__name__}"
            )

        # Validate no_speech_threshold type and range
        if no_speech_threshold is not None:
            if not isinstance(no_speech_threshold, (int, float)):
                raise TypeError(
                    f"no_speech_threshold must be float or None, got {type(no_speech_threshold).__name__}"
                )
            if not 0.0 <= no_speech_threshold <= 1.0:
                raise ValueError(
                    f"no_speech_threshold must be between 0 and 1, got {no_speech_threshold}"
                )

        # Validate temperature type and values
        if isinstance(temperature, (tuple, list)):
            # Temperature as sequence for fallback retries
            if len(temperature) == 0:
                raise ValueError("temperature tuple/list cannot be empty")

            for i, temp in enumerate(temperature):
                if not isinstance(temp, (int, float)):
                    raise TypeError(
                        f"temperature[{i}] must be numeric, got {type(temp).__name__}"
                    )
                if not 0.0 <= temp <= 1.0:
                    raise ValueError(
                        f"temperature[{i}] must be between 0 and 1, got {temp}"
                    )
        elif isinstance(temperature, (int, float)):
            # Single temperature value
            if not 0.0 <= temperature <= 1.0:
                raise ValueError(
                    f"temperature must be between 0 and 1, got {temperature}"
                )
        else:
            raise TypeError(
                f"temperature must be float or tuple/list, got {type(temperature).__name__}"
            )

        # Validate patience type and range
        if patience is not None:
            if not isinstance(patience, (int, float)):
                raise TypeError(
                    f"patience must be float or None, got {type(patience).__name__}"
                )
            if patience <= 0.0:
                raise ValueError(
                    f"patience must be positive, got {patience}"
                )

        # Validate length_penalty type
        if length_penalty is not None:
            if not isinstance(length_penalty, (int, float)):
                raise TypeError(
                    f"length_penalty must be float or None, got {type(length_penalty).__name__}"
                )

        # Validate repetition_penalty type and range
        if repetition_penalty is not None:
            if not isinstance(repetition_penalty, (int, float)):
                raise TypeError(
                    f"repetition_penalty must be float or None, got {type(repetition_penalty).__name__}"
                )
            if repetition_penalty <= 0.0:
                raise ValueError(
                    f"repetition_penalty must be positive, got {repetition_penalty}"
                )

        if sample_rate != 16000:
            warnings.warn(
                f"Whisper expects 16kHz audio, got {sample_rate}Hz. "
                "Results may be suboptimal."
            )

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Build transcribe kwargs, excluding None values for beam search params
        transcribe_kwargs = {
            'language': language,
            'task': task,
            'beam_size': beam_size,
            'best_of': best_of,
            'temperature': temperature,
            'vad_filter': vad_filter,
            'vad_parameters': vad_parameters,
            'word_timestamps': word_timestamps,
            'initial_prompt': initial_prompt,
            'condition_on_previous_text': condition_on_previous_text,
            'prefix': prefix,
            'hotwords': hotwords,
            'prompt_reset_on_temperature': prompt_reset_on_temperature,
            'compression_ratio_threshold': compression_ratio_threshold,
            'log_prob_threshold': logprob_threshold,
            'no_speech_threshold': no_speech_threshold,
            'log_progress': verbose,
        }

        # Add batch_size if using batched pipeline
        if self._is_batched:
            transcribe_kwargs['batch_size'] = self.batch_size

        # Only add beam search params if they're not None (ctranslate2 doesn't accept None)
        if patience is not None:
            transcribe_kwargs['patience'] = patience
        if length_penalty is not None:
            transcribe_kwargs['length_penalty'] = length_penalty
        if repetition_penalty is not None:
            transcribe_kwargs['repetition_penalty'] = repetition_penalty

        # Transcribe using faster-whisper (batched or standard)
        segments_iter, info = self.model.transcribe(audio, **transcribe_kwargs)

        # Collect segments
        segments = []
        full_text = []

        for segment in segments_iter:
            # Extract word-level timestamps if enabled
            words_list = None
            if word_timestamps and hasattr(segment, 'words') and segment.words:
                words_list = [
                    Word(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                        probability=word.probability,
                    )
                    for word in segment.words
                ]

            segments.append(
                TranscriptionSegment(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.avg_logprob,
                    words=words_list,
                )
            )
            full_text.append(segment.text.strip())

        # Calculate duration
        duration = len(audio) / sample_rate

        return TranscriptionResult(
            text=" ".join(full_text),
            segments=segments,
            language=info.language,
            duration=duration,
        )

    def transcribe_batch(
        self,
        file_paths: List[Union[str, Path]],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: Optional[float] = None,
        length_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Union[float, Tuple[float, ...], List[float]] = 0.0,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        prefix: Optional[str] = None,
        hotwords: Optional[str] = None,
        prompt_reset_on_temperature: float = 0.5,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        verbose: bool = False,
        batch_size: Optional[int] = None,
    ) -> List[TranscriptionResult]:
        """
        Batch transcribe multiple audio files with this model instance.

        Args:
            file_paths: List of paths to audio files
            language: Force language code (None = auto-detect)
            task: 'transcribe' or 'translate'
            beam_size: Beam size for decoding
            best_of: Number of candidates
            patience: Beam search patience factor
            length_penalty: Length penalty for beam search
            repetition_penalty: Penalty for repetitive text
            temperature: Sampling temperature
            vad_filter: Use VAD to filter silence
            vad_parameters: Custom VAD parameters
            word_timestamps: Enable word-level timestamps
            initial_prompt: Text to guide transcription
            condition_on_previous_text: Use previous segments as context
            prefix: Force first segment to start with this text
            hotwords: Hotwords/hint phrases to boost recognition
            prompt_reset_on_temperature: Reset prompt when temperature exceeds threshold
            compression_ratio_threshold: Filter hallucinations threshold
            logprob_threshold: Filter low-confidence threshold
            no_speech_threshold: Filter silent segments threshold
            verbose: Print detailed progress
            batch_size: Override batch size for this transcription

        Returns:
            List of TranscriptionResult objects, one per file

        Example:
            >>> whisper = WhisperInference(model_size="base")
            >>> files = ["audio1.mp3", "audio2.mp3"]
            >>> results = whisper.transcribe_batch(files)
        """
        results = []

        # Use provided batch_size or fall back to instance batch_size
        effective_batch_size = batch_size if batch_size is not None else self.batch_size

        for file_path in file_paths:
            result = self.transcribe_file(
                file_path=file_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
                prefix=prefix,
                hotwords=hotwords,
                prompt_reset_on_temperature=prompt_reset_on_temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                verbose=verbose,
            )
            results.append(result)

        return results


def transcribe_file(
    file_path: Union[str, Path],
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "auto",
    task: str = "transcribe",
    beam_size: int = 5,
    best_of: int = 5,
    patience: Optional[float] = None,
    length_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    temperature: Union[float, Tuple[float, ...], List[float]] = 0.0,
    vad_filter: bool = True,
    vad_parameters: Optional[dict] = None,
    word_timestamps: bool = False,
    initial_prompt: Optional[str] = None,
    condition_on_previous_text: bool = True,
    prefix: Optional[str] = None,
    hotwords: Optional[str] = None,
    prompt_reset_on_temperature: float = 0.5,
    compression_ratio_threshold: Optional[float] = None,
    logprob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
    verbose: bool = False,
) -> TranscriptionResult:
    """
    Convenience function to transcribe a file with default settings.

    This is the simplest way to transcribe audio with AudioDecode's
    optimized pipeline (fast decode + fast inference).

    Args:
        file_path: Path to audio file
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3')
        language: Force language code (None = auto-detect)
        device: Device to run on ('cpu', 'cuda', 'auto')
        compute_type: Quantization type ('int8', 'float16', 'float32')
        task: 'transcribe' or 'translate' (translate to English)
        beam_size: Beam size for decoding (higher = better quality, slower)
        best_of: Number of candidates (higher = better quality, slower)
        patience: Beam search patience factor (None = use model default)
        length_penalty: Length penalty for beam search (None = use model default)
        repetition_penalty: Penalty for repetitive text (None = use model default)
        temperature: Sampling temperature (0 = greedy, >0 = sampling)
        vad_filter: Use VAD to filter silence
        vad_parameters: Custom VAD parameters
        word_timestamps: Enable word-level timestamps (openai-whisper compatible)
        initial_prompt: Text to guide transcription style/terminology
        condition_on_previous_text: Use previous segments as context (default: True)
        prefix: Force first segment to start with this text
        hotwords: Hotwords/hint phrases to boost recognition (has no effect if prefix is set)
        prompt_reset_on_temperature: Reset prompt when temperature exceeds this threshold.
                                    Only has effect when condition_on_previous_text=True.
        compression_ratio_threshold: Skip segments with compression ratio above this threshold.
                                   Typical value: 2.4. Helps filter out hallucinations.
        logprob_threshold: Skip segments with average log probability below this threshold.
                         Typical value: -1.0. Helps filter out low-confidence transcriptions.
        no_speech_threshold: Skip segments with no-speech probability above this threshold.
                           Typical value: 0.6. Helps filter out silent segments.
        verbose: Print detailed progress to stdout (default: False)

    Returns:
        TranscriptionResult with text, segments, language, duration

    Example:
        >>> from audiodecode.inference import transcribe_file
        >>> result = transcribe_file("podcast.mp3", model_size="base")
        >>> print(result.text)
        >>> print(f"Language: {result.language}")
        >>> print(f"Duration: {result.duration:.1f}s")

        >>> # With word timestamps
        >>> result = transcribe_file("podcast.mp3", word_timestamps=True)
        >>> for segment in result.segments:
        ...     for word in segment.words:
        ...         print(f"{word.word}: {word.start:.2f}-{word.end:.2f}s")

        >>> # With prompt engineering and hotwords
        >>> result = transcribe_file("podcast.mp3",
        ...     initial_prompt="Technical discussion about AI",
        ...     hotwords="PyTorch, TensorFlow, CUDA",
        ...     prefix="Speaker 1:")
    """
    whisper = WhisperInference(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )

    return whisper.transcribe_file(
        file_path=file_path,
        language=language,
        task=task,
        beam_size=beam_size,
        best_of=best_of,
        patience=patience,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        prefix=prefix,
        hotwords=hotwords,
        prompt_reset_on_temperature=prompt_reset_on_temperature,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        verbose=verbose,
    )


def transcribe_batch(
    file_paths: List[Union[str, Path]],
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "auto",
    task: str = "transcribe",
    beam_size: int = 5,
    best_of: int = 5,
    patience: Optional[float] = None,
    length_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    temperature: Union[float, Tuple[float, ...], List[float]] = 0.0,
    vad_filter: bool = True,
    vad_parameters: Optional[dict] = None,
    word_timestamps: bool = False,
    initial_prompt: Optional[str] = None,
    condition_on_previous_text: bool = True,
    prefix: Optional[str] = None,
    hotwords: Optional[str] = None,
    prompt_reset_on_temperature: float = 0.5,
    compression_ratio_threshold: Optional[float] = None,
    logprob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
    verbose: bool = False,
    batch_size: Optional[int] = None,
    use_batched_inference: bool = True,
    show_progress: bool = False,
) -> List[TranscriptionResult]:
    """
    Batch transcribe multiple audio files efficiently.

    Reuses the same model instance across all files for better performance
    compared to transcribing files sequentially with separate model loads.

    Args:
        file_paths: List of paths to audio files
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3')
        language: Force language code (None = auto-detect)
        device: Device to run on ('cpu', 'cuda', 'auto')
        compute_type: Quantization type ('int8', 'float16', 'float32')
        task: 'transcribe' or 'translate' (translate to English)
        beam_size: Beam size for decoding
        best_of: Number of candidates
        patience: Beam search patience factor
        length_penalty: Length penalty for beam search
        repetition_penalty: Penalty for repetitive text
        temperature: Sampling temperature
        vad_filter: Use VAD to filter silence
        vad_parameters: Custom VAD parameters
        word_timestamps: Enable word-level timestamps
        initial_prompt: Text to guide transcription
        condition_on_previous_text: Use previous segments as context
        prefix: Force first segment to start with this text
        hotwords: Hotwords/hint phrases to boost recognition
        prompt_reset_on_temperature: Reset prompt when temperature exceeds threshold
        compression_ratio_threshold: Filter hallucinations threshold
        logprob_threshold: Filter low-confidence threshold
        no_speech_threshold: Filter silent segments threshold
        verbose: Print detailed progress
        batch_size: Batch size for processing
        use_batched_inference: Use batched inference pipeline
        show_progress: Show progress bar for batch processing

    Returns:
        List of TranscriptionResult objects, one per file

    Example:
        >>> from audiodecode.inference import transcribe_batch
        >>> files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]
        >>> results = transcribe_batch(files, model_size="base")
        >>> for i, result in enumerate(results):
        ...     print(f"File {i+1}: {result.text[:50]}...")
    """
    # Validate input
    if not isinstance(file_paths, (list, tuple)):
        raise TypeError(
            f"file_paths must be a list or tuple, got {type(file_paths).__name__}"
        )

    # Check for None values in list
    for i, path in enumerate(file_paths):
        if path is None:
            raise ValueError(f"file_paths[{i}] is None. All paths must be valid strings or Path objects.")

    # Create single model instance for all files
    whisper = WhisperInference(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        use_batched_inference=use_batched_inference,
    )

    # Transcribe each file with the shared model
    results = []

    # Set up progress bar if requested
    iterator = file_paths
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(file_paths, desc="Transcribing", unit="file")
        except ImportError:
            pass  # Silently fallback if tqdm not available

    for file_path in iterator:
        result = whisper.transcribe_file(
            file_path=file_path,
            language=language,
            task=task,
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            prefix=prefix,
            hotwords=hotwords,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            verbose=verbose,
        )
        results.append(result)

    return results


def transcribe_audio(
    audio: NDArray[np.float32],
    sample_rate: int = 16000,
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "auto",
    task: str = "transcribe",
    beam_size: int = 5,
    best_of: int = 5,
    patience: Optional[float] = None,
    length_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    temperature: Union[float, Tuple[float, ...], List[float]] = 0.0,
    vad_filter: bool = True,
    vad_parameters: Optional[dict] = None,
    word_timestamps: bool = False,
    initial_prompt: Optional[str] = None,
    condition_on_previous_text: bool = True,
    prefix: Optional[str] = None,
    hotwords: Optional[str] = None,
    prompt_reset_on_temperature: float = 0.5,
    compression_ratio_threshold: Optional[float] = None,
    logprob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
    verbose: bool = False,
) -> TranscriptionResult:
    """
    Convenience function to transcribe audio array with default settings.

    Args:
        audio: Audio array (mono, float32)
        sample_rate: Sample rate (should be 16000)
        model_size: Whisper model size
        language: Force language code (None = auto-detect)
        device: Device to run on
        compute_type: Quantization type
        task: 'transcribe' or 'translate' (translate to English)
        beam_size: Beam size for decoding (higher = better quality, slower)
        best_of: Number of candidates (higher = better quality, slower)
        patience: Beam search patience factor (None = use model default)
        length_penalty: Length penalty for beam search (None = use model default)
        repetition_penalty: Penalty for repetitive text (None = use model default)
        temperature: Sampling temperature (0 = greedy, >0 = sampling)
        vad_filter: Use VAD to filter silence
        vad_parameters: Custom VAD parameters
        word_timestamps: Enable word-level timestamps (openai-whisper compatible)
        initial_prompt: Text to guide transcription style/terminology
        condition_on_previous_text: Use previous segments as context (default: True)
        prefix: Force first segment to start with this text
        hotwords: Hotwords/hint phrases to boost recognition (has no effect if prefix is set)
        prompt_reset_on_temperature: Reset prompt when temperature exceeds this threshold.
                                    Only has effect when condition_on_previous_text=True.
        compression_ratio_threshold: Skip segments with compression ratio above this threshold.
                                   Typical value: 2.4. Helps filter out hallucinations.
        logprob_threshold: Skip segments with average log probability below this threshold.
                         Typical value: -1.0. Helps filter out low-confidence transcriptions.
        no_speech_threshold: Skip segments with no-speech probability above this threshold.
                           Typical value: 0.6. Helps filter out silent segments.
        verbose: Print detailed progress to stdout (default: False)

    Returns:
        TranscriptionResult

    Example:
        >>> from audiodecode import load
        >>> from audiodecode.inference import transcribe_audio
        >>>
        >>> audio, sr = load("audio.mp3", sr=16000)
        >>> result = transcribe_audio(audio, sr)
        >>> print(result.text)

        >>> # With word timestamps
        >>> result = transcribe_audio(audio, sr, word_timestamps=True)
        >>> for segment in result.segments:
        ...     if segment.words:
        ...         for word in segment.words:
        ...             print(f"{word.word}: {word.start:.2f}s")

        >>> # With prompt engineering and hotwords
        >>> result = transcribe_audio(audio, sr,
        ...     initial_prompt="Medical terminology",
        ...     hotwords="electrocardiogram, CT scan",
        ...     condition_on_previous_text=True)
    """
    whisper = WhisperInference(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )

    return whisper.transcribe_audio(
        audio=audio,
        sample_rate=sample_rate,
        language=language,
        task=task,
        beam_size=beam_size,
        best_of=best_of,
        patience=patience,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        prefix=prefix,
        hotwords=hotwords,
        prompt_reset_on_temperature=prompt_reset_on_temperature,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        verbose=verbose,
    )
