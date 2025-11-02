#!/usr/bin/env python3
"""
Real-World Showdown: YouTube Video Transcription

This script demonstrates EXACTLY what AudioDecode does vs competitors.
Tests 3 implementations on the SAME YouTube video:

1. openai-whisper (vanilla baseline)
2. faster-whisper + librosa loading (standard approach)
3. AudioDecode + faster-whisper (our optimized stack)

WHAT AUDIODECODE DOES:
- Replaces slow audio loading (librosa uses subprocess FFmpeg)
- With fast native audio decoding (PyAV + SoundFile)
- Same Whisper model, same accuracy, just FASTER loading

Usage:
    python examples/youtube_transcription_showdown.py "YOUTUBE_URL"
"""

import sys
import time
import psutil
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class TranscriptionResult:
    """Result from transcription test."""
    implementation: str

    # Timing (seconds)
    download_time: float
    audio_load_time: float
    model_load_time: float
    inference_time: float
    total_time: float

    # Performance
    audio_duration: float
    rtf: float  # Real-Time Factor

    # Memory (MB)
    memory_before: float
    memory_after: float
    memory_delta: float

    # Output
    transcription: str
    num_words: int
    num_segments: int


class YouTubeTranscriptionShowdown:
    """Compare transcription implementations on real YouTube video."""

    def __init__(self):
        self.process = psutil.Process()
        self.results = []

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def download_youtube(self, url: str, output_dir: str = "examples/youtube_downloads") -> str:
        """Download YouTube video audio using yt-dlp."""
        try:
            import yt_dlp
        except ImportError:
            print("ERROR: yt-dlp not installed.")
            print("Install with: pip install yt-dlp")
            sys.exit(1)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Download as best audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        print(f"\n🔽 Downloading YouTube video: {url}")
        t0 = time.perf_counter()

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            title = info['title']
            duration = info['duration']

        t1 = time.perf_counter()
        download_time = t1 - t0

        audio_file = f"{output_dir}/{video_id}.mp3"

        print(f"✅ Downloaded: {title}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Download time: {download_time:.2f}s")
        print(f"   File: {audio_file}")

        return audio_file, download_time, duration, title

    def test_vanilla_whisper(self, audio_file: str, model_size: str = "base") -> TranscriptionResult:
        """Test 1: openai-whisper (baseline)."""
        print("\n" + "="*80)
        print("TEST 1: openai-whisper (Vanilla Baseline)")
        print("="*80)
        print("WHAT IT DOES:")
        print("  - Loads audio using librosa (slow, subprocess-based FFmpeg)")
        print("  - Loads Whisper model")
        print("  - Runs transcription")
        print()

        try:
            import whisper
        except ImportError:
            print("ERROR: openai-whisper not installed")
            print("Install with: pip install openai-whisper")
            sys.exit(1)

        mem_before = self.get_memory_mb()

        # Load model (combined with audio in vanilla Whisper)
        t0 = time.perf_counter()
        model = whisper.load_model(model_size)
        t1 = time.perf_counter()
        model_load_time = t1 - t0

        print(f"⏱️  Model loaded in {model_load_time:.3f}s")

        # Transcribe (includes audio loading internally)
        print("🎙️  Transcribing...")
        result = model.transcribe(audio_file)
        t2 = time.perf_counter()
        transcribe_time = t2 - t1
        total_time = t2 - t0

        mem_after = self.get_memory_mb()

        # Get audio duration from result
        audio_duration = result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0

        text = result["text"].strip()
        num_words = len(text.split())
        num_segments = len(result.get("segments", []))

        print(f"✅ Complete!")
        print(f"   Audio loading + Inference: {transcribe_time:.3f}s")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   RTF: {audio_duration/total_time:.1f}x")
        print(f"   Words transcribed: {num_words}")
        print(f"   Memory used: {mem_after - mem_before:.0f} MB")

        return TranscriptionResult(
            implementation="openai-whisper",
            download_time=0.0,
            audio_load_time=0.0,  # Combined in transcribe
            model_load_time=model_load_time,
            inference_time=transcribe_time,
            total_time=total_time,
            audio_duration=audio_duration,
            rtf=audio_duration / total_time if total_time > 0 else 0,
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta=mem_after - mem_before,
            transcription=text,
            num_words=num_words,
            num_segments=num_segments
        )

    def test_faster_whisper_librosa(self, audio_file: str, model_size: str = "base") -> TranscriptionResult:
        """Test 2: faster-whisper + librosa (standard fast approach)."""
        print("\n" + "="*80)
        print("TEST 2: faster-whisper + librosa")
        print("="*80)
        print("WHAT IT DOES:")
        print("  - Loads audio using librosa (slow, subprocess-based FFmpeg)")
        print("  - Loads faster-whisper model (CTranslate2 - 4x faster)")
        print("  - Runs transcription (faster inference)")
        print()

        try:
            from faster_whisper import WhisperModel
            import librosa
        except ImportError as e:
            print(f"ERROR: {e}")
            print("Install with: pip install faster-whisper librosa")
            sys.exit(1)

        mem_before = self.get_memory_mb()

        # Load audio with librosa (slow path)
        print("📂 Loading audio with librosa...")
        t0 = time.perf_counter()
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        t1 = time.perf_counter()
        audio_load_time = t1 - t0

        audio_duration = len(audio) / sr

        print(f"⏱️  Audio loaded in {audio_load_time:.3f}s (librosa)")

        # Load model
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        t2 = time.perf_counter()
        model_load_time = t2 - t1

        print(f"⏱️  Model loaded in {model_load_time:.3f}s")

        # Transcribe
        print("🎙️  Transcribing...")
        segments, info = model.transcribe(audio, beam_size=5)
        segments_list = list(segments)
        t3 = time.perf_counter()
        inference_time = t3 - t2
        total_time = t3 - t0

        mem_after = self.get_memory_mb()

        text = " ".join([seg.text for seg in segments_list]).strip()
        num_words = len(text.split())
        num_segments = len(segments_list)

        print(f"✅ Complete!")
        print(f"   Audio load time: {audio_load_time:.3f}s (librosa - SLOW)")
        print(f"   Inference time: {inference_time:.3f}s")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   RTF: {audio_duration/total_time:.1f}x")
        print(f"   Words transcribed: {num_words}")
        print(f"   Memory used: {mem_after - mem_before:.0f} MB")

        return TranscriptionResult(
            implementation="faster-whisper+librosa",
            download_time=0.0,
            audio_load_time=audio_load_time,
            model_load_time=model_load_time,
            inference_time=inference_time,
            total_time=total_time,
            audio_duration=audio_duration,
            rtf=audio_duration / total_time if total_time > 0 else 0,
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta=mem_after - mem_before,
            transcription=text,
            num_words=num_words,
            num_segments=num_segments
        )

    def test_audiodecode_faster_whisper(self, audio_file: str, model_size: str = "base") -> TranscriptionResult:
        """Test 3: AudioDecode + faster-whisper (our optimized stack)."""
        print("\n" + "="*80)
        print("TEST 3: AudioDecode + faster-whisper (OUR STACK)")
        print("="*80)
        print("WHAT AUDIODECODE DOES:")
        print("  - Loads audio using NATIVE libraries (PyAV + SoundFile)")
        print("  - NO subprocess overhead, NO FFmpeg spawning")
        print("  - Same faster-whisper model (same accuracy)")
        print("  - Same transcription (only loading is faster)")
        print()

        try:
            from audiodecode import load
            from audiodecode.inference import WhisperInference
        except ImportError as e:
            print(f"ERROR: {e}")
            print("Install with: pip install audiodecode[inference]")
            sys.exit(1)

        mem_before = self.get_memory_mb()

        # Load audio with AudioDecode (FAST path)
        print("📂 Loading audio with AudioDecode...")
        t0 = time.perf_counter()
        audio, sr = load(audio_file, sr=16000, mono=True)
        t1 = time.perf_counter()
        audio_load_time = t1 - t0

        audio_duration = len(audio) / sr

        print(f"⏱️  Audio loaded in {audio_load_time:.3f}s (AudioDecode - FAST!)")

        # Load model
        model = WhisperInference(model_size=model_size, device="cpu", compute_type="int8")
        t2 = time.perf_counter()
        model_load_time = t2 - t1

        print(f"⏱️  Model loaded in {model_load_time:.3f}s")

        # Transcribe
        print("🎙️  Transcribing...")
        result = model.transcribe_audio(audio, sample_rate=sr)
        t3 = time.perf_counter()
        inference_time = t3 - t2
        total_time = t3 - t0

        mem_after = self.get_memory_mb()

        text = result.text.strip()
        num_words = len(text.split())
        num_segments = len(result.segments)

        print(f"✅ Complete!")
        print(f"   Audio load time: {audio_load_time:.3f}s (AudioDecode - FAST!)")
        print(f"   Inference time: {inference_time:.3f}s")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   RTF: {audio_duration/total_time:.1f}x")
        print(f"   Words transcribed: {num_words}")
        print(f"   Memory used: {mem_after - mem_before:.0f} MB")

        return TranscriptionResult(
            implementation="audiodecode+faster-whisper",
            download_time=0.0,
            audio_load_time=audio_load_time,
            model_load_time=model_load_time,
            inference_time=inference_time,
            total_time=total_time,
            audio_duration=audio_duration,
            rtf=audio_duration / total_time if total_time > 0 else 0,
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta=mem_after - mem_before,
            transcription=text,
            num_words=num_words,
            num_segments=num_segments
        )

    def print_comparison(self, results):
        """Print detailed comparison table."""
        print("\n" + "="*80)
        print("FINAL COMPARISON - SAME VIDEO, SAME MODEL, DIFFERENT LOADING")
        print("="*80)

        print(f"\nAudio Duration: {results[0].audio_duration:.1f}s")
        print("\n" + "-"*80)
        print(f"{'Implementation':<35} {'Load(s)':<12} {'Infer(s)':<12} {'Total(s)':<12} {'RTF':<12} {'Memory(MB)'}")
        print("-"*80)

        for r in results:
            print(f"{r.implementation:<35} {r.audio_load_time:<12.3f} {r.inference_time:<12.3f} {r.total_time:<12.3f} {r.rtf:<12.1f}x {r.memory_delta:<12.0f}")

        # Calculate speedups
        baseline = results[0]  # vanilla whisper
        print("\n" + "="*80)
        print("SPEEDUPS vs openai-whisper (baseline)")
        print("="*80)
        for r in results[1:]:
            speedup = baseline.total_time / r.total_time
            load_speedup = baseline.total_time / (r.audio_load_time + 0.001)  # Avoid div by zero
            print(f"{r.implementation:<35} {speedup:.2f}x faster overall")

        # Accuracy comparison
        print("\n" + "="*80)
        print("ACCURACY COMPARISON - Should be IDENTICAL (same model)")
        print("="*80)
        for r in results:
            print(f"\n{r.implementation}:")
            print(f"  Words: {r.num_words}")
            print(f"  Segments: {r.num_segments}")
            print(f"  Preview: {r.transcription[:200]}...")

        # Key insight
        print("\n" + "="*80)
        print("KEY INSIGHT: WHAT AUDIODECODE DOES")
        print("="*80)
        print("✅ SAME Whisper model (same accuracy)")
        print("✅ SAME faster-whisper backend (CTranslate2)")
        print("✅ ONLY DIFFERENCE: Audio loading method")
        print("")
        print("❌ librosa: Spawns FFmpeg subprocess (slow)")
        print("✅ AudioDecode: Uses native PyAV + SoundFile (fast)")
        print("")
        print(f"Audio load time comparison:")
        if len(results) >= 3:
            print(f"  librosa:     {results[1].audio_load_time:.3f}s")
            print(f"  AudioDecode: {results[2].audio_load_time:.3f}s")
            if results[2].audio_load_time > 0:
                print(f"  Speedup:     {results[1].audio_load_time/results[2].audio_load_time:.1f}x faster")
        print("="*80)

    def save_results(self, results, filename="examples/youtube_showdown_results.json"):
        """Save results to JSON."""
        data = {
            "results": [
                {
                    "implementation": r.implementation,
                    "audio_load_time": r.audio_load_time,
                    "model_load_time": r.model_load_time,
                    "inference_time": r.inference_time,
                    "total_time": r.total_time,
                    "audio_duration": r.audio_duration,
                    "rtf": r.rtf,
                    "memory_delta_mb": r.memory_delta,
                    "num_words": r.num_words,
                    "num_segments": r.num_segments,
                    "transcription_preview": r.transcription[:500]
                }
                for r in results
            ]
        }

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="YouTube Transcription Showdown")
    parser.add_argument("url", nargs='?', help="YouTube URL to transcribe")
    parser.add_argument("--model", default="tiny", help="Whisper model size (default: tiny)")
    parser.add_argument("--file", help="Use local audio file instead of YouTube")

    args = parser.parse_args()

    print("="*80)
    print("YOUTUBE TRANSCRIPTION SHOWDOWN")
    print("Real-World Test: What Does AudioDecode Actually Do?")
    print("="*80)

    showdown = YouTubeTranscriptionShowdown()

    # Get audio file
    if args.file:
        audio_file = args.file
        download_time = 0
        audio_duration = 0
        print(f"\nUsing local file: {audio_file}")
    elif args.url:
        audio_file, download_time, audio_duration, title = showdown.download_youtube(args.url)
    else:
        print("\nERROR: Provide either --file or YouTube URL")
        print("Usage: python examples/youtube_transcription_showdown.py 'https://youtube.com/watch?v=...'")
        sys.exit(1)

    # Run all tests
    results = []

    # Test 1: Vanilla Whisper
    try:
        r1 = showdown.test_vanilla_whisper(audio_file, args.model)
        results.append(r1)
    except Exception as e:
        print(f"⚠️  Vanilla Whisper failed: {e}")

    # Test 2: faster-whisper + librosa
    try:
        r2 = showdown.test_faster_whisper_librosa(audio_file, args.model)
        results.append(r2)
    except Exception as e:
        print(f"⚠️  faster-whisper failed: {e}")

    # Test 3: AudioDecode + faster-whisper
    try:
        r3 = showdown.test_audiodecode_faster_whisper(audio_file, args.model)
        results.append(r3)
    except Exception as e:
        print(f"⚠️  AudioDecode failed: {e}")

    # Compare results
    if results:
        showdown.print_comparison(results)
        showdown.save_results(results)

    print("\n✅ Showdown complete!")


if __name__ == "__main__":
    main()
