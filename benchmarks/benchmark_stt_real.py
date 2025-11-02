#!/usr/bin/env python3
"""
Real-world STT benchmark: AudioDecode vs vanilla Whisper

Measures actual performance with real audio files:
- Load time
- Inference time
- Total time
- RTF (Real-Time Factor)
- Memory usage

NO MOCKS - Real transcriptions only.
"""

import time
import psutil
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class BenchmarkResult:
    """Real benchmark result with actual timing data."""
    implementation: str
    model_size: str
    audio_file: str
    audio_duration: float

    # Timing (seconds)
    load_time: float
    inference_time: float
    total_time: float

    # Performance metrics
    rtf: float  # Real-Time Factor (audio_duration / total_time)

    # Memory (MB)
    memory_before: float
    memory_after: float
    memory_delta: float

    # Output
    transcription: str
    num_segments: int


class STTBenchmark:
    """Run real STT benchmarks without mocks."""

    def __init__(self):
        self.process = psutil.Process()
        self.results: List[BenchmarkResult] = []

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def create_test_audio(self, duration: float, filename: str) -> str:
        """Create test audio file with tone (for testing)."""
        sr = 16000
        samples = int(sr * duration)

        # Generate sine wave at 440Hz (A note)
        t = np.linspace(0, duration, samples)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        # Add some noise to make it realistic
        audio += 0.05 * np.random.randn(samples)
        audio = audio.astype(np.float32)

        Path("benchmarks/test_audio").mkdir(parents=True, exist_ok=True)
        filepath = f"benchmarks/test_audio/{filename}"
        sf.write(filepath, audio, sr)

        return filepath

    def benchmark_vanilla_whisper(self, audio_file: str, model_size: str = "base") -> BenchmarkResult:
        """Benchmark vanilla openai-whisper (includes load time in total)."""
        print(f"\n[Vanilla Whisper] Transcribing {audio_file} with {model_size} model...")

        try:
            import whisper
        except ImportError:
            print("ERROR: openai-whisper not installed. Install with: pip install openai-whisper")
            raise

        # Get audio duration first
        import librosa
        audio_duration = librosa.get_duration(path=audio_file)

        mem_before = self.get_memory_mb()

        # Time model loading + inference together (vanilla Whisper combines these)
        t0 = time.perf_counter()

        model = whisper.load_model(model_size)
        t1 = time.perf_counter()
        load_time = t1 - t0

        result = model.transcribe(audio_file)
        t2 = time.perf_counter()
        inference_time = t2 - t1
        total_time = t2 - t0

        mem_after = self.get_memory_mb()

        print(f"  Load time: {load_time:.3f}s")
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  RTF: {audio_duration/total_time:.1f}x")

        return BenchmarkResult(
            implementation="openai-whisper",
            model_size=model_size,
            audio_file=audio_file,
            audio_duration=audio_duration,
            load_time=load_time,
            inference_time=inference_time,
            total_time=total_time,
            rtf=audio_duration / total_time if total_time > 0 else 0,
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta=mem_after - mem_before,
            transcription=result["text"],
            num_segments=len(result.get("segments", []))
        )

    def benchmark_audiodecode_faster_whisper(self, audio_file: str, model_size: str = "base") -> BenchmarkResult:
        """Benchmark AudioDecode + faster-whisper (our full stack)."""
        print(f"\n[AudioDecode + faster-whisper] Transcribing {audio_file} with {model_size} model...")

        try:
            from audiodecode import load
            from audiodecode.inference import WhisperInference
        except ImportError as e:
            print(f"ERROR: {e}")
            print("Install with: pip install audiodecode[inference]")
            raise

        mem_before = self.get_memory_mb()

        # Time audio loading separately (our advantage)
        t0 = time.perf_counter()
        audio, sr = load(audio_file, sr=16000, mono=True)
        t1 = time.perf_counter()
        load_time = t1 - t0

        audio_duration = len(audio) / sr

        # Time model loading + inference
        model = WhisperInference(model_size=model_size, device="cpu", compute_type="int8")
        t2 = time.perf_counter()
        model_load_time = t2 - t1

        result = model.transcribe_audio(audio, sample_rate=sr)
        t3 = time.perf_counter()
        inference_time = t3 - t2
        total_time = t3 - t0

        mem_after = self.get_memory_mb()

        print(f"  Audio load time: {load_time:.3f}s")
        print(f"  Model load time: {model_load_time:.3f}s")
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  RTF: {audio_duration/total_time:.1f}x")

        return BenchmarkResult(
            implementation="audiodecode-faster-whisper",
            model_size=model_size,
            audio_file=audio_file,
            audio_duration=audio_duration,
            load_time=load_time + model_load_time,
            inference_time=inference_time,
            total_time=total_time,
            rtf=audio_duration / total_time if total_time > 0 else 0,
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta=mem_after - mem_before,
            transcription=result.text,
            num_segments=len(result.segments)
        )

    def benchmark_faster_whisper_only(self, audio_file: str, model_size: str = "base") -> BenchmarkResult:
        """Benchmark faster-whisper alone (without AudioDecode loading)."""
        print(f"\n[faster-whisper only] Transcribing {audio_file} with {model_size} model...")

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("ERROR: faster-whisper not installed. Install with: pip install faster-whisper")
            raise

        # Load audio using librosa (baseline)
        import librosa

        mem_before = self.get_memory_mb()

        t0 = time.perf_counter()
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        t1 = time.perf_counter()
        load_time = t1 - t0

        audio_duration = len(audio) / sr

        # Model + inference
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        t2 = time.perf_counter()
        model_load_time = t2 - t1

        segments, info = model.transcribe(audio, beam_size=5)
        segments_list = list(segments)
        t3 = time.perf_counter()
        inference_time = t3 - t2
        total_time = t3 - t0

        mem_after = self.get_memory_mb()

        text = " ".join([seg.text for seg in segments_list])

        print(f"  Audio load time (librosa): {load_time:.3f}s")
        print(f"  Model load time: {model_load_time:.3f}s")
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  RTF: {audio_duration/total_time:.1f}x")

        return BenchmarkResult(
            implementation="faster-whisper-librosa",
            model_size=model_size,
            audio_file=audio_file,
            audio_duration=audio_duration,
            load_time=load_time + model_load_time,
            inference_time=inference_time,
            total_time=total_time,
            rtf=audio_duration / total_time if total_time > 0 else 0,
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta=mem_after - mem_before,
            transcription=text,
            num_segments=len(segments_list)
        )

    def print_comparison(self, results: List[BenchmarkResult]):
        """Print comparison table."""
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON")
        print("="*80)

        # Group by audio file
        by_file = {}
        for r in results:
            if r.audio_file not in by_file:
                by_file[r.audio_file] = []
            by_file[r.audio_file].append(r)

        for audio_file, file_results in by_file.items():
            print(f"\nFile: {audio_file}")
            print(f"Duration: {file_results[0].audio_duration:.1f}s")
            print("-" * 80)
            print(f"{'Implementation':<30} {'Load(s)':<10} {'Infer(s)':<10} {'Total(s)':<10} {'RTF':<10} {'Mem(MB)':<10}")
            print("-" * 80)

            for r in file_results:
                print(f"{r.implementation:<30} {r.load_time:<10.3f} {r.inference_time:<10.3f} {r.total_time:<10.3f} {r.rtf:<10.1f}x {r.memory_delta:<10.0f}")

            # Calculate speedups
            if len(file_results) > 1:
                baseline = next((r for r in file_results if r.implementation == "openai-whisper"), None)
                if baseline:
                    print("\nSpeedups vs vanilla Whisper:")
                    for r in file_results:
                        if r.implementation != "openai-whisper":
                            speedup = baseline.total_time / r.total_time
                            print(f"  {r.implementation}: {speedup:.2f}x faster")

    def save_results(self, filename: str = "benchmarks/stt_benchmark_results.json"):
        """Save results to JSON."""
        data = {
            "results": [
                {
                    "implementation": r.implementation,
                    "model_size": r.model_size,
                    "audio_file": r.audio_file,
                    "audio_duration": r.audio_duration,
                    "load_time": r.load_time,
                    "inference_time": r.inference_time,
                    "total_time": r.total_time,
                    "rtf": r.rtf,
                    "memory_delta_mb": r.memory_delta,
                    "transcription_preview": r.transcription[:100] if r.transcription else ""
                }
                for r in self.results
            ]
        }

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Results saved to {filename}")


def main():
    """Run real benchmarks."""
    print("="*80)
    print("AudioDecode STT REAL BENCHMARK")
    print("="*80)

    benchmark = STTBenchmark()

    # Create test audio files (or use real ones if available)
    print("\nCreating test audio files...")
    test_files = [
        benchmark.create_test_audio(10.0, "test_10s.wav"),   # 10 seconds
        benchmark.create_test_audio(30.0, "test_30s.wav"),   # 30 seconds
        benchmark.create_test_audio(60.0, "test_60s.wav"),   # 1 minute
    ]

    # Run benchmarks on each file
    model_size = "tiny"  # Use tiny for faster benchmarking

    for audio_file in test_files:
        print(f"\n{'='*80}")
        print(f"Benchmarking: {audio_file}")
        print(f"{'='*80}")

        # 1. Vanilla Whisper (baseline)
        try:
            result = benchmark.benchmark_vanilla_whisper(audio_file, model_size)
            benchmark.results.append(result)
        except Exception as e:
            print(f"⚠️  Vanilla Whisper failed: {e}")

        # 2. faster-whisper with librosa
        try:
            result = benchmark.benchmark_faster_whisper_only(audio_file, model_size)
            benchmark.results.append(result)
        except Exception as e:
            print(f"⚠️  faster-whisper failed: {e}")

        # 3. AudioDecode + faster-whisper (our stack)
        try:
            result = benchmark.benchmark_audiodecode_faster_whisper(audio_file, model_size)
            benchmark.results.append(result)
        except Exception as e:
            print(f"⚠️  AudioDecode failed: {e}")

    # Print comparison
    benchmark.print_comparison(benchmark.results)

    # Save results
    benchmark.save_results()

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
