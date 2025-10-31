"""
Real-World Comprehensive Benchmark
===================================

Tests AudioDecode vs librosa across realistic ML pipeline scenarios.
Runs on both macOS and Linux (Docker) for complete comparison.

Scenarios:
1. Cold start (first decode) - common in serverless/batch jobs
2. Warm processing (different files) - typical ML training
3. Cached processing (same file) - data augmentation loops
4. Mixed batch (various formats) - real dataset diversity
5. Batch processing - parallel decode of many files

Usage:
    # macOS
    python benchmark_real_world.py

    # Linux (Docker)
    docker run --rm -v $(pwd):/app audiodecode:linux-test python3 /app/benchmark_real_world.py
"""

import sys
import os
sys.path.insert(0, "src")

import time
import platform
from pathlib import Path
from typing import List, Dict, Any
import json

# Import both libraries
try:
    from audiodecode import AudioDecoder, clear_cache
    from audiodecode._rust import batch_decode
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

import librosa
import numpy as np


class RealWorldBenchmark:
    """Comprehensive real-world benchmark suite."""

    def __init__(self, fixtures_dir: str = "fixtures/audio"):
        self.fixtures_dir = Path(fixtures_dir)
        self.results = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor() if platform.system() == "Darwin" else "N/A",
            },
            "scenarios": {}
        }

    def get_test_files(self) -> Dict[str, List[Path]]:
        """Get diverse set of test files."""
        return {
            "mp3_small": [self.fixtures_dir / "wav_1s_mono_16000.mp3"],
            "mp3_medium": [self.fixtures_dir / "test_10s_mono_16000.mp3"] if (self.fixtures_dir / "test_10s_mono_16000.mp3").exists() else [],
            "wav_small": [self.fixtures_dir / "wav_1s_mono_16000.wav"],
            "wav_large": [self.fixtures_dir / "test_10s_mono_16000.wav"] if (self.fixtures_dir / "test_10s_mono_16000.wav").exists() else [],
            "flac": [self.fixtures_dir / "wav_1s_mono_8000.flac"],
            "stereo": [self.fixtures_dir / "wav_1s_stereo_44100.wav"],
            "mixed": [
                self.fixtures_dir / "wav_1s_mono_16000.mp3",
                self.fixtures_dir / "wav_1s_mono_16000.wav",
                self.fixtures_dir / "wav_1s_mono_8000.flac",
                self.fixtures_dir / "wav_1s_stereo_44100.wav",
            ]
        }

    def scenario_1_cold_start(self):
        """Scenario 1: Cold start - first time decoding (serverless, batch jobs)."""
        print("\n" + "="*70)
        print("SCENARIO 1: Cold Start (Serverless/Batch Jobs)")
        print("="*70)
        print("Simulates: Lambda function, batch job startup, first decode")

        test_files = self.get_test_files()
        mp3_file = str(test_files["mp3_small"][0])

        # Simulate cold start - clear everything
        clear_cache()

        # librosa cold
        print("\n1. librosa (cold start):")
        start = time.perf_counter()
        audio_lr, sr = librosa.load(mp3_file, sr=None)
        lr_time = time.perf_counter() - start
        print(f"   Time: {lr_time*1000:.2f}ms")
        print(f"   Shape: {audio_lr.shape}")

        # AudioDecode cold
        print("\n2. AudioDecode (cold start):")
        start = time.perf_counter()
        audio_ad = AudioDecoder(mp3_file).decode()
        ad_time = time.perf_counter() - start
        print(f"   Time: {ad_time*1000:.2f}ms")
        print(f"   Shape: {audio_ad.shape}")

        speedup = lr_time / ad_time
        print(f"\n   🚀 Speedup: {speedup:.1f}x")

        self.results["scenarios"]["cold_start"] = {
            "description": "First decode (cold start)",
            "librosa_ms": lr_time * 1000,
            "audiodecode_ms": ad_time * 1000,
            "speedup": speedup,
            "winner": "AudioDecode" if speedup > 1 else "librosa"
        }

    def scenario_2_warm_different_files(self):
        """Scenario 2: Processing different files (typical ML training)."""
        print("\n" + "="*70)
        print("SCENARIO 2: Warm Different Files (ML Training)")
        print("="*70)
        print("Simulates: Training loop processing diverse dataset")

        test_files = self.get_test_files()
        files = test_files["mixed"][:20] * 5  # 20 unique files, 5 times each = 100 total

        # Warm up both
        librosa.load(str(files[0]), sr=None)
        AudioDecoder(str(files[0])).decode()

        # librosa
        print("\n1. librosa (100 different files):")
        start = time.perf_counter()
        for f in files:
            audio, sr = librosa.load(str(f), sr=None)
        lr_time = time.perf_counter() - start
        print(f"   Total: {lr_time:.3f}s")
        print(f"   Per file: {lr_time/len(files)*1000:.2f}ms")

        # AudioDecode (clear cache to force decode)
        print("\n2. AudioDecode (100 different files):")
        clear_cache()
        start = time.perf_counter()
        for f in files:
            audio = AudioDecoder(str(f)).decode()
        ad_time = time.perf_counter() - start
        print(f"   Total: {ad_time:.3f}s")
        print(f"   Per file: {ad_time/len(files)*1000:.2f}ms")

        speedup = lr_time / ad_time
        print(f"\n   🚀 Speedup: {speedup:.1f}x")

        self.results["scenarios"]["warm_different_files"] = {
            "description": "Processing 100 different files (warm)",
            "num_files": len(files),
            "librosa_total_s": lr_time,
            "audiodecode_total_s": ad_time,
            "librosa_per_file_ms": lr_time / len(files) * 1000,
            "audiodecode_per_file_ms": ad_time / len(files) * 1000,
            "speedup": speedup,
            "winner": "AudioDecode" if speedup > 1 else "librosa"
        }

    def scenario_3_cached_same_file(self):
        """Scenario 3: Repeated access to same file (data augmentation)."""
        print("\n" + "="*70)
        print("SCENARIO 3: Cached Same File (Data Augmentation)")
        print("="*70)
        print("Simulates: Augmentation loop accessing same file repeatedly")

        test_files = self.get_test_files()
        mp3_file = str(test_files["mp3_small"][0])
        iterations = 100

        # librosa
        print(f"\n1. librosa ({iterations} iterations, same file):")
        librosa.load(mp3_file, sr=None)  # Warm up
        start = time.perf_counter()
        for _ in range(iterations):
            audio, sr = librosa.load(mp3_file, sr=None)
        lr_time = time.perf_counter() - start
        print(f"   Total: {lr_time:.3f}s")
        print(f"   Per decode: {lr_time/iterations*1000:.3f}ms")

        # AudioDecode with cache
        print(f"\n2. AudioDecode WITH cache ({iterations} iterations):")
        clear_cache()
        AudioDecoder(mp3_file).decode()  # Warm up
        start = time.perf_counter()
        for _ in range(iterations):
            audio = AudioDecoder(mp3_file).decode(use_cache=True)
        ad_cached_time = time.perf_counter() - start
        print(f"   Total: {ad_cached_time:.3f}s")
        print(f"   Per decode: {ad_cached_time/iterations*1000:.3f}ms")

        # AudioDecode without cache (for comparison)
        print(f"\n3. AudioDecode NO cache ({iterations} iterations):")
        start = time.perf_counter()
        for _ in range(iterations):
            audio = AudioDecoder(mp3_file).decode(use_cache=False)
        ad_nocache_time = time.perf_counter() - start
        print(f"   Total: {ad_nocache_time:.3f}s")
        print(f"   Per decode: {ad_nocache_time/iterations*1000:.3f}ms")

        speedup_cached = lr_time / ad_cached_time
        speedup_nocache = lr_time / ad_nocache_time
        print(f"\n   🚀 AudioDecode (cached) vs librosa: {speedup_cached:.1f}x")
        print(f"   🚀 AudioDecode (no cache) vs librosa: {speedup_nocache:.1f}x")
        print(f"   💾 Cache benefit: {ad_nocache_time/ad_cached_time:.1f}x")

        self.results["scenarios"]["cached_same_file"] = {
            "description": f"Repeated decode of same file ({iterations}x)",
            "iterations": iterations,
            "librosa_per_decode_ms": lr_time / iterations * 1000,
            "audiodecode_cached_ms": ad_cached_time / iterations * 1000,
            "audiodecode_nocache_ms": ad_nocache_time / iterations * 1000,
            "speedup_vs_librosa_cached": speedup_cached,
            "speedup_vs_librosa_nocache": speedup_nocache,
            "cache_benefit": ad_nocache_time / ad_cached_time,
            "winner": "AudioDecode (cached)" if speedup_cached > 1 else "librosa"
        }

    def scenario_4_mixed_formats(self):
        """Scenario 4: Mixed format batch (real dataset)."""
        print("\n" + "="*70)
        print("SCENARIO 4: Mixed Format Batch (Real Dataset)")
        print("="*70)
        print("Simulates: Processing real-world dataset with various formats")

        test_files = self.get_test_files()
        mixed_files = test_files["mixed"] * 25  # 100 files, mixed formats

        # Warm up
        librosa.load(str(mixed_files[0]), sr=None)
        AudioDecoder(str(mixed_files[0])).decode()

        # librosa
        print(f"\n1. librosa ({len(mixed_files)} mixed format files):")
        start = time.perf_counter()
        for f in mixed_files:
            audio, sr = librosa.load(str(f), sr=None)
        lr_time = time.perf_counter() - start
        print(f"   Total: {lr_time:.3f}s")
        print(f"   Per file: {lr_time/len(mixed_files)*1000:.2f}ms")

        # AudioDecode
        print(f"\n2. AudioDecode ({len(mixed_files)} mixed format files):")
        clear_cache()
        start = time.perf_counter()
        for f in mixed_files:
            audio = AudioDecoder(str(f)).decode()
        ad_time = time.perf_counter() - start
        print(f"   Total: {ad_time:.3f}s")
        print(f"   Per file: {ad_time/len(mixed_files)*1000:.2f}ms")

        speedup = lr_time / ad_time
        print(f"\n   🚀 Speedup: {speedup:.1f}x")

        # Show format breakdown
        format_counts = {}
        for f in mixed_files:
            ext = Path(f).suffix
            format_counts[ext] = format_counts.get(ext, 0) + 1

        print("\n   Format distribution:")
        for ext, count in sorted(format_counts.items()):
            print(f"     {ext}: {count} files")

        self.results["scenarios"]["mixed_formats"] = {
            "description": "Mixed format batch processing",
            "num_files": len(mixed_files),
            "format_distribution": format_counts,
            "librosa_total_s": lr_time,
            "audiodecode_total_s": ad_time,
            "librosa_per_file_ms": lr_time / len(mixed_files) * 1000,
            "audiodecode_per_file_ms": ad_time / len(mixed_files) * 1000,
            "speedup": speedup,
            "winner": "AudioDecode" if speedup > 1 else "librosa"
        }

    def scenario_5_batch_processing(self):
        """Scenario 5: Parallel batch processing with Rust."""
        if not HAS_RUST:
            print("\n" + "="*70)
            print("SCENARIO 5: Batch Processing (SKIPPED - Rust not available)")
            print("="*70)
            return

        print("\n" + "="*70)
        print("SCENARIO 5: Parallel Batch Processing (Rust)")
        print("="*70)
        print("Simulates: High-throughput batch processing")

        test_files = self.get_test_files()
        files = [str(test_files["mp3_small"][0])] * 100

        # Serial AudioDecode
        print(f"\n1. AudioDecode Serial ({len(files)} files):")
        clear_cache()
        start = time.perf_counter()
        serial_audios = [AudioDecoder(f).decode(use_cache=False) for f in files]
        serial_time = time.perf_counter() - start
        print(f"   Total: {serial_time:.3f}s")
        print(f"   Per file: {serial_time/len(files)*1000:.2f}ms")

        # Rust parallel (2 workers)
        print(f"\n2. Rust Parallel - 2 workers ({len(files)} files):")
        start = time.perf_counter()
        parallel_2 = batch_decode(files, num_workers=2)
        parallel_2_time = time.perf_counter() - start
        print(f"   Total: {parallel_2_time:.3f}s")
        print(f"   Per file: {parallel_2_time/len(files)*1000:.2f}ms")
        print(f"   🚀 Speedup: {serial_time/parallel_2_time:.1f}x")

        # Rust parallel (4 workers)
        print(f"\n3. Rust Parallel - 4 workers ({len(files)} files):")
        start = time.perf_counter()
        parallel_4 = batch_decode(files, num_workers=4)
        parallel_4_time = time.perf_counter() - start
        print(f"   Total: {parallel_4_time:.3f}s")
        print(f"   Per file: {parallel_4_time/len(files)*1000:.2f}ms")
        print(f"   🚀 Speedup: {serial_time/parallel_4_time:.1f}x")

        # Rust parallel (8 workers)
        print(f"\n4. Rust Parallel - 8 workers ({len(files)} files):")
        start = time.perf_counter()
        parallel_8 = batch_decode(files, num_workers=8)
        parallel_8_time = time.perf_counter() - start
        print(f"   Total: {parallel_8_time:.3f}s")
        print(f"   Per file: {parallel_8_time/len(files)*1000:.2f}ms")
        print(f"   🚀 Speedup: {serial_time/parallel_8_time:.1f}x")

        self.results["scenarios"]["batch_processing"] = {
            "description": "Parallel batch decode with Rust",
            "num_files": len(files),
            "serial_total_s": serial_time,
            "parallel_2_total_s": parallel_2_time,
            "parallel_4_total_s": parallel_4_time,
            "parallel_8_total_s": parallel_8_time,
            "speedup_2_workers": serial_time / parallel_2_time,
            "speedup_4_workers": serial_time / parallel_4_time,
            "speedup_8_workers": serial_time / parallel_8_time,
            "best_speedup": serial_time / min(parallel_2_time, parallel_4_time, parallel_8_time),
        }

    def print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "="*70)
        print("COMPREHENSIVE SUMMARY")
        print("="*70)

        print(f"\nPlatform: {self.results['platform']['system']} {self.results['platform']['release']}")
        print(f"Machine: {self.results['platform']['machine']}")

        print("\n" + "-"*70)
        print("SCENARIO RESULTS")
        print("-"*70)

        for scenario_name, data in self.results["scenarios"].items():
            print(f"\n{scenario_name.upper().replace('_', ' ')}:")
            print(f"  {data['description']}")
            print(f"  Winner: {data.get('winner', 'N/A')}")

            if 'speedup' in data:
                print(f"  Speedup: {data['speedup']:.1f}x")

        # Overall winner
        print("\n" + "="*70)
        wins = {"AudioDecode": 0, "librosa": 0}
        for data in self.results["scenarios"].values():
            winner = data.get("winner", "")
            if "AudioDecode" in winner:
                wins["AudioDecode"] += 1
            elif "librosa" in winner:
                wins["librosa"] += 1

        print("OVERALL WINNER:")
        print(f"  AudioDecode wins: {wins['AudioDecode']}/{len(self.results['scenarios'])} scenarios")
        print(f"  librosa wins: {wins['librosa']}/{len(self.results['scenarios'])} scenarios")

        if wins["AudioDecode"] > wins["librosa"]:
            print("\n  🏆 AudioDecode is the overall winner!")
        elif wins["librosa"] > wins["AudioDecode"]:
            print("\n  🏆 librosa is the overall winner!")
        else:
            print("\n  🤝 It's a tie!")

        print("="*70)

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON."""
        output_file = f"benchmark_results_{self.results['platform']['system'].lower()}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📊 Results saved to: {output_file}")

    def run_all(self):
        """Run all benchmark scenarios."""
        print("="*70)
        print("REAL-WORLD COMPREHENSIVE BENCHMARK")
        print("="*70)
        print(f"Platform: {self.results['platform']['system']} {self.results['platform']['machine']}")
        print("="*70)

        self.scenario_1_cold_start()
        self.scenario_2_warm_different_files()
        self.scenario_3_cached_same_file()
        self.scenario_4_mixed_formats()
        self.scenario_5_batch_processing()

        self.print_summary()
        self.save_results()


if __name__ == "__main__":
    benchmark = RealWorldBenchmark()
    benchmark.run_all()
