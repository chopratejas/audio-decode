#!/usr/bin/env python3
"""
A10G GPU Benchmark - CTranslate2 Direct (avoiding PyTorch cuDNN issues)
"""

import time
import sys
import os
from pathlib import Path

# Prevent PyTorch from loading (causes cuDNN issues)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def benchmark_audiodecode(audio_file: str, device: str, model_size: str = "base"):
    """Benchmark AudioDecode on specific device."""
    print(f"\n{'='*80}")
    print(f"  AudioDecode Benchmark - {device.upper()}")
    print(f"{'='*80}")

    # Load model
    print(f"\n📦 Loading Whisper model...")
    print(f"   Model: {model_size}")
    print(f"   Device: {device}")

    start_load = time.time()

    try:
        from audiodecode import WhisperInference

        whisper = WhisperInference(
            model_size=model_size,
            device=device,
        )
        load_time = time.time() - start_load

        print(f"   ✓ Model loaded in {load_time:.2f}s")
        print(f"   Compute type: {whisper.compute_type}")
        print(f"   Batch size: {whisper.batch_size}")

    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Transcribe
    print(f"\n🎤 Transcribing {Path(audio_file).name}...")
    start_transcribe = time.time()

    try:
        result = whisper.transcribe_file(audio_file, word_timestamps=True)
        transcribe_time = time.time() - start_transcribe

        rtf = result.duration / transcribe_time if transcribe_time > 0 else 0
        words = len(result.text.split())

        # Count word timestamps
        word_timestamps = sum(len(seg.words) for seg in result.segments if seg.words)

        print(f"   ✓ Transcription complete!")
        print(f"   Audio duration: {result.duration:.1f}s ({result.duration/60:.1f} min)")
        print(f"   Transcribe time: {transcribe_time:.2f}s")
        print(f"   Total time: {load_time + transcribe_time:.2f}s")
        print(f"   RTF: {rtf:.1f}x realtime")
        print(f"   Words: {words}")
        print(f"   Segments: {len(result.segments)}")
        print(f"   Word timestamps: {word_timestamps}")

        return {
            'device': device,
            'load_time': load_time,
            'transcribe_time': transcribe_time,
            'total_time': load_time + transcribe_time,
            'rtf': rtf,
            'duration': result.duration,
            'words': words,
            'segments': len(result.segments),
            'word_timestamps': word_timestamps,
            'text_preview': result.text[:300] + "..." if len(result.text) > 300 else result.text
        }

    except Exception as e:
        print(f"   ✗ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run A10G GPU vs CPU benchmark."""
    print("="*80)
    print("  AudioDecode A10G GPU Benchmark")
    print("  (CTranslate2 backend - no PyTorch dependencies)")
    print("="*80)

    # Check CTranslate2 GPU
    try:
        import ctranslate2
        cuda_devices = ctranslate2.get_cuda_device_count()
        print(f"\n✓ CTranslate2 version: {ctranslate2.__version__}")
        print(f"✓ CUDA devices available: {cuda_devices}")
        has_gpu = cuda_devices > 0
    except Exception as e:
        print(f"\n✗ CTranslate2 GPU check failed: {e}")
        has_gpu = False

    # Get audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "audio.mp3"

    if not Path(audio_file).exists():
        print(f"\n❌ Audio file not found: {audio_file}")
        print("Usage: python benchmark_ctranslate2_only.py <audio_file>")
        sys.exit(1)

    print(f"\n📁 Audio file: {audio_file}")
    print(f"   Size: {Path(audio_file).stat().st_size / 1024 / 1024:.1f} MB")

    # Run CPU benchmark
    print("\n" + "="*80)
    print("  🖥️  CPU BENCHMARK")
    print("="*80)

    cpu_result = benchmark_audiodecode(audio_file, "cpu")

    if cpu_result is None:
        print("\n❌ CPU benchmark failed!")
        sys.exit(1)

    # Run GPU benchmark if available
    if has_gpu:
        print("\n" + "="*80)
        print("  🚀 GPU (A10G) BENCHMARK")
        print("="*80)

        gpu_result = benchmark_audiodecode(audio_file, "cuda")

        if gpu_result is None:
            print("\n⚠️  GPU benchmark failed")
            print("CPU-only results are available above")
            sys.exit(1)

        # Detailed comparison
        print("\n" + "="*80)
        print("  📊 PERFORMANCE COMPARISON: A10G GPU vs CPU")
        print("="*80)

        speedup_load = cpu_result['load_time'] / gpu_result['load_time']
        speedup_transcribe = cpu_result['transcribe_time'] / gpu_result['transcribe_time']
        speedup_total = cpu_result['total_time'] / gpu_result['total_time']
        rtf_improvement = gpu_result['rtf'] / cpu_result['rtf']

        print(f"\n{'Metric':<30} {'CPU':<20} {'GPU (A10G)':<20} {'Speedup':<15}")
        print("-"*85)
        print(f"{'Model Load Time':<30} {cpu_result['load_time']:.2f}s{'':<15} "
              f"{gpu_result['load_time']:.2f}s{'':<15} "
              f"{speedup_load:.2f}x")
        print(f"{'Transcription Time':<30} {cpu_result['transcribe_time']:.2f}s{'':<15} "
              f"{gpu_result['transcribe_time']:.2f}s{'':<15} "
              f"**{speedup_transcribe:.2f}x**")
        print(f"{'Total Pipeline Time':<30} {cpu_result['total_time']:.2f}s{'':<15} "
              f"{gpu_result['total_time']:.2f}s{'':<15} "
              f"**{speedup_total:.2f}x**")
        print(f"{'RTF (realtime factor)':<30} {cpu_result['rtf']:.1f}x{'':<16} "
              f"{gpu_result['rtf']:.1f}x{'':<16} "
              f"{rtf_improvement:.2f}x better")

        print(f"\n{'Quality Metrics':<30} {'Value':<20}")
        print("-"*50)
        print(f"{'Audio Duration':<30} {cpu_result['duration']:.1f}s ({cpu_result['duration']/60:.1f} min)")
        print(f"{'Words Transcribed':<30} {cpu_result['words']}")
        print(f"{'Segments':<30} {cpu_result['segments']}")
        print(f"{'Word Timestamps':<30} {cpu_result['word_timestamps']}")

        print("\n" + "="*80)
        print("  🎉 KEY RESULTS")
        print("="*80)
        print(f"\n  ✓ A10G GPU is {speedup_transcribe:.2f}x FASTER than CPU")
        print(f"  ✓ GPU achieves {gpu_result['rtf']:.1f}x realtime factor")
        print(f"  ✓ Total pipeline speedup: {speedup_total:.2f}x")

        # Estimates vs OpenAI Whisper
        print(f"\n  📈 Estimated Performance vs OpenAI Whisper:")

        # From documentation: AudioDecode CPU is ~6x faster on Linux
        cpu_vs_openai = 6.0
        gpu_vs_openai = cpu_vs_openai * speedup_transcribe

        print(f"     • AudioDecode CPU: ~{cpu_vs_openai:.1f}x faster than OpenAI Whisper (from docs)")
        print(f"     • AudioDecode GPU: ~{gpu_vs_openai:.1f}x faster than OpenAI Whisper (extrapolated)")
        print(f"     • OpenAI Whisper estimated time: ~{cpu_result['transcribe_time'] * cpu_vs_openai:.1f}s")

        # Save comprehensive results
        print(f"\n  💾 Saving results to A10G_BENCHMARK_RESULTS.md...")

        with open("A10G_BENCHMARK_RESULTS.md", "w") as f:
            f.write("# AudioDecode A10G GPU Benchmark Results\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Platform:** NVIDIA A10G GPU\n")
            f.write(f"**Audio File:** {audio_file}\n")
            f.write(f"**Audio Duration:** {cpu_result['duration']:.1f}s ({cpu_result['duration']/60:.1f} minutes)\n")
            f.write(f"**Model:** Whisper base\n\n")

            f.write("## Performance Comparison\n\n")
            f.write("| Metric | CPU | GPU (A10G) | Speedup |\n")
            f.write("|--------|-----|------------|----------|\n")
            f.write(f"| Model Load | {cpu_result['load_time']:.2f}s | {gpu_result['load_time']:.2f}s | **{speedup_load:.2f}x** |\n")
            f.write(f"| Transcription | {cpu_result['transcribe_time']:.2f}s | {gpu_result['transcribe_time']:.2f}s | **{speedup_transcribe:.2f}x** |\n")
            f.write(f"| Total Pipeline | {cpu_result['total_time']:.2f}s | {gpu_result['total_time']:.2f}s | **{speedup_total:.2f}x** |\n")
            f.write(f"| RTF (realtime) | {cpu_result['rtf']:.1f}x | {gpu_result['rtf']:.1f}x | **{rtf_improvement:.2f}x** |\n\n")

            f.write("## Quality Metrics\n\n")
            f.write(f"- **Words:** {cpu_result['words']}\n")
            f.write(f"- **Segments:** {cpu_result['segments']}\n")
            f.write(f"- **Word Timestamps:** {cpu_result['word_timestamps']}\n\n")

            f.write("## Key Findings\n\n")
            f.write(f"1. **A10G GPU is {speedup_transcribe:.2f}x faster** than CPU for transcription\n")
            f.write(f"2. **GPU RTF:** {gpu_result['rtf']:.1f}x realtime (process {gpu_result['rtf']:.1f} seconds of audio per second)\n")
            f.write(f"3. **Estimated vs OpenAI Whisper:** ~{gpu_vs_openai:.1f}x faster on GPU\n\n")

            f.write("## Text Preview\n\n")
            f.write(f"```\n{cpu_result['text_preview']}\n```\n")

        print(f"     ✓ Results saved!")

    else:
        print("\n⚠️  No GPU available - CPU results only")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
