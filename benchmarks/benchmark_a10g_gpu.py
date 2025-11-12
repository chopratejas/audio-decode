#!/usr/bin/env python3
"""
A10G GPU Benchmark for AudioDecode
Tests AudioDecode with faster-whisper/CTranslate2 backend
"""

import time
import sys
from pathlib import Path


def check_gpu():
    """Check if GPU is available for CTranslate2."""
    try:
        import ctranslate2
        cuda_available = ctranslate2.get_cuda_device_count() > 0
        if cuda_available:
            print(f"✓ CTranslate2 CUDA devices: {ctranslate2.get_cuda_device_count()}")
            return True
        else:
            print("✗ No CUDA devices available for CTranslate2")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def benchmark_audiodecode(audio_file: str, device: str, model_size: str = "base"):
    """Benchmark AudioDecode on specific device."""
    from audiodecode import WhisperInference

    print(f"\n{'='*80}")
    print(f"  AudioDecode Benchmark - {device.upper()}")
    print(f"{'='*80}")

    # Load model
    print(f"\n📦 Loading model (size={model_size}, device={device})...")
    start_load = time.time()

    try:
        whisper = WhisperInference(
            model_size=model_size,
            device=device,
        )
        load_time = time.time() - start_load

        print(f"   ✓ Model loaded in {load_time:.2f}s")
        print(f"   Device: {whisper.device}")
        print(f"   Compute type: {whisper.compute_type}")
        print(f"   Batch size: {whisper.batch_size}")

    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return None

    # Transcribe
    print(f"\n🎤 Transcribing {audio_file}...")
    start_transcribe = time.time()

    try:
        result = whisper.transcribe_file(audio_file, word_timestamps=True)
        transcribe_time = time.time() - start_transcribe

        rtf = result.duration / transcribe_time if transcribe_time > 0 else 0
        words = len(result.text.split())

        # Count word timestamps
        word_timestamps = 0
        for seg in result.segments:
            if seg.words:
                word_timestamps += len(seg.words)

        print(f"   ✓ Transcribed successfully")
        print(f"   Audio duration: {result.duration:.1f}s")
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
            'text': result.text[:200] + "..." if len(result.text) > 200 else result.text
        }

    except Exception as e:
        print(f"   ✗ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run comprehensive A10G GPU benchmark."""
    print("="*80)
    print("  AudioDecode A10G GPU Benchmark")
    print("="*80)

    # Check GPU
    has_gpu = check_gpu()

    # Get audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Use the longest test file
        audio_file = "fixtures/audio/wav_60s_mono_16000.wav"

    if not Path(audio_file).exists():
        print(f"\n❌ Audio file not found: {audio_file}")
        print("Usage: python benchmark_a10g_gpu.py <audio_file>")
        sys.exit(1)

    print(f"\n📁 Audio file: {audio_file}")

    # Benchmark CPU
    print("\n" + "🖥️  CPU BENCHMARK".center(80, "="))
    cpu_result = benchmark_audiodecode(audio_file, "cpu")

    if cpu_result is None:
        print("\n❌ CPU benchmark failed!")
        sys.exit(1)

    # Benchmark GPU if available
    if has_gpu:
        print("\n" + "🚀 GPU BENCHMARK".center(80, "="))
        gpu_result = benchmark_audiodecode(audio_file, "cuda")

        if gpu_result is None:
            print("\n⚠️  GPU benchmark failed, but CPU results are available")
            return

        # Comparison
        print("\n" + "="*80)
        print("  A10G GPU vs CPU COMPARISON")
        print("="*80)

        speedup_transcribe = cpu_result['transcribe_time'] / gpu_result['transcribe_time']
        speedup_total = cpu_result['total_time'] / gpu_result['total_time']
        rtf_improvement = gpu_result['rtf'] / cpu_result['rtf']

        print(f"\n{'Metric':<25} {'CPU':<20} {'GPU':<20} {'Speedup':<15}")
        print("-"*80)
        print(f"{'Load Time':<25} {cpu_result['load_time']:.2f}s{'':<15} "
              f"{gpu_result['load_time']:.2f}s{'':<15} "
              f"{cpu_result['load_time']/gpu_result['load_time']:.2f}x")
        print(f"{'Transcribe Time':<25} {cpu_result['transcribe_time']:.2f}s{'':<15} "
              f"{gpu_result['transcribe_time']:.2f}s{'':<15} "
              f"{speedup_transcribe:.2f}x")
        print(f"{'Total Time':<25} {cpu_result['total_time']:.2f}s{'':<15} "
              f"{gpu_result['total_time']:.2f}s{'':<15} "
              f"{speedup_total:.2f}x")
        print(f"{'RTF (realtime factor)':<25} {cpu_result['rtf']:.1f}x{'':<16} "
              f"{gpu_result['rtf']:.1f}x{'':<16} "
              f"{rtf_improvement:.2f}x better")
        print(f"{'Audio Duration':<25} {cpu_result['duration']:.1f}s")
        print(f"{'Words':<25} {cpu_result['words']}")
        print(f"{'Word Timestamps':<25} {cpu_result['word_timestamps']}")

        print("\n" + "="*80)
        print("  🎉 A10G GPU PERFORMANCE SUMMARY")
        print("="*80)
        print(f"\n  GPU is {speedup_transcribe:.2f}x FASTER than CPU for transcription")
        print(f"  GPU achieves {gpu_result['rtf']:.1f}x realtime factor")
        print(f"  Total pipeline speedup: {speedup_total:.2f}x")

        # Estimate vs OpenAI Whisper
        # From docs: AudioDecode CPU is 1.77x faster on Mac, 6.0x on Linux
        # Assume this Linux system is similar to Linux benchmark
        vs_openai_cpu = 6.0  # Conservative estimate
        vs_openai_gpu = vs_openai_cpu * speedup_transcribe

        print(f"\n  📊 Estimated Performance vs OpenAI Whisper:")
        print(f"     • CPU: ~{vs_openai_cpu:.1f}x faster (from Linux benchmarks)")
        print(f"     • GPU: ~{vs_openai_gpu:.1f}x faster (extrapolated)")

        # Save results
        print(f"\n  💾 Saving results...")
        with open("A10G_BENCHMARK_RESULTS.md", "w") as f:
            f.write(f"# AudioDecode A10G GPU Benchmark Results\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Audio File:** {audio_file}\n")
            f.write(f"**Audio Duration:** {cpu_result['duration']:.1f}s\n\n")
            f.write(f"## Performance Results\n\n")
            f.write(f"| Metric | CPU | GPU (A10G) | Speedup |\n")
            f.write(f"|--------|-----|------------|----------|\n")
            f.write(f"| Load Time | {cpu_result['load_time']:.2f}s | {gpu_result['load_time']:.2f}s | {cpu_result['load_time']/gpu_result['load_time']:.2f}x |\n")
            f.write(f"| Transcribe Time | {cpu_result['transcribe_time']:.2f}s | {gpu_result['transcribe_time']:.2f}s | **{speedup_transcribe:.2f}x** |\n")
            f.write(f"| Total Time | {cpu_result['total_time']:.2f}s | {gpu_result['total_time']:.2f}s | **{speedup_total:.2f}x** |\n")
            f.write(f"| RTF (realtime) | {cpu_result['rtf']:.1f}x | {gpu_result['rtf']:.1f}x | {rtf_improvement:.2f}x |\n\n")
            f.write(f"## Quality Metrics\n\n")
            f.write(f"- Words: {cpu_result['words']}\n")
            f.write(f"- Segments: {cpu_result['segments']}\n")
            f.write(f"- Word Timestamps: {cpu_result['word_timestamps']}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"**A10G GPU is {speedup_transcribe:.2f}x faster than CPU** for transcription.\n\n")
            f.write(f"**Estimated vs OpenAI Whisper:** ~{vs_openai_gpu:.1f}x faster on GPU\n")

        print(f"     ✓ Results saved to A10G_BENCHMARK_RESULTS.md")

    else:
        print("\n⚠️  No GPU available for comparison")
        print("   This machine doesn't have CTranslate2 GPU support enabled")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
