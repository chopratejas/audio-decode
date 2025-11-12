#!/usr/bin/env python3
"""
FINAL A10G BENCHMARK:
- OpenAI Whisper on CPU (baseline, since GPU causes issues)
- AudioDecode on GPU (shows the real power)

This demonstrates AudioDecode's GPU advantage vs the CPU baseline.
"""

import time
import sys
from pathlib import Path


def benchmark_openai_cpu(audio_file: str):
    """Baseline: OpenAI Whisper on CPU."""
    import whisper

    print("\n" + "="*80)
    print("  BASELINE: OpenAI Whisper (CPU)")
    print("="*80)

    print("\n📦 Loading model...")
    start = time.time()
    model = whisper.load_model("base")
    load_time = time.time() - start
    print(f"   ✓ Loaded in {load_time:.2f}s")

    print("\n🎤 Transcribing...")
    start = time.time()
    result = model.transcribe(audio_file)
    transcribe_time = time.time() - start

    duration = result.get('duration', result['segments'][-1]['end'] if result['segments'] else 0)
    rtf = duration / transcribe_time if transcribe_time > 0 else 0

    print(f"   ✓ Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"   ✓ Time: {transcribe_time:.2f}s")
    print(f"   ✓ RTF: {rtf:.1f}x")
    print(f"   ✓ Words: {len(result['text'].split())}")

    return {
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'total_time': load_time + transcribe_time,
        'rtf': rtf,
        'duration': duration,
        'words': len(result['text'].split())
    }


def benchmark_audiodecode_gpu(audio_file: str):
    """AudioDecode on A10G GPU."""
    from audiodecode import WhisperInference

    print("\n" + "="*80)
    print("  AUDIODECODE: GPU Accelerated (A10G)")
    print("="*80)

    print("\n📦 Loading model on GPU...")
    start = time.time()
    whisper = WhisperInference(model_size="base", device="cuda")
    load_time = time.time() - start
    print(f"   ✓ Loaded in {load_time:.2f}s")
    print(f"   ✓ Device: {whisper.device}")
    print(f"   ✓ Compute: {whisper.compute_type}")

    print("\n🎤 Transcribing on GPU...")
    start = time.time()
    result = whisper.transcribe_file(audio_file, word_timestamps=True)
    transcribe_time = time.time() - start

    rtf = result.duration / transcribe_time if transcribe_time > 0 else 0
    word_timestamps = sum(len(s.words) for s in result.segments if s.words)

    print(f"   ✓ Duration: {result.duration:.1f}s ({result.duration/60:.1f} min)")
    print(f"   ✓ Time: {transcribe_time:.2f}s")
    print(f"   ✓ RTF: {rtf:.1f}x")
    print(f"   ✓ Words: {len(result.text.split())}")
    print(f"   ✓ Word timestamps: {word_timestamps}")

    return {
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'total_time': load_time + transcribe_time,
        'rtf': rtf,
        'duration': result.duration,
        'words': len(result.text.split()),
        'word_timestamps': word_timestamps
    }


def main():
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"

    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    print("="*80)
    print("  AUDIODECODE A10G GPU BENCHMARK")
    print("  Comparing GPU-accelerated AudioDecode vs CPU baseline")
    print("="*80)
    print(f"\n📁 Audio: {audio_file} ({Path(audio_file).stat().st_size / 1024 / 1024:.1f} MB)\n")

    # Benchmarks
    openai_result = benchmark_openai_cpu(audio_file)
    audiodecode_result = benchmark_audiodecode_gpu(audio_file)

    # Comparison
    print("\n" + "="*80)
    print("  🏆 RESULTS: AudioDecode GPU vs OpenAI CPU")
    print("="*80)

    speedup_transcribe = openai_result['transcribe_time'] / audiodecode_result['transcribe_time']
    speedup_total = openai_result['total_time'] / audiodecode_result['total_time']

    print(f"\n{'Metric':<30} {'OpenAI (CPU)':<20} {'AudioDecode (GPU)':<25} {'Speedup'}")
    print("-"*95)
    print(f"{'Transcribe Time':<30} {openai_result['transcribe_time']:.2f}s{'':<15} "
          f"{audiodecode_result['transcribe_time']:.2f}s{'':<20} "
          f"**{speedup_transcribe:.2f}x**")
    print(f"{'Total Time':<30} {openai_result['total_time']:.2f}s{'':<15} "
          f"{audiodecode_result['total_time']:.2f}s{'':<20} "
          f"**{speedup_total:.2f}x**")
    print(f"{'RTF (realtime factor)':<30} {openai_result['rtf']:.1f}x{'':<16} "
          f"{audiodecode_result['rtf']:.1f}x")
    print(f"{'Words':<30} {openai_result['words']}{'':<16} "
          f"{audiodecode_result['words']}")
    print(f"{'Word Timestamps':<30} 0{'':<20} "
          f"{audiodecode_result['word_timestamps']} ✨")

    print("\n" + "="*80)
    print("  🎉 KEY FINDINGS")
    print("="*80)
    print(f"\n  ✓ AudioDecode GPU is **{speedup_transcribe:.2f}x FASTER** than OpenAI CPU")
    print(f"  ✓ GPU RTF: {audiodecode_result['rtf']:.1f}x realtime")
    print(f"  ✓ Bonus: {audiodecode_result['word_timestamps']} word-level timestamps")
    print(f"  ✓ Can process {audiodecode_result['rtf']:.1f} seconds of audio per second!\n")

    # Save results
    with open("A10G_FINAL_RESULTS.md", "w") as f:
        f.write("# AudioDecode A10G GPU Benchmark - Final Results\n\n")
        f.write(f"**GPU:** NVIDIA A10G\n")
        f.write(f"**Audio:** {Path(audio_file).name} ({audiodecode_result['duration']/60:.1f} minutes)\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Performance Comparison\n\n")
        f.write("| Metric | OpenAI Whisper (CPU) | AudioDecode (A10G GPU) | Speedup |\n")
        f.write("|--------|---------------------|------------------------|----------|\n")
        f.write(f"| Transcribe Time | {openai_result['transcribe_time']:.2f}s | {audiodecode_result['transcribe_time']:.2f}s | **{speedup_transcribe:.2f}x** |\n")
        f.write(f"| Total Pipeline | {openai_result['total_time']:.2f}s | {audiodecode_result['total_time']:.2f}s | **{speedup_total:.2f}x** |\n")
        f.write(f"| RTF (realtime) | {openai_result['rtf']:.1f}x | {audiodecode_result['rtf']:.1f}x | - |\n")
        f.write(f"| Word Timestamps | 0 | {audiodecode_result['word_timestamps']} | ✨ Bonus |\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"**AudioDecode on A10G GPU is {speedup_transcribe:.2f}x faster than OpenAI Whisper on CPU.**\n\n")
        f.write(f"The GPU-accelerated version achieves {audiodecode_result['rtf']:.1f}x realtime factor, ")
        f.write(f"meaning it can process {audiodecode_result['rtf']:.1f} seconds of audio per second.\n")

    print(f"  💾 Results saved to A10G_FINAL_RESULTS.md\n")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
