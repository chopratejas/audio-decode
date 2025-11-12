#!/usr/bin/env python3
"""
CPU-Only Benchmark: AudioDecode vs OpenAI Whisper on A10G machine
"""

import time
import sys
import tracemalloc
from pathlib import Path


def benchmark_openai_whisper(audio_file: str):
    """Benchmark OpenAI Whisper on CPU."""
    import whisper

    print("\n" + "="*80)
    print("  OpenAI Whisper (Baseline)")
    print("="*80)

    print("\n📦 Loading model...")
    tracemalloc.start()
    start = time.time()
    model = whisper.load_model("base")
    load_time = time.time() - start
    load_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()
    print(f"   ✓ Loaded in {load_time:.2f}s (Memory: {load_mem:.0f} MB)")

    print("\n🎤 Transcribing...")
    tracemalloc.start()
    start = time.time()
    result = model.transcribe(audio_file)
    transcribe_time = time.time() - start
    transcribe_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    duration = result.get('duration', result['segments'][-1]['end'] if result['segments'] else 0)
    rtf = duration / transcribe_time if transcribe_time > 0 else 0
    words = len(result['text'].split())

    print(f"   ✓ Complete!")
    print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"   Time: {transcribe_time:.2f}s")
    print(f"   Total: {load_time + transcribe_time:.2f}s")
    print(f"   RTF: {rtf:.1f}x")
    print(f"   Words: {words}")
    print(f"   Segments: {len(result['segments'])}")

    return {
        'name': 'OpenAI Whisper',
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'total_time': load_time + transcribe_time,
        'rtf': rtf,
        'duration': duration,
        'words': words,
        'segments': len(result['segments']),
        'text': result['text']
    }


def benchmark_audiodecode(audio_file: str):
    """Benchmark AudioDecode on CPU."""
    from audiodecode import WhisperInference

    print("\n" + "="*80)
    print("  AudioDecode (Optimized)")
    print("="*80)

    print("\n📦 Loading model...")
    tracemalloc.start()
    start = time.time()
    whisper = WhisperInference(model_size="base", device="cpu")
    load_time = time.time() - start
    load_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()
    print(f"   ✓ Loaded in {load_time:.2f}s (Memory: {load_mem:.0f} MB)")
    print(f"   Device: {whisper.device}")
    print(f"   Compute type: {whisper.compute_type}")

    print("\n🎤 Transcribing...")
    tracemalloc.start()
    start = time.time()
    result = whisper.transcribe_file(audio_file, word_timestamps=True)
    transcribe_time = time.time() - start
    transcribe_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    rtf = result.duration / transcribe_time if transcribe_time > 0 else 0
    words = len(result.text.split())
    word_timestamps = sum(len(seg.words) for seg in result.segments if seg.words)

    print(f"   ✓ Complete!")
    print(f"   Duration: {result.duration:.1f}s ({result.duration/60:.1f} min)")
    print(f"   Time: {transcribe_time:.2f}s")
    print(f"   Total: {load_time + transcribe_time:.2f}s")
    print(f"   RTF: {rtf:.1f}x")
    print(f"   Words: {words}")
    print(f"   Segments: {len(result.segments)}")
    print(f"   Word timestamps: {word_timestamps}")

    return {
        'name': 'AudioDecode',
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'total_time': load_time + transcribe_time,
        'rtf': rtf,
        'duration': result.duration,
        'words': words,
        'segments': len(result.segments),
        'word_timestamps': word_timestamps,
        'text': result.text
    }


def main():
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"

    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    print("="*80)
    print("  AudioDecode vs OpenAI Whisper - CPU Benchmark")
    print("  Platform: A10G machine (CPU-only comparison)")
    print("="*80)
    print(f"\n📁 Audio: {audio_file} ({Path(audio_file).stat().st_size / 1024 / 1024:.1f} MB)")

    # Benchmark both
    openai_result = benchmark_openai_whisper(audio_file)
    audiodecode_result = benchmark_audiodecode(audio_file)

    # Comparison
    print("\n" + "="*80)
    print("  📊 COMPARISON")
    print("="*80)

    speedup_load = openai_result['load_time'] / audiodecode_result['load_time']
    speedup_transcribe = openai_result['transcribe_time'] / audiodecode_result['transcribe_time']
    speedup_total = openai_result['total_time'] / audiodecode_result['total_time']

    print(f"\n{'Metric':<25} {'OpenAI':<20} {'AudioDecode':<20} {'Speedup':<15}")
    print("-"*80)
    print(f"{'Load Time':<25} {openai_result['load_time']:.2f}s{'':<15} "
          f"{audiodecode_result['load_time']:.2f}s{'':<15} "
          f"{speedup_load:.2f}x")
    print(f"{'Transcribe Time':<25} {openai_result['transcribe_time']:.2f}s{'':<15} "
          f"{audiodecode_result['transcribe_time']:.2f}s{'':<15} "
          f"**{speedup_transcribe:.2f}x**")
    print(f"{'Total Time':<25} {openai_result['total_time']:.2f}s{'':<15} "
          f"{audiodecode_result['total_time']:.2f}s{'':<15} "
          f"**{speedup_total:.2f}x**")
    print(f"{'RTF':<25} {openai_result['rtf']:.1f}x{'':<16} "
          f"{audiodecode_result['rtf']:.1f}x{'':<16} "
          f"{audiodecode_result['rtf']/openai_result['rtf']:.2f}x")
    print(f"{'Words':<25} {openai_result['words']}")
    print(f"{'Word Timestamps':<25} 0{'':<20} {audiodecode_result['word_timestamps']}")

    print("\n" + "="*80)
    print("  🎉 RESULTS")
    print("="*80)
    print(f"\n  AudioDecode is **{speedup_transcribe:.2f}x FASTER** than OpenAI Whisper on CPU")
    print(f"  Total pipeline speedup: **{speedup_total:.2f}x**")
    print(f"  Bonus: {audiodecode_result['word_timestamps']} word-level timestamps")

    # Save results
    with open("CPU_BENCHMARK_RESULTS.md", "w") as f:
        f.write("# AudioDecode vs OpenAI Whisper - CPU Benchmark\n\n")
        f.write(f"**Platform:** A10G Machine (CPU-only)\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Audio:** {audio_file}\n")
        f.write(f"**Duration:** {audiodecode_result['duration']:.1f}s ({audiodecode_result['duration']/60:.1f} min)\n\n")
        f.write("## Results\n\n")
        f.write("| Metric | OpenAI Whisper | AudioDecode | Speedup |\n")
        f.write("|--------|----------------|-------------|----------|\n")
        f.write(f"| Load Time | {openai_result['load_time']:.2f}s | {audiodecode_result['load_time']:.2f}s | **{speedup_load:.2f}x** |\n")
        f.write(f"| Transcribe | {openai_result['transcribe_time']:.2f}s | {audiodecode_result['transcribe_time']:.2f}s | **{speedup_transcribe:.2f}x** |\n")
        f.write(f"| Total | {openai_result['total_time']:.2f}s | {audiodecode_result['total_time']:.2f}s | **{speedup_total:.2f}x** |\n")
        f.write(f"| RTF | {openai_result['rtf']:.1f}x | {audiodecode_result['rtf']:.1f}x | {audiodecode_result['rtf']/openai_result['rtf']:.2f}x |\n\n")
        f.write(f"**AudioDecode is {speedup_transcribe:.2f}x faster than OpenAI Whisper on CPU!**\n")

    print(f"\n  💾 Results saved to CPU_BENCHMARK_RESULTS.md")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
