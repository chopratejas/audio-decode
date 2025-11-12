#!/usr/bin/env python3
"""Clean A10G GPU Benchmark - Both systems on GPU"""
import time
import sys
from pathlib import Path

audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"
if not Path(audio_file).exists():
    print(f"File not found: {audio_file}")
    sys.exit(1)

print("="*80)
print("  A10G GPU BENCHMARK: AudioDecode vs OpenAI Whisper")
print("="*80)
print(f"\nAudio: {audio_file} ({Path(audio_file).stat().st_size/1024/1024:.1f} MB)\n")

# OpenAI Whisper on GPU
print("="*80)
print("BASELINE: OpenAI Whisper on GPU")
print("="*80)
import whisper
import torch

print(f"\nLoading model on GPU...")
t0 = time.time()
model = whisper.load_model("base", device="cuda")
load_time_openai = time.time() - t0
print(f"Loaded in {load_time_openai:.2f}s")

print(f"\nTranscribing...")
t0 = time.time()
result_openai = model.transcribe(audio_file)
transcribe_time_openai = time.time() - t0
duration = result_openai.get('duration', result_openai['segments'][-1]['end'])
rtf_openai = duration / transcribe_time_openai
print(f"Done in {transcribe_time_openai:.2f}s")
print(f"Duration: {duration:.1f}s, RTF: {rtf_openai:.1f}x")

# AudioDecode on GPU
print("\n" + "="*80)
print("AUDIODECODE: Optimized GPU Inference")
print("="*80)
from audiodecode import WhisperInference

print(f"\nLoading model on GPU...")
t0 = time.time()
whisper_ad = WhisperInference(model_size="base", device="cuda")
load_time_ad = time.time() - t0
print(f"Loaded in {load_time_ad:.2f}s")

print(f"\nTranscribing...")
t0 = time.time()
result_ad = whisper_ad.transcribe_file(audio_file, word_timestamps=True)
transcribe_time_ad = time.time() - t0
rtf_ad = result_ad.duration / transcribe_time_ad
word_ts = sum(len(s.words) for s in result_ad.segments if s.words)
print(f"Done in {transcribe_time_ad:.2f}s")
print(f"Duration: {result_ad.duration:.1f}s, RTF: {rtf_ad:.1f}x")

# Results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
speedup = transcribe_time_openai / transcribe_time_ad
print(f"\nOpenAI Whisper GPU:  {transcribe_time_openai:.2f}s  (RTF: {rtf_openai:.1f}x)")
print(f"AudioDecode GPU:     {transcribe_time_ad:.2f}s  (RTF: {rtf_ad:.1f}x)")
print(f"\nSpeedup: {speedup:.2f}x FASTER")
print(f"Word timestamps: {word_ts}")

# Save
with open("A10G_RESULTS.md", "w") as f:
    f.write(f"# A10G GPU Benchmark Results\n\n")
    f.write(f"Audio: {duration/60:.1f} min\n\n")
    f.write(f"| System | Time | RTF | Speedup |\n")
    f.write(f"|--------|------|-----|----------|\n")
    f.write(f"| OpenAI Whisper (GPU) | {transcribe_time_openai:.2f}s | {rtf_openai:.1f}x | baseline |\n")
    f.write(f"| AudioDecode (GPU) | {transcribe_time_ad:.2f}s | {rtf_ad:.1f}x | **{speedup:.2f}x** |\n")
    f.write(f"\n**AudioDecode is {speedup:.2f}x faster on A10G GPU!**\n")

print(f"\nResults saved to A10G_RESULTS.md\n")
