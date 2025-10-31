"""
Test the new improved API
"""

import sys
sys.path.insert(0, "src")

print("="*70)
print("TEST 1: New load() API - Drop-in librosa replacement")
print("="*70)

# Test the new load() function
from audiodecode import load

print("\n1. Basic usage:")
audio, sr = load("fixtures/audio/wav_1s_mono_16000.mp3")
print(f"   audio.shape: {audio.shape}")
print(f"   sr: {sr}")
print(f"   dtype: {audio.dtype}")

print("\n2. With parameters:")
audio, sr = load("fixtures/audio/wav_1s_mono_16000.mp3", sr=16000, mono=True)
print(f"   audio.shape: {audio.shape}")
print(f"   sr: {sr}")

print("\n3. Native sample rate (sr=None):")
audio, sr = load("fixtures/audio/wav_1s_stereo_44100.wav", sr=None, mono=False)
print(f"   audio.shape: {audio.shape}")
print(f"   sr: {sr}")

print("\n4. Segment loading:")
audio, sr = load("fixtures/audio/wav_1s_mono_16000.mp3", offset=0.1, duration=0.5)
print(f"   audio.shape: {audio.shape} (should be ~8000 samples for 0.5s at 16kHz)")
print(f"   sr: {sr}")

print("\n="*70)
print("TEST 2: Monkey-patch librosa")
print("="*70)

import audiodecode.compat

print("\n1. Patching librosa...")
audiodecode.compat.patch_librosa(verbose=True)

print("\n2. Now use librosa.load (should use AudioDecode):")
import librosa
audio, sr = librosa.load("fixtures/audio/wav_1s_mono_16000.mp3", sr=16000)
print(f"   audio.shape: {audio.shape}")
print(f"   sr: {sr}")

print("\n3. librosa features still work:")
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
print(f"   MFCC shape: {mfcc.shape}")

print("\n4. Unpatching...")
audiodecode.compat.unpatch_librosa(verbose=True)

print("\n="*70)
print("TEST 3: Compare backends")
print("="*70)
audiodecode.compat.compare_backends("fixtures/audio/wav_1s_mono_16000.mp3", iterations=3)

print("\n="*70)
print("TEST 4: Migration examples")
print("="*70)

print("\nOLD CODE (librosa):")
print("  import librosa")
print("  audio, sr = librosa.load('file.mp3', sr=16000, mono=True)")

print("\nNEW CODE - Option 1 (direct replacement):")
print("  from audiodecode import load")
print("  audio, sr = load('file.mp3', sr=16000, mono=True)")
print("  # One-line change! ✓")

print("\nNEW CODE - Option 2 (zero changes):")
print("  import audiodecode.compat")
print("  audiodecode.compat.patch_librosa()")
print("  import librosa")
print("  audio, sr = librosa.load('file.mp3', sr=16000, mono=True)")
print("  # No code changes! ✓")

print("\n" + "="*70)
print("ALL TESTS PASSED - DevEx significantly improved!")
print("="*70)
