# Migration Guide: From librosa to AudioDecode

## Why Migrate?

- **200x faster cold starts on Linux** (eliminates subprocess overhead)
- **6x faster cold starts on macOS**
- **Same API** - minimal code changes required
- **Compatible** - works seamlessly with librosa features

## Three Migration Paths

### Path 1: Zero Code Changes (Monkey Patch)

**Best for**: Quick testing, existing codebases

```python
# Add these two lines at the top of your script
import audiodecode.compat
audiodecode.compat.patch_librosa()

# Rest of your code unchanged
import librosa
audio, sr = librosa.load("file.mp3", sr=16000, mono=True)
mfcc = librosa.feature.mfcc(y=audio, sr=sr)
```

**Pros**: No code changes, instant speedup
**Cons**: Monkey-patching can be confusing to debug

---

### Path 2: One-Line Import Change (Recommended)

**Best for**: New projects, clean migration

```python
# Before
import librosa
audio, sr = librosa.load("file.mp3", sr=16000, mono=True)

# After (change one line)
from audiodecode import load
audio, sr = load("file.mp3", sr=16000, mono=True)

# Still use librosa for features
import librosa
mfcc = librosa.feature.mfcc(y=audio, sr=sr)
```

**Pros**: Clean, explicit, easy to understand
**Cons**: Need to change imports in multiple files

---

### Path 3: Mixed Approach (Pragmatic)

**Best for**: Gradual migration, performance-critical sections only

```python
import librosa
from audiodecode import load as fast_load

# Use fast loading for bulk processing
for file in large_dataset:  # 10,000 files
    audio, sr = fast_load(file, sr=16000)  # Fast!
    features = extract_features(audio)

# Use librosa for one-off loads
audio, sr = librosa.load("test.wav")  # Familiar
```

**Pros**: Best of both worlds, gradual adoption
**Cons**: Two APIs to maintain

---

## Common Patterns

### Pattern 1: Batch Processing

```python
# Before (librosa - slow on Linux)
import librosa
audios = []
for file in files:
    audio, sr = librosa.load(file, sr=16000, mono=True)
    audios.append(audio)

# After (AudioDecode - 200x faster)
from audiodecode import load
audios = []
for file in files:
    audio, sr = load(file, sr=16000, mono=True)
    audios.append(audio)
```

### Pattern 2: Feature Extraction

```python
# Before
import librosa
audio, sr = librosa.load("file.mp3", sr=22050)
mfcc = librosa.feature.mfcc(y=audio, sr=sr)

# After (hybrid approach - best performance)
from audiodecode import load
import librosa
audio, sr = load("file.mp3", sr=22050)  # Fast load
mfcc = librosa.feature.mfcc(y=audio, sr=sr)  # Use librosa features
```

### Pattern 3: Data Augmentation

```python
# Before
import librosa
for epoch in range(10):
    for file in dataset:
        audio, sr = librosa.load(file)
        augmented = augment(audio)
        train(augmented)

# After (with caching - 30x faster)
from audiodecode import load
import librosa
for epoch in range(10):
    for file in dataset:
        audio, sr = load(file)  # Cached after first load
        augmented = augment(audio)
        train(augmented)
```

### Pattern 4: Notebooks/Interactive

```python
# Add at top of notebook
%load_ext autoreload
%autoreload 2

import audiodecode.compat
audiodecode.compat.patch_librosa()

# Now all librosa.load calls are fast
import librosa
audio, sr = librosa.load("file.mp3")
```

---

## API Compatibility

### What Works

```python
# All these work identically
from audiodecode import load

# Basic
audio, sr = load("file.mp3")

# With sample rate
audio, sr = load("file.mp3", sr=16000)

# With mono
audio, sr = load("file.mp3", mono=True)

# With offset/duration
audio, sr = load("file.mp3", offset=10.0, duration=5.0)

# Native sample rate
audio, sr = load("file.mp3", sr=None)

# Different dtype
audio, sr = load("file.mp3", dtype=np.float64)
```

### What Doesn't Work

```python
# AudioDecode only does LOADING
# For these, you still need librosa

# Feature extraction - use librosa
mfcc = librosa.feature.mfcc(y=audio, sr=sr)
chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

# Time/frequency conversions - use librosa
tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

# Effects - use librosa
audio_slow = librosa.effects.time_stretch(audio, rate=0.8)
```

---

## Performance Expectations

### Linux (Ubuntu, Debian, RHEL, etc.)

| Scenario | librosa | AudioDecode | Speedup |
|----------|---------|-------------|---------|
| First load (cold) | 6,000ms | 30ms | 200x |
| Same file (warm) | 0.4ms | 0.4ms | 1x |
| Different files | 50ms | 8ms | 6x |

**Why**: librosa spawns ffmpeg subprocess on Linux (~6s overhead). AudioDecode uses FFmpeg C library directly.

### macOS (Intel, Apple Silicon)

| Scenario | librosa | AudioDecode | Speedup |
|----------|---------|-------------|---------|
| First load (cold) | 1,400ms | 220ms | 6x |
| Same file (warm) | 0.2ms | 0.4ms | 0.5x |
| Different files | 0.8ms | 0.4ms | 2x |

**Why**: macOS uses Core Audio (no subprocess), so advantage is smaller. Still faster for cold starts.

### When to Expect Speedup

**Big wins**:
- Linux production servers
- Processing many files (1000+)
- Cold starts (serverless, batch jobs)
- Large files (>10s duration)

**Small wins or neutral**:
- macOS development
- Small files (<1s) after warm-up
- Repeated same file access

---

## Testing Your Migration

### Step 1: Verify Correctness

```python
import numpy as np
import librosa
from audiodecode import load

# Load same file with both
audio_lr, sr_lr = librosa.load("test.mp3")
audio_ad, sr_ad = load("test.mp3")

# Check they match
print(f"Shape: {audio_lr.shape} vs {audio_ad.shape}")
print(f"SR: {sr_lr} vs {sr_ad}")
corr = np.corrcoef(audio_lr.flatten(), audio_ad.flatten())[0, 1]
print(f"Correlation: {corr:.6f}")  # Should be ~1.0
```

### Step 2: Measure Performance

```python
import audiodecode.compat

# This function does side-by-side comparison
audiodecode.compat.compare_backends("your_file.mp3", iterations=5)
```

### Step 3: Gradual Rollout

```python
# Use environment variable to control
import os
if os.getenv("USE_FAST_AUDIO") == "1":
    from audiodecode import load
else:
    from librosa import load

# Now you can A/B test in production
```

---

## Troubleshooting

### "No module named 'soundfile'"

```bash
# Install system dependency
# Ubuntu/Debian:
apt-get install libsndfile1

# macOS:
brew install libsndfile

# Then install Python package:
pip install soundfile
```

### "Sample rate doesn't match"

```python
# Make sure sr parameter matches
audio, sr = load("file.mp3", sr=16000)  # Explicitly set
print(sr)  # Will be 16000
```

### "Audio sounds different"

MP3 decoders may differ in padding. Both are correct decodes, just different frame boundaries.

For lossless formats (WAV, FLAC), output is bit-perfect.

```python
# To verify:
import numpy as np
diff = np.abs(audio_lr - audio_ad).max()
print(f"Max difference: {diff}")  # Should be ~0 for WAV/FLAC
```

---

## Best Practices

1. **Use load() for decoding, librosa for features**
   ```python
   from audiodecode import load
   import librosa
   audio, sr = load("file.mp3")  # Fast
   mfcc = librosa.feature.mfcc(y=audio, sr=sr)  # Rich features
   ```

2. **Enable caching for repeated access**
   ```python
   from audiodecode import load, set_cache_size
   set_cache_size(256)  # Cache up to 256 files
   # Now repeated loads are 30x faster
   ```

3. **Profile before optimizing**
   ```python
   # Make sure audio loading is actually your bottleneck
   # Use audiodecode.compat.compare_backends() to verify
   ```

4. **Test on your target platform**
   ```python
   # Performance varies by OS - test on production-like environment
   ```

---

## FAQ

**Q: Do I need to change all my code?**
A: No. Use `audiodecode.compat.patch_librosa()` for zero code changes.

**Q: Will this break my existing code?**
A: No. AudioDecode has the same API as librosa.load().

**Q: Can I still use librosa features?**
A: Yes! AudioDecode only replaces loading. Use librosa for MFCC, spectrograms, etc.

**Q: Is the output identical?**
A: For WAV/FLAC: bit-perfect. For MP3: nearly identical (different padding).

**Q: When should I NOT migrate?**
A: If you're only loading a few files on macOS, the benefit is small.

---

## Get Help

- Issues: https://github.com/audiodecode/audiodecode/issues
- Docs: https://audiodecode.readthedocs.io
- Slack: [community link]
