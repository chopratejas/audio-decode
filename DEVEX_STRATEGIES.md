# DevEx Improvement Strategies - Ultra-Thinking

## Current Pain Points

1. Two-step API (create decoder, then decode)
2. Sample rate not returned
3. Still need librosa for features
4. Installation can fail (system dependencies)
5. Not a drop-in replacement

## Strategy 1: Drop-in librosa.load() Replacement ⭐⭐⭐⭐⭐

### Approach: Monkey-patch or provide compatible function

```python
# User code - BEFORE
import librosa
audio, sr = librosa.load("file.mp3", sr=16000, mono=True)

# User code - AFTER (Option A: Monkey patch)
import audiodecode.librosa_compat
audiodecode.librosa_compat.install()  # Patches librosa.load
import librosa
audio, sr = librosa.load("file.mp3", sr=16000, mono=True)  # Now uses AudioDecode!

# User code - AFTER (Option B: Direct replacement)
from audiodecode import load  # Drop-in replacement
audio, sr = load("file.mp3", sr=16000, mono=True)
```

### Implementation
```python
# src/audiodecode/librosa_compat.py
def load(path, *, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type="soxr_hq"):
    """Drop-in replacement for librosa.load with same signature"""
    decoder = AudioDecoder(path, target_sr=sr, mono=mono)
    audio = decoder.decode(offset=offset, duration=duration)

    # Return tuple like librosa
    actual_sr = sr if sr is not None else decoder.info()["sample_rate"]
    return audio.astype(dtype), actual_sr

def install():
    """Monkey-patch librosa to use AudioDecode backend"""
    import librosa.core.audio
    librosa.core.audio.load = load
    librosa.load = load
```

**Pros**:
- Zero API change for users
- Returns sample rate (fixes major friction)
- Can be opt-in (call install() to activate)
- Works with existing code

**Cons**:
- Monkey-patching is controversial
- May break if librosa internals change
- Users might not know which backend they're using

**DevEx Score**: 9/10 - Minimal friction, works with existing code

---

## Strategy 2: Make AudioDecode a librosa Backend (like audioread)

### Approach: Integrate into librosa's backend system

Looking at librosa source, it uses:
1. `soundfile` for WAV/FLAC
2. `audioread` for everything else (which spawns ffmpeg)

We could make AudioDecode an audioread alternative.

```python
# User code
import librosa
# Set environment variable or config
os.environ['LIBROSA_AUDIOREAD_BACKEND'] = 'audiodecode'
audio, sr = librosa.load("file.mp3")  # Automatically uses AudioDecode
```

**Pros**:
- Clean integration with librosa
- No monkey patching
- Users still use librosa.load()

**Cons**:
- Requires changes to librosa (PR to their repo)
- Long timeline (waiting for librosa maintainers)
- May not be accepted

**DevEx Score**: 10/10 IF accepted, 0/10 if not

---

## Strategy 3: Better API - Match librosa Exactly

### Approach: Provide exact librosa.load() API

```python
# src/audiodecode/__init__.py
def load(path, *, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    """
    Load audio file. Drop-in replacement for librosa.load().

    This is the recommended API - matches librosa exactly.
    """
    decoder = AudioDecoder(path, target_sr=sr, mono=mono)
    audio = decoder.decode(offset=offset, duration=duration)
    actual_sr = sr if sr is not None else decoder.info()["sample_rate"]
    return audio.astype(dtype), actual_sr

# Keep AudioDecoder for advanced usage
__all__ = ["load", "AudioDecoder", "clear_cache", ...]
```

```python
# User code - MIGRATION IS EASY
# Change this:
import librosa
audio, sr = librosa.load("file.mp3", sr=16000)

# To this (one-line change):
from audiodecode import load
audio, sr = load("file.mp3", sr=16000)
```

**Pros**:
- One-line migration
- Returns sample rate
- No monkey patching
- We control the API

**Cons**:
- Users have to change imports
- Still need librosa for features

**DevEx Score**: 8/10 - Very easy migration

---

## Strategy 4: Improve Installation - Auto-install System Dependencies

### Approach: Use post-install script or wheels with bundled libs

```python
# setup.py or pyproject.toml
[tool.setuptools]
post-install = "audiodecode.install:post_install"

# src/audiodecode/install.py
def post_install():
    """Try to install system dependencies if missing"""
    import platform
    import subprocess

    if platform.system() == "Linux":
        # Try to install via package manager
        try:
            subprocess.run(["apt-get", "install", "-y", "libsndfile1"], check=False)
        except:
            print("Warning: Could not install libsndfile. Please install manually.")

    elif platform.system() == "Darwin":
        try:
            subprocess.run(["brew", "install", "libsndfile"], check=False)
        except:
            print("Warning: Could not install libsndfile. Please install manually.")
```

**OR BETTER**: Bundle FFmpeg/libsndfile in wheels

```bash
# Build wheels with bundled libraries
python -m build --wheel
# Wheel contains: audiodecode + libavcodec + libsndfile
```

**Pros**:
- pip install "just works"
- No manual system dependency installation

**Cons**:
- Larger wheel size
- Security concerns (bundled libs may be outdated)
- Platform-specific wheels needed

**DevEx Score**: 9/10 - Frictionless install

---

## Strategy 5: Pure Python Fallback

### Approach: If PyAV/soundfile missing, use pure Python decoder

```python
# src/audiodecode/backends/fallback.py
class PurePythonBackend:
    """Slow but works everywhere - uses only stdlib"""
    def decode(self, filepath):
        # Use wave module for WAV
        # Use minimp3 (pure Python) for MP3
        # Slower but no dependencies
```

**Pros**:
- pip install always works
- Graceful degradation

**Cons**:
- Slower (defeats the purpose)
- Complex to maintain

**DevEx Score**: 6/10 - Works but slow

---

## Strategy 6: Provide Pre-configured Docker/Conda

### Approach: Easy environment setup

```bash
# Option A: Docker
docker run -v $(pwd):/data audiodecode/runtime python script.py

# Option B: Conda (includes all system deps)
conda install -c conda-forge audiodecode
# Pulls in FFmpeg, libsndfile automatically

# Option C: UV with system deps
uvx --with audiodecode[all] python script.py
```

**Pros**:
- Guaranteed working environment
- Good for production

**Cons**:
- Overhead of container/conda
- Not ideal for quick scripts

**DevEx Score**: 7/10 - Good for production, overkill for dev

---

## RECOMMENDATION: Combination Strategy

### Phase 1: Better API (Immediate)
```python
# Provide librosa-compatible load() function
from audiodecode import load  # Drop-in replacement
audio, sr = load("file.mp3", sr=16000, mono=True)
```

### Phase 2: Bundled Wheels (Better Install)
```bash
# Build manylinux wheels with bundled FFmpeg/libsndfile
pip install audiodecode  # Just works, no system deps
```

### Phase 3: Optional Monkey-patch (Advanced Users)
```python
# For zero-change migration
import audiodecode.compat
audiodecode.compat.patch_librosa()  # Now librosa.load uses AudioDecode
import librosa
audio, sr = librosa.load("file.mp3")  # Fast!
```

### Phase 4: Upstream Integration (Long-term)
```python
# PR to librosa to add AudioDecode as backend option
# Then it "just works" with librosa
```

---

## What to Build NOW

### Priority 1: `load()` function (1 hour)
```python
# src/audiodecode/__init__.py
def load(path, *, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    decoder = AudioDecoder(path, target_sr=sr, mono=mono)
    audio = decoder.decode(offset=offset, duration=duration)
    actual_sr = sr if sr is not None else decoder.info()["sample_rate"]
    return audio.astype(dtype), actual_sr
```

### Priority 2: Monkey-patch helper (30 min)
```python
# src/audiodecode/compat.py
def patch_librosa():
    """Make librosa.load use AudioDecode backend"""
    import librosa.core.audio
    from audiodecode import load
    librosa.core.audio.load = load
    librosa.load = load
```

### Priority 3: Better error messages (30 min)
```python
# If soundfile/av missing, show helpful message
try:
    import soundfile
except ImportError:
    raise ImportError(
        "soundfile not found. Install with:\n"
        "  pip install soundfile\n"
        "Or install system package:\n"
        "  Ubuntu/Debian: apt-get install libsndfile1\n"
        "  macOS: brew install libsndfile"
    )
```

### Priority 4: Migration guide (1 hour)
Create doc showing exact before/after for common librosa patterns.

---

## Expected DevEx After Changes

### Before
```python
# Confusing, two-step, no sr returned
from audiodecode import AudioDecoder
audio = AudioDecoder("file.mp3", target_sr=16000, mono=True).decode()
# sr = ??? (you have to remember)
```

### After
```python
# Familiar, one-step, returns sr
from audiodecode import load
audio, sr = load("file.mp3", sr=16000, mono=True)
# Exactly like librosa!

# Or zero-change migration:
import audiodecode.compat
audiodecode.compat.patch_librosa()
import librosa
audio, sr = librosa.load("file.mp3")  # Fast backend!
```

**DevEx Score**: 3/10 → 9/10

---

## The Killer Feature: Hot-swap Backend

```python
# Cool idea: Let users compare easily
from audiodecode import load as fast_load
import librosa

# Compare performance
with timer():
    audio1, sr1 = librosa.load("file.mp3")  # ~6s on Linux
with timer():
    audio2, sr2 = fast_load("file.mp3")     # ~0.02s on Linux

# Use fast one in production
load = fast_load if IS_PRODUCTION else librosa.load
```

This makes it easy to A/B test and validate correctness.
