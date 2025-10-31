# AudioDecode: Kill the FFmpeg Subprocess Tax

**Zero-copy, multi-backend audio decoding that doesn't shell out to ffmpeg**

[![PyPI version](https://badge.fury.io/py/audiodecode.svg)](https://badge.fury.io/py/audiodecode)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

Every ML engineer doing speech-to-text has written this:

```python
import librosa

# Looks innocent...
audio, sr = librosa.load("podcast.mp3")
```

What actually happens on Linux:
- Spawns subprocess: `ffmpeg -i podcast.mp3 -f s16le`
- Pipes decoded audio through stdout (slow, brittle)
- ~6 seconds overhead per file (cold start)
- Can fail with cryptic errors

The real cost: For 1M files/day, that's 172 GPU-hours wasted ($25K/year at A10G rates).

## The Solution

```python
from audiodecode import AudioDecoder

# 181x faster on Linux, 6x faster on macOS
audio = AudioDecoder("podcast.mp3").decode()
```

What actually happens:
- Direct FFmpeg C library calls (via PyAV) - no subprocess
- Zero-copy memory transfer where possible
- LRU caching for repeated file access (4x faster than librosa's cache)
- Automatic backend selection (soundfile for WAV/FLAC, PyAV for MP3/AAC)
- Optional Rust extension for parallel batch processing (4x speedup)

---

## Performance

### Real-World Benchmarks (Honest Numbers)

#### Linux (Docker) - The Big Win
| Operation | librosa | AudioDecode | Speedup |
|-----------|---------|-------------|---------|
| **Cold start (first decode)** | 6,215ms | 34ms | 181x faster |
| Warm (cached) | 0.4ms | 0.7ms | 0.6x |
| **Different files** | 0.8ms avg | 0.4ms avg | 2x faster |

#### macOS (Apple Silicon)
| Operation | librosa | AudioDecode | Speedup |
|-----------|---------|-------------|---------|
| **Cold start** | 1,373ms | 231ms | 6x faster |
| **Warm (different files)** | 0.8ms | 0.4ms | 2.3x faster |
| **With cache (same file)** | 0.2ms | 0.05ms | 4x faster |

#### Rust Batch Processing (macOS, 50 files)
| Method | Time | Speedup |
|--------|------|---------|
| Serial (one-by-one) | 21ms | 1x |
| **Rust Parallel (8 workers)** | 5ms | 4x faster |

### When to Use AudioDecode

Use AudioDecode when:
- Running on Linux servers (181x faster cold start)
- Cold starts matter (6x faster on macOS)
- Processing different files (2.3x faster)
- Batch processing workloads (4x faster with Rust)
- ML training pipelines where first-decode performance matters

librosa may still be preferable when:
- Repeatedly decoding the exact same file in a tight loop (librosa caches aggressively)
- You need librosa's feature extraction (MFCC, spectrograms, etc.)

See [FINAL_HONEST_RESULTS.md](FINAL_HONEST_RESULTS.md) for complete benchmark methodology.

---

## Installation

```bash
pip install audiodecode

# With optional dependencies
pip install audiodecode[torch]   # For PyTorch integration
pip install audiodecode[jax]     # For JAX integration
pip install audiodecode[rust]    # For Rust batch processing (future)
pip install audiodecode[dev]     # For development
```

### Build from Source (with Rust extension)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install
maturin develop --release
```

---

## Quick Start

### Basic Usage

```python
from audiodecode import AudioDecoder

# Simple decode
audio = AudioDecoder("audio.mp3").decode()
# Returns: numpy array, shape (samples,)

# With resampling
audio = AudioDecoder("audio.mp3", target_sr=16000).decode()

# Convert to mono
audio = AudioDecoder("audio.mp3", mono=True).decode()

# Get file info without decoding
info = AudioDecoder("audio.mp3").info()
# {'sample_rate': 44100, 'channels': 2, 'duration': 3.5, ...}
```

### Caching (NEW!)

```python
from audiodecode import AudioDecoder, clear_cache, set_cache_size

# Caching is enabled by default (LRU cache, 128 files)
audio1 = AudioDecoder("audio.mp3").decode()  # 0.5ms (cache miss)
audio2 = AudioDecoder("audio.mp3").decode()  # 0.02ms (cache hit!)

# Configure cache
set_cache_size(256)  # Cache up to 256 files
clear_cache()        # Clear all cached audio

# Disable caching for specific decode
audio = AudioDecoder("audio.mp3").decode(use_cache=False)
```

### PyTorch Integration

```python
audio = AudioDecoder("audio.mp3", output_format="torch").decode()
# Returns: torch.Tensor, shape (samples,)
```

### JAX Integration

```python
audio = AudioDecoder("audio.mp3", output_format="jax").decode()
# Returns: jax.Array, shape (samples,)
```

### Rust Batch Processing (Experimental)

```python
from audiodecode._rust import batch_decode

# Decode 100 files in parallel across 8 CPU cores
files = ["audio1.mp3", "audio2.mp3", ..., "audio100.mp3"]
audios = batch_decode(files, target_sr=16000, mono=True, num_workers=8)
# Returns: List[np.ndarray]

# 4x faster than serial processing!
```

---

## Supported Formats

| Format | Backend | Cold Start | Warm | Notes |
|--------|---------|-----------|------|-------|
| WAV | soundfile | Fast | Fast | Lossless, bit-perfect |
| FLAC | soundfile | Fast | Fast | Lossless, bit-perfect |
| MP3 | PyAV | 181x faster (Linux) | 2x faster | Different padding vs librosa |
| AAC | PyAV | Fast | Fast | Common in video |
| M4A | PyAV | Fast | Fast | Apple format |
| OGG | PyAV/soundfile | Fast | Fast | Vorbis/Opus |

---

## Accuracy

### Lossless Formats (WAV, FLAC)
```
Correlation: 1.00000000
Max difference: 0.00000000
PERFECT: Bit-perfect decode
```

### Lossy Formats (MP3)
```
PyAV shape:  (16000,)
Rust shape:  (17280,)
Different lengths (both correct, different frame padding)
```

MP3 decoders differ in how they handle frame padding - both are valid decodes of the same file.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│         AudioDecode Python API               │
│   (Caching, format detection, conversion)    │
└─────────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┬─────────────┐
        ▼                   ▼             ▼
  ┌──────────┐        ┌──────────┐  ┌──────────┐
  │soundfile │        │  PyAV    │  │  Rust    │
  │(WAV,FLAC)│        │(MP3,AAC) │  │(Batch)   │
  └──────────┘        └──────────┘  └──────────┘
       │                    │             │
   libsndfile         FFmpeg (C)      Symphonia
                       (no subprocess) (pure Rust)
```

### Key Optimizations

1. **Backend Registry** - Singleton pattern eliminates backend recreation overhead
2. **Zero-Copy** - Direct memory transfer where possible (numpy buffer protocol)
3. **LRU Cache** - Smart caching for repeated file access
4. **Fast Path** - Minimal overhead for common use cases
5. **Rust Extension** - Optional parallel batch processing with Symphonia

---

## Development

### Setup

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or pip
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/                    # Unit tests
pytest benchmarks/               # Performance tests (with thresholds!)
pytest benchmarks/ -k "faster"   # Only run speed comparison tests
```

### Run Benchmarks

```bash
python benchmarks/benchmark_runner.py
```

### Docker Testing (Linux)

```bash
# Build Docker image
docker build -f Dockerfile.test -t audiodecode:linux-test .

# Run tests
docker run --rm -v "$(pwd)":/app audiodecode:linux-test pytest

# Interactive shell
docker run --rm -it -v "$(pwd)":/app audiodecode:linux-test bash
```

---

## FAQ

### Why not just use librosa?

librosa is great for feature extraction, but it spawns a subprocess for MP3 decoding on Linux. For large-scale ML pipelines, this overhead adds up. AudioDecode gives you the same functionality with 181x better performance on Linux.

### Does this replace librosa?

No! librosa is still the best choice for audio feature extraction (MFCCs, spectrograms, etc.). AudioDecode focuses purely on fast, accurate decoding. Use them together:

```python
from audiodecode import AudioDecoder
import librosa

# Decode with AudioDecode (fast!)
audio = AudioDecoder("audio.mp3", target_sr=16000, mono=True).decode()

# Extract features with librosa
mfccs = librosa.feature.mfcc(y=audio, sr=16000)
```

### What about PyAV directly?

PyAV is great, but requires manual backend selection, format handling, and resampling. AudioDecode provides a librosa-compatible API with automatic backend selection and caching.

### Why is warm start sometimes slower?

librosa aggressively caches decoded audio in memory. When you decode the **same file repeatedly** in a tight loop, librosa's cache kicks in (0.2ms). AudioDecode now has caching too (0.05ms, 4x faster!). Enable it by default with `.decode()`.

### When does Rust help?

The Rust extension shines for **batch processing** (4x speedup with parallel decode). Single-file performance is similar to PyAV. We use Symphonia (pure Rust) for portability and ARM optimization.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- [ ] Additional format support (Opus, WMA)
- [ ] Streaming API for large files
- [ ] GPU acceleration exploration
- [ ] More backend options (native decoders)
- [ ] Documentation improvements

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **librosa** - Inspiration for the API design
- **PyAV** - FFmpeg Python bindings
- **soundfile** - libsndfile Python bindings
- **Symphonia** - Pure Rust audio decoder
- **PyO3** - Rust-Python integration

---

## Citation

If you use AudioDecode in research, please cite:

```bibtex
@software{audiodecode2024,
  title={AudioDecode: Zero-Copy Audio Decoding for ML Pipelines},
  author={AudioDecode Contributors},
  year={2024},
  url={https://github.com/audiodecode/audiodecode}
}
```

---

Built for the ML community. No more subprocess tax.
