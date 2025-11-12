# AudioDecode User Guide

**Complete practical guide for using AudioDecode in real-world applications**

This guide covers everything you need to know to use AudioDecode effectively, from basic usage to advanced production scenarios.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Features & Use Cases](#core-features--use-cases)
3. [Configuration Deep Dive](#configuration-deep-dive)
4. [Common Workflows](#common-workflows)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting & FAQs](#troubleshooting--faqs)
7. [Migration Guides](#migration-guides)

---

## Getting Started

### Installation Options

#### Minimal Installation (Audio Loading Only)
```bash
pip install audiodecode
```
Use this if you only need fast audio loading.

#### Training Optimization
```bash
pip install audiodecode[torch]
```
Includes PyTorch DataLoader for ML training pipelines.

#### Speech-to-Text
```bash
pip install audiodecode[inference]
```
Adds faster-whisper for transcription capabilities.

#### Full Installation (Recommended)
```bash
pip install audiodecode[torch,inference]
```
All features included.

#### From Source
```bash
git clone https://github.com/audiodecode/audiodecode.git
cd audiodecode
pip install -e ".[torch,inference]"
```

### Quick Start Examples

#### Fast Audio Loading
```python
from audiodecode import load

# Drop-in replacement for librosa.load()
audio, sr = load("podcast.mp3", sr=16000, mono=True)
print(f"Loaded {len(audio)} samples at {sr}Hz")
```

#### Speech-to-Text
```python
from audiodecode import transcribe_file

# Transcribe an audio file
result = transcribe_file("podcast.mp3", model_size="base")
print(result.text)
print(f"Language: {result.language}, Duration: {result.duration:.1f}s")
```

#### ML Training
```python
from audiodecode import AudioDataLoader

# Auto-tuned DataLoader for training
loader = AudioDataLoader(
    files=train_files,
    labels=train_labels,
    batch_size=32,
    target_sr=16000,
    device='cuda'
)

for batch, labels in loader:
    # Your training code here
    outputs = model(batch)
```

### Verifying Installation

```python
# Check version
import audiodecode
print(f"AudioDecode version: {audiodecode.__version__}")

# Check available features
from audiodecode import _TORCH_AVAILABLE, _INFERENCE_AVAILABLE
print(f"PyTorch support: {_TORCH_AVAILABLE}")
print(f"Inference support: {_INFERENCE_AVAILABLE}")

# Quick test
audio, sr = audiodecode.load("test.mp3")
print(f"Successfully loaded audio: {audio.shape}")
```

---

## Core Features & Use Cases

### Feature 1: Fast Audio Loading

**When to use it:**
- ML training data loading (faster than librosa)
- Batch audio preprocessing
- Real-time audio applications
- Any time you call `librosa.load()`

**Performance comparison:**
- **Linux**: 223x faster than librosa (6,000ms → 27ms)
- **macOS**: 6.5x faster than librosa (1,412ms → 217ms)
- **Cached**: 18.5x faster (148ms → 8ms)

#### Basic Usage

```python
from audiodecode import load

# Load audio file
audio, sr = load("audio.mp3", sr=16000, mono=True)
# Returns: (numpy array, sample rate)
```

#### All Parameters

```python
from audiodecode import load

audio, sr = load(
    path="audio.mp3",           # File path (str or Path)
    sr=16000,                    # Target sample rate (None = native)
    mono=True,                   # Convert to mono (False = stereo)
    offset=0.0,                  # Start time in seconds
    duration=None,               # Duration to load in seconds (None = all)
    dtype=np.float32            # Output data type
)
```

#### Supported Formats

```python
# WAV, FLAC (native support, fastest)
audio, sr = load("audio.wav")

# MP3, AAC, M4A, OGG (via PyAV)
audio, sr = load("audio.mp3")
audio, sr = load("audio.aac")
audio, sr = load("audio.m4a")
audio, sr = load("audio.ogg")
```

#### Performance Expectations

```python
import time
from audiodecode import load

# Typical performance on Linux
start = time.time()
audio, sr = load("large_file.mp3", sr=16000)
elapsed = time.time() - start

print(f"Loaded in {elapsed*1000:.1f}ms")
# Expected: 20-50ms (vs 5-6 seconds with librosa)
```

#### Common Use Cases

**ML Training Data Loading**
```python
from audiodecode import load
import numpy as np

def load_training_batch(file_paths):
    """Fast batch loading for ML training"""
    audios = []
    for path in file_paths:
        audio, sr = load(path, sr=16000, mono=True)
        audios.append(audio)
    return np.array(audios)

# 10-100x faster than librosa for batch loading
batch = load_training_batch(training_files[:32])
```

**Audio Preprocessing Pipeline**
```python
from audiodecode import load
import librosa  # Still use librosa for features

def extract_features(file_path):
    """Fast loading + feature extraction"""
    # Fast load (180x speedup)
    audio, sr = load(file_path, sr=22050)

    # Extract features (use librosa as usual)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

    return {
        'mfcc': mfcc,
        'spectral_centroid': spectral_centroid
    }
```

**Segment Extraction**
```python
from audiodecode import load

# Load only a specific segment (fast)
audio_segment, sr = load(
    "long_podcast.mp3",
    sr=16000,
    offset=60.0,      # Start at 1 minute
    duration=10.0     # Load 10 seconds
)
```

#### OOP API (AudioDecoder)

```python
from audiodecode import AudioDecoder

# Create decoder instance
decoder = AudioDecoder(
    "audio.mp3",
    target_sr=16000,
    mono=True,
    output_format="numpy"  # or "torch", "jax", "bytes"
)

# Get file info without decoding (fast)
info = decoder.info()
print(f"Duration: {info['duration']:.1f}s")
print(f"Sample rate: {info['sample_rate']}Hz")
print(f"Channels: {info['channels']}")

# Decode audio
audio = decoder.decode()

# Decode specific segment
audio_segment = decoder.decode(offset=10.0, duration=5.0)

# Control caching
audio = decoder.decode(use_cache=True)  # Default
audio = decoder.decode(use_cache=False)  # Skip cache
```

#### Caching Control

```python
from audiodecode import load, clear_cache, set_cache_size, get_cache

# Load with caching (default)
audio1, sr = load("audio.mp3")  # First load: decodes
audio2, sr = load("audio.mp3")  # Second load: from cache (18x faster)

# Adjust cache size
set_cache_size(256)  # Store up to 256 files (default: 128)

# Get cache info
cache = get_cache()
info = cache.info()
print(f"Cached files: {info['entries']}")
print(f"Memory usage: {info['total_memory_mb']:.1f} MB")

# Clear cache
clear_cache()
```

---

### Feature 2: Speech-to-Text Inference

**When to use it:**
- Self-hosted transcription (vs Deepgram/AssemblyAI)
- Podcast/video transcription
- Meeting notes automation
- Batch audio processing
- Custom speech-to-text pipelines

**Performance comparison:**
- **CPU**: 1.8-6.0x faster than vanilla Whisper
- **GPU**: 2.4x faster (A10G tested)
- **RTF**: 43.8-108.3x realtime on GPU

**When NOT to use it:**
- Real-time streaming (future feature)
- Need highest accuracy at any cost (use large-v3 with GPU)
- Already using Deepgram API and happy with cost

#### Complete API Examples

**Simple Transcription**
```python
from audiodecode import transcribe_file

# Basic transcription
result = transcribe_file("podcast.mp3")
print(result.text)
# Output: "Welcome to our podcast about machine learning..."
```

**Full Parameters**
```python
from audiodecode import transcribe_file

result = transcribe_file(
    file_path="podcast.mp3",

    # Model selection
    model_size="base",              # tiny, base, small, medium, large-v3

    # Device & compute
    device="auto",                  # auto, cpu, cuda
    compute_type="auto",            # auto, int8, float16, float32

    # Language
    language=None,                  # None = auto-detect, or "en", "es", etc.
    task="transcribe",              # transcribe or translate (to English)

    # Quality controls
    beam_size=5,                    # Higher = better quality, slower
    best_of=5,                      # Number of candidates to consider
    patience=None,                  # Beam search patience (None = default)
    length_penalty=None,            # Favor longer/shorter segments
    repetition_penalty=None,        # Discourage repetition
    temperature=0.0,                # 0 = greedy, >0 = sampling

    # VAD (Voice Activity Detection)
    vad_filter=True,                # True, False, or "auto"
    vad_parameters=None,            # Custom VAD settings

    # Timestamps
    word_timestamps=False,          # Enable word-level timestamps

    # Prompt engineering
    initial_prompt=None,            # Guide with context/terminology
    condition_on_previous_text=True, # Use previous segments as context
    prefix=None,                    # Force first segment to start with text
    hotwords=None,                  # Boost recognition of specific words

    # Quality filters
    compression_ratio_threshold=None,  # Filter hallucinations (e.g., 2.4)
    logprob_threshold=None,            # Filter low-confidence (e.g., -1.0)
    no_speech_threshold=None,          # Filter silence (e.g., 0.6)

    # Output
    verbose=False                   # Print progress
)

# Access results
print(f"Text: {result.text}")
print(f"Language: {result.language}")
print(f"Duration: {result.duration:.1f}s")
print(f"Segments: {len(result.segments)}")
```

#### Configuration Options

**Model Size Selection**
```python
# Performance vs Accuracy tradeoff
models = {
    "tiny": "32x realtime, lowest accuracy, 39M params",
    "base": "16x realtime, good for most uses, 74M params",
    "small": "7x realtime, good accuracy, 244M params",
    "medium": "3x realtime, high accuracy, 769M params",
    "large-v3": "1-2x realtime, best accuracy, 1550M params"
}

# Example: Fast transcription
result = transcribe_file("audio.mp3", model_size="tiny")

# Example: High accuracy
result = transcribe_file("audio.mp3", model_size="large-v3")
```

**Language Detection vs Specification**
```python
# Auto-detect language (default)
result = transcribe_file("audio.mp3", language=None)
print(f"Detected: {result.language}")

# Force specific language (faster, no detection overhead)
result = transcribe_file("audio.mp3", language="en")
result = transcribe_file("audio.mp3", language="es")
result = transcribe_file("audio.mp3", language="zh")

# Translate to English
result = transcribe_file("spanish.mp3", task="translate")
# Output will be in English regardless of input language
```

**Batch Size Configuration**
```python
from audiodecode.inference import WhisperInference

# Create inference instance with custom batch size
whisper = WhisperInference(
    model_size="base",
    batch_size=16  # Default: 16 (optimal for most GPUs)
)

# Smaller batch size (less memory)
whisper = WhisperInference(model_size="large-v3", batch_size=8)

# Larger batch size (more throughput if you have VRAM)
whisper = WhisperInference(model_size="tiny", batch_size=32)
```

#### GPU vs CPU Usage

**Auto-Detection (Recommended)**
```python
# Automatically uses GPU if available
result = transcribe_file("audio.mp3", device="auto")
```

**Explicit Device Selection**
```python
# Force CPU (e.g., for development/testing)
result = transcribe_file("audio.mp3", device="cpu", compute_type="int8")

# Force GPU
result = transcribe_file("audio.mp3", device="cuda", compute_type="float16")
```

**Compute Type Selection**
```python
# Auto (recommended): int8 for CPU, float16 for GPU
result = transcribe_file("audio.mp3", compute_type="auto")

# CPU optimizations
result = transcribe_file("audio.mp3",
                        device="cpu",
                        compute_type="int8")  # Best CPU performance

# GPU optimizations
result = transcribe_file("audio.mp3",
                        device="cuda",
                        compute_type="float16")  # Best GPU performance
```

**Performance Expectations**
```python
# CPU (MacBook Pro M1)
# base model: ~18x realtime
# tiny model: ~32x realtime

# GPU (A10G)
# base model: ~108x realtime
# small model: ~43.8x realtime

# Example: 1-hour podcast on GPU (base model)
# Transcription time: ~33 seconds
```

#### Output Formats and Post-Processing

**TranscriptionResult Object**
```python
result = transcribe_file("audio.mp3")

# Full text
print(result.text)

# Segments with timestamps
for segment in result.segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s]")
    print(f"Text: {segment.text}")
    print(f"Confidence: {segment.confidence:.2f}")
    print()

# Metadata
print(f"Language: {result.language}")
print(f"Duration: {result.duration:.1f}s")
```

**Export Formats**
```python
# Save as different formats
result.save("transcript.txt")      # Plain text
result.save("subtitles.srt")       # SubRip subtitles
result.save("captions.vtt")        # WebVTT captions
result.save("data.json")           # JSON with full data

# Or get format as string
txt_content = result.text
srt_content = result.to_srt()
vtt_content = result.to_vtt()
json_content = result.to_json()
```

**SRT (SubRip) Format**
```python
result = transcribe_file("video.mp4")
srt = result.to_srt()

# Output:
# 1
# 00:00:00,000 --> 00:00:03,500
# Welcome to our podcast about machine learning.
#
# 2
# 00:00:03,500 --> 00:00:07,200
# Today we're discussing neural networks.

with open("subtitles.srt", "w") as f:
    f.write(srt)
```

**VTT (WebVTT) Format**
```python
result = transcribe_file("video.mp4")
vtt = result.to_vtt()

# Output:
# WEBVTT
#
# 00:00:00.000 --> 00:00:03.500
# Welcome to our podcast about machine learning.
#
# 00:00:03.500 --> 00:00:07.200
# Today we're discussing neural networks.

with open("captions.vtt", "w") as f:
    f.write(vtt)
```

**JSON Format**
```python
result = transcribe_file("audio.mp3")
json_data = result.to_json()

# Output:
# {
#   "text": "Full transcription text...",
#   "language": "en",
#   "duration": 123.45,
#   "segments": [
#     {
#       "text": "Welcome to our podcast",
#       "start": 0.0,
#       "end": 3.5,
#       "confidence": -0.23
#     },
#     ...
#   ]
# }
```

#### Batch Processing

**Process Multiple Files**
```python
from audiodecode import transcribe_batch

# Transcribe multiple files efficiently
files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]
results = transcribe_batch(
    files,
    model_size="base",
    show_progress=True  # Shows progress bar
)

# Process results
for i, result in enumerate(results):
    print(f"\nFile {i+1}: {files[i]}")
    print(f"Text: {result.text[:100]}...")
    print(f"Duration: {result.duration:.1f}s")
```

**Batch with Custom Settings**
```python
from audiodecode import transcribe_batch

results = transcribe_batch(
    file_paths=audio_files,
    model_size="base",
    language="en",           # Force English
    batch_size=16,           # Process 16 files in parallel
    device="cuda",
    show_progress=True,
    verbose=False            # Don't print each file's progress
)
```

**OOP API for Batch Processing**
```python
from audiodecode.inference import WhisperInference

# Create model instance once
whisper = WhisperInference(model_size="base", device="cuda")

# Reuse for multiple files (faster)
results = []
for audio_file in audio_files:
    result = whisper.transcribe_file(audio_file)
    results.append(result)
    print(f"Completed: {audio_file}")
```

---

### Feature 3: ML Training Optimization

**When to use it:**
- Training audio ML models (speech recognition, classification, etc.)
- Custom audio augmentation pipelines
- Large-scale audio preprocessing
- Any PyTorch training with audio data

#### DataLoader Usage Examples

**Basic Auto-Tuned DataLoader**
```python
from audiodecode import AudioDataLoader

# Zero-config, auto-tuned DataLoader
loader = AudioDataLoader(
    files=train_files,          # List of audio file paths
    labels=train_labels,        # List of labels (same length)
    batch_size=32,
    target_sr=16000,
    mono=True,
    device='cuda'               # Auto-tunes for GPU
)

# Training loop
for epoch in range(num_epochs):
    for batch, labels in loader:
        # batch: torch.Tensor of shape [32, samples]
        # labels: torch.Tensor of shape [32]

        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**All Parameters**
```python
from audiodecode import AudioDataLoader

loader = AudioDataLoader(
    # Data
    files=audio_files,              # List[str] or List[Path]
    labels=labels,                  # Optional[List]

    # Audio processing
    target_sr=16000,                # Target sample rate
    mono=True,                      # Convert to mono
    duration=None,                  # Fixed duration (None = full)
    offset=0.0,                     # Start offset in seconds

    # Transforms
    transform=augmentation_fn,      # Optional audio transform
    feature_extractor=mfcc_fn,      # Optional feature extraction

    # DataLoader settings
    batch_size=32,
    shuffle=True,
    drop_last=False,

    # Device
    device='cuda',                  # 'cpu', 'cuda', or torch.device

    # Auto-tuning (None = auto)
    num_workers=None,               # Auto-tunes based on CPU
    prefetch_factor=None,           # Auto-tunes based on RAM
    persistent_workers=None,        # Auto-enables if num_workers > 0

    # Caching
    cache_decoded=False,            # Cache decoded audio in RAM
    max_cache_size=None             # Max cached files
)
```

#### Auto-Tuning Behavior

**How Auto-Tuning Works**
```python
from audiodecode import AudioDataLoader

# Create loader (auto-tuning happens here)
loader = AudioDataLoader(
    files=train_files,
    labels=train_labels,
    batch_size=32,
    device='cuda'
)

# Inspect auto-tuned settings
config = loader.get_config()
print(config)

# Output:
# {
#   'num_workers': 6,           # Auto-tuned based on CPU cores
#   'prefetch_factor': 3,       # Auto-tuned based on batch size & RAM
#   'persistent_workers': True, # Auto-enabled
#   'pin_memory': True,         # Auto-enabled for CUDA
#   'batch_size': 32,
#   'device': 'cuda',
#   'num_files': 10000,
#   'num_batches': 313
# }

# Pretty print
loader.print_config()
```

**Platform-Specific Auto-Tuning**
```python
# Linux: Uses up to 3/4 of CPU cores (efficient for PyAV)
# loader on 16-core Linux: num_workers=8

# macOS: More conservative (Core Audio is efficient)
# loader on 8-core MacBook: num_workers=4

# Windows: Conservative (process spawning overhead)
# loader on 8-core Windows: num_workers=4
```

**Override Auto-Tuning**
```python
# Use custom settings instead of auto-tuning
loader = AudioDataLoader(
    files=train_files,
    labels=train_labels,
    batch_size=32,
    num_workers=4,              # Override auto-tuning
    prefetch_factor=2,          # Override auto-tuning
    persistent_workers=True,
    device='cuda'
)
```

#### Custom Transforms

**Audio Augmentation**
```python
import numpy as np
from audiodecode import AudioDataLoader

def augmentation(audio):
    """Custom augmentation function"""
    # Add Gaussian noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.005, audio.shape)
        audio = audio + noise

    # Time stretch (simplified example)
    if np.random.rand() > 0.5:
        indices = np.random.choice(len(audio), int(len(audio) * 0.95))
        audio = audio[sorted(indices)]

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    return audio

loader = AudioDataLoader(
    files=train_files,
    labels=train_labels,
    batch_size=32,
    transform=augmentation  # Applied to each sample
)
```

**Feature Extraction**
```python
import librosa
from audiodecode import AudioDataLoader

def extract_mfcc(audio):
    """Extract MFCC features"""
    # audio is already at target_sr
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    # Return flattened features
    return mfcc.T.flatten()

loader = AudioDataLoader(
    files=train_files,
    labels=train_labels,
    batch_size=32,
    target_sr=16000,
    feature_extractor=extract_mfcc
)

# Now batches contain MFCC features instead of raw audio
for features, labels in loader:
    # features: torch.Tensor of shape [32, num_features]
    outputs = model(features)
```

**Chaining Transforms**
```python
def augment_and_extract(audio):
    """Chain augmentation and feature extraction"""
    # 1. Augmentation
    if np.random.rand() > 0.5:
        audio = audio + np.random.normal(0, 0.005, audio.shape)

    # 2. Feature extraction
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)

    return mfcc.T.flatten()

loader = AudioDataLoader(
    files=train_files,
    labels=train_labels,
    feature_extractor=augment_and_extract
)
```

#### Caching Strategies

**When to Use Caching**
```python
# Use caching when:
# - Dataset is small enough to fit in RAM
# - Using heavy augmentation (cache before augment)
# - Multiple epochs over same data

from audiodecode import AudioDataLoader

# Without caching (default)
loader = AudioDataLoader(
    files=train_files,
    labels=train_labels,
    batch_size=32,
    cache_decoded=False  # Each epoch reloads from disk
)

# With caching (recommended for small datasets)
loader = AudioDataLoader(
    files=train_files,
    labels=train_labels,
    batch_size=32,
    cache_decoded=True,      # Cache in RAM
    max_cache_size=1000      # Limit to 1000 files
)
```

**Cache Size Management**
```python
import psutil
from audiodecode import AudioDataLoader

# Calculate cache size based on available RAM
available_ram_gb = psutil.virtual_memory().available / (1024**3)

# Rule of thumb: use 25% of available RAM for cache
# Assume ~1MB per audio file
max_cache_files = int(available_ram_gb * 0.25 * 1024)

loader = AudioDataLoader(
    files=train_files,
    labels=train_labels,
    cache_decoded=True,
    max_cache_size=max_cache_files
)
```

**Low-Level Cache Control**
```python
from audiodecode import AudioDatasetWithCache

# Create dataset with caching
dataset = AudioDatasetWithCache(
    files=train_files,
    labels=train_labels,
    target_sr=16000,
    cache_decoded=True,
    max_cache_size=500
)

# Check cache stats
print(f"Cached: {dataset.get_cache_size()} files")

# Clear cache
dataset.clear_cache()
```

#### Train/Val Split Helper

```python
from audiodecode import create_train_val_loaders

# Automatic train/val loader creation
train_loader, val_loader = create_train_val_loaders(
    train_files=train_files,
    train_labels=train_labels,
    val_files=val_files,
    val_labels=val_labels,
    batch_size=32,
    val_batch_size=64,      # Larger batch for validation
    target_sr=16000,
    device='cuda'
)

# Training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch, labels in train_loader:
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for batch, labels in val_loader:
            outputs = model(batch)
            # Validation logic
```

---

## Configuration Deep Dive

### Audio Loading Configuration

#### Sample Rate Control
```python
from audiodecode import load

# Native sample rate (fastest, no resampling)
audio, sr = load("audio.mp3", sr=None)
print(f"Native sample rate: {sr}Hz")

# Common sample rates
audio_16k, _ = load("audio.mp3", sr=16000)  # Whisper/Speech
audio_22k, _ = load("audio.mp3", sr=22050)  # Music/librosa
audio_44k, _ = load("audio.mp3", sr=44100)  # CD quality
audio_48k, _ = load("audio.mp3", sr=48000)  # Professional audio
```

#### Channel Control
```python
# Mono (recommended for ML/speech)
audio_mono, sr = load("stereo.mp3", mono=True)
print(audio_mono.shape)  # (samples,)

# Stereo (preserve channels)
audio_stereo, sr = load("stereo.mp3", mono=False)
print(audio_stereo.shape)  # (2, samples)
```

#### Segment Loading
```python
# Load specific time range (no need to load entire file)
segment, sr = load(
    "long_podcast.mp3",
    offset=60.0,      # Start at 1:00
    duration=30.0,    # Load 30 seconds
    sr=16000
)
# Much faster than loading entire file then slicing
```

### Inference Configuration

#### VAD (Voice Activity Detection) Configuration

**Smart VAD Mode (Recommended)**
```python
# "auto" mode: Uses VAD only for long audio (>60s)
result = transcribe_file("audio.mp3", vad_filter="auto")

# For short audio (<60s): skips VAD (faster)
# For long audio (>60s): uses VAD (better quality)
```

**Manual VAD Control**
```python
# Always use VAD (best quality, slower)
result = transcribe_file("audio.mp3", vad_filter=True)

# Never use VAD (faster, may include silence)
result = transcribe_file("audio.mp3", vad_filter=False)
```

**Custom VAD Parameters**
```python
# Fine-tune VAD sensitivity
vad_params = {
    'threshold': 0.5,          # Speech probability threshold
    'min_speech_duration_ms': 250,
    'min_silence_duration_ms': 2000
}

result = transcribe_file(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=vad_params
)
```

#### Beam Search Tuning

**Quality vs Speed Tradeoff**
```python
# Fast (lower quality)
result = transcribe_file("audio.mp3",
                        beam_size=1,      # Greedy search
                        best_of=1)        # Single candidate

# Balanced (default)
result = transcribe_file("audio.mp3",
                        beam_size=5,      # Good quality
                        best_of=5)

# Best quality (slower)
result = transcribe_file("audio.mp3",
                        beam_size=10,     # More thorough
                        best_of=10)
```

**Advanced Beam Search Parameters**
```python
result = transcribe_file(
    "audio.mp3",
    beam_size=5,
    patience=2.0,              # Higher = more thorough search
    length_penalty=1.0,        # Positive = favor longer segments
    repetition_penalty=1.2     # Discourage repetition
)
```

#### Temperature Fallback
```python
# Single temperature (deterministic)
result = transcribe_file("audio.mp3", temperature=0.0)

# Temperature fallback (retry on failure)
result = transcribe_file(
    "audio.mp3",
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
)
# Tries 0.0 first, falls back to 0.2 if fails, etc.
```

#### Quality Thresholds

**Filtering Low-Quality Segments**
```python
result = transcribe_file(
    "audio.mp3",

    # Filter hallucinations
    compression_ratio_threshold=2.4,

    # Filter low-confidence transcriptions
    logprob_threshold=-1.0,

    # Filter silent segments
    no_speech_threshold=0.6
)

# Segments failing these thresholds are excluded
```

### Performance Tuning Guide

#### CPU Optimization
```python
import os

# Set OpenMP threads before import (affects CTranslate2)
os.environ["OMP_NUM_THREADS"] = "6"  # Optimal for most CPUs

from audiodecode.inference import WhisperInference

whisper = WhisperInference(
    model_size="base",
    device="cpu",
    compute_type="int8",        # Best CPU performance
    batch_size=16               # Optimal for CPU
)
```

#### GPU Optimization
```python
from audiodecode.inference import WhisperInference

whisper = WhisperInference(
    model_size="base",
    device="cuda",
    compute_type="float16",     # Best GPU performance
    batch_size=16,              # Optimal for A10G (tested)
    use_batched_inference=True  # 2-3x speedup
)
```

#### Memory Management

**Audio Loading Memory**
```python
from audiodecode import set_cache_size

# Reduce cache size to save memory
set_cache_size(32)  # Only cache 32 files (default: 128)

# Disable cache entirely
set_cache_size(0)
```

**Inference Memory**
```python
# Use smaller model
whisper = WhisperInference(model_size="tiny")  # 39M params

# Reduce batch size
whisper = WhisperInference(model_size="base", batch_size=8)

# Use int8 quantization (CPU only)
whisper = WhisperInference(
    model_size="base",
    device="cpu",
    compute_type="int8"  # ~4x smaller than float32
)
```

### Device Selection

**Auto-Detection**
```python
from audiodecode.inference import WhisperInference

# Automatically uses GPU if available, else CPU
whisper = WhisperInference(device="auto")
```

**Explicit Selection**
```python
import torch

# Force CPU
whisper = WhisperInference(device="cpu")

# Force GPU
whisper = WhisperInference(device="cuda")

# Specific GPU
whisper = WhisperInference(device=torch.device("cuda:0"))
```

**Check Available Devices**
```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

---

## Common Workflows

### Workflow 1: Transcribing Podcasts/Videos

**Basic Podcast Transcription**
```python
from audiodecode import transcribe_file

# Transcribe podcast
result = transcribe_file(
    "podcast_episode.mp3",
    model_size="base",      # Good balance
    language="en",          # English podcast
    vad_filter="auto"       # Smart VAD
)

# Save multiple formats
result.save("transcript.txt")           # For reading
result.save("subtitles.srt")           # For video
result.save("data.json")               # For further processing

print(f"Transcribed {result.duration/60:.1f} minutes")
print(f"Word count: {len(result.text.split())}")
```

**Video Subtitles with Timestamps**
```python
from audiodecode import transcribe_file

# Transcribe video for subtitles
result = transcribe_file(
    "video.mp4",
    model_size="small",     # Better accuracy for subtitles
    language="en"
)

# Generate SRT subtitles
with open("subtitles.srt", "w", encoding="utf-8") as f:
    f.write(result.to_srt())

# Or VTT for web
with open("captions.vtt", "w", encoding="utf-8") as f:
    f.write(result.to_vtt())
```

**Long-Form Content (2+ hours)**
```python
from audiodecode import transcribe_file

# Transcribe long podcast/lecture
result = transcribe_file(
    "long_lecture.mp3",
    model_size="base",
    language="en",
    vad_filter=True,                    # Remove silence
    compression_ratio_threshold=2.4,    # Filter hallucinations
    no_speech_threshold=0.6,            # Filter silent segments
    verbose=True                        # Show progress
)

# Save with chapters (manual for now)
with open("transcript_with_chapters.txt", "w") as f:
    f.write("LECTURE TRANSCRIPT\n\n")

    current_chapter = 0
    for segment in result.segments:
        # Add chapter markers every 15 minutes
        timestamp = segment.start / 60
        if timestamp // 15 > current_chapter:
            current_chapter = int(timestamp // 15)
            f.write(f"\n\n=== CHAPTER {current_chapter} ({timestamp:.1f} min) ===\n\n")

        f.write(f"{segment.text} ")
```

### Workflow 2: Building Custom ML Pipelines

**Audio Classification Training**
```python
import torch
import torch.nn as nn
from audiodecode import create_train_val_loaders

# Define model
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=80, stride=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc = nn.Linear(64, num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# Create data loaders (auto-tuned)
train_loader, val_loader = create_train_val_loaders(
    train_files=train_files,
    train_labels=train_labels,
    val_files=val_files,
    val_labels=val_labels,
    batch_size=32,
    target_sr=16000,
    device='cuda'
)

# Training loop
model = AudioClassifier().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for batch, labels in train_loader:
        batch, labels = batch.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, labels in val_loader:
            batch, labels = batch.cuda(), labels.cuda()
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}: Val Acc = {100*correct/total:.2f}%")
```

**Feature Extraction Pipeline**
```python
from audiodecode import AudioDataLoader
import librosa
import numpy as np

def extract_features(audio):
    """Extract multiple audio features"""
    # Compute features
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=16000)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=16000)

    # Combine features
    features = np.concatenate([
        mfcc.mean(axis=1),
        chroma.mean(axis=1),
        spectral_contrast.mean(axis=1)
    ])

    return features

# Create loader with feature extraction
loader = AudioDataLoader(
    files=audio_files,
    labels=labels,
    batch_size=32,
    target_sr=16000,
    feature_extractor=extract_features
)

# Extract features for entire dataset
all_features = []
all_labels = []

for features, labels in loader:
    all_features.append(features.numpy())
    all_labels.append(labels.numpy())

features_array = np.vstack(all_features)
labels_array = np.concatenate(all_labels)

print(f"Extracted features: {features_array.shape}")

# Now use with sklearn, etc.
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(features_array, labels_array)
```

### Workflow 3: Batch Processing Large Audio Datasets

**Batch Transcription**
```python
from audiodecode import transcribe_batch
from pathlib import Path
import json

# Find all audio files
audio_dir = Path("audio_dataset")
audio_files = list(audio_dir.glob("**/*.mp3"))

print(f"Found {len(audio_files)} files")

# Batch transcribe (reuses model)
results = transcribe_batch(
    audio_files,
    model_size="base",
    language="en",
    device="cuda",
    show_progress=True,
    batch_size=16
)

# Save results
output_dir = Path("transcriptions")
output_dir.mkdir(exist_ok=True)

for audio_file, result in zip(audio_files, results):
    # Save transcript
    output_file = output_dir / f"{audio_file.stem}.txt"
    output_file.write_text(result.text)

    # Save metadata
    meta_file = output_dir / f"{audio_file.stem}.json"
    meta_file.write_text(json.dumps({
        'file': str(audio_file),
        'duration': result.duration,
        'language': result.language,
        'word_count': len(result.text.split()),
        'num_segments': len(result.segments)
    }, indent=2))

print(f"Completed {len(results)} transcriptions")
```

**Parallel Preprocessing**
```python
from audiodecode import load
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def preprocess_file(file_path):
    """Preprocess single audio file"""
    try:
        # Fast load
        audio, sr = load(file_path, sr=16000, mono=True)

        # Compute features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr)

        # Save processed
        output = file_path.with_suffix('.npy')
        np.save(output, mfcc)

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

# Find files
audio_files = list(Path("dataset").glob("*.mp3"))

# Parallel processing
with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(
        executor.map(preprocess_file, audio_files),
        total=len(audio_files),
        desc="Preprocessing"
    ))

print(f"Successfully processed: {sum(results)}/{len(audio_files)}")
```

### Workflow 4: Real-Time Audio Processing

**Stream Processing (Chunk-Based)**
```python
from audiodecode import load
import numpy as np

def process_audio_stream(file_path, chunk_duration=5.0):
    """Process audio in chunks for pseudo-real-time processing"""
    # Get file info first
    from audiodecode import AudioDecoder
    decoder = AudioDecoder(file_path, target_sr=16000)
    info = decoder.info()
    total_duration = info['duration']

    # Process in chunks
    offset = 0.0
    results = []

    while offset < total_duration:
        # Load chunk
        audio, sr = load(
            file_path,
            sr=16000,
            offset=offset,
            duration=chunk_duration
        )

        # Process chunk (your processing here)
        # Example: Compute RMS energy
        rms = np.sqrt(np.mean(audio**2))
        results.append({
            'time': offset,
            'rms': rms,
            'duration': len(audio) / sr
        })

        print(f"Processed chunk at {offset:.1f}s, RMS: {rms:.3f}")

        offset += chunk_duration

    return results

# Process long audio file in chunks
chunks = process_audio_stream("long_audio.mp3", chunk_duration=5.0)
```

---

## Advanced Usage

### Custom Audio Preprocessing

**Normalization**
```python
from audiodecode import load
import numpy as np

def load_and_normalize(file_path, target_rms=0.1):
    """Load audio and normalize to target RMS"""
    audio, sr = load(file_path, sr=16000, mono=True)

    # Compute current RMS
    current_rms = np.sqrt(np.mean(audio**2))

    # Normalize
    if current_rms > 0:
        audio = audio * (target_rms / current_rms)

    # Clip to prevent overflow
    audio = np.clip(audio, -1.0, 1.0)

    return audio, sr

audio, sr = load_and_normalize("audio.mp3")
```

**Silence Removal**
```python
from audiodecode import load
import numpy as np

def remove_silence(audio, sr, threshold=0.01, min_duration=0.1):
    """Remove silent sections from audio"""
    # Compute energy
    frame_length = int(sr * 0.02)  # 20ms frames
    hop_length = frame_length // 2

    energy = np.array([
        np.sum(audio[i:i+frame_length]**2)
        for i in range(0, len(audio)-frame_length, hop_length)
    ])

    # Find speech frames
    speech_frames = energy > threshold

    # Reconstruct audio
    speech_audio = []
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * hop_length
            end = start + hop_length
            speech_audio.append(audio[start:end])

    return np.concatenate(speech_audio) if speech_audio else audio

# Usage
audio, sr = load("audio_with_silence.mp3", sr=16000)
audio_clean = remove_silence(audio, sr)
```

### Integrating with PyTorch/TensorFlow

**PyTorch Integration**
```python
from audiodecode import AudioDataset
import torch
from torch.utils.data import DataLoader

# Custom collate function
def collate_variable_length(batch):
    """Handle variable-length audio"""
    audios, labels = zip(*batch)

    # Pad to max length in batch
    max_len = max(len(audio) for audio in audios)
    padded = torch.zeros(len(audios), max_len)

    for i, audio in enumerate(audios):
        padded[i, :len(audio)] = audio

    return padded, torch.tensor(labels)

# Create dataset
dataset = AudioDataset(
    files=audio_files,
    labels=labels,
    target_sr=16000,
    duration=None  # Variable length
)

# Create loader with custom collate
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_variable_length
)
```

**TensorFlow Integration**
```python
from audiodecode import load
import tensorflow as tf
import numpy as np

def audio_generator(file_paths, labels, target_sr=16000):
    """Generator for tf.data.Dataset"""
    for file_path, label in zip(file_paths, labels):
        audio, sr = load(file_path, sr=target_sr, mono=True)
        yield audio, label

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_generator(
    lambda: audio_generator(train_files, train_labels),
    output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# Add padding and batching
dataset = dataset.padded_batch(
    batch_size=32,
    padded_shapes=([None], [])
)

# Training
for batch, labels in dataset:
    # Your TensorFlow training code
    pass
```

### Prompt Engineering for Better Transcription

**Domain-Specific Terminology**
```python
from audiodecode import transcribe_file

# Medical transcription
result = transcribe_file(
    "medical_consultation.mp3",
    initial_prompt="""
    Medical terminology: electrocardiogram, myocardial infarction,
    hypertension, CT scan, MRI, diagnosis, treatment, prescription
    """,
    model_size="base"
)

# Technical podcast
result = transcribe_file(
    "tech_podcast.mp3",
    initial_prompt="""
    Tech terms: Kubernetes, PyTorch, TensorFlow, API, microservices,
    AWS, Docker, GPU, neural networks, transformers
    """
)

# Legal transcription
result = transcribe_file(
    "deposition.mp3",
    initial_prompt="""
    Legal terminology: plaintiff, defendant, objection, sustained,
    overruled, testimony, evidence, hearsay
    """
)
```

**Speaker Names and Formatting**
```python
# Use prefix to force specific formatting
result = transcribe_file(
    "interview.mp3",
    prefix="Interviewer:",
    initial_prompt="Interview with John Smith about climate change"
)

# For subsequent segments, use condition_on_previous_text
result = transcribe_file(
    "interview.mp3",
    initial_prompt="Interview format: Speaker name followed by colon",
    condition_on_previous_text=True
)
```

**Context for Better Accuracy**
```python
# Provide context about the audio content
result = transcribe_file(
    "lecture.mp3",
    initial_prompt="""
    University lecture on quantum mechanics, discussing wave-particle
    duality, Schrödinger equation, and quantum entanglement
    """,
    condition_on_previous_text=True
)
```

### Word-Level Timestamps

**Basic Word Timestamps**
```python
from audiodecode import transcribe_file

result = transcribe_file(
    "audio.mp3",
    word_timestamps=True  # Enable word-level timestamps
)

# Access word timestamps
for segment in result.segments:
    print(f"\nSegment: [{segment.start:.1f}s - {segment.end:.1f}s]")
    if segment.words:
        for word in segment.words:
            print(f"  {word.word}: {word.start:.2f}s - {word.end:.2f}s "
                  f"(confidence: {word.probability:.2f})")
```

**Karaoke-Style Subtitles**
```python
from audiodecode import transcribe_file

result = transcribe_file("song.mp3", word_timestamps=True)

# Generate karaoke format
with open("karaoke.txt", "w") as f:
    for segment in result.segments:
        if segment.words:
            for word in segment.words:
                f.write(f"[{word.start:.2f}] {word.word}\n")
```

**Word-Level Analysis**
```python
from audiodecode import transcribe_file
import pandas as pd

result = transcribe_file("speech.mp3", word_timestamps=True)

# Extract all words
words_data = []
for segment in result.segments:
    if segment.words:
        for word in segment.words:
            words_data.append({
                'word': word.word,
                'start': word.start,
                'end': word.end,
                'duration': word.end - word.start,
                'confidence': word.probability
            })

df = pd.DataFrame(words_data)

# Analysis
print(f"Total words: {len(df)}")
print(f"Average word duration: {df['duration'].mean():.3f}s")
print(f"Low confidence words (<0.5):")
print(df[df['confidence'] < 0.5])
```

### Language Detection

**Auto-Detection**
```python
from audiodecode import transcribe_file

# Detect language automatically
result = transcribe_file("unknown_language.mp3", language=None)

print(f"Detected language: {result.language}")
# Output: "en", "es", "fr", "de", "zh", etc.

# Language-specific processing
if result.language == "en":
    print("English transcription")
elif result.language == "es":
    print("Spanish transcription")
```

**Multi-Language Dataset Processing**
```python
from audiodecode import transcribe_batch
from collections import Counter

# Transcribe multiple files
files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]
results = transcribe_batch(files, language=None)  # Auto-detect

# Analyze languages
languages = Counter(r.language for r in results)
print(f"Language distribution: {languages}")

# Group by language
language_groups = {}
for file, result in zip(files, results):
    lang = result.language
    if lang not in language_groups:
        language_groups[lang] = []
    language_groups[lang].append((file, result))

# Process each language group
for lang, group in language_groups.items():
    print(f"\n{lang.upper()} files: {len(group)}")
    for file, result in group:
        print(f"  {file}: {result.text[:50]}...")
```

---

## Troubleshooting & FAQs

### Common Errors and Solutions

#### ImportError: faster-whisper not installed
```python
# Error:
# ImportError: faster-whisper is not installed

# Solution:
pip install audiodecode[inference]

# Or:
pip install faster-whisper
```

#### ImportError: torch not installed
```python
# Error:
# ImportError: PyTorch not installed

# Solution:
pip install audiodecode[torch]

# Or:
pip install torch
```

#### FileNotFoundError: Audio file not found
```python
# Error:
# FileNotFoundError: Audio file not found: audio.mp3

# Solution:
from pathlib import Path

# Check file exists
file_path = Path("audio.mp3")
if not file_path.exists():
    print(f"File not found: {file_path.absolute()}")

# Use absolute paths
from audiodecode import load
audio, sr = load("/absolute/path/to/audio.mp3")
```

#### CUDA out of memory
```python
# Error:
# RuntimeError: CUDA out of memory

# Solution 1: Reduce batch size
from audiodecode.inference import WhisperInference

whisper = WhisperInference(
    model_size="base",
    batch_size=8  # Reduce from default 16
)

# Solution 2: Use smaller model
whisper = WhisperInference(
    model_size="tiny",  # Instead of base/small
    device="cuda"
)

# Solution 3: Use CPU
whisper = WhisperInference(
    model_size="base",
    device="cpu",
    compute_type="int8"
)
```

#### Audio quality issues
```python
# Problem: Transcription quality is poor

# Solution 1: Use larger model
result = transcribe_file("audio.mp3", model_size="small")  # or "medium"

# Solution 2: Enable VAD
result = transcribe_file("audio.mp3", vad_filter=True)

# Solution 3: Adjust quality thresholds
result = transcribe_file(
    "audio.mp3",
    compression_ratio_threshold=2.4,  # Filter hallucinations
    logprob_threshold=-1.0,           # Filter low confidence
    no_speech_threshold=0.6           # Filter silence
)

# Solution 4: Use prompt engineering
result = transcribe_file(
    "audio.mp3",
    initial_prompt="Technical podcast about AI and machine learning"
)
```

### Performance Optimization Tips

**Audio Loading Performance**
```python
# Tip 1: Enable caching for repeated access
from audiodecode import set_cache_size
set_cache_size(256)  # Cache more files

# Tip 2: Use native sample rate when possible
audio, sr = load("audio.mp3", sr=None)  # No resampling

# Tip 3: Load only what you need
audio, sr = load("long_file.mp3", offset=60, duration=30)

# Tip 4: Use mono for speech/ML
audio, sr = load("audio.mp3", mono=True)  # Faster than stereo
```

**Inference Performance**
```python
# Tip 1: Use batched inference
from audiodecode.inference import WhisperInference

whisper = WhisperInference(
    model_size="base",
    use_batched_inference=True  # 2-3x speedup
)

# Tip 2: Disable VAD for short audio
result = transcribe_file("short.mp3", vad_filter=False)

# Tip 3: Use int8 quantization on CPU
whisper = WhisperInference(
    device="cpu",
    compute_type="int8"  # 4x smaller, faster
)

# Tip 4: Reuse model instance for multiple files
whisper = WhisperInference(model_size="base")
for file in files:
    result = whisper.transcribe_file(file)  # Reuses loaded model
```

**Training Performance**
```python
# Tip 1: Use auto-tuned DataLoader
from audiodecode import AudioDataLoader

loader = AudioDataLoader(
    files=files,
    labels=labels,
    device='cuda'  # Auto-tunes for your system
)

# Tip 2: Enable caching for small datasets
loader = AudioDataLoader(
    files=files,
    labels=labels,
    cache_decoded=True,
    max_cache_size=1000
)

# Tip 3: Use persistent workers
loader = AudioDataLoader(
    files=files,
    labels=labels,
    num_workers=4,
    persistent_workers=True  # Keep workers alive
)
```

### Memory Issues

**Reduce Memory Usage**
```python
# Audio loading
from audiodecode import set_cache_size

# Reduce or disable cache
set_cache_size(32)  # Small cache
set_cache_size(0)   # Disable cache

# Inference
from audiodecode.inference import WhisperInference

# Use smaller model
whisper = WhisperInference(model_size="tiny")

# Reduce batch size
whisper = WhisperInference(batch_size=8)

# Use int8 quantization
whisper = WhisperInference(compute_type="int8")

# Training
from audiodecode import AudioDataLoader

# Disable caching
loader = AudioDataLoader(files=files, labels=labels, cache_decoded=False)

# Reduce workers
loader = AudioDataLoader(files=files, labels=labels, num_workers=2)
```

**Monitor Memory Usage**
```python
import psutil

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024**2:.1f} MB")

# Monitor during processing
print_memory_usage()
result = transcribe_file("audio.mp3")
print_memory_usage()
```

### GPU Setup Problems

**Check CUDA Installation**
```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available - using CPU")
```

**Force CPU if GPU Issues**
```python
from audiodecode import transcribe_file

# Force CPU even if GPU available
result = transcribe_file("audio.mp3", device="cpu")
```

**Install CUDA-Enabled PyTorch**
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### FAQs

**Q: Is AudioDecode compatible with librosa?**

A: Yes! The `load()` function is a drop-in replacement:
```python
# Old code
import librosa
audio, sr = librosa.load("audio.mp3", sr=16000, mono=True)

# New code (just change import)
from audiodecode import load
audio, sr = load("audio.mp3", sr=16000, mono=True)
```

**Q: Can I use AudioDecode with existing PyTorch code?**

A: Yes! AudioDataset works like any PyTorch Dataset:
```python
from torch.utils.data import DataLoader
from audiodecode import AudioDataset

dataset = AudioDataset(files, labels, target_sr=16000)
loader = DataLoader(dataset, batch_size=32)  # Standard PyTorch
```

**Q: What's the difference between transcribe_file and WhisperInference?**

A: `transcribe_file()` is a convenience function for one-off transcriptions. `WhisperInference` is for advanced usage and batch processing:
```python
# Simple: one-off transcription
from audiodecode import transcribe_file
result = transcribe_file("audio.mp3")

# Advanced: batch processing, reuse model
from audiodecode.inference import WhisperInference
whisper = WhisperInference(model_size="base")
results = [whisper.transcribe_file(f) for f in files]
```

**Q: How accurate is the transcription?**

A: AudioDecode uses faster-whisper, which is as accurate as OpenAI Whisper:
- Same models (tiny, base, small, medium, large)
- Same accuracy
- Just faster (2-6x speedup)

**Q: Can I use custom audio formats?**

A: AudioDecode supports all formats that FFmpeg supports:
- WAV, FLAC (native, fastest)
- MP3, AAC, M4A, OGG (via PyAV)
- Any format FFmpeg can decode

**Q: Is AudioDecode production-ready?**

A: Yes! It's used in production for:
- Speech-to-text services
- ML training pipelines
- Audio preprocessing
- Batch transcription jobs

---

## Migration Guides

### Migrating from librosa

**Step 1: Replace load() calls**
```python
# Before
import librosa
audio, sr = librosa.load("audio.mp3", sr=16000, mono=True)

# After
from audiodecode import load
audio, sr = load("audio.mp3", sr=16000, mono=True)
```

**Step 2: Keep librosa for features**
```python
from audiodecode import load
import librosa

# Fast loading with audiodecode
audio, sr = load("audio.mp3", sr=22050)

# Feature extraction with librosa (unchanged)
mfcc = librosa.feature.mfcc(y=audio, sr=sr)
chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
```

**Step 3: Update batch loading**
```python
# Before (slow)
import librosa
audios = [librosa.load(f, sr=16000)[0] for f in files]

# After (fast)
from audiodecode import load
audios = [load(f, sr=16000)[0] for f in files]
```

**Complete Migration Example**
```python
# BEFORE: librosa
import librosa
import numpy as np

def extract_features_old(file_path):
    # Slow: librosa.load() spawns FFmpeg
    audio, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    return mfcc

# AFTER: audiodecode + librosa
from audiodecode import load  # Only change!
import librosa
import numpy as np

def extract_features_new(file_path):
    # Fast: audiodecode.load() uses native libraries
    audio, sr = load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    return mfcc

# 180x faster on Linux, same output!
```

### Migrating from openai-whisper

**Step 1: Install audiodecode**
```bash
pip install audiodecode[inference]
```

**Step 2: Replace transcribe() calls**
```python
# Before
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
text = result["text"]

# After
from audiodecode import transcribe_file
result = transcribe_file("audio.mp3", model_size="base")
text = result.text
```

**Step 3: Update word timestamps**
```python
# Before
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.mp3", word_timestamps=True)
for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']}: {word['start']}-{word['end']}")

# After
from audiodecode import transcribe_file
result = transcribe_file("audio.mp3", word_timestamps=True)
for segment in result.segments:
    for word in segment.words:
        print(f"{word.word}: {word.start}-{word.end}")
```

**Step 4: Migrate advanced features**
```python
# Before
result = model.transcribe(
    "audio.mp3",
    language="en",
    initial_prompt="Technical podcast",
    temperature=(0.0, 0.2, 0.4),
    word_timestamps=True
)

# After
result = transcribe_file(
    "audio.mp3",
    language="en",
    initial_prompt="Technical podcast",
    temperature=(0.0, 0.2, 0.4),
    word_timestamps=True
)
```

**Complete Migration Example**
```python
# BEFORE: openai-whisper
import whisper

model = whisper.load_model("base")
result = model.transcribe("podcast.mp3")

with open("transcript.txt", "w") as f:
    f.write(result["text"])

# AFTER: audiodecode (2-6x faster)
from audiodecode import transcribe_file

result = transcribe_file("podcast.mp3", model_size="base")
result.save("transcript.txt")

# Even easier with built-in export!
result.save("subtitles.srt")
result.save("captions.vtt")
result.save("data.json")
```

### Migrating from faster-whisper

**Step 1: Replace imports**
```python
# Before
from faster_whisper import WhisperModel
import librosa

# After
from audiodecode.inference import WhisperInference
from audiodecode import load  # Faster than librosa
```

**Step 2: Replace model initialization**
```python
# Before
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8")

# After
from audiodecode.inference import WhisperInference
model = WhisperInference(model_size="base", device="cpu", compute_type="int8")
```

**Step 3: Replace transcription**
```python
# Before
from faster_whisper import WhisperModel
import librosa

model = WhisperModel("base")
audio, sr = librosa.load("audio.mp3", sr=16000)  # Slow
segments, info = model.transcribe(audio)
text = " ".join([seg.text for seg in segments])

# After
from audiodecode import transcribe_file

result = transcribe_file("audio.mp3", model_size="base")
text = result.text  # Much simpler!
```

**Complete Migration Example**
```python
# BEFORE: faster-whisper + librosa
from faster_whisper import WhisperModel
import librosa

# Load model
model = WhisperModel("base", device="cpu", compute_type="int8")

# Process files
for audio_file in audio_files:
    # Slow: librosa uses subprocess
    audio, sr = librosa.load(audio_file, sr=16000)

    # Transcribe
    segments, info = model.transcribe(audio)

    # Extract text
    text = " ".join([seg.text for seg in segments])

    # Save
    output_file = audio_file.replace(".mp3", ".txt")
    with open(output_file, "w") as f:
        f.write(text)

# AFTER: audiodecode (simpler + faster)
from audiodecode import transcribe_batch

# Batch transcribe (reuses model, fast loading)
results = transcribe_batch(
    audio_files,
    model_size="base",
    device="cpu",
    compute_type="int8"
)

# Save results
for audio_file, result in zip(audio_files, results):
    output_file = audio_file.replace(".mp3", ".txt")
    result.save(output_file)  # Built-in save method!

# 180x faster loading + cleaner code!
```

---

## Next Steps

### Learn More

- **API Reference**: Complete documentation of all functions and classes
- **Benchmarks**: Performance comparisons with librosa and vanilla Whisper
- **Examples**: Additional real-world examples in `/examples` directory
- **GitHub**: Source code, issues, and discussions

### Get Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share use cases
- **Documentation**: Additional guides and tutorials

### Contribute

AudioDecode is open source! Contributions welcome:
- Bug fixes and improvements
- New features and optimizations
- Documentation and examples
- Performance benchmarks

---

**AudioDecode: The Complete Audio Foundation Layer**

*Fast. Batteries-included. Production-ready.*
