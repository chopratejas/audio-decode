"""
Test Developer UX and Friction Points
======================================

This script tests the actual developer experience of using AudioDecode
from a fresh perspective. Tests:
1. Import speed
2. API intuitiveness
3. Error messages
4. Edge cases
5. Real-world usage patterns
"""

import sys
import os
sys.path.insert(0, "src")

import time
from pathlib import Path


def test_import_speed():
    """Test: How fast is the import?"""
    print("\n" + "="*70)
    print("TEST 1: Import Speed")
    print("="*70)

    start = time.perf_counter()
    from audiodecode import AudioDecoder
    import_time = time.perf_counter() - start

    print(f"Import time: {import_time*1000:.2f}ms")
    if import_time > 0.1:
        print(f"WARNING: Import is slow! ({import_time*1000:.0f}ms)")
    return import_time < 0.1


def test_simple_usage():
    """Test: Can a developer use it with zero docs?"""
    print("\n" + "="*70)
    print("TEST 2: Simple Usage (Zero Documentation)")
    print("="*70)

    from audiodecode import AudioDecoder

    # Test 1: Most basic usage
    try:
        audio = AudioDecoder("fixtures/audio/wav_1s_mono_16000.mp3").decode()
        print(f"Basic decode: SUCCESS - shape {audio.shape}")
    except Exception as e:
        print(f"Basic decode: FAILED - {e}")
        return False

    # Test 2: With parameters (can user guess the API?)
    try:
        audio = AudioDecoder("fixtures/audio/wav_1s_mono_16000.mp3",
                            target_sr=16000,
                            mono=True).decode()
        print(f"With params: SUCCESS - shape {audio.shape}")
    except Exception as e:
        print(f"With params: FAILED - {e}")
        return False

    return True


def test_error_messages():
    """Test: Are error messages helpful?"""
    print("\n" + "="*70)
    print("TEST 3: Error Messages Quality")
    print("="*70)

    from audiodecode import AudioDecoder

    # Test 1: Non-existent file
    print("\n1. Non-existent file:")
    try:
        audio = AudioDecoder("nonexistent.mp3").decode()
        print("   FAILED: Should have raised error!")
        return False
    except Exception as e:
        print(f"   Error message: {e}")
        if "not found" in str(e).lower() or "exist" in str(e).lower():
            print("   GOOD: Clear error message")
        else:
            print("   BAD: Unclear error message")

    # Test 2: Invalid format
    print("\n2. Invalid file format:")
    # Create a dummy file
    Path("test_invalid.xyz").write_text("not audio")
    try:
        audio = AudioDecoder("test_invalid.xyz").decode()
        print("   FAILED: Should have raised error!")
        Path("test_invalid.xyz").unlink()
        return False
    except Exception as e:
        print(f"   Error message: {e}")
        if "format" in str(e).lower() or "support" in str(e).lower():
            print("   GOOD: Clear error message")
        else:
            print("   BAD: Error message could be clearer")
        Path("test_invalid.xyz").unlink()

    return True


def test_api_discoverability():
    """Test: Can users discover features via IDE autocomplete?"""
    print("\n" + "="*70)
    print("TEST 4: API Discoverability")
    print("="*70)

    from audiodecode import AudioDecoder
    import inspect

    # What methods are available?
    decoder = AudioDecoder("fixtures/audio/wav_1s_mono_16000.mp3")
    public_methods = [m for m in dir(decoder) if not m.startswith('_')]
    print(f"\nPublic methods: {public_methods}")

    # Are docstrings present?
    print(f"\nDecoder class docstring:")
    print(f"  {AudioDecoder.__doc__[:100] if AudioDecoder.__doc__ else 'MISSING!'}")

    print(f"\ndecode() method docstring:")
    print(f"  {decoder.decode.__doc__[:100] if decoder.decode.__doc__ else 'MISSING!'}")

    # Check __init__ signature
    sig = inspect.signature(AudioDecoder.__init__)
    print(f"\n__init__ signature: {sig}")

    return True


def test_edge_cases():
    """Test: How does it handle edge cases?"""
    print("\n" + "="*70)
    print("TEST 5: Edge Cases")
    print("="*70)

    from audiodecode import AudioDecoder, clear_cache

    # Test 1: Very small file
    print("\n1. Very small audio file (1s):")
    try:
        audio = AudioDecoder("fixtures/audio/wav_1s_mono_16000.mp3").decode()
        print(f"   SUCCESS: {audio.shape}")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Test 2: Stereo to mono conversion
    print("\n2. Stereo to mono conversion:")
    try:
        audio = AudioDecoder("fixtures/audio/wav_1s_stereo_44100.wav", mono=True).decode()
        print(f"   SUCCESS: {audio.shape} (should be 1D)")
        if audio.ndim != 1:
            print(f"   WARNING: Expected 1D, got {audio.ndim}D")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Test 3: Resampling
    print("\n3. Resampling (44100 -> 16000):")
    try:
        audio = AudioDecoder("fixtures/audio/wav_1s_stereo_44100.wav", target_sr=16000).decode()
        print(f"   SUCCESS: {audio.shape}")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Test 4: Cache behavior
    print("\n4. Cache behavior:")
    try:
        clear_cache()
        file_path = "fixtures/audio/wav_1s_mono_16000.mp3"

        start = time.perf_counter()
        audio1 = AudioDecoder(file_path).decode(use_cache=False)
        no_cache_time = time.perf_counter() - start

        start = time.perf_counter()
        audio2 = AudioDecoder(file_path).decode(use_cache=True)
        cache_miss_time = time.perf_counter() - start

        start = time.perf_counter()
        audio3 = AudioDecoder(file_path).decode(use_cache=True)
        cache_hit_time = time.perf_counter() - start

        print(f"   No cache: {no_cache_time*1000:.3f}ms")
        print(f"   Cache miss: {cache_miss_time*1000:.3f}ms")
        print(f"   Cache hit: {cache_hit_time*1000:.3f}ms")

        if cache_hit_time < cache_miss_time * 0.5:
            print(f"   SUCCESS: Cache is {cache_miss_time/cache_hit_time:.1f}x faster")
        else:
            print(f"   WARNING: Cache benefit unclear")

    except Exception as e:
        print(f"   FAILED: {e}")

    return True


def test_real_world_patterns():
    """Test: Common real-world usage patterns"""
    print("\n" + "="*70)
    print("TEST 6: Real-World Usage Patterns")
    print("="*70)

    from audiodecode import AudioDecoder
    import numpy as np

    # Pattern 1: Batch processing loop
    print("\n1. Batch processing loop (10 files):")
    try:
        files = ["fixtures/audio/wav_1s_mono_16000.mp3"] * 10
        start = time.perf_counter()
        audios = [AudioDecoder(f).decode() for f in files]
        batch_time = time.perf_counter() - start
        print(f"   SUCCESS: {len(audios)} files in {batch_time*1000:.2f}ms")
        print(f"   Per file: {batch_time/len(files)*1000:.2f}ms")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Pattern 2: Process and pass to librosa
    print("\n2. Integration with librosa (feature extraction):")
    try:
        import librosa
        audio = AudioDecoder("fixtures/audio/wav_1s_mono_16000.mp3",
                            target_sr=16000,
                            mono=True).decode()
        # Compute MFCC on AudioDecode output
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
        print(f"   SUCCESS: MFCCs shape {mfccs.shape}")
    except ImportError:
        print("   SKIPPED: librosa not installed")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Pattern 3: Multiple output formats
    print("\n3. Different output formats:")
    try:
        # NumPy (default)
        audio_np = AudioDecoder("fixtures/audio/wav_1s_mono_16000.mp3").decode()
        print(f"   NumPy: {type(audio_np).__name__} - {audio_np.shape}")

        # Try PyTorch if available
        try:
            audio_torch = AudioDecoder("fixtures/audio/wav_1s_mono_16000.mp3",
                                      output_format="torch").decode()
            print(f"   PyTorch: {type(audio_torch).__name__} - {audio_torch.shape}")
        except ImportError:
            print("   PyTorch: SKIPPED (not installed)")

    except Exception as e:
        print(f"   FAILED: {e}")

    return True


def test_installation_friction():
    """Test: What dependencies are actually needed?"""
    print("\n" + "="*70)
    print("TEST 7: Installation Friction")
    print("="*70)

    # Check what's imported
    print("\nRequired dependencies:")
    required = ["numpy", "soundfile", "av"]  # PyAV

    for pkg in required:
        try:
            __import__(pkg)
            print(f"  {pkg}: INSTALLED")
        except ImportError:
            print(f"  {pkg}: MISSING (REQUIRED)")

    # Optional dependencies
    print("\nOptional dependencies:")
    optional = ["torch", "jax"]

    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  {pkg}: INSTALLED")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED (optional)")

    # Rust extension
    print("\nRust extension:")
    try:
        from audiodecode._rust import batch_decode
        print("  _rust: AVAILABLE")
    except ImportError:
        print("  _rust: NOT AVAILABLE (optional)")

    return True


def main():
    """Run all UX tests"""
    print("="*70)
    print("DEVELOPER UX EVALUATION")
    print("="*70)

    tests = [
        ("Import Speed", test_import_speed),
        ("Simple Usage", test_simple_usage),
        ("Error Messages", test_error_messages),
        ("API Discoverability", test_api_discoverability),
        ("Edge Cases", test_edge_cases),
        ("Real-World Patterns", test_real_world_patterns),
        ("Installation", test_installation_friction),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nTEST CRASHED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nTests passed: {passed}/{total}")

    print("\nFriction points identified:")
    print("- TODO: List any issues found during testing")


if __name__ == "__main__":
    main()
