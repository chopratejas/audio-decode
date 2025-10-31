/// High-performance audio decoding with Rust + Symphonia
///
/// This module provides parallel batch processing and optimized FLAC decoding
/// that complements the Python PyAV backend.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArray1, ToPyArray};
use rayon::prelude::*;

mod decoder;
mod resampler;

use decoder::AudioDecoder;

/// Decode a single audio file to numpy array
///
/// This releases the GIL during decoding for true parallelism.
///
/// Args:
///     filepath: Path to audio file
///     target_sr: Target sample rate (None = keep original)
///     mono: Convert to mono if True
///
/// Returns:
///     numpy.ndarray: Audio as float32 array
#[pyfunction]
#[pyo3(signature = (filepath, target_sr=None, mono=false))]
fn decode<'py>(
    py: Python<'py>,
    filepath: &str,
    target_sr: Option<u32>,
    mono: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    // Release GIL during CPU-intensive work
    let samples: Vec<f32> = py.allow_threads(|| {
        let decoder = AudioDecoder::new(filepath)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open file: {}", e)))?;

        decoder.decode(target_sr, mono)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to decode: {}", e)))
    })?;

    // Convert to numpy (zero-copy)
    Ok(samples.to_pyarray_bound(py))
}

/// Decode multiple audio files in parallel
///
/// This is where the magic happens! We decode files simultaneously across
/// multiple CPU cores, dramatically speeding up batch processing.
///
/// Args:
///     filepaths: List of file paths
///     target_sr: Target sample rate (None = keep original)
///     mono: Convert to mono if True
///     num_workers: Number of parallel workers (None = auto-detect)
///
/// Returns:
///     List[numpy.ndarray]: List of audio arrays
#[pyfunction]
#[pyo3(signature = (filepaths, target_sr=None, mono=false, num_workers=None))]
fn batch_decode<'py>(
    py: Python<'py>,
    filepaths: Vec<String>,
    target_sr: Option<u32>,
    mono: bool,
    num_workers: Option<usize>,
) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
    // Configure thread pool
    if let Some(n) = num_workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    // Decode all files in parallel (GIL released!)
    let results: Vec<Result<Vec<f32>, String>> = py.allow_threads(|| {
        filepaths.par_iter()
            .map(|path| {
                let decoder = AudioDecoder::new(path)
                    .map_err(|e| format!("Failed to open {}: {}", path, e))?;

                decoder.decode(target_sr, mono)
                    .map_err(|e| format!("Failed to decode {}: {}", path, e))
            })
            .collect()
    });

    // Convert results to numpy arrays (GIL reacquired)
    results.into_iter()
        .map(|r| {
            let samples = r.map_err(PyRuntimeError::new_err)?;
            Ok(samples.to_pyarray_bound(py))
        })
        .collect()
}

/// Get audio file metadata without decoding
///
/// Args:
///     filepath: Path to audio file
///
/// Returns:
///     dict: {"sample_rate": int, "channels": int, "duration": float, "samples": int}
#[pyfunction]
fn get_info(filepath: &str) -> PyResult<(u32, u32, f64, u64)> {
    let decoder = AudioDecoder::new(filepath)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open file: {}", e)))?;

    let info = decoder.get_info()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get info: {}", e)))?;

    Ok((info.sample_rate, info.channels, info.duration, info.samples))
}

/// Python module definition
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(batch_decode, m)?)?;
    m.add_function(wrap_pyfunction!(get_info, m)?)?;
    Ok(())
}
