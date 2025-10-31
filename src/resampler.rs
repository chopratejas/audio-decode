/// High-quality audio resampling using rubato
///
/// This provides better quality than scipy's resample and is faster than soxr in many cases.

use rubato::{
    Resampler as RubatoResampler, SincFixedIn, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ResampleError {
    #[error("Rubato error: {0}")]
    Rubato(String),

    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
}

pub struct Resampler {
    resampler: SincFixedIn<f32>,
    channels: usize,
}

impl Resampler {
    /// Create a new resampler
    ///
    /// Args:
    ///     from_sr: Source sample rate
    ///     to_sr: Target sample rate
    ///     channels: Number of audio channels
    pub fn new(from_sr: u32, to_sr: u32, channels: usize) -> Result<Self, ResampleError> {
        if from_sr == 0 || to_sr == 0 {
            return Err(ResampleError::InvalidParams(
                "Sample rates must be > 0".to_string(),
            ));
        }

        if channels == 0 {
            return Err(ResampleError::InvalidParams(
                "Number of channels must be > 0".to_string(),
            ));
        }

        // High-quality resampling parameters
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let resample_ratio = to_sr as f64 / from_sr as f64;

        // We don't know input length ahead of time, so use a reasonable chunk size
        let chunk_size = 4096;

        let resampler = SincFixedIn::<f32>::new(
            resample_ratio,
            2.0,
            params,
            chunk_size,
            channels,
        )
        .map_err(|e| ResampleError::Rubato(e.to_string()))?;

        Ok(Self {
            resampler,
            channels,
        })
    }

    /// Process audio samples
    ///
    /// Args:
    ///     samples: Interleaved audio samples [L, R, L, R, ...]
    ///
    /// Returns:
    ///     Resampled audio samples
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, ResampleError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Convert interleaved to separate channels
        let num_frames = samples.len() / self.channels;
        let mut channels_in: Vec<Vec<f32>> = vec![Vec::with_capacity(num_frames); self.channels];

        for (frame_idx, frame) in samples.chunks_exact(self.channels).enumerate() {
            for (ch_idx, &sample) in frame.iter().enumerate() {
                channels_in[ch_idx].push(sample);
            }
        }

        // Resample each channel
        // Note: For production, we'd want to handle longer audio by chunking,
        // but for typical ML audio files (<30s), this is fine
        let channels_out = self
            .resampler
            .process(&channels_in, None)
            .map_err(|e| ResampleError::Rubato(e.to_string()))?;

        // Convert back to interleaved
        let out_frames = channels_out[0].len();
        let mut interleaved = Vec::with_capacity(out_frames * self.channels);

        for frame_idx in 0..out_frames {
            for ch in 0..self.channels {
                interleaved.push(channels_out[ch][frame_idx]);
            }
        }

        Ok(interleaved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_resampling() {
        // Simple sine wave at 1kHz
        let from_sr = 44100;
        let to_sr = 16000;
        let channels = 1;

        let duration = 1.0; // 1 second
        let num_samples = (from_sr as f64 * duration) as usize;

        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f32 / from_sr as f32;
            samples.push((2.0 * std::f32::consts::PI * 1000.0 * t).sin());
        }

        let resampler = Resampler::new(from_sr, to_sr, channels).unwrap();
        let resampled = resampler.process(&samples).unwrap();

        // Check output length is approximately correct
        let expected_len = (num_samples as f64 * to_sr as f64 / from_sr as f64) as usize;
        assert!((resampled.len() as isize - expected_len as isize).abs() < 100);
    }

    #[test]
    fn test_stereo_resampling() {
        let from_sr = 48000;
        let to_sr = 16000;
        let channels = 2;

        // Generate stereo test signal (interleaved)
        let num_frames = 48000;
        let mut samples = Vec::with_capacity(num_frames * 2);

        for i in 0..num_frames {
            let t = i as f32 / from_sr as f32;
            // Left: 440 Hz
            samples.push((2.0 * std::f32::consts::PI * 440.0 * t).sin());
            // Right: 880 Hz
            samples.push((2.0 * std::f32::consts::PI * 880.0 * t).sin());
        }

        let resampler = Resampler::new(from_sr, to_sr, channels).unwrap();
        let resampled = resampler.process(&samples).unwrap();

        // Check stereo output
        assert!(resampled.len() % 2 == 0);
        let expected_frames = (num_frames as f64 * to_sr as f64 / from_sr as f64) as usize;
        assert!((resampled.len() / 2 - expected_frames).abs() < 100);
    }
}
