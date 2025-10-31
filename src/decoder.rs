/// Symphonia-based audio decoder
///
/// This provides high-performance audio decoding using the Symphonia pure-Rust library.

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

use crate::resampler::Resampler;

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Symphonia error: {0}")]
    Symphonia(#[from] SymphoniaError),

    #[error("No default track found")]
    NoTrack,

    #[error("Unsupported format")]
    UnsupportedFormat,

    #[error("Resampling error: {0}")]
    Resampling(String),
}

pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: u32,
    pub duration: f64,
    pub samples: u64,
}

pub struct AudioDecoder {
    filepath: String,
}

impl AudioDecoder {
    /// Create a new decoder for the given file
    pub fn new(filepath: &str) -> Result<Self, DecodeError> {
        // Verify file exists
        if !Path::new(filepath).exists() {
            return Err(DecodeError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", filepath),
            )));
        }

        Ok(Self {
            filepath: filepath.to_string(),
        })
    }

    /// Get audio file metadata without decoding
    pub fn get_info(&self) -> Result<AudioInfo, DecodeError> {
        let file = File::open(&self.filepath)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Probe format
        let mut hint = Hint::new();
        if let Some(ext) = Path::new(&self.filepath).extension() {
            hint.with_extension(ext.to_str().unwrap_or(""));
        }

        let probed = symphonia::default::get_probe().format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?;

        let track = probed
            .format
            .default_track()
            .ok_or(DecodeError::NoTrack)?;

        let codec_params = &track.codec_params;

        let sample_rate = codec_params.sample_rate.unwrap_or(0);
        let channels = codec_params.channels.map(|c| c.count()).unwrap_or(0) as u32;

        // Calculate duration
        let samples = codec_params.n_frames.unwrap_or(0);
        let duration = if sample_rate > 0 {
            samples as f64 / sample_rate as f64
        } else {
            0.0
        };

        Ok(AudioInfo {
            sample_rate,
            channels,
            duration,
            samples,
        })
    }

    /// Decode audio file to Vec<f32> samples
    ///
    /// This is the core decoding function that:
    /// 1. Opens the file and probes the format
    /// 2. Creates a decoder for the audio codec
    /// 3. Decodes all packets to PCM samples
    /// 4. Optionally resamples to target sample rate
    /// 5. Optionally converts to mono
    pub fn decode(&self, target_sr: Option<u32>, mono: bool) -> Result<Vec<f32>, DecodeError> {
        let file = File::open(&self.filepath)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Probe format with hint
        let mut hint = Hint::new();
        if let Some(ext) = Path::new(&self.filepath).extension() {
            hint.with_extension(ext.to_str().unwrap_or(""));
        }

        let mut probed = symphonia::default::get_probe().format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?;

        let track = probed
            .format
            .default_track()
            .ok_or(DecodeError::NoTrack)?;

        let track_id = track.id;
        let codec_params = track.codec_params.clone();
        let original_sr = codec_params.sample_rate.unwrap_or(44100);
        let original_channels = codec_params.channels.map(|c| c.count()).unwrap_or(1);

        // Create decoder
        let mut decoder = symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;

        // Decode all packets
        let mut samples: Vec<f32> = Vec::new();

        loop {
            match probed.format.next_packet() {
                Ok(packet) => {
                    // Only decode packets from our track
                    if packet.track_id() != track_id {
                        continue;
                    }

                    match decoder.decode(&packet) {
                        Ok(decoded) => {
                            // Convert to f32 samples
                            let audio_samples = self.extract_samples(&decoded, original_channels, mono);
                            samples.extend_from_slice(&audio_samples);
                        }
                        Err(SymphoniaError::DecodeError(_)) => {
                            // Skip decode errors (corrupted frames)
                            continue;
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
                Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // End of stream
                    break;
                }
                Err(e) => return Err(e.into()),
            }
        }

        // Resample if needed
        if let Some(sr) = target_sr {
            if sr != original_sr {
                let num_channels = if mono { 1 } else { original_channels };
                let mut resampler = Resampler::new(original_sr, sr, num_channels)
                    .map_err(|e| DecodeError::Resampling(e.to_string()))?;
                samples = resampler.process(&samples)
                    .map_err(|e| DecodeError::Resampling(e.to_string()))?;
            }
        }

        Ok(samples)
    }

    /// Extract samples from decoded audio buffer
    ///
    /// Handles different sample formats and channel configurations
    fn extract_samples(&self, decoded: &AudioBufferRef, channels: usize, mono: bool) -> Vec<f32> {
        // Convert to f32 based on buffer type
        let mut samples = match decoded {
            AudioBufferRef::F32(buf) => {
                // Already f32, just copy
                let mut out = Vec::with_capacity(buf.frames() * channels);
                for ch in 0..channels {
                    let channel = buf.chan(ch);
                    out.extend_from_slice(channel);
                }
                out
            }
            AudioBufferRef::S16(buf) => {
                // Convert i16 to f32
                let mut out = Vec::with_capacity(buf.frames() * channels);
                for ch in 0..channels {
                    for &sample in buf.chan(ch) {
                        out.push(sample as f32 / 32768.0);
                    }
                }
                out
            }
            AudioBufferRef::S32(buf) => {
                // Convert i32 to f32
                let mut out = Vec::with_capacity(buf.frames() * channels);
                for ch in 0..channels {
                    for &sample in buf.chan(ch) {
                        out.push(sample as f32 / 2147483648.0);
                    }
                }
                out
            }
            AudioBufferRef::U8(buf) => {
                // Convert u8 to f32 (centered around 0)
                let mut out = Vec::with_capacity(buf.frames() * channels);
                for ch in 0..channels {
                    for &sample in buf.chan(ch) {
                        out.push((sample as f32 - 128.0) / 128.0);
                    }
                }
                out
            }
            _ => {
                // Unsupported format, return zeros
                vec![0.0; decoded.frames() * channels]
            }
        };

        // Convert to mono if requested
        if mono && channels > 1 {
            samples = self.to_mono(&samples, channels);
        }

        samples
    }

    /// Convert multi-channel audio to mono by averaging
    fn to_mono(&self, samples: &[f32], channels: usize) -> Vec<f32> {
        let num_frames = samples.len() / channels;
        let mut mono = Vec::with_capacity(num_frames);

        for frame in 0..num_frames {
            let mut sum = 0.0;
            for ch in 0..channels {
                sum += samples[frame * channels + ch];
            }
            mono.push(sum / channels as f32);
        }

        mono
    }
}
