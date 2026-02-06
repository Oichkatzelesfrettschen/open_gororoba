//! NANOGrav 15-year Gravitational Wave Background dataset.
//!
//! NANOGrav (North American Nanohertz Observatory for Gravitational Waves)
//! detected a gravitational wave background in their 15-year dataset.
//! This module provides the free spectrum (per-frequency bin) strain estimates.
//!
//! Source: Zenodo, https://doi.org/10.5281/zenodo.8067081
//! Reference: Agazie et al. (2023), ApJL 951, L8

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file, validate_not_html};
use std::fs;
use std::path::{Path, PathBuf};

/// A single frequency bin from the NANOGrav free spectrum.
#[derive(Debug, Clone)]
pub struct FreeSpectrumPoint {
    /// Frequency (Hz), typically in nHz range.
    pub frequency: f64,
    /// Log10 of characteristic strain squared.
    pub log10_rho: f64,
    /// Lower 95% credible bound on log10(rho).
    pub log10_rho_lo: f64,
    /// Upper 95% credible bound on log10(rho).
    pub log10_rho_hi: f64,
}

/// Parse NANOGrav free spectrum CSV.
///
/// Expected columns: frequency, log10_rho, log10_rho_lo, log10_rho_hi
pub fn parse_nanograv_free_spectrum(path: &Path) -> Result<Vec<FreeSpectrumPoint>, FetchError> {
    let content = fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    let mut points = Vec::new();
    let mut header_seen = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !header_seen {
            header_seen = true;
            continue;
        }

        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() < 4 {
            continue;
        }

        let parse = |s: &str| -> f64 {
            s.trim().parse::<f64>().unwrap_or(f64::NAN)
        };

        points.push(FreeSpectrumPoint {
            frequency: parse(fields[0]),
            log10_rho: parse(fields[1]),
            log10_rho_lo: parse(fields[2]),
            log10_rho_hi: parse(fields[3]),
        });
    }

    Ok(points)
}

/// NANOGrav 15-year free spectrum data provider.
///
/// Fetches the per-frequency-bin gravitational wave strain estimates
/// from the NANOGrav 15yr dataset on Zenodo.
const NANOGRAV_URLS: &[&str] = &[
    "https://zenodo.org/records/8067081/files/NANOGrav15yr_FreeSpectrum.csv",
];

pub struct NanoGrav15yrProvider;

impl DatasetProvider for NanoGrav15yrProvider {
    fn name(&self) -> &str { "NANOGrav 15yr Free Spectrum" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("nanograv_15yr_freespectrum.csv");
        if config.skip_existing && output.exists() {
            eprintln!("  {} already cached at {}", self.name(), output.display());
            return Ok(output);
        }

        for url in NANOGRAV_URLS {
            eprintln!("  Downloading {} from {}...", self.name(), url);
            match download_to_file(url, &output) {
                Ok(bytes) => {
                    let data = fs::read(&output)?;
                    if let Err(e) = validate_not_html(&data) {
                        eprintln!("  Validation failed: {}", e);
                        fs::remove_file(&output).ok();
                        continue;
                    }
                    eprintln!("  Saved {} bytes to {}", bytes, output.display());
                    return Ok(output);
                }
                Err(e) => {
                    eprintln!("  Failed: {}", e);
                }
            }
        }

        Err(FetchError::AllUrlsFailed {
            dataset: self.name().to_string(),
        })
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("nanograv_15yr_freespectrum.csv").exists()
    }
}
