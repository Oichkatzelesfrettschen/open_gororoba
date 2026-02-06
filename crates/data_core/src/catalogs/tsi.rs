//! TSIS-1 Total Solar Irradiance (TSI) measurements.
//!
//! TSIS-1 (Total and Spectral Solar Irradiance Sensor) aboard the ISS provides
//! the most accurate TSI measurements to date, with ~0.01% absolute accuracy.
//!
//! Source: LASP Interactive Solar Irradiance Datacenter (LISIRD)
//! Reference: Kopp (2021), https://doi.org/10.1007/s11207-021-01853-x

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_with_fallbacks};
use std::path::{Path, PathBuf};

/// A single TSI measurement from TSIS-1.
#[derive(Debug, Clone)]
pub struct TsiMeasurement {
    /// Date as Julian Day Number.
    pub jd: f64,
    /// ISO date string (YYYY-MM-DD).
    pub date: String,
    /// Total Solar Irradiance (W/m^2).
    pub tsi: f64,
    /// TSI measurement uncertainty (W/m^2).
    pub tsi_uncertainty: f64,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "NaN" || s == "nan" || s == "-99" || s == "-999" {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse TSIS-1 TSI CSV data.
pub fn parse_tsi_csv(path: &Path) -> Result<Vec<TsiMeasurement>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    let mut measurements = Vec::new();
    let mut header_seen = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }

        // Skip header line(s)
        if !header_seen && (trimmed.contains("jd") || trimmed.contains("date")
            || trimmed.contains("TSI") || trimmed.contains("irradiance"))
        {
            header_seen = true;
            continue;
        }
        if !header_seen {
            // First non-comment, non-header line -- try parsing
            header_seen = true;
        }

        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() < 3 {
            continue;
        }

        let jd = parse_f64(fields[0]);
        let date = fields.get(1).unwrap_or(&"").trim().to_string();
        let tsi = parse_f64(fields.get(2).unwrap_or(&""));
        let unc = parse_f64(fields.get(3).unwrap_or(&""));

        if tsi.is_nan() {
            continue;
        }

        measurements.push(TsiMeasurement {
            jd,
            date,
            tsi,
            tsi_uncertainty: unc,
        });
    }

    Ok(measurements)
}

/// LASP LISIRD TSIS-1 daily TSI data URL.
const TSIS_URLS: &[&str] = &[
    "https://lasp.colorado.edu/lisird/latis/dap/tsis_tsi_24hr.csv",
];

/// TSIS-1 Total Solar Irradiance dataset provider.
pub struct TsisTsiProvider;

impl DatasetProvider for TsisTsiProvider {
    fn name(&self) -> &str { "TSIS-1 TSI Daily" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("tsis1_tsi_daily.csv");
        download_with_fallbacks(self.name(), TSIS_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("tsis1_tsi_daily.csv").exists()
    }
}
