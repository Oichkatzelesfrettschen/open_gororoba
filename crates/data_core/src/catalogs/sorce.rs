//! SORCE TIM total solar irradiance series.
//!
//! SORCE extends the long-term irradiance record and is useful as a legacy
//! complement to TSIS-1.
//!
//! Source: LASP LISIRD
//! https://lasp.colorado.edu/lisird/

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

/// One SORCE TSI record.
#[derive(Debug, Clone)]
pub struct SorceMeasurement {
    pub jd: f64,
    pub date: String,
    pub tsi: f64,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "nan" || s == "NaN" || s == "-99" {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse SORCE CSV rows.
pub fn parse_sorce_csv(path: &Path) -> Result<Vec<SorceMeasurement>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    let mut rows = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with(';')
            || trimmed.contains("date")
        {
            continue;
        }

        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() < 3 {
            continue;
        }
        let tsi = parse_f64(fields[2]);
        if tsi.is_nan() {
            continue;
        }
        rows.push(SorceMeasurement {
            jd: parse_f64(fields[0]),
            date: fields[1].trim().to_string(),
            tsi,
        });
    }
    Ok(rows)
}

const SORCE_URLS: &[&str] = &[
    "https://lasp.colorado.edu/lisird/latis/dap/sorce_tsi_24hr.csv",
    "https://lasp.colorado.edu/lisird/latis/dap/sorce_tsi_6hr.csv",
];

/// SORCE TSI dataset provider.
pub struct SorceTsiProvider;

impl DatasetProvider for SorceTsiProvider {
    fn name(&self) -> &str {
        "SORCE TSI Daily"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("sorce_tsi_daily.csv");
        download_with_fallbacks(self.name(), SORCE_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("sorce_tsi_daily.csv").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_parse_sorce_if_available() {
        let path = Path::new("data/external/sorce_tsi_daily.csv");
        if !path.exists() {
            eprintln!("Skipping: SORCE data not available");
            return;
        }
        let rows = parse_sorce_csv(path).expect("failed to parse SORCE CSV");
        assert!(!rows.is_empty(), "SORCE rows should not be empty");
    }
}
