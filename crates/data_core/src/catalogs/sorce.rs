//! SORCE TIM total solar irradiance series.
//!
//! SORCE extends the long-term irradiance record and is useful as a legacy
//! complement to TSIS-1.
//!
//! Source: LASP LISIRD / SORCE data archive
//! https://lasp.colorado.edu/lisird/data/sorce_tsi_24hr_l3
//!
//! The old LaTiS endpoint (`sorce_tsi_24hr`) was renamed to `sorce_tsi_24hr_l3`
//! (Level 3 suffix). The most stable URL is the direct file link at
//! `/data/sorce/tsi_data/daily/`.

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

/// Parse SORCE data from either CSV (LaTiS) or space-delimited TXT (direct).
///
/// CSV format: `time (Julian Date),tsi_1au (W/m^2),...`
/// TXT format: `YYYYMMDD.5  JDN  avg_jdn  stdev  tsi_1au  ...` (`;` comments)
pub fn parse_sorce_csv(path: &Path) -> Result<Vec<SorceMeasurement>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    let mut rows = Vec::new();
    let is_csv = content.lines().any(|l| l.contains("tsi_1au") && l.contains(','));

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with(';')
            || trimmed.contains("date")
            || trimmed.contains("tsi_1au")
        {
            continue;
        }

        if is_csv {
            // LaTiS CSV: col 0 = JD, col 1 = TSI
            let fields: Vec<&str> = trimmed.split(',').collect();
            if fields.len() < 2 {
                continue;
            }
            let jd = parse_f64(fields[0]);
            let tsi = parse_f64(fields[1]);
            if tsi.is_nan() || tsi <= 0.0 {
                continue;
            }
            rows.push(SorceMeasurement {
                jd,
                date: String::new(),
                tsi,
            });
        } else {
            // Direct TXT: col 1 = JDN, col 4 = tsi_1au (space-separated)
            let fields: Vec<&str> = trimmed.split_whitespace().collect();
            if fields.len() < 5 {
                continue;
            }
            let jd = parse_f64(fields[1]);
            let tsi = parse_f64(fields[4]);
            if tsi.is_nan() || tsi <= 0.0 {
                continue;
            }
            let date_str = fields[0];
            // date_str is like "20030225.500"; strip decimal
            let date = date_str.split('.').next().unwrap_or("").to_string();
            rows.push(SorceMeasurement {
                jd,
                date,
                tsi,
            });
        }
    }
    Ok(rows)
}

const SORCE_URLS: &[&str] = &[
    // Direct file link (most stable, 933 KB, Version 20, Aug 2025)
    "https://lasp.colorado.edu/data/sorce/tsi_data/daily/sorce_tsi_L3_c24h_latest.txt",
    // LaTiS API (renamed from sorce_tsi_24hr to sorce_tsi_24hr_l3)
    "https://lasp.colorado.edu/lisird/latis/dap/sorce_tsi_24hr_l3.csv",
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
