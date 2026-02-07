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

/// Number of fields in the TsiMeasurement struct.
pub const TSI_FIELD_COUNT: usize = 4;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_synthetic_tsi() {
        let csv = "\
jd,date,TSI,uncertainty
2459215.5,2021-01-01,1360.52,0.14
2459216.5,2021-01-02,1360.48,0.13
2459217.5,2021-01-03,1360.55,0.15
";
        let f = write_temp_csv(csv);
        let measurements = parse_tsi_csv(f.path()).unwrap();
        assert_eq!(measurements.len(), 3);

        let m1 = &measurements[0];
        assert!((m1.jd - 2459215.5).abs() < 0.01);
        assert_eq!(m1.date, "2021-01-01");
        assert!((m1.tsi - 1360.52).abs() < 0.01);
        assert!((m1.tsi_uncertainty - 0.14).abs() < 0.01);
    }

    #[test]
    fn test_tsi_nan_rows_skipped() {
        let csv = "\
jd,date,TSI,uncertainty
2459215.5,2021-01-01,NaN,0
2459216.5,2021-01-02,1360.5,0.1
";
        let f = write_temp_csv(csv);
        let measurements = parse_tsi_csv(f.path()).unwrap();
        assert_eq!(measurements.len(), 1, "NaN TSI rows should be skipped");
        assert!((measurements[0].tsi - 1360.5).abs() < 0.01);
    }

    #[test]
    fn test_tsi_comment_lines_skipped() {
        let csv = "\
# TSIS-1 Total Solar Irradiance
; Version 03
jd,date,TSI,uncertainty
2459215.5,2021-01-01,1360.52,0.14
";
        let f = write_temp_csv(csv);
        let measurements = parse_tsi_csv(f.path()).unwrap();
        assert_eq!(measurements.len(), 1);
    }

    #[test]
    fn test_tsi_sentinel_values() {
        let csv = "\
jd,date,TSI,uncertainty
2459215.5,2021-01-01,-99,0
2459216.5,2021-01-02,-999,0
2459217.5,2021-01-03,1360.5,0.1
";
        let f = write_temp_csv(csv);
        let measurements = parse_tsi_csv(f.path()).unwrap();
        assert_eq!(measurements.len(), 1, "sentinel values -99/-999 should be NaN -> skipped");
    }

    #[test]
    fn test_tsi_field_count() {
        assert_eq!(TSI_FIELD_COUNT, 4);
    }
}
