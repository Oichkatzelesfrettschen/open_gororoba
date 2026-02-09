//! ESA Swarm magnetic field sample provider.
//!
//! Fetches a short, reproducible sample via the VirES HAPI endpoint for
//! magnetic field vectors and total intensity.
//!
//! Source: https://vires.services/
//!
//! HAPI CSV format: ISO 8601 timestamp, then parameter columns.

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

const SWARM_URLS: &[&str] = &[
    "https://vires.services/hapi/data?dataset=SW_OPER_MAGA_LR_1B&parameters=Latitude,Longitude,Radius,F,B_NEC&start=2014-01-01T00:00:00Z&stop=2014-01-01T00:30:00Z&format=csv&include=header",
    "https://vires.services/hapi/data?dataset=SW_OPER_MAGB_LR_1B&parameters=Latitude,Longitude,Radius,F,B_NEC&start=2014-01-01T00:00:00Z&stop=2014-01-01T00:30:00Z&format=csv&include=header",
];

/// Expected column names in the HAPI CSV header.
pub const SWARM_EXPECTED_COLUMNS: &[&str] = &["Timestamp", "Latitude", "Longitude", "Radius", "F"];

/// One Swarm magnetic field measurement.
#[derive(Debug, Clone)]
pub struct SwarmRecord {
    /// ISO 8601 timestamp string.
    pub timestamp: String,
    /// Geographic latitude (degrees).
    pub latitude: f64,
    /// Geographic longitude (degrees).
    pub longitude: f64,
    /// Geocentric radius (m).
    pub radius: f64,
    /// Total field intensity F (nT).
    pub f_total: f64,
}

fn parse_f64(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse Swarm HAPI CSV data.
///
/// Validates that the header row contains expected column names
/// and parses each data row into a SwarmRecord.
pub fn parse_swarm_csv(path: &Path) -> Result<Vec<SwarmRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    let mut records = Vec::new();
    let mut header_validated = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // First non-comment line should be the header
        if !header_validated {
            let lower = trimmed.to_lowercase();
            for col in SWARM_EXPECTED_COLUMNS {
                if !lower.contains(&col.to_lowercase()) {
                    return Err(FetchError::Validation(format!(
                        "Swarm CSV header missing expected column '{}': {}",
                        col, trimmed
                    )));
                }
            }
            header_validated = true;
            continue;
        }

        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() < 5 {
            continue;
        }

        records.push(SwarmRecord {
            timestamp: fields[0].trim().to_string(),
            latitude: parse_f64(fields[1]),
            longitude: parse_f64(fields[2]),
            radius: parse_f64(fields[3]),
            f_total: parse_f64(fields[4]),
        });
    }

    Ok(records)
}

/// Check that timestamps are strictly monotonically increasing.
///
/// Returns Ok(()) if timestamps are monotonic, or an error with the first
/// out-of-order pair.
pub fn check_timestamp_monotonicity(records: &[SwarmRecord]) -> Result<(), FetchError> {
    for pair in records.windows(2) {
        if pair[1].timestamp <= pair[0].timestamp {
            return Err(FetchError::Validation(format!(
                "Swarm timestamps not monotonic: '{}' >= '{}'",
                pair[0].timestamp, pair[1].timestamp
            )));
        }
    }
    Ok(())
}

/// Swarm provider.
pub struct SwarmMagAProvider;

impl DatasetProvider for SwarmMagAProvider {
    fn name(&self) -> &str {
        "Swarm L1B Magnetic Sample"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("swarm_magnetic_sample.csv");
        download_with_fallbacks(self.name(), SWARM_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("swarm_magnetic_sample.csv").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_swarm_synthetic() {
        let csv = "\
# HAPI CSV output
Timestamp,Latitude,Longitude,Radius,F,B_NEC_N,B_NEC_E,B_NEC_C
2014-01-01T00:00:00Z,-45.123,120.456,6871200.0,48234.5,12345.0,-2345.0,46789.0
2014-01-01T00:00:01Z,-45.100,120.460,6871205.0,48240.1,12350.0,-2340.0,46800.0
2014-01-01T00:00:02Z,-45.077,120.464,6871210.0,48245.7,12355.0,-2335.0,46810.0
";
        let f = write_temp(csv);
        let records = parse_swarm_csv(f.path()).unwrap();
        assert_eq!(records.len(), 3, "Should parse 3 data rows");
        assert_eq!(records[0].timestamp, "2014-01-01T00:00:00Z");
        assert!((records[0].latitude - (-45.123)).abs() < 1e-6);
        assert!((records[0].f_total - 48234.5).abs() < 0.1);
        assert!((records[2].radius - 6871210.0).abs() < 0.1);
    }

    #[test]
    fn test_swarm_header_validation_rejects_bad_header() {
        let csv = "\
wrong,columns,here,oops,bad
1,2,3,4,5
";
        let f = write_temp(csv);
        let result = parse_swarm_csv(f.path());
        assert!(result.is_err(), "Should reject CSV with wrong header");
    }

    #[test]
    fn test_swarm_timestamp_monotonicity_ok() {
        let records = vec![
            SwarmRecord {
                timestamp: "2014-01-01T00:00:00Z".into(),
                latitude: 0.0,
                longitude: 0.0,
                radius: 0.0,
                f_total: 0.0,
            },
            SwarmRecord {
                timestamp: "2014-01-01T00:00:01Z".into(),
                latitude: 0.0,
                longitude: 0.0,
                radius: 0.0,
                f_total: 0.0,
            },
            SwarmRecord {
                timestamp: "2014-01-01T00:00:02Z".into(),
                latitude: 0.0,
                longitude: 0.0,
                radius: 0.0,
                f_total: 0.0,
            },
        ];
        assert!(check_timestamp_monotonicity(&records).is_ok());
    }

    #[test]
    fn test_swarm_timestamp_monotonicity_fails() {
        let records = vec![
            SwarmRecord {
                timestamp: "2014-01-01T00:00:02Z".into(),
                latitude: 0.0,
                longitude: 0.0,
                radius: 0.0,
                f_total: 0.0,
            },
            SwarmRecord {
                timestamp: "2014-01-01T00:00:01Z".into(),
                latitude: 0.0,
                longitude: 0.0,
                radius: 0.0,
                f_total: 0.0,
            },
        ];
        assert!(check_timestamp_monotonicity(&records).is_err());
    }

    #[test]
    fn test_swarm_empty_records_monotonic() {
        assert!(check_timestamp_monotonicity(&[]).is_ok());
    }

    #[test]
    fn test_parse_swarm_if_available() {
        let path = std::path::Path::new("data/external/swarm_magnetic_sample.csv");
        if !path.exists() {
            eprintln!("Skipping: Swarm data not available");
            return;
        }
        let records = parse_swarm_csv(path).expect("failed to parse Swarm CSV");
        assert!(!records.is_empty(), "Swarm records should not be empty");
        check_timestamp_monotonicity(&records).expect("timestamps should be monotonic");
    }
}
