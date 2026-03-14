//! ACE SWEPAM Level 2 solar wind hourly data provider.
//!
//! The Advanced Composition Explorer (ACE) SWEPAM instrument measures
//! solar wind proton density, bulk speed, and temperature. Level 2
//! hourly averages are published in fixed-width ASCII by Caltech/ACE
//! Science Center.
//!
//! Source: <https://izw1.caltech.edu/ACE/ASC/level2/>
//! Reference: McComas et al. (1998), Space Sci. Rev. 86, 563

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::path::{Path, PathBuf};

/// A single hourly ACE SWEPAM solar wind measurement.
#[derive(Debug, Clone)]
pub struct SwepamRecord {
    /// Decimal year (e.g., 2024.5).
    pub decimal_year: f64,
    /// Day of year (1-366).
    pub doy: u16,
    /// Hour of day (0-23).
    pub hour: u8,
    /// Proton number density (cm^-3). NaN if fill value (-9999.9).
    pub proton_density: f64,
    /// Bulk solar wind speed (km/s). NaN if fill value.
    pub bulk_speed: f64,
    /// Proton temperature (K). NaN if fill value (-1.00e+05).
    pub ion_temperature: f64,
}

const FILL_DENSITY: f64 = -9999.9;
const FILL_SPEED: f64 = -9999.9;
const FILL_TEMP: f64 = -1.00e+05;

/// Parse ACE SWEPAM Level 2 hourly ASCII data.
///
/// Format (header lines start with `#`):
/// ```text
/// Year DOY  Hr  Np   V   Tpr
/// 2024   1   0  4.2 370.1 5.1e+04
/// ```
pub fn parse_swepam_hourly(content: &str) -> Vec<SwepamRecord> {
    let mut records = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("Year") {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 6 {
            continue;
        }
        let year: f64 = match fields[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let doy: u16 = match fields[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let hour: u8 = match fields[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let np: f64 = fields[3].parse().unwrap_or(f64::NAN);
        let v: f64 = fields[4].parse().unwrap_or(f64::NAN);
        let tpr: f64 = fields[5].parse().unwrap_or(f64::NAN);

        let decimal_year = year + (doy as f64 - 1.0 + hour as f64 / 24.0) / 365.25;

        records.push(SwepamRecord {
            decimal_year,
            doy,
            hour,
            proton_density: if (np - FILL_DENSITY).abs() < 1.0 {
                f64::NAN
            } else {
                np
            },
            bulk_speed: if (v - FILL_SPEED).abs() < 1.0 {
                f64::NAN
            } else {
                v
            },
            ion_temperature: if (tpr - FILL_TEMP).abs() < 1.0e3 {
                f64::NAN
            } else {
                tpr
            },
        });
    }
    records
}

/// Parse an ACE SWEPAM file from disk.
pub fn parse_swepam_file(path: &Path) -> Result<Vec<SwepamRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    if path.extension().and_then(|value| value.to_str()) == Some("csv") {
        Ok(parse_swepam_hapi_csv(&content))
    } else {
        Ok(parse_swepam_hourly(&content))
    }
}

pub fn parse_swepam_hapi_csv(content: &str) -> Vec<SwepamRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut records = Vec::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let decimal_year = year as f64 + (doy as f64 - 1.0 + hour as f64 / 24.0) / 365.25;
        records.push(SwepamRecord {
            decimal_year,
            doy,
            hour,
            proton_density: parse_f64_or_nan(record.get(1).unwrap_or("")),
            bulk_speed: parse_f64_or_nan(record.get(2).unwrap_or("")),
            ion_temperature: parse_f64_or_nan(record.get(3).unwrap_or("")),
        });
    }
    records
}

const ACE_SWEPAM_HAPI_DATASET: &str = "AC_H2_SWE";

/// ACE SWEPAM dataset provider.
///
/// Fetches hourly solar wind data for a specified year range.
pub struct AceSwepamProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for AceSwepamProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for AceSwepamProvider {
    fn name(&self) -> &str {
        "ACE SWEPAM Solar Wind"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ace_swepam");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("ac_h2_swe_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                ACE_SWEPAM_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&["Time", "Np", "Vp", "Tpr"]),
            ) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("Saved {}", fname);
                }
                Err(e) => {
                    log::warn!("Failed to download ACE SWEPAM {}: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("ace_swepam").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_swepam_hourly_basic() {
        let data = "\
# ACE SWEPAM Level 2 hourly data
# Year DOY Hr Np V Tpr
2024   1  0    4.2   370.1  5.10e+04
2024   1  1    4.5   365.3  4.80e+04
2024   1  2 -9999.9 -9999.9  -1.00e+05
";
        let records = parse_swepam_hourly(data);
        assert_eq!(records.len(), 3);

        assert_eq!(records[0].doy, 1);
        assert_eq!(records[0].hour, 0);
        assert!((records[0].proton_density - 4.2).abs() < 0.01);
        assert!((records[0].bulk_speed - 370.1).abs() < 0.1);
        assert!((records[0].ion_temperature - 5.1e4).abs() < 1.0e3);

        // Fill values should become NaN
        assert!(records[2].proton_density.is_nan());
        assert!(records[2].bulk_speed.is_nan());
        assert!(records[2].ion_temperature.is_nan());
    }

    #[test]
    fn test_parse_swepam_empty() {
        let data = "# header only\n";
        let records = parse_swepam_hourly(data);
        assert!(records.is_empty());
    }

    #[test]
    fn test_swepam_physical_ranges() {
        // Typical solar wind: density 1-50 cm^-3, speed 250-800 km/s, temp 1e4-5e6 K
        let data = "\
2024  180  12    8.0   450.0  1.20e+05
";
        let records = parse_swepam_hourly(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(
            r.proton_density > 0.5 && r.proton_density < 100.0,
            "density out of range: {}",
            r.proton_density
        );
        assert!(
            r.bulk_speed > 200.0 && r.bulk_speed < 1000.0,
            "speed out of range: {}",
            r.bulk_speed
        );
        assert!(
            r.ion_temperature > 1.0e3 && r.ion_temperature < 1.0e7,
            "temperature out of range: {}",
            r.ion_temperature
        );
    }

    #[test]
    fn test_swepam_decimal_year() {
        let data = "2024  1  0  4.0 370.0 5.0e+04\n";
        let records = parse_swepam_hourly(data);
        // Day 1, hour 0 -> decimal_year ~ 2024.0
        assert!((records[0].decimal_year - 2024.0).abs() < 0.01);
    }
}
