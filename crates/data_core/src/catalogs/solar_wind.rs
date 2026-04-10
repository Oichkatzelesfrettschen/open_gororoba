//! ACE SWEPAM Level 2 solar wind hourly data provider.
//!
//! The Advanced Composition Explorer (ACE) SWEPAM instrument measures
//! solar wind proton density, bulk speed, and temperature. Level 2
//! hourly averages are published in fixed-width ASCII by Caltech/ACE
//! Science Center.
//!
//! Source: <https://izw1.caltech.edu/ACE/ASC/level2/>
//! Reference: McComas et al. (1998), Space Sci. Rev. 86, 563
//!
//! Fetch/provider support is in `solar_wind_fetch` (feature-gated on `fetch`).

use crate::{
    catalogs::omni::OmniRecord,
    fetcher::FetchError,
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::path::Path;

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
            proton_density: parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or("")),
            bulk_speed: parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or("")),
            ion_temperature: parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or("")),
        });
    }
    records
}

/// Convert ACE SWEPAM records to OmniRecord format.
///
/// B-field fields are set to NaN (SWEPAM is a plasma-only instrument);
/// callers that need magnetic data should merge with an ACE MAG or
/// WIND MFI source via `data_core::catalogs::ace_mag::ace_mag_to_omni`.
pub fn swepam_to_omni(records: &[SwepamRecord]) -> Vec<OmniRecord> {
    records
        .iter()
        .filter(|r| !r.proton_density.is_nan() || !r.bulk_speed.is_nan())
        .map(|r| OmniRecord {
            year: r.decimal_year as u16,
            doy: r.doy,
            hour: r.hour,
            b_magnitude: f64::NAN,
            bx_gse: f64::NAN,
            by_gse: f64::NAN,
            bz_gse: f64::NAN,
            proton_temperature: r.ion_temperature,
            proton_density: r.proton_density,
            bulk_speed: r.bulk_speed,
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 0,
            r_au: 1.0,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
        })
        .collect()
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
