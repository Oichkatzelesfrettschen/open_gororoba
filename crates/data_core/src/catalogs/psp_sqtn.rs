//! Parker Solar Probe FIELDS SQTN density/temperature parser via CDAWeb HAPI.
//!
//! Public official source:
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=PSP_FLD_L3_SQTN_RFS_V1V2>
//!
//! This product provides electron-density and core-temperature estimates and
//! complements the merged hourly and FIELDS MAG lanes for later mission windows.
//!
//! Fetch logic lives in `psp_sqtn_fetch`.

use crate::parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

pub const EV_TO_K: f64 = 11_604.518_121_550_08;

#[derive(Debug, Clone)]
pub struct PspSqtnRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub electron_density_cm3: f64,
    pub electron_core_temperature_k: f64,
}

#[derive(Default)]
pub(crate) struct SqtnHourAccumulator {
    pub density_sum: f64,
    pub density_count: usize,
    pub temperature_sum: f64,
    pub temperature_count: usize,
}

pub fn parse_psp_sqtn_csv(content: &str) -> Vec<PspSqtnRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let density_col = headers.iter().position(|value| value == "electron_density");
    let quality_col = headers
        .iter()
        .position(|value| value == "density_quality_flag");
    let temp_col = headers
        .iter()
        .position(|value| value == "electron_core_temperature");
    let (Some(density_col), Some(quality_col), Some(temp_col)) =
        (density_col, quality_col, temp_col)
    else {
        return Vec::new();
    };
    let mut hourly: BTreeMap<(u16, u16, u8), SqtnHourAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let quality = record
            .get(quality_col)
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(0);
        if quality <= 0 {
            continue;
        }
        let density = parse_hapi_spacephysics_f64_or_nan(record.get(density_col).unwrap_or(""));
        let temp_ev = parse_hapi_spacephysics_f64_or_nan(record.get(temp_col).unwrap_or(""));
        let entry = hourly.entry((year, doy, hour)).or_default();
        if density.is_finite() {
            entry.density_sum += density;
            entry.density_count += 1;
        }
        if temp_ev.is_finite() {
            entry.temperature_sum += temp_ev * EV_TO_K;
            entry.temperature_count += 1;
        }
    }
    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            if acc.density_count == 0 && acc.temperature_count == 0 {
                return None;
            }
            Some(PspSqtnRecord {
                year,
                doy,
                hour,
                electron_density_cm3: if acc.density_count > 0 {
                    acc.density_sum / acc.density_count as f64
                } else {
                    f64::NAN
                },
                electron_core_temperature_k: if acc.temperature_count > 0 {
                    acc.temperature_sum / acc.temperature_count as f64
                } else {
                    f64::NAN
                },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_psp_sqtn_csv() {
        let csv = "Time,electron_density,density_quality_flag,electron_core_temperature\n\
2025-09-23T22:00:00Z,120.0,2,20.0\n\
2025-09-23T22:10:00Z,180.0,2,10.0\n\
2025-09-23T22:20:00Z,-1.0E31,0,-1.0E31\n";
        let rows = parse_psp_sqtn_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert!((row.electron_density_cm3 - 150.0).abs() < 1.0e-9);
        assert!(row.electron_core_temperature_k > 100_000.0);
    }
}
