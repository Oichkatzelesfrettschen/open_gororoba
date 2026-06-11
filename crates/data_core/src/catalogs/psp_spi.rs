//! Parker Solar Probe SWEAP/SPI SF00 level-3 moment parser via CDAWeb HAPI.
//!
//! Official public sources:
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=PSP_SWP_SPI_SF00_L3_MOM>
//!   <https://cdaweb.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/>
//!
//! The direct CDAWeb daily CDF directory is used as the authoritative day
//! manifest, while the executed Rust science path stages parser-friendly daily
//! HAPI CSV slices for the available public days.
//!
//! Fetch logic lives in `psp_spi_fetch`.

use crate::parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

pub const EV_TO_K: f64 = 11_604.518_121_550_08;

#[derive(Debug, Clone)]
pub struct PspSpiMomRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub density_cm3: f64,
    pub speed_kms: f64,
    pub temperature_k: f64,
}

#[cfg(feature = "fetch")]
pub use super::psp_spi_fetch::{PspSpiMomProvider, parse_psp_spi_mom_file};

#[derive(Default)]
pub(crate) struct SpiHourAccumulator {
    pub density_sum: f64,
    pub density_count: usize,
    pub speed_sum: f64,
    pub speed_count: usize,
    pub temperature_sum: f64,
    pub temperature_count: usize,
}

pub fn parse_psp_spi_mom_csv(content: &str) -> Vec<PspSpiMomRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let density_col = headers.iter().position(|value| value == "DENS");
    let temp_col = headers.iter().position(|value| value == "TEMP");
    let quality_col = headers.iter().position(|value| value == "QUALITY_FLAG");
    let vr_col = headers.iter().position(|value| value == "VEL_RTN_SUN_0");
    let vt_col = headers.iter().position(|value| value == "VEL_RTN_SUN_1");
    let vn_col = headers.iter().position(|value| value == "VEL_RTN_SUN_2");
    let (
        Some(density_col),
        Some(temp_col),
        Some(quality_col),
        Some(vr_col),
        Some(vt_col),
        Some(vn_col),
    ) = (density_col, temp_col, quality_col, vr_col, vt_col, vn_col)
    else {
        return Vec::new();
    };
    let mut hourly: BTreeMap<(u16, u16, u8), SpiHourAccumulator> = BTreeMap::new();
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
        if quality == 65_535 {
            continue;
        }
        let density = parse_hapi_spacephysics_f64_or_nan(record.get(density_col).unwrap_or(""));
        let temp_ev = parse_hapi_spacephysics_f64_or_nan(record.get(temp_col).unwrap_or(""));
        let vr = parse_hapi_spacephysics_f64_or_nan(record.get(vr_col).unwrap_or(""));
        let vt = parse_hapi_spacephysics_f64_or_nan(record.get(vt_col).unwrap_or(""));
        let vn = parse_hapi_spacephysics_f64_or_nan(record.get(vn_col).unwrap_or(""));
        let speed = if vr.is_finite() || vt.is_finite() || vn.is_finite() {
            let vr = if vr.is_finite() { vr } else { 0.0 };
            let vt = if vt.is_finite() { vt } else { 0.0 };
            let vn = if vn.is_finite() { vn } else { 0.0 };
            (vr * vr + vt * vt + vn * vn).sqrt()
        } else {
            f64::NAN
        };
        let entry = hourly.entry((year, doy, hour)).or_default();
        if density.is_finite() {
            entry.density_sum += density;
            entry.density_count += 1;
        }
        if speed.is_finite() {
            entry.speed_sum += speed;
            entry.speed_count += 1;
        }
        if temp_ev.is_finite() {
            entry.temperature_sum += temp_ev * EV_TO_K;
            entry.temperature_count += 1;
        }
    }
    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            if acc.density_count == 0 && acc.speed_count == 0 && acc.temperature_count == 0 {
                return None;
            }
            Some(PspSpiMomRecord {
                year,
                doy,
                hour,
                density_cm3: if acc.density_count > 0 {
                    acc.density_sum / acc.density_count as f64
                } else {
                    f64::NAN
                },
                speed_kms: if acc.speed_count > 0 {
                    acc.speed_sum / acc.speed_count as f64
                } else {
                    f64::NAN
                },
                temperature_k: if acc.temperature_count > 0 {
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
    fn test_parse_psp_spi_mom_csv() {
        let csv = "Time,CNTS,QUALITY_FLAG,DENS,VEL_RTN_SUN_0,VEL_RTN_SUN_1,VEL_RTN_SUN_2,TEMP,SUN_DIST\n\
2025-07-01T00:03:42.373199744Z,152.0,4288,0.045442,292.95,88.471,30.321,64.683,64683000\n\
2025-07-01T00:07:26.069820160Z,145.0,4288,0.040847,310.63,87.622,35.143,64.729,64692000\n";
        let rows = parse_psp_spi_mom_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert!(row.density_cm3.is_finite());
        assert!(row.speed_kms > 300.0);
        assert!(row.temperature_k > 700_000.0);
    }
}
