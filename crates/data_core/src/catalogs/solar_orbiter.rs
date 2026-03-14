//! Solar Orbiter merged hourly plasma and magnetic-field parser.
//!
//! Solar Orbiter provides modern inner-heliosphere coverage from 2020 onward.
//! The official Rust fetch lane uses the CDAWeb HAPI dataset:
//!   SOLO_COHO1HR_MERGED_MAG_PLASMA
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=SOLO_COHO1HR_MERGED_MAG_PLASMA>
//!
//! B-field coordinates are RTN (Radial-Tangential-Normal).

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord, parse_spdf_merged, spdf_to_omni},
    },
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::path::PathBuf;

/// Column layout for Solar Orbiter merged hourly data.
pub const SOLO_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 13,
    col_year: 0,
    col_doy: 1,
    col_hour: 2,
    col_distance_au: Some(3),
    col_lat_deg: Some(4),
    col_lon_deg: Some(5),
    col_b_mag: Some(6),
    col_br: Some(7),
    col_bt: Some(8),
    col_bn: Some(9),
    col_density: Some(10),
    col_speed: Some(11),
    col_temperature: Some(12),
    fill_b: 9999.99,
    fill_density: 999.9,
    fill_speed: 9999.9,
    fill_temperature: 999999.0,
    fill_distance: 999.999,
    b_is_se: false,
};

/// Parse Solar Orbiter merged hourly data from a string.
pub fn parse_solar_orbiter_merged(content: &str) -> Vec<SpdfMergedRecord> {
    parse_spdf_merged(content, &SOLO_LAYOUT)
}

/// Parse Solar Orbiter merged hourly data from a file.
pub fn parse_solar_orbiter_file(
    path: &std::path::Path,
) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    if path.extension().and_then(|value| value.to_str()) == Some("csv") {
        Ok(parse_solo_hapi_csv(&content))
    } else {
        Ok(parse_solar_orbiter_merged(&content))
    }
}

/// Convert Solar Orbiter merged records to OmniRecord format.
pub fn solar_orbiter_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    spdf_to_omni(records, false)
}

pub fn parse_solo_hapi_csv(content: &str) -> Vec<SpdfMergedRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut rows = Vec::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        rows.push(SpdfMergedRecord {
            year,
            doy,
            hour,
            distance_au: parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or("")),
            lat_deg: parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or("")),
            lon_deg: parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or("")),
            br: parse_hapi_spacephysics_f64_or_nan(record.get(4).unwrap_or("")),
            bt: parse_hapi_spacephysics_f64_or_nan(record.get(5).unwrap_or("")),
            bn: parse_hapi_spacephysics_f64_or_nan(record.get(6).unwrap_or("")),
            b_magnitude: parse_hapi_spacephysics_f64_or_nan(record.get(7).unwrap_or("")),
            bulk_speed: parse_hapi_spacephysics_f64_or_nan(record.get(11).unwrap_or("")),
            proton_density: parse_hapi_spacephysics_f64_or_nan(record.get(14).unwrap_or("")),
            proton_temperature: parse_hapi_spacephysics_f64_or_nan(record.get(15).unwrap_or("")),
        });
    }
    rows
}

const SOLO_HAPI_DATASET: &str = "SOLO_COHO1HR_MERGED_MAG_PLASMA";

/// Solar Orbiter merged-hourly dataset provider.
pub struct SolarOrbiterProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for SolarOrbiterProvider {
    fn default() -> Self {
        Self {
            year_start: 2020,
            year_end: 2020,
        }
    }
}

impl DatasetProvider for SolarOrbiterProvider {
    fn name(&self) -> &str {
        "Solar Orbiter Merged Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("solar_orbiter");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("solo_coho1hr_merged_mag_plasma_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                SOLO_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&[
                    "Time",
                    "radialDistance",
                    "heliographicLatitude",
                    "heliographicLongitude",
                    "BR",
                    "BT",
                    "BN",
                    "B",
                    "VR",
                    "VT",
                    "VN",
                    "ProtonSpeed",
                    "flow_theta",
                    "flow_lon",
                    "protonDensity",
                    "protonTemp",
                ]),
            ) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download Solar Orbiter {} via HAPI: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("solar_orbiter");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("solo_coho1hr_merged_mag_plasma_") && name.ends_with(".csv")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solo_layout_validity() {
        assert_eq!(SOLO_LAYOUT.min_columns, 13);
        const { assert!(!SOLO_LAYOUT.b_is_se) };
        assert_eq!(SOLO_LAYOUT.col_distance_au, Some(3));
        assert_eq!(SOLO_LAYOUT.col_b_mag, Some(6));
    }

    #[test]
    fn test_parse_solo_inner_heliosphere() {
        let data = "2020 180 12 0.65 4.0 120.0 18.0 16.0 -6.0 4.0 35.0 360.0 250000.0\n";
        let records = parse_solar_orbiter_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 0.65).abs() < 0.001);
        assert!((r.b_magnitude - 18.0).abs() < 0.1);
        assert!((r.proton_density - 35.0).abs() < 0.1);
        assert!((r.bulk_speed - 360.0).abs() < 0.1);
        assert!((r.proton_temperature - 250000.0).abs() < 1.0);
    }

    #[test]
    fn test_solo_fill_values() {
        let data =
            "2020 180 12 999.999 999.99 999.99 9999.99 9999.99 9999.99 9999.99 999.9 9999.9 999999.0\n";
        let records = parse_solar_orbiter_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan());
        assert!(r.b_magnitude.is_nan());
        assert!(r.proton_density.is_nan());
        assert!(r.bulk_speed.is_nan());
        assert!(r.proton_temperature.is_nan());
    }

    #[test]
    fn test_solo_to_omni_rtn_conversion() {
        let data = "2020 180 12 0.65 4.0 120.0 18.0 16.0 -6.0 4.0 35.0 360.0 250000.0\n";
        let spdf = parse_solar_orbiter_merged(data);
        let omni = solar_orbiter_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        let o = &omni[0];
        assert!((o.r_au - 0.65).abs() < 0.001);
        assert!((o.bx_gse - (-2.804)).abs() < 0.02);
        assert!((o.by_gse - 16.856).abs() < 0.02);
        assert!((o.bz_gse - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_solo_hapi_csv() {
        let data = "Time,radialDistance,heliographicLatitude,heliographicLongitude,BR,BT,BN,B,VR,VT,VN,ProtonSpeed,flow_theta,flow_lon,protonDensity,protonTemp\n2020-06-28T12:00:00Z,0.65,4.0,120.0,16.0,-6.0,4.0,18.0,350.0,5.0,1.0,360.0,0.0,0.0,35.0,250000.0\n";
        let rows = parse_solo_hapi_csv(data);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2020);
        assert_eq!(row.hour, 12);
        assert!((row.distance_au - 0.65).abs() < 0.001);
        assert!((row.br - 16.0).abs() < 0.001);
        assert!((row.proton_density - 35.0).abs() < 0.001);
    }
}
