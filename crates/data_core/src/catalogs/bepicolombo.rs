//! BepiColombo heliocentric position support data parser.
//!
//! Official source:
//!   BEPICOLOMBO_HELIO1HR_POSITION via CDAWeb HAPI
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=BEPICOLOMBO_HELIO1HR_POSITION>
//!
//! This lane currently provides trajectory support only. Plasma and magnetic
//! field channels remain absent here until a governed official merged product
//! is added to the Rust fetch path.
//!
//! Fetch/provider support is in `bepicolombo_fetch` (feature-gated on `fetch`).

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_fleet::SpdfMission,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord},
    },
    fetcher::FetchError,
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;

/// SPDF-style layout for BepiColombo position-only HAPI support data.
pub const BEPICOLOMBO_POSITION_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 6,
    col_year: 0,
    col_doy: 1,
    col_hour: 2,
    col_distance_au: Some(3),
    col_lat_deg: Some(4),
    col_lon_deg: Some(5),
    col_b_mag: None,
    col_br: None,
    col_bt: None,
    col_bn: None,
    col_density: None,
    col_speed: None,
    col_temperature: None,
    fill_b: 9999.9,
    fill_density: 9999.9,
    fill_speed: 9999.9,
    fill_temperature: 9999999.0,
    fill_distance: 999.999,
    b_is_se: true,
};

/// `SpdfMission` config for BepiColombo trajectory support rows.
pub static BEPICOLOMBO_MISSION: SpdfMission = SpdfMission {
    layout: &BEPICOLOMBO_POSITION_LAYOUT,
    b_is_se: true,
    year_fixup: None,
};

/// Parse BepiColombo position support rows from HAPI CSV.
pub fn parse_bepicolombo_position_hapi_csv(content: &str) -> Vec<SpdfMergedRecord> {
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
            b_magnitude: f64::NAN,
            br: f64::NAN,
            bt: f64::NAN,
            bn: f64::NAN,
            proton_density: f64::NAN,
            bulk_speed: f64::NAN,
            proton_temperature: f64::NAN,
        });
    }
    rows
}

#[cfg(feature = "fetch")]
pub use super::bepicolombo_fetch::BepicolomboProvider;

/// Parse a BepiColombo position file from disk.
pub fn parse_bepicolombo_file(path: &std::path::Path) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    Ok(parse_bepicolombo_position_hapi_csv(&content))
}

/// Convert BepiColombo position rows to the shared overlay format.
pub fn bepicolombo_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    BEPICOLOMBO_MISSION.to_omni(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bepicolombo_hapi_csv() {
        let data = "Time,RAD_AU,HG_LAT,HG_LON\n2020-01-01T00:00:00Z,0.95,1.25,44.0\n";
        let rows = parse_bepicolombo_position_hapi_csv(data);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2020);
        assert_eq!(row.doy, 1);
        assert_eq!(row.hour, 0);
        assert!((row.distance_au - 0.95).abs() < 1e-12);
        assert!((row.lat_deg - 1.25).abs() < 1e-12);
        assert!((row.lon_deg - 44.0).abs() < 1e-12);
        assert!(row.b_magnitude.is_nan());
    }

    #[test]
    fn test_parse_bepicolombo_hapi_fill_values() {
        let data = "Time,RAD_AU,HG_LAT,HG_LON\n2020-01-01T00:00:00Z,-1.0E31,-1.0E31,-1.0E31\n";
        let rows = parse_bepicolombo_position_hapi_csv(data);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert!(row.distance_au.is_nan());
        assert!(row.lat_deg.is_nan());
        assert!(row.lon_deg.is_nan());
    }
}
