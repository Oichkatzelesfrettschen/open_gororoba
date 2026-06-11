//! Parker Solar Probe merged hourly data parser.
//!
//! PSP provides the innermost heliospheric sampling (0.05-0.25 AU perihelion):
//!   Launched 2018-08-12, first perihelion 2018-11-05
//!   Instruments: SWEAP (Solar Wind Electrons Alphas and Protons),
//!                FIELDS (electromagnetic fields and waves)
//!
//! Official source:
//!   PSP_COHO1HR_MERGED_MAG_PLASMA via CDAWeb HAPI
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=PSP_COHO1HR_MERGED_MAG_PLASMA>
//!
//! B-field coordinate system: RTN (Radial-Tangential-Normal).
//!
//! Fetch/provider support is in `psp_fetch` (feature-gated on `fetch`).

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord, parse_spdf_merged, spdf_to_omni},
    },
    fetcher::FetchError,
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;

/// Column layout for PSP AMDA-derived merged hourly data.
///
/// The 13-column format matches the standard SPDF merged convention
/// used by Voyager/Pioneer, with RTN B-field coordinates from SPC/FIELDS.
pub const PSP_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
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

/// PSP B magnitude fill (nT).
pub const PSP_FILL_B_MAG: f64 = 9999.99;
/// PSP density fill (cm^-3).
pub const PSP_FILL_DENSITY: f64 = 999.9;
/// PSP speed fill (km/s).
pub const PSP_FILL_SPEED: f64 = 9999.9;
/// PSP temperature fill (K).
pub const PSP_FILL_TEMP: f64 = 999999.0;
/// PSP distance fill (AU).
pub const PSP_FILL_DISTANCE: f64 = 999.999;

/// Parse PSP merged hourly data from a string.
pub fn parse_psp_merged(content: &str) -> Vec<SpdfMergedRecord> {
    parse_spdf_merged(content, &PSP_LAYOUT)
}

#[cfg(feature = "fetch")]
pub use super::psp_fetch::PspProvider;

/// Parse PSP merged hourly data from a file.
pub fn parse_psp_file(path: &std::path::Path) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    if path.extension().and_then(|value| value.to_str()) == Some("csv") {
        Ok(parse_psp_hapi_csv(&content))
    } else {
        Ok(parse_psp_merged(&content))
    }
}

/// Convert PSP records to OmniRecord format.
///
/// B-field is in RTN coordinates (sign flip on Bt for GSE conversion).
pub fn psp_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    spdf_to_omni(records, false) // RTN coordinates
}

pub fn parse_psp_hapi_csv(content: &str) -> Vec<SpdfMergedRecord> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psp_layout_validity() {
        assert_eq!(PSP_LAYOUT.min_columns, 13);
        const { assert!(!PSP_LAYOUT.b_is_se) };
        assert_eq!(PSP_LAYOUT.col_distance_au, Some(3));
        assert_eq!(PSP_LAYOUT.col_b_mag, Some(6));
    }

    #[test]
    fn test_parse_psp_inner_heliosphere() {
        // PSP at 0.1 AU perihelion: B ~ 100 nT, n ~ 300 cm^-3, V ~ 300 km/s
        // B ceiling at 0.1 AU: 200/(0.01) = 20000 nT -- no practical ceiling issue
        // Density ceiling at 0.1 AU: 500/(0.01) = 50000 cm^-3 -- no issue
        let data = "2022 310 12 0.100 3.5 120.0 95.0 80.0 -40.0 20.0 300.0 350.0 500000.0\n";
        let records = parse_psp_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 0.1).abs() < 0.001);
        assert!((r.b_magnitude - 95.0).abs() < 0.1);
        assert!((r.proton_density - 300.0).abs() < 0.1);
        assert!((r.bulk_speed - 350.0).abs() < 0.1);
        assert!((r.proton_temperature - 500000.0).abs() < 1.0);
    }

    #[test]
    fn test_psp_fill_values() {
        let data = "2022 310 12 999.999 999.99 999.99 9999.99 9999.99 9999.99 9999.99 999.9 9999.9 999999.0\n";
        let records = parse_psp_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan());
        assert!(r.b_magnitude.is_nan());
        assert!(r.proton_density.is_nan());
        assert!(r.bulk_speed.is_nan());
        assert!(r.proton_temperature.is_nan());
    }

    #[test]
    fn test_psp_to_omni_rtn_conversion() {
        let data = "2022 310 12 0.100 3.5 120.0 95.0 80.0 -40.0 20.0 300.0 350.0 500000.0\n";
        let spdf = parse_psp_merged(data);
        let omni = psp_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        let o = &omni[0];
        // RTN -> GSE via rotation by spacecraft longitude (120 deg)
        assert!((o.bx_gse - (-5.359)).abs() < 0.01);
        assert!((o.by_gse - 89.282).abs() < 0.01);
        assert!((o.bz_gse - 20.0).abs() < 0.1);
        assert!((o.r_au - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_parse_psp_hapi_csv() {
        let data = "Time,radialDistance,heliographicLatitude,heliographicLongitude,BR,BT,BN,B,VR,VT,VN,ProtonSpeed,flow_theta,flow_lon,protonDensity,protonTemp\n2020-01-01T00:00:00Z,0.17,1.5,120.0,15.0,-4.0,2.0,15.7,300.0,5.0,1.0,300.0,0.0,0.0,120.0,450000.0\n";
        let rows = parse_psp_hapi_csv(data);
        assert_eq!(rows.len(), 1);
        assert!((rows[0].distance_au - 0.17).abs() < 0.001);
        assert!((rows[0].proton_density - 120.0).abs() < 0.01);
        assert!((rows[0].bulk_speed - 300.0).abs() < 0.01);
    }
}
