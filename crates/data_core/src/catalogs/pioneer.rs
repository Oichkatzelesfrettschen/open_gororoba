//! Pioneer 10 & 11 merged hourly data parser.
//!
//! Pioneer spacecraft fill the 5-80 AU gap in heliospheric coverage:
//!   P10: launched 1972, first to cross Jupiter (1973) and Saturn orbit
//!        distance. Data to ~80 AU (signal lost 2003). Ecliptic plane.
//!   P11: launched 1973, Jupiter (1974) + Saturn flyby (1979).
//!        Data to ~45 AU. Inclined ~17 deg from ecliptic.
//!
//! Pioneer anomaly context: unexplained ~8.7e-10 m/s^2 sunward
//! acceleration observed on both spacecraft at r > 20 AU.
//! Now attributed to asymmetric thermal radiation but DM coupling
//! remains an open (if unlikely) hypothesis.
//!
//! Canonical NSSDC/SPDF/PDS merged hourly format columns (Pioneer 10/11):
//!   0: Year, 1: DOY, 2: Hour
//!   3: Heliocentric distance (AU)
//!   4: HGI latitude (deg)
//!   5: HGI longitude (deg)
//!   6: Br RTN (nT), 7: Bt RTN (nT), 8: Bn RTN (nT)
//!   9: |B| scalar (nT)
//!   10: Bulk speed (km/s)
//!   11: Flow theta (deg)
//!   12: Flow phi (deg)
//!   13: Proton density (cm^-3)
//!   14: Proton temperature (K)
//!
//! Fill values are derived from the FORTRAN field widths in `p10_mgd.txt` and
//! `p11mgd.txt`, which are also bundled inside reachable PDS/PPI encounter
//! subsets: 999.9999 (B and density), 9999.9 (speed), 9999999.0 (temperature),
//! 999.99 (distance), and 9999.9 for angular position/velocity fields.
//!
//! Coordinate systems:
//! - Spacecraft position: Heliographic Inertial (HGI)
//! - B-field: RTN (Radial-Tangential-Normal)
//!
//! Source: <https://spdf.gsfc.nasa.gov/pub/data/pioneer/>

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_fleet::SpdfMission,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord},
    },
    fetcher::FetchError,
};

/// SPDF column layout for Pioneer 10 merged hourly data.
pub const PIONEER10_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 15,
    col_year: 0,
    col_doy: 1,
    col_hour: 2,
    col_distance_au: Some(3),
    col_lat_deg: Some(4),
    col_lon_deg: Some(5),
    col_b_mag: Some(9),
    col_br: Some(6),
    col_bt: Some(7),
    col_bn: Some(8),
    col_density: Some(13),
    col_speed: Some(10),
    col_temperature: Some(14),
    fill_b: 999.9999,
    fill_density: 999.9999,
    fill_speed: 9999.9,
    fill_temperature: 9999999.0,
    fill_distance: 999.99,
    b_is_se: false,
};

/// SPDF column layout for Pioneer 11 merged hourly data.
/// Same format as Pioneer 10.
pub const PIONEER11_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 15,
    col_year: 0,
    col_doy: 1,
    col_hour: 2,
    col_distance_au: Some(3),
    col_lat_deg: Some(4),
    col_lon_deg: Some(5),
    col_b_mag: Some(9),
    col_br: Some(6),
    col_bt: Some(7),
    col_bn: Some(8),
    col_density: Some(13),
    col_speed: Some(10),
    col_temperature: Some(14),
    fill_b: 999.9999,
    fill_density: 999.9999,
    fill_speed: 9999.9,
    fill_temperature: 9999999.0,
    fill_distance: 999.99,
    b_is_se: false,
};

// Pioneer-specific fill values for reference.
/// B magnitude fill (nT).
pub const PIONEER_FILL_B_MAG: f64 = 999.9999;
/// B component fill (nT).
pub const PIONEER_FILL_B_COMP: f64 = 999.9999;
/// Density fill (cm^-3).
pub const PIONEER_FILL_DENSITY: f64 = 999.9999;
/// Speed fill (km/s).
pub const PIONEER_FILL_SPEED: f64 = 9999.9;
/// Temperature fill (K).
pub const PIONEER_FILL_TEMP: f64 = 9999999.0;
/// Distance fill (AU).
pub const PIONEER_FILL_DISTANCE: f64 = 999.99;

/// Which Pioneer spacecraft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PioneerSpacecraft {
    P10,
    P11,
}

/// Post-parse year fixup for Pioneer: 2-digit year (e.g. 72) -> 4-digit (1972).
fn pioneer_year_fixup(r: &mut SpdfMergedRecord) {
    if r.year < 100 {
        r.year += 1900;
    }
}

/// `SpdfMission` config for Pioneer 10 merged hourly data.
pub static PIONEER10_MISSION: SpdfMission = SpdfMission {
    layout: &PIONEER10_LAYOUT,
    b_is_se: false,
    year_fixup: Some(pioneer_year_fixup),
};

/// `SpdfMission` config for Pioneer 11 merged hourly data.
pub static PIONEER11_MISSION: SpdfMission = SpdfMission {
    layout: &PIONEER11_LAYOUT,
    b_is_se: false,
    year_fixup: Some(pioneer_year_fixup),
};

/// Parse Pioneer merged hourly data from a string.
pub fn parse_pioneer_merged(content: &str, spacecraft: PioneerSpacecraft) -> Vec<SpdfMergedRecord> {
    let mission = match spacecraft {
        PioneerSpacecraft::P10 => &PIONEER10_MISSION,
        PioneerSpacecraft::P11 => &PIONEER11_MISSION,
    };
    mission.parse_merged(content)
}

/// Parse Pioneer merged hourly data from a file.
pub fn parse_pioneer_file(
    path: &std::path::Path,
    spacecraft: PioneerSpacecraft,
) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let mission = match spacecraft {
        PioneerSpacecraft::P10 => &PIONEER10_MISSION,
        PioneerSpacecraft::P11 => &PIONEER11_MISSION,
    };
    mission.parse_file(path)
}

/// Convert Pioneer records to OmniRecord format.
///
/// Sets r_au, lat_deg, lon_deg from parsed position columns.
/// B-field is in RTN coordinates and is mapped through the shared RTN->GSE
/// adapter path used by other radially-outbound deep-space spacecraft.
/// Both P10 and P11 use RTN coordinates, so no spacecraft dispatch is needed here.
pub fn pioneer_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    // Both PIONEER10_MISSION and PIONEER11_MISSION have b_is_se=false; same result.
    PIONEER10_MISSION.to_omni(records)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pioneer10_layout_validity() {
        assert_eq!(PIONEER10_LAYOUT.min_columns, 15);
        const { assert!(!PIONEER10_LAYOUT.b_is_se) };
        assert_eq!(PIONEER10_LAYOUT.col_distance_au, Some(3));
    }

    #[test]
    fn test_pioneer11_layout_validity() {
        assert_eq!(PIONEER11_LAYOUT.min_columns, 15);
        const { assert!(!PIONEER11_LAYOUT.b_is_se) };
    }

    #[test]
    fn test_parse_pioneer10_inner_heliosphere() {
        // P10 at ~10 AU: B ceiling = 200/100 = 2.0, density ceiling = 500/100 = 5.0
        let data = "1975 100 12 10.0 1.0 180.0 0.3 -0.1 0.05 0.5 400.0 5.0 180.0 0.5 50000.0\n";
        let records = parse_pioneer_merged(data, PioneerSpacecraft::P10);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 10.0).abs() < 0.1);
        assert!((r.b_magnitude - 0.5).abs() < 0.01);
        assert!((r.proton_density - 0.5).abs() < 0.01);
        assert!((r.bulk_speed - 400.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_pioneer10_outer_heliosphere() {
        // P10 at ~60 AU: B ceiling = 200/3600 = 0.056, density ceiling = 500/3600 = 0.139
        let data = "1993 200 6 60.0 3.0 150.0 0.02 -0.01 0.005 0.03 400.0 1.0 180.0 0.01 10000.0\n";
        let records = parse_pioneer_merged(data, PioneerSpacecraft::P10);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 60.0).abs() < 0.1);
        assert!(!r.b_magnitude.is_nan());
        assert!(!r.proton_density.is_nan());
    }

    #[test]
    fn test_parse_pioneer11_saturn_distance() {
        // P11 near Saturn (~9.5 AU), inclined ~17 deg from ecliptic
        // B ceiling = 200/90.25 = 2.22, density ceiling = 500/90.25 = 5.54
        let data = "1979 245 12 9.5 -15.0 200.0 0.4 -0.2 0.1 0.8 450.0 -5.0 180.0 0.8 80000.0\n";
        let records = parse_pioneer_merged(data, PioneerSpacecraft::P11);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 9.5).abs() < 0.1);
        assert!((r.lat_deg - (-15.0)).abs() < 0.1);
        assert!((r.bulk_speed - 450.0).abs() < 0.1);
    }

    #[test]
    fn test_pioneer_fill_values() {
        let data = "1980 100 12 999.99 9999.9 9999.9 999.9999 999.9999 999.9999 999.9999 9999.9 9999.9 9999.9 999.9999 9999999.0\n";
        let records = parse_pioneer_merged(data, PioneerSpacecraft::P10);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan());
        assert!(r.lat_deg.is_nan());
        assert!(r.lon_deg.is_nan());
        assert!(r.b_magnitude.is_nan());
        assert!(r.proton_density.is_nan());
        assert!(r.bulk_speed.is_nan());
        assert!(r.proton_temperature.is_nan());
    }

    #[test]
    fn test_pioneer_to_omni_populates_r_au() {
        let data = "1975 200 12 10.0 1.0 180.0 0.3 -0.1 0.05 0.5 400.0 5.0 180.0 0.5 50000.0\n";
        let spdf = parse_pioneer_merged(data, PioneerSpacecraft::P10);
        let omni = pioneer_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        assert!(
            (omni[0].r_au - 10.0).abs() < 0.1,
            "r_au should be populated"
        );
    }

    #[test]
    fn test_pioneer_to_omni_rtn_to_gse() {
        let rec = SpdfMergedRecord {
            year: 1975,
            doy: 200,
            hour: 12,
            distance_au: 10.0,
            lat_deg: 1.0,
            lon_deg: 180.0,
            b_magnitude: 0.5,
            br: 0.3,
            bt: -0.1,
            bn: 0.05,
            proton_density: 0.5,
            bulk_speed: 400.0,
            proton_temperature: 50000.0,
        };
        let omni = pioneer_to_omni(&[rec]);
        // RTN -> GSE via rotation by spacecraft longitude (180 deg)
        assert!((omni[0].bx_gse - (-0.3)).abs() < 1e-6);
        assert!((omni[0].by_gse - 0.1).abs() < 1e-6);
        assert!((omni[0].bz_gse - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_parse_reachable_p11_saturn_encounter_row() {
        let data = " 79 212  0   9.21   -5.0   90.6   0.0265  -0.3685   0.0334   0.3755  394.9   -1.8   -2.6   0.0850   30073.\n";
        let records = parse_pioneer_merged(data, PioneerSpacecraft::P11);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert_eq!(r.year, 1979);
        assert!((r.distance_au - 9.21).abs() < 0.01);
        assert!((r.b_magnitude - 0.3755).abs() < 1e-6);
        assert!((r.bulk_speed - 394.9).abs() < 0.1);
        assert!((r.proton_density - 0.0850).abs() < 1e-6);
    }
}
