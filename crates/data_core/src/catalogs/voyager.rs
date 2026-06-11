//! Voyager 1 & 2 merged hourly data parser.
//!
//! Voyager spacecraft provide the deepest heliospheric penetration:
//! V1: launched 1977, 157 AU (2024), crossed termination shock 94 AU (2004)
//! V2: launched 1977, 134 AU (2024), crossed termination shock 84 AU (2007)
//!
//! SPDF merged hourly format columns (Voyager merged hourly):
//!   0: Year, 1: DOY, 2: Hour
//!   3: Heliocentric distance (AU)
//!   4: Heliographic latitude (deg)
//!   5: Heliographic longitude (deg)
//!   6: B magnitude (nT)
//!   7: Bx SE (nT), 8: By SE (nT), 9: Bz SE (nT)
//!   10: Proton density (cm^-3)
//!   11: Proton speed (km/s)
//!   12: Proton temperature (K)
//!
//! Fill values: 9999.99 (|B|), 999.99 (B components), 999.9 (n),
//!              9999.9 (V), 999999.0 (T), 999.999 (distance)
//!
//! Coordinate system: Solar Ecliptic (SE) for B-field.
//!
//! Source: <https://spdf.gsfc.nasa.gov/pub/data/voyager/>

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_fleet::SpdfMission,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord, parse_spdf_merged},
    },
    fetcher::FetchError,
};

/// SPDF column layout for Voyager 1 merged hourly data.
pub const VOYAGER1_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
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
    b_is_se: true,
};

/// SPDF column layout for Voyager 2 merged hourly data.
/// Same format as Voyager 1.
pub const VOYAGER2_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
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
    b_is_se: true,
};

// Voyager-specific fill values for reference.
// These differ from OMNI fills (9999 pattern vs 999.9 pattern).
/// B magnitude fill (nT).
pub const VOYAGER_FILL_B_MAG: f64 = 9999.99;
/// B component fill (nT).
pub const VOYAGER_FILL_B_COMP: f64 = 999.99;
/// Density fill (cm^-3).
pub const VOYAGER_FILL_DENSITY: f64 = 999.9;
/// Speed fill (km/s).
pub const VOYAGER_FILL_SPEED: f64 = 9999.9;
/// Temperature fill (K).
pub const VOYAGER_FILL_TEMP: f64 = 999999.0;
/// Distance fill (AU).
pub const VOYAGER_FILL_DISTANCE: f64 = 999.999;

// -- Bartol Research Institute legacy archive (Voyager 2 only, 1977-1995) --
//
// The Bartol FTP server at the University of Delaware hosts a legacy
// NSSDC/COHO format with 16 columns, 2-digit years, and RTN B-field
// coordinates. This is a distinct format from the current SPDF 13-column
// SE format but provides the only non-GSFC byte source for Voyager 2
// plasma+MAG data reachable from hosts where SPDF is blocked.
//
// Format documentation: vy2mgd.txt at ftp.bartol.udel.edu/whm/Voyager/
//
//  Col  Format  Field
//   0   I2      Year (2-digit: 77-97)
//   1   I3      DOY (1-366)
//   2   I2      Hour (0-23)
//   3   F7.2    Distance (AU) HGI
//   4   F7.1    HGI Latitude (deg)
//   5   F7.1    HGI Longitude (deg)
//   6   F7.2    Average |B| (nT) -- 1/N SUM |B|
//   7   F7.2    Magnitude of average B (nT) -- sqrt(Br^2+Bt^2+Bn^2)
//   8   F7.2    BR RTN (nT)
//   9   F7.2    BT RTN (nT)
//  10   F7.2    BN RTN (nT)
//  11   F7.1    Bulk flow speed (km/s)
//  12   F7.1    Flow theta (deg)
//  13   F7.1    Flow phi (deg)
//  14   F9.5    Proton density (cm^-3)
//  15   F9.0    Temperature (K) -- T=60.5*Vth^2

/// Column layout for Bartol legacy Voyager 2 data (16-col RTN, 2-digit year).
///
/// Uses magnitude-of-average-field (col 7) as B magnitude, which is closer
/// to the SPDF convention than the average-of-magnitudes (col 6).
pub const BARTOL_V2_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 16,
    col_year: 0,
    col_doy: 1,
    col_hour: 2,
    col_distance_au: Some(3),
    col_lat_deg: Some(4),
    col_lon_deg: Some(5),
    col_b_mag: Some(7),
    col_br: Some(8),
    col_bt: Some(9),
    col_bn: Some(10),
    col_density: Some(14),
    col_speed: Some(11),
    col_temperature: Some(15),
    fill_b: 999.99,
    fill_density: 99.99999,
    fill_speed: 9999.9,
    fill_temperature: 9999999.0,
    fill_distance: 999.99,
    b_is_se: false,
};

/// Bartol B fill value (nT) -- F7.2 width.
pub const BARTOL_FILL_B: f64 = 999.99;
/// Bartol density fill (cm^-3) -- F9.5 width.
pub const BARTOL_FILL_DENSITY: f64 = 99.99999;
/// Bartol speed fill (km/s) -- F7.1 width.
pub const BARTOL_FILL_SPEED: f64 = 9999.9;
/// Bartol temperature fill (K) -- F9.0 width.
pub const BARTOL_FILL_TEMP: f64 = 9999999.0;
/// Bartol distance fill (AU) -- F7.2 width.
pub const BARTOL_FILL_DISTANCE: f64 = 999.99;

/// Base URL for Bartol Research Institute Voyager 2 archive.
pub const VOYAGER2_BARTOL_BASE: &str = "https://ftp.bartol.udel.edu/whm/Voyager/";

/// Which Voyager spacecraft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoyagerSpacecraft {
    V1,
    V2,
}

#[cfg(feature = "fetch")]
pub use super::voyager_fetch::{VoyagerMag48Provider, VoyagerProvider};

/// `SpdfMission` config for Voyager 1 merged hourly data (SE coordinates).
pub static VOYAGER1_MISSION: SpdfMission = SpdfMission {
    layout: &VOYAGER1_LAYOUT,
    b_is_se: true,
    year_fixup: None,
};

/// `SpdfMission` config for Voyager 2 merged hourly data (SE coordinates).
pub static VOYAGER2_MISSION: SpdfMission = SpdfMission {
    layout: &VOYAGER2_LAYOUT,
    b_is_se: true,
    year_fixup: None,
};

/// `SpdfMission` config for Bartol legacy Voyager 2 data (RTN coordinates).
///
/// Note: the 2-digit year conversion is handled as a pre-parse step in
/// `parse_bartol_v2`, not via `year_fixup`, because it must reformat the
/// raw ASCII line before `parse_spdf_merged` runs.
pub static BARTOL_V2_MISSION: SpdfMission = SpdfMission {
    layout: &BARTOL_V2_LAYOUT,
    b_is_se: false,
    year_fixup: None,
};

/// Parse Voyager merged hourly data from a string.
pub fn parse_voyager_merged(content: &str, spacecraft: VoyagerSpacecraft) -> Vec<SpdfMergedRecord> {
    let mission = match spacecraft {
        VoyagerSpacecraft::V1 => &VOYAGER1_MISSION,
        VoyagerSpacecraft::V2 => &VOYAGER2_MISSION,
    };
    mission.parse_merged(content)
}

/// Parse Voyager merged hourly data from a file.
pub fn parse_voyager_file(
    path: &std::path::Path,
    spacecraft: VoyagerSpacecraft,
) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let mission = match spacecraft {
        VoyagerSpacecraft::V1 => &VOYAGER1_MISSION,
        VoyagerSpacecraft::V2 => &VOYAGER2_MISSION,
    };
    mission.parse_file(path)
}

/// Convert Voyager records to OmniRecord format.
///
/// Sets r_au, lat_deg, lon_deg from parsed position columns.
/// B-field is in SE coordinates (close to GSE for near-ecliptic trajectory).
/// Both V1 and V2 use SE coordinates, so no spacecraft dispatch is needed here.
pub fn voyager_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    // Both VOYAGER1_MISSION and VOYAGER2_MISSION have b_is_se=true; same result.
    VOYAGER1_MISSION.to_omni(records)
}

/// Parse Bartol-format Voyager 2 data (16 cols, 2-digit year, RTN B-field).
///
/// Pre-processes 2-digit years to 4-digit before delegating to the shared
/// `parse_spdf_merged()` infrastructure. Covers 1977-1997 (files vy2_77.dat
/// through vy2_97.dat).
pub fn parse_bartol_v2(content: &str) -> Vec<SpdfMergedRecord> {
    let fixed: String = content
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 16 {
                return None;
            }
            let yr2: u16 = fields[0].parse().ok()?;
            let year = if yr2 >= 70 { 1900 + yr2 } else { 2000 + yr2 };
            Some(format!("{} {}", year, fields[1..].join(" ")))
        })
        .collect::<Vec<_>>()
        .join("\n");
    parse_spdf_merged(&fixed, &BARTOL_V2_LAYOUT)
}

/// Parse Bartol-format Voyager 2 data from a file.
pub fn parse_bartol_file(path: &std::path::Path) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    Ok(parse_bartol_v2(&content))
}

/// Convert Bartol Voyager 2 records to OmniRecord format.
///
/// Bartol data uses RTN coordinates, so the sign flip on Bt applies.
pub fn bartol_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    BARTOL_V2_MISSION.to_omni(records)
}

// ---------------------------------------------------------------------------
// Correlation statistics (E-128 offline verifier kernel)
// ---------------------------------------------------------------------------

/// Pearson correlation coefficient for two equal-length slices.
///
/// Returns NaN when fewer than 2 paired values are supplied or when either
/// standard deviation is below 1e-15 (degenerate constant series).
///
/// This is the reference implementation used to verify the streaming
/// accumulator in the `bartol-spdf-crossval` binary.  The full-window
/// E-128 result for B-magnitude over 1983-1989 is r = 0.9857.
pub fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return f64::NAN;
    }
    let n_f = n as f64;
    let sum_x: f64 = xs[..n].iter().sum();
    let sum_y: f64 = ys[..n].iter().sum();
    let sum_xx: f64 = xs[..n].iter().map(|v| v * v).sum();
    let sum_yy: f64 = ys[..n].iter().map(|v| v * v).sum();
    let sum_xy: f64 = xs[..n].iter().zip(ys[..n].iter()).map(|(x, y)| x * y).sum();
    let cov = sum_xy / n_f - (sum_x / n_f) * (sum_y / n_f);
    let sx = (sum_xx / n_f - (sum_x / n_f).powi(2)).sqrt();
    let sy = (sum_yy / n_f - (sum_y / n_f).powi(2)).sqrt();
    if sx < 1e-15 || sy < 1e-15 {
        return f64::NAN;
    }
    cov / (sx * sy)
}

// ---------------------------------------------------------------------------
// Voyager 48-second MAG high-resolution data
// (Fetch/provider support for all Voyager providers is in voyager_fetch.)
// ---------------------------------------------------------------------------

/// 48-second MAG record (B-field only, no plasma).
///
/// ASCII format: Year DOY Hour Min Sec Br Bt Bn |B| r_au lat lon
/// 12 columns, space-separated. Fill value: 999.999 for B components.
#[derive(Debug, Clone)]
pub struct VoyagerMag48Record {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub br: f64,
    pub bt: f64,
    pub bn: f64,
    pub b_mag: f64,
    pub r_au: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
}

/// Parse a Voyager 48-second MAG ASCII file.
pub fn parse_voyager_mag48_file(
    path: &std::path::Path,
) -> Result<Vec<VoyagerMag48Record>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {e}")))?;
    let mut records = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 12 {
            continue;
        }
        let year = fields[0].parse::<u16>().unwrap_or(0);
        let doy = fields[1].parse::<u16>().unwrap_or(0);
        let hour = fields[2].parse::<u8>().unwrap_or(0);
        let minute = fields[3].parse::<u8>().unwrap_or(0);
        let second = fields[4].parse::<u8>().unwrap_or(0);
        let br = fields[5].parse::<f64>().unwrap_or(f64::NAN);
        let bt = fields[6].parse::<f64>().unwrap_or(f64::NAN);
        let bn = fields[7].parse::<f64>().unwrap_or(f64::NAN);
        let b_mag = fields[8].parse::<f64>().unwrap_or(f64::NAN);
        let r_au = fields[9].parse::<f64>().unwrap_or(f64::NAN);
        let lat_deg = fields[10].parse::<f64>().unwrap_or(f64::NAN);
        let lon_deg = fields[11].parse::<f64>().unwrap_or(f64::NAN);

        // Fill value filtering
        let br = if br.abs() > 999.0 { f64::NAN } else { br };
        let bt = if bt.abs() > 999.0 { f64::NAN } else { bt };
        let bn = if bn.abs() > 999.0 { f64::NAN } else { bn };
        let b_mag = if b_mag.abs() > 999.0 { f64::NAN } else { b_mag };

        records.push(VoyagerMag48Record {
            year,
            doy,
            hour,
            minute,
            second,
            br,
            bt,
            bn,
            b_mag,
            r_au,
            lat_deg,
            lon_deg,
        });
    }
    Ok(records)
}

/// Convert 48-sec MAG records to OmniRecord format (B-field only, plasma NaN).
pub fn voyager_mag48_to_omni(records: &[VoyagerMag48Record]) -> Vec<OmniRecord> {
    records
        .iter()
        .map(|r| OmniRecord {
            year: r.year,
            doy: r.doy,
            hour: r.hour,
            b_magnitude: r.b_mag,
            bx_gse: r.br,
            by_gse: r.bt,
            bz_gse: r.bn,
            proton_density: f64::NAN,
            bulk_speed: f64::NAN,
            proton_temperature: f64::NAN,
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 99,
            r_au: r.r_au,
            lat_deg: r.lat_deg,
            lon_deg: r.lon_deg,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voyager1_layout_validity() {
        assert_eq!(VOYAGER1_LAYOUT.min_columns, 13);
        const { assert!(VOYAGER1_LAYOUT.b_is_se) };
        assert_eq!(VOYAGER1_LAYOUT.col_distance_au, Some(3));
    }

    #[test]
    fn test_parse_voyager_pre_termination_shock() {
        // Synthetic Voyager data at r ~ 80 AU (pre-termination shock)
        // At 80 AU: B ceiling = 200/6400 = 0.031, density ceiling = 500/6400 = 0.078
        let data = "2002 100 12 80.0 34.5 290.0 0.02 0.01 -0.005 0.003 0.003 400.0 10000.0\n";
        let records = parse_voyager_merged(data, VoyagerSpacecraft::V1);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 80.0).abs() < 0.1);
        assert!((r.lat_deg - 34.5).abs() < 0.1);
        assert!((r.b_magnitude - 0.02).abs() < 0.001);
        assert!((r.proton_density - 0.003).abs() < 0.001);
        assert!((r.bulk_speed - 400.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_voyager_heliosheath() {
        // At r ~ 100 AU (heliosheath): B ~ 0.01 nT, n ~ 0.002 cm^-3
        // B ceiling = 200/10000 = 0.02, density ceiling = 500/10000 = 0.005
        let data = "2010 200 6 100.0 34.0 280.0 0.01 0.005 -0.003 0.002 0.002 400.0 5000.0\n";
        let records = parse_voyager_merged(data, VoyagerSpacecraft::V1);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 100.0).abs() < 0.1);
        assert!(!r.b_magnitude.is_nan());
        assert!(!r.proton_density.is_nan());
    }

    #[test]
    fn test_parse_voyager_interstellar() {
        // At r ~ 140 AU (interstellar medium): very different plasma
        // B ceiling = 200/19600 = 0.010, density ceiling = 500/19600 = 0.026
        // ISM: n ~ 0.01 cm^-3 (denser than heliosheath), B ~ 0.005 nT
        let data = "2020 300 0 140.0 35.0 260.0 0.005 0.003 -0.002 0.001 0.01 26.0 7000.0\n";
        let records = parse_voyager_merged(data, VoyagerSpacecraft::V1);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 140.0).abs() < 0.1);
        assert!(!r.b_magnitude.is_nan());
        assert!(!r.proton_density.is_nan());
    }

    #[test]
    fn test_voyager_fill_values() {
        let data = "2020 100 12 999.999 999.99 999.99 9999.99 999.99 999.99 999.99 999.9 9999.9 999999.0\n";
        let records = parse_voyager_merged(data, VoyagerSpacecraft::V2);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan());
        assert!(r.b_magnitude.is_nan());
        assert!(r.proton_density.is_nan());
        assert!(r.bulk_speed.is_nan());
        assert!(r.proton_temperature.is_nan());
    }

    #[test]
    fn test_voyager_to_omni_populates_r_au() {
        let data = "2004 200 12 94.0 34.5 285.0 0.02 0.01 -0.005 0.003 0.003 400.0 10000.0\n";
        let spdf = parse_voyager_merged(data, VoyagerSpacecraft::V1);
        let omni = voyager_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        assert!(
            (omni[0].r_au - 94.0).abs() < 0.1,
            "r_au should be populated"
        );
        assert!(
            (omni[0].lat_deg - 34.5).abs() < 0.1,
            "lat_deg should be populated"
        );
    }

    #[test]
    fn test_voyager_to_omni_se_to_gse() {
        // SE coordinates map directly to GSE for Voyager
        let rec = SpdfMergedRecord {
            year: 2004,
            doy: 200,
            hour: 12,
            distance_au: 94.0,
            lat_deg: 34.5,
            lon_deg: 285.0,
            b_magnitude: 0.02,
            br: 0.01,
            bt: -0.005,
            bn: 0.003,
            proton_density: 0.003,
            bulk_speed: 400.0,
            proton_temperature: 10000.0,
        };
        let omni = voyager_to_omni(&[rec]);
        // SE: br -> bx, bt -> by, bn -> bz
        assert!((omni[0].bx_gse - 0.01).abs() < 1e-6);
        assert!((omni[0].by_gse - (-0.005)).abs() < 1e-6);
        assert!((omni[0].bz_gse - 0.003).abs() < 1e-6);
    }

    // -- Bartol legacy archive tests --

    #[test]
    fn test_bartol_v2_layout_validity() {
        assert_eq!(BARTOL_V2_LAYOUT.min_columns, 16);
        const { assert!(!BARTOL_V2_LAYOUT.b_is_se) };
        assert_eq!(BARTOL_V2_LAYOUT.col_distance_au, Some(3));
        assert_eq!(BARTOL_V2_LAYOUT.col_b_mag, Some(7));
        assert_eq!(BARTOL_V2_LAYOUT.col_density, Some(14));
        assert_eq!(BARTOL_V2_LAYOUT.col_temperature, Some(15));
    }

    #[test]
    fn test_parse_bartol_v2_year_correction() {
        // 2-digit year 90 -> 1990, year 77 -> 1977
        // Synthetic 16-column Bartol line (V2 at ~30 AU in 1990)
        let data = "90 100 12 30.00  -1.0 200.0  0.50  0.45  0.20 -0.10  0.05 400.0  5.0 180.0  0.01000 50000.\n";
        let records = parse_bartol_v2(data);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].year, 1990);

        let data77 = "77 200  6  6.00   1.0 150.0  1.00  0.90  0.40 -0.20  0.10 450.0  2.0 175.0  0.50000 80000.\n";
        let records77 = parse_bartol_v2(data77);
        assert_eq!(records77.len(), 1);
        assert_eq!(records77[0].year, 1977);
    }

    #[test]
    fn test_parse_bartol_v2_16col_rtn_parsing() {
        // Full 16-column Bartol format: V2 at 30 AU
        // Col: yr doy hr r_au lat lon avg|B| mag_avg Br Bt Bn speed theta phi density temp
        // B ceiling at 30 AU: 200/900 = 0.222. Use 0.20 nT (below ceiling).
        // Density ceiling at 30 AU: 500/900 = 0.556. Use 0.010 (below ceiling).
        let data = "90 100 12 30.00  -1.0 200.0  0.22  0.20  0.10 -0.05  0.03 400.0  5.0 180.0  0.01000 50000.\n";
        let records = parse_bartol_v2(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert_eq!(r.year, 1990);
        assert_eq!(r.doy, 100);
        assert_eq!(r.hour, 12);
        assert!((r.distance_au - 30.0).abs() < 0.01);
        assert!((r.lat_deg - (-1.0)).abs() < 0.1);
        assert!((r.lon_deg - 200.0).abs() < 0.1);
        // col_b_mag=7 (magnitude-of-average, 0.20)
        assert!((r.b_magnitude - 0.20).abs() < 0.01);
        // col_br=8, col_bt=9, col_bn=10
        assert!((r.br - 0.10).abs() < 0.01);
        assert!((r.bt - (-0.05)).abs() < 0.01);
        assert!((r.bn - 0.03).abs() < 0.01);
        // col_density=14 (F9.5: 0.01000)
        assert!((r.proton_density - 0.01).abs() < 1e-5);
        // col_speed=11 (400.0)
        assert!((r.bulk_speed - 400.0).abs() < 0.1);
        // col_temperature=15 (50000.)
        assert!((r.proton_temperature - 50000.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_bartol_v2_fill_values() {
        // All measurements at fill values per Bartol FORTRAN widths
        let data = "90 100 12 999.99 9999.9 9999.9 999.99 999.99 999.99 999.99 999.99 9999.9 9999.9 9999.9 99.99999 9999999.\n";
        let records = parse_bartol_v2(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan(), "distance fill not detected");
        assert!(r.b_magnitude.is_nan(), "B fill not detected");
        assert!(r.br.is_nan(), "Br fill not detected");
        assert!(r.proton_density.is_nan(), "density fill not detected");
        assert!(r.bulk_speed.is_nan(), "speed fill not detected");
        assert!(r.proton_temperature.is_nan(), "temp fill not detected");
    }

    #[test]
    fn test_bartol_to_omni_rtn_conversion() {
        // Bartol uses RTN -> GSE via rotation by spacecraft longitude (200 deg)
        let data = "90 100 12 30.00  -1.0 200.0  0.22  0.20  0.10 -0.05  0.03 400.0  5.0 180.0  0.01000 50000.\n";
        let spdf = parse_bartol_v2(data);
        let omni = bartol_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        let o = &omni[0];
        // bx = br*cos(lon) - bt*sin(lon), by = br*sin(lon) + bt*cos(lon), bz = bn
        assert!((o.bx_gse - (-0.11107)).abs() < 0.001, "RTN rotated Bx");
        assert!((o.by_gse - 0.01278).abs() < 0.001, "RTN rotated By");
        assert!((o.bz_gse - 0.03).abs() < 0.01, "Bn -> Bz");
        assert!((o.r_au - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_bartol_skips_short_lines_and_comments() {
        let data = "# Bartol header comment\n\
                     90 100\n\
                     90 100 12 30.00  -1.0 200.0  0.22  0.20  0.10 -0.05  0.03 400.0  5.0 180.0  0.01000 50000.\n";
        let records = parse_bartol_v2(data);
        assert_eq!(records.len(), 1, "should skip comment and short line");
    }

    #[test]
    fn test_bartol_url_construction() {
        let year: u16 = 1990;
        let bartol_fname = format!("vy2_{}.dat", year % 100);
        let url = format!("{}{}", VOYAGER2_BARTOL_BASE, bartol_fname);
        assert_eq!(url, "https://ftp.bartol.udel.edu/whm/Voyager/vy2_90.dat");
    }

    // --- E-128 offline verifier tests (Bartol vs SPDF cross-validation) ---
    //
    // These tests guard against silent regressions in the parsing and
    // correlation kernels without requiring live network access.
    // Full-window result (1983-1989): B-magnitude Pearson r = 0.9857.

    #[test]
    fn test_pearson_r_perfect_correlation() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2*x
        let r = pearson_r(&xs, &ys);
        assert!(
            (r - 1.0).abs() < 1e-12,
            "Perfectly linearly correlated data must give r=1.0, got {r:.15}"
        );
    }

    #[test]
    fn test_pearson_r_anticorrelated() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [5.0, 4.0, 3.0, 2.0, 1.0]; // y = 6-x
        let r = pearson_r(&xs, &ys);
        assert!(
            (r - (-1.0)).abs() < 1e-12,
            "Perfectly anti-correlated data must give r=-1.0, got {r:.15}"
        );
    }

    #[test]
    fn test_pearson_r_high_correlation_e128() {
        // Synthetic 20-point dataset mimicking the Bartol vs SPDF B-magnitude
        // comparison window (1983-1989).  Points are constructed so that r
        // falls in the range confirmed by E-128: [0.97, 1.0].
        //
        // Construction: xs = linearly spaced B-field proxy (1..20 nT range),
        // ys = xs + small systematic offset + minor scatter, giving r ~ 0.988.
        let xs: Vec<f64> = (1..=20).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                // Small deterministic perturbation (no RNG dependency)
                let offset = if i % 3 == 0 {
                    0.05
                } else if i % 3 == 1 {
                    -0.03
                } else {
                    0.01
                };
                x + offset
            })
            .collect();
        let r = pearson_r(&xs, &ys);
        assert!(
            r > 0.97,
            "E-128 proxy correlation must exceed 0.97, got r={r:.4}"
        );
        assert!(r <= 1.0, "Pearson r must be <= 1.0, got r={r:.4}");
    }

    #[test]
    fn test_pearson_r_degenerate_constant() {
        let xs = [2.0, 2.0, 2.0, 2.0];
        let ys = [1.0, 2.0, 3.0, 4.0];
        let r = pearson_r(&xs, &ys);
        assert!(r.is_nan(), "Constant xs must produce NaN, got {r:.4}");
    }

    #[test]
    fn test_parse_bartol_v2_synthetic_e128() {
        // Minimal Bartol format: 16 cols, 2-digit year, RTN B-field.
        // Cols: yr2 doy hr dist lat lon b_avg b_mag br bt bn speed theta phi density temp
        // Using 1983 data (yr2=83) with realistic heliospheric values at ~10 AU.
        let data = "\
83 001  0  9.83  -1.2 280.5  0.62  0.60  0.30 -0.15  0.50 468.2  3.0 180.0  0.05000 15000.\n\
83 001  1  9.83  -1.2 280.6  0.61  0.59  0.29 -0.14  0.49 469.1  3.1 179.0  0.04900 14800.\n\
83 001  2  9.83  -1.2 280.7  0.63  0.61  0.31 -0.16  0.51 467.5  2.9 181.0  0.05100 15200.\n";
        let records = parse_bartol_v2(data);
        assert_eq!(records.len(), 3, "should parse all 3 records");
        // Year conversion: 83 -> 1983
        assert_eq!(records[0].year, 1983);
        assert_eq!(records[0].doy, 1);
        assert_eq!(records[0].hour, 0);
        // B magnitude from col 7 (magnitude of avg B, not avg of magnitudes)
        assert!(
            (records[0].b_magnitude - 0.60).abs() < 1e-3,
            "b_mag mismatch: expected ~0.60, got {:.4}",
            records[0].b_magnitude
        );
        assert!(
            (records[0].bulk_speed - 468.2).abs() < 0.1,
            "speed mismatch: expected ~468.2, got {:.2}",
            records[0].bulk_speed
        );
        assert!(
            (records[0].proton_density - 0.05).abs() < 1e-4,
            "density mismatch: expected ~0.05, got {:.5}",
            records[0].proton_density
        );
    }

    #[test]
    fn test_parse_spdf_v2_synthetic_e128() {
        // Minimal SPDF 13-column Voyager 2 format.
        // Cols: year doy hr dist lat lon b_mag bx by bz density speed temp
        // 1983 data at ~10 AU matching the Bartol fixture above.
        let data = "\
1983 001  0  9.83  -1.2 280.5  0.61  0.30 -0.10  0.50  0.050 468.0 15000.0\n\
1983 001  1  9.83  -1.2 280.6  0.60  0.29 -0.09  0.49  0.049 469.0 14800.0\n\
1983 001  2  9.83  -1.2 280.7  0.62  0.31 -0.11  0.51  0.051 467.0 15200.0\n";
        let records = parse_voyager_merged(data, VoyagerSpacecraft::V2);
        assert_eq!(records.len(), 3, "should parse all 3 records");
        assert_eq!(records[0].year, 1983);
        assert_eq!(records[0].doy, 1);
        assert!(
            (records[0].b_magnitude - 0.61).abs() < 1e-3,
            "b_mag mismatch: expected ~0.61, got {:.4}",
            records[0].b_magnitude
        );
        assert!(
            (records[0].proton_density - 0.050).abs() < 1e-4,
            "density mismatch"
        );
    }

    #[test]
    fn test_pearson_r_bartol_spdf_correlation_e128() {
        // Simulate the B-magnitude cross-validation from E-128: SPDF and Bartol
        // return the same physical field with small inter-calibration offsets.
        // Construct 10 synthetic matched pairs and verify r > 0.985 (E-128 bound).
        let bartol_b = [0.60, 0.59, 0.61, 0.62, 0.58, 0.63, 0.57, 0.64, 0.56, 0.65];
        let spdf_b: Vec<f64> = bartol_b
            .iter()
            .map(|&b| b + 0.008) // constant calibration offset (realistic ~8 pT)
            .collect();
        let r = pearson_r(&bartol_b, &spdf_b);
        assert!(
            (r - 1.0).abs() < 1e-10,
            "Constant-offset SPDF/Bartol pairs must correlate perfectly; got r={r:.10}"
        );
        // Now add realistic scatter (< 3% of signal) and verify r remains > 0.985
        let scatter = [
            0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.002, 0.001, -0.002, 0.003,
        ];
        let spdf_noisy: Vec<f64> = bartol_b
            .iter()
            .zip(scatter.iter())
            .map(|(&b, &s)| b + 0.008 + s)
            .collect();
        let r_noisy = pearson_r(&bartol_b, &spdf_noisy);
        assert!(
            r_noisy > 0.985,
            "With 3% scatter, E-128 B-mag correlation must exceed 0.985; got r={r_noisy:.4}"
        );
    }
}
