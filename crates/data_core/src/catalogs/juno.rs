//! Juno JADE/MAG cruise phase data parser.
//!
//! Juno provides plasma and magnetic field measurements during its
//! cruise phase (2011-2016) at 1-5 AU before entering Jupiter orbit.
//!
//! Instruments:
//!   JADE: Jovian Auroral Distributions Experiment
//!         McComas et al. (2017), Space Sci. Rev. 213, 547
//!   MAG: Fluxgate Magnetometer
//!         Connerney et al. (2017), Space Sci. Rev. 213, 39
//!
//! Cruise data is available as merged hourly from SPDF.
//! Column layout follows standard SPDF merged format.
//!   0: Year, 1: DOY, 2: Hour
//!   3: Heliocentric distance (AU)
//!   4: Heliographic latitude (deg)
//!   5: Heliographic longitude (deg)
//!   6: |B| (nT), 7: Br (nT), 8: Bt (nT), 9: Bn (nT)
//!   10: Proton density (cm^-3)
//!   11: Proton speed (km/s)
//!   12: Proton temperature (K)
//!
//! Coordinate system: SE (Solar Ecliptic) for B-field during cruise.
//!
//! Source: <https://spdf.gsfc.nasa.gov/pub/data/juno/>

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_fleet::SpdfMission,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord},
    },
    fetcher::{
        DatasetProvider, FetchConfig, FetchError, download_amda_hapi_csv, download_hapi_csv,
    },
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::path::PathBuf;

/// SPDF column layout for Juno cruise merged hourly data.
pub const JUNO_CRUISE_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
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

/// `SpdfMission` config for Juno cruise merged hourly data.
pub static JUNO_MISSION: SpdfMission = SpdfMission {
    layout: &JUNO_CRUISE_LAYOUT,
    b_is_se: true,
    year_fixup: None,
};

/// Parse Juno cruise merged hourly data from a string.
pub fn parse_juno_cruise(content: &str) -> Vec<SpdfMergedRecord> {
    JUNO_MISSION.parse_merged(content)
}

/// Parse Juno cruise merged hourly data from a file.
pub fn parse_juno_cruise_file(path: &std::path::Path) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    if path.extension().and_then(|value| value.to_str()) == Some("csv") {
        Ok(parse_juno_helio1hr_hapi_csv(&content))
    } else {
        Ok(parse_juno_cruise(&content))
    }
}

pub fn parse_juno_helio1hr_hapi_csv(content: &str) -> Vec<SpdfMergedRecord> {
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

/// Convert Juno cruise records to OmniRecord format.
pub fn juno_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    JUNO_MISSION.to_omni(records)
}

const JUNO_POSITION_HAPI_DATASET: &str = "JUNO_HELIO1HR_POSITION";

/// NASA SPDF Juno cruise dataset provider.
pub struct JunoCruiseProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for JunoCruiseProvider {
    fn default() -> Self {
        Self {
            year_start: 2011,
            year_end: 2016,
        }
    }
}

impl DatasetProvider for JunoCruiseProvider {
    fn name(&self) -> &str {
        "Juno Cruise Merged Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("juno");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("juno_helio1hr_position_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                JUNO_POSITION_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&["Time", "RAD_AU", "HG_LAT", "HG_LON"]),
            ) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download Juno {}: {}", year, e);
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("juno").exists()
    }
}

// ---------------------------------------------------------------------------
// AMDA fallback: three-dataset lane (protons + MAG + orbit)
// ---------------------------------------------------------------------------
//
// WHY: JUNO_HELIO1HR_POSITION (CDAWeb) carries only orbital data, and the
// full plasma+MAG merged product is not available as a single CDAWeb dataset.
// AMDA provides independent L2 lanes that cover the 2011-2016 cruise phase.
//
// Lane:
//   1. CDAWeb HAPI: JUNO_HELIO1HR_POSITION (position-only, existing provider).
//   2. AMDA HAPI: juno-jadel5-protmom (proton moments) + juno-fgm-cruise60
//      (MAG 60-min averages) + juno-cruise-all (ephemeris), then merged.

/// AMDA dataset ID for Juno JADE L5 proton moments (cruise phase).
const JUNO_AMDA_PLASMA: &str = "juno-jadel5-protmom";

/// AMDA dataset ID for Juno FGM cruise 60-min averages in RTN coordinates.
const JUNO_AMDA_MAG: &str = "juno-fgm-cruise60";

/// AMDA dataset ID for Juno cruise ephemeris.
const JUNO_AMDA_ORB: &str = "juno-cruise-all";

/// Parse Juno AMDA proton-moment CSV (`juno-jadel5-protmom`).
///
/// Returns (year, doy, hour, density_cm3, speed_kms, temperature_k).
fn parse_juno_amda_plasma(content: &str) -> Vec<(u16, u16, u8, f64, f64, f64)> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut out = Vec::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let density = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let speed = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let temp = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        out.push((year, doy, hour, density, speed, temp));
    }
    out
}

/// Parse Juno AMDA MAG cruise CSV (`juno-fgm-cruise60`).
///
/// Returns (year, doy, hour, br, bt, bn, b_mag).
fn parse_juno_amda_mag(content: &str) -> Vec<(u16, u16, u8, f64, f64, f64, f64)> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut out = Vec::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let br = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let bt = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let bn = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        let b_mag = parse_hapi_spacephysics_f64_or_nan(record.get(4).unwrap_or(""));
        out.push((year, doy, hour, br, bt, bn, b_mag));
    }
    out
}

/// Parse Juno AMDA orbit CSV (`juno-cruise-all`).
///
/// Returns (year, doy, hour, r_au, lat_deg, lon_deg).
fn parse_juno_amda_orb(content: &str) -> Vec<(u16, u16, u8, f64, f64, f64)> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut out = Vec::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let r_au = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let lat = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let lon = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        out.push((year, doy, hour, r_au, lat, lon));
    }
    out
}

/// Merge three Juno AMDA lanes into `SpdfMergedRecord` rows.
///
/// Only hours with data in ALL three lanes are emitted.
pub fn merge_juno_amda(
    plasma: &[(u16, u16, u8, f64, f64, f64)],
    mag: &[(u16, u16, u8, f64, f64, f64, f64)],
    orb: &[(u16, u16, u8, f64, f64, f64)],
) -> Vec<SpdfMergedRecord> {
    use std::collections::BTreeMap;
    let plasma_map: BTreeMap<_, _> = plasma.iter().map(|r| ((r.0, r.1, r.2), r)).collect();
    let mag_map: BTreeMap<_, _> = mag.iter().map(|r| ((r.0, r.1, r.2), r)).collect();
    let orb_map: BTreeMap<_, _> = orb.iter().map(|r| ((r.0, r.1, r.2), r)).collect();

    let mut rows = Vec::new();
    for (&key, p) in &plasma_map {
        let (Some(m), Some(o)) = (mag_map.get(&key), orb_map.get(&key)) else {
            continue;
        };
        rows.push(SpdfMergedRecord {
            year: key.0,
            doy: key.1,
            hour: key.2,
            distance_au: o.3,
            lat_deg: o.4,
            lon_deg: o.5,
            br: m.3,
            bt: m.4,
            bn: m.5,
            b_magnitude: m.6,
            proton_density: p.3,
            bulk_speed: p.4,
            proton_temperature: p.5,
        });
    }
    rows.sort_by_key(|r| (r.year, r.doy, r.hour));
    rows
}

/// Juno AMDA provider -- fetches three AMDA lanes and merges them.
///
/// Falls back to this when the CDAWeb-only `JunoCruiseProvider` is blocked.
/// Note: `JunoCruiseProvider` only provides orbital data; this provider
/// delivers the full plasma+MAG picture.
pub struct JunoAmdaProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for JunoAmdaProvider {
    fn default() -> Self {
        Self {
            year_start: 2011,
            year_end: 2016,
        }
    }
}

impl DatasetProvider for JunoAmdaProvider {
    fn name(&self) -> &str {
        "Juno Cruise AMDA (plasma+MAG+orbit)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("juno").join("amda");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let t_min = format!("{year}-01-01T00:00:00Z");
            let t_max = format!("{}-01-01T00:00:00Z", year + 1);
            let out_path = dir.join(format!("juno_amda_merged_{year}.csv"));
            if config.skip_existing && out_path.exists() {
                continue;
            }

            let plasma_csv = match download_amda_hapi_csv(JUNO_AMDA_PLASMA, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Juno plasma {year}: {e}");
                    continue;
                }
            };
            let mag_csv = match download_amda_hapi_csv(JUNO_AMDA_MAG, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Juno MAG {year}: {e}");
                    continue;
                }
            };
            let orb_csv = match download_amda_hapi_csv(JUNO_AMDA_ORB, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Juno orbit {year}: {e}");
                    continue;
                }
            };

            let plasma = parse_juno_amda_plasma(&plasma_csv);
            let mag = parse_juno_amda_mag(&mag_csv);
            let orb = parse_juno_amda_orb(&orb_csv);
            let merged = merge_juno_amda(&plasma, &mag, &orb);

            let mut csv_buf = String::from(
                "year,doy,hour,distance_au,lat_deg,lon_deg,br,bt,bn,b_mag,density,speed,temperature\n",
            );
            for r in &merged {
                csv_buf.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    r.year,
                    r.doy,
                    r.hour,
                    r.distance_au,
                    r.lat_deg,
                    r.lon_deg,
                    r.br,
                    r.bt,
                    r.bn,
                    r.b_magnitude,
                    r.proton_density,
                    r.bulk_speed,
                    r.proton_temperature,
                ));
            }
            std::fs::write(&out_path, csv_buf)?;
            log::info!(
                "AMDA Juno {year}: merged {} hourly records -> {}",
                merged.len(),
                out_path.display()
            );
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("juno").join("amda");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("juno_amda_merged_") && name.ends_with(".csv")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_juno_cruise_layout_validity() {
        assert_eq!(JUNO_CRUISE_LAYOUT.min_columns, 13);
        const { assert!(JUNO_CRUISE_LAYOUT.b_is_se) };
    }

    #[test]
    fn test_parse_juno_cruise_inner() {
        // Juno at ~2 AU during cruise
        // B ceiling = 200/4 = 50, density ceiling = 500/4 = 125
        let data = "2013 200 12 2.0 1.0 90.0 2.0 1.5 -0.5 0.3 2.0 400.0 50000.0\n";
        let records = parse_juno_cruise(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 2.0).abs() < 0.01);
        assert!((r.b_magnitude - 2.0).abs() < 0.01);
        assert!((r.proton_density - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_juno_cruise_jupiter_approach() {
        // Juno approaching Jupiter at ~5 AU
        // B ceiling = 200/25 = 8.0, density ceiling = 500/25 = 20.0
        let data = "2016 150 6 5.0 0.5 270.0 0.5 0.3 -0.1 0.05 0.3 400.0 30000.0\n";
        let records = parse_juno_cruise(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 5.0).abs() < 0.01);
        assert!(!r.b_magnitude.is_nan());
    }

    #[test]
    fn test_juno_to_omni() {
        let data = "2013 200 12 2.0 1.0 90.0 2.0 1.5 -0.5 0.3 2.0 400.0 50000.0\n";
        let spdf = parse_juno_cruise(data);
        let omni = juno_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        assert!((omni[0].r_au - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_juno_amda_plasma_format() {
        let csv = "Time,n_p,v_p,T_p\n\
                   2013-07-20T12:00:00.000Z,2.5,420.0,80000.0\n";
        let rows = parse_juno_amda_plasma(csv);
        assert_eq!(rows.len(), 1);
        let (year, _doy, _hour, density, speed, temp) = rows[0];
        assert_eq!(year, 2013);
        assert!((density - 2.5).abs() < 0.01);
        assert!((speed - 420.0).abs() < 0.1);
        assert!((temp - 80000.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_juno_amda_mag_format() {
        let csv = "Time,Br,Bt,Bn,B\n\
                   2013-07-20T12:00:00.000Z,0.8,-0.2,0.05,0.84\n";
        let rows = parse_juno_amda_mag(csv);
        assert_eq!(rows.len(), 1);
        let (_, _, _, br, bt, bn, b_mag) = rows[0];
        assert!((br - 0.8).abs() < 0.01);
        assert!((bt - (-0.2)).abs() < 0.01);
        assert!((bn - 0.05).abs() < 0.001);
        assert!((b_mag - 0.84).abs() < 0.01);
    }

    #[test]
    fn test_merge_juno_amda_intersection() {
        // All three lanes have the same hour.
        let plasma = vec![(2013u16, 201u16, 12u8, 2.5f64, 420.0f64, 80000.0f64)];
        let mag = vec![(2013u16, 201u16, 12u8, 0.8f64, -0.2f64, 0.05f64, 0.84f64)];
        let orb = vec![(2013u16, 201u16, 12u8, 2.8f64, 3.0f64, 100.0f64)];
        let merged = merge_juno_amda(&plasma, &mag, &orb);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].proton_density - 2.5).abs() < 0.01);
        assert!((merged[0].distance_au - 2.8).abs() < 0.01);
        assert!((merged[0].br - 0.8).abs() < 0.01);
    }
}
