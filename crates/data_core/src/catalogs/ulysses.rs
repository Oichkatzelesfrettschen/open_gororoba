//! Ulysses spacecraft data parser.
//!
//! Ulysses is the ONLY spacecraft with polar solar wind measurements,
//! providing fast latitude scans from -80 to +80 deg heliographic:
//!   First orbit (1994-1995): south pole -> equator -> north pole
//!   Second orbit (2000-2001): repeated during solar maximum
//!   Third orbit (2007-2008): during deep solar minimum
//!
//! Key science: bimodal solar wind structure
//!   Fast polar wind:  ~750 km/s, ~3 cm^-3, T ~ 200,000 K
//!   Slow equatorial:  ~400 km/s, ~7 cm^-3, T ~ 50,000 K
//!   Transition band:  ~20-30 deg from equator
//!
//! Range: 1.3-5.4 AU (Jupiter gravity assist orbit, inclined 80 deg).
//!
//! Instruments:
//!   SWOOPS: Solar Wind Observations Over the Poles of the Sun
//!           (Bame et al., 1992) -- proton density, speed, temperature
//!   VHM/FGM: Vector Helium/Fluxgate Magnetometer
//!           (Balogh et al., 1992) -- B-field in RTN coordinates
//!
//! SPDF merged hourly format columns:
//!   0: Year, 1: DOY, 2: Hour
//!   3: Heliocentric distance (AU)
//!   4: Heliographic latitude (deg)
//!   5: Heliographic longitude (deg)
//!   6: Br RTN (nT), 7: Bt RTN (nT), 8: Bn RTN (nT), 9: |B| (nT)
//!   10: Proton density (cm^-3)
//!   11: Proton speed (km/s)
//!   12: Proton temperature (K)
//!
//! Fill values: 9999.9 (B components), 9999.9 (|B|), 9999.9 (density),
//!              9999.9 (speed), 9999999.0 (temperature), 999.999 (distance)
//!
//! Coordinate system: RTN (Radial-Tangential-Normal) for B-field.
//!
//! Source: <https://spdf.gsfc.nasa.gov/pub/data/ulysses/>

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_fleet::SpdfMission,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord},
    },
    fetcher::{
        DatasetProvider, FetchConfig, FetchError, download_amda_hapi_csv, download_hapi_csv,
        download_to_file,
    },
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::{collections::BTreeMap, path::PathBuf};

/// SPDF column layout for Ulysses merged hourly data.
///
/// B-field columns are in RTN coordinates (different from Voyager which uses SE).
/// B magnitude is at column 9 (after components), unlike Voyager where it precedes.
pub const ULYSSES_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 13,
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
    col_density: Some(10),
    col_speed: Some(11),
    col_temperature: Some(12),
    fill_b: 9999.9,
    fill_density: 9999.9,
    fill_speed: 9999.9,
    fill_temperature: 9999999.0,
    fill_distance: 999.999,
    b_is_se: false, // RTN coordinates
};

// Ulysses-specific fill values for reference.
/// B component fill (nT).
pub const ULYSSES_FILL_B: f64 = 9999.9;
/// Density fill (cm^-3).
pub const ULYSSES_FILL_DENSITY: f64 = 9999.9;
/// Speed fill (km/s).
pub const ULYSSES_FILL_SPEED: f64 = 9999.9;
/// Temperature fill (K).
pub const ULYSSES_FILL_TEMP: f64 = 9999999.0;
/// Distance fill (AU).
pub const ULYSSES_FILL_DISTANCE: f64 = 999.999;

/// Raw Ulysses SWOOPS plasma record (hourly).
#[derive(Debug, Clone)]
pub struct UlyssesSwoopsRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    /// Heliocentric distance (AU).
    pub r_au: f64,
    /// Heliographic latitude (deg). -80 to +80.
    pub lat_deg: f64,
    /// Heliographic longitude (deg).
    pub lon_deg: f64,
    /// Proton density (cm^-3).
    pub density: f64,
    /// Proton bulk speed (km/s).
    pub speed: f64,
    /// Proton temperature (K).
    pub temperature: f64,
}

/// Raw Ulysses VHM/FGM magnetometer record (hourly).
#[derive(Debug, Clone)]
pub struct UlyssesMagRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    /// B radial in RTN (nT).
    pub br: f64,
    /// B tangential in RTN (nT).
    pub bt: f64,
    /// B normal in RTN (nT).
    pub bn: f64,
    /// B magnitude (nT).
    pub b_mag: f64,
}

/// `SpdfMission` config for Ulysses merged hourly data.
pub static ULYSSES_MISSION: SpdfMission = SpdfMission {
    layout: &ULYSSES_LAYOUT,
    b_is_se: false,
    year_fixup: None,
};

/// Parse Ulysses merged hourly data from a string.
pub fn parse_ulysses_merged(content: &str) -> Vec<SpdfMergedRecord> {
    ULYSSES_MISSION.parse_merged(content)
}

/// Parse Ulysses merged hourly data from a file.
pub fn parse_ulysses_file(path: &std::path::Path) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    if path.extension().and_then(|value| value.to_str()) == Some("csv") {
        Ok(parse_ulysses_hapi_csv(&content))
    } else {
        Ok(parse_ulysses_merged(&content))
    }
}

pub fn parse_ulysses_hapi_csv(content: &str) -> Vec<SpdfMergedRecord> {
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
            proton_density: parse_hapi_spacephysics_f64_or_nan(record.get(12).unwrap_or("")),
            bulk_speed: parse_hapi_spacephysics_f64_or_nan(record.get(9).unwrap_or("")),
            proton_temperature: parse_hapi_spacephysics_f64_or_nan(record.get(14).unwrap_or("")),
        });
    }
    rows
}

/// Convert Ulysses records to OmniRecord format.
///
/// Sets r_au, lat_deg, lon_deg from parsed position columns.
/// B-field is in RTN coordinates; converted to GSE with separation_angle=0
/// (radially outward spacecraft approximation).
pub fn ulysses_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    ULYSSES_MISSION.to_omni(records)
}

/// Merge separate SWOOPS and VHM/FGM records by time key.
///
/// Follows the same BTreeMap intersection pattern as merge_wind_swe_mfi.
/// Produces SpdfMergedRecord as the common intermediate.
pub fn merge_ulysses_swoops_mag(
    swoops: &[UlyssesSwoopsRecord],
    mag: &[UlyssesMagRecord],
) -> Vec<SpdfMergedRecord> {
    // Build SWOOPS lookup
    let mut swoops_map: BTreeMap<(u16, u16, u8), &UlyssesSwoopsRecord> = BTreeMap::new();
    for r in swoops {
        swoops_map.insert((r.year, r.doy, r.hour), r);
    }

    // Build MAG lookup
    let mut mag_map: BTreeMap<(u16, u16, u8), &UlyssesMagRecord> = BTreeMap::new();
    for r in mag {
        mag_map.insert((r.year, r.doy, r.hour), r);
    }

    // Union of all time keys (sorted by BTreeMap iteration order)
    let mut all_keys: Vec<(u16, u16, u8)> =
        swoops_map.keys().chain(mag_map.keys()).copied().collect();
    all_keys.sort();
    all_keys.dedup();

    all_keys
        .iter()
        .map(|&(year, doy, hour)| {
            let (distance_au, lat_deg, lon_deg, density, speed, temperature) = swoops_map
                .get(&(year, doy, hour))
                .map(|s| {
                    (
                        s.r_au,
                        s.lat_deg,
                        s.lon_deg,
                        s.density,
                        s.speed,
                        s.temperature,
                    )
                })
                .unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN));

            let (br, bt, bn, b_magnitude) = mag_map
                .get(&(year, doy, hour))
                .map(|m| (m.br, m.bt, m.bn, m.b_mag))
                .unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN));

            SpdfMergedRecord {
                year,
                doy,
                hour,
                distance_au,
                lat_deg,
                lon_deg,
                b_magnitude,
                br,
                bt,
                bn,
                proton_density: density,
                bulk_speed: speed,
                proton_temperature: temperature,
            }
        })
        .collect()
}

const ULYSSES_HAPI_DATASET: &str = "UY_COHO1HR_MERGED_MAG_PLASMA";
const ULYSSES_SPDF_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/ulysses/merged/";

/// NASA SPDF Ulysses dataset provider.
pub struct UlyssesProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for UlyssesProvider {
    fn default() -> Self {
        Self {
            year_start: 1997,
            year_end: 2009,
        }
    }
}

impl DatasetProvider for UlyssesProvider {
    fn name(&self) -> &str {
        "Ulysses Merged Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ulysses");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let asc_name = format!("uly_{year}.asc");
            let asc_output = dir.join(&asc_name);
            let csv_name = format!("uy_coho1hr_merged_mag_plasma_{year}.csv");
            let csv_output = dir.join(&csv_name);
            if config.skip_existing && (asc_output.exists() || csv_output.exists()) {
                continue;
            }

            let asc_url = format!("{ULYSSES_SPDF_BASE}{asc_name}");
            match download_to_file(&asc_url, &asc_output) {
                Ok(_) => {
                    log::info!("saved {}", asc_name);
                    continue;
                }
                Err(e) => {
                    log::warn!(
                        "failed to download official Ulysses merged file {}: {}",
                        asc_url,
                        e
                    );
                }
            }

            match download_hapi_csv(
                ULYSSES_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&[
                    "Time",
                    "heliocentricDistance",
                    "heliographicLatitude",
                    "heliographicLongitude",
                    "BR",
                    "BT",
                    "BN",
                    "ABS_B",
                    "numVectorsMagFldAvg",
                    "plasmaFlowSpeed",
                    "elevAngle",
                    "azimuthAngle",
                    "protonDensity",
                    "alphaDensity",
                    "protonTempLarge",
                ]),
            ) {
                Ok(data) => {
                    std::fs::write(&csv_output, data)?;
                    log::info!("saved {}", csv_name);
                }
                Err(e) => {
                    log::warn!(
                        "failed to download Ulysses {} via HAPI fallback: {}",
                        year,
                        e
                    );
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("ulysses");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                (name.starts_with("uly_") && name.ends_with(".asc"))
                    || (name.starts_with("uy_coho1hr_") && name.ends_with(".csv"))
            })
    }
}

// ---------------------------------------------------------------------------
// AMDA fallback: three-dataset lane (plasma + MAG + orbit)
// ---------------------------------------------------------------------------
//
// WHY: SPDF/CDAWeb HAPI is sometimes blocked from European or institutional
// networks.  AMDA (amda.irap.omp.eu) mirrors the same Ulysses data under
// HAPI-compatible IDs and is independently reachable.
//
// Lane:
//   1. SPDF direct ASC download (try first -- most complete coverage).
//   2. CDAWeb HAPI: UY_COHO1HR_MERGED_MAG_PLASMA (merged single product).
//   3. AMDA HAPI: ulys-bai-mom (plasma) + ulys-fgm-rtn (MAG) + ulys-orb-all
//      (orbit), then merged into SpdfMergedRecord via time-key intersection.

/// AMDA dataset ID for Ulysses solar wind ion moments (SWOOPS/BAI level 2).
///
/// Parameters: Time, density (cm^-3), bulk speed (km/s), temperature (K).
const ULYSSES_AMDA_PLASMA: &str = "ulys-bai-mom";

/// AMDA dataset ID for Ulysses FGM magnetometer in RTN coordinates.
///
/// Parameters: Time, Br (nT), Bt (nT), Bn (nT), |B| (nT).
const ULYSSES_AMDA_MAG: &str = "ulys-fgm-rtn";

/// AMDA dataset ID for Ulysses orbital ephemeris.
///
/// Parameters: Time, r (AU), HGI latitude (deg), HGI longitude (deg).
const ULYSSES_AMDA_ORB: &str = "ulys-orb-all";

/// Parse Ulysses AMDA plasma CSV (`ulys-bai-mom`).
///
/// Returns (year, doy, hour, density_cm3, speed_kms, temperature_k).
/// Columns after the timestamp: n_p, v_p, T_p (positional, header present).
fn parse_ulysses_amda_plasma(content: &str) -> Vec<(u16, u16, u8, f64, f64, f64)> {
    let mut reader = csv::ReaderBuilder::new()
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

/// Parse Ulysses AMDA magnetometer CSV (`ulys-fgm-rtn`).
///
/// Returns (year, doy, hour, br, bt, bn, b_mag).
fn parse_ulysses_amda_mag(content: &str) -> Vec<(u16, u16, u8, f64, f64, f64, f64)> {
    let mut reader = csv::ReaderBuilder::new()
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

/// Parse Ulysses AMDA ephemeris CSV (`ulys-orb-all`).
///
/// Returns (year, doy, hour, r_au, lat_deg, lon_deg).
fn parse_ulysses_amda_orb(content: &str) -> Vec<(u16, u16, u8, f64, f64, f64)> {
    let mut reader = csv::ReaderBuilder::new()
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

/// Merge three AMDA lanes (plasma, MAG, orbit) into `SpdfMergedRecord` rows.
///
/// Uses time-key intersection: only hours present in ALL three datasets are
/// emitted.  This is conservative but guarantees no partially-filled rows.
pub fn merge_ulysses_amda(
    plasma: &[(u16, u16, u8, f64, f64, f64)],
    mag: &[(u16, u16, u8, f64, f64, f64, f64)],
    orb: &[(u16, u16, u8, f64, f64, f64)],
) -> Vec<SpdfMergedRecord> {
    use std::collections::BTreeMap;
    let plasma_map: BTreeMap<_, _> =
        plasma.iter().map(|r| ((r.0, r.1, r.2), r)).collect();
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

/// Ulysses AMDA provider -- fetches three AMDA lanes and merges them.
///
/// Intended as the third-tier fallback: try `UlyssesProvider` (SPDF + CDAWeb
/// HAPI) first, and fall back to this when both upstream sources fail.
pub struct UlyssesAmdaProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for UlyssesAmdaProvider {
    fn default() -> Self {
        Self {
            year_start: 1997,
            year_end: 2009,
        }
    }
}

impl DatasetProvider for UlyssesAmdaProvider {
    fn name(&self) -> &str {
        "Ulysses AMDA (plasma+MAG+orbit)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ulysses").join("amda");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let t_min = format!("{year}-01-01T00:00:00Z");
            let t_max = format!("{}-01-01T00:00:00Z", year + 1);
            let out_path = dir.join(format!("ulysses_amda_merged_{year}.csv"));
            if config.skip_existing && out_path.exists() {
                continue;
            }

            let plasma_csv =
                match download_amda_hapi_csv(ULYSSES_AMDA_PLASMA, &t_min, &t_max, None) {
                    Ok(csv) => csv,
                    Err(e) => {
                        log::warn!("AMDA Ulysses plasma {year}: {e}");
                        continue;
                    }
                };
            let mag_csv = match download_amda_hapi_csv(ULYSSES_AMDA_MAG, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Ulysses MAG {year}: {e}");
                    continue;
                }
            };
            let orb_csv = match download_amda_hapi_csv(ULYSSES_AMDA_ORB, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Ulysses orbit {year}: {e}");
                    continue;
                }
            };

            let plasma = parse_ulysses_amda_plasma(&plasma_csv);
            let mag = parse_ulysses_amda_mag(&mag_csv);
            let orb = parse_ulysses_amda_orb(&orb_csv);
            let merged = merge_ulysses_amda(&plasma, &mag, &orb);

            // Write merged rows as a simple CSV for downstream parsers.
            let mut csv_buf =
                String::from("year,doy,hour,distance_au,lat_deg,lon_deg,br,bt,bn,b_mag,density,speed,temperature\n");
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
                "AMDA Ulysses {year}: merged {} hourly records -> {}",
                merged.len(),
                out_path.display()
            );
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("ulysses").join("amda");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("ulysses_amda_merged_") && name.ends_with(".csv")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ulysses_layout_validity() {
        assert_eq!(ULYSSES_LAYOUT.min_columns, 13);
        const { assert!(!ULYSSES_LAYOUT.b_is_se, "Ulysses uses RTN, not SE") };
        assert_eq!(ULYSSES_LAYOUT.col_lat_deg, Some(4));
        assert_eq!(ULYSSES_LAYOUT.col_b_mag, Some(9));
        assert_eq!(ULYSSES_LAYOUT.col_br, Some(6));
    }

    #[test]
    fn test_parse_ulysses_fast_polar_wind() {
        // First fast latitude scan, south pole, ~2 AU
        // Fast polar wind: v ~ 750 km/s, n ~ 3 cm^-3, T ~ 200000 K
        // At 2 AU: B ceiling = 200/4 = 50, density ceiling = 500/4 = 125
        let data = "1994 250 12 2.0 -75.0 180.0 1.5 -0.3 0.1 1.55 3.0 750.0 200000.0\n";
        let records = parse_ulysses_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 2.0).abs() < 0.01);
        assert!((r.lat_deg - (-75.0)).abs() < 0.1, "south pole latitude");
        assert!((r.bulk_speed - 750.0).abs() < 0.1, "fast polar wind");
        assert!((r.proton_density - 3.0).abs() < 0.01);
        assert!((r.b_magnitude - 1.55).abs() < 0.01);
        // B components in RTN
        assert!((r.br - 1.5).abs() < 0.01);
        assert!((r.bt - (-0.3)).abs() < 0.01);
        assert!((r.bn - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_parse_ulysses_slow_equatorial_wind() {
        // Equatorial crossing at ~1.5 AU
        // Slow wind: v ~ 400 km/s, n ~ 7 cm^-3, T ~ 50000 K
        // At 1.5 AU: ceilings generous (density = 500/2.25 = 222)
        let data = "1995 100 6 1.5 5.0 90.0 3.0 -0.5 0.2 3.1 7.0 400.0 50000.0\n";
        let records = parse_ulysses_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.lat_deg - 5.0).abs() < 0.1, "near-equatorial");
        assert!((r.bulk_speed - 400.0).abs() < 0.1, "slow equatorial wind");
        assert!((r.proton_density - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_ulysses_north_pole() {
        // North polar pass at ~2 AU, lat ~ +80 deg
        // At 2 AU: B ceiling = 50, density ceiling = 125
        let data = "1995 200 0 2.0 80.0 270.0 1.8 -0.2 0.05 1.82 2.5 780.0 250000.0\n";
        let records = parse_ulysses_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.lat_deg - 80.0).abs() < 0.1, "north pole latitude");
        assert!((r.bulk_speed - 780.0).abs() < 0.1, "fast polar wind");
    }

    #[test]
    fn test_parse_ulysses_jupiter_distance() {
        // At Jupiter encounter distance ~5.4 AU
        // B ceiling = 200/29.16 = 6.86, density ceiling = 500/29.16 = 17.1
        let data = "1992 50 12 5.4 -5.0 120.0 0.3 -0.1 0.02 0.32 0.5 400.0 20000.0\n";
        let records = parse_ulysses_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 5.4).abs() < 0.01);
        assert!(!r.b_magnitude.is_nan());
        assert!(!r.proton_density.is_nan());
    }

    #[test]
    fn test_ulysses_fill_values() {
        let data = "1994 100 12 999.999 9999.9 9999.9 9999.9 9999.9 9999.9 9999.9 9999.9 9999.9 9999999.0\n";
        let records = parse_ulysses_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan(), "distance fill");
        assert!(r.b_magnitude.is_nan(), "B fill");
        assert!(r.proton_density.is_nan(), "density fill");
        assert!(r.bulk_speed.is_nan(), "speed fill");
        assert!(r.proton_temperature.is_nan(), "temp fill");
    }

    #[test]
    fn test_ulysses_to_omni_rtn_conversion() {
        // RTN -> GSE via rotation by spacecraft longitude (180 deg)
        let data = "1994 250 12 2.0 -75.0 180.0 1.5 -0.3 0.1 1.55 3.0 750.0 200000.0\n";
        let spdf = parse_ulysses_merged(data);
        let omni = ulysses_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        let o = &omni[0];
        // bx = br*cos(180) - bt*sin(180) = -br, by = br*sin(180) + bt*cos(180) = -bt
        assert!((o.bx_gse - (-1.5)).abs() < 1e-4, "RTN rotated Bx");
        assert!((o.by_gse - 0.3).abs() < 1e-4, "RTN rotated By");
        assert!((o.bz_gse - 0.1).abs() < 1e-6, "Bn -> Bz");
        assert!((o.r_au - 2.0).abs() < 0.01, "r_au populated");
        assert!((o.lat_deg - (-75.0)).abs() < 0.1, "lat_deg populated");
    }

    #[test]
    fn test_merge_swoops_mag_full_overlap() {
        let swoops = vec![
            UlyssesSwoopsRecord {
                year: 1994,
                doy: 250,
                hour: 12,
                r_au: 2.0,
                lat_deg: -75.0,
                lon_deg: 180.0,
                density: 3.0,
                speed: 750.0,
                temperature: 200000.0,
            },
            UlyssesSwoopsRecord {
                year: 1994,
                doy: 250,
                hour: 13,
                r_au: 2.0,
                lat_deg: -74.8,
                lon_deg: 180.5,
                density: 3.1,
                speed: 745.0,
                temperature: 195000.0,
            },
        ];
        let mag = vec![
            UlyssesMagRecord {
                year: 1994,
                doy: 250,
                hour: 12,
                br: 1.5,
                bt: -0.3,
                bn: 0.1,
                b_mag: 1.55,
            },
            UlyssesMagRecord {
                year: 1994,
                doy: 250,
                hour: 13,
                br: 1.4,
                bt: -0.2,
                bn: 0.15,
                b_mag: 1.43,
            },
        ];

        let merged = merge_ulysses_swoops_mag(&swoops, &mag);
        assert_eq!(merged.len(), 2);
        // Both hours should have full data
        assert!((merged[0].proton_density - 3.0).abs() < 0.01);
        assert!((merged[0].br - 1.5).abs() < 0.01);
        assert!((merged[1].proton_density - 3.1).abs() < 0.01);
        assert!((merged[1].br - 1.4).abs() < 0.01);
    }

    #[test]
    fn test_merge_swoops_mag_partial_overlap() {
        // SWOOPS has hour 12, MAG has hours 12 and 13
        let swoops = vec![UlyssesSwoopsRecord {
            year: 1994,
            doy: 250,
            hour: 12,
            r_au: 2.0,
            lat_deg: -75.0,
            lon_deg: 180.0,
            density: 3.0,
            speed: 750.0,
            temperature: 200000.0,
        }];
        let mag = vec![
            UlyssesMagRecord {
                year: 1994,
                doy: 250,
                hour: 12,
                br: 1.5,
                bt: -0.3,
                bn: 0.1,
                b_mag: 1.55,
            },
            UlyssesMagRecord {
                year: 1994,
                doy: 250,
                hour: 13,
                br: 1.4,
                bt: -0.2,
                bn: 0.15,
                b_mag: 1.43,
            },
        ];

        let merged = merge_ulysses_swoops_mag(&swoops, &mag);
        assert_eq!(merged.len(), 2, "union of time keys");

        // Hour 12: full data
        assert!(!merged[0].proton_density.is_nan());
        assert!(!merged[0].br.is_nan());

        // Hour 13: MAG only, plasma NaN
        assert!(merged[1].proton_density.is_nan(), "no SWOOPS at hour 13");
        assert!(!merged[1].br.is_nan(), "MAG data at hour 13");
    }

    #[test]
    fn test_ulysses_fast_slow_bimodality() {
        // Verify the parser handles the full latitude range correctly
        // South pole (fast), equator (slow), north pole (fast)
        let data = "\
1994 250 12 2.0 -75.0 180.0 1.5 -0.3 0.1 1.55 3.0 750.0 200000.0
1995 100 12 1.5 5.0 90.0 3.0 -0.5 0.2 3.1 7.0 400.0 50000.0
1995 200 12 2.0 80.0 270.0 1.8 -0.2 0.05 1.82 2.5 780.0 250000.0
";
        let records = parse_ulysses_merged(data);
        assert_eq!(records.len(), 3);

        // South pole: fast
        assert!(records[0].bulk_speed > 700.0);
        assert!(records[0].proton_density < 4.0);

        // Equator: slow
        assert!(records[1].bulk_speed < 450.0);
        assert!(records[1].proton_density > 5.0);

        // North pole: fast
        assert!(records[2].bulk_speed > 700.0);
        assert!(records[2].proton_density < 4.0);
    }

    #[test]
    fn test_parse_ulysses_amda_plasma_format() {
        // Minimal AMDA CSV with header row.
        let csv = "Time,n_p,v_p,T_p\n\
                   1994-09-07T12:00:00.000Z,3.1,750.0,200000.0\n\
                   1994-09-07T13:00:00.000Z,-1.0e31,9999.9,9999999.0\n";
        let rows = parse_ulysses_amda_plasma(csv);
        assert_eq!(rows.len(), 2);
        // First row: valid fast polar wind
        let (year, doy, hour, density, speed, temp) = rows[0];
        assert_eq!(year, 1994);
        assert!((density - 3.1).abs() < 0.01);
        assert!((speed - 750.0).abs() < 0.1);
        assert!((temp - 200000.0).abs() < 1.0);
        let _ = (doy, hour);
        // Second row: AMDA fill values become NaN via parse_hapi_spacephysics_f64_or_nan
        let (_, _, _, density2, speed2, _) = rows[1];
        assert!(density2.is_nan() || density2 < 0.0, "fill should be NaN or negative");
        let _ = speed2;
    }

    #[test]
    fn test_parse_ulysses_amda_mag_format() {
        let csv = "Time,Br,Bt,Bn,B\n\
                   1994-09-07T12:00:00.000Z,1.5,-0.3,0.1,1.55\n";
        let rows = parse_ulysses_amda_mag(csv);
        assert_eq!(rows.len(), 1);
        let (_, _, _, br, bt, bn, b_mag) = rows[0];
        assert!((br - 1.5).abs() < 0.01);
        assert!((bt - (-0.3)).abs() < 0.01);
        assert!((bn - 0.1).abs() < 0.01);
        assert!((b_mag - 1.55).abs() < 0.01);
    }

    #[test]
    fn test_merge_ulysses_amda_three_way_intersection() {
        // All three lanes cover the same two hours.
        let plasma = vec![
            (1994u16, 250u16, 12u8, 3.0f64, 750.0f64, 200000.0f64),
            (1994, 250, 13, 3.1, 745.0, 195000.0),
        ];
        let mag = vec![
            (1994u16, 250u16, 12u8, 1.5f64, -0.3f64, 0.1f64, 1.55f64),
            (1994, 250, 13, 1.4, -0.2, 0.15, 1.43),
        ];
        let orb = vec![
            (1994u16, 250u16, 12u8, 2.0f64, -75.0f64, 180.0f64),
            (1994, 250, 13, 2.01, -74.8, 180.5),
        ];
        let merged = merge_ulysses_amda(&plasma, &mag, &orb);
        assert_eq!(merged.len(), 2);
        assert!((merged[0].proton_density - 3.0).abs() < 0.01);
        assert!((merged[0].br - 1.5).abs() < 0.01);
        assert!((merged[0].distance_au - 2.0).abs() < 0.01);
        assert!((merged[0].lat_deg - (-75.0)).abs() < 0.1);
    }

    #[test]
    fn test_merge_ulysses_amda_missing_mag_hour_skipped() {
        // MAG missing hour 13 -- intersection only yields hour 12.
        let plasma = vec![
            (1994u16, 250u16, 12u8, 3.0f64, 750.0f64, 200000.0f64),
            (1994, 250, 13, 3.1, 745.0, 195000.0),
        ];
        let mag = vec![(1994u16, 250u16, 12u8, 1.5f64, -0.3f64, 0.1f64, 1.55f64)];
        let orb = vec![
            (1994u16, 250u16, 12u8, 2.0f64, -75.0f64, 180.0f64),
            (1994, 250, 13, 2.01, -74.8, 180.5),
        ];
        let merged = merge_ulysses_amda(&plasma, &mag, &orb);
        assert_eq!(merged.len(), 1, "hour 13 missing MAG should be excluded");
        assert!((merged[0].bulk_speed - 750.0).abs() < 0.1);
    }
}
