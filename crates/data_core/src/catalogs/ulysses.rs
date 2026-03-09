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
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord, parse_spdf_merged, spdf_to_omni},
    },
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string},
};
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

/// Parse Ulysses merged hourly data from a string.
pub fn parse_ulysses_merged(content: &str) -> Vec<SpdfMergedRecord> {
    parse_spdf_merged(content, &ULYSSES_LAYOUT)
}

/// Parse Ulysses merged hourly data from a file.
pub fn parse_ulysses_file(path: &std::path::Path) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    Ok(parse_ulysses_merged(&content))
}

/// Convert Ulysses records to OmniRecord format.
///
/// Sets r_au, lat_deg, lon_deg from parsed position columns.
/// B-field is in RTN coordinates; converted to GSE with separation_angle=0
/// (radially outward spacecraft approximation).
pub fn ulysses_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    spdf_to_omni(records, false) // RTN coordinates (b_is_se=false)
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

/// Base URL for Ulysses merged hourly data at SPDF.
const ULYSSES_MERGED_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/ulysses/merged/";

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
            year_start: 1994,
            year_end: 1995,
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
            let fname = format!("uly_{}_merged_hourly.asc", year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}", ULYSSES_MERGED_BASE, fname);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download Ulysses {}: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("ulysses").exists()
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
        // RTN: Br -> Bx, Bt -> -By, Bn -> Bz
        let data = "1994 250 12 2.0 -75.0 180.0 1.5 -0.3 0.1 1.55 3.0 750.0 200000.0\n";
        let spdf = parse_ulysses_merged(data);
        let omni = ulysses_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        let o = &omni[0];
        // RTN->GSE: br -> bx, -bt -> by, bn -> bz
        assert!((o.bx_gse - 1.5).abs() < 1e-6, "Br -> Bx");
        assert!((o.by_gse - 0.3).abs() < 1e-6, "-Bt -> By (sign flip)");
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
}
