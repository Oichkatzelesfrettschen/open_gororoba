//! New Horizons SWAP (Solar Wind Around Pluto) data parser.
//!
//! New Horizons is an active mission at 55+ AU (2024), providing
//! modern high-sensitivity plasma measurements in the outer heliosphere.
//!
//! Key limitation: NO magnetometer on board. All B-field values are NaN.
//! This makes NH SWAP complementary to Voyager (which has MAG but
//! degraded plasma sensitivity at large distances).
//!
//! SWAP instrument: Solar Wind Around Pluto
//!   McComas et al. (2008), Space Sci. Rev. 140, 261
//!   Measures solar wind proton density, speed, and temperature.
//!
//! Data format: SPDF merged hourly ASCII (PDS-derived).
//! NH SWAP columns differ from Voyager: no B-field columns.
//!   0: Year, 1: DOY, 2: Hour
//!   3: Heliocentric distance (AU)
//!   4: Heliographic latitude (deg)
//!   5: Heliographic longitude (deg)
//!   6: Proton density (cm^-3)
//!   7: Proton speed (km/s)
//!   8: Proton temperature (K)
//!
//! Fill values: 9999.9 (density, speed), 9999999.0 (temperature),
//!              999.999 (distance)
//!
//! Source: <https://spdf.gsfc.nasa.gov/pub/data/new-horizons/swap/>

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_fleet::SpdfMission,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord},
    },
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string},
};
use std::path::PathBuf;

/// SPDF column layout for New Horizons SWAP hourly data.
///
/// No magnetometer: col_b_mag, col_br, col_bt, col_bn are all None.
pub const NH_SWAP_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 9,
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
    col_density: Some(6),
    col_speed: Some(7),
    col_temperature: Some(8),
    fill_b: 9999.9,
    fill_density: 9999.9,
    fill_speed: 9999.9,
    fill_temperature: 9999999.0,
    fill_distance: 999.999,
    b_is_se: true, // irrelevant since no B columns
};

/// NH SWAP fill values for reference.
pub const NH_FILL_DENSITY: f64 = 9999.9;
pub const NH_FILL_SPEED: f64 = 9999.9;
pub const NH_FILL_TEMP: f64 = 9999999.0;
pub const NH_FILL_DISTANCE: f64 = 999.999;

/// `SpdfMission` config for New Horizons SWAP merged hourly data.
pub static NH_SWAP_MISSION: SpdfMission = SpdfMission {
    layout: &NH_SWAP_LAYOUT,
    b_is_se: true, // irrelevant since no B columns
    year_fixup: None,
};

/// Parse New Horizons SWAP hourly data from a string.
pub fn parse_nh_swap(content: &str) -> Vec<SpdfMergedRecord> {
    NH_SWAP_MISSION.parse_merged(content)
}

/// Parse New Horizons SWAP hourly data from a file.
pub fn parse_nh_swap_file(path: &std::path::Path) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    NH_SWAP_MISSION.parse_file(path)
}

/// Convert NH SWAP records to OmniRecord format.
///
/// B-field will be NaN (no magnetometer). Position (r_au, lat, lon)
/// populated from SWAP data.
pub fn nh_swap_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    NH_SWAP_MISSION.to_omni(records)
}

const NH_SWAP_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/new-horizons/swap/merged/";

/// NASA SPDF New Horizons SWAP dataset provider.
pub struct NhSwapProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for NhSwapProvider {
    fn default() -> Self {
        Self {
            year_start: 2015,
            year_end: 2020,
        }
    }
}

impl DatasetProvider for NhSwapProvider {
    fn name(&self) -> &str {
        "New Horizons SWAP Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("new_horizons");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("nh_swap_{}_hourly.asc", year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}", NH_SWAP_BASE, fname);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download NH SWAP {}: {}", year, e);
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("new_horizons").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nh_swap_layout_validity() {
        assert_eq!(NH_SWAP_LAYOUT.min_columns, 9);
        assert!(NH_SWAP_LAYOUT.col_b_mag.is_none(), "no magnetometer");
        assert!(NH_SWAP_LAYOUT.col_br.is_none());
        assert_eq!(NH_SWAP_LAYOUT.col_density, Some(6));
    }

    #[test]
    fn test_parse_nh_swap_pluto_flyby() {
        // NH at Pluto flyby ~33 AU (2015)
        // At 33 AU: density ceiling = 500/1089 = 0.459
        let data = "2015 195 12 33.0 1.5 180.0 0.01 400.0 20000.0\n";
        let records = parse_nh_swap(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 33.0).abs() < 0.1);
        assert!(r.b_magnitude.is_nan(), "no magnetometer");
        assert!(r.br.is_nan(), "no magnetometer");
        assert!((r.proton_density - 0.01).abs() < 0.001);
        assert!((r.bulk_speed - 400.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_nh_swap_outer() {
        // NH at ~55 AU (2024)
        // At 55 AU: density ceiling = 500/3025 = 0.165
        let data = "2024 100 0 55.0 2.0 160.0 0.003 380.0 10000.0\n";
        let records = parse_nh_swap(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 55.0).abs() < 0.1);
        assert!((r.proton_density - 0.003).abs() < 0.001);
    }

    #[test]
    fn test_nh_swap_to_omni_no_bfield() {
        let data = "2015 195 12 33.0 1.5 180.0 0.01 400.0 20000.0\n";
        let spdf = parse_nh_swap(data);
        let omni = nh_swap_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        assert!(omni[0].b_magnitude.is_nan(), "no magnetometer");
        assert!(omni[0].bx_gse.is_nan(), "no magnetometer");
        assert!((omni[0].r_au - 33.0).abs() < 0.1);
    }

    #[test]
    fn test_nh_swap_fill_values() {
        let data = "2020 100 12 999.999 9999.9 9999.9 9999.9 9999.9 9999999.0\n";
        let records = parse_nh_swap(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan());
        assert!(r.proton_density.is_nan());
        assert!(r.bulk_speed.is_nan());
        assert!(r.proton_temperature.is_nan());
    }
}
