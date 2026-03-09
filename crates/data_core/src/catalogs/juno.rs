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

use crate::catalogs::omni::OmniRecord;
use crate::catalogs::spdf_merged::{SpdfColumnLayout, SpdfMergedRecord, parse_spdf_merged, spdf_to_omni};
use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string};
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

/// Parse Juno cruise merged hourly data from a string.
pub fn parse_juno_cruise(content: &str) -> Vec<SpdfMergedRecord> {
    parse_spdf_merged(content, &JUNO_CRUISE_LAYOUT)
}

/// Parse Juno cruise merged hourly data from a file.
pub fn parse_juno_cruise_file(
    path: &std::path::Path,
) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    Ok(parse_juno_cruise(&content))
}

/// Convert Juno cruise records to OmniRecord format.
pub fn juno_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    spdf_to_omni(records, true) // SE coordinates
}

const JUNO_CRUISE_BASE: &str =
    "https://spdf.gsfc.nasa.gov/pub/data/juno/merged/";

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
            let fname = format!("juno_{}_merged_hourly.asc", year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}", JUNO_CRUISE_BASE, fname);
            match download_to_string(&url) {
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
}
