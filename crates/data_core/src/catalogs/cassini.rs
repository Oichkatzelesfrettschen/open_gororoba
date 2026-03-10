//! Cassini cruise and Saturn-orbit plasma / magnetic-field parser.
//!
//! Cassini provides plasma and magnetic field measurements during:
//!   Cruise phase (1997-2004): 1-9.5 AU, en route to Saturn
//!   Saturn orbit (2004-2017): ~9.5 AU, magnetospheric science
//!
//! For heliospheric DM studies, the cruise phase data is primary
//! (clean solar wind measurements). Saturn-orbit data is contaminated
//! by Saturn's magnetosphere and only useful during solar wind
//! excursions outside the bow shock.
//!
//! Instruments:
//!   CAPS: Cassini Plasma Spectrometer
//!         Young et al. (2004), Space Sci. Rev. 114, 1
//!   MAG: Dual Technique Magnetometer
//!         Dougherty et al. (2004), Space Sci. Rev. 114, 331
//!
//! Canonical SPDF merged hourly format.
//!   0: Year, 1: DOY, 2: Hour
//!   3: Heliocentric distance (AU)
//!   4: Heliographic latitude (deg)
//!   5: Heliographic longitude (deg)
//!   6: |B| (nT), 7: Br (nT), 8: Bt (nT), 9: Bn (nT)
//!   10: Proton density (cm^-3)
//!   11: Proton speed (km/s)
//!   12: Proton temperature (K)
//!
//! Coordinate system: RTN for B-field.
//!
//! Governed local note:
//!   The repo may also stage a hybrid AMDA-derived hourly lane in the same
//!   column layout. In that hybrid, trajectory comes from `cass-orb-cruise`,
//!   magnetic field from measured `cass-mag-rtn60`, and plasma from modeled
//!   `tao-cass-sw`. The parser intentionally accepts the shared SPDF-style
//!   layout, while provenance is carried separately by manifests and audits.
//!
//! Source: <https://spdf.gsfc.nasa.gov/pub/data/cassini/>

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord, parse_spdf_merged, spdf_to_omni},
    },
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string},
};
use std::path::PathBuf;

/// SPDF column layout for Cassini cruise merged hourly data.
pub const CASSINI_CRUISE_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
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
    b_is_se: false, // RTN coordinates
};

/// Parse Cassini cruise merged hourly data from a string.
pub fn parse_cassini_cruise(content: &str) -> Vec<SpdfMergedRecord> {
    parse_spdf_merged(content, &CASSINI_CRUISE_LAYOUT)
}

/// Parse Cassini cruise merged hourly data from a file.
pub fn parse_cassini_cruise_file(
    path: &std::path::Path,
) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    Ok(parse_cassini_cruise(&content))
}

/// Convert Cassini cruise records to OmniRecord format.
///
/// B-field is in RTN coordinates; converted to GSE with separation_angle=0.
pub fn cassini_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    spdf_to_omni(records, false) // RTN coordinates
}

const CASSINI_CRUISE_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/cassini/merged/";

/// NASA SPDF Cassini cruise dataset provider.
pub struct CassiniCruiseProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for CassiniCruiseProvider {
    fn default() -> Self {
        Self {
            year_start: 1997,
            year_end: 2004,
        }
    }
}

impl DatasetProvider for CassiniCruiseProvider {
    fn name(&self) -> &str {
        "Cassini Cruise Merged Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("cassini");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("cassini_{}_merged_hourly.asc", year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}", CASSINI_CRUISE_BASE, fname);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download Cassini {}: {}", year, e);
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("cassini").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cassini_cruise_layout_validity() {
        assert_eq!(CASSINI_CRUISE_LAYOUT.min_columns, 13);
        const { assert!(!CASSINI_CRUISE_LAYOUT.b_is_se, "Cassini uses RTN") };
    }

    #[test]
    fn test_parse_cassini_cruise_inner() {
        // Cassini at ~3 AU during cruise
        // B ceiling = 200/9 = 22.2, density ceiling = 500/9 = 55.6
        let data = "2000 200 12 3.0 1.0 120.0 1.5 1.0 -0.5 0.2 1.0 400.0 50000.0\n";
        let records = parse_cassini_cruise(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 3.0).abs() < 0.01);
        assert!((r.b_magnitude - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_cassini_saturn_approach() {
        // Cassini approaching Saturn at ~9.5 AU
        // B ceiling = 200/90.25 = 2.22, density ceiling = 500/90.25 = 5.54
        let data = "2004 150 6 9.5 0.5 200.0 0.5 0.3 -0.1 0.05 0.3 400.0 30000.0\n";
        let records = parse_cassini_cruise(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 9.5).abs() < 0.01);
    }

    #[test]
    fn test_cassini_to_omni_rtn_conversion() {
        // RTN: Br -> Bx, -Bt -> By, Bn -> Bz
        let data = "2000 200 12 3.0 1.0 120.0 1.5 1.0 -0.5 0.2 1.0 400.0 50000.0\n";
        let spdf = parse_cassini_cruise(data);
        let omni = cassini_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        assert!((omni[0].bx_gse - 1.0).abs() < 1e-6, "Br -> Bx");
        assert!((omni[0].by_gse - 0.5).abs() < 1e-6, "-Bt -> By");
        assert!((omni[0].bz_gse - 0.2).abs() < 1e-6, "Bn -> Bz");
        assert!((omni[0].r_au - 3.0).abs() < 0.01);
    }
}
