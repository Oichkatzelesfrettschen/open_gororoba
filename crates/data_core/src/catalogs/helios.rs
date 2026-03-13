//! Helios 1 & 2 merged hourly data parser.
//!
//! Helios spacecraft provide inner heliospheric coverage (0.29-0.98 AU):
//!   H1: launched 1974-12-10, perihelion 0.31 AU, data to 1985
//!   H2: launched 1976-01-15, perihelion 0.29 AU, data to 1980
//!
//! AMDA HAPI datasets (primary source when SPDF blocked):
//!   helios{1,2}-e1-all: proton corefit (density, speed, thermal_speed)
//!   helios{1,2}-e3-all: MAG 6-sec average (Bx, By, Bz, |B|)
//!   helios{1,2}-orb-all: orbit (r_au, lat, lon)
//!
//! The Python fetcher (bin/fetch_helios.py) translates AMDA HAPI CSV to
//! SPDF-style merged hourly ASCII with per-hour median aggregation.
//! E1 thermal_speed is converted to temperature via T = m_p * v_th^2 / (2*k_B).
//!
//! Fill values match the AMDA-derived output format.
//! B-field coordinate system: RTN from AMDA (ecliptic data available but
//! RTN is standard for inner heliosphere consistency with Voyager/Pioneer).
//!
//! Source: <https://amda.irap.omp.eu/service/hapi/info?id=helios1-e1-all>

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_fleet::SpdfMission,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord},
    },
    fetcher::{DatasetProvider, FetchConfig, FetchError},
};
use std::path::PathBuf;

/// Column layout for Helios 1 AMDA-derived merged hourly data.
pub const HELIOS1_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
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

/// Column layout for Helios 2 AMDA-derived merged hourly data.
/// Same format as Helios 1.
pub const HELIOS2_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
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

/// Helios B fill (nT).
pub const HELIOS_FILL_B: f64 = 9999.99;
/// Helios density fill (cm^-3).
pub const HELIOS_FILL_DENSITY: f64 = 999.9;
/// Helios speed fill (km/s).
pub const HELIOS_FILL_SPEED: f64 = 9999.9;
/// Helios temperature fill (K).
pub const HELIOS_FILL_TEMP: f64 = 999999.0;
/// Helios distance fill (AU).
pub const HELIOS_FILL_DISTANCE: f64 = 999.999;

/// Which Helios spacecraft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeliosSpacecraft {
    H1,
    H2,
}

/// `SpdfMission` config for Helios 1 merged hourly data.
pub static HELIOS1_MISSION: SpdfMission = SpdfMission {
    layout: &HELIOS1_LAYOUT,
    b_is_se: false,
    year_fixup: None,
};

/// `SpdfMission` config for Helios 2 merged hourly data.
pub static HELIOS2_MISSION: SpdfMission = SpdfMission {
    layout: &HELIOS2_LAYOUT,
    b_is_se: false,
    year_fixup: None,
};

/// Parse Helios merged hourly data from a string.
pub fn parse_helios_merged(content: &str, spacecraft: HeliosSpacecraft) -> Vec<SpdfMergedRecord> {
    let mission = match spacecraft {
        HeliosSpacecraft::H1 => &HELIOS1_MISSION,
        HeliosSpacecraft::H2 => &HELIOS2_MISSION,
    };
    mission.parse_merged(content)
}

/// Parse Helios merged hourly data from a file.
pub fn parse_helios_file(
    path: &std::path::Path,
    spacecraft: HeliosSpacecraft,
) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let mission = match spacecraft {
        HeliosSpacecraft::H1 => &HELIOS1_MISSION,
        HeliosSpacecraft::H2 => &HELIOS2_MISSION,
    };
    mission.parse_file(path)
}

/// Convert Helios records to OmniRecord format.
///
/// B-field is in RTN coordinates (sign flip on Bt for GSE conversion).
/// Both H1 and H2 use RTN coordinates, so no spacecraft dispatch is needed here.
pub fn helios_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    // Both HELIOS1_MISSION and HELIOS2_MISSION have b_is_se=false; same result.
    HELIOS1_MISSION.to_omni(records)
}

/// Base URL for Helios merged hourly data at SPDF (blocked from this host).
const HELIOS1_SPDF_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/merged/";
const HELIOS2_SPDF_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/helios/helios2/merged/";

/// NASA Helios dataset provider.
pub struct HeliosProvider {
    /// Which spacecraft (H1 or H2).
    pub spacecraft: HeliosSpacecraft,
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for HeliosProvider {
    fn default() -> Self {
        Self {
            spacecraft: HeliosSpacecraft::H1,
            year_start: 1976,
            year_end: 1980,
        }
    }
}

impl DatasetProvider for HeliosProvider {
    fn name(&self) -> &str {
        match self.spacecraft {
            HeliosSpacecraft::H1 => "Helios 1 Merged Hourly",
            HeliosSpacecraft::H2 => "Helios 2 Merged Hourly",
        }
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let subdir = match self.spacecraft {
            HeliosSpacecraft::H1 => "helios1",
            HeliosSpacecraft::H2 => "helios2",
        };
        let dir = config.output_dir.join("helios").join(subdir);
        std::fs::create_dir_all(&dir)?;

        let base = match self.spacecraft {
            HeliosSpacecraft::H1 => HELIOS1_SPDF_BASE,
            HeliosSpacecraft::H2 => HELIOS2_SPDF_BASE,
        };

        for year in self.year_start..=self.year_end {
            let fname = format!("{}_{}_amda_merged.asc", subdir, year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            // SPDF is blocked; log and continue. Data comes via Python AMDA fetcher.
            log::warn!(
                "Helios SPDF blocked ({}). Use bin/fetch_helios.py --source amda for year {}.",
                base,
                year,
            );
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let subdir = match self.spacecraft {
            HeliosSpacecraft::H1 => "helios1",
            HeliosSpacecraft::H2 => "helios2",
        };
        config.output_dir.join("helios").join(subdir).exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helios1_layout_validity() {
        assert_eq!(HELIOS1_LAYOUT.min_columns, 13);
        const { assert!(!HELIOS1_LAYOUT.b_is_se) };
        assert_eq!(HELIOS1_LAYOUT.col_distance_au, Some(3));
    }

    #[test]
    fn test_helios2_layout_validity() {
        assert_eq!(HELIOS2_LAYOUT.min_columns, 13);
        const { assert!(!HELIOS2_LAYOUT.b_is_se) };
    }

    #[test]
    fn test_parse_helios1_inner_heliosphere() {
        // H1 at 0.3 AU perihelion: B ~ 30 nT, n ~ 50 cm^-3
        // B ceiling at 0.3 AU: 200/(0.09) = 2222 nT
        // Density ceiling at 0.3 AU: 500/(0.09) = 5556 cm^-3
        let data = "1976 100 12 0.310 2.0 150.0 28.0 20.0 -15.0 8.0 45.0 400.0 200000.0\n";
        let records = parse_helios_merged(data, HeliosSpacecraft::H1);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 0.31).abs() < 0.01);
        assert!((r.b_magnitude - 28.0).abs() < 0.1);
        assert!((r.proton_density - 45.0).abs() < 0.1);
        assert!((r.bulk_speed - 400.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_helios2_aphelion() {
        // H2 at 0.97 AU (near 1 AU): B ~ 5 nT, n ~ 8 cm^-3
        let data = "1977 200 6 0.970 -1.0 200.0 5.0 3.0 -2.0 1.5 8.0 450.0 100000.0\n";
        let records = parse_helios_merged(data, HeliosSpacecraft::H2);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 0.97).abs() < 0.01);
        assert!((r.b_magnitude - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_helios_fill_values() {
        let data = "1976 100 12 999.999 999.99 999.99 9999.99 9999.99 9999.99 9999.99 999.9 9999.9 999999.0\n";
        let records = parse_helios_merged(data, HeliosSpacecraft::H1);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan());
        assert!(r.b_magnitude.is_nan());
        assert!(r.proton_density.is_nan());
        assert!(r.bulk_speed.is_nan());
        assert!(r.proton_temperature.is_nan());
    }

    #[test]
    fn test_helios_to_omni_rtn() {
        let data = "1976 100 12 0.310 2.0 150.0 28.0 20.0 -15.0 8.0 45.0 400.0 200000.0\n";
        let spdf = parse_helios_merged(data, HeliosSpacecraft::H1);
        let omni = helios_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        let o = &omni[0];
        // RTN -> GSE via rotation by spacecraft longitude (150 deg)
        assert!((o.bx_gse - (-9.8205)).abs() < 0.01);
        assert!((o.by_gse - 22.9904).abs() < 0.01);
        assert!((o.bz_gse - 8.0).abs() < 0.1);
        assert!((o.r_au - 0.31).abs() < 0.01);
    }
}
