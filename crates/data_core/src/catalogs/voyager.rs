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

use crate::catalogs::omni::OmniRecord;
use crate::catalogs::spdf_merged::{SpdfColumnLayout, SpdfMergedRecord, parse_spdf_merged, spdf_to_omni};
use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string};
use std::path::PathBuf;

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

/// Which Voyager spacecraft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoyagerSpacecraft {
    V1,
    V2,
}

/// Parse Voyager merged hourly data from a string.
pub fn parse_voyager_merged(content: &str, spacecraft: VoyagerSpacecraft) -> Vec<SpdfMergedRecord> {
    let layout = match spacecraft {
        VoyagerSpacecraft::V1 => &VOYAGER1_LAYOUT,
        VoyagerSpacecraft::V2 => &VOYAGER2_LAYOUT,
    };
    parse_spdf_merged(content, layout)
}

/// Parse Voyager merged hourly data from a file.
pub fn parse_voyager_file(
    path: &std::path::Path,
    spacecraft: VoyagerSpacecraft,
) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    Ok(parse_voyager_merged(&content, spacecraft))
}

/// Convert Voyager records to OmniRecord format.
///
/// Sets r_au, lat_deg, lon_deg from parsed position columns.
/// B-field is in SE coordinates (close to GSE for near-ecliptic trajectory).
pub fn voyager_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    spdf_to_omni(records, true) // SE coordinates
}

/// Base URLs for Voyager merged hourly data at SPDF.
const VOYAGER1_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/merged/";
const VOYAGER2_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/merged/";

/// NASA SPDF Voyager dataset provider.
pub struct VoyagerProvider {
    /// Which spacecraft (V1 or V2).
    pub spacecraft: VoyagerSpacecraft,
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for VoyagerProvider {
    fn default() -> Self {
        Self {
            spacecraft: VoyagerSpacecraft::V1,
            year_start: 2020,
            year_end: 2020,
        }
    }
}

impl DatasetProvider for VoyagerProvider {
    fn name(&self) -> &str {
        match self.spacecraft {
            VoyagerSpacecraft::V1 => "Voyager 1 Merged Hourly",
            VoyagerSpacecraft::V2 => "Voyager 2 Merged Hourly",
        }
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let subdir = match self.spacecraft {
            VoyagerSpacecraft::V1 => "voyager1",
            VoyagerSpacecraft::V2 => "voyager2",
        };
        let dir = config.output_dir.join(subdir);
        std::fs::create_dir_all(&dir)?;

        let base = match self.spacecraft {
            VoyagerSpacecraft::V1 => VOYAGER1_BASE,
            VoyagerSpacecraft::V2 => VOYAGER2_BASE,
        };

        for year in self.year_start..=self.year_end {
            let fname = format!("vy{}_{}_{}.asc",
                match self.spacecraft {
                    VoyagerSpacecraft::V1 => "1",
                    VoyagerSpacecraft::V2 => "2",
                },
                year,
                "merged_hourly",
            );
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}", base, fname);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download Voyager {}: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let subdir = match self.spacecraft {
            VoyagerSpacecraft::V1 => "voyager1",
            VoyagerSpacecraft::V2 => "voyager2",
        };
        config.output_dir.join(subdir).exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voyager1_layout_validity() {
        assert_eq!(VOYAGER1_LAYOUT.min_columns, 13);
        assert!(VOYAGER1_LAYOUT.b_is_se);
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
        assert!((omni[0].r_au - 94.0).abs() < 0.1, "r_au should be populated");
        assert!((omni[0].lat_deg - 34.5).abs() < 0.1, "lat_deg should be populated");
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
}
