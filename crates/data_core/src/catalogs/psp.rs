//! Parker Solar Probe merged hourly data parser.
//!
//! PSP provides the innermost heliospheric sampling (0.05-0.25 AU perihelion):
//!   Launched 2018-08-12, first perihelion 2018-11-05
//!   Instruments: SWEAP (Solar Wind Electrons Alphas and Protons),
//!                FIELDS (electromagnetic fields and waves)
//!
//! AMDA HAPI datasets (primary source when SPDF blocked):
//!   psp-spc-mom: SPC proton moments (density, speed, thermal_speed)
//!   psp-mag-1min: MAG 1-min cadence (Br, Bt, Bn, |B| in RTN)
//!   psp-orb-all: full mission orbit (r_au, lat, lon in HCI)
//!
//! The Python fetcher (bin/fetch_psp.py) translates AMDA HAPI CSV to
//! SPDF-style merged hourly ASCII with per-hour median aggregation.
//! SPC thermal_speed is converted to temperature via T = m_p * v_th^2 / (2*k_B).
//!
//! Fill values match the AMDA-derived output format.
//! B-field coordinate system: RTN (Radial-Tangential-Normal).
//!
//! Source: <https://amda.irap.omp.eu/service/hapi/info?id=psp-spc-mom>

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord, parse_spdf_merged, spdf_to_omni},
    },
    fetcher::{DatasetProvider, FetchConfig, FetchError},
};
use std::path::PathBuf;

/// Column layout for PSP AMDA-derived merged hourly data.
///
/// The 13-column format matches the standard SPDF merged convention
/// used by Voyager/Pioneer, with RTN B-field coordinates from SPC/FIELDS.
pub const PSP_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
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

/// PSP B magnitude fill (nT).
pub const PSP_FILL_B_MAG: f64 = 9999.99;
/// PSP density fill (cm^-3).
pub const PSP_FILL_DENSITY: f64 = 999.9;
/// PSP speed fill (km/s).
pub const PSP_FILL_SPEED: f64 = 9999.9;
/// PSP temperature fill (K).
pub const PSP_FILL_TEMP: f64 = 999999.0;
/// PSP distance fill (AU).
pub const PSP_FILL_DISTANCE: f64 = 999.999;

/// Parse PSP merged hourly data from a string.
pub fn parse_psp_merged(content: &str) -> Vec<SpdfMergedRecord> {
    parse_spdf_merged(content, &PSP_LAYOUT)
}

/// Parse PSP merged hourly data from a file.
pub fn parse_psp_file(path: &std::path::Path) -> Result<Vec<SpdfMergedRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    Ok(parse_psp_merged(&content))
}

/// Convert PSP records to OmniRecord format.
///
/// B-field is in RTN coordinates (sign flip on Bt for GSE conversion).
pub fn psp_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    spdf_to_omni(records, false) // RTN coordinates
}

/// Base URL for PSP merged hourly data at SPDF (blocked from this host).
const PSP_SPDF_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/psp/";

/// NASA PSP dataset provider.
pub struct PspProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for PspProvider {
    fn default() -> Self {
        Self {
            year_start: 2022,
            year_end: 2023,
        }
    }
}

impl DatasetProvider for PspProvider {
    fn name(&self) -> &str {
        "Parker Solar Probe Merged Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("psp");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("psp_{}_amda_merged.asc", year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            // SPDF is blocked; log and continue. Data comes via Python AMDA fetcher.
            let url = format!("{}sweap/spc/l3/{}/", PSP_SPDF_BASE, year);
            log::warn!(
                "PSP SPDF blocked ({}). Use bin/fetch_psp.py --source amda for year {}.",
                url,
                year,
            );
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("psp").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psp_layout_validity() {
        assert_eq!(PSP_LAYOUT.min_columns, 13);
        const { assert!(!PSP_LAYOUT.b_is_se) };
        assert_eq!(PSP_LAYOUT.col_distance_au, Some(3));
        assert_eq!(PSP_LAYOUT.col_b_mag, Some(6));
    }

    #[test]
    fn test_parse_psp_inner_heliosphere() {
        // PSP at 0.1 AU perihelion: B ~ 100 nT, n ~ 300 cm^-3, V ~ 300 km/s
        // B ceiling at 0.1 AU: 200/(0.01) = 20000 nT -- no practical ceiling issue
        // Density ceiling at 0.1 AU: 500/(0.01) = 50000 cm^-3 -- no issue
        let data = "2022 310 12 0.100 3.5 120.0 95.0 80.0 -40.0 20.0 300.0 350.0 500000.0\n";
        let records = parse_psp_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 0.1).abs() < 0.001);
        assert!((r.b_magnitude - 95.0).abs() < 0.1);
        assert!((r.proton_density - 300.0).abs() < 0.1);
        assert!((r.bulk_speed - 350.0).abs() < 0.1);
        assert!((r.proton_temperature - 500000.0).abs() < 1.0);
    }

    #[test]
    fn test_psp_fill_values() {
        let data = "2022 310 12 999.999 999.99 999.99 9999.99 9999.99 9999.99 9999.99 999.9 9999.9 999999.0\n";
        let records = parse_psp_merged(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan());
        assert!(r.b_magnitude.is_nan());
        assert!(r.proton_density.is_nan());
        assert!(r.bulk_speed.is_nan());
        assert!(r.proton_temperature.is_nan());
    }

    #[test]
    fn test_psp_to_omni_rtn_conversion() {
        let data = "2022 310 12 0.100 3.5 120.0 95.0 80.0 -40.0 20.0 300.0 350.0 500000.0\n";
        let spdf = parse_psp_merged(data);
        let omni = psp_to_omni(&spdf);
        assert_eq!(omni.len(), 1);
        let o = &omni[0];
        // RTN -> GSE via rotation by spacecraft longitude (120 deg)
        assert!((o.bx_gse - (-5.359)).abs() < 0.01);
        assert!((o.by_gse - 89.282).abs() < 0.01);
        assert!((o.bz_gse - 20.0).abs() < 0.1);
        assert!((o.r_au - 0.1).abs() < 0.001);
    }
}
