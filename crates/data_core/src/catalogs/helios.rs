//! Helios 1 & 2 merged hourly data parser.
//!
//! Helios spacecraft provide inner heliospheric coverage (0.29-0.98 AU):
//!   H1: launched 1974-12-10, perihelion 0.31 AU, data to 1985
//!   H2: launched 1976-01-15, perihelion 0.29 AU, data to 1980
//!
//! Official sources:
//!   SPDF merged yearly ASCII:
//!     <https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/merged/>
//!     <https://spdf.gsfc.nasa.gov/pub/data/helios/helios2/merged/>
//!   CDAWeb HAPI mirrors:
//!     HELIOS1_COHO1HR_MERGED_MAG_PLASMA
//!     HELIOS2_COHO1HR_MERGED_MAG_PLASMA
//!
//! Fill values and column order follow the SPDF merged readmes (`he1mgd.txt`,
//! `he2mgd.txt`). B-field coordinates are RTN.

use crate::{
    catalogs::{
        omni::OmniRecord,
        spdf_fleet::SpdfMission,
        spdf_merged::{SpdfColumnLayout, SpdfMergedRecord},
    },
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv, download_to_file},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::path::PathBuf;

/// Column layout for Helios 1 merged hourly data from SPDF.
pub const HELIOS1_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 20,
    col_year: 0,
    col_doy: 1,
    col_hour: 2,
    col_distance_au: Some(4),
    col_lat_deg: Some(5),
    col_lon_deg: Some(6),
    col_b_mag: Some(14),
    col_br: Some(11),
    col_bt: Some(12),
    col_bn: Some(13),
    col_density: Some(18),
    col_speed: Some(15),
    col_temperature: Some(19),
    fill_b: 99999.99,
    fill_density: 999.9,
    fill_speed: 9999.9,
    fill_temperature: 9999999.0,
    fill_distance: 999.99,
    b_is_se: false,
};

/// Column layout for Helios 2 merged hourly data.
pub const HELIOS2_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 20,
    col_year: 0,
    col_doy: 1,
    col_hour: 2,
    col_distance_au: Some(4),
    col_lat_deg: Some(5),
    col_lon_deg: Some(6),
    col_b_mag: Some(14),
    col_br: Some(11),
    col_bt: Some(12),
    col_bn: Some(13),
    col_density: Some(18),
    col_speed: Some(15),
    col_temperature: Some(19),
    fill_b: 99999.99,
    fill_density: 999.9,
    fill_speed: 9999.9,
    fill_temperature: 9999999.0,
    fill_distance: 999.99,
    b_is_se: false,
};

/// Helios B fill (nT).
pub const HELIOS_FILL_B: f64 = 99999.99;
/// Helios density fill (cm^-3).
pub const HELIOS_FILL_DENSITY: f64 = 999.9;
/// Helios speed fill (km/s).
pub const HELIOS_FILL_SPEED: f64 = 9999.9;
/// Helios temperature fill (K).
pub const HELIOS_FILL_TEMP: f64 = 9999999.0;
/// Helios distance fill (AU).
pub const HELIOS_FILL_DISTANCE: f64 = 999.99;

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
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    if path.extension().and_then(|value| value.to_str()) == Some("csv") {
        Ok(parse_helios_hapi_csv(&content))
    } else {
        Ok(parse_helios_merged(&content, spacecraft))
    }
}

/// Convert Helios records to OmniRecord format.
///
/// B-field is in RTN coordinates (sign flip on Bt for GSE conversion).
/// Both H1 and H2 use RTN coordinates, so no spacecraft dispatch is needed here.
pub fn helios_to_omni(records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
    // Both HELIOS1_MISSION and HELIOS2_MISSION have b_is_se=false; same result.
    HELIOS1_MISSION.to_omni(records)
}

fn parse_helios_hapi_csv(content: &str) -> Vec<SpdfMergedRecord> {
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
            distance_au: parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or("")),
            lat_deg: parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or("")),
            lon_deg: parse_hapi_spacephysics_f64_or_nan(record.get(4).unwrap_or("")),
            br: parse_hapi_spacephysics_f64_or_nan(record.get(9).unwrap_or("")),
            bt: parse_hapi_spacephysics_f64_or_nan(record.get(10).unwrap_or("")),
            bn: parse_hapi_spacephysics_f64_or_nan(record.get(11).unwrap_or("")),
            b_magnitude: parse_hapi_spacephysics_f64_or_nan(record.get(12).unwrap_or("")),
            bulk_speed: parse_hapi_spacephysics_f64_or_nan(record.get(13).unwrap_or("")),
            proton_density: parse_hapi_spacephysics_f64_or_nan(record.get(16).unwrap_or("")),
            proton_temperature: parse_hapi_spacephysics_f64_or_nan(record.get(17).unwrap_or("")),
        });
    }
    rows
}

const HELIOS1_HAPI_DATASET: &str = "HELIOS1_COHO1HR_MERGED_MAG_PLASMA";
const HELIOS2_HAPI_DATASET: &str = "HELIOS2_COHO1HR_MERGED_MAG_PLASMA";
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
        let hapi_dataset = match self.spacecraft {
            HeliosSpacecraft::H1 => HELIOS1_HAPI_DATASET,
            HeliosSpacecraft::H2 => HELIOS2_HAPI_DATASET,
        };

        for year in self.year_start..=self.year_end {
            let asc_name = match self.spacecraft {
                HeliosSpacecraft::H1 => format!("he1_{year}.asc"),
                HeliosSpacecraft::H2 => format!("he2_{year}.asc"),
            };
            let asc_output = dir.join(&asc_name);
            let csv_name = format!("{subdir}_{year}_merged_hapi.csv");
            let csv_output = dir.join(&csv_name);
            if config.skip_existing && (asc_output.exists() || csv_output.exists()) {
                continue;
            }

            let asc_url = format!("{base}{asc_name}");
            match download_to_file(&asc_url, &asc_output) {
                Ok(_) => {
                    log::info!("saved {}", asc_name);
                    continue;
                }
                Err(e) => {
                    log::warn!(
                        "failed to download official Helios merged file {}: {}",
                        asc_url,
                        e
                    );
                }
            }

            match download_hapi_csv(
                hapi_dataset,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&[
                    "Time",
                    "RotationNumber",
                    "heliocentricDistance",
                    "heliographicLatitude",
                    "heliographicLongitude",
                    "sepAngle",
                    "BX",
                    "BY",
                    "BZ",
                    "BR",
                    "BT",
                    "BN",
                    "B",
                    "flowSpeed",
                    "elevAngle",
                    "azimuthAngle",
                    "protonDensity",
                    "protonTemp",
                ]),
            ) {
                Ok(data) => {
                    std::fs::write(&csv_output, data)?;
                    log::info!("saved {}", csv_name);
                }
                Err(e) => {
                    log::warn!("failed to download Helios {} via HAPI fallback: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let subdir = match self.spacecraft {
            HeliosSpacecraft::H1 => "helios1",
            HeliosSpacecraft::H2 => "helios2",
        };
        let dir = config.output_dir.join("helios").join(subdir);
        let prefix = match self.spacecraft {
            HeliosSpacecraft::H1 => "he1_",
            HeliosSpacecraft::H2 => "he2_",
        };
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                (name.starts_with(prefix) && name.ends_with(".asc"))
                    || (name.starts_with(subdir) && name.ends_with(".csv"))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helios1_layout_validity() {
        assert_eq!(HELIOS1_LAYOUT.min_columns, 20);
        const { assert!(!HELIOS1_LAYOUT.b_is_se) };
        assert_eq!(HELIOS1_LAYOUT.col_distance_au, Some(4));
        assert_eq!(HELIOS1_LAYOUT.col_b_mag, Some(14));
    }

    #[test]
    fn test_helios2_layout_validity() {
        assert_eq!(HELIOS2_LAYOUT.min_columns, 20);
        const { assert!(!HELIOS2_LAYOUT.b_is_se) };
    }

    #[test]
    fn test_parse_helios1_inner_heliosphere() {
        // Official SPDF merged layout includes Carrington rotation and sepAngle.
        let data = "1976 100 12 1650 0.310 2.0 150.0 25.0 0.0 0.0 0.0 20.0 -15.0 8.0 28.0 400.0 0.0 0.0 45.0 200000.0\n";
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
        let data = "1977 200 6 1651 0.970 -1.0 200.0 12.0 0.0 0.0 0.0 3.0 -2.0 1.5 5.0 450.0 0.0 0.0 8.0 100000.0\n";
        let records = parse_helios_merged(data, HeliosSpacecraft::H2);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.distance_au - 0.97).abs() < 0.01);
        assert!((r.b_magnitude - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_helios_fill_values() {
        let data = "1976 100 12 9999 999.99 9999.9 9999.9 9999.9 99999.99 99999.99 99999.99 99999.99 99999.99 99999.99 99999.99 9999.9 9999.9 9999.9 999.9 9999999.0\n";
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
        let data = "1976 100 12 1650 0.310 2.0 150.0 25.0 0.0 0.0 0.0 20.0 -15.0 8.0 28.0 400.0 0.0 0.0 45.0 200000.0\n";
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
