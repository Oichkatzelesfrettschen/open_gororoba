//! Fetch/provider support for WIND SWE, MFI, and AMDA fallback data.

use crate::{
    catalogs::omni::OmniRecord,
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_amda_hapi_csv, download_to_file},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// SPDF direct download providers
// ---------------------------------------------------------------------------

const WIND_MFI_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/ascii/1hour_ascii/";
const WIND_SWE_BASE: &str =
    "https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/swe_kp_unspike/wind_kp_unspike";

/// WIND MFI dataset provider (1-hour magnetic field).
pub struct WindMfiProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for WindMfiProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for WindMfiProvider {
    fn name(&self) -> &str {
        "WIND MFI 1-hour Magnetic Field"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("wind_mfi");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            for month in 1..=12u8 {
                let fname = format!("{}{:02}_wind_mag_1hour.asc", year, month);
                let output = dir.join(&fname);
                if config.skip_existing && output.exists() {
                    continue;
                }
                let url = format!("{}{}", WIND_MFI_BASE, fname);
                match download_to_file(&url, &output) {
                    Ok(bytes) if bytes > 0 => {
                        log::info!("Saved {} ({} bytes)", fname, bytes);
                    }
                    _ => {
                        log::debug!("WIND MFI {} not found", fname);
                    }
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("wind_mfi").exists()
    }
}

/// WIND SWE dataset provider (KP unspiked plasma).
pub struct WindSweProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for WindSweProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for WindSweProvider {
    fn name(&self) -> &str {
        "WIND SWE KP Unspiked Plasma"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("wind_swe");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("wind_kp_unspike{}.txt", year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}.txt", WIND_SWE_BASE, year);
            match download_to_file(&url, &output) {
                Ok(bytes) if bytes > 0 => {
                    log::info!("Saved {} ({} bytes)", fname, bytes);
                }
                _ => {
                    log::warn!("Failed to download WIND SWE {}", year);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("wind_swe").exists()
    }
}

// ---------------------------------------------------------------------------
// AMDA fallback: Wind plasma + MAG from AMDA HAPI
// ---------------------------------------------------------------------------
//
// WHY: SPDF/CDAWeb HAPI can be blocked from non-NASA networks.  AMDA provides
// independent KP-quality mirrors of WIND SWE and MFI data under HAPI-compatible
// dataset IDs, covering the full mission from 1994 to present.
//
// Lane:
//   1. SPDF direct ASC download (WindMfiProvider / WindSweProvider above).
//   2. AMDA HAPI: wnd-swe-kp (plasma: density, speed, temperature) + wnd-mfi-kp
//      (MAG: Bx, By, Bz GSE + |B|), merged by hour into OmniRecord-compatible
//      format and staged as merged CSV.

const WIND_AMDA_SWE: &str = "wnd-swe-kp";
const WIND_AMDA_MFI: &str = "wnd-mfi-kp";

/// An hourly WIND AMDA-derived plasma record.
#[derive(Debug, Clone)]
pub struct WindAmdaPlasmaRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    /// Proton density (cm^-3).
    pub density: f64,
    /// Bulk flow speed (km/s).
    pub speed: f64,
    /// Proton temperature (K).
    pub temperature: f64,
}

/// An hourly WIND AMDA-derived MAG record (GSE coordinates).
#[derive(Debug, Clone)]
pub struct WindAmdaMagRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub bx_gse: f64,
    pub by_gse: f64,
    pub bz_gse: f64,
    pub b_magnitude: f64,
}

/// Parse WIND AMDA SWE KP CSV (`wnd-swe-kp`).
///
/// Columns after the timestamp: density, speed, temperature (positional).
pub fn parse_wind_amda_swe(content: &str) -> Vec<WindAmdaPlasmaRecord> {
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
        out.push(WindAmdaPlasmaRecord {
            year,
            doy,
            hour,
            density,
            speed,
            temperature: temp,
        });
    }
    out
}

/// Parse WIND AMDA MFI KP CSV (`wnd-mfi-kp`).
///
/// Columns after the timestamp: Bx_GSE, By_GSE, Bz_GSE, |B| (positional).
pub fn parse_wind_amda_mfi(content: &str) -> Vec<WindAmdaMagRecord> {
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
        let bx = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let by = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let bz = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        let b_mag = parse_hapi_spacephysics_f64_or_nan(record.get(4).unwrap_or(""));
        out.push(WindAmdaMagRecord {
            year,
            doy,
            hour,
            bx_gse: bx,
            by_gse: by,
            bz_gse: bz,
            b_magnitude: b_mag,
        });
    }
    out
}

/// Merge WIND AMDA plasma + MAG lanes into `OmniRecord` rows.
///
/// Only hours present in BOTH lanes are emitted.
pub fn merge_wind_amda(
    plasma: &[WindAmdaPlasmaRecord],
    mag: &[WindAmdaMagRecord],
) -> Vec<OmniRecord> {
    use std::collections::BTreeMap;
    let plasma_map: BTreeMap<(u16, u16, u8), &WindAmdaPlasmaRecord> = plasma
        .iter()
        .map(|r| ((r.year, r.doy, r.hour), r))
        .collect();
    let mag_map: BTreeMap<(u16, u16, u8), &WindAmdaMagRecord> =
        mag.iter().map(|r| ((r.year, r.doy, r.hour), r)).collect();

    let mut rows = Vec::new();
    for (&key, p) in &plasma_map {
        let Some(m) = mag_map.get(&key) else {
            continue;
        };
        rows.push(OmniRecord {
            year: key.0,
            doy: key.1,
            hour: key.2,
            b_magnitude: m.b_magnitude,
            bx_gse: m.bx_gse,
            by_gse: m.by_gse,
            bz_gse: m.bz_gse,
            proton_temperature: p.temperature,
            proton_density: p.density,
            bulk_speed: p.speed,
            // Fields not available from WIND AMDA KP datasets.
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 99,
            r_au: 1.0,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
        });
    }
    rows.sort_by_key(|r| (r.year, r.doy, r.hour));
    rows
}

/// WIND AMDA provider -- fetches SWE + MFI lanes and merges them.
///
/// Use as a fallback when the SPDF-based `WindSweProvider` / `WindMfiProvider`
/// are unreachable.  Produces yearly merged CSV files under
/// `data/external/wind/amda/`.
pub struct WindAmdaProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for WindAmdaProvider {
    fn default() -> Self {
        Self {
            year_start: 1995,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for WindAmdaProvider {
    fn name(&self) -> &str {
        "WIND AMDA (SWE+MFI)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("wind").join("amda");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let t_min = format!("{year}-01-01T00:00:00Z");
            let t_max = format!("{}-01-01T00:00:00Z", year + 1);
            let out_path = dir.join(format!("wind_amda_merged_{year}.csv"));
            if config.skip_existing && out_path.exists() {
                continue;
            }

            let swe_csv = match download_amda_hapi_csv(WIND_AMDA_SWE, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA WIND SWE {year}: {e}");
                    continue;
                }
            };
            let mfi_csv = match download_amda_hapi_csv(WIND_AMDA_MFI, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA WIND MFI {year}: {e}");
                    continue;
                }
            };

            let plasma = parse_wind_amda_swe(&swe_csv);
            let mag = parse_wind_amda_mfi(&mfi_csv);
            let merged = merge_wind_amda(&plasma, &mag);

            let mut csv_buf = String::from(
                "year,doy,hour,bx_gse,by_gse,bz_gse,b_mag,density,speed,temperature\n",
            );
            for r in &merged {
                csv_buf.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},{}\n",
                    r.year,
                    r.doy,
                    r.hour,
                    r.bx_gse,
                    r.by_gse,
                    r.bz_gse,
                    r.b_magnitude,
                    r.proton_density,
                    r.bulk_speed,
                    r.proton_temperature,
                ));
            }
            std::fs::write(&out_path, csv_buf)?;
            log::info!(
                "AMDA WIND {year}: merged {} hourly records -> {}",
                merged.len(),
                out_path.display()
            );
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("wind").join("amda");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("wind_amda_merged_") && name.ends_with(".csv")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_wind_amda_swe_format() {
        // Minimal AMDA SWE CSV with header.
        let csv = "Time,n_p,v_p,T_p\n\
                   2010-03-15T12:00:00.000Z,5.2,380.0,90000.0\n";
        let rows = parse_wind_amda_swe(csv);
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.year, 2010);
        assert!((r.density - 5.2).abs() < 0.01);
        assert!((r.speed - 380.0).abs() < 0.1);
        assert!((r.temperature - 90000.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_wind_amda_mfi_format() {
        let csv = "Time,Bx_GSE,By_GSE,Bz_GSE,B\n\
                   2010-03-15T12:00:00.000Z,2.1,-1.5,0.8,2.72\n";
        let rows = parse_wind_amda_mfi(csv);
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.year, 2010);
        assert!((r.bx_gse - 2.1).abs() < 0.01);
        assert!((r.by_gse - (-1.5)).abs() < 0.01);
        assert!((r.bz_gse - 0.8).abs() < 0.01);
        assert!((r.b_magnitude - 2.72).abs() < 0.01);
    }

    #[test]
    fn test_merge_wind_amda_intersection() {
        let plasma = vec![WindAmdaPlasmaRecord {
            year: 2010,
            doy: 74,
            hour: 12,
            density: 5.2,
            speed: 380.0,
            temperature: 90000.0,
        }];
        let mag = vec![WindAmdaMagRecord {
            year: 2010,
            doy: 74,
            hour: 12,
            bx_gse: 2.1,
            by_gse: -1.5,
            bz_gse: 0.8,
            b_magnitude: 2.72,
        }];
        let merged = merge_wind_amda(&plasma, &mag);
        assert_eq!(merged.len(), 1);
        let r = &merged[0];
        assert_eq!(r.year, 2010);
        assert!((r.proton_density - 5.2).abs() < 0.01);
        assert!((r.bx_gse - 2.1).abs() < 0.01);
        assert!((r.bz_gse - 0.8).abs() < 0.01);
        assert_eq!(r.r_au, 1.0);
    }

    #[test]
    fn test_merge_wind_amda_no_overlap() {
        // Plasma at hour 12, MAG at hour 13 -- no intersection.
        let plasma = vec![WindAmdaPlasmaRecord {
            year: 2010,
            doy: 74,
            hour: 12,
            density: 5.2,
            speed: 380.0,
            temperature: 90000.0,
        }];
        let mag = vec![WindAmdaMagRecord {
            year: 2010,
            doy: 74,
            hour: 13,
            bx_gse: 2.1,
            by_gse: -1.5,
            bz_gse: 0.8,
            b_magnitude: 2.72,
        }];
        let merged = merge_wind_amda(&plasma, &mag);
        assert!(merged.is_empty(), "no shared hours -> empty merge");
    }
}
