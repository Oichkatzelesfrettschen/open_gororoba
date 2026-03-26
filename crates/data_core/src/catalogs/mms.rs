//! MMS Fluxgate Magnetometer (FGM) data provider.
//!
//! The Magnetospheric Multiscale (MMS) mission provides high-cadence
//! magnetic field measurements. This module fetches Survey (SRVY) mode
//! Level 2 data via CDAWeb HAPI.
//!
//! Source: <https://cdaweb.gsfc.nasa.gov/hapi/info?id=MMS1_FGM_SRVY_L2>
//! Reference: Russell et al. (2016), Space Sci. Rev. 199, 189

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
};

/// MMS FGM Survey Level 2 record.
#[derive(Debug, Clone)]
pub struct MmsFgmRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub bx_gse: f64,
    pub by_gse: f64,
    pub bz_gse: f64,
    pub b_magnitude: f64,
}

#[derive(Default)]
struct FgmHourAccumulator {
    bx_sum: f64,
    by_sum: f64,
    bz_sum: f64,
    bmag_sum: f64,
    count: usize,
}

const MMS_FGM_HAPI_DATASET: &str = "MMS1_FGM_SRVY_L2@0";

/// MMS FGM dataset provider.
pub struct MmsFgmProvider {
    pub year_start: u16,
    pub year_end: u16,
    pub doy_range: Option<(u16, u16)>,
}

impl Default for MmsFgmProvider {
    fn default() -> Self {
        Self {
            year_start: 2015,
            year_end: 2026,
            doy_range: None,
        }
    }
}

impl DatasetProvider for MmsFgmProvider {
    fn name(&self) -> &str {
        "MMS1 FGM Survey L2"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("mms");
        fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            // Respect the startDate of the dataset (2015-09-01)
            let (start_month, start_day) = if year == 2015 { (9, 1) } else { (1, 1) };

            // Monthly chunking to avoid 400 errors on large requests
            for month in start_month..=12 {
                let t_min = format!("{:04}-{:02}-{:02}T00:00:00Z", year, month, start_day);
                
                // End of month or end of year
                let (end_year, end_month) = if month == 12 { (year + 1, 1) } else { (year, month + 1) };
                let t_max = format!("{:04}-{:02}-01T00:00:00Z", end_year, end_month);

                // Stop if we exceed year_end
                if year == self.year_end && month == 12 && self.year_end < 2026 {
                    // This is simple logic, could be more precise with days but months are safe chunks
                }

                let fname = format!("mms1_fgm_srvy_l2_{:04}_{:02}.csv", year, month);
                let output = dir.join(&fname);
                
                if config.skip_existing && output.exists() {
                    continue;
                }

                println!("Fetching MMS1 FGM Survey L2 for {} month {} ({} to {})...", year, month, t_min, t_max);

                let body = download_hapi_csv(
                    MMS_FGM_HAPI_DATASET,
                    &t_min,
                    &t_max,
                    Some(&["Time", "mms1_fgm_b_gse_srvy_l2_clean"]),
                )?;
                fs::write(&output, body)?;
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("mms").exists()
    }
}

pub fn parse_mms_fgm_hapi_csv(content: &str) -> Vec<MmsFgmRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut rows = Vec::new();
    
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else { continue; };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else { continue; };
        
        rows.push(MmsFgmRecord {
            year,
            doy,
            hour,
            bx_gse: parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or("")),
            by_gse: parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or("")),
            bz_gse: parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or("")),
            b_magnitude: parse_hapi_spacephysics_f64_or_nan(record.get(4).unwrap_or("")),
        });
    }
    rows
}

/// Average high-cadence MMS records to hourly bins.
pub fn average_to_hourly(records: &[MmsFgmRecord]) -> Vec<MmsFgmRecord> {
    let mut hourly: BTreeMap<(u16, u16, u8), FgmHourAccumulator> = BTreeMap::new();
    for r in records {
        let entry = hourly.entry((r.year, r.doy, r.hour)).or_default();
        if r.bx_gse.is_finite() && r.by_gse.is_finite() && r.bz_gse.is_finite() {
            entry.bx_sum += r.bx_gse;
            entry.by_sum += r.by_gse;
            entry.bz_sum += r.bz_gse;
            entry.bmag_sum += r.b_magnitude;
            entry.count += 1;
        }
    }

    hourly.into_iter().filter_map(|((year, doy, hour), acc)| {
        if acc.count == 0 { return None; }
        let n = acc.count as f64;
        Some(MmsFgmRecord {
            year,
            doy,
            hour,
            bx_gse: acc.bx_sum / n,
            by_gse: acc.by_sum / n,
            bz_gse: acc.bz_sum / n,
            b_magnitude: acc.bmag_sum / n,
        })
    }).collect()
}
