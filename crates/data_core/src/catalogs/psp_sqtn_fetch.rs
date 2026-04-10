//! Fetch logic for Parker Solar Probe FIELDS SQTN density/temperature data.
//!
//! Parse logic and record types live in `psp_sqtn`.

use super::psp_sqtn::{PspSqtnRecord, parse_psp_sqtn_csv};
use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use chrono::{Datelike, NaiveDate};
use std::path::PathBuf;

const PSP_SQTN_DATASET: &str = "PSP_FLD_L3_SQTN_RFS_V1V2";

pub fn parse_psp_sqtn_file(path: &std::path::Path) -> Result<Vec<PspSqtnRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    let rows = parse_psp_sqtn_csv(&content);
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "PSP SQTN CSV {} had no finite hourly rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub struct PspSqtnProvider {
    pub year_start: u16,
    pub year_end: u16,
    pub month_start: u8,
    pub month_end: u8,
}

impl Default for PspSqtnProvider {
    fn default() -> Self {
        Self {
            year_start: 2025,
            year_end: 2025,
            month_start: 9,
            month_end: 9,
        }
    }
}

impl DatasetProvider for PspSqtnProvider {
    fn name(&self) -> &str {
        "Parker Solar Probe SQTN RFS V1V2"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("psp").join("sqtn_rfs_v1v2");
        std::fs::create_dir_all(&root)?;
        for year in self.year_start..=self.year_end {
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            for month in self.month_start..=self.month_end {
                let mut day =
                    NaiveDate::from_ymd_opt(year as i32, month as u32, 1).ok_or_else(|| {
                        FetchError::Validation(format!("invalid PSP SQTN month {year}-{month:02}"))
                    })?;
                let end = if month == 12 {
                    NaiveDate::from_ymd_opt(year as i32 + 1, 1, 1).expect("valid January date")
                } else {
                    NaiveDate::from_ymd_opt(year as i32, month as u32 + 1, 1)
                        .expect("valid next-month date")
                };
                while day < end {
                    let next_day = day
                        .succ_opt()
                        .ok_or_else(|| FetchError::Validation(format!("advance {day}")))?;
                    let output = year_dir.join(format!(
                        "psp_fld_l3_sqtn_rfs_v1v2_{}{:02}{:02}.csv",
                        day.year(),
                        day.month(),
                        day.day()
                    ));
                    if config.skip_existing && output.exists() {
                        day = next_day;
                        continue;
                    }
                    match download_hapi_csv(
                        PSP_SQTN_DATASET,
                        &format!("{}T00:00:00Z", day.format("%Y-%m-%d")),
                        &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                        Some(&[
                            "Time",
                            "electron_density",
                            "density_quality_flag",
                            "electron_core_temperature",
                        ]),
                    ) {
                        Ok(body) => {
                            std::fs::write(&output, body)?;
                            log::info!("saved {}", output.display());
                        }
                        Err(err) => {
                            log::warn!(
                                "failed to download PSP SQTN for {}-{:02}-{:02}: {}",
                                day.year(),
                                day.month(),
                                day.day(),
                                err
                            );
                        }
                    }
                    day = next_day;
                }
            }
        }
        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let root = config.output_dir.join("psp").join("sqtn_rfs_v1v2");
        std::fs::read_dir(root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| entry.path().is_dir())
    }
}
