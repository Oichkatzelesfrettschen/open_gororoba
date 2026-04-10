//! Fetch logic for Parker Solar Probe SWEAP SPC L3 ion-moment data.
//!
//! Parse logic and record types live in `psp_spc`.

use super::psp_spc::{PspSpcL3iRecord, parse_psp_spc_l3i_csv};
use crate::{
    cdf_support::filename_date_yyyymmdd,
    fetcher::{
        DatasetProvider, FetchConfig, FetchError, download_hapi_csv, download_to_file,
        download_to_string,
    },
};
use chrono::NaiveDate;
use regex::Regex;
use std::path::{Path, PathBuf};

const PSP_SPC_L3I_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/pub/data/psp/sweap/spc/l3/l3i/";
const PSP_SPC_L3I_HAPI_DATASET: &str = "PSP_SWP_SPC_L3I";

pub fn parse_psp_spc_l3i_file(path: &Path) -> Result<Vec<PspSpcL3iRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    let rows = parse_psp_spc_l3i_csv(&content);
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "PSP SPC L3I CSV {} had no finite hourly rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub struct PspSpcL3iProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for PspSpcL3iProvider {
    fn default() -> Self {
        Self {
            year_start: 2025,
            year_end: 2025,
        }
    }
}

impl DatasetProvider for PspSpcL3iProvider {
    fn name(&self) -> &str {
        "Parker Solar Probe SWEAP SPC L3 ion moments"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("psp").join("sweap_spc_l3i");
        std::fs::create_dir_all(&root)?;
        for year in self.year_start..=self.year_end {
            let year_url = format!("{PSP_SPC_L3I_ROOT}{year}/");
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            for entry in directory_entries(&year_url)? {
                if !entry.ends_with(".cdf") {
                    continue;
                }
                let output = year_dir.join(&entry);
                if !(config.skip_existing && output.exists()) {
                    let url = format!("{year_url}{entry}");
                    download_to_file(&url, &output)?;
                    log::info!("saved {}", output.display());
                }
                let Some((entry_year, month, day)) = filename_date_yyyymmdd(Path::new(&entry))
                else {
                    continue;
                };
                let Some(day_start) = NaiveDate::from_ymd_opt(
                    i32::from(entry_year),
                    u32::from(month),
                    u32::from(day),
                ) else {
                    continue;
                };
                let next_day = day_start
                    .succ_opt()
                    .ok_or_else(|| FetchError::Validation(format!("advance {day_start}")))?;
                let csv_output = year_dir.join(format!(
                    "psp_swp_spc_l3i_{entry_year:04}{month:02}{day:02}.csv"
                ));
                if config.skip_existing && csv_output.exists() {
                    continue;
                }
                match download_hapi_csv(
                    PSP_SPC_L3I_HAPI_DATASET,
                    &format!("{}T00:00:00Z", day_start.format("%Y-%m-%d")),
                    &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                    Some(&[
                        "Time",
                        "general_flag",
                        "vp_moment_RTN_gd",
                        "np_moment_gd",
                        "wp_moment_gd",
                    ]),
                ) {
                    Ok(body) => {
                        std::fs::write(&csv_output, body)?;
                        log::info!("saved {}", csv_output.display());
                    }
                    Err(err) => {
                        log::warn!(
                            "failed to download PSP SPC L3I for {}-{:02}-{:02}: {}",
                            entry_year,
                            month,
                            day,
                            err
                        );
                    }
                }
            }
        }
        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let root = config.output_dir.join("psp").join("sweap_spc_l3i");
        std::fs::read_dir(root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| entry.path().is_dir())
    }
}

fn directory_entries(url: &str) -> Result<Vec<String>, FetchError> {
    let html = download_to_string(url)?;
    let regex = Regex::new(r#"href="([^"]+)""#).expect("valid directory regex");
    let mut entries = Vec::new();
    for capture in regex.captures_iter(&html) {
        let href = capture
            .get(1)
            .map(|value| value.as_str())
            .unwrap_or_default();
        if href == "../" || href == "./" || href.starts_with('?') || href.starts_with('/') {
            continue;
        }
        entries.push(href.to_string());
    }
    entries.sort();
    entries.dedup();
    Ok(entries)
}
