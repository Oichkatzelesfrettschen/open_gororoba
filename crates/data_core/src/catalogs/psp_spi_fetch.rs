//! Fetch logic for Parker Solar Probe SWEAP/SPI SF00 level-3 moment data.
//!
//! Parse logic and record types live in `psp_spi`.

use super::psp_spi::{PspSpiMomRecord, parse_psp_spi_mom_csv};
use crate::{
    cdf_support::filename_date_yyyymmdd,
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv, download_to_string},
};
use chrono::NaiveDate;
use regex::Regex;
use std::path::PathBuf;

const PSP_SPI_SF00_L3_MOM_ROOT: &str =
    "https://cdaweb.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/";
const PSP_SPI_SF00_L3_MOM_HAPI_DATASET: &str = "PSP_SWP_SPI_SF00_L3_MOM";

pub fn parse_psp_spi_mom_file(path: &std::path::Path) -> Result<Vec<PspSpiMomRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    let rows = parse_psp_spi_mom_csv(&content);
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "PSP SPI moment CSV {} had no finite hourly rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub struct PspSpiMomProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for PspSpiMomProvider {
    fn default() -> Self {
        Self {
            year_start: 2025,
            year_end: 2025,
        }
    }
}

impl DatasetProvider for PspSpiMomProvider {
    fn name(&self) -> &str {
        "Parker Solar Probe SWEAP SPI SF00 L3 moments"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("psp").join("sweap_spi_sf00_l3_mom");
        std::fs::create_dir_all(&root)?;
        for year in self.year_start..=self.year_end {
            let year_url = format!("{PSP_SPI_SF00_L3_MOM_ROOT}{year}/");
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            for entry in directory_entries(&year_url)? {
                if !entry.ends_with(".cdf") {
                    continue;
                }
                let Some((entry_year, month, day)) =
                    filename_date_yyyymmdd(std::path::Path::new(&entry))
                else {
                    continue;
                };
                if entry_year != year {
                    continue;
                }
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
                    "psp_swp_spi_sf00_l3_mom_{}{:02}{:02}.csv",
                    entry_year, month, day
                ));
                if config.skip_existing && csv_output.exists() {
                    continue;
                }
                match download_hapi_csv(
                    PSP_SPI_SF00_L3_MOM_HAPI_DATASET,
                    &format!("{}T00:00:00Z", day_start.format("%Y-%m-%d")),
                    &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                    Some(&[
                        "Time",
                        "CNTS",
                        "QUALITY_FLAG",
                        "DENS",
                        "VEL_RTN_SUN",
                        "TEMP",
                        "SUN_DIST",
                    ]),
                ) {
                    Ok(body) => {
                        std::fs::write(&csv_output, body)?;
                        log::info!("saved {}", csv_output.display());
                    }
                    Err(err) => {
                        log::warn!(
                            "failed to download PSP SPI moments for {}-{:02}-{:02}: {}",
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
        let root = config.output_dir.join("psp").join("sweap_spi_sf00_l3_mom");
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
    let regex = Regex::new(r#"href="([^"]+)""#)
        .map_err(|err| FetchError::Validation(format!("invalid PSP SPI directory regex: {err}")))?;
    let mut entries = Vec::new();
    for capture in regex.captures_iter(&html) {
        let Some(href) = capture.get(1).map(|value| value.as_str()) else {
            continue;
        };
        if href.starts_with('/') || href.starts_with('?') || href == "Parent Directory" {
            continue;
        }
        entries.push(href.to_string());
    }
    entries.sort();
    entries.dedup();
    Ok(entries)
}
