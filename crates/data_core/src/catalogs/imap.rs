//! IMAP public-product staging providers.
//!
//! Current public official surfaces verified from the CDAWeb/SPDF archive:
//!   - IMAP helio1hr position support:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/helio1hr/>
//!   - IMAP-Hi L2 ENA h90 product family:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/hi/l2/h90-ena-h-sf-nsp-full-4deg-3mo/>
//!
//! These providers stage the official CDF products into the Rust fetch lane.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file, download_to_string};
use regex::Regex;
use std::path::PathBuf;

const IMAP_HELIO1HR_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/pub/data/imap/helio1hr/";
const IMAP_HI_L2_H90_ROOT: &str =
    "https://cdaweb.gsfc.nasa.gov/pub/data/imap/hi/l2/h90-ena-h-sf-nsp-full-4deg-3mo/";

fn directory_entries(url: &str) -> Result<Vec<String>, FetchError> {
    let html = download_to_string(url)?;
    let regex = Regex::new(r#"href="([^"]+)""#)
        .map_err(|err| FetchError::Validation(format!("invalid IMAP directory regex: {err}")))?;
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

fn has_matching_files(dir: &std::path::Path, suffix: &str) -> bool {
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => return false,
            };
            if !file_type.is_file() {
                return false;
            }
            entry.file_name().to_string_lossy().ends_with(suffix)
        })
}

/// Public IMAP heliocentric hourly position support from the official archive.
pub struct ImapHelio1hrProvider;

impl DatasetProvider for ImapHelio1hrProvider {
    fn name(&self) -> &str {
        "IMAP Helio1hr Position"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("imap").join("helio1hr");
        std::fs::create_dir_all(&dir)?;
        for entry in directory_entries(IMAP_HELIO1HR_ROOT)? {
            if !entry.ends_with(".cdf") {
                continue;
            }
            let output = dir.join(&entry);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{IMAP_HELIO1HR_ROOT}{entry}");
            download_to_file(&url, &output)?;
            log::info!("saved {}", output.display());
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        has_matching_files(&config.output_dir.join("imap").join("helio1hr"), ".cdf")
    }
}

/// Public IMAP-Hi L2 ENA h90 CDF staging provider.
pub struct ImapHiL2H90Provider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for ImapHiL2H90Provider {
    fn default() -> Self {
        Self {
            year_start: 2026,
            year_end: 2026,
        }
    }
}

impl DatasetProvider for ImapHiL2H90Provider {
    fn name(&self) -> &str {
        "IMAP-Hi L2 ENA h90"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("imap").join("hi").join("l2").join("h90");
        std::fs::create_dir_all(&root)?;
        for year in self.year_start..=self.year_end {
            let year_url = format!("{IMAP_HI_L2_H90_ROOT}{year}/");
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            for entry in directory_entries(&year_url)? {
                if !entry.ends_with(".cdf") {
                    continue;
                }
                let output = year_dir.join(&entry);
                if config.skip_existing && output.exists() {
                    continue;
                }
                let url = format!("{year_url}{entry}");
                download_to_file(&url, &output)?;
                log::info!("saved {}", output.display());
            }
        }
        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let root = config.output_dir.join("imap").join("hi").join("l2").join("h90");
        if has_matching_files(&root, ".cdf") {
            return true;
        }
        std::fs::read_dir(&root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let file_type = match entry.file_type() {
                    Ok(file_type) => file_type,
                    Err(_) => return false,
                };
                if !file_type.is_dir() {
                    return false;
                }
                has_matching_files(&entry.path(), ".cdf")
            })
    }
}
