//! Fetch/provider support for IBEX ENA sky maps and orbit data.

use crate::fetcher::{
    DatasetProvider, FetchConfig, FetchError, download_hapi_csv, download_to_file,
    download_to_string,
};
use regex::Regex;
use std::path::PathBuf;

const IBEX_RELEASE17_ROOT: &str = "https://spdf.gsfc.nasa.gov/pub/data/ibex/release17/";
const IBEX_LO_HYDROGEN_ROOT: &str =
    "https://spdf.gsfc.nasa.gov/pub/data/ibex/release17/Lo-Hydrogen/";
const IBEX_ORBIT_HAPI_DATASET: &str = "IBEX_OR_SSC";

fn directory_entries(url: &str) -> Result<Vec<String>, FetchError> {
    let html = download_to_string(url)?;
    let regex = Regex::new(r#"href="([^"]+)""#)
        .map_err(|err| FetchError::Validation(format!("invalid IBEX directory regex: {err}")))?;
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

fn has_flux_txt(root: &std::path::Path) -> bool {
    std::fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => return false,
            };
            if file_type.is_file() {
                return entry.file_name().to_string_lossy().ends_with("-flux.txt");
            }
            if file_type.is_dir() {
                return has_flux_txt(&entry.path());
            }
            false
        })
}

/// IBEX ENA sky map dataset provider.
pub struct IbexProvider {
    /// Start year (inclusive) from the official release17 archive.
    pub year_start: u16,
    /// End year (inclusive) from the official release17 archive.
    pub year_end: u16,
}

impl Default for IbexProvider {
    fn default() -> Self {
        Self {
            year_start: 2009,
            year_end: 2009,
        }
    }
}

impl DatasetProvider for IbexProvider {
    fn name(&self) -> &str {
        "IBEX ENA Sky Maps"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ibex").join("release17");
        std::fs::create_dir_all(&dir)?;

        let readme = dir.join("ibex17_readme.txt");
        if !config.skip_existing || !readme.exists() {
            let readme_url = format!("{IBEX_RELEASE17_ROOT}ibex17_readme.txt");
            if let Err(err) = download_to_file(&readme_url, &readme) {
                log::warn!(
                    "failed to download IBEX readme from {}: {}",
                    readme_url,
                    err
                );
            }
        }

        for year in self.year_start..=self.year_end {
            let year_slug = format!("lvset_h_cg_hb_{year}");
            let year_url = format!("{IBEX_LO_HYDROGEN_ROOT}{year_slug}/");
            let year_dir = dir.join(&year_slug);
            std::fs::create_dir_all(&year_dir)?;
            let entries = match directory_entries(&year_url) {
                Ok(entries) => entries,
                Err(err) => {
                    log::warn!("failed to list IBEX year {}: {}", year, err);
                    continue;
                }
            };
            for entry in entries {
                let keep = entry.ends_with("-flux.txt")
                    || entry.ends_with("-desc.txt")
                    || entry.ends_with("-ener.txt");
                if !keep {
                    continue;
                }
                let output = year_dir.join(&entry);
                if config.skip_existing && output.exists() {
                    continue;
                }
                let url = format!("{year_url}{entry}");
                match download_to_file(&url, &output) {
                    Ok(_) => log::info!("saved {}", output.display()),
                    Err(err) => log::warn!("failed to download {}: {}", url, err),
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let release_root = config.output_dir.join("ibex").join("release17");
        has_flux_txt(&release_root)
            || std::fs::read_dir(config.output_dir.join("ibex"))
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|entry| entry.ok())
                .any(|entry| entry.file_name().to_string_lossy().ends_with(".csv"))
    }
}

/// IBEX orbit support dataset provider.
pub struct IbexOrbitProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for IbexOrbitProvider {
    fn default() -> Self {
        Self {
            year_start: 2016,
            year_end: 2016,
        }
    }
}

impl DatasetProvider for IbexOrbitProvider {
    fn name(&self) -> &str {
        "IBEX Orbit SSC"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ibex").join("orbits");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("ibex_or_ssc_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                IBEX_ORBIT_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&["Time", "RADIUS"]),
            ) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download IBEX orbit support {}: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("ibex").join("orbits");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("ibex_or_ssc_") && name.ends_with(".csv")
            })
    }
}
