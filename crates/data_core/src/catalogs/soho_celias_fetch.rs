//! Fetch implementation for soho_celias. See soho_celias.rs for record types and parsers.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file, download_to_string};
use regex::Regex;
use std::path::{Path, PathBuf};

const SOHO_CELIAS_PM5_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/pub/data/soho/celias/pm_5min/";
const SOHO_CELIAS_PM5_BUNDLE_URL: &str =
    "https://soho.nascom.nasa.gov/data/EntireMissionBundles/CELIAS_Proton_Monitor_5min.tar.gz";
const SOHO_LASCO_LEVEL05_ROOT: &str = "https://umbra.nascom.nasa.gov/pub/lasco_level05/";
const SOHO_METADATA_URLS: &[(&str, &str)] = &[
    (
        "gsfc_index.html",
        "https://soho.nascom.nasa.gov/data/archive/index_gsfc.html",
    ),
    (
        "gsfc_archive.html",
        "https://soho.nascom.nasa.gov/data/archive.html",
    ),
    (
        "esa_cmdline.html",
        "https://www.cosmos.esa.int/web/soho/command-line",
    ),
    (
        "lasco_direct.html",
        "https://lasco-www.nrl.navy.mil/index.php?p=get_data",
    ),
];

fn href_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r#"href="([^"]+)""#).unwrap())
}

fn soho_directory_entries(url: &str) -> Result<Vec<String>, FetchError> {
    let html = download_to_string(url)?;
    let mut entries = Vec::new();
    for capture in href_regex().captures_iter(&html) {
        let Some(href) = capture.get(1).map(|value| value.as_str()) else {
            continue;
        };
        if href.starts_with('/')
            || href.starts_with('?')
            || href == "Parent Directory"
            || href == "../"
        {
            continue;
        }
        entries.push(href.to_string());
    }
    entries.sort();
    entries.dedup();
    Ok(entries)
}

fn stage_soho_metadata(meta_dir: &Path, skip_existing: bool) {
    if let Err(err) = std::fs::create_dir_all(meta_dir) {
        log::warn!(
            "failed to create SOHO metadata dir {}: {}",
            meta_dir.display(),
            err
        );
        return;
    }
    for (file_name, url) in SOHO_METADATA_URLS {
        let output = meta_dir.join(file_name);
        if skip_existing && output.exists() {
            continue;
        }
        if let Err(err) = download_to_file(url, &output) {
            log::warn!("failed to download SOHO metadata {}: {}", url, err);
        }
    }
}

fn dir_has_suffix(root: &Path, suffix: &str) -> bool {
    std::fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let Ok(file_type) = entry.file_type() else {
                return false;
            };
            if file_type.is_file() {
                return entry.file_name().to_string_lossy().ends_with(suffix);
            }
            if file_type.is_dir() {
                return dir_has_suffix(&entry.path(), suffix);
            }
            false
        })
}

/// Mission-long SOHO CELIAS Proton Monitor TXT bundle provider.
pub struct SohoCeliasBundleProvider;

impl DatasetProvider for SohoCeliasBundleProvider {
    fn name(&self) -> &str {
        "SOHO CELIAS PM 5min Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("soho").join("celias");
        std::fs::create_dir_all(&root)?;
        stage_soho_metadata(&root.join("metadata"), config.skip_existing);
        let output = root.join("CELIAS_Proton_Monitor_5min.tar.gz");
        if !config.skip_existing || !output.exists() {
            download_to_file(SOHO_CELIAS_PM5_BUNDLE_URL, &output)?;
        }
        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("soho")
            .join("celias")
            .join("CELIAS_Proton_Monitor_5min.tar.gz")
            .exists()
    }
}

/// Official SOHO CELIAS PM 5-minute daily CDF staging provider.
pub struct SohoCeliasPm5MinProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for SohoCeliasPm5MinProvider {
    fn default() -> Self {
        Self {
            year_start: 2020,
            year_end: 2023,
        }
    }
}

impl DatasetProvider for SohoCeliasPm5MinProvider {
    fn name(&self) -> &str {
        "SOHO CELIAS PM 5min CDF"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("soho").join("celias_pm_5min");
        std::fs::create_dir_all(&root)?;
        stage_soho_metadata(&root.join("metadata"), config.skip_existing);

        let readme = root.join("00readme.txt");
        if !config.skip_existing || !readme.exists() {
            let readme_url = format!("{SOHO_CELIAS_PM5_ROOT}00readme.txt");
            if let Err(err) = download_to_file(&readme_url, &readme) {
                log::warn!("failed to download SOHO CELIAS readme: {}", err);
            }
        }

        for year in self.year_start..=self.year_end {
            let year_url = format!("{SOHO_CELIAS_PM5_ROOT}{year}/");
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            let entries = match soho_directory_entries(&year_url) {
                Ok(entries) => entries,
                Err(err) => {
                    log::warn!("failed to list SOHO CELIAS {}: {}", year, err);
                    continue;
                }
            };
            for entry in entries {
                if !entry.ends_with(".cdf") {
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

        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        dir_has_suffix(
            &config.output_dir.join("soho").join("celias_pm_5min"),
            ".cdf",
        )
    }
}

/// Bounded SOHO LASCO level-0.5 day sample provider.
pub struct SohoLascoDaySampleProvider {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub max_files_per_camera: usize,
}

impl Default for SohoLascoDaySampleProvider {
    fn default() -> Self {
        Self {
            year: 2024,
            month: 1,
            day: 1,
            max_files_per_camera: 2,
        }
    }
}

impl DatasetProvider for SohoLascoDaySampleProvider {
    fn name(&self) -> &str {
        "SOHO LASCO L0.5 Day Sample"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let yymmdd = format!("{:02}{:02}{:02}", self.year % 100, self.month, self.day);
        let root = config
            .output_dir
            .join("soho")
            .join("lasco")
            .join("level05")
            .join(&yymmdd);
        std::fs::create_dir_all(&root)?;
        stage_soho_metadata(&root.join("metadata"), config.skip_existing);

        let day_url = format!("{SOHO_LASCO_LEVEL05_ROOT}{yymmdd}/");
        let day_index = root.join("index.html");
        if !config.skip_existing || !day_index.exists() {
            let _ = download_to_file(&day_url, &day_index);
        }

        for camera in ["c2", "c3"] {
            let cam_url = format!("{day_url}{camera}/");
            let cam_dir = root.join(camera);
            std::fs::create_dir_all(&cam_dir)?;
            let cam_index = cam_dir.join("index.html");
            if !config.skip_existing || !cam_index.exists() {
                let _ = download_to_file(&cam_url, &cam_index);
            }
            let img_hdr = cam_dir.join("img_hdr.txt");
            if !config.skip_existing || !img_hdr.exists() {
                let _ = download_to_file(&format!("{cam_url}img_hdr.txt"), &img_hdr);
            }
            let entries = match soho_directory_entries(&cam_url) {
                Ok(entries) => entries,
                Err(err) => {
                    log::warn!(
                        "failed to list LASCO {} day sample {}: {}",
                        camera,
                        yymmdd,
                        err
                    );
                    continue;
                }
            };
            for entry in entries
                .into_iter()
                .filter(|entry| entry.ends_with(".fts"))
                .take(self.max_files_per_camera)
            {
                let output = cam_dir.join(&entry);
                if config.skip_existing && output.exists() {
                    continue;
                }
                let url = format!("{cam_url}{entry}");
                match download_to_file(&url, &output) {
                    Ok(_) => log::info!("saved {}", output.display()),
                    Err(err) => log::warn!("failed to download {}: {}", url, err),
                }
            }
        }

        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        dir_has_suffix(
            &config.output_dir.join("soho").join("lasco").join("level05"),
            ".fts",
        )
    }
}
