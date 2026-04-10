//! Fetch/provider support for IMP 8 merged 1-minute data.

use crate::{
    catalogs::imp8::IMP8_MERGED_ROOT,
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file},
};
use std::path::PathBuf;

/// IMP 8 merged 1-minute dataset provider.
pub struct Imp8Provider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for Imp8Provider {
    fn default() -> Self {
        Self {
            year_start: 1976,
            year_end: 1980,
        }
    }
}

impl DatasetProvider for Imp8Provider {
    fn name(&self) -> &str {
        "IMP 8 Merged 1-minute"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("imp8");
        std::fs::create_dir_all(&dir)?;
        for year in self.year_start..=self.year_end {
            for month in 1..=12 {
                let file_name = format!("imp_min_merge{year}{month:02}.asc");
                let output = dir.join(&file_name);
                if config.skip_existing && output.exists() {
                    continue;
                }
                let url = format!("{IMP8_MERGED_ROOT}{file_name}");
                match download_to_file(&url, &output) {
                    Ok(_) => log::info!("saved {}", output.display()),
                    Err(err) => log::warn!("failed to download {}: {}", url, err),
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("imp8");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("imp_min_merge")
            })
    }
}
