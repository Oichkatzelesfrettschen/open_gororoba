//! Fetch implementation for voyager_pws. See voyager_pws.rs for record types and parsers.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use std::path::PathBuf;

const VOYAGER1_PWS_HAPI_DATASET: &str = "VG1_PWS_LR";
const VOYAGER2_PWS_HAPI_DATASET: &str = "VG2_PWS_LR";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoyagerPwsSpacecraft {
    V1,
    V2,
}

impl VoyagerPwsSpacecraft {
    fn dataset_id(self) -> &'static str {
        match self {
            Self::V1 => VOYAGER1_PWS_HAPI_DATASET,
            Self::V2 => VOYAGER2_PWS_HAPI_DATASET,
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }
}

pub struct VoyagerPwsProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for VoyagerPwsProvider {
    fn default() -> Self {
        Self {
            year_start: 2016,
            year_end: 2016,
        }
    }
}

impl DatasetProvider for VoyagerPwsProvider {
    fn name(&self) -> &str {
        "Voyager PWS Low Rate"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("voyager").join("pws");
        std::fs::create_dir_all(&root)?;

        for spacecraft in [VoyagerPwsSpacecraft::V1, VoyagerPwsSpacecraft::V2] {
            let dir = root.join(spacecraft.slug());
            std::fs::create_dir_all(&dir)?;
            for year in self.year_start..=self.year_end {
                let output = dir.join(format!("{}_pws_lr_{}.csv", spacecraft.slug(), year));
                if config.skip_existing && output.exists() {
                    continue;
                }
                match download_hapi_csv(
                    spacecraft.dataset_id(),
                    &format!("{year}-01-01T00:00:00Z"),
                    &format!("{}-01-01T00:00:00Z", year + 1),
                    Some(&["Time", "electric_field"]),
                ) {
                    Ok(body) => {
                        std::fs::write(&output, body)?;
                        log::info!("saved {}", output.display());
                    }
                    Err(err) => {
                        log::warn!(
                            "failed to download {} {} via HAPI: {}",
                            spacecraft.dataset_id(),
                            year,
                            err
                        );
                    }
                }
            }
        }

        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let root = config.output_dir.join("voyager").join("pws");
        std::fs::read_dir(&root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let path = entry.path();
                if path.is_file() {
                    return path.extension().and_then(|value| value.to_str()) == Some("csv");
                }
                std::fs::read_dir(path)
                    .ok()
                    .into_iter()
                    .flatten()
                    .filter_map(|child| child.ok())
                    .any(|child| {
                        child.path().extension().and_then(|value| value.to_str()) == Some("csv")
                    })
            })
    }
}
