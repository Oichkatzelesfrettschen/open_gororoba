//! Fetch/provider support for HST public metadata.

use super::hst::parse_hst_public_metadata_json;
use crate::{
    catalogs::mast_public_metadata::fetch_mast_public_observation_page,
    fetcher::{DatasetProvider, FetchConfig, FetchError},
};
use std::{fs, path::PathBuf};

pub struct HstPublicMetadataProvider {
    pub page_size: usize,
    pub max_pages: usize,
}

impl Default for HstPublicMetadataProvider {
    fn default() -> Self {
        Self {
            page_size: 1000,
            max_pages: 4,
        }
    }
}

impl DatasetProvider for HstPublicMetadataProvider {
    fn name(&self) -> &str {
        "HST Public Observation Metadata"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("hst_public_observations.csv");
        if config.skip_existing && output.exists() {
            return Ok(output);
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut writer = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&output)
            .map_err(|err| {
                FetchError::Validation(format!("create CSV {}: {err}", output.display()))
            })?;
        let mut total_rows = 0usize;
        for page in 1..=self.max_pages {
            let body = fetch_mast_public_observation_page("HST", page, self.page_size, 120)?;
            let rows = parse_hst_public_metadata_json(&body)?;
            let row_count = rows.len();
            for row in rows {
                writer.serialize(row).map_err(|err| {
                    FetchError::Validation(format!("write CSV {}: {err}", output.display()))
                })?;
            }
            total_rows += row_count;
            if row_count < self.page_size {
                break;
            }
        }
        writer.flush()?;
        if total_rows == 0 {
            return Err(FetchError::Validation(
                "HST public metadata fetch returned zero rows".to_string(),
            ));
        }
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("hst_public_observations.csv")
            .exists()
    }
}
