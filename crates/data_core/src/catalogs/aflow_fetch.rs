//! Fetch-side AFLOW provider split from the parser/model module.

use crate::{
    catalogs::aflow::aflux_url,
    fetcher::{DatasetProvider, FetchConfig, FetchError},
};
use std::path::PathBuf;

/// Records per AFLUX API page. AFLUX default maximum is 64 but we request
/// larger pages to reduce round-trips over the full database.
const PER_PAGE: usize = 500;

/// Download one page from the AFLUX API.
fn fetch_aflow_page(page: usize) -> Result<String, FetchError> {
    let url = aflux_url(page);
    crate::fetcher::download_to_string(&url)
}

/// Download the full AFLOW dataset (paginated) and save as a single JSON array.
///
/// Paginates through the entire AFLUX result set until an empty page is
/// returned, collecting all records. Progress is printed to stderr.
pub fn fetch_aflow_dataset(config: &FetchConfig) -> Result<PathBuf, FetchError> {
    let dest = config.output_dir.join("aflow_materials.json");
    if config.skip_existing && dest.exists() {
        return Ok(dest);
    }

    let mut all_records: Vec<serde_json::Value> = Vec::new();
    let mut page = 0;

    loop {
        log::debug!("AFLOW page {page} ...");
        let body = fetch_aflow_page(page)?;
        let page_records: Vec<serde_json::Value> = serde_json::from_str(&body).map_err(|e| {
            FetchError::Validation(format!("AFLOW page {page} JSON parse error: {e}"))
        })?;

        if page_records.is_empty() {
            break;
        }

        let n = page_records.len();
        all_records.extend(page_records);
        log::debug!("+{n} records (total: {})", all_records.len());

        if n < PER_PAGE {
            break;
        }
        page += 1;
    }

    log::info!("AFLOW total: {} records", all_records.len());

    let json_out = serde_json::to_string(&all_records)
        .map_err(|e| FetchError::Validation(format!("JSON serialize error: {e}")))?;

    std::fs::create_dir_all(&config.output_dir)?;
    std::fs::write(&dest, json_out)?;

    Ok(dest)
}

/// AFLOW materials database provider for the unified fetch pipeline.
pub struct AflowProvider;

impl DatasetProvider for AflowProvider {
    fn name(&self) -> &str {
        "AFLOW Materials Database"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_aflow_dataset(config)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("aflow_materials.json").exists()
    }
}
