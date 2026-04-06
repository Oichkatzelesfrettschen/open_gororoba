//! Fetch/provider support for the JARVIS catalog.

use super::jarvis::{FIGSHARE_ARTICLE_ID, FigshareFile, extract_json_from_zip};
use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file};
use std::path::PathBuf;

/// List files in a Figshare article via the public API.
pub fn list_figshare_files(article_id: u64) -> Result<Vec<FigshareFile>, FetchError> {
    let url = format!("https://api.figshare.com/v2/articles/{article_id}");
    let body = crate::fetcher::download_to_string(&url)?;

    let parsed: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| FetchError::Validation(format!("JSON parse error: {e}")))?;

    let files = parsed
        .get("files")
        .and_then(|f| f.as_array())
        .ok_or_else(|| FetchError::Validation("No 'files' array in Figshare response".into()))?;

    let mut result = Vec::new();
    for entry in files {
        let id = entry.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
        let name = entry
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let size = entry.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
        let download_url = entry
            .get("download_url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if !download_url.is_empty() {
            result.push(FigshareFile {
                id,
                name,
                size,
                download_url,
            });
        }
    }

    Ok(result)
}

/// Download the JARVIS-DFT JSON dataset to the configured output directory.
pub fn fetch_jarvis_json(config: &FetchConfig) -> Result<PathBuf, FetchError> {
    let dest = config.output_dir.join("jarvis_dft_3d.json");
    if config.skip_existing && dest.exists() {
        return Ok(dest);
    }

    let files = list_figshare_files(FIGSHARE_ARTICLE_ID)?;

    let mut json_files: Vec<&FigshareFile> =
        files.iter().filter(|f| f.name.ends_with(".json")).collect();
    if json_files.is_empty() {
        json_files = files
            .iter()
            .filter(|f| f.name.contains("json") || f.name.ends_with(".json"))
            .collect();
    }
    json_files.sort_by_key(|f| f.size);

    let target = json_files
        .first()
        .ok_or_else(|| FetchError::Validation("No JSON file in JARVIS Figshare article".into()))?;

    if target.name.ends_with(".zip") {
        let zip_dest = config.output_dir.join(&target.name);
        download_to_file(&target.download_url, &zip_dest)?;
        extract_json_from_zip(&zip_dest, &dest)?;
    } else {
        download_to_file(&target.download_url, &dest)?;
    }

    Ok(dest)
}

/// JARVIS-DFT dataset provider for the unified fetch pipeline.
pub struct JarvisProvider;

impl DatasetProvider for JarvisProvider {
    fn name(&self) -> &str {
        "JARVIS-DFT 3D"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_jarvis_json(config)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("jarvis_dft_3d.json").exists()
    }
}
