//! Landsat Collection 2 metadata sample via USGS STAC.
//!
//! The Landsat image archive is very large, so this provider fetches a stable
//! metadata item JSON that can anchor reproducible downstream pipelines.
//!
//! Source: https://landsatlook.usgs.gov/stac-server

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

const LANDSAT_URLS: &[&str] = &[
    "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/items/LC09_L2SP_009024_20211205_20230505_02_T1_SR",
    "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/items/LC08_L2SP_044034_20210508_20210517_02_T1_SR",
];

/// Basic shape check for Landsat STAC item JSON.
pub fn looks_like_landsat_stac_json(path: &Path) -> Result<bool, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    Ok(content.contains("\"type\"")
        && content.contains("\"Feature\"")
        && content.contains("\"assets\"")
        && content.contains("landsat"))
}

/// Landsat provider.
pub struct LandsatStacProvider;

impl DatasetProvider for LandsatStacProvider {
    fn name(&self) -> &str {
        "Landsat C2 L2 STAC Metadata"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("landsat_c2l2_sr_sample.json");
        download_with_fallbacks(self.name(), LANDSAT_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("landsat_c2l2_sr_sample.json")
            .exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_landsat_sample_if_available() {
        let path = Path::new("data/external/landsat_c2l2_sr_sample.json");
        if !path.exists() {
            eprintln!("Skipping: Landsat sample not available");
            return;
        }
        let ok = looks_like_landsat_stac_json(path).expect("failed to parse Landsat metadata");
        assert!(ok, "Landsat metadata should look like STAC JSON");
    }
}
