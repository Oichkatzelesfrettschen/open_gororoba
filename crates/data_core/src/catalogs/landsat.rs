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

/// Required top-level fields in a STAC Item.
const STAC_REQUIRED_FIELDS: &[&str] = &[
    "\"type\"",
    "\"stac_version\"",
    "\"id\"",
    "\"geometry\"",
    "\"properties\"",
    "\"assets\"",
];

/// Validate that a STAC item JSON contains all required fields.
pub fn validate_stac_schema(path: &Path) -> Result<(), FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    for field in STAC_REQUIRED_FIELDS {
        if !content.contains(field) {
            return Err(FetchError::Validation(format!(
                "STAC item missing required field {}: {}",
                field,
                path.display()
            )));
        }
    }
    Ok(())
}

/// Extract cloud cover percentage from a STAC item JSON.
///
/// Looks for `"eo:cloud_cover"` in the properties block. Returns None
/// if the field is absent.
pub fn extract_cloud_cover(path: &Path) -> Result<Option<f64>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    // Simple JSON field extraction without a full parser
    let key = "\"eo:cloud_cover\"";
    if let Some(pos) = content.find(key) {
        let after_key = &content[pos + key.len()..];
        // Skip optional whitespace and colon
        let after_colon = after_key
            .trim_start()
            .strip_prefix(':')
            .unwrap_or(after_key);
        let value_str = after_colon.trim_start();
        // Extract numeric value until comma, brace, or whitespace
        let end = value_str.find([',', '}', '\n']).unwrap_or(value_str.len());
        let num_str = value_str[..end].trim();
        if let Ok(val) = num_str.parse::<f64>() {
            return Ok(Some(val));
        }
    }
    Ok(None)
}

/// Count asset entries in a STAC item JSON.
pub fn count_stac_assets(path: &Path) -> Result<usize, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    // Count occurrences of "href" within assets block as a proxy
    Ok(content.matches("\"href\"").count())
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
    use std::io::Write;
    use std::path::Path;

    fn write_temp(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    const SYNTHETIC_STAC: &str = r#"{
  "type": "Feature",
  "stac_version": "1.0.0",
  "id": "LC09_L2SP_009024_20211205_20230505_02_T1_SR",
  "geometry": {"type": "Polygon", "coordinates": [[[-70,42],[-70,44],[-68,44],[-68,42],[-70,42]]]},
  "bbox": [-70.0, 42.0, -68.0, 44.0],
  "properties": {
    "datetime": "2021-12-05T15:30:00Z",
    "eo:cloud_cover": 12.5,
    "landsat:scene_id": "LC90090242021339LGN00"
  },
  "assets": {
    "blue": {"href": "https://example.com/B2.TIF", "type": "image/tiff"},
    "green": {"href": "https://example.com/B3.TIF", "type": "image/tiff"},
    "red": {"href": "https://example.com/B4.TIF", "type": "image/tiff"}
  },
  "links": []
}"#;

    #[test]
    fn test_validate_stac_schema_ok() {
        let f = write_temp(SYNTHETIC_STAC);
        assert!(validate_stac_schema(f.path()).is_ok());
    }

    #[test]
    fn test_validate_stac_schema_rejects_missing_field() {
        // Missing "assets"
        let bad = r#"{"type": "Feature", "stac_version": "1.0.0", "id": "test", "geometry": {}, "properties": {}}"#;
        let f = write_temp(bad);
        assert!(validate_stac_schema(f.path()).is_err());
    }

    #[test]
    fn test_extract_cloud_cover() {
        let f = write_temp(SYNTHETIC_STAC);
        let cc = extract_cloud_cover(f.path()).unwrap();
        assert_eq!(cc, Some(12.5));
    }

    #[test]
    fn test_extract_cloud_cover_absent() {
        let json = r#"{"type": "Feature", "properties": {"datetime": "2021-01-01"}}"#;
        let f = write_temp(json);
        let cc = extract_cloud_cover(f.path()).unwrap();
        assert_eq!(cc, None);
    }

    #[test]
    fn test_count_stac_assets() {
        let f = write_temp(SYNTHETIC_STAC);
        let n = count_stac_assets(f.path()).unwrap();
        assert_eq!(n, 3, "Should count 3 asset hrefs");
    }

    #[test]
    fn test_looks_like_landsat_stac_json() {
        let f = write_temp(SYNTHETIC_STAC);
        let ok = looks_like_landsat_stac_json(f.path()).unwrap();
        assert!(ok);
    }

    #[test]
    fn test_landsat_sample_if_available() {
        let path = Path::new("data/external/landsat_c2l2_sr_sample.json");
        if !path.exists() {
            eprintln!("Skipping: Landsat sample not available");
            return;
        }
        let ok = looks_like_landsat_stac_json(path).expect("failed to parse Landsat metadata");
        assert!(ok, "Landsat metadata should look like STAC JSON");
        validate_stac_schema(path).expect("STAC schema validation should pass");
    }
}
