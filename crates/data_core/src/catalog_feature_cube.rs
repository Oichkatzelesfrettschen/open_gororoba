//! Shared generic catalog feature-cube types for non-heliosphere datasets.
//!
//! This mirrors the heliosphere feature-cube pattern at a higher level:
//! support metadata stays explicit on each row, while dataset-specific numeric
//! channels are declared in the manifest.

use serde::{Deserialize, Serialize};

/// One declared numeric feature channel in a generic catalog cube.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFeatureChannel {
    pub name: String,
    pub description: String,
    pub unit: Option<String>,
    pub role: String,
    #[serde(default)]
    pub dictionary: Vec<String>,
}

/// One row in a generic catalog feature cube.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFeatureRow {
    pub cube_name: String,
    pub dataset: String,
    pub record_id: String,
    pub modality: String,
    pub ra_deg: Option<f64>,
    pub dec_deg: Option<f64>,
    pub time_utc: Option<String>,
    pub redshift: Option<f64>,
    pub distance_proxy: Option<f64>,
    pub program_id: Option<String>,
    pub instrument: Option<String>,
    pub features: Vec<f64>,
}

/// Manifest for a generic catalog feature cube.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFeatureCubeManifest {
    pub cube_name: String,
    pub generated_at_utc: String,
    pub row_count: usize,
    pub dataset_names: Vec<String>,
    pub source_paths: Vec<String>,
    pub notes: Vec<String>,
    pub channels: Vec<CatalogFeatureChannel>,
}

/// A generic catalog feature cube.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFeatureCube {
    pub manifest: CatalogFeatureCubeManifest,
    pub rows: Vec<CatalogFeatureRow>,
}

/// Build a stable sorted dictionary for a categorical metadata field.
pub fn stable_dictionary<I>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut dictionary = values
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>();
    dictionary.sort();
    dictionary.dedup();
    dictionary
}

/// Encode a categorical value against a stable dictionary.
pub fn encode_dictionary_value(dictionary: &[String], value: &str) -> f64 {
    dictionary
        .iter()
        .position(|entry| entry == value)
        .map(|index| index as f64)
        .unwrap_or(-1.0)
}

/// Count pipe-separated tokens such as MAST filter lists.
pub fn pipe_count(value: &str) -> f64 {
    let count = value
        .split('|')
        .filter(|token| !token.trim().is_empty())
        .count();
    count as f64
}

