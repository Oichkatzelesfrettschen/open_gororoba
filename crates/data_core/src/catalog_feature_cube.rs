//! Shared generic catalog feature-cube types for non-heliosphere datasets.
//!
//! This mirrors the heliosphere feature-cube pattern at a higher level:
//! support metadata stays explicit on each row, while dataset-specific numeric
//! channels are declared in the manifest.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residualized_features: Option<Vec<f64>>,
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

/// Declares the nuisance terms used to residualize one dataset family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogNuisanceModel {
    pub dataset: String,
    pub nuisance_terms: Vec<String>,
    pub ridge_lambda: f64,
    pub notes: Vec<String>,
}

/// Summarizes how much variance the nuisance model removed from one dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuisanceEffectReport {
    pub dataset: String,
    pub row_count: usize,
    pub nuisance_term_count: usize,
    pub feature_r2: Vec<f64>,
    pub mean_removed_energy: f64,
    pub mean_retained_energy: f64,
}

/// Residualized variant of a generic catalog cube with nuisance metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualizedCatalogFeatureCube {
    pub cube: CatalogFeatureCube,
    pub nuisance_models: Vec<CatalogNuisanceModel>,
    pub nuisance_reports: Vec<NuisanceEffectReport>,
}

/// Parse a generic catalog feature cube from JSON, tolerating `null` in
/// feature arrays that originated from non-finite floating-point values.
pub fn parse_catalog_feature_cube_json(content: &[u8]) -> Result<CatalogFeatureCube, serde_json::Error> {
    let value: Value = serde_json::from_slice(content)?;
    let manifest: CatalogFeatureCubeManifest = serde_json::from_value(value["manifest"].clone())?;
    let rows_value = value
        .get("rows")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let rows = rows_value
        .into_iter()
        .map(parse_catalog_feature_row_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CatalogFeatureCube { manifest, rows })
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

fn parse_catalog_feature_row_value(value: Value) -> Result<CatalogFeatureRow, serde_json::Error> {
    #[derive(Debug, Deserialize)]
    struct JsonCatalogFeatureRow {
        cube_name: String,
        dataset: String,
        record_id: String,
        modality: String,
        ra_deg: Option<f64>,
        dec_deg: Option<f64>,
        time_utc: Option<String>,
        redshift: Option<f64>,
        distance_proxy: Option<f64>,
        program_id: Option<String>,
        instrument: Option<String>,
        features: Vec<Option<f64>>,
        #[serde(default)]
        residualized_features: Option<Vec<Option<f64>>>,
    }

    let row: JsonCatalogFeatureRow = serde_json::from_value(value)?;
    Ok(CatalogFeatureRow {
        cube_name: row.cube_name,
        dataset: row.dataset,
        record_id: row.record_id,
        modality: row.modality,
        ra_deg: row.ra_deg,
        dec_deg: row.dec_deg,
        time_utc: row.time_utc,
        redshift: row.redshift,
        distance_proxy: row.distance_proxy,
        program_id: row.program_id,
        instrument: row.instrument,
        features: row
            .features
            .into_iter()
            .map(|value| value.unwrap_or(f64::NAN))
            .collect(),
        residualized_features: row.residualized_features.map(|values| {
            values
                .into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect()
        }),
    })
}
