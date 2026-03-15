//! JWST public-observation metadata via the official MAST API.
//!
//! Official sources:
//!   <https://mast.stsci.edu/api/v0/>
//!   <https://mast.stsci.edu/api/v0/_services.html>
//!
//! This bounded Rust lane stages public JWST observation metadata, not raw
//! calibrated products. It is intended to provide a governed, fetchable public
//! catalog surface that can later be expanded to instrument- or product-level
//! lanes.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, validate_not_html};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::{Path, PathBuf}, time::Duration};

const MAST_API_ROOT: &str = "https://mast.stsci.edu/api/v0/invoke";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwstPublicObservation {
    pub obsid: String,
    pub obs_collection: String,
    pub proposal_id: String,
    pub target_name: String,
    pub s_ra: f64,
    pub s_dec: f64,
    pub t_obs_release: String,
    pub filters: String,
    pub instrument_name: String,
    pub dataproduct_type: String,
    pub calib_level: String,
}

fn json_string(value: &Value, key: &str) -> String {
    match value.get(key) {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|entry| entry.as_str().map(ToOwned::to_owned))
            .collect::<Vec<_>>()
            .join("|"),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

fn json_f64(value: &Value, key: &str) -> f64 {
    value.get(key).and_then(Value::as_f64).unwrap_or(f64::NAN)
}

pub fn parse_jwst_public_metadata_json(
    content: &str,
) -> Result<Vec<JwstPublicObservation>, FetchError> {
    let parsed: Value = serde_json::from_str(content)
        .map_err(|err| FetchError::Validation(format!("invalid JWST MAST JSON: {err}")))?;
    let data = parsed
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| FetchError::Validation("JWST MAST response missing data array".to_string()))?;
    let rows = data
        .iter()
        .map(|row| JwstPublicObservation {
            obsid: json_string(row, "obsid"),
            obs_collection: json_string(row, "obs_collection"),
            proposal_id: json_string(row, "proposal_id"),
            target_name: json_string(row, "target_name"),
            s_ra: json_f64(row, "s_ra"),
            s_dec: json_f64(row, "s_dec"),
            t_obs_release: json_string(row, "t_obs_release"),
            filters: json_string(row, "filters"),
            instrument_name: json_string(row, "instrument_name"),
            dataproduct_type: json_string(row, "dataproduct_type"),
            calib_level: json_string(row, "calib_level"),
        })
        .filter(|row| !row.obsid.is_empty())
        .collect::<Vec<_>>();
    Ok(rows)
}

fn fetch_mast_page(page: usize, page_size: usize) -> Result<String, FetchError> {
    let request = json!({
        "service": "Mast.Caom.Filtered",
        "format": "json",
        "pagesize": page_size,
        "page": page,
        "params": {
            "columns": "obsid,obs_collection,proposal_id,target_name,s_ra,s_dec,t_obs_release,filters,instrument_name,dataproduct_type,calib_level",
            "filters": [
                { "paramName": "obs_collection", "values": ["JWST"] },
                { "paramName": "dataRights", "values": ["PUBLIC"] }
            ]
        }
    });
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|source| FetchError::HttpError {
            url: MAST_API_ROOT.to_string(),
            source: Box::new(source),
        })?;
    let response = client
        .post(MAST_API_ROOT)
        .form(&[("request", request.to_string())])
        .send()
        .map_err(|source| FetchError::HttpError {
            url: MAST_API_ROOT.to_string(),
            source: Box::new(source),
        })?;
    let status = response.status();
    if !status.is_success() {
        return Err(FetchError::HttpStatus {
            url: MAST_API_ROOT.to_string(),
            status: status.as_u16(),
        });
    }
    let body = response.text().map_err(|source| FetchError::HttpError {
        url: MAST_API_ROOT.to_string(),
        source: Box::new(source),
    })?;
    validate_not_html(body.as_bytes())?;
    Ok(body)
}

pub fn parse_jwst_public_metadata_csv(
    path: &Path,
) -> Result<Vec<JwstPublicObservation>, FetchError> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|err| FetchError::Validation(format!("CSV read error: {err}")))?;
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        rows.push(row.map_err(|err| {
            FetchError::Validation(format!("JWST metadata CSV parse error: {err}"))
        })?);
    }
    Ok(rows)
}

pub struct JwstPublicMetadataProvider {
    pub page_size: usize,
    pub max_pages: usize,
}

impl Default for JwstPublicMetadataProvider {
    fn default() -> Self {
        Self {
            page_size: 5000,
            max_pages: 8,
        }
    }
}

impl DatasetProvider for JwstPublicMetadataProvider {
    fn name(&self) -> &str {
        "JWST Public Observation Metadata"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("jwst_public_observations.csv");
        if config.skip_existing && output.exists() {
            return Ok(output);
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut writer = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&output)
            .map_err(|err| FetchError::Validation(format!("create CSV {}: {err}", output.display())))?;
        let mut total_rows = 0usize;
        for page in 1..=self.max_pages {
            let body = fetch_mast_page(page, self.page_size)?;
            let rows = parse_jwst_public_metadata_json(&body)?;
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
                "JWST public metadata fetch returned zero rows".to_string(),
            ));
        }
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("jwst_public_observations.csv").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_parse_jwst_public_metadata_json() {
        let content = r#"{
            "status": "COMPLETE",
            "data": [
                {
                    "obsid": 12345,
                    "obs_collection": "JWST",
                    "proposal_id": 2731,
                    "target_name": "NGC 3132",
                    "s_ra": 154.0,
                    "s_dec": -40.0,
                    "t_obs_release": "2024-01-01T00:00:00",
                    "filters": ["F200W", "CLEAR"],
                    "instrument_name": "NIRCAM/IMAGE",
                    "dataproduct_type": "image",
                    "calib_level": 3
                }
            ]
        }"#;
        let rows = parse_jwst_public_metadata_json(content).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].obsid, "12345");
        assert_eq!(rows[0].filters, "F200W|CLEAR");
        assert_eq!(rows[0].instrument_name, "NIRCAM/IMAGE");
    }

    #[test]
    fn test_parse_jwst_public_metadata_csv() {
        let mut temp = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            temp,
            "obsid,obs_collection,proposal_id,target_name,s_ra,s_dec,t_obs_release,filters,instrument_name,dataproduct_type,calib_level"
        )
        .unwrap();
        writeln!(
            temp,
            "12345,JWST,2731,NGC 3132,154.0,-40.0,2024-01-01T00:00:00,F200W|CLEAR,NIRCAM/IMAGE,image,3"
        )
        .unwrap();
        let rows = parse_jwst_public_metadata_csv(temp.path()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].proposal_id, "2731");
    }
}
