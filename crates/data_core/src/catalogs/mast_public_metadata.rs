//! Shared helpers for MAST public-observation metadata lanes.

use crate::fetcher::{FetchError, validate_not_html};
use reqwest::blocking::Client;
use serde_json::{Value, json};
use std::time::Duration;

#[cfg_attr(not(feature = "fetch"), allow(dead_code))]
const MAST_API_ROOT: &str = "https://mast.stsci.edu/api/v0/invoke";

pub(crate) fn json_string(value: &Value, key: &str) -> String {
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

pub(crate) fn json_f64(value: &Value, key: &str) -> f64 {
    value.get(key).and_then(Value::as_f64).unwrap_or(f64::NAN)
}

#[cfg_attr(not(feature = "fetch"), allow(dead_code))]
pub(crate) fn fetch_mast_public_observation_page(
    obs_collection: &str,
    page: usize,
    page_size: usize,
    timeout_secs: u64,
) -> Result<String, FetchError> {
    let request = json!({
        "service": "Mast.Caom.Filtered",
        "format": "json",
        "pagesize": page_size,
        "page": page,
        "params": {
            "columns": "obsid,obs_collection,proposal_id,target_name,s_ra,s_dec,t_obs_release,filters,instrument_name,dataproduct_type,calib_level",
            "filters": [
                { "paramName": "obs_collection", "values": [obs_collection] },
                { "paramName": "dataRights", "values": ["PUBLIC"] }
            ]
        }
    });
    let client = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
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
