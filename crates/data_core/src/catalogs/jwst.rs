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

use crate::{
    catalogs::mast_public_metadata::{json_f64, json_string},
    fetcher::FetchError,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

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

#[cfg(feature = "fetch")]
pub use super::jwst_fetch::JwstPublicMetadataProvider;

pub fn parse_jwst_public_metadata_json(
    content: &str,
) -> Result<Vec<JwstPublicObservation>, FetchError> {
    let parsed: Value = serde_json::from_str(content)
        .map_err(|err| FetchError::Validation(format!("invalid JWST MAST JSON: {err}")))?;
    let data = parsed
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            FetchError::Validation("JWST MAST response missing data array".to_string())
        })?;
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
