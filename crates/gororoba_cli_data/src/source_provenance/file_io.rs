//! File-loading helpers used by the source-provenance pipeline.
//!
//! Functions:
//!   * `load_toml_value`  -- read a TOML file into `toml::Value`
//!   * `read_text_lossy`  -- read bytes and decode UTF-8 with replacement
//!   * `read_tsv_rows`    -- read a tab-separated file into row maps
//!   * `derive_status`    -- compute a normalized status from a TSV
//!     row's status / result / http_code / is_pdf fields
//!
//! All items `pub(super)`. Pure file-I/O / data-shaping helpers with
//! no dependencies on other source_provenance submodules.

use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result};
use csv::ReaderBuilder;
use toml::Value;

pub(super) fn load_toml_value(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

pub(super) fn read_text_lossy(path: &Path) -> Result<String> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

pub(super) fn read_tsv_rows(path: &Path) -> Result<Vec<HashMap<String, String>>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .with_context(|| format!("open TSV {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("read TSV headers {}", path.display()))?
        .clone();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.with_context(|| format!("read TSV record {}", path.display()))?;
        let mut row = HashMap::new();
        for (header, field) in headers.iter().zip(record.iter()) {
            row.insert(header.to_string(), field.trim().to_string());
        }
        rows.push(row);
    }
    Ok(rows)
}

pub(super) fn derive_status(row: &HashMap<String, String>) -> String {
    let status = row.get("status").cloned().unwrap_or_default();
    if !status.is_empty() {
        return status;
    }
    let result = row.get("result").cloned().unwrap_or_default();
    if !result.is_empty() {
        return result;
    }
    let http_code = row.get("http_code").cloned().unwrap_or_default();
    let is_pdf_raw = row
        .get("is_pdf")
        .cloned()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let is_pdf = matches!(is_pdf_raw.as_str(), "yes" | "true" | "1");
    if http_code.starts_with('2') && is_pdf {
        return "pdf_ok".to_string();
    }
    if http_code.starts_with('2') {
        return "ok_nonpdf".to_string();
    }
    if !http_code.is_empty() {
        return format!("http_{http_code}");
    }
    "unknown".to_string()
}
