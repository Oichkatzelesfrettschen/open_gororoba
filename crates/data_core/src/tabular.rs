//! Feature-gated Polars tabular helpers for dataset inventory and flat-file ingestion.

use std::{collections::BTreeSet, fs::File, path::Path};

use polars::prelude::{Column, DataFrame, NamedFrom, Series};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TabularError {
    #[error("csv parse failed: {0}")]
    Csv(#[from] csv::Error),
    #[error("io failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("json parse failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("polars failed: {0}")]
    Polars(#[from] polars::error::PolarsError),
    #[error("json root must be an array of flat objects")]
    InvalidJsonShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabularOverview {
    pub rows: usize,
    pub columns: usize,
}

fn string_columns_to_frame(
    names: Vec<String>,
    values: Vec<Vec<String>>,
) -> Result<DataFrame, TabularError> {
    let height = values.first().map_or(0, Vec::len);
    let columns: Vec<Column> = names
        .into_iter()
        .zip(values)
        .map(|(name, column)| Series::new(name.into(), column).into())
        .collect();
    Ok(DataFrame::new(height, columns)?)
}

fn json_cell_to_string(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Bool(flag)) => flag.to_string(),
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Build a tabular provider inventory for the canonical dataset list.
pub fn provider_inventory_frame() -> Result<DataFrame, TabularError> {
    let providers = crate::known_provider_names();
    let provider_names: Vec<String> = providers.iter().map(|name| (*name).to_string()).collect();
    let pillars: Vec<String> = providers
        .iter()
        .map(|name| crate::provider_pillar(name).to_string())
        .collect();
    let claim_counts: Vec<u32> = providers
        .iter()
        .map(|name| crate::claims_for_provider(name).len() as u32)
        .collect();
    let columns = vec![
        Series::new("provider".into(), provider_names).into(),
        Series::new("pillar".into(), pillars).into(),
        Series::new("claim_count".into(), claim_counts).into(),
    ];
    Ok(DataFrame::new(providers.len(), columns)?)
}

/// Read a CSV file into a string-backed Polars DataFrame.
pub fn csv_records_to_frame(path: &Path) -> Result<DataFrame, TabularError> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader
        .headers()?
        .iter()
        .map(|header| header.to_string())
        .collect::<Vec<_>>();
    let mut columns = vec![Vec::<String>::new(); headers.len()];
    for record in reader.records() {
        let record = record?;
        for (column, value) in columns.iter_mut().zip(record.iter()) {
            column.push(value.to_string());
        }
    }
    string_columns_to_frame(headers, columns)
}

/// Read a JSON array of flat objects into a string-backed Polars DataFrame.
pub fn json_records_to_frame(path: &Path) -> Result<DataFrame, TabularError> {
    let root: Value = serde_json::from_reader(File::open(path)?)?;
    let records = root.as_array().ok_or(TabularError::InvalidJsonShape)?;
    let mut keys = BTreeSet::new();
    for record in records {
        let object = record.as_object().ok_or(TabularError::InvalidJsonShape)?;
        keys.extend(object.keys().cloned());
    }
    let names: Vec<String> = keys.into_iter().collect();
    let mut columns = vec![Vec::<String>::with_capacity(records.len()); names.len()];
    for record in records {
        let object = record.as_object().ok_or(TabularError::InvalidJsonShape)?;
        for (index, key) in names.iter().enumerate() {
            columns[index].push(json_cell_to_string(object.get(key)));
        }
    }
    string_columns_to_frame(names, columns)
}

pub fn frame_overview(frame: &DataFrame) -> TabularOverview {
    TabularOverview {
        rows: frame.height(),
        columns: frame.width(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_provider_inventory_frame_matches_dataset_count() {
        let frame = provider_inventory_frame().expect("provider inventory");
        assert_eq!(frame.height(), crate::DATASET_COUNT);
        assert_eq!(frame.width(), 3);
    }

    #[test]
    fn test_csv_records_to_frame_reads_headers_and_rows() {
        let mut temp = NamedTempFile::new().unwrap();
        writeln!(temp, "name,value").unwrap();
        writeln!(temp, "alpha,1").unwrap();
        writeln!(temp, "beta,2").unwrap();
        let frame = csv_records_to_frame(temp.path()).expect("csv frame");
        let overview = frame_overview(&frame);
        assert_eq!(overview.rows, 2);
        assert_eq!(overview.columns, 2);
    }

    #[test]
    fn test_json_records_to_frame_reads_flat_objects() {
        let mut temp = NamedTempFile::new().unwrap();
        writeln!(
            temp,
            "[{{\"name\":\"alpha\",\"value\":1}},{{\"name\":\"beta\",\"value\":2}}]"
        )
        .unwrap();
        let frame = json_records_to_frame(temp.path()).expect("json frame");
        let overview = frame_overview(&frame);
        assert_eq!(overview.rows, 2);
        assert_eq!(overview.columns, 2);
    }
}
