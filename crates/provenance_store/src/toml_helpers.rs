//! Pure TOML/JSON utility helpers for ProvenanceStore ingest and
//! compat-export paths.
//!
//! Covers:
//! - TOML file readers (`load_toml_value`, `load_toml_text`, `load_text`).
//! - Registry-table introspection (`load_registry_table_toml`,
//!   `render_toml_table`).
//! - Compat-export quoting/serialization (`compat_toml_quote`,
//!   `compat_toml_string_array`, `compat_json_string_array`).
//! - Table-walking helpers (`compat_root_table`, `compat_child_table`,
//!   `compat_table_string`, `compat_table_bool`, `compat_table_array`).
//! - Field extractors (`string_field`, `optional_string_field`,
//!   `string_array_field`, `bool_field`, `optional_integer_field`).
//! - Miscellaneous (`trim_trailing_blank_lines`, `host_for_url`,
//!   `join_refs`, `toml_array_to_json_string`).

use std::{fs, path::Path};

use anyhow::{Context, Result};
use toml::Value;

pub(crate) fn load_toml_value(path: &Path) -> Result<Value> {
    let raw = load_toml_text(path)?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

pub(crate) fn load_toml_text(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}

pub(crate) fn load_text(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}

pub(crate) fn load_registry_table_toml(raw: &str, key: &str) -> Result<Option<String>> {
    let value: Value = toml::from_str(raw).with_context(|| format!("parse {key} registry"))?;
    let Some(table) = value.get(key).and_then(Value::as_table) else {
        return Ok(None);
    };
    render_toml_table(table).map(Some)
}

pub(crate) fn render_toml_table(table: &toml::map::Map<String, Value>) -> Result<String> {
    toml::to_string(table).context("serialize TOML table")
}

pub(crate) fn compat_toml_quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

pub(crate) fn compat_json_string_array(raw: &str) -> Result<Vec<String>> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(raw).with_context(|| format!("parse JSON string array from {raw}"))
}

pub(crate) fn compat_toml_string_array(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| compat_toml_quote(value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(crate) fn compat_root_table<'a>(value: &'a Value, key: &str) -> Option<&'a toml::value::Table> {
    value.get(key).and_then(Value::as_table)
}

pub(crate) fn compat_child_table<'a>(
    table: Option<&'a toml::value::Table>,
    key: &str,
) -> Option<&'a toml::value::Table> {
    table
        .and_then(|table| table.get(key))
        .and_then(Value::as_table)
}

pub(crate) fn compat_table_string(
    table: Option<&toml::value::Table>,
    key: &str,
    default: &str,
) -> String {
    table
        .and_then(|table| table.get(key))
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

pub(crate) fn compat_table_bool(
    table: Option<&toml::value::Table>,
    key: &str,
    default: bool,
) -> bool {
    table
        .and_then(|table| table.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

pub(crate) fn compat_table_array(
    table: Option<&toml::value::Table>,
    key: &str,
    default: &[&str],
) -> Vec<String> {
    match table
        .and_then(|table| table.get(key))
        .and_then(Value::as_array)
    {
        Some(values) => values
            .iter()
            .filter_map(Value::as_str)
            .map(ToOwned::to_owned)
            .collect(),
        None => default.iter().map(|value| (*value).to_string()).collect(),
    }
}

pub(crate) fn trim_trailing_blank_lines(lines: &mut Vec<String>) {
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
}

pub(crate) fn string_field(table: &toml::map::Map<String, Value>, key: &str) -> String {
    table
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

pub(crate) fn optional_string_field(
    table: &toml::map::Map<String, Value>,
    key: &str,
) -> Option<String> {
    let value = table.get(key).and_then(Value::as_str).unwrap_or("").trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

pub(crate) fn string_array_field(table: &toml::map::Map<String, Value>, key: &str) -> Vec<String> {
    table
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn bool_field(table: &toml::map::Map<String, Value>, key: &str) -> bool {
    table.get(key).and_then(Value::as_bool).unwrap_or(false)
}

pub(crate) fn optional_integer_field(
    table: &toml::map::Map<String, Value>,
    key: &str,
) -> Option<i64> {
    table.get(key).and_then(Value::as_integer)
}

pub(crate) fn host_for_url(url: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(|host| host.to_string()))
}

pub(crate) fn join_refs(values: &[String]) -> String {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Extract an array field from a toml::Value and serialize it as a JSON string.
/// Works with Value (not just Map) so it can be used in the ingest methods.
pub(crate) fn toml_array_to_json_string(val: &Value, key: &str) -> String {
    val.get(key)
        .and_then(Value::as_array)
        .map(|items| {
            let strs: Vec<&str> = items.iter().filter_map(Value::as_str).collect();
            serde_json::to_string(&strs).unwrap_or_else(|_| "[]".to_string())
        })
        .unwrap_or_else(|| "[]".to_string())
}
