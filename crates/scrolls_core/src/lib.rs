use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use thiserror::Error;

const UPDATED_STAMP: &str = "2026-02-09";

#[derive(Debug, Error)]
pub enum ScrollError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV parse error: {0}")]
    Csv(#[from] csv::Error),
    #[error("TOML serialization error: {0}")]
    TomlSer(#[from] toml::ser::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollDataset {
    pub dataset: ScrollDatasetMeta,
    pub column: Vec<ScrollColumnProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollDatasetMeta {
    pub id: String,
    pub slug: String,
    pub source_csv: String,
    pub source_sha256: String,
    pub source_size_bytes: usize,
    pub has_header: bool,
    pub delimiter: String,
    pub quotechar: String,
    pub row_count: usize,
    pub column_count: usize,
    pub header_value_sha256: String,
    pub row_value_sha256: String,
    pub migrated_on: String,
    pub migrated_by: String,
    pub dataset_class: String,
    pub corpus_label: String,
    pub header: Vec<String>,
    pub original_header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollColumnProfile {
    pub index: usize,
    pub name: String,
    pub original_name: String,
    pub inferred_type: String,
    pub non_empty_count: usize,
    pub empty_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollIndexEntry {
    pub id: String,
    pub slug: String,
    pub source_csv: String,
    pub source_sha256: String,
    pub source_size_bytes: usize,
    pub canonical_toml: String,
    pub row_count: usize,
    pub column_count: usize,
    pub has_header: bool,
    pub delimiter: String,
    pub quotechar: String,
    pub header_value_sha256: String,
    pub row_value_sha256: String,
    pub dataset_class: String,
}

#[derive(Debug, Clone)]
pub struct ConversionOutput {
    pub dataset: ScrollDataset,
    pub index_entry: ScrollIndexEntry,
    pub rendered_dataset_toml: String,
}

#[derive(Debug, Clone)]
pub struct ConvertSpec<'a> {
    pub dataset_id: &'a str,
    pub slug: &'a str,
    pub source_csv: &'a str,
    pub canonical_toml: &'a str,
    pub dataset_class: &'a str,
    pub corpus_label: &'a str,
    pub migrated_by: &'a str,
}

#[derive(Debug, Clone)]
struct ParsedCsv {
    has_header: bool,
    delimiter: char,
    quotechar: char,
    header: Vec<String>,
    original_header: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn sha_text_ascii(blob: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(blob.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn escape_json_ascii(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ if ch <= '\u{1f}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            _ if ch.is_ascii() => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                let encoded = ch.encode_utf16(&mut units);
                for unit in encoded {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
        }
    }
    out
}

fn normalize_ascii_text(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            let mut units = [0u16; 2];
            let encoded = ch.encode_utf16(&mut units);
            for unit in encoded {
                out.push_str(&format!("\\u{:04X}", unit));
            }
        }
    }
    out
}

fn json_ascii_header(header: &[String]) -> String {
    let mut out = String::from("[");
    for (idx, value) in header.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&escape_json_ascii(value));
        out.push('"');
    }
    out.push(']');
    out
}

fn json_ascii_rows(rows: &[Vec<String>]) -> String {
    let mut out = String::from("[");
    for (row_idx, row) in rows.iter().enumerate() {
        if row_idx > 0 {
            out.push(',');
        }
        out.push('[');
        for (col_idx, value) in row.iter().enumerate() {
            if col_idx > 0 {
                out.push(',');
            }
            out.push('"');
            out.push_str(&escape_json_ascii(value));
            out.push('"');
        }
        out.push(']');
    }
    out.push(']');
    out
}

fn sanitize_header_token(token: &str, index: usize) -> String {
    let mut out = String::new();
    let lower = token.trim().to_ascii_lowercase();
    let mut prev_underscore = false;
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    out = out.trim_matches('_').to_string();
    if out.is_empty() {
        out = format!("col_{}", index + 1);
    }
    if out
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        out = format!("col_{}", out);
    }
    out
}

fn make_unique(tokens: &[String]) -> Vec<String> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::with_capacity(tokens.len());
    for token in tokens {
        let count = seen.entry(token.clone()).or_insert(0);
        *count += 1;
        if *count == 1 {
            out.push(token.clone());
        } else {
            out.push(format!("{}_{}", token, *count));
        }
    }
    out
}

fn try_parse_finite_float(value: &str) -> Option<f64> {
    let parsed = value.parse::<f64>().ok()?;
    if parsed.is_finite() {
        Some(parsed)
    } else {
        None
    }
}

fn infer_type(values: &[String]) -> String {
    let non_empty: Vec<String> = values
        .iter()
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .collect();
    if non_empty.is_empty() {
        return "empty".to_string();
    }
    let lowered: Vec<String> = non_empty.iter().map(|v| v.to_ascii_lowercase()).collect();
    if lowered
        .iter()
        .all(|v| matches!(v.as_str(), "true" | "false" | "yes" | "no"))
    {
        return "bool".to_string();
    }
    if non_empty.iter().all(|v| {
        let trimmed = v.trim();
        let tail = trimmed
            .strip_prefix('+')
            .or_else(|| trimmed.strip_prefix('-'))
            .unwrap_or(trimmed);
        !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit())
    }) {
        return "int".to_string();
    }
    if non_empty
        .iter()
        .all(|v| try_parse_finite_float(v.as_str()).is_some())
    {
        return "float".to_string();
    }
    "string".to_string()
}

fn detect_delimiter(sample: &str) -> u8 {
    let candidates = [b',', b';', b'\t', b'|'];
    let mut best = (b',', 0usize);
    for candidate in candidates {
        let mut score = 0usize;
        for line in sample.lines().take(64) {
            score += line.as_bytes().iter().filter(|&&b| b == candidate).count();
        }
        if score > best.1 {
            best = (candidate, score);
        }
    }
    best.0
}

fn looks_like_header(first: &[String], second: &[String]) -> bool {
    if first.is_empty() {
        return false;
    }
    let first_non_numeric = first
        .iter()
        .filter(|v| try_parse_finite_float(v.trim()).is_none())
        .count();
    let second_non_numeric = second
        .iter()
        .filter(|v| try_parse_finite_float(v.trim()).is_none())
        .count();
    let distinct = first.iter().collect::<BTreeSet<_>>().len() == first.len();
    first_non_numeric >= second_non_numeric && distinct
}

fn parse_csv(path: &Path) -> Result<ParsedCsv, ScrollError> {
    let raw = fs::read(path)?;
    let sample = String::from_utf8_lossy(&raw);
    let delimiter = detect_delimiter(&sample);
    let quotechar = b'"';
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(delimiter)
        .quote(quotechar)
        .flexible(true)
        .from_reader(raw.as_slice());

    let mut parsed: Vec<Vec<String>> = Vec::new();
    for record in reader.records() {
        let row = record?;
        let mut values: Vec<String> = row.iter().map(str::to_string).collect();
        if parsed.is_empty() && !values.is_empty() {
            values[0] = values[0].trim_start_matches('\u{feff}').to_string();
        }
        parsed.push(
            values
                .into_iter()
                .map(|value| normalize_ascii_text(&value))
                .collect(),
        );
    }

    let has_header = if parsed.len() >= 2 {
        looks_like_header(&parsed[0], &parsed[1])
    } else {
        true
    };

    let (original_header, data_rows): (Vec<String>, Vec<Vec<String>>) =
        if has_header && !parsed.is_empty() {
            (parsed[0].clone(), parsed.into_iter().skip(1).collect())
        } else {
            (Vec::new(), parsed)
        };

    let mut max_cols = original_header.len();
    for row in &data_rows {
        if row.len() > max_cols {
            max_cols = row.len();
        }
    }

    let header_tokens: Vec<String> = if max_cols == 0 {
        Vec::new()
    } else if has_header {
        let mut padded = original_header.clone();
        while padded.len() < max_cols {
            padded.push(String::new());
        }
        padded
            .iter()
            .enumerate()
            .map(|(idx, token)| sanitize_header_token(token, idx))
            .collect()
    } else {
        (0..max_cols)
            .map(|idx| format!("col_{}", idx + 1))
            .collect()
    };
    let header = make_unique(&header_tokens);

    let mut rows = Vec::with_capacity(data_rows.len());
    for row in data_rows {
        let mut padded = row;
        while padded.len() < max_cols {
            padded.push(String::new());
        }
        padded.truncate(max_cols);
        rows.push(padded);
    }

    Ok(ParsedCsv {
        has_header,
        delimiter: delimiter as char,
        quotechar: quotechar as char,
        header,
        original_header,
        rows,
    })
}

pub fn slugify(name: &str) -> String {
    let stem = name.rsplit_once('.').map_or(name, |(prefix, _)| prefix);
    let lower = stem.to_ascii_lowercase();
    let mut out = String::new();
    let mut prev_underscore = false;
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "dataset".to_string()
    } else {
        out
    }
}

pub fn convert_csv_to_scroll(
    path: &Path,
    spec: &ConvertSpec<'_>,
) -> Result<ConversionOutput, ScrollError> {
    let raw = fs::read(path)?;
    let parsed = parse_csv(path)?;
    let source_sha256 = format!("{:x}", Sha256::digest(&raw));
    let header_value_sha256 = sha_text_ascii(&json_ascii_header(&parsed.header));
    let row_value_sha256 = sha_text_ascii(&json_ascii_rows(&parsed.rows));
    let row_count = parsed.rows.len();
    let column_count = parsed.header.len();

    let mut columns = Vec::with_capacity(column_count);
    for idx in 0..column_count {
        let values: Vec<String> = parsed.rows.iter().map(|row| row[idx].clone()).collect();
        let non_empty_count = values
            .iter()
            .filter(|value| !value.trim().is_empty())
            .count();
        let empty_count = values.len() - non_empty_count;
        let original_name = parsed.original_header.get(idx).cloned().unwrap_or_default();
        columns.push(ScrollColumnProfile {
            index: idx + 1,
            name: parsed.header[idx].clone(),
            original_name,
            inferred_type: infer_type(&values),
            non_empty_count,
            empty_count,
        });
    }

    let dataset = ScrollDataset {
        dataset: ScrollDatasetMeta {
            id: spec.dataset_id.to_string(),
            slug: spec.slug.to_string(),
            source_csv: spec.source_csv.to_string(),
            source_sha256: source_sha256.clone(),
            source_size_bytes: raw.len(),
            has_header: parsed.has_header,
            delimiter: parsed.delimiter.to_string(),
            quotechar: parsed.quotechar.to_string(),
            row_count,
            column_count,
            header_value_sha256: header_value_sha256.clone(),
            row_value_sha256: row_value_sha256.clone(),
            migrated_on: UPDATED_STAMP.to_string(),
            migrated_by: spec.migrated_by.to_string(),
            dataset_class: spec.dataset_class.to_string(),
            corpus_label: spec.corpus_label.to_string(),
            header: parsed.header.clone(),
            original_header: parsed.original_header.clone(),
            rows: parsed.rows.clone(),
        },
        column: columns,
    };

    let rendered_dataset_toml = toml::to_string_pretty(&dataset)?;

    let index_entry = ScrollIndexEntry {
        id: spec.dataset_id.to_string(),
        slug: spec.slug.to_string(),
        source_csv: spec.source_csv.to_string(),
        source_sha256,
        source_size_bytes: raw.len(),
        canonical_toml: spec.canonical_toml.to_string(),
        row_count,
        column_count,
        has_header: parsed.has_header,
        delimiter: parsed.delimiter.to_string(),
        quotechar: parsed.quotechar.to_string(),
        header_value_sha256,
        row_value_sha256,
        dataset_class: spec.dataset_class.to_string(),
    };

    Ok(ConversionOutput {
        dataset,
        index_entry,
        rendered_dataset_toml,
    })
}

fn q(value: &str) -> String {
    serde_json::to_string(value).expect("string serialization should not fail")
}

pub fn render_scroll_index(
    entries: &[ScrollIndexEntry],
    index_table: &str,
    source_descriptor: &str,
    canonical_dir: &str,
    corpus_label: &str,
    generated_by: &str,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "# Canonical index for {} datasets migrated to TOML scrolls.",
        corpus_label
    ));
    lines.push(format!("# Generated by {}", generated_by));
    lines.push(String::new());
    lines.push(format!("[{}]", index_table));
    lines.push(format!("updated = {}", q(UPDATED_STAMP)));
    lines.push("authoritative = true".to_string());
    lines.push(format!("source_descriptor = {}", q(source_descriptor)));
    lines.push(format!("canonical_dir = {}", q(canonical_dir)));
    lines.push(format!("dataset_count = {}", entries.len()));
    lines.push(String::new());
    for entry in entries {
        lines.push("[[dataset]]".to_string());
        lines.push(format!("id = {}", q(&entry.id)));
        lines.push(format!("slug = {}", q(&entry.slug)));
        lines.push(format!("source_csv = {}", q(&entry.source_csv)));
        lines.push(format!("source_sha256 = {}", q(&entry.source_sha256)));
        lines.push(format!("source_size_bytes = {}", entry.source_size_bytes));
        lines.push(format!("canonical_toml = {}", q(&entry.canonical_toml)));
        lines.push(format!("row_count = {}", entry.row_count));
        lines.push(format!("column_count = {}", entry.column_count));
        lines.push(format!(
            "has_header = {}",
            if entry.has_header { "true" } else { "false" }
        ));
        lines.push(format!("delimiter = {}", q(&entry.delimiter)));
        lines.push(format!("quotechar = {}", q(&entry.quotechar)));
        lines.push(format!(
            "header_value_sha256 = {}",
            q(&entry.header_value_sha256)
        ));
        lines.push(format!("row_value_sha256 = {}", q(&entry.row_value_sha256)));
        lines.push(format!("dataset_class = {}", q(&entry.dataset_class)));
        lines.push(String::new());
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn slugify_normalizes_ascii() {
        assert_eq!(slugify("My Data File.csv"), "my_data_file");
        assert_eq!(slugify("____.csv"), "dataset");
        assert_eq!(slugify("128D_Cayley-Dickson.csv"), "128d_cayley_dickson");
    }

    #[test]
    fn conversion_preserves_row_and_header_checksums() {
        let mut temp = NamedTempFile::new().expect("tmp");
        writeln!(temp, "A,B").expect("write");
        writeln!(temp, "1,2").expect("write");
        writeln!(temp, "3,4").expect("write");

        let spec = ConvertSpec {
            dataset_id: "T-0001",
            slug: "sample",
            source_csv: "tmp/sample.csv",
            canonical_toml: "registry/data/test/T-0001_sample.toml",
            dataset_class: "canonical_dataset",
            corpus_label: "test corpus",
            migrated_by: "scrolls_core::tests",
        };
        let converted = convert_csv_to_scroll(temp.path(), &spec).expect("convert");
        assert_eq!(converted.dataset.dataset.row_count, 2);
        assert_eq!(converted.dataset.dataset.column_count, 2);
        assert_eq!(
            converted.dataset.dataset.header,
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(!converted.dataset.dataset.header_value_sha256.is_empty());
        assert!(!converted.dataset.dataset.row_value_sha256.is_empty());
        assert!(converted
            .rendered_dataset_toml
            .contains("dataset_class = \"canonical_dataset\""));
    }
}
