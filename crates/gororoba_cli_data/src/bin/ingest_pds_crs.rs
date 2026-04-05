use anyhow::{Context, Result, bail};
use arrow_array::{
    Array, ArrayRef, RecordBatch,
    builder::{Float64Builder, Int64Builder, StringBuilder, TimestampMillisecondBuilder},
};
use arrow_ipc::{reader::FileReader, writer::FileWriter};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::NaiveDateTime;
use clap::Parser;
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Parser, Debug)]
#[command(name = "ingest-pds-crs")]
#[command(
    about = "Ingest fetched PDS/PPI Voyager CRS bundles from a Python manifest into normalized Rust artifacts."
)]
struct Cli {
    /// Path to the Python-generated PDS fetch manifest.
    #[arg(long)]
    manifest: PathBuf,

    /// Output directory for normalized JSON artifacts.
    #[arg(long, default_value = "data/output/heliosphere/pds_ingest")]
    out_dir: PathBuf,

    /// Output format for parsed encounter products.
    #[arg(long, value_enum, default_value_t = ExportFormat::Json)]
    export_format: ExportFormat,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Eq, PartialEq)]
enum ExportFormat {
    Json,
    Arrow,
}

#[derive(Debug, Deserialize)]
struct FetchManifest {
    spacecraft: u8,
    product: String,
    pds_ppi_dataset: Option<String>,
    files: Vec<FetchFileEntry>,
}

#[derive(Debug, Deserialize)]
struct FetchFileEntry {
    path: String,
    status: String,
    role: Option<String>,
    sha256: Option<String>,
    bytes: Option<usize>,
    product_id: Option<String>,
}

#[derive(Debug)]
struct IndexRow {
    data_set_id: String,
    file_specification_name: String,
    product_id: String,
    start_time: String,
    stop_time: String,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(untagged)]
enum ScalarValue {
    Integer(i64),
    Float(f64),
    Text(String),
}

#[derive(Clone, Debug)]
struct ColumnLayout {
    name: String,
    data_type: String,
    start_byte: usize,
    bytes: usize,
    missing_constant: Option<String>,
}

#[derive(Clone, Debug)]
struct LabelLayout {
    record_bytes: usize,
    file_records: Option<usize>,
    columns: Vec<ColumnLayout>,
}

#[derive(Debug, Serialize)]
struct ProductArtifact {
    spacecraft: u8,
    dataset_id: String,
    data_set_id: String,
    file_specification_name: String,
    product_id: String,
    start_time: String,
    stop_time: String,
    label_path: String,
    data_path: String,
    record_bytes: usize,
    file_records: Option<usize>,
    rows: Vec<BTreeMap<String, Option<ScalarValue>>>,
}

#[derive(Debug, Serialize)]
struct IngestSummary {
    manifest: String,
    spacecraft: u8,
    dataset_id: String,
    verified_files: usize,
    export_format: String,
    products: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColumnEncoding {
    TimestampMillis,
    Float64,
    Int64,
    Utf8,
}

fn sha256_hex(path: &Path) -> Result<String> {
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn verify_manifest_files(manifest: &FetchManifest) -> Result<usize> {
    let mut verified = 0_usize;
    for file in &manifest.files {
        if file.status != "fetched" {
            continue;
        }
        let path = Path::new(&file.path);
        let metadata = fs::metadata(path)
            .with_context(|| format!("missing fetched file {}", path.display()))?;
        if let Some(expected_bytes) = file.bytes
            && metadata.len() as usize != expected_bytes
        {
            bail!(
                "byte-size mismatch for {}: expected {}, got {}",
                path.display(),
                expected_bytes,
                metadata.len()
            );
        }
        if let Some(expected_sha) = &file.sha256 {
            let actual_sha = sha256_hex(path)?;
            if &actual_sha != expected_sha {
                bail!(
                    "sha256 mismatch for {}: expected {}, got {}",
                    path.display(),
                    expected_sha,
                    actual_sha
                );
            }
        }
        verified += 1;
    }
    Ok(verified)
}

fn parse_index_rows(path: &Path) -> Result<Vec<IndexRow>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read headers from {}", path.display()))?
        .iter()
        .map(|header| header.trim().trim_matches('"').to_string())
        .collect::<Vec<_>>();
    let header_index = |name: &str| {
        headers
            .iter()
            .position(|header| header == name)
            .with_context(|| format!("missing {name} in {}", path.display()))
    };
    let data_set_id_idx = header_index("DATA_SET_ID")?;
    let file_spec_idx = header_index("FILE_SPECIFICATION_NAME")?;
    let product_id_idx = header_index("PRODUCT_ID")?;
    let start_time_idx = header_index("START_TIME")?;
    let stop_time_idx = header_index("STOP_TIME")?;
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.with_context(|| format!("failed to parse {}", path.display()))?;
        let field = |idx: usize, name: &str| {
            record
                .get(idx)
                .map(|value| value.trim().trim_matches('"').to_string())
                .with_context(|| format!("missing {name} field in {}", path.display()))
        };
        rows.push(IndexRow {
            data_set_id: field(data_set_id_idx, "DATA_SET_ID")?,
            file_specification_name: field(file_spec_idx, "FILE_SPECIFICATION_NAME")?,
            product_id: field(product_id_idx, "PRODUCT_ID")?,
            start_time: field(start_time_idx, "START_TIME")?,
            stop_time: field(stop_time_idx, "STOP_TIME")?,
        });
    }
    Ok(rows)
}

fn normalize_value(
    text: &str,
    data_type: &str,
    missing_constant: Option<&str>,
) -> Option<ScalarValue> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if missing_constant.is_some_and(|missing| trimmed == missing.trim_matches('"')) {
        return None;
    }
    if matches!(
        data_type,
        "ASCII_REAL" | "REAL" | "FLOAT" | "ASCII_FLOAT" | "ASCII_SCIENTIFIC_NOTATION"
    ) {
        return trimmed.parse::<f64>().ok().map(ScalarValue::Float);
    }
    if matches!(
        data_type,
        "ASCII_INTEGER" | "INTEGER" | "ASCII_NONNEGATIVE_INTEGER"
    ) {
        return trimmed.parse::<i64>().ok().map(ScalarValue::Integer);
    }
    Some(ScalarValue::Text(trimmed.to_string()))
}

fn parse_label_layout(path: &Path) -> Result<LabelLayout> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read label {}", path.display()))?;

    let mut record_bytes = None;
    let mut file_records = None;
    let mut columns = Vec::new();
    let mut in_column = false;
    let mut current_name = None;
    let mut current_data_type = None;
    let mut current_start_byte = None;
    let mut current_bytes = None;
    let mut current_missing_constant = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("/*") {
            continue;
        }
        let parsed_pair = line
            .split_once('=')
            .map(|(key, value)| (key.trim(), value.trim()));
        if parsed_pair == Some(("OBJECT", "COLUMN")) {
            in_column = true;
            current_name = None;
            current_data_type = None;
            current_start_byte = None;
            current_bytes = None;
            current_missing_constant = None;
            continue;
        }
        if parsed_pair == Some(("END_OBJECT", "COLUMN")) {
            in_column = false;
            if let (Some(name), Some(data_type), Some(start_byte), Some(bytes)) = (
                current_name.take(),
                current_data_type.take(),
                current_start_byte.take(),
                current_bytes.take(),
            ) {
                columns.push(ColumnLayout {
                    name,
                    data_type,
                    start_byte,
                    bytes,
                    missing_constant: current_missing_constant.take(),
                });
            }
            continue;
        }
        let Some((key, value)) = parsed_pair else {
            continue;
        };
        let value = value.trim().trim_matches('"');

        if !in_column {
            if key == "RECORD_BYTES" {
                record_bytes = value.parse::<usize>().ok();
            } else if key == "FILE_RECORDS" {
                file_records = value.parse::<usize>().ok();
            }
            continue;
        }

        match key {
            "NAME" => current_name = Some(value.to_string()),
            "DATA_TYPE" => current_data_type = Some(value.to_string()),
            "START_BYTE" => current_start_byte = value.parse::<usize>().ok(),
            "BYTES" => current_bytes = value.parse::<usize>().ok(),
            "MISSING_CONSTANT" => current_missing_constant = Some(value.to_string()),
            _ => {}
        }
    }

    Ok(LabelLayout {
        record_bytes: record_bytes.context("label missing RECORD_BYTES")?,
        file_records,
        columns,
    })
}

fn parse_fixed_width_rows(
    path: &Path,
    layout: &LabelLayout,
) -> Result<Vec<BTreeMap<String, Option<ScalarValue>>>> {
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let text = String::from_utf8_lossy(&data);
    let mut rows = Vec::new();
    for line in text.lines() {
        let bytes = line.as_bytes();
        if bytes.is_empty() {
            continue;
        }
        let mut row = BTreeMap::new();
        let tokenized = line.split_whitespace().collect::<Vec<_>>();
        let use_tokens = tokenized.len() == layout.columns.len();
        for (index, column) in layout.columns.iter().enumerate() {
            let value = if use_tokens {
                normalize_value(
                    tokenized[index],
                    &column.data_type,
                    column.missing_constant.as_deref(),
                )
            } else {
                let start = column.start_byte.saturating_sub(1);
                let end = start.saturating_add(column.bytes);
                let slice = if start >= bytes.len() {
                    ""
                } else {
                    std::str::from_utf8(&bytes[start..bytes.len().min(end)]).unwrap_or("")
                };
                normalize_value(slice, &column.data_type, column.missing_constant.as_deref())
            };
            row.insert(column.name.clone(), value);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn parse_time_millis(text: &str) -> Result<i64> {
    let dt = NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.3f")
        .with_context(|| format!("failed to parse timestamp {text}"))?;
    Ok(dt.and_utc().timestamp_millis())
}

fn column_encoding(data_type: &str) -> ColumnEncoding {
    match data_type.trim_matches('"') {
        "TIME" => ColumnEncoding::TimestampMillis,
        "ASCII_REAL" | "REAL" | "FLOAT" | "ASCII_FLOAT" | "ASCII_SCIENTIFIC_NOTATION" => {
            ColumnEncoding::Float64
        }
        "ASCII_INTEGER" | "INTEGER" | "ASCII_NONNEGATIVE_INTEGER" => ColumnEncoding::Int64,
        _ => ColumnEncoding::Utf8,
    }
}

fn build_arrow_batch(artifact: &ProductArtifact, layout: &LabelLayout) -> Result<RecordBatch> {
    let mut fields = Vec::with_capacity(layout.columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(layout.columns.len());

    for column in &layout.columns {
        match column_encoding(&column.data_type) {
            ColumnEncoding::TimestampMillis => {
                let mut builder = TimestampMillisecondBuilder::new();
                for row in &artifact.rows {
                    match row.get(&column.name).cloned().flatten() {
                        Some(ScalarValue::Text(text)) => {
                            builder.append_value(parse_time_millis(&text)?)
                        }
                        Some(other) => {
                            bail!(
                                "timestamp column {} had non-text value {:?}",
                                column.name,
                                other
                            )
                        }
                        None => builder.append_null(),
                    }
                }
                fields.push(Field::new(
                    &column.name,
                    DataType::Timestamp(TimeUnit::Millisecond, None),
                    true,
                ));
                arrays.push(Arc::new(builder.finish()) as ArrayRef);
            }
            ColumnEncoding::Float64 => {
                let mut builder = Float64Builder::new();
                for row in &artifact.rows {
                    match row.get(&column.name).cloned().flatten() {
                        Some(ScalarValue::Float(value)) => builder.append_value(value),
                        Some(ScalarValue::Integer(value)) => builder.append_value(value as f64),
                        Some(other) => {
                            bail!(
                                "float column {} had incompatible value {:?}",
                                column.name,
                                other
                            )
                        }
                        None => builder.append_null(),
                    }
                }
                fields.push(Field::new(&column.name, DataType::Float64, true));
                arrays.push(Arc::new(builder.finish()) as ArrayRef);
            }
            ColumnEncoding::Int64 => {
                let mut builder = Int64Builder::new();
                for row in &artifact.rows {
                    match row.get(&column.name).cloned().flatten() {
                        Some(ScalarValue::Integer(value)) => builder.append_value(value),
                        Some(other) => {
                            bail!(
                                "int column {} had incompatible value {:?}",
                                column.name,
                                other
                            )
                        }
                        None => builder.append_null(),
                    }
                }
                fields.push(Field::new(&column.name, DataType::Int64, true));
                arrays.push(Arc::new(builder.finish()) as ArrayRef);
            }
            ColumnEncoding::Utf8 => {
                let mut builder = StringBuilder::new();
                for row in &artifact.rows {
                    match row.get(&column.name).cloned().flatten() {
                        Some(ScalarValue::Text(text)) => builder.append_value(text),
                        Some(ScalarValue::Float(value)) => builder.append_value(value.to_string()),
                        Some(ScalarValue::Integer(value)) => {
                            builder.append_value(value.to_string())
                        }
                        None => builder.append_null(),
                    }
                }
                fields.push(Field::new(&column.name, DataType::Utf8, true));
                arrays.push(Arc::new(builder.finish()) as ArrayRef);
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).context("failed to build Arrow record batch")
}

fn write_arrow_file(path: &Path, batch: &RecordBatch) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = FileWriter::try_new(file, &batch.schema())
        .with_context(|| format!("failed to open Arrow writer for {}", path.display()))?;
    writer
        .write(batch)
        .with_context(|| format!("failed to write batch to {}", path.display()))?;
    writer
        .finish()
        .with_context(|| format!("failed to finish Arrow file {}", path.display()))?;
    verify_arrow_file(path, batch)
}

fn verify_arrow_file(path: &Path, expected: &RecordBatch) -> Result<()> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = FileReader::try_new(file, None)
        .with_context(|| format!("failed to read Arrow file {}", path.display()))?;
    let mut batches = Vec::new();
    for maybe_batch in &mut reader {
        batches.push(maybe_batch.with_context(|| format!("failed to decode {}", path.display()))?);
    }
    if batches.len() != 1 {
        bail!(
            "expected exactly one Arrow batch in {}, found {}",
            path.display(),
            batches.len()
        );
    }
    let actual = &batches[0];
    if actual.num_rows() != expected.num_rows() {
        bail!(
            "row-count mismatch in {}: expected {}, got {}",
            path.display(),
            expected.num_rows(),
            actual.num_rows()
        );
    }
    if actual.schema().fields().len() != expected.schema().fields().len() {
        bail!(
            "field-count mismatch in {}: expected {}, got {}",
            path.display(),
            expected.schema().fields().len(),
            actual.schema().fields().len()
        );
    }
    for (expected_array, actual_array) in expected.columns().iter().zip(actual.columns()) {
        if expected_array.null_count() != actual_array.null_count() {
            bail!(
                "null-count mismatch in {}: expected {}, got {}",
                path.display(),
                expected_array.null_count(),
                actual_array.null_count()
            );
        }
    }
    Ok(())
}

fn find_role_file<'a>(files: &'a [FetchFileEntry], role: &str) -> Option<&'a FetchFileEntry> {
    files
        .iter()
        .find(|file| file.status == "fetched" && file.role.as_deref() == Some(role))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.out_dir)?;

    let manifest_text = fs::read_to_string(&cli.manifest)
        .with_context(|| format!("failed to read {}", cli.manifest.display()))?;
    let manifest: FetchManifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("failed to parse {}", cli.manifest.display()))?;

    if manifest.product != "jupiter_encounter" {
        bail!(
            "ingest-pds-crs currently supports only jupiter_encounter manifests, got {}",
            manifest.product
        );
    }

    let verified_files = verify_manifest_files(&manifest)?;
    let dataset_id = manifest
        .pds_ppi_dataset
        .clone()
        .context("manifest missing pds_ppi_dataset")?;

    let index_tab = find_role_file(&manifest.files, "index_tab")
        .context("manifest missing fetched index_tab entry")?;
    let index_rows = parse_index_rows(Path::new(&index_tab.path))?;

    let mut product_names = Vec::new();
    for row in index_rows {
        let product_id = row.product_id.trim().to_string();
        let label_path = manifest
            .files
            .iter()
            .find(|file| {
                file.status == "fetched"
                    && file.role.as_deref() == Some("data_label")
                    && file
                        .product_id
                        .as_deref()
                        .is_some_and(|candidate| candidate.trim() == product_id)
            })
            .context("missing data label for product in manifest")?;
        let data_path = manifest
            .files
            .iter()
            .find(|file| {
                file.status == "fetched"
                    && file.role.as_deref() == Some("data_table")
                    && file
                        .product_id
                        .as_deref()
                        .is_some_and(|candidate| candidate.trim() == product_id)
            })
            .context("missing data table for product in manifest")?;

        let label_layout = parse_label_layout(Path::new(&label_path.path))?;
        let rows = parse_fixed_width_rows(Path::new(&data_path.path), &label_layout)?;
        let artifact = ProductArtifact {
            spacecraft: manifest.spacecraft,
            dataset_id: dataset_id.clone(),
            data_set_id: row.data_set_id,
            file_specification_name: row.file_specification_name,
            product_id: product_id.clone(),
            start_time: row.start_time,
            stop_time: row.stop_time,
            label_path: label_path.path.clone(),
            data_path: data_path.path.clone(),
            record_bytes: label_layout.record_bytes,
            file_records: label_layout.file_records,
            rows,
        };

        let output_stem = format!("{}_{}", dataset_id, product_id.trim().to_lowercase());
        match cli.export_format {
            ExportFormat::Json => {
                let output_path = cli.out_dir.join(format!("{output_stem}.json"));
                fs::write(&output_path, serde_json::to_vec_pretty(&artifact)?)
                    .with_context(|| format!("failed to write {}", output_path.display()))?;
                println!("wrote {}", output_path.display());
            }
            ExportFormat::Arrow => {
                let batch = build_arrow_batch(&artifact, &label_layout)?;
                let output_path = cli.out_dir.join(format!("{output_stem}.arrow"));
                write_arrow_file(&output_path, &batch)?;
                println!("wrote {}", output_path.display());
            }
        }
        product_names.push(product_id);
    }

    let summary = IngestSummary {
        manifest: cli.manifest.to_string_lossy().into_owned(),
        spacecraft: manifest.spacecraft,
        dataset_id,
        verified_files,
        export_format: match cli.export_format {
            ExportFormat::Json => "json",
            ExportFormat::Arrow => "arrow",
        }
        .to_string(),
        products: product_names,
    };
    let summary_path = cli.out_dir.join("ingest_summary.json");
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("wrote {}", summary_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_LABEL: &str = r#"
RECORD_BYTES = 20
FILE_RECORDS = 2
OBJECT = COLUMN
  NAME = "TIME"
  DATA_TYPE = "CHARACTER"
  START_BYTE = 1
  BYTES = 8
END_OBJECT = COLUMN
OBJECT = COLUMN
  NAME = "RATE"
  DATA_TYPE = "ASCII_REAL"
  START_BYTE = 9
  BYTES = 6
  MISSING_CONSTANT = -1.000E+31
END_OBJECT = COLUMN
"#;

    #[test]
    fn test_parse_label_layout_extracts_columns() {
        let path = std::env::temp_dir().join("sample_pds_label.lbl");
        fs::write(&path, SAMPLE_LABEL).expect("write sample label");
        let layout = parse_label_layout(&path).expect("parse layout");
        assert_eq!(layout.record_bytes, 20);
        assert_eq!(layout.file_records, Some(2));
        assert_eq!(layout.columns.len(), 2);
        assert_eq!(layout.columns[1].name, "RATE");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_normalize_value_honors_missing_constant() {
        assert_eq!(
            normalize_value("-1.000E+31", "ASCII_REAL", Some("-1.000E+31")),
            None
        );
        assert_eq!(
            normalize_value("42", "ASCII_INTEGER", None),
            Some(ScalarValue::Integer(42))
        );
    }

    #[test]
    fn test_parse_fixed_width_rows_prefers_whitespace_tokens_when_aligned() {
        let path = std::env::temp_dir().join("sample_pds_rows.tab");
        fs::write(
            &path,
            "1979-02-28T00:00:00.000   2.71267e-03  2.71267e-03 -9.99999e+10 -9.99999e+10\n",
        )
        .expect("write sample rows");
        let layout = LabelLayout {
            record_bytes: 78,
            file_records: Some(1),
            columns: vec![
                ColumnLayout {
                    name: "TIME".to_string(),
                    data_type: "TIME".to_string(),
                    start_byte: 1,
                    bytes: 23,
                    missing_constant: None,
                },
                ColumnLayout {
                    name: "A".to_string(),
                    data_type: "ASCII_REAL".to_string(),
                    start_byte: 26,
                    bytes: 12,
                    missing_constant: None,
                },
                ColumnLayout {
                    name: "STD_A".to_string(),
                    data_type: "ASCII_REAL".to_string(),
                    start_byte: 39,
                    bytes: 12,
                    missing_constant: None,
                },
                ColumnLayout {
                    name: "B".to_string(),
                    data_type: "ASCII_REAL".to_string(),
                    start_byte: 51,
                    bytes: 12,
                    missing_constant: Some("-9.99999e+10".to_string()),
                },
                ColumnLayout {
                    name: "STD_B".to_string(),
                    data_type: "ASCII_REAL".to_string(),
                    start_byte: 64,
                    bytes: 12,
                    missing_constant: Some("-9.99999e+10".to_string()),
                },
            ],
        };
        let rows = parse_fixed_width_rows(&path, &layout).expect("parse rows");
        assert_eq!(
            rows[0].get("TIME"),
            Some(&Some(ScalarValue::Text(
                "1979-02-28T00:00:00.000".to_string()
            )))
        );
        assert_eq!(rows[0].get("B"), Some(&None));
        assert_eq!(rows[0].get("STD_B"), Some(&None));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_build_arrow_batch_tracks_nulls_and_timestamps() {
        let artifact = ProductArtifact {
            spacecraft: 1,
            dataset_id: "VG1".to_string(),
            data_set_id: "VG1".to_string(),
            file_specification_name: "/DATA/TEST.LBL".to_string(),
            product_id: "TEST".to_string(),
            start_time: "1979-02-28T00:00:00.000".to_string(),
            stop_time: "1979-02-28T00:15:00.000".to_string(),
            label_path: "/tmp/test.lbl".to_string(),
            data_path: "/tmp/test.tab".to_string(),
            record_bytes: 10,
            file_records: Some(2),
            rows: vec![
                BTreeMap::from([
                    (
                        "TIME".to_string(),
                        Some(ScalarValue::Text("1979-02-28T00:00:00.000".to_string())),
                    ),
                    ("RATE".to_string(), Some(ScalarValue::Float(1.5))),
                ]),
                BTreeMap::from([
                    (
                        "TIME".to_string(),
                        Some(ScalarValue::Text("1979-02-28T00:15:00.000".to_string())),
                    ),
                    ("RATE".to_string(), None),
                ]),
            ],
        };
        let layout = LabelLayout {
            record_bytes: 10,
            file_records: Some(2),
            columns: vec![
                ColumnLayout {
                    name: "TIME".to_string(),
                    data_type: "TIME".to_string(),
                    start_byte: 1,
                    bytes: 23,
                    missing_constant: None,
                },
                ColumnLayout {
                    name: "RATE".to_string(),
                    data_type: "ASCII_REAL".to_string(),
                    start_byte: 24,
                    bytes: 12,
                    missing_constant: Some("-9.99999e+10".to_string()),
                },
            ],
        };
        let batch = build_arrow_batch(&artifact, &layout).expect("build batch");
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.column(1).null_count(), 1);
    }
}
