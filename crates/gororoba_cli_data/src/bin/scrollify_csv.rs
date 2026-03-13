use clap::{Parser, ValueEnum};
use glob::glob;
use rusqlite::{Connection, params};
use scrolls_core::{
    ConvertSpec, ScrollDataset, ScrollIndexEntry, ScrollRowPayloadRef, convert_csv_to_scroll,
    render_scroll_dataset, render_scroll_index, slugify,
};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

const DEFAULT_MAX_INLINE_TOML_BYTES: usize = 50_000_000;
const DEFAULT_ROWS_PREVIEW_COUNT: usize = 8;
const PAYLOAD_ROW_TABLE: &str = "dataset_rows";

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DatasetClassArg {
    CanonicalDataset,
    GeneratedArtifact,
    HoldingExternal,
    HoldingArchive,
}

impl DatasetClassArg {
    fn as_str(self) -> &'static str {
        match self {
            DatasetClassArg::CanonicalDataset => "canonical_dataset",
            DatasetClassArg::GeneratedArtifact => "generated_artifact",
            DatasetClassArg::HoldingExternal => "holding_external_csv",
            DatasetClassArg::HoldingArchive => "holding_archive_csv",
        }
    }
}

/// Convert CSV files into TOML scroll datasets and emit a canonical index.
#[derive(Debug, Parser)]
struct Args {
    /// Source glob for CSV files (alternative to --source-manifest).
    #[arg(long)]
    source_glob: Option<String>,
    /// Text manifest containing source CSV paths (one per line).
    #[arg(long)]
    source_manifest: Option<PathBuf>,
    /// Output index TOML path.
    #[arg(long)]
    out_index: PathBuf,
    /// Output directory for per-dataset TOML files.
    #[arg(long)]
    out_dir: PathBuf,
    /// Top-level index metadata table name.
    #[arg(long)]
    index_table: String,
    /// Dataset id prefix, e.g. PC/PG/AH/EH.
    #[arg(long)]
    dataset_prefix: String,
    /// Human-readable corpus label.
    #[arg(long)]
    corpus_label: String,
    /// Dataset class annotation.
    #[arg(long, value_enum, default_value_t = DatasetClassArg::CanonicalDataset)]
    dataset_class: DatasetClassArg,
    /// Optional SQLite store for oversized row payloads.
    #[arg(long)]
    sqlite_overflow_db: Option<PathBuf>,
    /// Spill to SQLite when the rendered dataset TOML would exceed this many bytes.
    #[arg(long, default_value_t = DEFAULT_MAX_INLINE_TOML_BYTES)]
    max_inline_toml_bytes: usize,
    /// Number of leading rows to keep inline as a human-readable preview when spilling.
    #[arg(long, default_value_t = DEFAULT_ROWS_PREVIEW_COUNT)]
    rows_preview_count: usize,
}

fn load_manifest(path: &Path) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("{}: {}", path.display(), err))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect())
}

fn collect_sources(args: &Args) -> Result<Vec<String>, String> {
    let mut paths: BTreeSet<String> = BTreeSet::new();
    if let Some(pattern) = &args.source_glob {
        let entries = glob(pattern).map_err(|err| format!("glob parse {}: {}", pattern, err))?;
        for entry in entries {
            let path = entry.map_err(|err| format!("glob match error: {}", err))?;
            paths.insert(path.to_string_lossy().to_string());
        }
    }
    if let Some(manifest) = &args.source_manifest {
        for item in load_manifest(manifest)? {
            paths.insert(item);
        }
    }
    if paths.is_empty() {
        return Err(
            "no input CSV files found; pass --source-glob and/or --source-manifest".to_string(),
        );
    }
    Ok(paths.into_iter().collect())
}

fn sqlite_path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn ensure_payload_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         CREATE TABLE IF NOT EXISTS datasets (
             dataset_id TEXT PRIMARY KEY,
             canonical_toml TEXT NOT NULL,
             source_csv TEXT NOT NULL,
             source_sha256 TEXT NOT NULL,
             source_size_bytes INTEGER NOT NULL,
             row_count INTEGER NOT NULL,
             column_count INTEGER NOT NULL,
             header_json TEXT NOT NULL,
             original_header_json TEXT NOT NULL,
             header_value_sha256 TEXT NOT NULL,
             row_value_sha256 TEXT NOT NULL,
             dataset_class TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS dataset_rows (
             dataset_id TEXT NOT NULL,
             row_index INTEGER NOT NULL,
             row_json TEXT NOT NULL,
             PRIMARY KEY(dataset_id, row_index),
             FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
         );",
    )
    .map_err(|err| format!("initialize SQLite payload schema: {}", err))
}

fn spill_rows_to_sqlite(
    sqlite_path: &Path,
    dataset: &ScrollDataset,
    canonical_toml: &str,
    rows: &[Vec<String>],
) -> Result<(), String> {
    if let Some(parent) = sqlite_path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("mkdir {}: {}", parent.display(), err))?;
    }
    let mut conn = Connection::open(sqlite_path)
        .map_err(|err| format!("open sqlite {}: {}", sqlite_path.display(), err))?;
    ensure_payload_schema(&conn)?;

    let dataset_id = dataset.dataset.id.clone();
    let header_json = serde_json::to_string(&dataset.dataset.header)
        .map_err(|err| format!("serialize header for {}: {}", dataset_id, err))?;
    let original_header_json = serde_json::to_string(&dataset.dataset.original_header)
        .map_err(|err| format!("serialize original_header for {}: {}", dataset_id, err))?;

    let tx = conn
        .transaction()
        .map_err(|err| format!("begin sqlite transaction for {}: {}", dataset_id, err))?;
    tx.execute(
        "DELETE FROM dataset_rows WHERE dataset_id = ?1",
        params![dataset_id.as_str()],
    )
    .map_err(|err| format!("clear sqlite rows for {}: {}", dataset_id, err))?;
    tx.execute(
        "DELETE FROM datasets WHERE dataset_id = ?1",
        params![dataset_id.as_str()],
    )
    .map_err(|err| format!("clear sqlite dataset for {}: {}", dataset_id, err))?;
    tx.execute(
        "INSERT INTO datasets (
             dataset_id,
             canonical_toml,
             source_csv,
             source_sha256,
             source_size_bytes,
             row_count,
             column_count,
             header_json,
             original_header_json,
             header_value_sha256,
             row_value_sha256,
             dataset_class
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            dataset_id.as_str(),
            canonical_toml,
            dataset.dataset.source_csv.as_str(),
            dataset.dataset.source_sha256.as_str(),
            dataset.dataset.source_size_bytes as i64,
            dataset.dataset.row_count as i64,
            dataset.dataset.column_count as i64,
            header_json,
            original_header_json,
            dataset.dataset.header_value_sha256.as_str(),
            dataset.dataset.row_value_sha256.as_str(),
            dataset.dataset.dataset_class.as_str(),
        ],
    )
    .map_err(|err| format!("insert sqlite dataset for {}: {}", dataset_id, err))?;
    {
        let mut stmt = tx
            .prepare(
                "INSERT INTO dataset_rows (dataset_id, row_index, row_json)
                 VALUES (?1, ?2, ?3)",
            )
            .map_err(|err| format!("prepare sqlite row insert for {}: {}", dataset_id, err))?;
        for (row_index, row) in rows.iter().enumerate() {
            let row_json = serde_json::to_string(row)
                .map_err(|err| format!("serialize row {} for {}: {}", row_index, dataset_id, err))?;
            stmt.execute(params![dataset_id.as_str(), row_index as i64, row_json])
                .map_err(|err| {
                    format!(
                        "insert sqlite row {} for {}: {}",
                        row_index, dataset_id, err
                    )
                })?;
        }
    }
    tx.commit()
        .map_err(|err| format!("commit sqlite payload for {}: {}", dataset_id, err))
}

fn compact_dataset_if_needed(
    rendered_len: usize,
    out_path: &Path,
    dataset: &mut ScrollDataset,
    index_entry: &mut ScrollIndexEntry,
    args: &Args,
) -> Result<Option<PathBuf>, String> {
    let Some(sqlite_path) = args.sqlite_overflow_db.clone() else {
        return Ok(None);
    };
    if rendered_len <= args.max_inline_toml_bytes {
        return Ok(None);
    }
    let rows = dataset
        .dataset
        .rows
        .clone()
        .ok_or_else(|| format!("dataset {} is missing inline rows", dataset.dataset.id))?;
    spill_rows_to_sqlite(&sqlite_path, dataset, &sqlite_path_string(out_path), &rows)?;
    let preview_rows = if args.rows_preview_count == 0 {
        None
    } else {
        Some(
            rows.iter()
                .take(args.rows_preview_count)
                .cloned()
                .collect::<Vec<_>>(),
        )
    };
    let sqlite_str = sqlite_path_string(&sqlite_path);
    dataset.dataset.rows = None;
    dataset.dataset.rows_preview = preview_rows;
    dataset.dataset.row_payload = Some(ScrollRowPayloadRef {
        backend: "sqlite".to_string(),
        sqlite_path: sqlite_str.clone(),
        sqlite_table: PAYLOAD_ROW_TABLE.to_string(),
        dataset_id: dataset.dataset.id.clone(),
        format: "json_row_array".to_string(),
        lfs_tracked: true,
    });
    index_entry.row_payload_backend = Some("sqlite".to_string());
    index_entry.row_payload_sqlite = Some(sqlite_str);
    index_entry.row_payload_table = Some(PAYLOAD_ROW_TABLE.to_string());
    Ok(Some(sqlite_path))
}

fn run(args: Args) -> Result<(), String> {
    let inputs = collect_sources(&args)?;
    fs::create_dir_all(&args.out_dir)
        .map_err(|err| format!("mkdir {}: {}", args.out_dir.display(), err))?;
    if let Some(parent) = args.out_index.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("mkdir {}: {}", parent.display(), err))?;
    }

    let mut entries: Vec<ScrollIndexEntry> = Vec::new();
    let mut touched_sqlite: BTreeSet<String> = BTreeSet::new();
    for (idx, source_csv) in inputs.iter().enumerate() {
        let source_path = Path::new(source_csv);
        if !source_path.exists() {
            return Err(format!("missing source CSV: {}", source_csv));
        }
        let dataset_id = format!("{}-{:04}", args.dataset_prefix, idx + 1);
        let file_name = source_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("dataset.csv");
        let slug = slugify(file_name);
        let out_name = format!("{}_{}.toml", dataset_id, slug);
        let out_path = args.out_dir.join(out_name);
        let canonical_toml = out_path.to_string_lossy().to_string();
        let spec = ConvertSpec {
            dataset_id: &dataset_id,
            slug: &slug,
            source_csv,
            canonical_toml: &canonical_toml,
            dataset_class: args.dataset_class.as_str(),
            corpus_label: &args.corpus_label,
            migrated_by: "gororoba_cli::scrollify-csv",
        };
        let mut converted = convert_csv_to_scroll(source_path, &spec)
            .map_err(|err| format!("{}: {}", source_csv, err))?;
        if let Some(sqlite_path) = compact_dataset_if_needed(
            converted.rendered_dataset_toml.len(),
            &out_path,
            &mut converted.dataset,
            &mut converted.index_entry,
            &args,
        )? {
            touched_sqlite.insert(sqlite_path_string(&sqlite_path));
            converted.rendered_dataset_toml = render_scroll_dataset(&converted.dataset)
                .map_err(|err| format!("render compact scroll for {}: {}", source_csv, err))?;
        }
        fs::write(&out_path, converted.rendered_dataset_toml)
            .map_err(|err| format!("write {}: {}", out_path.display(), err))?;
        entries.push(converted.index_entry);
    }

    let source_descriptor = if let Some(manifest) = &args.source_manifest {
        format!("manifest:{}", manifest.display())
    } else if let Some(glob) = &args.source_glob {
        format!("glob:{}", glob)
    } else {
        "manual".to_string()
    };
    let rendered_index = render_scroll_index(
        &entries,
        &args.index_table,
        &source_descriptor,
        &args.out_dir.to_string_lossy(),
        &args.corpus_label,
        "gororoba_cli::scrollify-csv",
    );
    fs::write(&args.out_index, rendered_index)
        .map_err(|err| format!("write {}: {}", args.out_index.display(), err))?;

    println!(
        "Wrote {} with {} datasets and {} scroll TOML files in {}.",
        args.out_index.display(),
        entries.len(),
        entries.len(),
        args.out_dir.display()
    );
    if !touched_sqlite.is_empty() {
        for path in touched_sqlite {
            println!("Spilled oversized row payloads into {}.", path);
        }
    }
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("ERROR: {}", err);
        std::process::exit(1);
    }
}
