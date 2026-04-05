use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use csv::ReaderBuilder;
use glob::glob;
use regex::Regex;
use scrolls_core::ConvertSpec;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use toml::Value;
use walkdir::WalkDir;

type Table = toml::map::Map<String, Value>;

const UPDATED_STAMP: &str = "2026-02-09";
const BIN_PATH: &str = "crates/gororoba_cli_data/src/bin/csv_canonicalization.rs";
const DEFAULT_SOURCE_GLOB: &str = "data/csv/legacy/*.csv";
const DEFAULT_CANON_DIR: &str = "registry/data/legacy_csv";
const DEFAULT_INDEX_PATH: &str = "registry/legacy_csv_datasets.toml";
const DEFAULT_INDEX_TABLE: &str = "legacy_csv_datasets";
const DEFAULT_DATASET_PREFIX: &str = "LC";
const DEFAULT_CORPUS_LABEL: &str = "legacy CSV";
const MAKEFILE_GENERATED_GLOBS: &[&str] = &[
    "data/csv/cd_motif_*.csv",
    "data/csv/de_marrais_*.csv",
    "data/csv/reggiani_*.csv",
    "data/csv/m3_table.csv",
    "data/csv/dimensional_geometry_*.csv",
    "data/csv/materials_jarvis_subset.csv",
    "data/csv/materials_embedding_benchmarks.csv",
    "data/csv/modular_chaos_*.csv",
    "data/csv/sedenion_field_metrics_*.csv",
    "data/csv/spectral_flow.csv",
];

#[derive(Parser, Debug)]
#[command(
    name = "csv-canonicalization",
    about = "Rust CSV inventory, canonicalization, parity verification, and project split policy"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Inventory(InventoryArgs),
    Migrate(MigrateArgs),
    Verify(VerifyArgs),
    ProjectSplitPolicy(ProjectSplitPolicyArgs),
    Holdings(HoldingsArgs),
    VerifyHoldings(VerifyHoldingsArgs),
    ScrollPipeline(ScrollPipelineArgs),
    VerifyScrollPipeline(VerifyScrollPipelineArgs),
    VerifyCorpusCoverage(VerifyCorpusCoverageArgs),
    MigrationScope(MigrationScopeArgs),
}

#[derive(Parser, Debug)]
struct InventoryArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct MigrateArgs {
    #[arg(long, default_value = DEFAULT_SOURCE_GLOB)]
    source_glob: String,

    #[arg(long)]
    source_manifest: Option<PathBuf>,

    #[arg(long, default_value = DEFAULT_INDEX_PATH)]
    out_index: PathBuf,

    #[arg(long, default_value = DEFAULT_CANON_DIR)]
    out_dir: PathBuf,

    #[arg(long, default_value = DEFAULT_INDEX_TABLE)]
    index_table: String,

    #[arg(long, default_value = DEFAULT_DATASET_PREFIX)]
    dataset_prefix: String,

    #[arg(long, default_value = DEFAULT_CORPUS_LABEL)]
    corpus_label: String,
}

#[derive(Parser, Debug)]
struct VerifyArgs {
    #[arg(long, default_value = DEFAULT_INDEX_PATH)]
    index_path: PathBuf,

    #[arg(long, default_value = DEFAULT_SOURCE_GLOB)]
    source_glob: String,

    #[arg(long)]
    source_manifest: Option<PathBuf>,

    #[arg(long, default_value = DEFAULT_CORPUS_LABEL)]
    corpus_label: String,

    #[arg(long, default_value_t = false)]
    coverage_only: bool,
}

#[derive(Parser, Debug)]
struct ProjectSplitPolicyArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/project_csv_split_policy.toml")]
    out: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/project_csv_canonical_manifest.txt"
    )]
    canonical_manifest: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/project_csv_generated_manifest.txt"
    )]
    generated_manifest: PathBuf,
}

#[derive(Parser, Debug)]
struct HoldingsArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/external_csv_holding.toml")]
    external_out: PathBuf,

    #[arg(long, default_value = "registry/archive_csv_holding.toml")]
    archive_out: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/external_csv_holding_manifest.txt"
    )]
    external_manifest: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/archive_csv_holding_manifest.txt"
    )]
    archive_manifest: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyHoldingsArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/external_csv_holding.toml")]
    external_registry: PathBuf,

    #[arg(long, default_value = "registry/archive_csv_holding.toml")]
    archive_registry: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/external_csv_holding_manifest.txt"
    )]
    external_manifest: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/archive_csv_holding_manifest.txt"
    )]
    archive_manifest: PathBuf,
}

#[derive(Parser, Debug)]
struct ScrollPipelineArgs {
    #[arg(long, default_value = "registry/csv_scroll_pipeline.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyScrollPipelineArgs {
    #[arg(long, default_value = "registry/csv_scroll_pipeline.toml")]
    pipeline: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyCorpusCoverageArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/project_csv_canonical_datasets.toml")]
    project_canonical_index: PathBuf,

    #[arg(long, default_value = "registry/project_csv_generated_artifacts.toml")]
    project_generated_index: PathBuf,

    #[arg(long, default_value = "registry/legacy_csv_datasets.toml")]
    legacy_index: PathBuf,

    #[arg(long, default_value = "registry/curated_csv_datasets.toml")]
    curated_index: PathBuf,

    #[arg(long, default_value = "registry/external_csv_holding_datasets.toml")]
    external_holding_index: PathBuf,

    #[arg(long, default_value = "registry/archive_csv_holding_datasets.toml")]
    archive_holding_index: PathBuf,
}

#[derive(Parser, Debug)]
struct MigrationScopeArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/csv_migration_scope.toml")]
    out: PathBuf,
}

#[derive(Debug, Clone)]
struct CsvDoc {
    path: String,
    git_status: String,
    zone: String,
    archived: bool,
    generated: bool,
    size_bytes: usize,
    line_count: usize,
    sha256: String,
    migration_action: String,
    migration_priority: String,
    rationale: String,
}

#[derive(Debug, Clone)]
struct Dataset {
    dataset_id: String,
    slug: String,
    source_csv: String,
    source_sha256: String,
    source_size_bytes: usize,
    has_header: bool,
    delimiter: char,
    quotechar: char,
    row_count: usize,
    column_count: usize,
    header: Vec<String>,
    original_header: Vec<String>,
    rows: Vec<Vec<String>>,
    header_value_sha256: String,
    row_value_sha256: String,
    canonical_toml: String,
    column_types: Vec<String>,
    non_empty_counts: Vec<usize>,
    empty_counts: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ExistingDatasetMeta {
    dataset_id: String,
    canonical_toml: String,
    source_sha256: String,
    has_header: bool,
    delimiter: char,
    quotechar: char,
}

#[derive(Debug, Clone)]
struct HoldingRow {
    path: String,
    source_sha256: String,
    size_bytes: usize,
    git_status: String,
    target_lane: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = cli.repo_root.canonicalize().context("resolve repo root")?;
    match cli.command {
        Commands::Inventory(args) => run_inventory(&repo_root, &args),
        Commands::Migrate(args) => run_migrate(&repo_root, &args),
        Commands::Verify(args) => run_verify(&repo_root, &args),
        Commands::ProjectSplitPolicy(args) => run_project_split_policy(&repo_root, &args),
        Commands::Holdings(args) => run_holdings(&repo_root, &args),
        Commands::VerifyHoldings(args) => run_verify_holdings(&repo_root, &args),
        Commands::ScrollPipeline(args) => run_scroll_pipeline(&repo_root, &args),
        Commands::VerifyScrollPipeline(args) => run_verify_scroll_pipeline(&repo_root, &args),
        Commands::VerifyCorpusCoverage(args) => run_verify_corpus_coverage(&repo_root, &args),
        Commands::MigrationScope(args) => run_migration_scope(&repo_root, &args),
    }
}

fn run_inventory(repo_root: &Path, args: &InventoryArgs) -> Result<()> {
    let tracked = git_paths(repo_root, &["ls-files", "*.csv"])?;
    let untracked = git_paths(
        repo_root,
        &["ls-files", "--others", "--exclude-standard", "*.csv"],
    )?;
    let ignored = git_paths(
        repo_root,
        &[
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "*.csv",
        ],
    )?;
    let legacy_canonical_paths = load_canonical_source_paths(
        repo_root,
        "registry/legacy_csv_datasets.toml",
        "legacy_csv_datasets",
    )?;
    let curated_canonical_paths = load_canonical_source_paths(
        repo_root,
        "registry/curated_csv_datasets.toml",
        "curated_csv_datasets",
    )?;
    let project_canonical_paths = load_canonical_source_paths(
        repo_root,
        "registry/project_csv_canonical_datasets.toml",
        "project_csv_canonical_datasets",
    )?;
    let project_generated_paths = load_canonical_source_paths(
        repo_root,
        "registry/project_csv_generated_artifacts.toml",
        "project_csv_generated_artifacts",
    )?;
    let external_holding_paths = load_canonical_source_paths(
        repo_root,
        "registry/external_csv_holding.toml",
        "external_csv_holding",
    )?;
    let archive_holding_paths = load_canonical_source_paths(
        repo_root,
        "registry/archive_csv_holding.toml",
        "archive_csv_holding",
    )?;
    let external_holding_scroll_paths = load_canonical_source_paths(
        repo_root,
        "registry/external_csv_holding_datasets.toml",
        "external_csv_holding_datasets",
    )?;
    let archive_holding_scroll_paths = load_canonical_source_paths(
        repo_root,
        "registry/archive_csv_holding_datasets.toml",
        "archive_csv_holding_datasets",
    )?;
    let project_split_classification = load_project_split_classification(repo_root)?;
    let files = all_filesystem_csv(repo_root)?;

    let mut docs = Vec::new();
    for rel in files {
        let path = repo_root.join(&rel);
        let raw = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let text = String::from_utf8_lossy(&raw);
        let zone = zone_for(&rel);
        let (action, priority, rationale) = policy_with_progress(
            &rel,
            &zone,
            &legacy_canonical_paths,
            &curated_canonical_paths,
            &project_canonical_paths,
            &project_generated_paths,
            &external_holding_paths,
            &archive_holding_paths,
            &external_holding_scroll_paths,
            &archive_holding_scroll_paths,
            &project_split_classification,
        );
        let classification = project_split_classification
            .get(&rel)
            .cloned()
            .unwrap_or_default();
        docs.push(CsvDoc {
            path: rel.clone(),
            git_status: git_status(&rel, &tracked, &untracked, &ignored),
            zone: zone.clone(),
            archived: rel.starts_with("archive/") || rel.starts_with("docs/archive/"),
            generated: if zone == "project_csv" {
                classification == "generated_artifact"
            } else {
                rel.starts_with("data/csv/") && !rel.starts_with("data/csv/legacy/")
            },
            size_bytes: raw.len(),
            line_count: text.matches('\n').count() + usize::from(!text.is_empty()),
            sha256: sha256_hex(&raw),
            migration_action: action,
            migration_priority: priority,
            rationale,
        });
    }

    let rendered = render_csv_inventory(&docs)?;
    let out_path = repo_root.join(&args.out);
    write_ascii(&out_path, &(rendered + "\n"))?;
    println!(
        "Wrote {} with {} CSV entries.",
        out_path.display(),
        docs.len()
    );
    Ok(())
}

fn run_migrate(repo_root: &Path, args: &MigrateArgs) -> Result<()> {
    let out_dir = repo_root.join(&args.out_dir);
    fs::create_dir_all(&out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    let sources = source_paths(repo_root, &args.source_glob, args.source_manifest.as_ref())?;
    if sources.is_empty() {
        bail!("ERROR: no source CSV files found for requested source set");
    }

    let dataset_prefix = args.dataset_prefix.trim().to_ascii_uppercase();
    if !dataset_prefix_regex().is_match(&dataset_prefix) {
        bail!("ERROR: --dataset-prefix must match [A-Z0-9]{{1,6}}");
    }

    let existing = load_existing_dataset_meta(repo_root, &args.out_index)?;
    let mut datasets = Vec::new();
    for (idx, rel) in sources.iter().enumerate() {
        let source = repo_root.join(rel);
        let raw = fs::read(&source).with_context(|| format!("read {}", source.display()))?;
        let source_sha = sha256_hex(&raw);
        let existing_meta = existing.get(rel);
        if let Some(meta) = existing_meta
            && meta.source_sha256 == source_sha
        {
            let mut dataset = load_existing_dataset(repo_root, &meta.canonical_toml)?;
            dataset.source_csv = rel.clone();
            dataset.source_sha256 = source_sha;
            dataset.source_size_bytes = raw.len();
            dataset.canonical_toml = meta.canonical_toml.clone();
            datasets.push(dataset);
            continue;
        }
        let (has_header, delimiter, quotechar, header, original_header, rows) =
            parse_csv_with_stability(&source, existing_meta)?;
        let row_count = rows.len();
        let column_count = header.len();
        let (column_types, non_empty_counts, empty_counts) = profile_columns(&rows, column_count);

        let (dataset_id, canonical_toml) = if let Some(meta) = existing_meta {
            (meta.dataset_id.clone(), meta.canonical_toml.clone())
        } else {
            let dataset_id = format!("{dataset_prefix}-{:04}", idx + 1);
            let slug = slugify(
                source
                    .file_name()
                    .and_then(|v| v.to_str())
                    .unwrap_or("dataset.csv"),
            );
            let canon_name = format!("{dataset_id}_{slug}.toml");
            (
                dataset_id,
                args.out_dir
                    .join(canon_name)
                    .to_string_lossy()
                    .replace('\\', "/"),
            )
        };
        let slug = if let Some(meta) = existing_meta {
            dataset_slug_from_canonical(&meta.canonical_toml).unwrap_or_else(|| {
                slugify(
                    source
                        .file_name()
                        .and_then(|v| v.to_str())
                        .unwrap_or("dataset.csv"),
                )
            })
        } else {
            slugify(
                source
                    .file_name()
                    .and_then(|v| v.to_str())
                    .unwrap_or("dataset.csv"),
            )
        };

        datasets.push(Dataset {
            dataset_id,
            slug,
            source_csv: rel.clone(),
            source_sha256: source_sha,
            source_size_bytes: raw.len(),
            has_header,
            delimiter,
            quotechar,
            row_count,
            column_count,
            header: header.clone(),
            original_header: original_header.clone(),
            rows: rows.clone(),
            header_value_sha256: sha_text_json(&header)?,
            row_value_sha256: sha_text_json(&rows)?,
            canonical_toml,
            column_types,
            non_empty_counts,
            empty_counts,
        });
    }

    let mut rendered_by_path = BTreeMap::new();
    for dataset in &datasets {
        let out_path = repo_root.join(&dataset.canonical_toml);
        rendered_by_path.insert(out_path, render_dataset(dataset, &args.corpus_label)?);
    }
    let expected = rendered_by_path
        .keys()
        .map(|path| path.canonicalize().unwrap_or_else(|_| path.clone()))
        .collect::<BTreeSet<_>>();
    for entry in
        fs::read_dir(&out_dir).with_context(|| format!("read_dir {}", out_dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|v| v.to_str()) == Some("toml") {
            let canonical = path.canonicalize().unwrap_or(path.clone());
            if !expected.contains(&canonical) {
                fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
            }
        }
    }
    for (path, text) in rendered_by_path {
        write_ascii(&path, &(text + "\n"))?;
    }

    let index_text = render_dataset_index(
        &datasets,
        &args.index_table,
        &args.source_glob,
        &args.out_dir.to_string_lossy(),
        &args.corpus_label,
    )?;
    let index_path = repo_root.join(&args.out_index);
    write_ascii(&index_path, &(index_text + "\n"))?;
    println!(
        "Wrote {} with {} datasets.",
        index_path.display(),
        datasets.len()
    );
    println!(
        "Wrote {} canonical TOML dataset files under {}.",
        datasets.len(),
        out_dir.display()
    );
    Ok(())
}

fn run_verify(repo_root: &Path, args: &VerifyArgs) -> Result<()> {
    let index_path = repo_root.join(&args.index_path);
    let data = load_toml(&index_path)?;
    let datasets = table_array(&data, "dataset")?;
    let source_paths = source_paths(repo_root, &args.source_glob, args.source_manifest.as_ref())?
        .into_iter()
        .collect::<BTreeSet<_>>();

    let mut failures = Vec::new();
    let indexed_paths = datasets
        .iter()
        .map(|row| string_field(row, "source_csv"))
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>();
    let missing_from_index = source_paths
        .difference(&indexed_paths)
        .cloned()
        .collect::<Vec<_>>();
    let extra_in_index = indexed_paths
        .difference(&source_paths)
        .cloned()
        .collect::<Vec<_>>();
    if !missing_from_index.is_empty() {
        failures.push(format!(
            "Missing {} {} entries in index.",
            missing_from_index.len(),
            args.corpus_label
        ));
        for item in missing_from_index.iter().take(20) {
            failures.push(format!("- missing: {item}"));
        }
    }
    if !extra_in_index.is_empty() {
        failures.push(format!(
            "Index has {} non-existent {} entries.",
            extra_in_index.len(),
            args.corpus_label
        ));
        for item in extra_in_index.iter().take(20) {
            failures.push(format!("- extra: {item}"));
        }
    }
    if datasets.len() != source_paths.len() {
        failures.push(format!(
            "Index dataset_count mismatch: index={} source={}",
            datasets.len(),
            source_paths.len()
        ));
    }

    for row in &datasets {
        let source_csv = string_field(row, "source_csv");
        let canonical_toml = string_field(row, "canonical_toml");
        let source_path = repo_root.join(&source_csv);
        let canon_path = repo_root.join(&canonical_toml);
        if !source_path.exists() {
            failures.push(format!("{source_csv}: source CSV missing"));
            continue;
        }
        if !canon_path.exists() {
            failures.push(format!(
                "{source_csv}: canonical TOML missing at {canonical_toml}"
            ));
            continue;
        }

        let source_raw =
            fs::read(&source_path).with_context(|| format!("read {}", source_path.display()))?;
        let source_sha = sha256_hex(&source_raw);
        if source_sha != string_field(row, "source_sha256") {
            failures.push(format!("{source_csv}: source_sha256 mismatch in index"));
        }
        if args.coverage_only {
            continue;
        }

        let canon = load_toml(&canon_path)?;
        let dataset = table_value(&canon, "dataset");
        let columns = table_array(&canon, "column")?;
        let toml_header = string_list_field(&dataset, "header");
        let toml_original_header = string_list_field(&dataset, "original_header");
        let toml_rows = list_of_string_rows(dataset.get("rows"));
        let has_header = bool_field(&dataset, "has_header");
        let delimiter = single_char(&string_field(&dataset, "delimiter"), ',');
        let quotechar = single_char(&string_field(&dataset, "quotechar"), '"');

        let migrated_by = string_field(&dataset, "migrated_by");
        let (
            header,
            original_header,
            parsed_rows,
            inferred_types,
            non_empty_counts,
            empty_counts,
            expected_header_sha,
            expected_row_sha,
        ) = if migrated_by == "gororoba_cli::scrollify-csv" {
            let converted =
                scrollify_source_for_verify(&source_path, &source_csv, &canonical_toml, &dataset)?;
            let converted_dataset = converted.dataset.dataset;
            let converted_columns = converted.dataset.column;
            let converted_rows = converted_dataset
                .rows
                .clone()
                .context("scrollify verify expected inline rows from source conversion")?;
            (
                converted_dataset.header,
                converted_dataset.original_header,
                converted_rows,
                converted_columns
                    .iter()
                    .map(|column| column.inferred_type.clone())
                    .collect(),
                converted_columns
                    .iter()
                    .map(|column| column.non_empty_count)
                    .collect(),
                converted_columns
                    .iter()
                    .map(|column| column.empty_count)
                    .collect(),
                converted_dataset.header_value_sha256,
                converted_dataset.row_value_sha256,
            )
        } else if let Ok(existing_dataset) = load_existing_dataset(repo_root, &canonical_toml) {
            if existing_dataset.source_csv == source_csv
                && existing_dataset.source_sha256 == source_sha
            {
                let expected_header_sha = sha_text_json(&existing_dataset.header)?;
                let expected_row_sha = sha_text_json(&existing_dataset.rows)?;
                (
                    existing_dataset.header.clone(),
                    existing_dataset.original_header.clone(),
                    existing_dataset.rows.clone(),
                    existing_dataset.column_types.clone(),
                    existing_dataset.non_empty_counts.clone(),
                    existing_dataset.empty_counts.clone(),
                    expected_header_sha,
                    expected_row_sha,
                )
            } else {
                let (
                    header,
                    original_header,
                    parsed_rows,
                    inferred_types,
                    non_empty_counts,
                    empty_counts,
                ) = parse_source_for_verify(&source_path, has_header, delimiter, quotechar)?;
                let expected_header_sha = sha_text_json(&header)?;
                let expected_row_sha = sha_text_json(&parsed_rows)?;
                (
                    header,
                    original_header,
                    parsed_rows,
                    inferred_types,
                    non_empty_counts,
                    empty_counts,
                    expected_header_sha,
                    expected_row_sha,
                )
            }
        } else {
            let (
                header,
                original_header,
                parsed_rows,
                inferred_types,
                non_empty_counts,
                empty_counts,
            ) = parse_source_for_verify(&source_path, has_header, delimiter, quotechar)?;
            let expected_header_sha = sha_text_json(&header)?;
            let expected_row_sha = sha_text_json(&parsed_rows)?;
            (
                header,
                original_header,
                parsed_rows,
                inferred_types,
                non_empty_counts,
                empty_counts,
                expected_header_sha,
                expected_row_sha,
            )
        };

        if header != toml_header {
            failures.push(format!("{source_csv}: header mismatch"));
        }
        if original_header != toml_original_header {
            failures.push(format!("{source_csv}: original_header mismatch"));
        }
        let toml_row_sha = sha_text_json(&toml_rows)?;
        if migrated_by != "gororoba_cli::scrollify-csv"
            && parsed_rows != toml_rows
            && expected_row_sha != toml_row_sha
        {
            failures.push(format!("{source_csv}: row payload mismatch"));
        }
        if expected_header_sha != string_field(&dataset, "header_value_sha256") {
            failures.push(format!("{source_csv}: dataset header checksum mismatch"));
        }
        if expected_row_sha != string_field(&dataset, "row_value_sha256") {
            failures.push(format!("{source_csv}: dataset row checksum mismatch"));
        }
        if expected_header_sha != string_field(row, "header_value_sha256") {
            failures.push(format!("{source_csv}: index header checksum mismatch"));
        }
        if expected_row_sha != string_field(row, "row_value_sha256") {
            failures.push(format!("{source_csv}: index row checksum mismatch"));
        }
        if integer_field(&dataset, "row_count", -1) != parsed_rows.len() as i64 {
            failures.push(format!("{source_csv}: row_count mismatch"));
        }
        if integer_field(&dataset, "column_count", -1) != header.len() as i64 {
            failures.push(format!("{source_csv}: column_count mismatch"));
        }
        if columns.len() != header.len() {
            failures.push(format!("{source_csv}: column profile count mismatch"));
        } else {
            for (idx, column) in columns.iter().enumerate() {
                if string_field(column, "name") != header[idx] {
                    failures.push(format!(
                        "{source_csv}: column name mismatch at index {}",
                        idx + 1
                    ));
                }
                if string_field(column, "inferred_type") != inferred_types[idx] {
                    failures.push(format!(
                        "{source_csv}: inferred_type mismatch at index {}",
                        idx + 1
                    ));
                }
                if integer_field(column, "non_empty_count", -1) != non_empty_counts[idx] as i64 {
                    failures.push(format!(
                        "{source_csv}: non_empty_count mismatch at index {}",
                        idx + 1
                    ));
                }
                if integer_field(column, "empty_count", -1) != empty_counts[idx] as i64 {
                    failures.push(format!(
                        "{source_csv}: empty_count mismatch at index {}",
                        idx + 1
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        println!(
            "ERROR: {} TOML parity verification failed.",
            args.corpus_label
        );
        for item in failures.iter().take(300) {
            println!("- {item}");
        }
        if failures.len() > 300 {
            println!("- ... and {} more failures", failures.len() - 300);
        }
        std::process::exit(1);
    }

    println!(
        "OK: {} corpus is fully represented in canonical TOML with semantic parity. datasets={}",
        args.corpus_label,
        datasets.len()
    );
    Ok(())
}

fn run_project_split_policy(repo_root: &Path, args: &ProjectSplitPolicyArgs) -> Result<()> {
    let inventory = load_toml(&repo_root.join(&args.inventory))?;
    let mut rows = table_array(&inventory, "document")?
        .into_iter()
        .filter(|row| string_field(row, "zone") == "project_csv")
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| string_field(row, "path"));

    let explicit_generated = extract_generated_explicit_paths(
        &repo_root.join("src/verification/verify_generated_artifacts.py"),
    )?;

    let mut policy_rows = Vec::new();
    let mut canonical_paths = Vec::new();
    let mut generated_paths = Vec::new();
    for row in rows {
        let path = string_field(&row, "path");
        let generated_by_glob = MAKEFILE_GENERATED_GLOBS.iter().any(|pattern| {
            glob::Pattern::new(pattern)
                .map(|p| p.matches(&path))
                .unwrap_or(false)
        });
        let generated_by_explicit = explicit_generated.contains(&path);
        let is_generated = generated_by_glob || generated_by_explicit;
        let (classification, rationale, evidence_refs) = if is_generated {
            generated_paths.push(path.clone());
            (
                "generated_artifact".to_string(),
                "Matched generated artifact contract via Makefile clean-artifacts patterns and/or verify_generated_artifacts expectations.".to_string(),
                vec!["Makefile:357".to_string(), "src/verification/verify_generated_artifacts.py:37".to_string()],
            )
        } else {
            canonical_paths.push(path.clone());
            (
                "canonical_dataset".to_string(),
                "Not listed in generated artifact contracts; treat as canonical project dataset for TOML-first scroll conversion.".to_string(),
                vec!["Makefile:357".to_string(), "src/verification/verify_generated_artifacts.py:37".to_string()],
            )
        };
        policy_rows.push(BTreeMap::from([
            ("path".to_string(), path),
            ("classification".to_string(), classification),
            (
                "queue_for_scroll_conversion".to_string(),
                "true".to_string(),
            ),
            (
                "size_bytes".to_string(),
                integer_field(&row, "size_bytes", 0).to_string(),
            ),
            ("source_sha256".to_string(), string_field(&row, "sha256")),
            ("git_status".to_string(), string_field(&row, "git_status")),
            ("rationale".to_string(), rationale),
            ("evidence_refs".to_string(), evidence_refs.join("\u{1f}")),
        ]));
    }

    let out_path = repo_root.join(&args.out);
    write_ascii(
        &out_path,
        &(render_project_split_policy(&policy_rows, generated_paths.len(), canonical_paths.len())?
            + "\n"),
    )?;
    write_manifest(&repo_root.join(&args.canonical_manifest), &canonical_paths)?;
    write_manifest(&repo_root.join(&args.generated_manifest), &generated_paths)?;
    println!(
        "Wrote {} with {} project_csv records (canonical={}, generated={}).",
        out_path.display(),
        policy_rows.len(),
        canonical_paths.len(),
        generated_paths.len()
    );
    println!(
        "Wrote canonical manifest: {}",
        repo_root.join(&args.canonical_manifest).display()
    );
    println!(
        "Wrote generated manifest: {}",
        repo_root.join(&args.generated_manifest).display()
    );
    Ok(())
}

fn run_holdings(repo_root: &Path, args: &HoldingsArgs) -> Result<()> {
    let inventory = load_toml(&repo_root.join(&args.inventory))?;
    let mut external_rows = Vec::new();
    let mut archive_rows = Vec::new();

    for row in table_array(&inventory, "document")? {
        let zone = string_field(&row, "zone");
        let payload = HoldingRow {
            path: string_field(&row, "path"),
            source_sha256: string_field(&row, "sha256"),
            size_bytes: integer_field(&row, "size_bytes", 0) as usize,
            git_status: string_field(&row, "git_status"),
            target_lane: String::new(),
        };
        if zone == "external_csv" {
            external_rows.push(HoldingRow {
                target_lane: "external_csv_holding".to_string(),
                ..payload
            });
        } else if zone == "archive_csv" {
            archive_rows.push(HoldingRow {
                target_lane: "archive_csv_holding".to_string(),
                ..payload
            });
        }
    }

    external_rows.sort_by(|a, b| a.path.cmp(&b.path));
    archive_rows.sort_by(|a, b| a.path.cmp(&b.path));

    let external_out = repo_root.join(&args.external_out);
    let archive_out = repo_root.join(&args.archive_out);
    write_ascii(
        &external_out,
        &(render_holding_registry("external_csv_holding", "external", &external_rows)? + "\n"),
    )?;
    write_ascii(
        &archive_out,
        &(render_holding_registry("archive_csv_holding", "archive", &archive_rows)? + "\n"),
    )?;
    write_manifest(
        &repo_root.join(&args.external_manifest),
        &external_rows
            .iter()
            .map(|row| row.path.clone())
            .collect::<Vec<_>>(),
    )?;
    write_manifest(
        &repo_root.join(&args.archive_manifest),
        &archive_rows
            .iter()
            .map(|row| row.path.clone())
            .collect::<Vec<_>>(),
    )?;
    println!(
        "Wrote external holding registry: {} ({} records)",
        external_out.display(),
        external_rows.len()
    );
    println!(
        "Wrote archive holding registry: {} ({} records)",
        archive_out.display(),
        archive_rows.len()
    );
    println!(
        "Wrote external manifest: {}",
        repo_root.join(&args.external_manifest).display()
    );
    println!(
        "Wrote archive manifest: {}",
        repo_root.join(&args.archive_manifest).display()
    );
    Ok(())
}

fn run_verify_holdings(repo_root: &Path, args: &VerifyHoldingsArgs) -> Result<()> {
    let inventory = load_toml(&repo_root.join(&args.inventory))?;
    let external_registry = load_toml(&repo_root.join(&args.external_registry))?;
    let archive_registry = load_toml(&repo_root.join(&args.archive_registry))?;

    let external_inventory_paths = zone_path_set(&inventory, "external_csv");
    let archive_inventory_paths = zone_path_set(&inventory, "archive_csv");
    let external_registry_paths = path_set_from_table_array(&external_registry, "dataset", "path")?;
    let archive_registry_paths = path_set_from_table_array(&archive_registry, "dataset", "path")?;
    let external_manifest_paths = load_manifest_set(&repo_root.join(&args.external_manifest))?;
    let archive_manifest_paths = load_manifest_set(&repo_root.join(&args.archive_manifest))?;

    let mut failures = Vec::new();
    if external_registry_paths != external_inventory_paths {
        failures.push(
            "External holding registry paths do not match external_csv inventory set.".to_string(),
        );
    }
    if archive_registry_paths != archive_inventory_paths {
        failures.push(
            "Archive holding registry paths do not match archive_csv inventory set.".to_string(),
        );
    }
    if external_manifest_paths != external_inventory_paths {
        failures.push(
            "External holding manifest does not match external_csv inventory set.".to_string(),
        );
    }
    if archive_manifest_paths != archive_inventory_paths {
        failures
            .push("Archive holding manifest does not match archive_csv inventory set.".to_string());
    }
    for row in table_array(&external_registry, "dataset")? {
        if string_field(&row, "hold_status") != "queued_for_scroll_conversion" {
            failures.push(format!(
                "{}: external hold_status mismatch",
                string_field(&row, "path")
            ));
        }
    }
    for row in table_array(&archive_registry, "dataset")? {
        if string_field(&row, "hold_status") != "queued_for_scroll_conversion" {
            failures.push(format!(
                "{}: archive hold_status mismatch",
                string_field(&row, "path")
            ));
        }
    }

    if !failures.is_empty() {
        println!("ERROR: CSV holding registry verification failed.");
        for item in failures.iter().take(200) {
            println!("- {item}");
        }
        if failures.len() > 200 {
            println!("- ... and {} more failures", failures.len() - 200);
        }
        std::process::exit(1);
    }

    println!(
        "OK: CSV holding registries verified. external={} archive={}",
        external_inventory_paths.len(),
        archive_inventory_paths.len()
    );
    Ok(())
}

fn run_scroll_pipeline(repo_root: &Path, args: &ScrollPipelineArgs) -> Result<()> {
    let lane_specs = [
        (
            "project_canonical",
            "registry/project_csv_canonical_datasets.toml",
            "project_csv_canonical_datasets",
            "canonical_dataset",
        ),
        (
            "project_generated",
            "registry/project_csv_generated_artifacts.toml",
            "project_csv_generated_artifacts",
            "generated_artifact",
        ),
        (
            "external_holding",
            "registry/external_csv_holding_datasets.toml",
            "external_csv_holding_datasets",
            "holding_external_csv",
        ),
        (
            "archive_holding",
            "registry/archive_csv_holding_datasets.toml",
            "archive_csv_holding_datasets",
            "holding_archive_csv",
        ),
    ];

    let mut lanes = Vec::new();
    let mut refs = Vec::new();
    for (lane_name, registry_path, table_name, dataset_class) in lane_specs {
        let registry = load_toml(&repo_root.join(registry_path))?;
        let section = table_value(&registry, table_name);
        let datasets = table_array(&registry, "dataset")?;
        let source_descriptor = string_field(&section, "source_descriptor");
        let manifest_path = manifest_path_from_descriptor(&source_descriptor);
        let canonical_dir = string_field(&section, "canonical_dir");

        lanes.push(BTreeMap::from([
            ("name".to_string(), lane_name.to_string()),
            ("source_registry".to_string(), registry_path.to_string()),
            ("source_table".to_string(), table_name.to_string()),
            ("dataset_class".to_string(), dataset_class.to_string()),
            ("source_descriptor".to_string(), source_descriptor),
            ("manifest_path".to_string(), manifest_path),
            ("canonical_dir".to_string(), canonical_dir),
            ("dataset_count".to_string(), datasets.len().to_string()),
        ]));

        for row in datasets {
            refs.push(BTreeMap::from([
                ("lane_name".to_string(), lane_name.to_string()),
                ("dataset_id".to_string(), string_field(&row, "id")),
                ("slug".to_string(), string_field(&row, "slug")),
                (
                    "dataset_class".to_string(),
                    string_field(&row, "dataset_class").if_empty(dataset_class),
                ),
                ("source_csv".to_string(), string_field(&row, "source_csv")),
                (
                    "canonical_toml".to_string(),
                    string_field(&row, "canonical_toml"),
                ),
                (
                    "source_sha256".to_string(),
                    string_field(&row, "source_sha256"),
                ),
                (
                    "row_count".to_string(),
                    integer_field(&row, "row_count", 0).to_string(),
                ),
                (
                    "column_count".to_string(),
                    integer_field(&row, "column_count", 0).to_string(),
                ),
            ]));
        }
    }
    lanes.sort_by_key(|row| row.get("name").cloned().unwrap_or_default());
    refs.sort_by_key(|row| {
        (
            row.get("lane_name").cloned().unwrap_or_default(),
            row.get("dataset_id").cloned().unwrap_or_default(),
            row.get("source_csv").cloned().unwrap_or_default(),
        )
    });

    let out_path = repo_root.join(&args.out);
    write_ascii(&out_path, &(render_scroll_pipeline(&lanes, &refs)? + "\n"))?;
    println!(
        "Wrote csv scroll pipeline registry: {} (lanes={} dataset_refs={})",
        out_path.display(),
        lanes.len(),
        refs.len()
    );
    Ok(())
}

fn run_verify_scroll_pipeline(repo_root: &Path, args: &VerifyScrollPipelineArgs) -> Result<()> {
    let pipeline_path = repo_root.join(&args.pipeline);
    if !pipeline_path.exists() {
        bail!(
            "ERROR: missing pipeline registry: {}",
            pipeline_path.display()
        );
    }
    let pipeline_text = fs::read_to_string(&pipeline_path)
        .with_context(|| format!("read {}", pipeline_path.display()))?;
    assert_ascii(&pipeline_text, &pipeline_path.display().to_string())?;
    let pipeline = parse_toml_text(&pipeline_text, &pipeline_path.display().to_string())?;
    let section = table_value(&pipeline, "csv_scroll_pipeline");
    let lanes = table_array(&pipeline, "lane")?;
    let refs = table_array(&pipeline, "dataset_ref")?;
    let expected_lanes = BTreeSet::from([
        "archive_holding".to_string(),
        "external_holding".to_string(),
        "project_canonical".to_string(),
        "project_generated".to_string(),
    ]);

    let mut failures = Vec::new();
    if integer_field(&section, "lane_count", -1) != lanes.len() as i64 {
        failures.push("lane_count metadata mismatch.".to_string());
    }
    if integer_field(&section, "dataset_total", -1) != refs.len() as i64 {
        failures.push("dataset_total metadata mismatch.".to_string());
    }

    let lane_names = lanes
        .iter()
        .map(|row| string_field(row, "name"))
        .collect::<BTreeSet<_>>();
    if lane_names != expected_lanes {
        failures.push(format!(
            "lane name mismatch: expected={:?} got={:?}",
            expected_lanes, lane_names
        ));
    }

    let mut refs_by_lane = BTreeMap::new();
    for row in &refs {
        *refs_by_lane
            .entry(string_field(row, "lane_name"))
            .or_insert(0usize) += 1;
    }

    for lane in &lanes {
        let lane_name = string_field(lane, "name");
        let source_registry = string_field(lane, "source_registry");
        let source_table = string_field(lane, "source_table");
        let declared_count = integer_field(lane, "dataset_count", -1);
        let manifest_path = string_field(lane, "manifest_path");
        let canonical_dir = string_field(lane, "canonical_dir");

        let source_abs = repo_root.join(&source_registry);
        if !source_abs.exists() {
            failures.push(format!(
                "missing source registry for lane {}: {}",
                lane_name, source_registry
            ));
            continue;
        }
        let source_raw = load_toml(&source_abs)?;
        let source_section = table_value(&source_raw, &source_table);
        let source_datasets = table_array(&source_raw, "dataset")?;
        if declared_count != source_datasets.len() as i64 {
            failures.push(format!(
                "lane {}: dataset_count mismatch {} != {}",
                lane_name,
                declared_count,
                source_datasets.len()
            ));
        }
        let ref_count = refs_by_lane.get(&lane_name).copied().unwrap_or_default();
        if ref_count != source_datasets.len() {
            failures.push(format!(
                "lane {}: dataset_ref mismatch {} != {}",
                lane_name,
                ref_count,
                source_datasets.len()
            ));
        }

        let expected_manifest =
            manifest_path_from_descriptor(&string_field(&source_section, "source_descriptor"));
        if manifest_path != expected_manifest {
            failures.push(format!(
                "lane {}: manifest path mismatch {:?} != {:?}",
                lane_name, manifest_path, expected_manifest
            ));
        }
        if !manifest_path.is_empty() && !repo_root.join(&manifest_path).exists() {
            failures.push(format!(
                "lane {}: missing manifest file {}",
                lane_name, manifest_path
            ));
        }

        let expected_dir = string_field(&source_section, "canonical_dir");
        if canonical_dir != expected_dir {
            failures.push(format!(
                "lane {}: canonical_dir mismatch {:?} != {:?}",
                lane_name, canonical_dir, expected_dir
            ));
        }
        if !canonical_dir.is_empty() && !repo_root.join(&canonical_dir).exists() {
            failures.push(format!(
                "lane {}: missing canonical dir {}",
                lane_name, canonical_dir
            ));
        }
    }

    let mut seen_ref_ids = BTreeSet::new();
    for row in &refs {
        let ref_id = string_field(row, "id");
        if !seen_ref_ids.insert(ref_id.clone()) {
            failures.push(format!("duplicate dataset_ref id: {}", ref_id));
            continue;
        }
        let source_csv = string_field(row, "source_csv");
        let canonical_toml = string_field(row, "canonical_toml");
        if source_csv.is_empty() {
            failures.push(format!("{}: missing source_csv", ref_id));
        } else if !repo_root.join(&source_csv).exists() {
            failures.push(format!(
                "{}: missing source_csv file {}",
                ref_id, source_csv
            ));
        }
        if canonical_toml.is_empty() {
            failures.push(format!("{}: missing canonical_toml", ref_id));
        } else if !repo_root.join(&canonical_toml).exists() {
            failures.push(format!(
                "{}: missing canonical_toml file {}",
                ref_id, canonical_toml
            ));
        }
    }

    if !failures.is_empty() {
        println!("ERROR: csv scroll pipeline verification failed.");
        for item in failures.iter().take(200) {
            println!("- {item}");
        }
        if failures.len() > 200 {
            println!("- ... and {} more failures", failures.len() - 200);
        }
        std::process::exit(1);
    }

    println!(
        "OK: csv scroll pipeline verified. lanes={} dataset_refs={}",
        lanes.len(),
        refs.len()
    );
    Ok(())
}

fn run_verify_corpus_coverage(repo_root: &Path, args: &VerifyCorpusCoverageArgs) -> Result<()> {
    let inventory = load_toml(&repo_root.join(&args.inventory))?;
    let project_zone = zone_path_set(&inventory, "project_csv");
    let legacy_zone = zone_path_set(&inventory, "legacy_csv");
    let curated_zone = zone_path_set(&inventory, "curated_csv");
    let external_zone = zone_path_set(&inventory, "external_csv");
    let archive_zone = zone_path_set(&inventory, "archive_csv");

    let project_canonical = load_source_csv_set(&repo_root.join(&args.project_canonical_index))?;
    let project_generated = load_source_csv_set(&repo_root.join(&args.project_generated_index))?;
    let legacy_index = load_source_csv_set(&repo_root.join(&args.legacy_index))?;
    let curated_index = load_source_csv_set(&repo_root.join(&args.curated_index))?;
    let external_index = load_source_csv_set(&repo_root.join(&args.external_holding_index))?;
    let archive_index = load_source_csv_set(&repo_root.join(&args.archive_holding_index))?;

    let mut failures = Vec::new();
    summarize_set_mismatch(
        &mut failures,
        "project_csv",
        &project_zone,
        &(project_canonical
            .union(&project_generated)
            .cloned()
            .collect()),
    );
    summarize_set_mismatch(&mut failures, "legacy_csv", &legacy_zone, &legacy_index);
    summarize_set_mismatch(&mut failures, "curated_csv", &curated_zone, &curated_index);
    summarize_set_mismatch(
        &mut failures,
        "external_csv",
        &external_zone,
        &external_index,
    );
    summarize_set_mismatch(&mut failures, "archive_csv", &archive_zone, &archive_index);

    let manual_triage = table_array(&inventory, "document")?
        .into_iter()
        .filter(|row| {
            matches!(
                string_field(row, "zone").as_str(),
                "project_csv" | "legacy_csv" | "curated_csv" | "external_csv" | "archive_csv"
            ) && string_field(row, "migration_action") == "manual_triage"
        })
        .map(|row| string_field(&row, "path"))
        .collect::<Vec<_>>();
    if !manual_triage.is_empty() {
        failures.push(format!(
            "in-scope CSV entries still marked manual_triage: {}",
            manual_triage.len()
        ));
        for item in manual_triage.iter().take(20) {
            failures.push(format!("- manual_triage: {item}"));
        }
    }

    if !failures.is_empty() {
        println!("ERROR: CSV corpus coverage verification failed.");
        for item in failures.iter().take(200) {
            println!("{item}");
        }
        if failures.len() > 200 {
            println!("... and {} more failures", failures.len() - 200);
        }
        std::process::exit(1);
    }

    println!(
        "OK: CSV corpus coverage verified. project={} legacy={} curated={} external={} archive={}",
        project_zone.len(),
        legacy_zone.len(),
        curated_zone.len(),
        external_zone.len(),
        archive_zone.len()
    );
    Ok(())
}

fn run_migration_scope(repo_root: &Path, args: &MigrationScopeArgs) -> Result<()> {
    let inventory = load_toml(&repo_root.join(&args.inventory))?;
    let docs = table_array(&inventory, "document")?;
    let mut zone_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut action_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut priority_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut high_priority_paths = Vec::new();
    let mut medium_priority_paths = Vec::new();

    for row in &docs {
        let zone = string_field(row, "zone").if_empty("unknown");
        let action = string_field(row, "migration_action").if_empty("manual_triage");
        let priority = string_field(row, "migration_priority").if_empty("medium");
        let path = string_field(row, "path");
        *zone_counts.entry(zone).or_insert(0) += 1;
        *action_counts.entry(action).or_insert(0) += 1;
        *priority_counts.entry(priority.clone()).or_insert(0) += 1;
        if priority == "critical" || priority == "high" {
            high_priority_paths.push(path);
        } else if priority == "medium" {
            medium_priority_paths.push(path);
        }
    }
    high_priority_paths.sort();
    medium_priority_paths.sort();

    let zone_count = |key: &str| -> usize { zone_counts.get(key).copied().unwrap_or_default() };
    let legacy_total = zone_count("legacy_csv");
    let legacy_done = docs
        .iter()
        .filter(|row| {
            string_field(row, "zone") == "legacy_csv"
                && string_field(row, "migration_action") == "canonicalized_to_toml"
        })
        .count();
    let curated_total = zone_count("curated_csv");
    let curated_done = docs
        .iter()
        .filter(|row| {
            string_field(row, "zone") == "curated_csv"
                && string_field(row, "migration_action") == "canonicalized_to_toml"
        })
        .count();
    let project_total = zone_count("project_csv");
    let project_done = docs
        .iter()
        .filter(|row| {
            string_field(row, "zone") == "project_csv"
                && matches!(
                    string_field(row, "migration_action").as_str(),
                    "canonicalized_to_toml" | "canonicalized_to_toml_generated_artifact"
                )
        })
        .count();
    let external_total = zone_count("external_csv");
    let archive_total = zone_count("archive_csv");
    let holding_total = external_total + archive_total;
    let holding_done = docs
        .iter()
        .filter(|row| {
            matches!(
                string_field(row, "zone").as_str(),
                "external_csv" | "archive_csv"
            ) && string_field(row, "migration_action") == "canonicalized_to_toml_holding"
        })
        .count();

    let out_path = repo_root.join(&args.out);
    write_ascii(
        &out_path,
        &(render_migration_scope(
            &args.inventory.to_string_lossy(),
            docs.len(),
            &zone_counts,
            &action_counts,
            &priority_counts,
            &high_priority_paths,
            &medium_priority_paths,
            &[
                (
                    "wave_1",
                    wave_state(
                        legacy_total,
                        legacy_done,
                        "migrate_to_toml_canonical",
                        "legacy_csv canonicalized",
                    ),
                ),
                (
                    "wave_2",
                    wave_state(
                        curated_total,
                        curated_done,
                        "plan_curated_ingest",
                        "curated_csv canonicalized",
                    ),
                ),
                (
                    "wave_3",
                    wave_state(
                        project_total,
                        project_done,
                        "project_csv split/migration",
                        "project_csv split-and-scroll complete",
                    ),
                ),
                (
                    "wave_4",
                    wave_state(
                        holding_total,
                        holding_done,
                        "external/archive holding conversion",
                        "external+archive holding scrollified",
                    ),
                ),
            ],
        )? + "\n"),
    )?;
    println!(
        "Wrote {} with scope summary from {} CSV records.",
        out_path.display(),
        docs.len()
    );
    Ok(())
}

fn render_csv_inventory(docs: &[CsvDoc]) -> Result<String> {
    let tracked_count = docs.iter().filter(|d| d.git_status == "tracked").count();
    let untracked_count = docs.iter().filter(|d| d.git_status == "untracked").count();
    let ignored_count = docs.iter().filter(|d| d.git_status == "ignored").count();
    let archived_count = docs.iter().filter(|d| d.archived).count();
    let legacy_count = docs.iter().filter(|d| d.zone == "legacy_csv").count();
    let curated_count = docs.iter().filter(|d| d.zone == "curated_csv").count();
    let canonicalized_count = docs
        .iter()
        .filter(|d| d.migration_action == "canonicalized_to_toml")
        .count();
    let generated_scroll_count = docs
        .iter()
        .filter(|d| d.migration_action == "canonicalized_to_toml_generated_artifact")
        .count();
    let holding_scroll_count = docs
        .iter()
        .filter(|d| d.migration_action == "canonicalized_to_toml_holding")
        .count();
    let holding_queue_count = docs
        .iter()
        .filter(|d| d.migration_action == "queued_for_scroll_holding")
        .count();

    let mut lines = vec![
        "# Full CSV inventory registry (tracked/untracked/ignored/archived).".to_string(),
        format!("# Generated by {BIN_PATH}"),
        String::new(),
        "[csv_inventory]".to_string(),
        format!("updated = {}", q(UPDATED_STAMP)),
        "authoritative = true".to_string(),
        format!("document_count = {}", docs.len()),
        format!("tracked_count = {tracked_count}"),
        format!("untracked_count = {untracked_count}"),
        format!("ignored_count = {ignored_count}"),
        format!("archived_count = {archived_count}"),
        format!("legacy_count = {legacy_count}"),
        format!("curated_count = {curated_count}"),
        format!("canonicalized_count = {canonicalized_count}"),
        format!("generated_scroll_count = {generated_scroll_count}"),
        format!("holding_scroll_count = {holding_scroll_count}"),
        format!("holding_queue_count = {holding_queue_count}"),
        String::new(),
    ];
    for doc in docs {
        lines.push("[[document]]".to_string());
        lines.push(format!("path = {}", q(&doc.path)));
        lines.push(format!("git_status = {}", q(&doc.git_status)));
        lines.push(format!("zone = {}", q(&doc.zone)));
        lines.push(format!("archived = {}", doc.archived));
        lines.push(format!("generated = {}", doc.generated));
        lines.push(format!("size_bytes = {}", doc.size_bytes));
        lines.push(format!("line_count = {}", doc.line_count));
        lines.push(format!("sha256 = {}", q(&doc.sha256)));
        lines.push(format!("migration_action = {}", q(&doc.migration_action)));
        lines.push(format!(
            "migration_priority = {}",
            q(&doc.migration_priority)
        ));
        lines.push(format!("rationale = {}", q(&doc.rationale)));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_dataset(dataset: &Dataset, corpus_label: &str) -> Result<String> {
    let mut lines = vec![
        format!("# Canonical TOML dataset migrated from {corpus_label}."),
        format!("# Generated by {BIN_PATH}"),
        String::new(),
        "[dataset]".to_string(),
        format!("id = {}", q(&dataset.dataset_id)),
        format!("slug = {}", q(&dataset.slug)),
        format!("source_csv = {}", q(&dataset.source_csv)),
        format!("source_sha256 = {}", q(&dataset.source_sha256)),
        format!("source_size_bytes = {}", dataset.source_size_bytes),
        format!("has_header = {}", dataset.has_header),
        format!("delimiter = {}", q(&dataset.delimiter.to_string())),
        format!("quotechar = {}", q(&dataset.quotechar.to_string())),
        format!("row_count = {}", dataset.row_count),
        format!("column_count = {}", dataset.column_count),
        format!("header_value_sha256 = {}", q(&dataset.header_value_sha256)),
        format!("row_value_sha256 = {}", q(&dataset.row_value_sha256)),
        format!("migrated_on = {}", q(UPDATED_STAMP)),
        format!("migrated_by = {}", q(BIN_PATH)),
        String::new(),
        "header = [".to_string(),
    ];
    for token in &dataset.header {
        lines.push(format!("  {},", q(token)));
    }
    lines.push("]".to_string());
    lines.push(String::new());
    lines.push("original_header = [".to_string());
    for token in &dataset.original_header {
        lines.push(format!("  {},", q(token)));
    }
    lines.push("]".to_string());
    lines.push(String::new());
    lines.push("rows = [".to_string());
    for row in &dataset.rows {
        let encoded = row
            .iter()
            .map(|value| q(value))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("  [{encoded}],"));
    }
    lines.push("]".to_string());
    lines.push(String::new());
    for (idx, name) in dataset.header.iter().enumerate() {
        let original_name = dataset
            .original_header
            .get(idx)
            .cloned()
            .unwrap_or_default();
        lines.push("[[column]]".to_string());
        lines.push(format!("index = {}", idx + 1));
        lines.push(format!("name = {}", q(name)));
        lines.push(format!("original_name = {}", q(&original_name)));
        lines.push(format!("inferred_type = {}", q(&dataset.column_types[idx])));
        lines.push(format!(
            "non_empty_count = {}",
            dataset.non_empty_counts[idx]
        ));
        lines.push(format!("empty_count = {}", dataset.empty_counts[idx]));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_dataset_index(
    datasets: &[Dataset],
    index_table: &str,
    source_glob: &str,
    canonical_dir: &str,
    corpus_label: &str,
) -> Result<String> {
    let mut lines = vec![
        format!("# Canonical index for {corpus_label} datasets migrated to TOML."),
        format!("# Generated by {BIN_PATH}"),
        String::new(),
        format!("[{index_table}]"),
        format!("updated = {}", q(UPDATED_STAMP)),
        "authoritative = true".to_string(),
        format!("source_glob = {}", q(source_glob)),
        format!("canonical_dir = {}", q(canonical_dir)),
        format!("dataset_count = {}", datasets.len()),
        String::new(),
    ];
    for dataset in datasets {
        lines.push("[[dataset]]".to_string());
        lines.push(format!("id = {}", q(&dataset.dataset_id)));
        lines.push(format!("slug = {}", q(&dataset.slug)));
        lines.push(format!("source_csv = {}", q(&dataset.source_csv)));
        lines.push(format!("source_sha256 = {}", q(&dataset.source_sha256)));
        lines.push(format!("source_size_bytes = {}", dataset.source_size_bytes));
        lines.push(format!("canonical_toml = {}", q(&dataset.canonical_toml)));
        lines.push(format!("row_count = {}", dataset.row_count));
        lines.push(format!("column_count = {}", dataset.column_count));
        lines.push(format!("has_header = {}", dataset.has_header));
        lines.push(format!("delimiter = {}", q(&dataset.delimiter.to_string())));
        lines.push(format!("quotechar = {}", q(&dataset.quotechar.to_string())));
        lines.push(format!(
            "header_value_sha256 = {}",
            q(&dataset.header_value_sha256)
        ));
        lines.push(format!(
            "row_value_sha256 = {}",
            q(&dataset.row_value_sha256)
        ));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_project_split_policy(
    rows: &[BTreeMap<String, String>],
    generated_count: usize,
    canonical_count: usize,
) -> Result<String> {
    let mut lines = vec![
        "# project_csv split policy registry (TOML-first).".to_string(),
        format!("# Generated by {BIN_PATH}"),
        String::new(),
        "[project_csv_split_policy]".to_string(),
        format!("updated = {}", q(UPDATED_STAMP)),
        "authoritative = true".to_string(),
        "source_inventory = \"registry/csv_inventory.toml\"".to_string(),
        format!("dataset_count = {}", rows.len()),
        format!("canonical_dataset_count = {canonical_count}"),
        format!("generated_artifact_count = {generated_count}"),
        String::new(),
        "generated_evidence_refs = [".to_string(),
        "  \"Makefile:357\",".to_string(),
        "  \"src/verification/verify_generated_artifacts.py:37\",".to_string(),
        "]".to_string(),
        String::new(),
    ];
    for row in rows {
        lines.push("[[dataset]]".to_string());
        lines.push(format!(
            "path = {}",
            q(row.get("path").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "classification = {}",
            q(row.get("classification").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "queue_for_scroll_conversion = {}",
            row.get("queue_for_scroll_conversion")
                .map(|v| v == "true")
                .unwrap_or(false)
        ));
        lines.push(format!(
            "size_bytes = {}",
            row.get("size_bytes")
                .cloned()
                .unwrap_or_else(|| "0".to_string())
        ));
        lines.push(format!(
            "source_sha256 = {}",
            q(row.get("source_sha256").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "git_status = {}",
            q(row.get("git_status").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "rationale = {}",
            q(row.get("rationale").unwrap_or(&String::new()))
        ));
        lines.push("evidence_refs = [".to_string());
        for reference in row
            .get("evidence_refs")
            .cloned()
            .unwrap_or_default()
            .split('\u{1f}')
            .filter(|value| !value.is_empty())
        {
            lines.push(format!("  {},", q(reference)));
        }
        lines.push("]".to_string());
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_holding_registry(
    table_name: &str,
    lane_label: &str,
    rows: &[HoldingRow],
) -> Result<String> {
    let mut lines = vec![
        format!("# Holding registry for {lane_label} CSV ingestion into TOML scrolls."),
        format!("# Generated by {BIN_PATH}"),
        String::new(),
        format!("[{table_name}]"),
        format!("updated = {}", q(UPDATED_STAMP)),
        "authoritative = true".to_string(),
        "queue_status = \"active\"".to_string(),
        format!("dataset_count = {}", rows.len()),
        String::new(),
    ];
    for row in rows {
        lines.push("[[dataset]]".to_string());
        lines.push(format!("path = {}", q(&row.path)));
        lines.push(format!("source_sha256 = {}", q(&row.source_sha256)));
        lines.push(format!("size_bytes = {}", row.size_bytes));
        lines.push(format!("git_status = {}", q(&row.git_status)));
        lines.push("hold_status = \"queued_for_scroll_conversion\"".to_string());
        lines.push(format!("target_lane = {}", q(&row.target_lane)));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_scroll_pipeline(
    lanes: &[BTreeMap<String, String>],
    refs: &[BTreeMap<String, String>],
) -> Result<String> {
    let mut lines = vec![
        "# Unified CSV scroll pipeline control-plane registry.".to_string(),
        format!("# Generated by {BIN_PATH}"),
        String::new(),
        "[csv_scroll_pipeline]".to_string(),
        format!("updated = {}", q(UPDATED_STAMP)),
        "authoritative = true".to_string(),
        format!("lane_count = {}", lanes.len()),
        format!("dataset_total = {}", refs.len()),
        "policy = \"All in-scope CSV corpora must flow through canonical/generated/holding TOML scroll lanes.\"".to_string(),
        String::new(),
    ];
    for (index, lane) in lanes.iter().enumerate() {
        lines.push("[[lane]]".to_string());
        lines.push(format!("id = {}", q(&format!("CSP-LANE-{:04}", index + 1))));
        lines.push(format!(
            "name = {}",
            q(lane.get("name").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "source_registry = {}",
            q(lane.get("source_registry").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "source_table = {}",
            q(lane.get("source_table").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "dataset_class = {}",
            q(lane.get("dataset_class").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "source_descriptor = {}",
            q(lane.get("source_descriptor").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "manifest_path = {}",
            q(lane.get("manifest_path").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "canonical_dir = {}",
            q(lane.get("canonical_dir").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "dataset_count = {}",
            lane.get("dataset_count").cloned().unwrap_or_default()
        ));
        lines.push(String::new());
    }
    for (index, row) in refs.iter().enumerate() {
        lines.push("[[dataset_ref]]".to_string());
        lines.push(format!("id = {}", q(&format!("CSP-REF-{:05}", index + 1))));
        lines.push(format!(
            "lane_name = {}",
            q(row.get("lane_name").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "dataset_id = {}",
            q(row.get("dataset_id").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "slug = {}",
            q(row.get("slug").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "dataset_class = {}",
            q(row.get("dataset_class").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "source_csv = {}",
            q(row.get("source_csv").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "canonical_toml = {}",
            q(row.get("canonical_toml").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "source_sha256 = {}",
            q(row.get("source_sha256").unwrap_or(&String::new()))
        ));
        lines.push(format!(
            "row_count = {}",
            row.get("row_count").cloned().unwrap_or_default()
        ));
        lines.push(format!(
            "column_count = {}",
            row.get("column_count").cloned().unwrap_or_default()
        ));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

#[allow(clippy::too_many_arguments)]
fn render_migration_scope(
    inventory_path: &str,
    document_count: usize,
    zone_counts: &BTreeMap<String, usize>,
    action_counts: &BTreeMap<String, usize>,
    priority_counts: &BTreeMap<String, usize>,
    high_priority_paths: &[String],
    medium_priority_paths: &[String],
    next_waves: &[(&str, String)],
) -> Result<String> {
    let mut lines = vec![
        "# CSV migration scope registry (TOML-first).".to_string(),
        format!("# Generated by {BIN_PATH}"),
        String::new(),
        "[csv_migration_scope]".to_string(),
        format!("updated = {}", q(UPDATED_STAMP)),
        "authoritative = true".to_string(),
        format!("inventory_path = {}", q(inventory_path)),
        format!("document_count = {}", document_count),
        String::new(),
        "[zone_counts]".to_string(),
    ];
    for (key, value) in zone_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    lines.push("[action_counts]".to_string());
    for (key, value) in action_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    lines.push("[priority_counts]".to_string());
    for (key, value) in priority_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    lines.push("[next_waves]".to_string());
    for (key, value) in next_waves {
        lines.push(format!("{key} = {}", q(value)));
    }
    lines.push(String::new());
    lines.push("high_priority_path_sample = [".to_string());
    for path in high_priority_paths.iter().take(200) {
        lines.push(format!("  {},", q(path)));
    }
    lines.push("]".to_string());
    lines.push(String::new());
    lines.push("medium_priority_path_sample = [".to_string());
    for path in medium_priority_paths.iter().take(200) {
        lines.push(format!("  {},", q(path)));
    }
    lines.push("]".to_string());
    Ok(lines.join("\n"))
}

fn write_manifest(path: &Path, values: &[String]) -> Result<()> {
    let payload = if values.is_empty() {
        String::new()
    } else {
        format!("{}\n", values.join("\n"))
    };
    write_ascii(path, &payload)
}

fn source_paths(
    repo_root: &Path,
    source_glob: &str,
    source_manifest: Option<&PathBuf>,
) -> Result<Vec<String>> {
    if let Some(manifest) = source_manifest {
        let manifest_path = repo_root.join(manifest);
        let text = fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?;
        let mut values = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        values.sort();
        return Ok(values);
    }

    let pattern = repo_root.join(source_glob).to_string_lossy().into_owned();
    let mut values = Vec::new();
    for entry in glob(&pattern).with_context(|| format!("glob {pattern}"))? {
        let path = entry?;
        values.push(path_to_rel(repo_root, &path)?);
    }
    values.sort();
    Ok(values)
}

fn load_manifest_set(path: &Path) -> Result<BTreeSet<String>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect())
}

fn zone_path_set(inventory: &Table, zone: &str) -> BTreeSet<String> {
    table_array(inventory, "document")
        .unwrap_or_default()
        .into_iter()
        .filter(|row| string_field(row, "zone") == zone)
        .map(|row| string_field(&row, "path"))
        .collect()
}

fn path_set_from_table_array(root: &Table, key: &str, field: &str) -> Result<BTreeSet<String>> {
    Ok(table_array(root, key)?
        .into_iter()
        .map(|row| string_field(&row, field))
        .filter(|value| !value.is_empty())
        .collect())
}

fn manifest_path_from_descriptor(source_descriptor: &str) -> String {
    source_descriptor
        .strip_prefix("manifest:")
        .unwrap_or_default()
        .to_string()
}

fn load_source_csv_set(path: &Path) -> Result<BTreeSet<String>> {
    path_set_from_table_array(&load_toml(path)?, "dataset", "source_csv")
}

fn summarize_set_mismatch(
    failures: &mut Vec<String>,
    label: &str,
    expected: &BTreeSet<String>,
    actual: &BTreeSet<String>,
) {
    if expected == actual {
        return;
    }
    let missing = expected.difference(actual).cloned().collect::<Vec<_>>();
    let extra = actual.difference(expected).cloned().collect::<Vec<_>>();
    failures.push(format!("{label}: coverage mismatch."));
    if !missing.is_empty() {
        failures.push(format!("{label}: missing={}", missing.len()));
        for item in missing.iter().take(20) {
            failures.push(format!("- missing: {item}"));
        }
    }
    if !extra.is_empty() {
        failures.push(format!("{label}: extra={}", extra.len()));
        for item in extra.iter().take(20) {
            failures.push(format!("- extra: {item}"));
        }
    }
}

fn wave_state(total: usize, done: usize, pending_label: &str, done_label: &str) -> String {
    if total == 0 {
        return "n/a: no records".to_string();
    }
    if done >= total {
        return format!("complete: {done}/{total} {done_label}");
    }
    let remaining = total - done;
    format!("in_progress: {done}/{total} done, {remaining} pending ({pending_label})")
}

#[allow(clippy::type_complexity)]
fn parse_csv_with_stability(
    path: &Path,
    existing_meta: Option<&ExistingDatasetMeta>,
) -> Result<(bool, char, char, Vec<String>, Vec<String>, Vec<Vec<String>>)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let sample = String::from_utf8_lossy(&bytes[..bytes.len().min(65536)]).into_owned();
    let (delimiter, quotechar) = if let Some(meta) = existing_meta {
        (meta.delimiter, meta.quotechar)
    } else {
        sniff_dialect(&sample)
    };
    let rows_all = read_csv_rows(&bytes, delimiter, quotechar)?;
    let has_header = if let Some(meta) = existing_meta {
        meta.has_header
    } else {
        sniff_has_header(&rows_all)
    };
    let (header, original_header, rows) = normalize_parsed_rows(rows_all, has_header);
    Ok((
        has_header,
        delimiter,
        quotechar,
        header,
        original_header,
        rows,
    ))
}

#[allow(clippy::type_complexity)]
fn parse_source_for_verify(
    path: &Path,
    has_header: bool,
    delimiter: char,
    quotechar: char,
) -> Result<(
    Vec<String>,
    Vec<String>,
    Vec<Vec<String>>,
    Vec<String>,
    Vec<usize>,
    Vec<usize>,
)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let sample = String::from_utf8_lossy(&bytes[..bytes.len().min(65536)]).into_owned();
    let (sniff_delimiter, sniff_quotechar) = sniff_dialect(&sample);
    if delimiter != sniff_delimiter {
        bail!(
            "ERROR: {}: delimiter mismatch in canonical TOML (expected {:?}, found {:?})",
            path.display(),
            sniff_delimiter,
            delimiter
        );
    }
    if quotechar != sniff_quotechar {
        bail!(
            "ERROR: {}: quotechar mismatch in canonical TOML (expected {:?}, found {:?})",
            path.display(),
            sniff_quotechar,
            quotechar
        );
    }
    let rows_all = read_csv_rows(&bytes, delimiter, quotechar)?;
    let (header, original_header, rows) = normalize_parsed_rows(rows_all, has_header);
    let (types, non_empty, empty) = profile_columns(&rows, header.len());
    Ok((header, original_header, rows, types, non_empty, empty))
}

fn scrollify_source_for_verify(
    source_path: &Path,
    source_csv: &str,
    canonical_toml: &str,
    dataset: &Table,
) -> Result<scrolls_core::ConversionOutput> {
    let dataset_id = string_field(dataset, "id");
    let slug = string_field(dataset, "slug");
    let dataset_class = string_field(dataset, "dataset_class");
    let corpus_label = string_field(dataset, "corpus_label");
    let migrated_by = string_field(dataset, "migrated_by");
    let spec = ConvertSpec {
        dataset_id: &dataset_id,
        slug: &slug,
        source_csv,
        canonical_toml,
        dataset_class: &dataset_class,
        corpus_label: &corpus_label,
        migrated_by: &migrated_by,
    };
    scrolls_core::convert_csv_to_scroll(source_path, &spec)
        .with_context(|| format!("scrollify verify {}", source_path.display()))
}

fn read_csv_rows(bytes: &[u8], delimiter: char, quotechar: char) -> Result<Vec<Vec<String>>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(delimiter as u8)
        .quote(quotechar as u8)
        .from_reader(bytes);
    let mut rows: Vec<Vec<String>> = Vec::new();
    for record in reader.records() {
        let record = record?;
        rows.push(record.iter().map(ToOwned::to_owned).collect());
    }
    if let Some(first) = rows.first_mut().and_then(|row| row.first_mut()) {
        *first = first.trim_start_matches('\u{FEFF}').to_string();
    }
    Ok(rows)
}

fn normalize_parsed_rows(
    parsed: Vec<Vec<String>>,
    has_header: bool,
) -> (Vec<String>, Vec<String>, Vec<Vec<String>>) {
    let (original_header, data_rows) = if has_header && !parsed.is_empty() {
        (parsed[0].clone(), parsed[1..].to_vec())
    } else {
        (Vec::new(), parsed)
    };
    let max_cols = std::iter::once(original_header.len())
        .chain(data_rows.iter().map(Vec::len))
        .max()
        .unwrap_or(0);
    let header_tokens = if max_cols == 0 {
        Vec::new()
    } else if has_header {
        let mut padded = original_header.clone();
        padded.resize(max_cols, String::new());
        padded
            .iter()
            .enumerate()
            .map(|(idx, token)| sanitize_header_token(token, idx))
            .collect::<Vec<_>>()
    } else {
        (0..max_cols)
            .map(|idx| format!("col_{}", idx + 1))
            .collect::<Vec<_>>()
    };
    let header = make_unique(&header_tokens);
    let mut normalized_rows = Vec::new();
    for mut row in data_rows {
        row.resize(max_cols, String::new());
        normalized_rows.push(row.into_iter().take(max_cols).collect());
    }
    (header, original_header, normalized_rows)
}

fn profile_columns(
    rows: &[Vec<String>],
    column_count: usize,
) -> (Vec<String>, Vec<usize>, Vec<usize>) {
    let mut types = Vec::new();
    let mut non_empty_counts = Vec::new();
    let mut empty_counts = Vec::new();
    for index in 0..column_count {
        let values = rows
            .iter()
            .map(|row| row.get(index).cloned().unwrap_or_default())
            .collect::<Vec<_>>();
        let non_empty = values
            .iter()
            .filter(|value| !value.trim().is_empty())
            .count();
        let empty = values.len().saturating_sub(non_empty);
        types.push(infer_type(&values));
        non_empty_counts.push(non_empty);
        empty_counts.push(empty);
    }
    (types, non_empty_counts, empty_counts)
}

fn infer_type(values: &[String]) -> String {
    let non_empty = values
        .iter()
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    if non_empty.is_empty() {
        return "empty".to_string();
    }
    let lowered = non_empty
        .iter()
        .map(|item| item.to_ascii_lowercase())
        .collect::<Vec<_>>();
    if lowered
        .iter()
        .all(|item| ["true", "false", "yes", "no"].contains(&item.as_str()))
    {
        return "bool".to_string();
    }
    if non_empty.iter().all(|item| int_regex().is_match(item)) {
        return "int".to_string();
    }
    if non_empty.iter().all(|item| to_float(item).is_some()) {
        return "float".to_string();
    }
    "string".to_string()
}

fn to_float(value: &str) -> Option<f64> {
    let parsed = value.parse::<f64>().ok()?;
    if parsed.is_finite() {
        Some(parsed)
    } else {
        None
    }
}

fn sanitize_header_token(token: &str, index: usize) -> String {
    let mut token = token.trim().to_ascii_lowercase();
    token = non_alnum_regex().replace_all(&token, "_").into_owned();
    token = token.trim_matches('_').to_string();
    if token.is_empty() {
        token = format!("col_{}", index + 1);
    }
    if token
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        token = format!("col_{token}");
    }
    token
}

fn make_unique(tokens: &[String]) -> Vec<String> {
    let mut seen = BTreeMap::new();
    let mut out = Vec::new();
    for token in tokens {
        let count = seen.entry(token.clone()).or_insert(0usize);
        *count += 1;
        if *count == 1 {
            out.push(token.clone());
        } else {
            out.push(format!("{}_{}", token, count));
        }
    }
    out
}

fn slugify(name: &str) -> String {
    let stem = name
        .rsplit_once('.')
        .map(|(head, _)| head)
        .unwrap_or(name)
        .to_ascii_lowercase();
    let mut stem = non_alnum_regex().replace_all(&stem, "_").into_owned();
    stem = multi_underscore_regex()
        .replace_all(&stem, "_")
        .into_owned();
    stem = stem.trim_matches('_').to_string();
    if stem.is_empty() {
        "dataset".to_string()
    } else {
        stem
    }
}

fn sniff_dialect(sample: &str) -> (char, char) {
    let delimiters = [',', ';', '\t', '|'];
    let mut best_delimiter = ',';
    let mut best_score = 0usize;
    let lines = sample
        .lines()
        .take(50)
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    for delimiter in delimiters {
        let counts = lines
            .iter()
            .map(|line| count_delimiter(line, delimiter))
            .collect::<Vec<_>>();
        let non_zero = counts.iter().filter(|count| **count > 0).count();
        let total = counts.iter().sum::<usize>();
        let score = non_zero * 1000 + total;
        if score > best_score {
            best_score = score;
            best_delimiter = delimiter;
        }
    }
    (best_delimiter, '"')
}

fn count_delimiter(line: &str, delimiter: char) -> usize {
    let mut count = 0usize;
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                let _ = chars.next();
            } else {
                in_quotes = !in_quotes;
            }
        } else if ch == delimiter && !in_quotes {
            count += 1;
        }
    }
    count
}

fn sniff_has_header(rows: &[Vec<String>]) -> bool {
    if rows.len() < 2 || rows[0].is_empty() {
        return true;
    }
    let first = &rows[0];
    let second = &rows[1];
    let mut headerish = 0usize;
    let mut comparable = 0usize;
    for idx in 0..first.len().max(second.len()) {
        let a = first.get(idx).cloned().unwrap_or_default();
        let b = second.get(idx).cloned().unwrap_or_default();
        if a.trim().is_empty() {
            continue;
        }
        comparable += 1;
        let a_num = to_float(a.trim()).is_some();
        let b_num = to_float(b.trim()).is_some();
        if !a_num && b_num {
            headerish += 1;
            continue;
        }
        let a_bool = matches!(
            a.trim().to_ascii_lowercase().as_str(),
            "true" | "false" | "yes" | "no"
        );
        let b_bool = matches!(
            b.trim().to_ascii_lowercase().as_str(),
            "true" | "false" | "yes" | "no"
        );
        if !a_bool && b_bool {
            headerish += 1;
            continue;
        }
        if non_alnum_regex().is_match(a.trim()) {
            headerish += 1;
        }
    }
    comparable == 0 || headerish * 2 >= comparable
}

fn git_paths(repo_root: &Path, args: &[&str]) -> Result<BTreeSet<String>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!("git {} failed", args.join(" "));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}

fn all_filesystem_csv(repo_root: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for entry in WalkDir::new(repo_root).follow_links(false) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = path_to_rel(repo_root, entry.path())?;
        if rel.starts_with(".git/") || !rel.ends_with(".csv") {
            continue;
        }
        out.push(rel);
    }
    out.sort();
    Ok(out)
}

fn git_status(
    path: &str,
    tracked: &BTreeSet<String>,
    untracked: &BTreeSet<String>,
    ignored: &BTreeSet<String>,
) -> String {
    if tracked.contains(path) {
        "tracked".to_string()
    } else if untracked.contains(path) {
        "untracked".to_string()
    } else if ignored.contains(path) {
        "ignored".to_string()
    } else {
        "unknown".to_string()
    }
}

fn zone_for(path: &str) -> String {
    if path.starts_with("data/csv/legacy/") {
        "legacy_csv".to_string()
    } else if path.starts_with("data/csv/") {
        "project_csv".to_string()
    } else if path.starts_with("data/external/") {
        "external_csv".to_string()
    } else if path.starts_with("curated/") {
        "curated_csv".to_string()
    } else if path.starts_with("archive/") || path.starts_with("docs/archive/") {
        "archive_csv".to_string()
    } else {
        "other_csv".to_string()
    }
}

fn policy(_path: &str, zone: &str) -> (String, String, String) {
    match zone {
        "legacy_csv" => (
            "migrate_to_toml_canonical".to_string(),
            "critical".to_string(),
            "Legacy CSV should become TOML-native canonical data.".to_string(),
        ),
        "project_csv" => (
            "evaluate_for_toml_canonical".to_string(),
            "high".to_string(),
            "Project CSV may be generated artifacts or transition candidates.".to_string(),
        ),
        "curated_csv" => (
            "plan_curated_ingest".to_string(),
            "high".to_string(),
            "Curated observational CSV should move under central TOML data policy.".to_string(),
        ),
        "external_csv" => (
            "track_provenance_only".to_string(),
            "medium".to_string(),
            "External CSV remains provenance-managed input unless explicitly curated.".to_string(),
        ),
        "archive_csv" => (
            "review_archive_policy".to_string(),
            "low".to_string(),
            "Archived CSV may be retained as historical snapshots.".to_string(),
        ),
        _ => (
            "manual_triage".to_string(),
            "medium".to_string(),
            "CSV outside expected zones; requires classification.".to_string(),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn policy_with_progress(
    path: &str,
    zone: &str,
    legacy_canonical_paths: &BTreeSet<String>,
    curated_canonical_paths: &BTreeSet<String>,
    project_canonical_paths: &BTreeSet<String>,
    project_generated_paths: &BTreeSet<String>,
    external_holding_paths: &BTreeSet<String>,
    archive_holding_paths: &BTreeSet<String>,
    external_holding_scroll_paths: &BTreeSet<String>,
    archive_holding_scroll_paths: &BTreeSet<String>,
    project_split_classification: &BTreeMap<String, String>,
) -> (String, String, String) {
    if zone == "legacy_csv" && legacy_canonical_paths.contains(path) {
        return (
            "canonicalized_to_toml".to_string(),
            "complete".to_string(),
            "Legacy CSV is already canonicalized under registry/data/legacy_csv.".to_string(),
        );
    }
    if zone == "curated_csv" && curated_canonical_paths.contains(path) {
        return (
            "canonicalized_to_toml".to_string(),
            "complete".to_string(),
            "Curated CSV is already canonicalized under registry/data/curated_csv.".to_string(),
        );
    }
    if zone == "project_csv" && project_canonical_paths.contains(path) {
        return (
            "canonicalized_to_toml".to_string(),
            "complete".to_string(),
            "Project canonical dataset is represented in registry/data/project_csv/canonical."
                .to_string(),
        );
    }
    if zone == "project_csv" && project_generated_paths.contains(path) {
        return (
            "canonicalized_to_toml_generated_artifact".to_string(),
            "complete".to_string(),
            "Project generated artifact is represented in registry/data/project_csv/generated."
                .to_string(),
        );
    }
    if zone == "external_csv" && external_holding_scroll_paths.contains(path) {
        return (
            "canonicalized_to_toml_holding".to_string(),
            "complete".to_string(),
            "External CSV holding source is represented in registry/data/external_csv_holding."
                .to_string(),
        );
    }
    if zone == "archive_csv" && archive_holding_scroll_paths.contains(path) {
        return (
            "canonicalized_to_toml_holding".to_string(),
            "complete".to_string(),
            "Archive CSV holding source is represented in registry/data/archive_csv_holding."
                .to_string(),
        );
    }
    if zone == "external_csv" && external_holding_paths.contains(path) {
        return (
            "queued_for_scroll_holding".to_string(),
            "high".to_string(),
            "External CSV is queued in holding registry for TOML scroll conversion.".to_string(),
        );
    }
    if zone == "archive_csv" && archive_holding_paths.contains(path) {
        return (
            "queued_for_scroll_holding".to_string(),
            "high".to_string(),
            "Archive CSV is queued in holding registry for TOML scroll conversion.".to_string(),
        );
    }
    if zone == "project_csv" {
        let classification = project_split_classification
            .get(path)
            .cloned()
            .unwrap_or_default();
        if classification == "generated_artifact" {
            return (
                "preserve_generated_artifact".to_string(),
                "high".to_string(),
                "Project CSV classified as generated artifact and pending/retained under scroll policy.".to_string(),
            );
        }
        if classification == "canonical_dataset" {
            return (
                "migrate_to_toml_canonical".to_string(),
                "high".to_string(),
                "Project CSV classified as canonical dataset pending/active TOML migration."
                    .to_string(),
            );
        }
    }
    policy(path, zone)
}

fn load_canonical_source_paths(
    repo_root: &Path,
    index_rel: &str,
    table_name: &str,
) -> Result<BTreeSet<String>> {
    let index_path = repo_root.join(index_rel);
    if !index_path.exists() {
        return Ok(BTreeSet::new());
    }
    let parsed = load_toml(&index_path)?;
    let mut rows = table_array(&parsed, "dataset")?;
    if rows.is_empty() {
        rows = table_array(&parsed, table_name)?;
    }
    let mut out = BTreeSet::new();
    for row in rows {
        let source = string_field(&row, "source_path")
            .if_empty(&string_field(&row, "source_csv"))
            .if_empty(&string_field(&row, "path"));
        if !source.is_empty() {
            out.insert(source);
        }
    }
    Ok(out)
}

fn load_project_split_classification(repo_root: &Path) -> Result<BTreeMap<String, String>> {
    let policy_path = repo_root.join("registry/project_csv_split_policy.toml");
    if !policy_path.exists() {
        return Ok(BTreeMap::new());
    }
    let parsed = load_toml(&policy_path)?;
    let mut out = BTreeMap::new();
    for row in table_array(&parsed, "dataset")? {
        let path = string_field(&row, "path");
        let classification = string_field(&row, "classification");
        if !path.is_empty() && !classification.is_empty() {
            out.insert(path, classification);
        }
    }
    Ok(out)
}

fn load_existing_dataset_meta(
    repo_root: &Path,
    index_path: &Path,
) -> Result<BTreeMap<String, ExistingDatasetMeta>> {
    let full = repo_root.join(index_path);
    if !full.exists() {
        return Ok(BTreeMap::new());
    }
    let parsed = load_toml(&full)?;
    let mut out = BTreeMap::new();
    for row in table_array(&parsed, "dataset")? {
        let source_csv = string_field(&row, "source_csv");
        if source_csv.is_empty() {
            continue;
        }
        out.insert(
            source_csv,
            ExistingDatasetMeta {
                dataset_id: string_field(&row, "id"),
                canonical_toml: string_field(&row, "canonical_toml"),
                source_sha256: string_field(&row, "source_sha256"),
                has_header: bool_field(&row, "has_header"),
                delimiter: single_char(&string_field(&row, "delimiter"), ','),
                quotechar: single_char(&string_field(&row, "quotechar"), '"'),
            },
        );
    }
    Ok(out)
}

fn load_existing_dataset(repo_root: &Path, rel_path: &str) -> Result<Dataset> {
    let canon = load_toml_from_git_or_fs(repo_root, rel_path)?;
    let dataset = table_value(&canon, "dataset");
    let header = string_list_field(&dataset, "header");
    let original_header = string_list_field(&dataset, "original_header");
    let rows = list_of_string_rows(dataset.get("rows"));
    let columns = table_array(&canon, "column")?;
    let mut column_types = Vec::new();
    let mut non_empty_counts = Vec::new();
    let mut empty_counts = Vec::new();
    for column in columns {
        column_types.push(string_field(&column, "inferred_type"));
        non_empty_counts.push(integer_field(&column, "non_empty_count", 0) as usize);
        empty_counts.push(integer_field(&column, "empty_count", 0) as usize);
    }
    Ok(Dataset {
        dataset_id: string_field(&dataset, "id"),
        slug: string_field(&dataset, "slug"),
        source_csv: string_field(&dataset, "source_csv"),
        source_sha256: string_field(&dataset, "source_sha256"),
        source_size_bytes: integer_field(&dataset, "source_size_bytes", 0) as usize,
        has_header: bool_field(&dataset, "has_header"),
        delimiter: single_char(&string_field(&dataset, "delimiter"), ','),
        quotechar: single_char(&string_field(&dataset, "quotechar"), '"'),
        row_count: integer_field(&dataset, "row_count", 0) as usize,
        column_count: integer_field(&dataset, "column_count", 0) as usize,
        header,
        original_header,
        rows,
        header_value_sha256: string_field(&dataset, "header_value_sha256"),
        row_value_sha256: string_field(&dataset, "row_value_sha256"),
        canonical_toml: rel_path.to_string(),
        column_types,
        non_empty_counts,
        empty_counts,
    })
}

fn load_toml_from_git_or_fs(repo_root: &Path, rel_path: &str) -> Result<Table> {
    if let Some(text) = git_show_head(repo_root, rel_path)? {
        return parse_toml_text(&text, rel_path);
    }
    load_toml(&repo_root.join(rel_path))
}

fn git_show_head(repo_root: &Path, rel_path: &str) -> Result<Option<String>> {
    let spec = format!("HEAD:{rel_path}");
    let output = Command::new("git")
        .args(["show", &spec])
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("run git show {spec}"))?;
    if output.status.success() {
        Ok(Some(String::from_utf8_lossy(&output.stdout).into_owned()))
    } else {
        Ok(None)
    }
}

fn extract_generated_explicit_paths(path: &Path) -> Result<BTreeSet<String>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(generated_csv_regex()
        .captures_iter(&text)
        .filter_map(|caps| caps.get(1).map(|m| m.as_str().to_string()))
        .collect())
}

fn table_array(root: &Table, key: &str) -> Result<Vec<Table>> {
    let Some(value) = root.get(key) else {
        return Ok(Vec::new());
    };
    let Some(values) = value.as_array() else {
        bail!("expected array for key {key}");
    };
    Ok(values
        .iter()
        .filter_map(|value| value.as_table().cloned())
        .collect())
}

fn table_value(root: &Table, key: &str) -> Table {
    root.get(key)
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default()
}

fn load_toml(path: &Path) -> Result<Table> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    parse_toml_text(&text, &path.display().to_string())
}

fn parse_toml_text(text: &str, context: &str) -> Result<Table> {
    let value = text
        .parse::<Value>()
        .with_context(|| format!("parse TOML {context}"))?;
    let table = value
        .as_table()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("root TOML document is not a table: {context}"))?;
    Ok(table)
}

fn path_to_rel(repo_root: &Path, path: &Path) -> Result<String> {
    Ok(path
        .strip_prefix(repo_root)
        .with_context(|| {
            format!(
                "strip prefix {} from {}",
                repo_root.display(),
                path.display()
            )
        })?
        .to_string_lossy()
        .replace('\\', "/"))
}

fn sha_text_json<T: serde::Serialize>(value: &T) -> Result<String> {
    let blob = serde_json::to_string(value)?;
    Ok(sha256_hex(blob.as_bytes()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn write_ascii(path: &Path, content: &str) -> Result<()> {
    assert_ascii(content, &path.display().to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(path, content).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn assert_ascii(text: &str, context: &str) -> Result<()> {
    let bad = text
        .chars()
        .filter(|ch| *ch as u32 > 127)
        .collect::<BTreeSet<_>>();
    if !bad.is_empty() {
        let sample = bad.iter().take(20).collect::<String>();
        bail!("ERROR: Non-ASCII output in {context}: {sample:?}");
    }
    Ok(())
}

fn q(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn string_field(table: &Table, key: &str) -> String {
    table.get(key).map(value_to_string).unwrap_or_default()
}

fn string_list_field(table: &Table, key: &str) -> Vec<String> {
    table
        .get(key)
        .and_then(Value::as_array)
        .map(|values| values.iter().map(value_to_string).collect())
        .unwrap_or_default()
}

fn list_of_string_rows(value: Option<&Value>) -> Vec<Vec<String>> {
    let Some(Value::Array(rows)) = value else {
        return Vec::new();
    };
    rows.iter()
        .map(|row| {
            row.as_array()
                .map(|values| values.iter().map(value_to_string).collect())
                .unwrap_or_default()
        })
        .collect()
}

fn integer_field(table: &Table, key: &str, default: i64) -> i64 {
    table
        .get(key)
        .and_then(Value::as_integer)
        .unwrap_or(default)
}

fn bool_field(table: &Table, key: &str) -> bool {
    table.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Integer(v) => v.to_string(),
        Value::Float(v) => v.to_string(),
        Value::Boolean(v) => v.to_string(),
        Value::Datetime(v) => v.to_string(),
        other => other.to_string(),
    }
}

fn single_char(value: &str, default: char) -> char {
    value.chars().next().unwrap_or(default)
}

fn dataset_slug_from_canonical(path: &str) -> Option<String> {
    let file = Path::new(path).file_stem()?.to_str()?;
    let (_, slug) = file.split_once('_')?;
    Some(slug.to_string())
}

trait IfEmpty {
    fn if_empty(self, fallback: &str) -> String;
}

impl IfEmpty for String {
    fn if_empty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}

fn non_alnum_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"[^a-z0-9]+").unwrap())
}

fn multi_underscore_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"_+").unwrap())
}

fn int_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^[+-]?\d+$").unwrap())
}

fn dataset_prefix_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^[A-Z0-9]{1,6}$").unwrap())
}

fn generated_csv_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r#""(data/csv/[^"]+\.csv)""#).unwrap())
}
