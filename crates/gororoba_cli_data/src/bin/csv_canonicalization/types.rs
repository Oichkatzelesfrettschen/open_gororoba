//! Type definitions for the `csv-canonicalization` binary:
//! Cli + Commands enum + all per-subcommand Args structs, plus
//! shared DEFAULT_* constants and the Table type alias.
//!
//! Fields and constants are pub(crate) so the bin root can match
//! and reference. Uses #[path] indirection because the binary has
//! an explicit Cargo.toml path.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use toml::Value;

pub(crate) type Table = toml::map::Map<String, Value>;

pub(crate) const UPDATED_STAMP: &str = "2026-02-09";
pub(crate) const BIN_PATH: &str = "crates/gororoba_cli_data/src/bin/csv_canonicalization.rs";
pub(crate) const DEFAULT_SOURCE_GLOB: &str = "data/csv/legacy/*.csv";
pub(crate) const DEFAULT_CANON_DIR: &str = "registry/data/legacy_csv";
pub(crate) const DEFAULT_INDEX_PATH: &str = "registry/legacy_csv_datasets.toml";
pub(crate) const DEFAULT_INDEX_TABLE: &str = "legacy_csv_datasets";
pub(crate) const DEFAULT_DATASET_PREFIX: &str = "LC";
pub(crate) const DEFAULT_CORPUS_LABEL: &str = "legacy CSV";
pub(crate) const MAKEFILE_GENERATED_GLOBS: &[&str] = &[
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
pub(crate) struct Cli {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,

    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
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
pub(crate) struct InventoryArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    pub(crate) out: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct MigrateArgs {
    #[arg(long, default_value = DEFAULT_SOURCE_GLOB)]
    pub(crate) source_glob: String,

    #[arg(long)]
    pub(crate) source_manifest: Option<PathBuf>,

    #[arg(long, default_value = DEFAULT_INDEX_PATH)]
    pub(crate) out_index: PathBuf,

    #[arg(long, default_value = DEFAULT_CANON_DIR)]
    pub(crate) out_dir: PathBuf,

    #[arg(long, default_value = DEFAULT_INDEX_TABLE)]
    pub(crate) index_table: String,

    #[arg(long, default_value = DEFAULT_DATASET_PREFIX)]
    pub(crate) dataset_prefix: String,

    #[arg(long, default_value = DEFAULT_CORPUS_LABEL)]
    pub(crate) corpus_label: String,
}

#[derive(Parser, Debug)]
pub(crate) struct VerifyArgs {
    #[arg(long, default_value = DEFAULT_INDEX_PATH)]
    pub(crate) index_path: PathBuf,

    #[arg(long, default_value = DEFAULT_SOURCE_GLOB)]
    pub(crate) source_glob: String,

    #[arg(long)]
    pub(crate) source_manifest: Option<PathBuf>,

    #[arg(long, default_value = DEFAULT_CORPUS_LABEL)]
    pub(crate) corpus_label: String,

    #[arg(long, default_value_t = false)]
    pub(crate) coverage_only: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct ProjectSplitPolicyArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    pub(crate) inventory: PathBuf,

    #[arg(long, default_value = "registry/project_csv_split_policy.toml")]
    pub(crate) out: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/project_csv_canonical_manifest.txt"
    )]
    pub(crate) canonical_manifest: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/project_csv_generated_manifest.txt"
    )]
    pub(crate) generated_manifest: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct HoldingsArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    pub(crate) inventory: PathBuf,

    #[arg(long, default_value = "registry/external_csv_holding.toml")]
    pub(crate) external_out: PathBuf,

    #[arg(long, default_value = "registry/archive_csv_holding.toml")]
    pub(crate) archive_out: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/external_csv_holding_manifest.txt"
    )]
    pub(crate) external_manifest: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/archive_csv_holding_manifest.txt"
    )]
    pub(crate) archive_manifest: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct VerifyHoldingsArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    pub(crate) inventory: PathBuf,

    #[arg(long, default_value = "registry/external_csv_holding.toml")]
    pub(crate) external_registry: PathBuf,

    #[arg(long, default_value = "registry/archive_csv_holding.toml")]
    pub(crate) archive_registry: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/external_csv_holding_manifest.txt"
    )]
    pub(crate) external_manifest: PathBuf,

    #[arg(
        long,
        default_value = "registry/manifests/archive_csv_holding_manifest.txt"
    )]
    pub(crate) archive_manifest: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct ScrollPipelineArgs {
    #[arg(long, default_value = "registry/csv_scroll_pipeline.toml")]
    pub(crate) out: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct VerifyScrollPipelineArgs {
    #[arg(long, default_value = "registry/csv_scroll_pipeline.toml")]
    pub(crate) pipeline: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct VerifyCorpusCoverageArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    pub(crate) inventory: PathBuf,

    #[arg(long, default_value = "registry/project_csv_canonical_datasets.toml")]
    pub(crate) project_canonical_index: PathBuf,

    #[arg(long, default_value = "registry/project_csv_generated_artifacts.toml")]
    pub(crate) project_generated_index: PathBuf,

    #[arg(long, default_value = "registry/legacy_csv_datasets.toml")]
    pub(crate) legacy_index: PathBuf,

    #[arg(long, default_value = "registry/curated_csv_datasets.toml")]
    pub(crate) curated_index: PathBuf,

    #[arg(long, default_value = "registry/external_csv_holding_datasets.toml")]
    pub(crate) external_holding_index: PathBuf,

    #[arg(long, default_value = "registry/archive_csv_holding_datasets.toml")]
    pub(crate) archive_holding_index: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct MigrationScopeArgs {
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    pub(crate) inventory: PathBuf,

    #[arg(long, default_value = "registry/csv_migration_scope.toml")]
    pub(crate) out: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct CsvDoc {
    pub(crate) path: String,
    pub(crate) git_status: String,
    pub(crate) zone: String,
    pub(crate) archived: bool,
    pub(crate) generated: bool,
    pub(crate) size_bytes: usize,
    pub(crate) line_count: usize,
    pub(crate) sha256: String,
    pub(crate) migration_action: String,
    pub(crate) migration_priority: String,
    pub(crate) rationale: String,
}

#[derive(Debug, Clone)]
pub(crate) struct Dataset {
    pub(crate) dataset_id: String,
    pub(crate) slug: String,
    pub(crate) source_csv: String,
    pub(crate) source_sha256: String,
    pub(crate) source_size_bytes: usize,
    pub(crate) has_header: bool,
    pub(crate) delimiter: char,
    pub(crate) quotechar: char,
    pub(crate) row_count: usize,
    pub(crate) column_count: usize,
    pub(crate) header: Vec<String>,
    pub(crate) original_header: Vec<String>,
    pub(crate) rows: Vec<Vec<String>>,
    pub(crate) header_value_sha256: String,
    pub(crate) row_value_sha256: String,
    pub(crate) canonical_toml: String,
    pub(crate) column_types: Vec<String>,
    pub(crate) non_empty_counts: Vec<usize>,
    pub(crate) empty_counts: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct ExistingDatasetMeta {
    pub(crate) dataset_id: String,
    pub(crate) canonical_toml: String,
    pub(crate) source_sha256: String,
    pub(crate) has_header: bool,
    pub(crate) delimiter: char,
    pub(crate) quotechar: char,
}

#[derive(Debug, Clone)]
pub(crate) struct HoldingRow {
    pub(crate) path: String,
    pub(crate) source_sha256: String,
    pub(crate) size_bytes: usize,
    pub(crate) git_status: String,
    pub(crate) target_lane: String,
}
