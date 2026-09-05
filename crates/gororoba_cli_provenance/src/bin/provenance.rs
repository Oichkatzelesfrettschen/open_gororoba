use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use lit_search::{
    SearchEngine, VerificationReport, check_novelty, search::SourceTier, sources::ApiKeys,
    verify_citations,
};
use provenance_core::{
    ArtifactQueryResult, BinaryRecord, ClaimRecord, ControlPlaneCounts, DoctorReport,
    DocumentQueryResult, DownloadCampaignQueryResult, DownloadQueryResult, ExperimentRecord,
    InsightRecord, LiteratureNoveltySimilarPaperRecord, LiteratureVerificationQueryResult,
    LiteratureVerificationResultRecord, LiteratureVerificationRunRecord, PantheonSeedSummary,
    TheoremRecord,
};
use provenance_ops::source_provenance;
use provenance_store::{
    ExternalSourceContractPatch, ProvenanceStore, RegistryImportPaths, ReimportOptions,
};
use regex::Regex;
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Parser, Debug)]
#[command(name = "provenance", about = "SQLite-backed provenance operator CLI")]
struct Cli {
    /// Repository root.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// SQLite database path.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Load compatibility registries into the normalized SQLite provenance store.
    Index(IndexArgs),
    /// Deprecated alias of export-artifact-scan. Prints the new name and exits nonzero.
    #[command(hide = true)]
    Export(ExportArgs),
    /// Rebuild the artifact registry and lane projections by scanning host
    /// filesystem state. Reads this checkout's working tree, including
    /// gitignored intake directories, so its output depends on which files the
    /// running host holds.
    ExportArtifactScan(ExportArtifactScanArgs),
    /// Report which artifact paths this host materializes, into a gitignored manifest.
    MaterializeStatus(MaterializeStatusArgs),
    /// Seal the exact linkless inventory correction frontier for offline replay.
    PlanLinklessMaterializability {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        repaired_at: String,
    },
    /// Correct sealed linkless inventory statuses and regenerate their projections.
    RepairLinklessMaterializability {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        audit: PathBuf,
    },
    /// Verify SQLite invariants and optionally compatibility export invariants.
    Verify(VerifyArgs),
    /// Legacy/bootstrap import from compatibility TOML/proof manifests into the canonical SQLite control plane.
    #[command(visible_alias = "import-legacy-control-plane")]
    IndexControlPlane(IndexControlPlaneArgs),
    /// Export compatibility TOML and theorem markdown views from the canonical SQLite control plane.
    ExportControlPlane(ExportControlPlaneArgs),
    /// Verify canonical SQLite control-plane invariants and generated compatibility exports.
    VerifyControlPlane(VerifyControlPlaneArgs),
    /// Load external source contracts and dossiers into the canonical SQLite control plane.
    IndexExternalSources(IndexExternalSourcesArgs),
    /// Export external source compatibility files and markdown dossiers from the canonical SQLite control plane.
    ExportExternalSources(ExportExternalSourcesArgs),
    /// Verify external source SQLite invariants and generated compatibility exports.
    VerifyExternalSources(VerifyExternalSourcesArgs),
    /// Update one SQLite-authored external source contract and optionally re-export compatibility views.
    UpdateExternalSource(Box<UpdateExternalSourceArgs>),
    /// Query one artifact or document from the SQLite index.
    Query(QueryArgs),
    /// Print operator-focused health and drift summary from the SQLite index.
    Doctor(DoctorArgs),
    /// Audit missing mirrors and blocked artifacts using the SQLite index.
    LinkAudit(LinkAuditArgs),
    /// Emit a Rust-generated recovery plan for artifacts missing a working mirror.
    Recover(RecoverArgs),
    /// Seed the Pantheon/PhysicsForge migration SQLite memoization tables.
    PantheonSeed(PantheonSeedArgs),
    /// Verify a real bibliography input with lit_search and persist results into provenance storage.
    LiteratureVerify(LiteratureVerifyArgs),
}

#[derive(Parser, Debug)]
struct IndexArgs {
    /// Rebuild compatibility exports before indexing them into SQLite.
    #[arg(long, default_value_t = false)]
    refresh_compat_exports: bool,

    #[arg(long, default_value = "registry/artifact_source_of_truth.toml")]
    artifact_registry: PathBuf,

    #[arg(long, default_value = "registry/knowledge_sources.toml")]
    knowledge_sources: PathBuf,

    #[arg(long, default_value = "registry/source_lanes")]
    lane_dir: PathBuf,
}

#[derive(Parser, Debug)]
struct ExportArgs {
    #[arg(long, default_value = "registry/artifact_source_of_truth.toml")]
    out_registry: PathBuf,

    #[arg(
        long,
        default_value = "reports/artifact_source_of_truth_reconciliation_2026_02_15.toml"
    )]
    out_artifact_report: PathBuf,

    #[arg(long, default_value = "registry/source_infrastructure.toml")]
    out_infrastructure: PathBuf,

    #[arg(long, default_value = "registry/source_lanes")]
    lane_dir: PathBuf,

    #[arg(
        long,
        default_value = "reports/source_infrastructure_reconciliation_2026_02_15.toml"
    )]
    out_infrastructure_report: PathBuf,

    /// Reindex the SQLite store after exporting compatibility files. Defaults
    /// off: the reindex writes an export run into the canonical control plane,
    /// which is not implied by regenerating a filesystem projection.
    #[arg(long, default_value_t = false)]
    reindex_after: bool,
}

#[derive(Parser, Debug)]
struct ExportArtifactScanArgs {
    #[command(flatten)]
    export: ExportArgs,

    /// Fraction of rows an output may lose before the write is refused.
    #[arg(long, default_value_t = source_provenance::DEFAULT_SHRINK_THRESHOLD)]
    shrink_threshold: f64,

    /// Accept a row loss beyond the shrink threshold.
    #[arg(long, default_value_t = false)]
    allow_shrink: bool,

    /// Gitignored per-host materialization manifest written alongside the export.
    #[arg(
        long,
        default_value = "data/output/external_manifests/host_materialization.toml"
    )]
    manifest_out: PathBuf,
}

#[derive(Parser, Debug)]
struct MaterializeStatusArgs {
    #[arg(long, default_value = "registry/artifact_source_of_truth.toml")]
    artifact_registry: PathBuf,

    #[arg(
        long,
        default_value = "data/output/external_manifests/host_materialization.toml"
    )]
    out_manifest: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyArgs {
    #[arg(long, default_value_t = true)]
    verify_exports: bool,

    #[arg(long, default_value = "registry/artifact_source_of_truth.toml")]
    artifact_registry: PathBuf,

    #[arg(long, default_value = "registry/source_infrastructure.toml")]
    infrastructure: PathBuf,
}

#[derive(Parser, Debug)]
struct IndexControlPlaneArgs {
    #[arg(long, default_value = "registry/claims.toml")]
    claims: PathBuf,

    #[arg(long, default_value = "registry/insights.toml")]
    insights: PathBuf,

    #[arg(long, default_value = "registry/experiments.toml")]
    experiments: PathBuf,

    #[arg(long, default_value = "registry/binaries.toml")]
    binaries: PathBuf,

    #[arg(long, default_value = "proofs/_RocqProject")]
    rocq_project: PathBuf,

    /// Acknowledge that this run overwrites canonical values with mirror values.
    #[arg(long)]
    allow_destructive_reimport: bool,
}

#[derive(Parser, Debug)]
struct ExportControlPlaneArgs {
    #[arg(long, default_value = "registry/claims.toml")]
    claims: PathBuf,

    #[arg(long, default_value = "registry/insights.toml")]
    insights: PathBuf,

    #[arg(long, default_value = "registry/experiments.toml")]
    experiments: PathBuf,

    #[arg(long, default_value = "registry/binaries.toml")]
    binaries: PathBuf,

    #[arg(long, default_value = "docs/THEOREMS.md")]
    theorems: PathBuf,

    #[arg(long, default_value = "docs/generated/THEOREMS_REGISTRY_MIRROR.md")]
    theorems_mirror: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyControlPlaneArgs {
    #[arg(long, default_value_t = true)]
    verify_compat_exports: bool,

    #[arg(long, default_value = "registry/claims.toml")]
    claims: PathBuf,

    #[arg(long, default_value = "registry/insights.toml")]
    insights: PathBuf,

    #[arg(long, default_value = "registry/experiments.toml")]
    experiments: PathBuf,

    #[arg(long, default_value = "registry/binaries.toml")]
    binaries: PathBuf,

    #[arg(long, default_value = "docs/THEOREMS.md")]
    theorems: PathBuf,

    #[arg(long, default_value = "docs/generated/THEOREMS_REGISTRY_MIRROR.md")]
    theorems_mirror: PathBuf,
}

#[derive(Parser, Debug)]
struct IndexExternalSourcesArgs {
    #[arg(long, default_value = "data/external/SOURCES.toml")]
    source_contracts: PathBuf,

    #[arg(long, default_value = "registry/external_sources.toml")]
    dossiers_registry: PathBuf,
}

#[derive(Parser, Debug)]
struct ExportExternalSourcesArgs {
    #[arg(long, default_value = "data/external/SOURCES.toml")]
    source_contracts: PathBuf,

    #[arg(long, default_value = "registry/external_sources.toml")]
    dossiers_registry: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyExternalSourcesArgs {
    #[arg(long, default_value_t = true)]
    verify_compat_exports: bool,

    #[arg(long, default_value = "data/external/SOURCES.toml")]
    source_contracts: PathBuf,

    #[arg(long, default_value = "registry/external_sources.toml")]
    dossiers_registry: PathBuf,
}

#[derive(Parser, Debug)]
struct UpdateExternalSourceArgs {
    #[arg(long)]
    id: String,

    #[arg(long, default_value_t = false)]
    create_if_missing: bool,

    #[arg(long)]
    path_glob: Option<String>,

    #[arg(long)]
    canonical_url: Option<String>,

    #[arg(long = "mirror-url")]
    mirror_urls: Vec<String>,

    #[arg(long)]
    access_class: Option<String>,

    #[arg(long)]
    status: Option<String>,

    #[arg(long)]
    retrieval_method: Option<String>,

    #[arg(long)]
    attempt_deadline_utc: Option<String>,

    #[arg(long)]
    resolution_deadline_utc: Option<String>,

    #[arg(long)]
    blocker_note: Option<String>,

    #[arg(long = "evidence-ref")]
    evidence_refs: Vec<String>,

    #[arg(long = "manual-manifest-ref")]
    manual_manifest_refs: Vec<String>,

    #[arg(long = "blocked-action")]
    blocked_action_plan: Vec<String>,

    #[arg(long = "validator-ref")]
    scientific_validator_refs: Vec<String>,

    #[arg(long, default_value_t = true)]
    export_after: bool,

    #[arg(long, default_value_t = true)]
    verify_after: bool,

    #[arg(long, default_value = "data/external/SOURCES.toml")]
    source_contracts: PathBuf,

    #[arg(long, default_value = "registry/external_sources.toml")]
    dossiers_registry: PathBuf,
}

#[derive(Parser, Debug)]
struct QueryArgs {
    #[command(subcommand)]
    kind: QueryKind,
}

#[derive(Subcommand, Debug)]
enum QueryKind {
    Artifact {
        needle: String,
    },
    Claim {
        needle: String,
    },
    Download {
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        needle: Option<String>,
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        status: Option<String>,
        #[arg(long)]
        backend: Option<String>,
    },
    Campaign {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    Document {
        needle: String,
    },
    Insight {
        needle: String,
    },
    Experiment {
        needle: String,
    },
    Binary {
        needle: String,
    },
    Literature {
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    Theorem {
        needle: String,
    },
}

#[derive(Parser, Debug)]
struct DoctorArgs {
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct LinkAuditArgs {
    #[arg(long, default_value_t = 25)]
    limit: usize,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct RecoverArgs {
    #[arg(long, default_value_t = 50)]
    limit: usize,

    #[arg(long, default_value = "reports/provenance_recovery_plan.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct PantheonSeedArgs {
    #[arg(
        long,
        default_value = "archive/registry/pantheon_physicsforge/pantheon_physicsforge_migration_findings.toml"
    )]
    findings: PathBuf,

    #[arg(
        long,
        default_value = "archive/registry/pantheon_physicsforge/pantheon_physicsforge_overflow_tracker.toml"
    )]
    overflow: PathBuf,
}

#[derive(Parser, Debug)]
struct LiteratureVerifyArgs {
    #[arg(long)]
    bib: PathBuf,

    #[arg(long)]
    topic: Option<String>,

    #[arg(long)]
    hypotheses: Option<PathBuf>,

    #[arg(long = "domain")]
    domains: Vec<String>,

    #[arg(long, value_enum, default_value_t = SearchTierArg::All)]
    tier: SearchTierArg,

    #[arg(long, default_value_t = 300)]
    inter_verify_delay_ms: u64,

    #[arg(long, default_value_t = 30)]
    novelty_limit: usize,

    #[arg(long, default_value_t = 0.25)]
    similarity_threshold: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Toml,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SearchTierArg {
    Core,
    Open,
    All,
}

impl From<SearchTierArg> for SourceTier {
    fn from(value: SearchTierArg) -> Self {
        match value {
            SearchTierArg::Core => SourceTier::Core,
            SearchTierArg::Open => SourceTier::Open,
            SearchTierArg::All => SourceTier::All,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = resolve_repo_root(&cli.repo_root);
    let db_path = repo_path(&repo_root, &cli.db);
    match cli.command {
        Commands::Index(args) => run_index(&repo_root, &db_path, args),
        Commands::Export(_) => {
            eprintln!(
                "provenance export is renamed to provenance export-artifact-scan; \
                 it scans host filesystem state rather than reading repository truth."
            );
            std::process::exit(2);
        }
        Commands::ExportArtifactScan(args) => run_export_artifact_scan(&repo_root, &db_path, args),
        Commands::MaterializeStatus(args) => run_materialize_status(&repo_root, args),
        Commands::PlanLinklessMaterializability { spec, repaired_at } => {
            let count = source_provenance::inventory_repair::write_linkless_materializability_spec(
                &repo_root,
                &spec,
                &repaired_at,
            )?;
            println!("Sealed {count} linkless inventory witnesses");
            Ok(())
        }
        Commands::RepairLinklessMaterializability { spec, audit } => {
            let summary = source_provenance::inventory_repair::repair_linkless_materializability(
                &repo_root, &spec, &audit,
            )?;
            println!(
                "Inventory correction: {} witnesses, {} total artifacts, already_applied={}",
                summary.repaired_count, summary.artifact_count, summary.already_applied
            );
            Ok(())
        }
        Commands::Verify(args) => run_verify(&repo_root, &db_path, args),
        Commands::IndexControlPlane(args) => run_index_control_plane(&repo_root, &db_path, args),
        Commands::ExportControlPlane(args) => run_export_control_plane(&repo_root, &db_path, args),
        Commands::VerifyControlPlane(args) => run_verify_control_plane(&repo_root, &db_path, args),
        Commands::IndexExternalSources(args) => {
            run_index_external_sources(&repo_root, &db_path, args)
        }
        Commands::ExportExternalSources(args) => {
            run_export_external_sources(&repo_root, &db_path, args)
        }
        Commands::VerifyExternalSources(args) => {
            run_verify_external_sources(&repo_root, &db_path, args)
        }
        Commands::UpdateExternalSource(args) => {
            run_update_external_source(&repo_root, &db_path, *args)
        }
        Commands::Query(args) => run_query(&db_path, args),
        Commands::Doctor(args) => run_doctor(&db_path, args),
        Commands::LinkAudit(args) => run_link_audit(&db_path, args),
        Commands::Recover(args) => run_recover(&db_path, &repo_root, args),
        Commands::PantheonSeed(args) => run_pantheon_seed(&db_path, &repo_root, args),
        Commands::LiteratureVerify(args) => run_literature_verify(&repo_root, &db_path, args),
    }
}

fn run_index(repo_root: &Path, db_path: &Path, args: IndexArgs) -> Result<()> {
    ProvenanceStore::ensure_artifact_reimport_safe(db_path)?;
    if args.refresh_compat_exports {
        rebuild_compatibility_exports(
            repo_root,
            &ExportArgs {
                out_registry: PathBuf::from("registry/artifact_source_of_truth.toml"),
                out_artifact_report: PathBuf::from(
                    "reports/artifact_source_of_truth_reconciliation_2026_02_15.toml",
                ),
                out_infrastructure: PathBuf::from("registry/source_infrastructure.toml"),
                lane_dir: PathBuf::from("registry/source_lanes"),
                out_infrastructure_report: PathBuf::from(
                    "reports/source_infrastructure_reconciliation_2026_02_15.toml",
                ),
                reindex_after: false,
            },
        )?;
    }

    let artifact_registry = repo_path(repo_root, &args.artifact_registry);
    let knowledge_sources = repo_path(repo_root, &args.knowledge_sources);
    let lane_dir = repo_path(repo_root, &args.lane_dir);
    let mut store = ProvenanceStore::open(db_path)?;
    let stats = store.reindex_from_registries(
        repo_root,
        &artifact_registry,
        &knowledge_sources,
        &lane_dir,
    )?;
    println!(
        "Indexed provenance sqlite: artifacts={} documents={} lanes={} mirrors={} indexed_at={}",
        stats.artifact_count,
        stats.document_count,
        stats.lane_assignment_count,
        stats.mirror_observation_count,
        stats.indexed_at
    );
    Ok(())
}

fn run_export_artifact_scan(
    repo_root: &Path,
    db_path: &Path,
    args: ExportArtifactScanArgs,
) -> Result<()> {
    let ExportArtifactScanArgs {
        export,
        shrink_threshold,
        allow_shrink,
        manifest_out,
    } = args;
    // The reindex reads the lane files back through
    // provenance_store::loaders::load_lane_assignments, which requires an
    // artifact_ref array in each lane. Prove that before any rename, so a
    // reindex that cannot succeed never leaves a rewritten registry behind.
    if export.reindex_after {
        ProvenanceStore::ensure_artifact_reimport_safe(db_path)?;
        ensure_reindex_preconditions(repo_root, &export)?;
    }
    let policy = source_provenance::ShrinkPolicy {
        max_shrink_fraction: shrink_threshold,
        allow_shrink,
    };
    let summary = rebuild_compatibility_exports_with_policy(repo_root, &export, &policy)?;
    for report in &summary.row_counts {
        println!(
            "rows {}: before={} after={}",
            to_repo_display_path(repo_root, &report.path),
            report
                .before
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            report.after
        );
    }
    let generated_at = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let manifest_path = repo_path(repo_root, &manifest_out);
    source_provenance::write_host_materialization(
        &manifest_path,
        &source_provenance::render_host_materialization(
            &summary.host_materialization,
            &generated_at,
        ),
    )?;
    println!(
        "Host materialization manifest: rows={} path={}",
        summary.host_materialization.len(),
        to_repo_display_path(repo_root, &manifest_path)
    );
    println!(
        "Scanned host filesystem state: artifacts={} downloaded={} remotely_materializable={} materializable_without_url={}",
        summary.artifact_count,
        summary.downloaded_count,
        summary.remotely_materializable_count,
        summary.materializable_without_url_count
    );
    if export.reindex_after {
        let mut store = ProvenanceStore::open(db_path)?;
        let stats = store.reindex_from_registries(
            repo_root,
            &repo_path(repo_root, &export.out_registry),
            &repo_path(repo_root, &PathBuf::from("registry/knowledge_sources.toml")),
            &repo_path(repo_root, &export.lane_dir),
        )?;
        store.record_export_run(
            "export-artifact-scan",
            stats.artifact_count,
            stats.document_count,
            &json!({
                "artifact_registry": export.out_registry,
                "infrastructure": export.out_infrastructure,
                "lane_dir": export.lane_dir,
            })
            .to_string(),
        )?;
        println!(
            "Refreshed sqlite: artifacts={} documents={}",
            stats.artifact_count, stats.document_count
        );
    }
    Ok(())
}

/// Fails before any write when a lane file the reindex will read carries no
/// artifact_ref array.
fn ensure_reindex_preconditions(repo_root: &Path, args: &ExportArgs) -> Result<()> {
    let lane_dir = repo_path(repo_root, &args.lane_dir);
    for lane in [
        "datasets",
        "slides_artifacts",
        "papers_pdf",
        "web_references",
    ] {
        let path = lane_dir.join(format!("{lane}.toml"));
        if !path.exists() {
            continue;
        }
        let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let value: toml::Value =
            toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        if value
            .get("artifact_ref")
            .and_then(toml::Value::as_array)
            .is_none()
        {
            bail!(
                "artifact_ref table missing in {}; reindex would fail after the export wrote, \
                 so the export is refused",
                path.display()
            );
        }
    }
    Ok(())
}

/// Re-checks per-host presence without rewriting any registry. The paths come
/// from the existing manifest, which is where host state lives, plus the
/// downloaded rows the registry itself carries.
fn run_materialize_status(repo_root: &Path, args: MaterializeStatusArgs) -> Result<()> {
    let retention = source_provenance::RetentionSet::from_git_index(repo_root);
    let mut seen: Vec<(String, String, String, String)> = Vec::new();

    let manifest_path = repo_path(repo_root, &args.out_manifest);
    if manifest_path.exists() {
        let text = fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?;
        let value: toml::Value =
            toml::from_str(&text).with_context(|| format!("parse {}", manifest_path.display()))?;
        for row in value
            .get("materialized")
            .and_then(toml::Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let Some(table) = row.as_table() else {
                continue;
            };
            let field = |name: &str| {
                table
                    .get(name)
                    .and_then(toml::Value::as_str)
                    .unwrap_or_default()
                    .to_string()
            };
            seen.push((
                field("artifact_id"),
                field("key"),
                field("status"),
                field("path"),
            ));
        }
    }

    let registry_path = repo_path(repo_root, &args.artifact_registry);
    let text = fs::read_to_string(&registry_path)
        .with_context(|| format!("read {}", registry_path.display()))?;
    let value: toml::Value =
        toml::from_str(&text).with_context(|| format!("parse {}", registry_path.display()))?;
    for artifact in value
        .get("artifact")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let Some(table) = artifact.as_table() else {
            continue;
        };
        let field = |name: &str| {
            table
                .get(name)
                .and_then(toml::Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        if field("status") != "downloaded" {
            continue;
        }
        if let Some(items) = table
            .get("downloaded_paths")
            .and_then(toml::Value::as_array)
        {
            for path in items.iter().filter_map(toml::Value::as_str) {
                seen.push((
                    field("id"),
                    field("key"),
                    "downloaded".to_string(),
                    path.to_string(),
                ));
            }
        }
    }

    seen.sort();
    seen.dedup();
    let rows = seen
        .iter()
        .map(|(id, key, status, path)| {
            source_provenance::observe_host_materialization(
                repo_root, &retention, id, key, status, path,
            )
        })
        .collect::<Vec<_>>();
    let generated_at = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let manifest = source_provenance::render_host_materialization(&rows, &generated_at);
    source_provenance::write_host_materialization(&manifest_path, &manifest)?;
    let present = rows.iter().filter(|row| row.present).count();
    println!(
        "Host materialization: rows={} present={} absent={} manifest={}",
        rows.len(),
        present,
        rows.len() - present,
        to_repo_display_path(repo_root, &manifest_path)
    );
    Ok(())
}

fn run_index_control_plane(
    repo_root: &Path,
    db_path: &Path,
    args: IndexControlPlaneArgs,
) -> Result<()> {
    let mut store = ProvenanceStore::open(db_path)?;
    let stats = store.reindex_control_plane_from_registries(
        repo_root,
        RegistryImportPaths {
            claims: &repo_path(repo_root, &args.claims),
            insights: &repo_path(repo_root, &args.insights),
            experiments: &repo_path(repo_root, &args.experiments),
            binaries: &repo_path(repo_root, &args.binaries),
            rocq_project: &repo_path(repo_root, &args.rocq_project),
        },
        // db_path is named in both modes so the refusal text identifies the
        // database it protected.
        ReimportOptions {
            allow_destructive_reimport: args.allow_destructive_reimport,
            db_path: Some(db_path),
        },
    )?;
    println!(
        "Indexed canonical control plane: claims={} insights={} experiments={} binaries={} theorems={} indexed_at={}",
        stats.claim_count,
        stats.insight_count,
        stats.experiment_count,
        stats.binary_count,
        stats.theorem_count,
        stats.indexed_at
    );
    Ok(())
}

fn run_export_control_plane(
    repo_root: &Path,
    db_path: &Path,
    args: ExportControlPlaneArgs,
) -> Result<()> {
    let mut store = ProvenanceStore::open(db_path)?;
    let claims = repo_path(repo_root, &args.claims);
    let insights = repo_path(repo_root, &args.insights);
    let experiments = repo_path(repo_root, &args.experiments);
    let binaries = repo_path(repo_root, &args.binaries);
    let theorems = repo_path(repo_root, &args.theorems);
    let theorems_mirror = repo_path(repo_root, &args.theorems_mirror);
    store.export_control_plane_compat_paths(
        repo_root,
        provenance_store::CompatExportPaths {
            claims: &claims,
            insights: &insights,
            experiments: &experiments,
            binaries: &binaries,
            theorems: &theorems,
            theorems_mirror: &theorems_mirror,
        },
    )?;
    let counts = store.control_plane_counts()?;
    println!(
        "Exported control-plane compatibility outputs: claims={} insights={} experiments={} binaries={} theorems={}",
        counts.claim_count,
        counts.insight_count,
        counts.experiment_count,
        counts.binary_count,
        counts.theorem_count
    );
    Ok(())
}

fn run_verify_control_plane(
    repo_root: &Path,
    db_path: &Path,
    args: VerifyControlPlaneArgs,
) -> Result<()> {
    let mut store = ProvenanceStore::open(db_path)?;
    store.verify_control_plane_invariants(repo_root)?;
    let counts = store.control_plane_counts()?;
    if args.verify_compat_exports {
        let claims = repo_path(repo_root, &args.claims);
        let insights = repo_path(repo_root, &args.insights);
        let experiments = repo_path(repo_root, &args.experiments);
        let binaries = repo_path(repo_root, &args.binaries);
        let theorems = repo_path(repo_root, &args.theorems);
        let theorems_mirror = repo_path(repo_root, &args.theorems_mirror);
        store.verify_control_plane_compat_exports_paths(
            repo_root,
            provenance_store::CompatExportPaths {
                claims: &claims,
                insights: &insights,
                experiments: &experiments,
                binaries: &binaries,
                theorems: &theorems,
                theorems_mirror: &theorems_mirror,
            },
        )?;
    }
    println!("{}", render_control_plane_counts(&counts));
    Ok(())
}

fn run_index_external_sources(
    repo_root: &Path,
    db_path: &Path,
    args: IndexExternalSourcesArgs,
) -> Result<()> {
    let mut store = ProvenanceStore::open(db_path)?;
    let (contract_count, dossier_count) = store.reindex_external_sources_from_compat(
        repo_root,
        &repo_path(repo_root, &args.source_contracts),
        &repo_path(repo_root, &args.dossiers_registry),
    )?;
    println!(
        "Indexed external sources into SQLite: source_contracts={} dossiers={}",
        contract_count, dossier_count
    );
    Ok(())
}

fn run_export_external_sources(
    repo_root: &Path,
    db_path: &Path,
    args: ExportExternalSourcesArgs,
) -> Result<()> {
    let mut store = ProvenanceStore::open(db_path)?;
    store.export_external_sources_compat(
        repo_root,
        &repo_path(repo_root, &args.source_contracts),
        &repo_path(repo_root, &args.dossiers_registry),
    )?;
    println!(
        "Exported external-source compatibility outputs: contracts={} dossiers={}",
        args.source_contracts.display(),
        args.dossiers_registry.display()
    );
    Ok(())
}

fn run_verify_external_sources(
    repo_root: &Path,
    db_path: &Path,
    args: VerifyExternalSourcesArgs,
) -> Result<()> {
    let mut store = ProvenanceStore::open(db_path)?;
    store.verify_external_source_invariants(repo_root)?;
    if args.verify_compat_exports {
        store.verify_external_sources_compat_exports(
            repo_root,
            &repo_path(repo_root, &args.source_contracts),
            &repo_path(repo_root, &args.dossiers_registry),
        )?;
    }
    println!("Verified SQLite-backed external-source control plane.");
    Ok(())
}

fn run_update_external_source(
    repo_root: &Path,
    db_path: &Path,
    args: UpdateExternalSourceArgs,
) -> Result<()> {
    if args.path_glob.is_none()
        && args.canonical_url.is_none()
        && args.mirror_urls.is_empty()
        && args.access_class.is_none()
        && args.status.is_none()
        && args.retrieval_method.is_none()
        && args.attempt_deadline_utc.is_none()
        && args.resolution_deadline_utc.is_none()
        && args.blocker_note.is_none()
        && args.evidence_refs.is_empty()
        && args.manual_manifest_refs.is_empty()
        && args.blocked_action_plan.is_empty()
        && args.scientific_validator_refs.is_empty()
    {
        bail!("no external-source fields were provided to update");
    }

    let mut store = ProvenanceStore::open(db_path)?;
    let patch = ExternalSourceContractPatch {
        path_glob: args.path_glob.as_deref(),
        canonical_url: args.canonical_url.as_deref(),
        mirror_urls: (!args.mirror_urls.is_empty()).then_some(args.mirror_urls.as_slice()),
        access_class: args.access_class.as_deref(),
        status: args.status.as_deref(),
        retrieval_method: args.retrieval_method.as_deref(),
        attempt_deadline_utc: args.attempt_deadline_utc.as_deref(),
        resolution_deadline_utc: args.resolution_deadline_utc.as_deref(),
        blocker_note: args.blocker_note.as_deref(),
        evidence_refs: (!args.evidence_refs.is_empty()).then_some(args.evidence_refs.as_slice()),
        manual_manifest_refs: (!args.manual_manifest_refs.is_empty())
            .then_some(args.manual_manifest_refs.as_slice()),
        blocked_action_plan: (!args.blocked_action_plan.is_empty())
            .then_some(args.blocked_action_plan.as_slice()),
        scientific_validator_refs: (!args.scientific_validator_refs.is_empty())
            .then_some(args.scientific_validator_refs.as_slice()),
    };
    let created = if args.create_if_missing {
        store.upsert_external_source_contract(&args.id, patch)?
    } else {
        let updated = store.update_external_source_contract(&args.id, patch)?;
        if !updated {
            bail!("external source contract not found: {}", args.id);
        }
        false
    };
    if !args.create_if_missing && created {
        bail!("external source contract not found: {}", args.id);
    }

    if args.export_after {
        store.export_external_sources_compat(
            repo_root,
            &repo_path(repo_root, &args.source_contracts),
            &repo_path(repo_root, &args.dossiers_registry),
        )?;
    }
    if args.verify_after {
        store.verify_external_source_invariants(repo_root)?;
        store.verify_external_sources_compat_exports(
            repo_root,
            &repo_path(repo_root, &args.source_contracts),
            &repo_path(repo_root, &args.dossiers_registry),
        )?;
    }
    if created {
        println!("Created SQLite-backed external source contract {}", args.id);
    } else {
        println!("Updated SQLite-backed external source contract {}", args.id);
    }
    Ok(())
}

fn run_verify(repo_root: &Path, db_path: &Path, args: VerifyArgs) -> Result<()> {
    let store = ProvenanceStore::open(db_path)?;
    store.verify_invariants(repo_root)?;
    if args.verify_exports {
        let artifact_summary = source_provenance::verify_artifact_source_of_truth(
            repo_root,
            &repo_path(repo_root, &args.artifact_registry),
        )?;
        let infrastructure_summary = source_provenance::verify_source_infrastructure(
            repo_root,
            &repo_path(repo_root, &args.infrastructure),
        )?;
        println!(
            "Verified sqlite + compatibility exports: artifacts={} downloaded={} lanes={}",
            artifact_summary.artifact_count,
            artifact_summary.downloaded_count,
            infrastructure_summary.lane_counts.len()
        );
    } else {
        println!("Verified sqlite provenance invariants.");
    }
    Ok(())
}

fn run_literature_verify(
    repo_root: &Path,
    db_path: &Path,
    args: LiteratureVerifyArgs,
) -> Result<()> {
    let bib_path = repo_path(repo_root, &args.bib);
    let bib_text = fs::read_to_string(&bib_path)
        .with_context(|| format!("read bibliography {}", bib_path.display()))?;
    let hypotheses_path = args
        .hypotheses
        .as_ref()
        .map(|path| repo_path(repo_root, path));
    let hypotheses_text = if let Some(path) = &hypotheses_path {
        Some(
            fs::read_to_string(path)
                .with_context(|| format!("read hypotheses {}", path.display()))?,
        )
    } else {
        None
    };

    let engine = SearchEngine::new(ApiKeys::from_env(), args.tier.into());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime for literature verification")?;

    let verify_report = runtime.block_on(verify_citations(
        &bib_text,
        &engine,
        Duration::from_millis(args.inter_verify_delay_ms),
    ));

    let papers_already_seen = verify_report
        .results
        .iter()
        .filter_map(|result| result.matched_paper.clone())
        .collect::<Vec<_>>();

    let novelty_report = if args.topic.is_some() || hypotheses_text.is_some() {
        Some(runtime.block_on(check_novelty(
            &engine,
            args.topic.as_deref().unwrap_or(""),
            hypotheses_text.as_deref().unwrap_or(""),
            &args.domains,
            &papers_already_seen,
            args.novelty_limit,
            args.similarity_threshold,
        )))
    } else {
        None
    };

    let entry_metadata = parse_bibtex_metadata(&bib_text);
    let created_at = chrono::Utc::now().to_rfc3339();
    let run = LiteratureVerificationRunRecord {
        id: None,
        input_path: to_repo_display_path(repo_root, &bib_path),
        topic: args.topic.clone(),
        hypotheses_path: hypotheses_path
            .as_ref()
            .map(|path| to_repo_display_path(repo_root, path)),
        domains: args.domains.clone(),
        search_queries: novelty_report
            .as_ref()
            .map(|report| report.search_queries.clone())
            .unwrap_or_default(),
        total_entries: verify_report.total,
        verified_count: verify_report.verified,
        suspicious_count: verify_report.suspicious,
        hallucinated_count: verify_report.hallucinated,
        skipped_count: verify_report.skipped,
        integrity_score: verify_report.integrity_score() as f64,
        novelty_score: novelty_report
            .as_ref()
            .map(|report| report.novelty_score as f64),
        novelty_assessment: novelty_report
            .as_ref()
            .map(|report| report.assessment.clone()),
        recommendation: novelty_report
            .as_ref()
            .map(|report| report.recommendation.clone()),
        search_coverage: novelty_report
            .as_ref()
            .map(|report| report.search_coverage.clone()),
        total_papers_retrieved: novelty_report
            .as_ref()
            .map(|report| report.total_papers_retrieved),
        created_at,
    };
    let result_rows = build_literature_result_rows(&verify_report, &entry_metadata);
    let similar_rows = novelty_report
        .as_ref()
        .map(build_literature_similar_rows)
        .unwrap_or_default();

    let mut store = ProvenanceStore::open(db_path)?;
    let run_id = store.record_literature_verification_run(&run, &result_rows, &similar_rows)?;

    println!("Recorded literature verification run {run_id}");
    println!("Input: {}", run.input_path);
    println!(
        "Totals: total={} verified={} suspicious={} hallucinated={} skipped={} integrity_score={:.3}",
        verify_report.total,
        verify_report.verified,
        verify_report.suspicious,
        verify_report.hallucinated,
        verify_report.skipped,
        verify_report.integrity_score(),
    );
    if let Some(novelty) = novelty_report {
        println!(
            "Novelty: score={:.3} assessment={} recommendation={} coverage={} similar_papers={} retrieved={}",
            novelty.novelty_score,
            novelty.assessment,
            novelty.recommendation,
            novelty.search_coverage,
            novelty.similar_papers_found,
            novelty.total_papers_retrieved,
        );
    }
    Ok(())
}

fn run_query(db_path: &Path, args: QueryArgs) -> Result<()> {
    let store = ProvenanceStore::open(db_path)?;
    match args.kind {
        QueryKind::Artifact { needle } => {
            let Some(result) = store.artifact_by_needle(&needle)? else {
                bail!("no artifact matched {needle}");
            };
            print_artifact_query(&result);
        }
        QueryKind::Claim { needle } => {
            let claim = store
                .list_claims()?
                .into_iter()
                .find(|row| {
                    row.id == needle
                        || row
                            .statement
                            .to_lowercase()
                            .contains(&needle.to_lowercase())
                })
                .with_context(|| format!("no claim matched {needle}"))?;
            print_claim_query(&claim);
        }
        QueryKind::Download {
            limit,
            needle,
            host,
            status,
            backend,
        } => {
            let results = store.query_download_jobs(
                limit,
                needle.as_deref(),
                host.as_deref(),
                status.as_deref(),
                backend.as_deref(),
            )?;
            if results.is_empty() {
                bail!("no download jobs matched the requested filters");
            }
            print_download_queries(&results);
        }
        QueryKind::Campaign { limit } => {
            let results = store.recent_download_campaigns(limit)?;
            if results.is_empty() {
                bail!("no download campaigns found");
            }
            print_download_campaigns(&results);
        }
        QueryKind::Document { needle } => {
            let Some(result) = store.document_by_needle(&needle)? else {
                bail!("no document matched {needle}");
            };
            print_document_query(&result);
        }
        QueryKind::Insight { needle } => {
            let insight = store
                .list_insights()?
                .into_iter()
                .find(|row| {
                    row.id == needle || row.title.to_lowercase().contains(&needle.to_lowercase())
                })
                .with_context(|| format!("no insight matched {needle}"))?;
            print_insight_query(&insight);
        }
        QueryKind::Experiment { needle } => {
            let experiment = store
                .list_experiments()?
                .into_iter()
                .find(|row| {
                    row.id == needle
                        || row.title.to_lowercase().contains(&needle.to_lowercase())
                        || row.binary.as_deref() == Some(needle.as_str())
                })
                .with_context(|| format!("no experiment matched {needle}"))?;
            print_experiment_query(&experiment);
        }
        QueryKind::Binary { needle } => {
            let binary = store
                .list_binaries()?
                .into_iter()
                .find(|row| {
                    row.name == needle
                        || row.crate_name == needle
                        || row
                            .description
                            .to_lowercase()
                            .contains(&needle.to_lowercase())
                })
                .with_context(|| format!("no binary matched {needle}"))?;
            print_binary_query(&binary);
        }
        QueryKind::Literature { limit } => {
            let runs = store.recent_literature_verification_runs(limit)?;
            if runs.is_empty() {
                bail!("no literature verification runs found");
            }
            print_literature_runs(&runs);
        }
        QueryKind::Theorem { needle } => {
            let theorem = store
                .list_theorems()?
                .into_iter()
                .find(|row| {
                    row.id == needle || row.title.to_lowercase().contains(&needle.to_lowercase())
                })
                .with_context(|| format!("no theorem matched {needle}"))?;
            print_theorem_query(&theorem);
        }
    }
    Ok(())
}

fn run_doctor(db_path: &Path, args: DoctorArgs) -> Result<()> {
    let store = ProvenanceStore::open(db_path)?;
    let report = store.doctor_report()?;
    let rendered = render_doctor_report(&report, args.format)?;
    if let Some(out) = args.out {
        fs::write(&out, rendered).with_context(|| format!("write {}", out.display()))?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn run_link_audit(db_path: &Path, args: LinkAuditArgs) -> Result<()> {
    let store = ProvenanceStore::open(db_path)?;
    let report = store.doctor_report()?;
    let candidates = store.recovery_candidates(args.limit)?;
    let rendered = render_link_audit(&report, &candidates);
    if let Some(out) = args.out {
        fs::write(&out, rendered).with_context(|| format!("write {}", out.display()))?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn run_recover(db_path: &Path, repo_root: &Path, args: RecoverArgs) -> Result<()> {
    let store = ProvenanceStore::open(db_path)?;
    let candidates = store.recovery_candidates(args.limit)?;
    let rendered = render_recovery_plan(&candidates);
    let out = repo_path(repo_root, &args.out);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(&out, rendered).with_context(|| format!("write {}", out.display()))?;
    println!(
        "Wrote provenance recovery plan: {} candidates={}",
        out.display(),
        candidates.len()
    );
    Ok(())
}

fn run_pantheon_seed(db_path: &Path, repo_root: &Path, args: PantheonSeedArgs) -> Result<()> {
    let mut store = ProvenanceStore::open(db_path)?;
    let summary = store.seed_pantheon_physicsforge_migration(
        &repo_path(repo_root, &args.findings),
        &repo_path(repo_root, &args.overflow),
    )?;
    print_pantheon_seed_summary(&summary);
    Ok(())
}

fn rebuild_compatibility_exports(repo_root: &Path, args: &ExportArgs) -> Result<()> {
    rebuild_compatibility_exports_with_policy(
        repo_root,
        args,
        &source_provenance::ShrinkPolicy::permissive(),
    )?;
    Ok(())
}

/// Stages the master registry, its report, the four lane files and the
/// infrastructure manifest, then renames the whole set in one pass. A shrink
/// beyond the policy threshold aborts before any rename.
fn rebuild_compatibility_exports_with_policy(
    repo_root: &Path,
    args: &ExportArgs,
    policy: &source_provenance::ShrinkPolicy,
) -> Result<source_provenance::BuildSummary> {
    let out_registry = repo_path(repo_root, &args.out_registry);
    let out_artifact_report = repo_path(repo_root, &args.out_artifact_report);
    let out_infrastructure = repo_path(repo_root, &args.out_infrastructure);
    let lane_dir = repo_path(repo_root, &args.lane_dir);
    let out_infrastructure_report = repo_path(repo_root, &args.out_infrastructure_report);

    let retention = source_provenance::RetentionSet::from_git_index(repo_root);
    let mut set = source_provenance::StagedWriteSet::new();
    let (mut summary, master_text) = source_provenance::stage_artifact_source_of_truth(
        repo_root,
        &out_registry,
        &out_artifact_report,
        &retention,
        &mut set,
    )?;
    source_provenance::stage_source_truth_infrastructure(
        repo_root,
        &master_text,
        &out_registry,
        &out_infrastructure,
        &lane_dir,
        &out_infrastructure_report,
        &mut set,
    )?;
    summary.row_counts = set.commit(policy)?;
    Ok(summary)
}

fn resolve_repo_root(path: &Path) -> PathBuf {
    if path == Path::new(".") {
        source_provenance::default_repo_root()
    } else {
        path.to_path_buf()
    }
}

fn repo_path(repo_root: &Path, maybe_relative: &Path) -> PathBuf {
    if maybe_relative.is_absolute() {
        maybe_relative.to_path_buf()
    } else {
        repo_root.join(maybe_relative)
    }
}

fn to_repo_display_path(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn parse_bibtex_metadata(
    bib_text: &str,
) -> std::collections::BTreeMap<String, (Option<String>, Option<String>)> {
    let mut rows = std::collections::BTreeMap::new();
    let entry_head_re = Regex::new(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,").expect("valid regex");
    let mut current_key: Option<String> = None;
    let mut current_doi: Option<String> = None;
    let mut current_arxiv: Option<String> = None;

    for raw_line in bib_text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(captures) = entry_head_re.captures(line) {
            if let Some(key) = current_key.take() {
                rows.insert(key, (current_doi.take(), current_arxiv.take()));
            }
            current_key = captures.get(2).map(|m| m.as_str().to_string());
            current_doi = None;
            current_arxiv = None;
            continue;
        }
        if line == "}" {
            if let Some(key) = current_key.take() {
                rows.insert(key, (current_doi.take(), current_arxiv.take()));
            }
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        let normalized = normalize_bibtex_field(value);
        match name.trim().to_ascii_lowercase().as_str() {
            "doi" => current_doi = Some(normalized),
            "eprint" => current_arxiv = Some(normalized),
            _ => {}
        }
    }
    if let Some(key) = current_key.take() {
        rows.insert(key, (current_doi.take(), current_arxiv.take()));
    }
    rows
}

fn normalize_bibtex_field(value: &str) -> String {
    value
        .trim()
        .trim_end_matches(',')
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim()
        .to_string()
}

fn build_literature_result_rows(
    report: &VerificationReport,
    entry_metadata: &std::collections::BTreeMap<String, (Option<String>, Option<String>)>,
) -> Vec<LiteratureVerificationResultRecord> {
    report
        .results
        .iter()
        .map(|result| {
            let (doi, arxiv_id) = entry_metadata
                .get(&result.cite_key)
                .cloned()
                .unwrap_or((None, None));
            LiteratureVerificationResultRecord {
                id: None,
                run_id: None,
                cite_key: result.cite_key.clone(),
                title: result.title.clone(),
                status: result.status.as_str().to_string(),
                confidence: result.confidence as f64,
                method: result.method.clone(),
                details: result.details.clone(),
                doi,
                arxiv_id,
                matched_paper_title: result
                    .matched_paper
                    .as_ref()
                    .map(|paper| paper.title.clone()),
                matched_paper_source: result
                    .matched_paper
                    .as_ref()
                    .map(|paper| paper.source.clone()),
                matched_paper_year: result.matched_paper.as_ref().map(|paper| paper.year as i64),
                matched_paper_url: result.matched_paper.as_ref().map(|paper| paper.url.clone()),
                relevance_score: result.relevance_score.map(|value| value as f64),
            }
        })
        .collect()
}

fn build_literature_similar_rows(
    report: &lit_search::NoveltyReport,
) -> Vec<LiteratureNoveltySimilarPaperRecord> {
    report
        .similar_papers
        .iter()
        .map(|paper| LiteratureNoveltySimilarPaperRecord {
            id: None,
            run_id: None,
            title: paper.title.clone(),
            paper_id: paper.paper_id.clone(),
            year: i64::from(paper.year),
            venue: paper.venue.clone(),
            citation_count: i64::from(paper.citation_count),
            similarity: paper.similarity as f64,
            url: paper.url.clone(),
            cite_key: paper.cite_key.clone(),
        })
        .collect()
}

fn print_artifact_query(result: &ArtifactQueryResult) {
    println!("Artifact: {}", result.artifact.id);
    println!("Key: {}", result.artifact.key);
    println!("Title: {}", result.artifact.title);
    println!("Status: {}", result.artifact.status.as_str());
    println!(
        "Canonical URL: {}",
        result
            .artifact
            .canonical_functional_url
            .as_deref()
            .unwrap_or("")
    );
    println!(
        "Canonical Path: {}",
        result
            .artifact
            .canonical_download_path
            .as_ref()
            .map(|v| v.as_str())
            .unwrap_or("")
    );
    println!("Lanes: {}", result.lanes.join(", "));
    println!("Source Refs: {}", result.artifact.source_refs.join(", "));
    if !result.mirror_observations.is_empty() {
        println!("Mirrors:");
        for mirror in &result.mirror_observations {
            println!("  - [{}] {}", mirror.mirror_kind.as_str(), mirror.url);
        }
    }
}

fn print_document_query(result: &DocumentQueryResult) {
    println!("Document: {}", result.document.id);
    println!("Path: {}", result.document.path);
    println!("Title: {}", result.document.title);
    println!("Kind: {}", result.document.kind);
    println!("Authoring Mode: {}", result.document.authoring_mode);
    println!(
        "Backing: {}",
        result
            .document
            .toml_backing
            .as_ref()
            .map(|v| v.as_str())
            .unwrap_or("")
    );
    println!("Source Refs: {}", result.source_refs.join(", "));
}

fn print_claim_query(claim: &ClaimRecord) {
    println!("Claim: {}", claim.id);
    println!("Status: {}", claim.status);
    println!("Statement: {}", claim.statement);
    println!("Where Stated: {}", claim.where_stated);
    println!("Last Verified: {}", claim.last_verified);
    println!(
        "Formal Proof: {}",
        claim.formal_proof.as_deref().unwrap_or("")
    );
    println!(
        "Status Note: {}",
        claim.status_note.as_deref().unwrap_or("")
    );
}

fn print_download_queries(results: &[DownloadQueryResult]) {
    for result in results {
        println!(
            "Download Job: {}",
            result
                .job
                .id
                .map(|value| value.to_string())
                .unwrap_or_default()
        );
        println!("  URL: {}", result.job.requested_url);
        println!("  Kind: {}", result.job.transfer_kind);
        println!("  Requested Backend: {}", result.job.requested_backend);
        println!("  Route Scheme: {}", result.job.route_scheme);
        println!(
            "  Route Host: {}",
            result.job.route_host.as_deref().unwrap_or("")
        );
        println!("  Route Backends: {}", result.job.route_backends.join(", "));
        println!("  Status: {}", result.job.status);
        println!(
            "  Final URL: {}",
            result.job.final_url.as_deref().unwrap_or("")
        );
        println!(
            "  Output Path: {}",
            result.job.output_path.as_deref().unwrap_or("")
        );
        println!("  Created At: {}", result.job.created_at);
        println!("  Note: {}", result.job.note.as_deref().unwrap_or(""));
        if !result.attempts.is_empty() {
            println!("  Attempts:");
            for attempt in &result.attempts {
                println!(
                    "    - backend={} succeeded={} failure_class={} http_code={} bytes={} is_pdf={} error={}",
                    attempt.backend,
                    attempt.succeeded,
                    attempt.failure_class.as_deref().unwrap_or(""),
                    attempt
                        .http_code
                        .map(|value| value.to_string())
                        .unwrap_or_default(),
                    attempt.bytes,
                    attempt.is_pdf,
                    attempt.error_message.as_deref().unwrap_or("")
                );
            }
        }
    }
}

fn print_download_campaigns(results: &[DownloadCampaignQueryResult]) {
    for result in results {
        println!(
            "Download Campaign: {}",
            result
                .campaign
                .id
                .map(|value| value.to_string())
                .unwrap_or_default()
        );
        println!("  Name: {}", result.campaign.name);
        println!("  Command Kind: {}", result.campaign.command_kind);
        println!("  Input Path: {}", result.campaign.input_path);
        println!(
            "  Out Ledger: {}",
            result.campaign.out_ledger_path.as_deref().unwrap_or("")
        );
        println!(
            "  Dest Dir: {}",
            result.campaign.dest_dir.as_deref().unwrap_or("")
        );
        println!("  Created At: {}", result.campaign.created_at);
        println!("  Note: {}", result.campaign.note.as_deref().unwrap_or(""));
        println!("  Job Count: {}", result.job_count);
        println!("  Success Count: {}", result.success_count);
        println!("  Failure Count: {}", result.failure_count);
    }
}

fn print_insight_query(insight: &InsightRecord) {
    println!("Insight: {}", insight.id);
    println!("Status: {}", insight.status);
    println!("Title: {}", insight.title);
    println!("Claim Refs: {}", insight.claim_refs.join(", "));
}

fn print_experiment_query(experiment: &ExperimentRecord) {
    println!("Experiment: {}", experiment.id);
    println!("Status: {}", experiment.status);
    println!("Title: {}", experiment.title);
    println!("Binary: {}", experiment.binary.as_deref().unwrap_or(""));
    println!("Claim Refs: {}", experiment.claim_refs.join(", "));
}

fn print_binary_query(binary: &BinaryRecord) {
    println!("Binary: {}", binary.name);
    println!("Crate: {}", binary.crate_name);
    println!("Description: {}", binary.description);
    println!("Experiment: {}", binary.experiment.as_deref().unwrap_or(""));
    println!("Source: {}", binary.source);
}

fn print_literature_runs(runs: &[LiteratureVerificationQueryResult]) {
    for run in runs {
        println!(
            "Literature run {} | input={} | created_at={}",
            run.run.id.unwrap_or_default(),
            run.run.input_path,
            run.run.created_at
        );
        println!(
            "  totals: total={} verified={} suspicious={} hallucinated={} skipped={} integrity_score={:.3}",
            run.run.total_entries,
            run.run.verified_count,
            run.run.suspicious_count,
            run.run.hallucinated_count,
            run.run.skipped_count,
            run.run.integrity_score
        );
        if let Some(score) = run.run.novelty_score {
            println!(
                "  novelty: score={:.3} assessment={} recommendation={} coverage={}",
                score,
                run.run.novelty_assessment.as_deref().unwrap_or(""),
                run.run.recommendation.as_deref().unwrap_or(""),
                run.run.search_coverage.as_deref().unwrap_or(""),
            );
        }
        println!(
            "  stored rows: verification_results={} similar_papers={}",
            run.results.len(),
            run.similar_papers.len()
        );
    }
}

fn print_pantheon_seed_summary(summary: &PantheonSeedSummary) {
    println!(
        "Seeded Pantheon/PhysicsForge migration sqlite: db={} findings={} risks={} overflow_tasks={} max_active_overflow={}",
        summary.db_path,
        summary.findings_count,
        summary.risk_count,
        summary.overflow_task_count,
        summary.max_active_overflow
    );
}

fn print_theorem_query(theorem: &TheoremRecord) {
    println!("Stable Theorem ID: {}", theorem.id);
    println!("Legacy Name: {}", theorem.legacy_name);
    println!("Title: {}", theorem.title);
    println!("Proof Path: {}", theorem.proof_path);
    println!("Status: {}", theorem.status);
    println!("Identity Kind: {}", theorem.identity_kind);
    println!("Linked Claims: {}", theorem.linked_claim_ids.join(", "));
    println!("Source: {}", theorem.source);
}

fn render_control_plane_counts(counts: &ControlPlaneCounts) -> String {
    format!(
        "claims={}\ninsights={}\nexperiments={}\ncomplete_experiments={}\nbinaries={}\ntheorems={}\nkernel_checked_claims={}\nproof_files={}",
        counts.claim_count,
        counts.insight_count,
        counts.experiment_count,
        counts.complete_experiment_count,
        counts.binary_count,
        counts.theorem_count,
        counts.kernel_checked_claim_count,
        counts.proof_file_count
    )
}

fn render_doctor_report(report: &DoctorReport, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Text => {
            let failed_hosts = report
                .top_failed_download_hosts
                .iter()
                .map(|entry| format!("{}:{}", entry.key, entry.count))
                .collect::<Vec<_>>()
                .join(", ");
            let active_hosts = report
                .top_active_download_hosts
                .iter()
                .map(|entry| format!("{}:{}", entry.key, entry.count))
                .collect::<Vec<_>>()
                .join(", ");
            let backend_health = report
                .backend_health
                .iter()
                .map(|entry| {
                    format!(
                        "{}(ok={},fail={},bytes={})",
                        entry.backend, entry.success_count, entry.failure_count, entry.total_bytes
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!(
                "generated_at={}\nartifacts={}\ndocuments={}\nmissing_minimum={}\nblocked={}\nunverified={}\ncitation_only={}\nmissing_lane_assignments={}\ndocuments_without_backing={}\ndownload_jobs={}\ndownload_attempts={}\ntop_failed_download_hosts={}\ntop_active_download_hosts={}\nbackend_health={}\nlast_indexed_at={}\nlast_exported_at={}",
                report.generated_at,
                report.artifact_count,
                report.document_count,
                report.missing_minimum_count,
                report.blocked_count,
                report.unverified_count,
                report.citation_only_count,
                report.missing_lane_assignment_count,
                report.documents_without_backing_count,
                report.download_job_count,
                report.download_attempt_count,
                failed_hosts,
                active_hosts,
                backend_health,
                report.last_indexed_at.as_deref().unwrap_or(""),
                report.last_exported_at.as_deref().unwrap_or(""),
            ))
        }
        OutputFormat::Json => serde_json::to_string_pretty(report).context("serialize doctor JSON"),
        OutputFormat::Toml => toml::to_string_pretty(report).context("serialize doctor TOML"),
    }
}

fn render_link_audit(
    report: &DoctorReport,
    candidates: &[provenance_core::ArtifactRecord],
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "link_audit generated_at={}\nartifacts={} missing_minimum={} blocked={} unverified={}\n",
        report.generated_at,
        report.artifact_count,
        report.missing_minimum_count,
        report.blocked_count,
        report.unverified_count
    ));
    if !candidates.is_empty() {
        out.push_str("candidates:\n");
        for artifact in candidates {
            out.push_str(&format!(
                "  - {} [{}] {}\n",
                artifact.id,
                artifact.status.as_str(),
                artifact.title
            ));
        }
    }
    out
}

fn render_recovery_plan(candidates: &[provenance_core::ArtifactRecord]) -> String {
    let mut out = String::new();
    out.push_str("# Rust-generated provenance recovery plan\n\n");
    out.push_str("[plan]\n");
    out.push_str(&format!(
        "generated_at = \"{}\"\n",
        chrono::Utc::now().to_rfc3339()
    ));
    out.push_str(&format!("candidate_count = {}\n\n", candidates.len()));
    for artifact in candidates {
        out.push_str("[[candidate]]\n");
        out.push_str(&format!("id = \"{}\"\n", artifact.id.replace('"', "\\\"")));
        out.push_str(&format!(
            "key = \"{}\"\n",
            artifact.key.replace('"', "\\\"")
        ));
        out.push_str(&format!(
            "title = \"{}\"\n",
            artifact.title.replace('"', "\\\"")
        ));
        out.push_str(&format!("status = \"{}\"\n", artifact.status.as_str()));
        out.push_str(&format!(
            "canonical_functional_url = \"{}\"\n",
            artifact
                .canonical_functional_url
                .as_deref()
                .unwrap_or("")
                .replace('"', "\\\"")
        ));
        out.push('\n');
    }
    out
}
