use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use provenance_core::{
    ArtifactQueryResult, BinaryRecord, ClaimRecord, ControlPlaneCounts, DoctorReport,
    DocumentQueryResult, DownloadCampaignQueryResult, DownloadQueryResult, ExperimentRecord,
    InsightRecord, PantheonSeedSummary, TheoremRecord,
};
use provenance_ops::source_provenance;
use provenance_store::{ExternalSourceContractPatch, ProvenanceStore};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
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
    /// Rebuild compatibility registry/report exports using the Rust seam.
    Export(ExportArgs),
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
    UpdateExternalSource(UpdateExternalSourceArgs),
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

    /// Reindex the SQLite store after exporting compatibility files.
    #[arg(long, default_value_t = true)]
    reindex_after: bool,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Toml,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = resolve_repo_root(&cli.repo_root);
    let db_path = repo_path(&repo_root, &cli.db);
    match cli.command {
        Commands::Index(args) => run_index(&repo_root, &db_path, args),
        Commands::Export(args) => run_export(&repo_root, &db_path, args),
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
            run_update_external_source(&repo_root, &db_path, args)
        }
        Commands::Query(args) => run_query(&db_path, args),
        Commands::Doctor(args) => run_doctor(&db_path, args),
        Commands::LinkAudit(args) => run_link_audit(&db_path, args),
        Commands::Recover(args) => run_recover(&db_path, &repo_root, args),
        Commands::PantheonSeed(args) => run_pantheon_seed(&db_path, &repo_root, args),
    }
}

fn run_index(repo_root: &Path, db_path: &Path, args: IndexArgs) -> Result<()> {
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

fn run_export(repo_root: &Path, db_path: &Path, args: ExportArgs) -> Result<()> {
    rebuild_compatibility_exports(repo_root, &args)?;
    if args.reindex_after {
        let mut store = ProvenanceStore::open(db_path)?;
        let stats = store.reindex_from_registries(
            repo_root,
            &repo_path(repo_root, &args.out_registry),
            &repo_path(repo_root, &PathBuf::from("registry/knowledge_sources.toml")),
            &repo_path(repo_root, &args.lane_dir),
        )?;
        store.record_export_run(
            "export",
            stats.artifact_count,
            stats.document_count,
            &json!({
                "artifact_registry": args.out_registry,
                "infrastructure": args.out_infrastructure,
                "lane_dir": args.lane_dir,
            })
            .to_string(),
        )?;
        println!(
            "Re-exported compatibility outputs and refreshed sqlite: artifacts={} documents={}",
            stats.artifact_count, stats.document_count
        );
    } else {
        println!(
            "Re-exported compatibility outputs: registry={} infrastructure={}",
            args.out_registry.display(),
            args.out_infrastructure.display()
        );
    }
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
        &repo_path(repo_root, &args.claims),
        &repo_path(repo_root, &args.insights),
        &repo_path(repo_root, &args.experiments),
        &repo_path(repo_root, &args.binaries),
        &repo_path(repo_root, &args.rocq_project),
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
    store.export_control_plane_compat(
        repo_root,
        &repo_path(repo_root, &args.claims),
        &repo_path(repo_root, &args.insights),
        &repo_path(repo_root, &args.experiments),
        &repo_path(repo_root, &args.binaries),
        &repo_path(repo_root, &args.theorems),
        &repo_path(repo_root, &args.theorems_mirror),
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
        store.verify_control_plane_compat_exports(
            repo_root,
            &repo_path(repo_root, &args.claims),
            &repo_path(repo_root, &args.insights),
            &repo_path(repo_root, &args.experiments),
            &repo_path(repo_root, &args.binaries),
            &repo_path(repo_root, &args.theorems),
            &repo_path(repo_root, &args.theorems_mirror),
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
        && args.access_class.is_none()
        && args.status.is_none()
        && args.retrieval_method.is_none()
        && args.attempt_deadline_utc.is_none()
        && args.resolution_deadline_utc.is_none()
        && args.blocker_note.is_none()
    {
        bail!("no external-source fields were provided to update");
    }

    let mut store = ProvenanceStore::open(db_path)?;
    let patch = ExternalSourceContractPatch {
        path_glob: args.path_glob.as_deref(),
        canonical_url: args.canonical_url.as_deref(),
        access_class: args.access_class.as_deref(),
        status: args.status.as_deref(),
        retrieval_method: args.retrieval_method.as_deref(),
        attempt_deadline_utc: args.attempt_deadline_utc.as_deref(),
        resolution_deadline_utc: args.resolution_deadline_utc.as_deref(),
        blocker_note: args.blocker_note.as_deref(),
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
    let out_registry = repo_path(repo_root, &args.out_registry);
    let out_artifact_report = repo_path(repo_root, &args.out_artifact_report);
    let out_infrastructure = repo_path(repo_root, &args.out_infrastructure);
    let lane_dir = repo_path(repo_root, &args.lane_dir);
    let out_infrastructure_report = repo_path(repo_root, &args.out_infrastructure_report);

    source_provenance::build_artifact_source_of_truth(
        repo_root,
        &out_registry,
        &out_artifact_report,
    )?;
    source_provenance::build_source_truth_infrastructure(
        repo_root,
        &out_registry,
        &out_infrastructure,
        &lane_dir,
        &out_infrastructure_report,
    )?;
    Ok(())
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
    println!("Theorem: {}", theorem.id);
    println!("Title: {}", theorem.title);
    println!("Proof Path: {}", theorem.proof_path);
    println!("Status: {}", theorem.status);
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
