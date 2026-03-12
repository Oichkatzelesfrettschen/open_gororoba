use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use gororoba_cli_data::source_provenance;
use provenance_core::{ArtifactQueryResult, DoctorReport, DocumentQueryResult, PantheonSeedSummary};
use provenance_store::ProvenanceStore;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "provenance", about = "SQLite-backed provenance operator CLI")]
struct Cli {
    /// Repository root.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// SQLite database path.
    #[arg(long, default_value = ".cache/provenance/provenance.sqlite3")]
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
struct QueryArgs {
    #[command(subcommand)]
    kind: QueryKind,
}

#[derive(Subcommand, Debug)]
enum QueryKind {
    Artifact { needle: String },
    Document { needle: String },
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
        Commands::Query(args) => run_query(&db_path, args),
        Commands::Doctor(args) => run_doctor(&db_path, args),
        Commands::LinkAudit(args) => run_link_audit(&db_path, args),
        Commands::Recover(args) => run_recover(&db_path, &repo_root, args),
        Commands::PantheonSeed(args) => run_pantheon_seed(&db_path, &repo_root, args),
    }
}

fn run_index(repo_root: &Path, db_path: &Path, args: IndexArgs) -> Result<()> {
    if args.refresh_compat_exports {
        rebuild_compatibility_exports(repo_root, &ExportArgs {
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
        })?;
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
        QueryKind::Document { needle } => {
            let Some(result) = store.document_by_needle(&needle)? else {
                bail!("no document matched {needle}");
            };
            print_document_query(&result);
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

    source_provenance::build_artifact_source_of_truth(repo_root, &out_registry, &out_artifact_report)?;
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

fn render_doctor_report(report: &DoctorReport, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Text => Ok(format!(
            "generated_at={}\nartifacts={}\ndocuments={}\nmissing_minimum={}\nblocked={}\nunverified={}\ncitation_only={}\nmissing_lane_assignments={}\ndocuments_without_backing={}\nlast_indexed_at={}\nlast_exported_at={}",
            report.generated_at,
            report.artifact_count,
            report.document_count,
            report.missing_minimum_count,
            report.blocked_count,
            report.unverified_count,
            report.citation_only_count,
            report.missing_lane_assignment_count,
            report.documents_without_backing_count,
            report.last_indexed_at.as_deref().unwrap_or(""),
            report.last_exported_at.as_deref().unwrap_or(""),
        )),
        OutputFormat::Json => serde_json::to_string_pretty(report).context("serialize doctor JSON"),
        OutputFormat::Toml => toml::to_string_pretty(report).context("serialize doctor TOML"),
    }
}

fn render_link_audit(report: &DoctorReport, candidates: &[provenance_core::ArtifactRecord]) -> String {
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
    out.push_str(&format!("generated_at = \"{}\"\n", chrono::Utc::now().to_rfc3339()));
    out.push_str(&format!("candidate_count = {}\n\n", candidates.len()));
    for artifact in candidates {
        out.push_str("[[candidate]]\n");
        out.push_str(&format!("id = \"{}\"\n", artifact.id.replace('"', "\\\"")));
        out.push_str(&format!("key = \"{}\"\n", artifact.key.replace('"', "\\\"")));
        out.push_str(&format!("title = \"{}\"\n", artifact.title.replace('"', "\\\"")));
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
