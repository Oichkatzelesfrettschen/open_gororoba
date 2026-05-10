//! Claims registry consolidation pipeline.
//!
//! Deduplicates, normalizes, enriches, cross-links, and synthesizes the
//! claims registry into a higher-quality knowledge graph.
//!
//! Usage:
//!   claims-consolidate analyze      # Read-only analysis report
//!   claims-consolidate normalize    # Normalize statuses
//!   claims-consolidate enrich       # Auto-fill metadata
//!   claims-consolidate crosslink    # Build cross-reference graph
//!   claims-consolidate merge        # Execute pre-identified merges
//!   claims-consolidate full         # Run all steps in sequence

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use gororoba_cli::claims::consolidate;
use provenance_store::{ControlPlaneCompatKind, ProvenanceStore};

#[derive(Parser)]
#[command(
    name = "claims-consolidate",
    about = "Claims registry consolidation pipeline"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Registry directory.
    #[arg(long, default_value = "registry", global = true)]
    registry_dir: PathBuf,

    /// Canonical control-plane database. When present, claims/insights/experiments
    /// load from SQLite-backed compatibility text first.
    #[arg(
        long,
        default_value = "registry/canonical/control_plane.sqlite3",
        global = true
    )]
    canonical_db: PathBuf,

    /// Report what would change without writing.
    #[arg(long, default_value_t = false, global = true)]
    dry_run: bool,

    /// Write output to a specific file instead of in-place.
    #[arg(long, global = true)]
    output: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Read-only analysis report.
    Analyze,
    /// Normalize statuses (case, variant collapse).
    Normalize,
    /// Auto-fill metadata (phase, confidence, insight links).
    Enrich,
    /// Build cross-reference graph (bidirectional claim links).
    Crosslink,
    /// Execute pre-identified claim merges.
    Merge,
    /// Run all steps in sequence (normalize -> enrich -> crosslink -> merge).
    Full,
}

fn main() {
    let cli = Cli::parse();

    let repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let claims_path = cli.registry_dir.join("claims.toml");
    let insights_path = cli.registry_dir.join("insights.toml");
    let experiments_path = cli.registry_dir.join("experiments.toml");
    let binaries_path = cli.registry_dir.join("binaries.toml");
    let conflict_markers_path = cli.registry_dir.join("conflict_markers.toml");
    let proofs_project_path = repo_root.join("proofs/_RocqProject");

    // Load registries
    let claims_text = match load_registry_compat_text(
        &cli.canonical_db,
        &claims_path,
        ControlPlaneCompatKind::Claims,
    ) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    };
    let mut claims =
        match consolidate::load_claims_from_str(&claims_text, &claims_path.display().to_string()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ERROR: {e}");
                std::process::exit(1);
            }
        };

    let insights = load_registry_compat_text(
        &cli.canonical_db,
        &insights_path,
        ControlPlaneCompatKind::Insights,
    )
    .and_then(|text| {
        consolidate::load_insights_from_str(&text, &insights_path.display().to_string())
    })
    .unwrap_or_else(|e| {
        eprintln!("WARNING: {e}");
        Vec::new()
    });

    let experiments = load_registry_compat_text(
        &cli.canonical_db,
        &experiments_path,
        ControlPlaneCompatKind::Experiments,
    )
    .and_then(|text| {
        consolidate::load_experiments_from_str(&text, &experiments_path.display().to_string())
    })
    .unwrap_or_else(|e| {
        eprintln!("WARNING: {e}");
        Vec::new()
    });

    let (cm_header, mut markers) = if conflict_markers_path.exists() {
        consolidate::load_conflict_markers(&conflict_markers_path).unwrap_or_else(|e| {
            eprintln!("WARNING: {e}");
            (None, Vec::new())
        })
    } else {
        (None, Vec::new())
    };

    match cli.command {
        Command::Analyze => {
            let report = consolidate::analyze(&claims, &insights, &experiments, &markers);
            println!("{report}");
        }
        Command::Normalize => {
            let modified = consolidate::normalize_all_statuses(&mut claims);
            println!("Statuses normalized: {modified}");
            if !cli.dry_run {
                write_claims_output(
                    &cli,
                    &repo_root,
                    &claims_path,
                    &insights_path,
                    &experiments_path,
                    &binaries_path,
                    &proofs_project_path,
                    &claims,
                );
            } else {
                println!("(dry-run: no files written)");
            }
        }
        Command::Enrich => {
            let enriched = consolidate::enrich_metadata(&mut claims, &insights, &experiments);
            println!("Fields enriched: {enriched}");
            if !cli.dry_run {
                write_claims_output(
                    &cli,
                    &repo_root,
                    &claims_path,
                    &insights_path,
                    &experiments_path,
                    &binaries_path,
                    &proofs_project_path,
                    &claims,
                );
            } else {
                println!("(dry-run: no files written)");
            }
        }
        Command::Crosslink => {
            let added = consolidate::build_crossref_graph(&mut claims, &insights, &experiments);
            println!("Cross-links added: {added}");
            if !cli.dry_run {
                write_claims_output(
                    &cli,
                    &repo_root,
                    &claims_path,
                    &insights_path,
                    &experiments_path,
                    &binaries_path,
                    &proofs_project_path,
                    &claims,
                );
            } else {
                println!("(dry-run: no files written)");
            }
        }
        Command::Merge => {
            let merged = consolidate::merge_claims(&mut claims);
            println!("Claims merged: {merged}");
            if !cli.dry_run {
                write_claims_output(
                    &cli,
                    &repo_root,
                    &claims_path,
                    &insights_path,
                    &experiments_path,
                    &binaries_path,
                    &proofs_project_path,
                    &claims,
                );
            } else {
                println!("(dry-run: no files written)");
            }
        }
        Command::Full => {
            let result = consolidate::run_full(&mut claims, &insights, &experiments, &mut markers);
            println!("{result}");
            if !cli.dry_run {
                write_claims_output(
                    &cli,
                    &repo_root,
                    &claims_path,
                    &insights_path,
                    &experiments_path,
                    &binaries_path,
                    &proofs_project_path,
                    &claims,
                );
                // Also write updated conflict markers
                if let Err(e) = consolidate::write_conflict_markers(
                    &conflict_markers_path,
                    &cm_header,
                    &markers,
                ) {
                    eprintln!("ERROR writing conflict markers: {e}");
                } else {
                    println!("Updated: {}", conflict_markers_path.display());
                }
            } else {
                println!("(dry-run: no files written)");
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_claims_output(
    cli: &Cli,
    repo_root: &std::path::Path,
    default_path: &std::path::Path,
    insights_path: &std::path::Path,
    experiments_path: &std::path::Path,
    binaries_path: &std::path::Path,
    proofs_project_path: &std::path::Path,
    claims: &[consolidate::FullClaimEntry],
) {
    let target = cli.output.as_deref().unwrap_or(default_path);
    match consolidate::write_claims(target, claims) {
        Ok(()) => {
            println!("Updated: {}", target.display());
            if cli.output.is_none() {
                sync_control_plane_after_claim_write(
                    &cli.canonical_db,
                    repo_root,
                    target,
                    insights_path,
                    experiments_path,
                    binaries_path,
                    proofs_project_path,
                );
            } else {
                println!(
                    "SKIP: canonical DB sync disabled because --output wrote a non-canonical file."
                );
            }
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    }
}

fn load_registry_compat_text(
    canonical_db: &std::path::Path,
    fallback_path: &std::path::Path,
    kind: ControlPlaneCompatKind,
) -> Result<String, String> {
    if canonical_db.exists() {
        let mut store = ProvenanceStore::open(canonical_db)
            .map_err(|err| format!("open canonical db {}: {err}", canonical_db.display()))?;
        return store.control_plane_compat_text(kind).map_err(|err| {
            format!(
                "render {:?} compatibility text from canonical db {}: {err}",
                kind,
                canonical_db.display()
            )
        });
    }
    std::fs::read_to_string(fallback_path)
        .map_err(|err| format!("Failed to read {}: {err}", fallback_path.display()))
}

fn sync_control_plane_after_claim_write(
    canonical_db: &std::path::Path,
    repo_root: &std::path::Path,
    claims_path: &std::path::Path,
    insights_path: &std::path::Path,
    experiments_path: &std::path::Path,
    binaries_path: &std::path::Path,
    proofs_project_path: &std::path::Path,
) {
    if !canonical_db.exists() {
        println!(
            "SKIP: canonical DB {} does not exist; claims output left as compatibility TOML only.",
            canonical_db.display()
        );
        return;
    }
    let theorems_path = repo_root.join("docs/THEOREMS.md");
    let theorems_mirror_path = repo_root.join("docs/generated/THEOREMS_REGISTRY_MIRROR.md");
    let mut store = match ProvenanceStore::open(canonical_db) {
        Ok(store) => store,
        Err(err) => {
            eprintln!(
                "ERROR: open canonical db {} for post-write sync: {err}",
                canonical_db.display()
            );
            std::process::exit(1);
        }
    };
    if let Err(err) = store.reindex_control_plane_from_registries(
        repo_root,
        claims_path,
        insights_path,
        experiments_path,
        binaries_path,
        proofs_project_path,
    ) {
        eprintln!("ERROR: reindex canonical control plane after claims write: {err}");
        std::process::exit(1);
    }
    if let Err(err) = store.export_control_plane_compat(
        repo_root,
        provenance_store::CompatExportPaths {
            claims: claims_path,
            insights: insights_path,
            experiments: experiments_path,
            binaries: binaries_path,
            theorems: &theorems_path,
            theorems_mirror: &theorems_mirror_path,
        },
    ) {
        eprintln!("ERROR: export control-plane compatibility after claims write: {err}");
        std::process::exit(1);
    }
    println!(
        "Synchronized canonical DB {} after in-place claims update.",
        canonical_db.display()
    );
}
