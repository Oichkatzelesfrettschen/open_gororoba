//! Claims registry consolidation pipeline.
//!
//! Deduplicates, normalizes, enriches, cross-links, and synthesizes the
//! claims registry (registry/claims.toml) into a higher-quality knowledge graph.
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

    let claims_path = cli.registry_dir.join("claims.toml");
    let insights_path = cli.registry_dir.join("insights.toml");
    let experiments_path = cli.registry_dir.join("experiments.toml");
    let conflict_markers_path = cli.registry_dir.join("conflict_markers.toml");

    // Load registries
    let mut claims = match consolidate::load_claims(&claims_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    };

    let insights = consolidate::load_insights(&insights_path).unwrap_or_else(|e| {
        eprintln!("WARNING: {e}");
        Vec::new()
    });

    let experiments = consolidate::load_experiments(&experiments_path).unwrap_or_else(|e| {
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
                write_claims_output(&cli, &claims_path, &claims);
            } else {
                println!("(dry-run: no files written)");
            }
        }
        Command::Enrich => {
            let enriched = consolidate::enrich_metadata(&mut claims, &insights, &experiments);
            println!("Fields enriched: {enriched}");
            if !cli.dry_run {
                write_claims_output(&cli, &claims_path, &claims);
            } else {
                println!("(dry-run: no files written)");
            }
        }
        Command::Crosslink => {
            let added = consolidate::build_crossref_graph(&mut claims, &insights, &experiments);
            println!("Cross-links added: {added}");
            if !cli.dry_run {
                write_claims_output(&cli, &claims_path, &claims);
            } else {
                println!("(dry-run: no files written)");
            }
        }
        Command::Merge => {
            let merged = consolidate::merge_claims(&mut claims);
            println!("Claims merged: {merged}");
            if !cli.dry_run {
                write_claims_output(&cli, &claims_path, &claims);
            } else {
                println!("(dry-run: no files written)");
            }
        }
        Command::Full => {
            let result = consolidate::run_full(&mut claims, &insights, &experiments, &mut markers);
            println!("{result}");
            if !cli.dry_run {
                write_claims_output(&cli, &claims_path, &claims);
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

fn write_claims_output(
    cli: &Cli,
    default_path: &std::path::Path,
    claims: &[consolidate::FullClaimEntry],
) {
    let target = cli.output.as_deref().unwrap_or(default_path);
    match consolidate::write_claims(target, claims) {
        Ok(()) => println!("Updated: {}", target.display()),
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    }
}
