//! Claims audit report generator.
//!
//! Replaces 7 Python analysis scripts with a single Rust binary.
//! Generates combined or individual audit reports.
//!
//! Usage:
//!   claims-audit                           # all reports to stdout
//!   claims-audit --report id               # just ID inventory
//!   claims-audit --report status           # just status inventory
//!   claims-audit --report staleness        # just staleness report
//!   claims-audit --report contradictions   # just contradictions
//!   claims-audit --report bold-tokens      # just bold tokens
//!   claims-audit --report priority         # just priority ranking
//!   claims-audit --out reports/audit.md    # write to file

use clap::Parser;
use std::path::PathBuf;
use std::process;

use gororoba_cli::claims::audit;
use gororoba_cli::claims::parser::parse_claim_rows;

#[derive(Parser)]
#[command(name = "claims-audit", about = "Generate claims audit reports")]
struct Cli {
    /// Repository root directory.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// Path to claims matrix file (relative to repo root).
    #[arg(long, default_value = "docs/CLAIMS_EVIDENCE_MATRIX.md")]
    matrix: String,

    /// Specific report to generate (default: all).
    #[arg(long, value_parser = ["id", "status", "staleness", "contradictions", "bold-tokens", "priority", "all"])]
    report: Option<String>,

    /// Staleness threshold date (claims verified before this are stale).
    #[arg(long, default_value = "2025-06-01")]
    stale_before: String,

    /// Output file (default: stdout).
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();
    let repo_root = cli.repo_root.canonicalize().unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot resolve repo root {:?}: {e}", cli.repo_root);
        process::exit(2);
    });

    let matrix_path = repo_root.join(&cli.matrix);
    if !matrix_path.exists() {
        eprintln!("ERROR: Missing matrix: {}", matrix_path.display());
        process::exit(2);
    }

    let matrix_text = std::fs::read_to_string(&matrix_path).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot read matrix: {e}");
        process::exit(2);
    });

    let report_type = cli.report.as_deref().unwrap_or("all");

    let output = match report_type {
        "all" => {
            let docs_dir = repo_root.join("docs");
            let doc_corpus = audit::collect_doc_corpus(&docs_dir);
            audit::run_all_audits(
                &matrix_text,
                &cli.matrix,
                &doc_corpus,
                &cli.stale_before,
            )
        }
        "id" => {
            let claims = parse_claim_rows(&matrix_text);
            let inv = audit::id_inventory(&claims);
            audit::render_id_inventory(&inv, &cli.matrix)
        }
        "status" => {
            let claims = parse_claim_rows(&matrix_text);
            let inv = audit::status_inventory(&claims);
            audit::render_status_inventory(&inv, &cli.matrix)
        }
        "staleness" => {
            let claims = parse_claim_rows(&matrix_text);
            let report = audit::staleness_report(&claims, &cli.stale_before);
            audit::render_staleness_report(&report, &cli.stale_before, &cli.matrix)
        }
        "contradictions" => {
            let claims = parse_claim_rows(&matrix_text);
            let contras = audit::status_contradictions(&claims);
            audit::render_contradictions(&contras, &cli.matrix)
        }
        "bold-tokens" => {
            let claims = parse_claim_rows(&matrix_text);
            let tokens = audit::bold_tokens_inventory(&claims);
            audit::render_bold_tokens(&tokens, &cli.matrix)
        }
        "priority" => {
            let claims = parse_claim_rows(&matrix_text);
            let docs_dir = repo_root.join("docs");
            let doc_corpus = audit::collect_doc_corpus(&docs_dir);
            let prio = audit::priority_ranking(&claims, &doc_corpus);
            audit::render_priority_ranking(&prio, &cli.matrix)
        }
        _ => unreachable!("clap validates report type"),
    };

    match cli.out {
        Some(out_path) => {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(&out_path, &output).unwrap_or_else(|e| {
                eprintln!("ERROR: Cannot write {}: {e}", out_path.display());
                process::exit(2);
            });
            eprintln!("Wrote: {}", out_path.display());
        }
        None => print!("{output}"),
    }
}
