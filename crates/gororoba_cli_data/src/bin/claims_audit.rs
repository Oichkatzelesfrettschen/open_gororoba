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
use provenance_store::{ControlPlaneCompatKind, ProvenanceStore};
use std::{path::PathBuf, process};
use toml::Value;

use gororoba_cli::claims::{
    audit,
    parser::{ClaimRow, parse_claim_rows},
};

#[derive(Parser)]
#[command(name = "claims-audit", about = "Generate claims audit reports")]
struct Cli {
    /// Repository root directory.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// Canonical SQLite control-plane DB used to render the live claims matrix.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,

    /// Path to legacy claims matrix compatibility file (relative to repo root).
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
    let (claims, matrix_label) = load_claim_rows(&repo_root, &cli, &matrix_path);

    let report_type = cli.report.as_deref().unwrap_or("all");

    let output = match report_type {
        "all" => {
            let docs_dir = repo_root.join("docs");
            let doc_corpus = audit::collect_doc_corpus(&docs_dir);
            render_all_audits(&claims, &matrix_label, &doc_corpus, &cli.stale_before)
        }
        "id" => {
            let inv = audit::id_inventory(&claims);
            audit::render_id_inventory(&inv, &matrix_label)
        }
        "status" => {
            let inv = audit::status_inventory(&claims);
            audit::render_status_inventory(&inv, &matrix_label)
        }
        "staleness" => {
            let report = audit::staleness_report(&claims, &cli.stale_before);
            audit::render_staleness_report(&report, &cli.stale_before, &matrix_label)
        }
        "contradictions" => {
            let contras = audit::status_contradictions(&claims);
            audit::render_contradictions(&contras, &matrix_label)
        }
        "bold-tokens" => {
            let tokens = audit::bold_tokens_inventory(&claims);
            audit::render_bold_tokens(&tokens, &matrix_label)
        }
        "priority" => {
            let docs_dir = repo_root.join("docs");
            let doc_corpus = audit::collect_doc_corpus(&docs_dir);
            let prio = audit::priority_ranking(&claims, &doc_corpus);
            audit::render_priority_ranking(&prio, &matrix_label)
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

fn load_claim_rows(
    repo_root: &std::path::Path,
    cli: &Cli,
    matrix_path: &std::path::Path,
) -> (Vec<ClaimRow>, String) {
    let canonical_db_path = repo_root.join(&cli.canonical_db);
    if canonical_db_path.exists() {
        let mut store = ProvenanceStore::open(&canonical_db_path).unwrap_or_else(|e| {
            eprintln!(
                "ERROR: Cannot open canonical DB {}: {e}",
                canonical_db_path.display()
            );
            process::exit(2);
        });
        let claims_toml = store
            .control_plane_compat_text(ControlPlaneCompatKind::Claims)
            .unwrap_or_else(|e| {
                eprintln!(
                    "ERROR: Cannot render claims compatibility text from {}: {e}",
                    canonical_db_path.display()
                );
                process::exit(2);
            });
        let claims = parse_claim_rows_from_toml(&claims_toml).unwrap_or_else(|e| {
            eprintln!("ERROR: Cannot build claims audit rows from canonical DB: {e}");
            process::exit(2);
        });
        return (
            claims,
            "registry/canonical/control_plane.sqlite3 (rendered legacy matrix)".to_string(),
        );
    }

    if !matrix_path.exists() {
        eprintln!("ERROR: Missing matrix: {}", matrix_path.display());
        process::exit(2);
    }

    let matrix_text = std::fs::read_to_string(matrix_path).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot read matrix: {e}");
        process::exit(2);
    });
    (parse_claim_rows(&matrix_text), cli.matrix.clone())
}

fn parse_claim_rows_from_toml(raw: &str) -> Result<Vec<ClaimRow>, String> {
    let value: Value = toml::from_str(raw).map_err(|e| format!("parse claims TOML: {e}"))?;
    let claims = value
        .get("claim")
        .and_then(Value::as_array)
        .ok_or_else(|| "claims array missing".to_string())?;
    let mut out = Vec::with_capacity(claims.len());
    for (idx, claim) in claims.iter().enumerate() {
        let table = claim
            .as_table()
            .ok_or_else(|| "claim row must be table".to_string())?;
        let claim_id = table_str(table, "id").to_string();
        let claim_num = claim_id
            .strip_prefix("C-")
            .ok_or_else(|| format!("invalid claim id {claim_id}"))?
            .parse::<u32>()
            .map_err(|e| format!("parse numeric claim id {claim_id}: {e}"))?;
        let last_verified = table_str(table, "last_verified").to_string();
        let last_verified_date = if last_verified.len() >= 10 {
            Some(last_verified[..10].to_string())
        } else {
            None
        };
        let status = table_str(table, "status").to_string();
        let evidence_notes = table
            .get("what_would_verify_refute")
            .and_then(Value::as_str)
            .or_else(|| table.get("status_note").and_then(Value::as_str))
            .unwrap_or("")
            .to_string();
        out.push(ClaimRow {
            claim_id,
            claim_num,
            claim_text: table_str(table, "statement").to_string(),
            where_stated: table_str(table, "where_stated").to_string(),
            status_cell: format!("**{}**", status),
            status_token: status,
            last_verified,
            last_verified_date,
            evidence_notes,
            lineno: idx + 1,
        });
    }
    Ok(out)
}

fn table_str<'a>(table: &'a toml::map::Map<String, Value>, key: &str) -> &'a str {
    table.get(key).and_then(Value::as_str).unwrap_or("")
}

fn render_all_audits(
    claims: &[ClaimRow],
    matrix_label: &str,
    doc_corpus: &str,
    stale_before: &str,
) -> String {
    let mut output = String::new();
    let inv = audit::id_inventory(claims);
    output.push_str(&audit::render_id_inventory(&inv, matrix_label));
    output.push_str("---\n\n");

    let status = audit::status_inventory(claims);
    output.push_str(&audit::render_status_inventory(&status, matrix_label));
    output.push_str("---\n\n");

    let stale = audit::staleness_report(claims, stale_before);
    output.push_str(&audit::render_staleness_report(
        &stale,
        stale_before,
        matrix_label,
    ));
    output.push_str("---\n\n");

    let contras = audit::status_contradictions(claims);
    output.push_str(&audit::render_contradictions(&contras, matrix_label));
    output.push_str("---\n\n");

    let tokens = audit::bold_tokens_inventory(claims);
    output.push_str(&audit::render_bold_tokens(&tokens, matrix_label));
    output.push_str("---\n\n");

    let prio = audit::priority_ranking(claims, doc_corpus);
    output.push_str(&audit::render_priority_ranking(&prio, matrix_label));
    output
}
