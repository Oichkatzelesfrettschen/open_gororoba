//! Claims verification runner.
//!
//! Replaces 11 Python verification scripts with a single Rust binary.
//! Runs all checks and exits with code 0 (pass) or 2 (fail).
//!
//! Usage:
//!   claims-verify                           # all checks
//!   claims-verify --check metadata          # just matrix metadata
//!   claims-verify --check evidence          # just evidence links
//!   claims-verify --check where-stated      # just Where stated pointers
//!   claims-verify --check tasks             # tasks metadata + consistency
//!   claims-verify --check domains           # domain mapping
//!   claims-verify --check providers         # dataset manifest vs source

use clap::Parser;
use std::path::PathBuf;
use std::process;

use gororoba_cli::claims::verify;

#[derive(Parser)]
#[command(
    name = "claims-verify",
    about = "Verify claims infrastructure integrity"
)]
struct Cli {
    /// Repository root directory.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// Path to claims matrix file (relative to repo root).
    #[arg(long, default_value = "docs/CLAIMS_EVIDENCE_MATRIX.md")]
    matrix: String,

    /// Specific check to run (default: all).
    #[arg(long, value_parser = ["metadata", "evidence", "where-stated", "tasks", "domains", "providers", "all"])]
    check: Option<String>,
}

fn run_check(name: &str, failures: Vec<String>) -> bool {
    if failures.is_empty() {
        eprintln!("OK: {name}");
        true
    } else {
        for (i, msg) in failures.iter().enumerate() {
            if i < 50 {
                eprintln!("FAIL [{name}]: {msg}");
            }
        }
        if failures.len() > 50 {
            eprintln!("... plus {} more failures", failures.len() - 50);
        }
        eprintln!("FAIL: {name} ({} issues)", failures.len());
        false
    }
}

fn main() {
    let cli = Cli::parse();
    let repo_root = cli.repo_root.canonicalize().unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot resolve repo root {:?}: {e}", cli.repo_root);
        process::exit(2);
    });

    let check_type = cli.check.as_deref().unwrap_or("all");

    let read_matrix = || -> String {
        let matrix_path = repo_root.join(&cli.matrix);
        if !matrix_path.exists() {
            eprintln!("ERROR: Missing matrix: {}", matrix_path.display());
            process::exit(2);
        }
        std::fs::read_to_string(&matrix_path).unwrap_or_else(|e| {
            eprintln!("ERROR: Cannot read matrix: {e}");
            process::exit(2);
        })
    };

    let mut all_ok = true;

    match check_type {
        "all" => match verify::run_all_verifications(&repo_root) {
            Ok(summary) => {
                eprintln!("OK: all verification checks passed");
                eprintln!("{summary}");
            }
            Err(failures) => {
                for (i, msg) in failures.iter().enumerate() {
                    if i < 100 {
                        eprintln!("FAIL: {msg}");
                    }
                }
                if failures.len() > 100 {
                    eprintln!("... plus {} more", failures.len() - 100);
                }
                eprintln!("\nTotal failures: {}", failures.len());
                all_ok = false;
            }
        },
        "metadata" => {
            let matrix_text = read_matrix();
            let f = verify::verify_matrix_metadata(&matrix_text);
            if !run_check("matrix_metadata", f) {
                all_ok = false;
            }
        }
        "evidence" => {
            let f = verify::verify_evidence_links(&repo_root);
            if !run_check("evidence_links", f) {
                all_ok = false;
            }
        }
        "where-stated" => {
            let matrix_text = read_matrix();
            let f = verify::verify_where_stated_pointers(&matrix_text);
            if !run_check("where_stated", f) {
                all_ok = false;
            }
        }
        "tasks" => {
            let matrix_text = read_matrix();
            let tasks_path = repo_root.join("docs/CLAIMS_TASKS.md");
            if tasks_path.exists() {
                if let Ok(tasks_text) = std::fs::read_to_string(&tasks_path) {
                    let f = verify::verify_tasks_metadata(&tasks_text);
                    if !run_check("tasks_metadata", f) {
                        all_ok = false;
                    }
                    let f = verify::verify_tasks_consistency(&matrix_text, &tasks_text);
                    if !run_check("tasks_consistency", f) {
                        all_ok = false;
                    }
                    let f = verify::verify_task_artifact_links(&tasks_text, &repo_root);
                    if !run_check("task_artifacts", f) {
                        all_ok = false;
                    }
                }
            } else {
                eprintln!("SKIP: docs/CLAIMS_TASKS.md not found");
            }
        }
        "domains" => {
            let matrix_text = read_matrix();
            let domain_path = repo_root.join("docs/claims/CLAIMS_DOMAIN_MAP.csv");
            if domain_path.exists() {
                if let Ok(csv_text) = std::fs::read_to_string(&domain_path) {
                    let f = verify::verify_domain_mapping(&matrix_text, &csv_text);
                    if !run_check("domain_mapping", f) {
                        all_ok = false;
                    }
                }
            } else {
                eprintln!("SKIP: docs/claims/CLAIMS_DOMAIN_MAP.csv not found");
            }
        }
        "providers" => {
            let manifest_path = repo_root.join("docs/DATASET_MANIFEST.md");
            let fetch_path = repo_root.join("crates/gororoba_cli/src/bin/fetch_datasets.rs");
            if manifest_path.exists() && fetch_path.exists() {
                if let (Ok(manifest), Ok(fetch_src)) = (
                    std::fs::read_to_string(&manifest_path),
                    std::fs::read_to_string(&fetch_path),
                ) {
                    let f = verify::verify_dataset_providers(&manifest, &fetch_src);
                    if !run_check("dataset_providers", f) {
                        all_ok = false;
                    }
                }
            } else {
                eprintln!("SKIP: manifest or fetch source not found");
            }
        }
        _ => unreachable!("clap validates check type"),
    }

    if all_ok {
        process::exit(0);
    } else {
        process::exit(2);
    }
}
