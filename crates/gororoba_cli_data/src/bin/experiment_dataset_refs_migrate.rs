//! Experiment `dataset_refs` migration binary (plan P6A.S4.T4).
//!
//! Populates the `dataset_refs = [...]` field inside
//! `experiments_cp.compat_toml_text` in
//! `registry/canonical/control_plane.sqlite3`. Reads a migration
//! mapping TOML describing which dataset IDs (PC- / EX- / XS-) to
//! attach to each experiment E-id.
//!
//! RCA-style design choices:
//! - Writes only `compat_toml_text`, not any other column. The
//!   downstream TOML exporter regenerates experiments.toml from this
//!   blob, so the patch propagates through the canonical chain.
//! - Line-oriented edit of the blob preserves ordering and comments.
//! - Idempotent: running twice yields identical state (no duplicate
//!   IDs added).
//! - --dry-run mode prints the would-be diff without touching the DB.
//!
//! Mapping file schema (TOML):
//!
//!   [[migration]]
//!   experiment_id = "E-003"
//!   dataset_refs = ["EX-0029", "EX-0031"]
//!   rationale = "Pantheon+ primary + DESI BAO secondary"

use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::Parser;
use rusqlite::{Connection, params};
use toml::Value;

const DEFAULT_DB: &str = "registry/canonical/control_plane.sqlite3";
const DEFAULT_MAPPING: &str = "registry/experiment_dataset_refs_migration.toml";

#[derive(Parser)]
#[command(
    name = "experiment-dataset-refs-migrate",
    about = "Patch dataset_refs into experiments_cp.compat_toml_text (plan P6A.S4.T4)."
)]
struct Args {
    #[arg(long, default_value = DEFAULT_DB)]
    db: PathBuf,

    #[arg(long, default_value = DEFAULT_MAPPING)]
    mapping: PathBuf,

    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let mapping = match load_mapping(&args.mapping) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: load mapping {}: {e}", args.mapping.display());
            return ExitCode::FAILURE;
        }
    };

    if mapping.is_empty() {
        eprintln!("nothing to migrate (mapping file empty)");
        return ExitCode::SUCCESS;
    }

    let mut conn = match Connection::open(&args.db) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: open {}: {e}", args.db.display());
            return ExitCode::FAILURE;
        }
    };

    let tx = match conn.transaction() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: begin tx: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut patched = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    for entry in &mapping {
        match process_one(&tx, &entry.experiment_id, &entry.dataset_refs, args.dry_run) {
            Ok(true) => {
                patched += 1;
                println!(
                    "[migrate] {} <- {}{}",
                    entry.experiment_id,
                    entry.dataset_refs.join(","),
                    if args.dry_run { " (dry-run)" } else { "" }
                );
            }
            Ok(false) => {
                skipped += 1;
                println!(
                    "[migrate] {} already populated, skipped",
                    entry.experiment_id
                );
            }
            Err(e) => {
                failed += 1;
                eprintln!("[migrate] {}: FAIL {e}", entry.experiment_id);
            }
        }
    }

    if args.dry_run {
        println!(
            "[migrate] dry-run summary: {} would patch, {} already populated, {} failed",
            patched, skipped, failed
        );
    } else if let Err(e) = tx.commit() {
        eprintln!("ERROR: commit: {e}");
        return ExitCode::FAILURE;
    } else {
        println!(
            "[migrate] {} patched, {} skipped, {} failed (committed)",
            patched, skipped, failed
        );
    }

    if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[derive(Debug, Clone)]
struct MigrationEntry {
    experiment_id: String,
    dataset_refs: Vec<String>,
}

fn load_mapping(path: &Path) -> Result<Vec<MigrationEntry>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let doc: Value = toml::from_str(&text).map_err(|e| format!("{}: {}", path.display(), e))?;
    let empty: Vec<Value> = Vec::new();
    let rows = doc
        .get("migration")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let mut out = Vec::new();
    for r in rows {
        let experiment_id = r
            .get("experiment_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing experiment_id".to_string())?
            .to_string();
        let dataset_refs = r
            .get("dataset_refs")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{experiment_id}: missing dataset_refs"))?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect::<Vec<_>>();
        if dataset_refs.is_empty() {
            return Err(format!("{experiment_id}: empty dataset_refs not allowed"));
        }
        out.push(MigrationEntry {
            experiment_id,
            dataset_refs,
        });
    }
    Ok(out)
}

/// Returns Ok(true) if a patch was applied (or would be in dry-run),
/// Ok(false) if the row already had non-empty dataset_refs.
fn process_one(
    tx: &rusqlite::Transaction,
    experiment_id: &str,
    new_refs: &[String],
    dry_run: bool,
) -> Result<bool, String> {
    let compat: String = tx
        .query_row(
            "SELECT compat_toml_text FROM experiments_cp WHERE id = ?1",
            params![experiment_id],
            |row| row.get(0),
        )
        .map_err(|e| format!("select: {e}"))?;

    let patched = match patch_dataset_refs(&compat, new_refs) {
        PatchOutcome::AlreadyPopulated => return Ok(false),
        PatchOutcome::Missing => {
            return Err("compat_toml_text has no dataset_refs line".to_string());
        }
        PatchOutcome::Patched(s) => s,
    };

    if !dry_run {
        tx.execute(
            "UPDATE experiments_cp SET compat_toml_text = ?1 WHERE id = ?2",
            params![patched, experiment_id],
        )
        .map_err(|e| format!("update: {e}"))?;
    }
    Ok(true)
}

enum PatchOutcome {
    Missing,
    AlreadyPopulated,
    Patched(String),
}

/// Line-oriented edit: find `dataset_refs = []` (whitespace-tolerant)
/// and replace with `dataset_refs = [...]`. If the existing list is
/// non-empty, return AlreadyPopulated.
fn patch_dataset_refs(compat: &str, new_refs: &[String]) -> PatchOutcome {
    let mut out = String::with_capacity(compat.len() + 128);
    let mut found = false;
    let mut already = false;
    for line in compat.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("dataset_refs") && trimmed.contains('=') {
            found = true;
            if trimmed.contains("dataset_refs = []")
                || trimmed.contains("dataset_refs=[]")
                || trimmed.contains("dataset_refs = [ ]")
            {
                let ids = new_refs
                    .iter()
                    .map(|s| format!("\"{s}\""))
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push_str(&format!("dataset_refs = [{ids}]\n"));
                continue;
            } else {
                // Has content already -- don't clobber.
                already = true;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    if !found {
        PatchOutcome::Missing
    } else if already {
        PatchOutcome::AlreadyPopulated
    } else {
        PatchOutcome::Patched(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_empty_refs() {
        let compat = "id = \"E-003\"\ndataset_refs = []\nstatus = \"active\"\n";
        let new = vec!["EX-0029".to_string(), "EX-0031".to_string()];
        match patch_dataset_refs(compat, &new) {
            PatchOutcome::Patched(s) => {
                assert!(s.contains("dataset_refs = [\"EX-0029\", \"EX-0031\"]"));
                assert!(s.contains("id = \"E-003\""));
                assert!(s.contains("status = \"active\""));
            }
            _ => panic!("expected Patched"),
        }
    }

    #[test]
    fn patch_skips_populated() {
        let compat = "dataset_refs = [\"PC-0001\"]\n";
        match patch_dataset_refs(compat, &["EX-0029".to_string()]) {
            PatchOutcome::AlreadyPopulated => {}
            _ => panic!("expected AlreadyPopulated"),
        }
    }

    #[test]
    fn patch_missing_line() {
        let compat = "id = \"E-003\"\nstatus = \"active\"\n";
        match patch_dataset_refs(compat, &["EX-0029".to_string()]) {
            PatchOutcome::Missing => {}
            _ => panic!("expected Missing"),
        }
    }
}
