//! No-Dataset-Left-Behind (NDLB) gate (plan P6A.S3.T2).
//!
//! Enforces the invariant that every dataset on disk under data/external/
//! corresponds to a registered row in `registry/datasets.toml`, that every
//! dataset row carries a resolvable `server_ref` into
//! `registry/data_servers.toml`, and that every dataset with
//! `status = "active"` has at least one experiment binding OR is marked
//! `"(synthetic)"` for locally-generated artifacts.
//!
//! Pending / non-fatal findings are emitted as WARN lines; structural
//! violations are emitted as ERROR lines and produce a non-zero exit.
//!
//! This binary is intentionally dependency-light. It reads the four
//! TOML registry files as untyped `toml::Value` trees so schema
//! additions do not immediately break compilation. Strict typed schemas
//! and full dataset inventory walking will arrive with P6A.S2.T3
//! (`dataset-hash-sweep`) and P6A.S3.T3 (wire into governance-gate).

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::Parser;
use toml::Value;

#[derive(Parser)]
#[command(
    name = "ndlb-gate",
    about = "No-Dataset-Left-Behind gate: enforces the dataset/server/experiment invariants of plan Phase 6A."
)]
struct Args {
    /// Registry directory.
    #[arg(long, default_value = "registry")]
    registry_dir: PathBuf,

    /// Data external root to walk for dark-dir detection.
    #[arg(long, default_value = "data/external")]
    data_root: PathBuf,

    /// Treat classified_pending rows as WARN (default) or ERROR.
    #[arg(long, default_value_t = false)]
    strict_pending: bool,

    /// Only WARN; never fail. Useful for early-stage integration.
    #[arg(long, default_value_t = false)]
    warn_only: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let datasets = match load_toml(&args.registry_dir.join("datasets.toml")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: failed to load datasets.toml: {e}");
            return ExitCode::FAILURE;
        }
    };
    let servers = match load_toml(&args.registry_dir.join("data_servers.toml")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: failed to load data_servers.toml: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tombstones = match load_toml(&args.registry_dir.join("tombstone_datasets.toml")) {
        Ok(v) => v,
        Err(_) => Value::Table(toml::map::Map::new()), // optional
    };

    // Build lookup: server ids.
    let server_ids: BTreeSet<String> = servers
        .get("server")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.get("id").and_then(Value::as_str).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // Build lookup: tombstoned dataset ids.
    let tombstoned_ids: BTreeSet<String> = tombstones
        .get("tombstone")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.get("id").and_then(Value::as_str).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let empty: Vec<Value> = Vec::new();
    let dataset_rows: &Vec<Value> = datasets
        .get("dataset")
        .and_then(Value::as_array)
        .unwrap_or(&empty);

    // Build lookup: registered dataset ids and their paths.
    let mut dataset_by_path: BTreeMap<String, &Value> = BTreeMap::new();
    for ds in dataset_rows {
        if let Some(p) = ds.get("local_path").and_then(Value::as_str) {
            dataset_by_path.insert(p.to_string(), ds);
        }
    }

    let mut errors: Vec<String> = Vec::new();
    let mut warns: Vec<String> = Vec::new();

    // Rule A: every dataset row's server_ref must resolve (unless synthetic/unknown).
    for ds in dataset_rows {
        let id = ds.get("id").and_then(Value::as_str).unwrap_or("(no-id)");
        let server = ds
            .get("server_ref")
            .and_then(Value::as_str)
            .unwrap_or("");
        match server {
            "" => errors.push(format!("[NDLB-A] {id}: empty server_ref")),
            "(synthetic)" => {}
            "(unknown)" => {
                warns.push(format!("[NDLB-A] {id}: server_ref is (unknown); triage."));
            }
            s if !server_ids.contains(s) => errors.push(format!(
                "[NDLB-A] {id}: server_ref '{s}' not found in data_servers.toml"
            )),
            _ => {}
        }
    }

    // Rule B: status=active requires experiment_bindings non-empty OR server_ref=(synthetic).
    // Rule C: status=deferred requires tombstone row.
    // Rule D: status=classified_pending is WARN by default, ERROR under --strict-pending.
    for ds in dataset_rows {
        let id = ds.get("id").and_then(Value::as_str).unwrap_or("(no-id)");
        let status = ds
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("");
        let bindings_empty = ds
            .get("experiment_bindings")
            .and_then(Value::as_array)
            .map(|a| a.is_empty())
            .unwrap_or(true);
        let server = ds.get("server_ref").and_then(Value::as_str).unwrap_or("");

        match status {
            "active" => {
                if bindings_empty && server != "(synthetic)" {
                    errors.push(format!(
                        "[NDLB-B] {id}: status=active but experiment_bindings is empty and not synthetic"
                    ));
                }
            }
            "deferred" => {
                if !tombstoned_ids.contains(id) {
                    errors.push(format!(
                        "[NDLB-C] {id}: status=deferred but no row in tombstone_datasets.toml"
                    ));
                }
            }
            "archived" => {
                // Acceptable terminal state.
            }
            "classified_pending" => {
                let msg = format!("[NDLB-D] {id}: status=classified_pending (awaiting triage)");
                if args.strict_pending {
                    errors.push(msg);
                } else {
                    warns.push(msg);
                }
            }
            other => {
                errors.push(format!(
                    "[NDLB-D] {id}: unknown status '{other}' (expected active|deferred|archived|classified_pending)"
                ));
            }
        }
    }

    // Rule E: every on-disk data/external/<subdir> must have a registry row.
    if args.data_root.is_dir() {
        for entry in fs::read_dir(&args.data_root).unwrap_or_else(|e| {
            panic!("ERROR: cannot read {}: {}", args.data_root.display(), e)
        }) {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let ft = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if !ft.is_dir() {
                continue;
            }
            let rel = entry.path();
            let rel_str = rel.to_string_lossy().to_string();
            // Normalize (data/external/X is how we store it in TOML)
            let normalized = rel_str.trim_start_matches("./").to_string();
            if !dataset_by_path.contains_key(&normalized) {
                errors.push(format!(
                    "[NDLB-E] {}: on-disk dataset not registered in datasets.toml",
                    normalized
                ));
            }
        }
    }

    // Emit
    for w in &warns {
        println!("WARN  {}", w);
    }
    for e in &errors {
        println!("ERROR {}", e);
    }
    println!(
        "[ndlb-gate] {} errors, {} warns (datasets={} server_ids={} tombstones={})",
        errors.len(),
        warns.len(),
        dataset_rows.len(),
        server_ids.len(),
        tombstoned_ids.len()
    );

    if !errors.is_empty() && !args.warn_only {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn load_toml(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    toml::from_str::<Value>(&text).map_err(|e| format!("{}: {}", path.display(), e))
}
