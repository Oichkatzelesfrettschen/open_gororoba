use clap::Parser;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(about = "Verify project_csv split policy and TOML scroll coverage")]
struct Args {
    /// Repository root directory.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// CSV inventory registry path.
    #[arg(long, default_value = "registry/csv_inventory.toml")]
    inventory: PathBuf,

    /// Project CSV split policy path.
    #[arg(long, default_value = "registry/project_csv_split_policy.toml")]
    policy: PathBuf,

    /// Canonical manifest path.
    #[arg(
        long,
        default_value = "registry/manifests/project_csv_canonical_manifest.txt"
    )]
    canonical_manifest: PathBuf,

    /// Generated manifest path.
    #[arg(
        long,
        default_value = "registry/manifests/project_csv_generated_manifest.txt"
    )]
    generated_manifest: PathBuf,

    /// Canonical index path.
    #[arg(long, default_value = "registry/project_csv_canonical_datasets.toml")]
    canonical_index: PathBuf,

    /// Generated index path.
    #[arg(long, default_value = "registry/project_csv_generated_artifacts.toml")]
    generated_index: PathBuf,
}

#[derive(Debug, Deserialize)]
struct CsvInventory {
    #[serde(default)]
    document: Vec<CsvInventoryRow>,
}

#[derive(Debug, Deserialize)]
struct CsvInventoryRow {
    #[serde(default)]
    path: String,
    #[serde(default)]
    zone: String,
}

#[derive(Debug, Deserialize)]
struct PolicyRegistry {
    #[serde(default)]
    dataset: Vec<PolicyRow>,
}

#[derive(Debug, Deserialize)]
struct PolicyRow {
    #[serde(default)]
    path: String,
    #[serde(default)]
    classification: String,
}

#[derive(Debug, Deserialize)]
struct ScrollIndex {
    #[serde(default)]
    dataset: Vec<ScrollRow>,
}

#[derive(Debug, Deserialize)]
struct ScrollRow {
    #[serde(default)]
    source_csv: String,
    #[serde(default)]
    dataset_class: String,
}

fn load_toml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let blob = fs::read_to_string(path).map_err(|err| format!("{}: {}", path.display(), err))?;
    toml::from_str(&blob).map_err(|err| format!("{}: {}", path.display(), err))
}

fn load_manifest(path: &Path) -> Result<BTreeSet<String>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("{}: {}", path.display(), err))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect())
}

fn sample(items: &BTreeSet<String>, limit: usize) -> Vec<String> {
    items.iter().take(limit).cloned().collect()
}

fn main() {
    let args = Args::parse();
    let root = args
        .repo_root
        .canonicalize()
        .unwrap_or_else(|_| args.repo_root.clone());

    let inventory_path = root.join(&args.inventory);
    let policy_path = root.join(&args.policy);
    let canonical_manifest_path = root.join(&args.canonical_manifest);
    let generated_manifest_path = root.join(&args.generated_manifest);
    let canonical_index_path = root.join(&args.canonical_index);
    let generated_index_path = root.join(&args.generated_index);

    let inventory: CsvInventory = match load_toml(&inventory_path) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("ERROR: {}", err);
            std::process::exit(1);
        }
    };
    let policy: PolicyRegistry = match load_toml(&policy_path) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("ERROR: {}", err);
            std::process::exit(1);
        }
    };
    let canonical_index: ScrollIndex = match load_toml(&canonical_index_path) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("ERROR: {}", err);
            std::process::exit(1);
        }
    };
    let generated_index: ScrollIndex = match load_toml(&generated_index_path) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("ERROR: {}", err);
            std::process::exit(1);
        }
    };

    let canonical_manifest = match load_manifest(&canonical_manifest_path) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("ERROR: {}", err);
            std::process::exit(1);
        }
    };
    let generated_manifest = match load_manifest(&generated_manifest_path) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("ERROR: {}", err);
            std::process::exit(1);
        }
    };

    let project_paths: BTreeSet<String> = inventory
        .document
        .iter()
        .filter(|row| row.zone == "project_csv")
        .map(|row| row.path.clone())
        .collect();
    let policy_paths: BTreeSet<String> =
        policy.dataset.iter().map(|row| row.path.clone()).collect();
    let canonical_policy: BTreeSet<String> = policy
        .dataset
        .iter()
        .filter(|row| row.classification == "canonical_dataset")
        .map(|row| row.path.clone())
        .collect();
    let generated_policy: BTreeSet<String> = policy
        .dataset
        .iter()
        .filter(|row| row.classification == "generated_artifact")
        .map(|row| row.path.clone())
        .collect();
    let canonical_index_paths: BTreeSet<String> = canonical_index
        .dataset
        .iter()
        .map(|row| row.source_csv.clone())
        .collect();
    let generated_index_paths: BTreeSet<String> = generated_index
        .dataset
        .iter()
        .map(|row| row.source_csv.clone())
        .collect();

    let mut failures: Vec<String> = Vec::new();

    if policy_paths != project_paths {
        let missing: BTreeSet<String> = project_paths.difference(&policy_paths).cloned().collect();
        let extra: BTreeSet<String> = policy_paths.difference(&project_paths).cloned().collect();
        if !missing.is_empty() {
            failures.push(format!(
                "Policy missing {} project_csv paths.",
                missing.len()
            ));
            for item in sample(&missing, 20) {
                failures.push(format!("- missing: {}", item));
            }
        }
        if !extra.is_empty() {
            failures.push(format!("Policy has {} non-project paths.", extra.len()));
            for item in sample(&extra, 20) {
                failures.push(format!("- extra: {}", item));
            }
        }
    }

    if !canonical_policy.is_disjoint(&generated_policy) {
        failures.push("Policy has overlapping canonical/generated path assignments.".to_string());
    }

    let all_policy: BTreeSet<String> = canonical_policy.union(&generated_policy).cloned().collect();
    if all_policy != project_paths {
        failures.push(
            "Policy canonical/generated partition does not cover project_csv set.".to_string(),
        );
    }

    if canonical_manifest != canonical_policy {
        failures
            .push("Canonical manifest does not match policy canonical_dataset set.".to_string());
    }
    if generated_manifest != generated_policy {
        failures
            .push("Generated manifest does not match policy generated_artifact set.".to_string());
    }

    if canonical_index_paths != canonical_policy {
        failures.push(
            "Canonical index source_csv set does not match policy canonical_dataset set."
                .to_string(),
        );
    }
    if generated_index_paths != generated_policy {
        failures.push(
            "Generated index source_csv set does not match policy generated_artifact set."
                .to_string(),
        );
    }

    for row in &canonical_index.dataset {
        if row.dataset_class != "canonical_dataset" {
            failures.push(format!(
                "{}: canonical index has wrong dataset_class",
                row.source_csv
            ));
        }
    }
    for row in &generated_index.dataset {
        if row.dataset_class != "generated_artifact" {
            failures.push(format!(
                "{}: generated index has wrong dataset_class",
                row.source_csv
            ));
        }
    }

    if !failures.is_empty() {
        eprintln!("ERROR: project_csv split policy verification failed.");
        for item in failures.iter().take(200) {
            eprintln!("{}", item);
        }
        if failures.len() > 200 {
            eprintln!("- ... and {} more failures", failures.len() - 200);
        }
        std::process::exit(1);
    }

    println!(
        "OK: project_csv split policy and scroll coverage verified. canonical={} generated={}",
        canonical_policy.len(),
        generated_policy.len()
    );
}
