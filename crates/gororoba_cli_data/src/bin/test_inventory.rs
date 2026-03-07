//! Validate smoke/regression/heavy test taxonomy and stale-count policy.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use clap::Parser;
use regex::Regex;
use serde::Deserialize;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "test-inventory")]
#[command(about = "Validate canonical test taxonomy and stale test-count docs policy")]
struct Args {
    /// Path to the test taxonomy registry.
    #[arg(long, default_value = "registry/test_taxonomy.toml")]
    taxonomy: PathBuf,

    /// Check-only mode: exit non-zero if violations are found.
    #[arg(long, default_value_t = false)]
    check: bool,
}

#[derive(Debug, Deserialize)]
struct TaxonomyRegistry {
    test_taxonomy: TaxonomyMeta,
    #[serde(default)]
    rust_smoke_target: Vec<RustSmokeTarget>,
    #[serde(default)]
    rust_heavy_package: Vec<RustHeavyPackage>,
    #[serde(default)]
    python_test_file: Vec<PythonTestFile>,
    #[serde(default)]
    doc_no_count: Vec<DocNoCount>,
}

#[derive(Debug, Deserialize)]
struct TaxonomyMeta {
    updated: String,
    stale_test_count_pattern: String,
}

#[derive(Debug, Deserialize)]
struct RustSmokeTarget {
    package: String,
    target: String,
    kind: String,
    description: String,
}

#[derive(Debug, Deserialize)]
struct RustHeavyPackage {
    name: String,
    cargo_profile: String,
    nextest_profile: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
struct PythonTestFile {
    path: String,
    lane: String,
}

#[derive(Debug, Deserialize)]
struct DocNoCount {
    path: String,
    reason: String,
}

fn read_registry(path: &Path) -> Result<TaxonomyRegistry, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    toml::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))
}

fn repo_root_from_taxonomy(path: &Path) -> Result<PathBuf, String> {
    path.parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("cannot infer repo root from {}", path.display()))
}

fn discover_python_tests(repo_root: &Path) -> Result<BTreeSet<String>, String> {
    let tests_root = repo_root.join("tests");
    if !tests_root.is_dir() {
        return Err(format!(
            "missing tests directory at {}",
            tests_root.display()
        ));
    }

    let mut files = BTreeSet::new();
    for entry in WalkDir::new(&tests_root).min_depth(1).max_depth(1) {
        let entry = entry.map_err(|err| format!("walk tests/: {err}"))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("py") {
            continue;
        }
        let rel = path
            .strip_prefix(repo_root)
            .map_err(|err| format!("strip prefix {}: {err}", path.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        files.insert(rel);
    }
    Ok(files)
}

fn validate_python_files(
    repo_root: &Path,
    registry: &TaxonomyRegistry,
    failures: &mut Vec<String>,
) -> usize {
    let discovered = match discover_python_tests(repo_root) {
        Ok(files) => files,
        Err(err) => {
            failures.push(format!("ERROR: {err}"));
            return 0;
        }
    };

    let configured: BTreeSet<String> = registry
        .python_test_file
        .iter()
        .map(|row| row.path.clone())
        .collect();

    for missing in discovered.difference(&configured) {
        failures.push(format!(
            "ERROR: unclassified pytest file `{missing}` is missing from registry/test_taxonomy.toml"
        ));
    }
    for extra in configured.difference(&discovered) {
        failures.push(format!(
            "ERROR: configured pytest file `{extra}` does not exist on disk"
        ));
    }

    let mut smoke_count = 0usize;
    for row in &registry.python_test_file {
        let path = repo_root.join(&row.path);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) => {
                failures.push(format!("ERROR: read {}: {err}", path.display()));
                continue;
            }
        };

        let has_smoke = text.contains("pytest.mark.smoke");
        let has_regression = text.contains("pytest.mark.regression");

        match row.lane.as_str() {
            "smoke" => {
                smoke_count += 1;
                if !has_smoke {
                    failures.push(format!(
                        "ERROR: {} is configured as smoke but lacks pytest.mark.smoke",
                        row.path
                    ));
                }
            }
            "regression" => {
                if !has_regression {
                    failures.push(format!(
                        "ERROR: {} is configured as regression but lacks pytest.mark.regression",
                        row.path
                    ));
                }
            }
            other => failures.push(format!(
                "ERROR: {} declares unsupported python lane `{other}`",
                row.path
            )),
        }

        if !has_smoke && !has_regression {
            failures.push(format!(
                "ERROR: {} has no smoke/regression pytest marker",
                row.path
            ));
        }
    }

    smoke_count
}

fn validate_rust_smoke_targets(
    repo_root: &Path,
    registry: &TaxonomyRegistry,
    failures: &mut Vec<String>,
) -> usize {
    for row in &registry.rust_smoke_target {
        if row.kind != "integration" {
            failures.push(format!(
                "ERROR: rust smoke target {}:{} uses unsupported kind `{}`",
                row.package, row.target, row.kind
            ));
            continue;
        }
        let path = repo_root
            .join("crates")
            .join(&row.package)
            .join("tests")
            .join(format!("{}.rs", row.target));
        if !path.is_file() {
            failures.push(format!(
                "ERROR: rust smoke target {}:{} missing file {}",
                row.package,
                row.target,
                path.display()
            ));
        }
        if row.description.trim().is_empty() {
            failures.push(format!(
                "ERROR: rust smoke target {}:{} is missing a description",
                row.package, row.target
            ));
        }
    }
    registry.rust_smoke_target.len()
}

fn validate_heavy_packages(
    repo_root: &Path,
    registry: &TaxonomyRegistry,
    failures: &mut Vec<String>,
) -> usize {
    for row in &registry.rust_heavy_package {
        let path = repo_root.join("crates").join(&row.name);
        if !path.is_dir() {
            failures.push(format!(
                "ERROR: heavy package `{}` is missing at {}",
                row.name,
                path.display()
            ));
        }
        if row.cargo_profile.trim().is_empty() || row.nextest_profile.trim().is_empty() {
            failures.push(format!(
                "ERROR: heavy package `{}` must declare cargo_profile and nextest_profile",
                row.name
            ));
        }
        if row.reason.trim().is_empty() {
            failures.push(format!(
                "ERROR: heavy package `{}` is missing a reason",
                row.name
            ));
        }
    }
    registry.rust_heavy_package.len()
}

fn validate_docs_without_counts(
    repo_root: &Path,
    registry: &TaxonomyRegistry,
    failures: &mut Vec<String>,
) -> usize {
    let pattern = match Regex::new(&registry.test_taxonomy.stale_test_count_pattern) {
        Ok(pattern) => pattern,
        Err(err) => {
            failures.push(format!(
                "ERROR: invalid stale_test_count_pattern `{}`: {err}",
                registry.test_taxonomy.stale_test_count_pattern
            ));
            return 0;
        }
    };

    for row in &registry.doc_no_count {
        let path = repo_root.join(&row.path);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) => {
                failures.push(format!("ERROR: read {}: {err}", path.display()));
                continue;
            }
        };
        if pattern.is_match(&text) {
            failures.push(format!(
                "ERROR: {} still contains a hard-coded test count matching `{}` ({})",
                row.path, registry.test_taxonomy.stale_test_count_pattern, row.reason
            ));
        }
    }
    registry.doc_no_count.len()
}

fn main() {
    let args = Args::parse();
    let taxonomy_path = args.taxonomy;
    let registry = match read_registry(&taxonomy_path) {
        Ok(registry) => registry,
        Err(err) => {
            eprintln!("ERROR: {err}");
            std::process::exit(1);
        }
    };

    let repo_root = match repo_root_from_taxonomy(&taxonomy_path) {
        Ok(root) => root,
        Err(err) => {
            eprintln!("ERROR: {err}");
            std::process::exit(1);
        }
    };

    let mut failures = Vec::new();
    let rust_smoke_count = validate_rust_smoke_targets(&repo_root, &registry, &mut failures);
    let rust_heavy_count = validate_heavy_packages(&repo_root, &registry, &mut failures);
    let python_smoke_count = validate_python_files(&repo_root, &registry, &mut failures);
    let docs_checked = validate_docs_without_counts(&repo_root, &registry, &mut failures);

    println!(
        "test-inventory: updated={} rust_smoke_targets={} rust_heavy_packages={} python_files={} python_smoke_files={} docs_without_totals={}",
        registry.test_taxonomy.updated,
        rust_smoke_count,
        rust_heavy_count,
        registry.python_test_file.len(),
        python_smoke_count,
        docs_checked,
    );

    if failures.is_empty() {
        println!("OK: canonical test taxonomy is classified and stale-count docs are clean.");
        return;
    }

    for failure in &failures {
        eprintln!("{failure}");
    }

    if args.check {
        std::process::exit(1);
    }
}
