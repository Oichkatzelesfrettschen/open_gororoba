//! Validate smoke/regression/heavy test taxonomy and stale-count policy.

use std::{
    collections::{BTreeMap, BTreeSet},
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
    rust_runtime_policy: RustRuntimePolicy,
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
struct RustRuntimePolicy {
    require_ignore_runtime_seconds: u64,
    runtime_comment_pattern: String,
    #[serde(default)]
    name_hints_require_ignore: Vec<String>,
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

#[derive(Debug)]
struct RustTestCase {
    rel_path: String,
    name: String,
    has_ignore: bool,
    runtime_seconds: Option<u64>,
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
    // If no Python test files are configured this is a pure-Rust repo -- skip all Python checks.
    if registry.python_test_file.is_empty() {
        return 0;
    }
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

fn extract_runtime_seconds(runtime_regex: &Regex, text: &str) -> Result<Option<u64>, String> {
    let Some(captures) = runtime_regex.captures(text) else {
        return Ok(None);
    };
    let lower = captures
        .get(1)
        .ok_or_else(|| "missing runtime lower bound capture".to_string())?
        .as_str()
        .parse::<u64>()
        .map_err(|err| format!("parse runtime lower bound: {err}"))?;
    let upper = captures
        .get(2)
        .map(|m| m.as_str().parse::<u64>())
        .transpose()
        .map_err(|err| format!("parse runtime upper bound: {err}"))?
        .unwrap_or(lower);
    let magnitude = lower.max(upper);
    let unit = captures
        .get(3)
        .ok_or_else(|| "missing runtime unit capture".to_string())?
        .as_str()
        .to_ascii_lowercase();
    let seconds = match unit.as_str() {
        "s" | "sec" | "secs" | "second" | "seconds" => magnitude,
        "min" | "mins" | "minute" | "minutes" => magnitude * 60,
        "hr" | "hrs" | "hour" | "hours" => magnitude * 60 * 60,
        other => return Err(format!("unsupported runtime unit `{other}`")),
    };
    Ok(Some(seconds))
}

fn discover_rust_tests_in_file(
    repo_root: &Path,
    path: &Path,
    runtime_regex: &Regex,
    failures: &mut Vec<String>,
) -> Vec<RustTestCase> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            failures.push(format!("ERROR: read {}: {err}", path.display()));
            return Vec::new();
        }
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut tests = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let Some(name_part) = trimmed.strip_prefix("fn test_") else {
            continue;
        };
        let Some(paren_index) = name_part.find('(') else {
            continue;
        };
        let name = format!("test_{}", &name_part[..paren_index]);

        let mut has_test = false;
        let mut has_ignore = false;
        let mut context = String::new();
        let mut cursor = idx;
        while cursor > 0 {
            cursor -= 1;
            let prev = lines[cursor].trim();
            if prev.is_empty() {
                continue;
            }
            if prev.starts_with("#[test]") {
                has_test = true;
            }
            if prev.starts_with("#[ignore") {
                has_ignore = true;
            }
            if prev.starts_with("#[")
                || prev.starts_with("//")
                || prev.starts_with("///")
                || prev.starts_with("//!")
            {
                context.push_str(prev);
                context.push('\n');
                continue;
            }
            break;
        }
        if !has_test {
            continue;
        }

        let runtime_seconds = match extract_runtime_seconds(runtime_regex, &context) {
            Ok(runtime) => runtime,
            Err(err) => {
                failures.push(format!(
                    "ERROR: parse runtime comment for {} in {}: {err}",
                    name,
                    path.display()
                ));
                None
            }
        };

        let rel_path = path
            .strip_prefix(repo_root)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"));

        tests.push(RustTestCase {
            rel_path,
            name,
            has_ignore,
            runtime_seconds,
        });
    }

    tests
}

fn validate_rust_runtime_policy(
    repo_root: &Path,
    registry: &TaxonomyRegistry,
    failures: &mut Vec<String>,
) -> usize {
    let heavy_packages: BTreeSet<&str> = registry
        .rust_heavy_package
        .iter()
        .map(|row| row.name.as_str())
        .collect();
    let runtime_regex = match Regex::new(&registry.rust_runtime_policy.runtime_comment_pattern) {
        Ok(regex) => regex,
        Err(err) => {
            failures.push(format!(
                "ERROR: invalid rust runtime regex `{}`: {err}",
                registry.rust_runtime_policy.runtime_comment_pattern
            ));
            return 0;
        }
    };

    let mut package_tests: BTreeMap<String, Vec<RustTestCase>> = BTreeMap::new();
    for package in &heavy_packages {
        let crate_root = repo_root.join("crates").join(package);
        if !crate_root.is_dir() {
            continue;
        }
        let mut discovered = Vec::new();
        for entry in WalkDir::new(&crate_root) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(err) => {
                    failures.push(format!("ERROR: walk {}: {err}", crate_root.display()));
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            discovered.extend(discover_rust_tests_in_file(
                repo_root,
                path,
                &runtime_regex,
                failures,
            ));
        }
        package_tests.insert((*package).to_string(), discovered);
    }

    let mut audited = 0usize;
    for (package, tests) in &package_tests {
        for test in tests {
            audited += 1;
            let name_requires_ignore = registry
                .rust_runtime_policy
                .name_hints_require_ignore
                .iter()
                .any(|hint| test.name.contains(hint));
            let runtime_requires_ignore = test.runtime_seconds.is_some_and(|seconds| {
                seconds >= registry.rust_runtime_policy.require_ignore_runtime_seconds
            });

            if (name_requires_ignore || runtime_requires_ignore) && !test.has_ignore {
                let reason = if let Some(seconds) = test.runtime_seconds {
                    format!(
                        "runtime={}s threshold={}s",
                        seconds, registry.rust_runtime_policy.require_ignore_runtime_seconds
                    )
                } else {
                    "name hint matched heavy-runtime policy".to_string()
                };
                failures.push(format!(
                    "ERROR: heavy-package Rust test `{}` in {} ({}) should be #[ignore] for the heavy lane: {}",
                    test.name, test.rel_path, package, reason
                ));
            }
        }
    }

    audited
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
    let rust_runtime_audited = validate_rust_runtime_policy(&repo_root, &registry, &mut failures);
    let python_smoke_count = validate_python_files(&repo_root, &registry, &mut failures);
    let docs_checked = validate_docs_without_counts(&repo_root, &registry, &mut failures);

    println!(
        "test-inventory: updated={} rust_smoke_targets={} rust_heavy_packages={} rust_runtime_audited={} python_files={} python_smoke_files={} docs_without_totals={}",
        registry.test_taxonomy.updated,
        rust_smoke_count,
        rust_heavy_count,
        rust_runtime_audited,
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
