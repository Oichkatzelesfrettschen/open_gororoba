use anyhow::{Context, Result, anyhow};
use clap::Parser;
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

const CI_WORKSPACE_TRIGGERS: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "Makefile",
    "agents.toml",
    "registry/test_taxonomy.toml",
    "registry/engineering_standards.toml",
    ".github/workflows/ci.yml",
    ".cargo/config.toml",
    ".config/nextest.toml",
];

const CI_WORKSPACE_TRIGGER_PREFIXES: &[&str] = &["mk/"];

const LOCAL_SHARED_RUST_TRIGGERS: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "Makefile",
    "agents.toml",
    "registry/test_taxonomy.toml",
    "registry/engineering_standards.toml",
];

const LOCAL_WORKSPACE_TRIGGERS: &[&str] = &["rust-toolchain.toml"];

const RUST_IRRELEVANT_PREFIXES: &[&str] = &[
    "registry/",
    "docs/",
    "papers/",
    "proofs/",
    "src/gemini_physics/",
    "src/ghost_stats/",
    "src/verification/",
    "bin/",
    "tests/",
    "data/",
    ".github/",
    ".githooks/",
    "scripts/",
    "plans/",
    ".horusec/",
];

const RUST_IRRELEVANT_FILES: &[&str] = &[".gitignore", "pyproject.toml", "pytest.ini"];

const GOVERNANCE_PREFIXES: &[&str] = &[
    "registry/",
    "docs/",
    "reports/",
    "data/artifacts/",
    "data/output/audit/",
    "proofs/",
    "bin/",
    "src/verification/",
    "tests/",
];

/// Whether a path is a Cargo manifest, which makes it governance-relevant.
///
/// `registry/binaries.toml` must match the `[[bin]]` set `cargo metadata`
/// reports exactly, and `execution-planning --verify` resolves every experiment
/// and lineage row against that set. A manifest edit can add, remove or rename
/// a target without touching anything under `registry/`, so keying the
/// governance lane on the registry tree alone lets a binary-set change reach
/// the branch unchecked.
fn is_cargo_manifest(file: &str) -> bool {
    file == "Cargo.toml" || file.ends_with("/Cargo.toml")
}

#[derive(Parser, Debug)]
#[command(
    name = "workspace-routing",
    about = "Map changed files to affected Rust crate scope and governance participation"
)]
struct Cli {
    #[arg(long)]
    local: bool,
    #[arg(long)]
    base: Option<String>,
    #[arg(long)]
    verbose: bool,
    /// Emit only the DIRECTLY changed crates as the scope (no
    /// reverse-dependency transitive closure). Use this for clippy
    /// runs, where lints fire on the package owning the source: a
    /// change in a hub crate cannot induce a new lint on a downstream
    /// consumer whose source has not changed.
    ///
    /// Without this flag (the default), the scope is the full
    /// transitive reverse-closure, which is correct for nextest
    /// because consumer tests exercise the changed hub crate.
    #[arg(long)]
    direct_only: bool,
}

#[derive(Debug)]
struct BaseRefError(String);

impl std::fmt::Display for BaseRefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BaseRefError {}

#[derive(Debug, Default)]
struct DependencyGraph {
    all_crates: BTreeSet<String>,
    deps: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Debug, Default)]
struct ChangeClassification {
    affected_crates: BTreeSet<String>,
    force_workspace: bool,
    has_governance_changes: bool,
    has_shared_rust_changes: bool,
    /// Any changed file can contain text covered by the hygiene gates,
    /// including Rust comments and strings that clippy leaves unchecked.
    has_check_relevant_changes: bool,
}

fn repo_root() -> PathBuf {
    repo_root::resolve!()
}

fn crates_dir(root: &Path) -> PathBuf {
    root.join("crates")
}

fn in_ci() -> bool {
    env::var("GITHUB_ACTIONS")
        .map(|value| value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn git(root: &Path, args: &[&str]) -> Result<(bool, String, String)> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .with_context(|| format!("run git {}", args.join(" ")))?;
    Ok((
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).trim().to_string(),
        String::from_utf8_lossy(&output.stderr).trim().to_string(),
    ))
}

fn ref_exists(root: &Path, reference: &str) -> Result<bool> {
    let (ok, _, _) = git(root, &["rev-parse", "--verify", reference])?;
    Ok(ok)
}

fn detect_base_ref(root: &Path, explicit_base: Option<&str>) -> Result<String> {
    if let Some(base) = explicit_base {
        return Ok(base.to_string());
    }

    if let Ok(base_ref) = env::var("GITHUB_BASE_REF")
        && !base_ref.trim().is_empty()
    {
        return Ok(format!("origin/{base_ref}"));
    }

    if let Ok(before) = env::var("GITHUB_EVENT_BEFORE")
        && before != "0".repeat(40)
        && !before.trim().is_empty()
    {
        return Ok(before);
    }

    for candidate in ["origin/main", "origin/master"] {
        if ref_exists(root, candidate)? {
            return Ok(candidate.to_string());
        }
    }

    Ok("HEAD~1".to_string())
}

fn validate_base_ref(root: &Path, base: &str) -> std::result::Result<(), BaseRefError> {
    match ref_exists(root, base) {
        Ok(true) => Ok(()),
        Ok(false) => Err(BaseRefError(format!(
            "cannot resolve diff base `{base}`. Fetch the base history before running CI routing (for GitHub Actions, use actions/checkout with fetch-depth: 0)."
        ))),
        Err(err) => Err(BaseRefError(format!(
            "cannot resolve diff base `{base}`: {err}"
        ))),
    }
}

fn changed_files(
    root: &Path,
    base: Option<&str>,
) -> std::result::Result<Vec<String>, BaseRefError> {
    let mut paths = BTreeSet::new();

    if let Some(base_ref) = base {
        let (mut ok, mut committed, _) = git(
            root,
            &["diff", "--name-only", &format!("{base_ref}...HEAD")],
        )
        .map_err(|err| BaseRefError(format!("failed to diff against base `{base_ref}`: {err}")))?;
        if !ok || committed.trim().is_empty() {
            let retry = git(root, &["diff", "--name-only", base_ref, "HEAD"]).map_err(|err| {
                BaseRefError(format!("failed to diff against base `{base_ref}`: {err}"))
            })?;
            ok = retry.0;
            committed = retry.1;
        }
        if !ok {
            return Err(BaseRefError(format!(
                "failed to diff against base `{base_ref}`. Fetch the base history before running CI routing."
            )));
        }
        committed
            .lines()
            .filter(|line| !line.trim().is_empty())
            .for_each(|line| {
                paths.insert(line.to_string());
            });
    }

    let (_, working_tree, _) = git(root, &["diff", "--name-only", "HEAD"])
        .map_err(|err| BaseRefError(format!("failed to diff working tree: {err}")))?;
    working_tree
        .lines()
        .filter(|line| !line.trim().is_empty())
        .for_each(|line| {
            paths.insert(line.to_string());
        });

    let (_, untracked, _) = git(root, &["ls-files", "--others", "--exclude-standard"])
        .map_err(|err| BaseRefError(format!("failed to list untracked files: {err}")))?;
    untracked
        .lines()
        .filter(|line| !line.trim().is_empty())
        .for_each(|line| {
            paths.insert(line.to_string());
        });

    Ok(paths.into_iter().collect())
}

fn parse_workspace_path_deps(root: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let cargo_toml = fs::read_to_string(root.join("Cargo.toml")).context("read root Cargo.toml")?;
    let value: toml::Value = toml::from_str(&cargo_toml).context("parse root Cargo.toml")?;
    let Some(workspace) = value.get("workspace").and_then(toml::Value::as_table) else {
        return Ok(BTreeMap::new());
    };
    let Some(dependencies) = workspace
        .get("dependencies")
        .and_then(toml::Value::as_table)
    else {
        return Ok(BTreeMap::new());
    };
    let mut path_map = BTreeMap::new();
    for (dep_name, dep_value) in dependencies {
        let Some(dep_table) = dep_value.as_table() else {
            continue;
        };
        let Some(path) = dep_table.get("path").and_then(toml::Value::as_str) else {
            continue;
        };
        path_map.insert(
            dep_name.clone(),
            root.join(path)
                .canonicalize()
                .with_context(|| format!("resolve workspace dependency {dep_name}: {path}"))?,
        );
    }
    Ok(path_map)
}

fn manifest_dependency_paths(
    manifest_directory: &Path,
    manifest: &toml::Value,
    workspace_paths: &BTreeMap<String, PathBuf>,
) -> Result<BTreeSet<PathBuf>> {
    let target_tables = manifest
        .get("target")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flat_map(|targets| targets.values());
    let mut paths = BTreeSet::new();
    for configuration in std::iter::once(manifest).chain(target_tables) {
        for dependency_kind in ["dependencies", "dev-dependencies", "build-dependencies"] {
            let Some(dependencies) = configuration
                .get(dependency_kind)
                .and_then(toml::Value::as_table)
            else {
                continue;
            };
            for (dependency_name, dependency) in dependencies {
                if let Some(path) = dependency.get("path").and_then(toml::Value::as_str) {
                    paths.insert(manifest_directory.join(path).canonicalize().with_context(
                        || format!("resolve dependency {dependency_name}: {path}"),
                    )?);
                } else if dependency.get("workspace").and_then(toml::Value::as_bool) == Some(true)
                    && let Some(path) = workspace_paths.get(dependency_name)
                {
                    paths.insert(path.clone());
                }
            }
        }
    }
    Ok(paths)
}

fn build_dependency_graph(root: &Path) -> Result<DependencyGraph> {
    let crates_root = crates_dir(root);
    if !crates_root.is_dir() {
        return Ok(DependencyGraph::default());
    }

    let mut graph = DependencyGraph::default();
    for entry in fs::read_dir(&crates_root).context("read crates directory")? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let manifest = entry.path().join("Cargo.toml");
        if manifest.is_file() {
            graph
                .all_crates
                .insert(entry.file_name().to_string_lossy().to_string());
        }
    }
    // `xtask/` is a top-level workspace member that lives outside the
    // `crates/` tree. Add it explicitly so the classifier's
    // `affected_crates.insert("xtask")` branch (for files matching
    // `xtask/...`) resolves to a known target instead of falling
    // through to the force_workspace path.
    if root.join("xtask").join("Cargo.toml").is_file() {
        graph.all_crates.insert("xtask".to_string());
    }

    let ws_path_map = parse_workspace_path_deps(root)?;
    let mut manifests = BTreeMap::new();
    let mut owners = BTreeMap::new();
    for crate_name in &graph.all_crates {
        // xtask lives at root/xtask, not crates_root/xtask; every
        // other entry in all_crates is under crates_root.
        let cargo_toml = if crate_name == "xtask" {
            root.join("xtask").join("Cargo.toml")
        } else {
            crates_root.join(crate_name).join("Cargo.toml")
        };
        let text = fs::read_to_string(&cargo_toml)
            .with_context(|| format!("read crate manifest {}", cargo_toml.display()))?;
        let manifest: toml::Value = toml::from_str(&text)
            .with_context(|| format!("parse crate manifest {}", cargo_toml.display()))?;
        let directory = cargo_toml
            .parent()
            .context("manifest directory")?
            .canonicalize()?;
        owners.insert(directory.clone(), crate_name.clone());
        manifests.insert(crate_name.clone(), (directory, manifest));
    }
    for (crate_name, (directory, manifest)) in manifests {
        for path in manifest_dependency_paths(&directory, &manifest, &ws_path_map)? {
            if let Some(dependency_name) = owners.get(&path)
                && dependency_name != &crate_name
            {
                graph
                    .deps
                    .entry(crate_name.clone())
                    .or_default()
                    .insert(dependency_name.clone());
            }
        }
    }

    Ok(graph)
}

fn invert_graph(deps: &BTreeMap<String, BTreeSet<String>>) -> BTreeMap<String, BTreeSet<String>> {
    let mut reverse = BTreeMap::new();
    for (crate_name, upstreams) in deps {
        for upstream in upstreams {
            reverse
                .entry(upstream.clone())
                .or_insert_with(BTreeSet::new)
                .insert(crate_name.clone());
        }
    }
    reverse
}

fn transitive_closure(
    seeds: &BTreeSet<String>,
    reverse: &BTreeMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    let mut visited = BTreeSet::new();
    let mut stack: Vec<String> = seeds.iter().cloned().collect();
    while let Some(crate_name) = stack.pop() {
        if !visited.insert(crate_name.clone()) {
            continue;
        }
        if let Some(dependents) = reverse.get(&crate_name) {
            for dependent in dependents {
                if !visited.contains(dependent) {
                    stack.push(dependent.clone());
                }
            }
        }
    }
    visited
}

fn classify_changes(
    files: &[String],
    all_crates: &BTreeSet<String>,
    local_mode: bool,
) -> ChangeClassification {
    let workspace_triggers: BTreeSet<&str> = if local_mode {
        LOCAL_WORKSPACE_TRIGGERS.iter().copied().collect()
    } else {
        CI_WORKSPACE_TRIGGERS.iter().copied().collect()
    };
    let shared_rust_triggers: BTreeSet<&str> = if local_mode {
        LOCAL_SHARED_RUST_TRIGGERS.iter().copied().collect()
    } else {
        BTreeSet::new()
    };

    let mut classification = ChangeClassification::default();

    for file in files {
        // Source comments and strings need the same hygiene checks as prose.
        classification.has_check_relevant_changes = true;

        if GOVERNANCE_PREFIXES
            .iter()
            .any(|prefix| file.starts_with(prefix))
            || (!file.contains('/') && file.ends_with(".md"))
            || is_cargo_manifest(file)
        {
            classification.has_governance_changes = true;
        }

        if workspace_triggers.contains(file.as_str())
            || (!local_mode
                && CI_WORKSPACE_TRIGGER_PREFIXES
                    .iter()
                    .any(|prefix| file.starts_with(prefix)))
        {
            classification.force_workspace = true;
            continue;
        }
        if shared_rust_triggers.contains(file.as_str()) {
            classification.has_shared_rust_changes = true;
            continue;
        }

        if RUST_IRRELEVANT_FILES.contains(&file.as_str()) {
            continue;
        }
        if RUST_IRRELEVANT_PREFIXES
            .iter()
            .any(|prefix| file.starts_with(prefix))
        {
            continue;
        }

        if let Some(remainder) = file.strip_prefix("crates/") {
            if let Some(crate_dir) = remainder.split('/').next() {
                if all_crates.contains(crate_dir) {
                    classification.affected_crates.insert(crate_dir.to_string());
                } else {
                    classification.force_workspace = true;
                }
            } else {
                classification.force_workspace = true;
            }
            continue;
        }

        // `xtask/` is a top-level workspace member that lives outside
        // `crates/`. Treat it as its own package so xtask-only edits
        // do not fall through to the `--workspace` fallback.
        //
        // # Why this fix exists
        //
        // Before this branch, a change to `xtask/src/main.rs` was not
        // matched by any of the prefix rules (it does not start with
        // `crates/`, is not a top-level `.rs`/`.toml`, and is not in
        // the RUST_IRRELEVANT_PREFIXES list). The classifier returned
        // an empty change set, which then promoted to `--workspace`
        // downstream via the local_base_fallback path. Pre-push
        // therefore recompiled and re-tested the ENTIRE workspace
        // (~3-8 minutes) for a one-line xtask edit.
        if file.starts_with("xtask/") {
            classification.affected_crates.insert("xtask".to_string());
            continue;
        }

        if !file.contains('/') && (file.ends_with(".rs") || file.ends_with(".toml")) {
            classification.force_workspace = true;
        }
    }

    classification
}

fn format_cargo_scope(crates: &BTreeSet<String>) -> String {
    crates
        .iter()
        .map(|crate_name| format!("-p {crate_name}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn emit_local(scope: &str, run_rust: bool, run_governance: bool, run_check: bool, verbose: bool) {
    if verbose {
        eprintln!("[ci-routing] run_rust={}", python_bool(run_rust));
        eprintln!(
            "[ci-routing] run_governance={}",
            python_bool(run_governance)
        );
        eprintln!("[ci-routing] run_check={}", python_bool(run_check));
        eprintln!(
            "[ci-routing] scope={}",
            if scope.is_empty() { "(skip)" } else { scope }
        );
    }
    println!("{scope}");
}

fn emit_github(
    scope: &str,
    run_rust: bool,
    run_governance: bool,
    run_check: bool,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("[ci-routing] run_rust={}", python_bool(run_rust));
        eprintln!(
            "[ci-routing] run_governance={}",
            python_bool(run_governance)
        );
        eprintln!("[ci-routing] run_check={}", python_bool(run_check));
        eprintln!(
            "[ci-routing] rust_scope={}",
            if scope.is_empty() { "(skip)" } else { scope }
        );
    }

    if let Ok(output_path) = env::var("GITHUB_OUTPUT")
        && !output_path.trim().is_empty()
    {
        let mut body = String::new();
        body.push_str(&format!("rust_scope={scope}\n"));
        body.push_str(&format!(
            "run_rust={}\n",
            if run_rust { "true" } else { "false" }
        ));
        body.push_str(&format!(
            "run_governance={}\n",
            if run_governance { "true" } else { "false" }
        ));
        body.push_str(&format!(
            "run_check={}\n",
            if run_check { "true" } else { "false" }
        ));
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)
            .with_context(|| format!("open GITHUB_OUTPUT {}", output_path))?
            .write_all(body.as_bytes())
            .with_context(|| format!("write GITHUB_OUTPUT {}", output_path))?;
        return Ok(());
    }

    println!("::set-output name=rust_scope::{scope}");
    println!(
        "::set-output name=run_rust::{}",
        if run_rust { "true" } else { "false" }
    );
    println!(
        "::set-output name=run_governance::{}",
        if run_governance { "true" } else { "false" }
    );
    println!(
        "::set-output name=run_check::{}",
        if run_check { "true" } else { "false" }
    );
    Ok(())
}

use std::io::Write;

fn run(cli: &Cli) -> Result<i32> {
    let root = repo_root();
    let mut base = detect_base_ref(&root, cli.base.as_deref())?;
    let mut local_base_fallback = false;
    if cli.verbose {
        eprintln!("[ci-routing] base_ref={base}");
    }

    if let Err(err) = validate_base_ref(&root, &base) {
        if !cli.local || in_ci() {
            return Err(anyhow!(err));
        }
        local_base_fallback = true;
        eprintln!(
            "[ci-routing] WARNING: {err} Falling back to working-tree and untracked files only."
        );
        base.clear();
    }

    let files = changed_files(
        &root,
        if base.is_empty() {
            None
        } else {
            Some(base.as_str())
        },
    )
    .map_err(anyhow::Error::msg)?;
    if cli.verbose {
        eprintln!("[ci-routing] {} changed files", files.len());
    }

    let graph = build_dependency_graph(&root)?;
    let reverse = invert_graph(&graph.deps);
    let classification = classify_changes(&files, &graph.all_crates, cli.local);

    let (scope, run_rust) = if classification.force_workspace {
        ("--workspace".to_string(), true)
    } else if !classification.affected_crates.is_empty() {
        // --direct-only (clippy lane) skips the transitive closure
        // expansion: lints fire on the package owning the source, so
        // pulling in dependents only inflates compile time without
        // catching new diagnostics. The default branch keeps the full
        // reverse-closure so consumer tests run against the hub change.
        let target = if cli.direct_only {
            classification.affected_crates.clone()
        } else {
            transitive_closure(&classification.affected_crates, &reverse)
        };
        (format_cargo_scope(&target), true)
    } else if classification.has_shared_rust_changes {
        ("--workspace".to_string(), true)
    } else {
        (String::new(), false)
    };

    let mut final_scope = scope;
    let mut has_governance = classification.has_governance_changes;
    let mut final_run_rust = run_rust;
    let mut run_check = classification.has_check_relevant_changes;

    if local_base_fallback && files.is_empty() {
        final_scope = "--workspace".to_string();
        final_run_rust = true;
        has_governance = true;
        run_check = true;
        if cli.verbose {
            eprintln!(
                "[ci-routing] local fallback could not inspect committed branch deltas; promoting to --workspace."
            );
        }
    }

    if cli.local {
        emit_local(
            &final_scope,
            final_run_rust,
            has_governance,
            run_check,
            cli.verbose,
        );
    } else {
        emit_github(
            &final_scope,
            final_run_rust,
            has_governance,
            run_check,
            cli.verbose,
        )?;
    }

    Ok(0)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(code) => ExitCode::from(code as u8),
        Err(err) => {
            let message = err.to_string();
            if message.contains("cannot resolve diff base") {
                eprintln!("[ci-routing] ERROR: {message}");
                ExitCode::from(2)
            } else {
                eprintln!("[ci-routing] ERROR: {message} -- falling back to --workspace");
                if cli.local {
                    println!("--workspace");
                } else if let Err(output_err) =
                    emit_github("--workspace", true, true, true, cli.verbose)
                {
                    eprintln!("[ci-routing] ERROR: failed to emit fallback outputs: {output_err}");
                    return ExitCode::from(1);
                }
                ExitCode::SUCCESS
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::is_cargo_manifest;

    #[test]
    fn root_member_dependencies_enter_reverse_closure() -> anyhow::Result<()> {
        let graph = super::build_dependency_graph(&super::repo_root())?;
        let reverse = super::invert_graph(&graph.deps);
        for dependency in ["provenance_store", "verified_core", "repo_utilities"] {
            assert!(graph.deps["xtask"].contains(dependency), "{dependency}");
            let closure = super::transitive_closure(&[dependency.to_string()].into(), &reverse);
            assert!(closure.contains("xtask"), "{dependency}");
        }
        Ok(())
    }

    #[test]
    fn toml_dependency_tables_resolve_relative_paths_and_aliases() -> anyhow::Result<()> {
        let root = super::repo_root();
        let manifest: toml::Value = toml::from_str(
            r#"
            [dependencies.renamed]
            package = "provenance_store"
            path = "../crates/provenance_store"
            [build-dependencies]
            repo_root = { workspace = true }
            [target.'cfg(unix)'.dev-dependencies]
            verified_core = { path = "../crates/verified_core" }
            [[bin]]
            name = "foreign-source"
            path = "../crates/gororoba_cli_data/src/bin/workspace_routing.rs"
            "#,
        )?;
        let paths = super::manifest_dependency_paths(
            &root.join("xtask"),
            &manifest,
            &super::parse_workspace_path_deps(&root)?,
        )?;
        let expected = ["provenance_store", "repo_root", "verified_core"]
            .into_iter()
            .map(|name| root.join("crates").join(name).canonicalize())
            .collect::<std::io::Result<std::collections::BTreeSet<_>>>()?;
        assert_eq!(paths, expected);
        Ok(())
    }

    #[test]
    fn rust_sources_trigger_hygiene_checks() {
        for local_mode in [false, true] {
            let classification = super::classify_changes(
                &["crates/cosmology_core/src/distance.rs".to_string()],
                &["cosmology_core".to_string()].into(),
                local_mode,
            );
            assert!(classification.has_check_relevant_changes);
        }
    }

    #[test]
    fn workspace_triggers_retain_governance_classification() {
        for local_mode in [false, true] {
            for path in ["Cargo.toml", "registry/test_taxonomy.toml"] {
                let classification =
                    super::classify_changes(&[path.to_string()], &Default::default(), local_mode);
                assert!(classification.has_governance_changes, "{path}");
                assert_eq!(classification.force_workspace, !local_mode, "{path}");
                assert_eq!(classification.has_shared_rust_changes, local_mode, "{path}");
            }
        }
    }

    #[test]
    fn cloud_build_configuration_routes_to_the_workspace() {
        for path in [
            ".github/workflows/ci.yml",
            ".cargo/config.toml",
            ".config/nextest.toml",
            "mk/cache_roots.mk",
        ] {
            let cloud = super::classify_changes(&[path.to_string()], &Default::default(), false);
            assert!(cloud.force_workspace, "{path}");
            let local = super::classify_changes(&[path.to_string()], &Default::default(), true);
            assert!(!local.force_workspace, "{path}");
        }
        let paper = super::classify_changes(
            &[".github/workflows/paper.yml".to_string()],
            &Default::default(),
            false,
        );
        assert!(!paper.force_workspace);
    }

    #[test]
    fn audit_artifact_edits_trigger_governance_without_rust() {
        for local_mode in [false, true] {
            let classification = super::classify_changes(
                &["data/output/audit/frb-physical-calibration-replay/findings.json".to_string()],
                &Default::default(),
                local_mode,
            );
            assert!(classification.has_governance_changes);
            assert!(classification.has_check_relevant_changes);
            assert!(!classification.force_workspace);
            assert!(classification.affected_crates.is_empty());
        }
    }

    #[test]
    fn provenance_sources_route_to_compiling_owners() -> anyhow::Result<()> {
        let root = super::repo_root();
        let graph = super::build_dependency_graph(&root)?;
        let reverse = super::invert_graph(&graph.deps);
        for path in [
            "crates/provenance_ops/src/source_provenance.rs",
            "crates/provenance_ops/src/source_provenance/host_observations.rs",
            "crates/provenance_ops/src/source_provenance/metadata_completeness.rs",
        ] {
            assert!(root.join(path).is_file());
            let classification =
                super::classify_changes(&[path.to_string()], &graph.all_crates, true);
            assert_eq!(
                classification.affected_crates,
                ["provenance_ops".to_string()].into()
            );
            assert!(!classification.force_workspace);
            let closure = super::transitive_closure(&classification.affected_crates, &reverse);
            for owner in [
                "provenance_ops",
                "gororoba_cli_provenance",
                "gororoba_cli_data",
            ] {
                assert!(closure.contains(owner), "{path}: missing {owner}");
            }
            eprintln!("{path}: dependency closure = {closure:?}");
        }
        let manifest: toml::Value = toml::from_str(&std::fs::read_to_string(
            root.join("crates/gororoba_cli_provenance/Cargo.toml"),
        )?)?;
        for target in manifest["bin"].as_array().expect("binary targets") {
            let relative = target["path"].as_str().expect("binary path");
            assert!(relative.starts_with("src/bin/"));
            let path = format!("crates/gororoba_cli_provenance/{relative}");
            assert!(root.join(&path).is_file());
            let classification = super::classify_changes(&[path], &graph.all_crates, true);
            assert_eq!(
                classification.affected_crates,
                ["gororoba_cli_provenance".to_string()].into()
            );
            assert!(!classification.force_workspace);
        }
        Ok(())
    }

    #[test]
    fn manifests_at_any_depth_are_governance_relevant() {
        assert!(is_cargo_manifest("Cargo.toml"));
        assert!(is_cargo_manifest("crates/gororoba_cli_physics/Cargo.toml"));
        assert!(is_cargo_manifest("xtask/Cargo.toml"));
    }

    #[test]
    fn near_misses_stay_out_of_the_governance_lane() {
        // A lockfile records resolution, not the `[[bin]]` set, so it cannot
        // desynchronize `registry/binaries.toml`.
        assert!(!is_cargo_manifest("Cargo.lock"));
        assert!(!is_cargo_manifest("crates/cd_kernel/Cargo.lock"));
        // Suffix matching must not fire on a name that merely ends the same way.
        assert!(!is_cargo_manifest("docs/NotCargo.toml"));
        assert!(!is_cargo_manifest("registry/claims.toml"));
    }
}
