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
    ".cargo/config.toml",
    "rust-toolchain.toml",
    "Makefile",
    "agents.toml",
    ".config/nextest.toml",
    "registry/test_taxonomy.toml",
    "registry/engineering_standards.toml",
];

const LOCAL_SHARED_RUST_TRIGGERS: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    ".cargo/config.toml",
    "Makefile",
    "agents.toml",
    ".config/nextest.toml",
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
    "proofs/",
    "bin/",
    "src/verification/",
    "tests/",
];

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
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate must be nested under repo/crates")
        .to_path_buf()
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

    if let Ok(base_ref) = env::var("GITHUB_BASE_REF") {
        if !base_ref.trim().is_empty() {
            return Ok(format!("origin/{base_ref}"));
        }
    }

    if let Ok(before) = env::var("GITHUB_EVENT_BEFORE") {
        if before != "0".repeat(40) && !before.trim().is_empty() {
            return Ok(before);
        }
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

fn parse_workspace_path_deps(root: &Path) -> Result<BTreeMap<String, String>> {
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
        let Some(crate_dir) = path.strip_prefix("crates/") else {
            continue;
        };
        let crate_dir = crate_dir.trim_end_matches('/');
        if !crate_dir.is_empty() {
            path_map.insert(dep_name.clone(), crate_dir.to_string());
        }
    }
    Ok(path_map)
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

    let ws_path_map = parse_workspace_path_deps(root)?;
    let inline_path_re =
        regex::Regex::new(r#"(?m)^\s*([\w-]+)\s*=\s*\{[^}]*path\s*=\s*"\.\./([\w-]+)""#)
            .expect("valid inline path regex");
    let workspace_dep_re =
        regex::Regex::new(r#"(?m)^\s*([\w-]+)\s*=\s*\{[^}]*workspace\s*=\s*true"#)
            .expect("valid workspace dep regex");

    for crate_name in &graph.all_crates {
        let cargo_toml = crates_root.join(crate_name).join("Cargo.toml");
        let text = fs::read_to_string(&cargo_toml)
            .with_context(|| format!("read crate manifest {}", cargo_toml.display()))?;

        for captures in inline_path_re.captures_iter(&text) {
            let dep_dir = captures.get(2).map(|m| m.as_str()).unwrap_or_default();
            if graph.all_crates.contains(dep_dir) {
                graph
                    .deps
                    .entry(crate_name.clone())
                    .or_default()
                    .insert(dep_dir.to_string());
            }
        }

        for captures in workspace_dep_re.captures_iter(&text) {
            let dep_name = captures.get(1).map(|m| m.as_str()).unwrap_or_default();
            if let Some(dep_dir) = ws_path_map.get(dep_name) {
                if dep_dir != crate_name && graph.all_crates.contains(dep_dir) {
                    graph
                        .deps
                        .entry(crate_name.clone())
                        .or_default()
                        .insert(dep_dir.clone());
                }
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
        if workspace_triggers.contains(file.as_str()) {
            classification.force_workspace = true;
            continue;
        }
        if shared_rust_triggers.contains(file.as_str()) {
            classification.has_shared_rust_changes = true;
            continue;
        }

        if GOVERNANCE_PREFIXES
            .iter()
            .any(|prefix| file.starts_with(prefix))
            || (!file.contains('/') && file.ends_with(".md"))
        {
            classification.has_governance_changes = true;
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

fn emit_local(scope: &str, run_rust: bool, run_governance: bool, verbose: bool) {
    if verbose {
        eprintln!("[ci-routing] run_rust={}", python_bool(run_rust));
        eprintln!(
            "[ci-routing] run_governance={}",
            python_bool(run_governance)
        );
        eprintln!(
            "[ci-routing] scope={}",
            if scope.is_empty() { "(skip)" } else { scope }
        );
    }
    println!("{scope}");
}

fn emit_github(scope: &str, run_rust: bool, run_governance: bool, verbose: bool) -> Result<()> {
    if verbose {
        eprintln!("[ci-routing] run_rust={}", python_bool(run_rust));
        eprintln!(
            "[ci-routing] run_governance={}",
            python_bool(run_governance)
        );
        eprintln!(
            "[ci-routing] rust_scope={}",
            if scope.is_empty() { "(skip)" } else { scope }
        );
    }

    if let Ok(output_path) = env::var("GITHUB_OUTPUT") {
        if !output_path.trim().is_empty() {
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
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&output_path)
                .with_context(|| format!("open GITHUB_OUTPUT {}", output_path))?
                .write_all(body.as_bytes())
                .with_context(|| format!("write GITHUB_OUTPUT {}", output_path))?;
            return Ok(());
        }
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
        let full_set = transitive_closure(&classification.affected_crates, &reverse);
        (format_cargo_scope(&full_set), true)
    } else if classification.has_shared_rust_changes {
        ("--workspace".to_string(), true)
    } else {
        (String::new(), false)
    };

    let mut final_scope = scope;
    let mut has_governance = classification.has_governance_changes;
    let mut final_run_rust = run_rust;

    if local_base_fallback && files.is_empty() {
        final_scope = "--workspace".to_string();
        final_run_rust = true;
        has_governance = true;
        if cli.verbose {
            eprintln!(
                "[ci-routing] local fallback could not inspect committed branch deltas; promoting to --workspace."
            );
        }
    }

    if cli.local {
        emit_local(&final_scope, final_run_rust, has_governance, cli.verbose);
    } else {
        emit_github(&final_scope, final_run_rust, has_governance, cli.verbose)?;
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
                } else if let Err(output_err) = emit_github("--workspace", true, true, cli.verbose)
                {
                    eprintln!("[ci-routing] ERROR: failed to emit fallback outputs: {output_err}");
                    return ExitCode::from(1);
                }
                ExitCode::SUCCESS
            }
        }
    }
}
