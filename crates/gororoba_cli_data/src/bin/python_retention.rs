use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use glob::Pattern;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "python-retention",
    about = "Classify remaining Python files under a machine-readable Rust-vs-Python retention policy"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Sync(SyncArgs),
    Verify(VerifyArgs),
}

#[derive(Parser, Debug)]
struct SyncArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/python_retention.toml")]
    policy: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct VerifyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/python_retention.toml")]
    policy: PathBuf,
    #[arg(long)]
    inventory: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct RetentionPolicyRoot {
    python_retention: RetentionPolicyMeta,
    rule: Vec<RuleRow>,
}

#[derive(Debug, Deserialize)]
struct RetentionPolicyMeta {
    version: u32,
    policy: String,
    inventory_output: String,
    #[allow(dead_code)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RuleRow {
    id: String,
    globs: Vec<String>,
    classification: String,
    action: String,
    canonical: bool,
    owner: String,
    rationale: String,
}

#[derive(Debug)]
struct CompiledRule {
    id: String,
    patterns: Vec<Pattern>,
    classification: String,
    action: String,
    canonical: bool,
    owner: String,
    rationale: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct InventoryRoot {
    python_retention_inventory: InventoryMeta,
    file: Vec<InventoryRow>,
}

#[derive(Debug, Serialize, Deserialize)]
struct InventoryMeta {
    version: u32,
    policy: String,
    file_count: usize,
    classified_count: usize,
    unmatched_count: usize,
    counts_by_classification: BTreeMap<String, usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct InventoryRow {
    path: String,
    matched_rule: String,
    classification: String,
    action: String,
    canonical: bool,
    owner: String,
    rationale: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Sync(args) => sync(args),
        Command::Verify(args) => verify(args),
    }
}

fn sync(args: SyncArgs) -> Result<()> {
    let repo_root = args.repo_root.canonicalize().context("resolve repo root")?;
    let policy_path = repo_root.join(&args.policy);
    let policy = load_policy(&policy_path)?;
    let inventory = build_inventory(&repo_root, &policy)?;
    let output_path = args
        .output
        .map(|path| repo_root.join(path))
        .unwrap_or_else(|| repo_root.join(&policy.python_retention.inventory_output));
    write_inventory(&output_path, &inventory)?;
    println!(
        "OK: wrote Python retention inventory: {} files={} unmatched={}",
        output_path.display(),
        inventory.python_retention_inventory.file_count,
        inventory.python_retention_inventory.unmatched_count
    );
    if inventory.python_retention_inventory.unmatched_count > 0 {
        bail!("python retention sync produced unmatched files");
    }
    Ok(())
}

fn verify(args: VerifyArgs) -> Result<()> {
    let repo_root = args.repo_root.canonicalize().context("resolve repo root")?;
    let policy_path = repo_root.join(&args.policy);
    let policy = load_policy(&policy_path)?;
    let expected = build_inventory(&repo_root, &policy)?;
    if expected.python_retention_inventory.unmatched_count > 0 {
        bail!(
            "python retention policy has {} unmatched files",
            expected.python_retention_inventory.unmatched_count
        );
    }
    let inventory_path = args
        .inventory
        .map(|path| repo_root.join(path))
        .unwrap_or_else(|| repo_root.join(&policy.python_retention.inventory_output));
    let actual_text = fs::read_to_string(&inventory_path)
        .with_context(|| format!("read inventory {}", inventory_path.display()))?;
    let actual: InventoryRoot =
        toml::from_str(&actual_text).context("parse python retention inventory")?;

    let expected_text = toml::to_string_pretty(&expected)?;
    let actual_normalized = toml::to_string_pretty(&actual)?;
    if expected_text != actual_normalized {
        bail!(
            "python retention inventory is stale: run `cargo run -p gororoba_cli_data --bin python-retention -- sync`"
        );
    }
    println!(
        "OK: Python retention inventory verified: files={} classifications={}",
        expected.python_retention_inventory.file_count,
        expected
            .python_retention_inventory
            .counts_by_classification
            .len()
    );
    Ok(())
}

fn load_policy(path: &Path) -> Result<RetentionPolicyRoot> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let raw: RetentionPolicyRoot =
        toml::from_str(&text).context("parse python retention policy")?;
    if raw.python_retention.version == 0 {
        bail!("python_retention.version must be positive");
    }
    if raw.python_retention.policy.trim().is_empty() {
        bail!("python_retention.policy must be non-empty");
    }
    if raw.rule.is_empty() {
        bail!("python retention policy must define at least one [[rule]]");
    }
    Ok(raw)
}

fn compile_rules(raw: &RetentionPolicyRoot) -> Result<Vec<CompiledRule>> {
    raw.rule
        .iter()
        .map(|rule| {
            if rule.id.trim().is_empty() {
                bail!("python retention rule missing id");
            }
            if rule.globs.is_empty() {
                bail!("python retention rule {} has no globs", rule.id);
            }
            let patterns = rule
                .globs
                .iter()
                .map(|glob| Pattern::new(glob).with_context(|| format!("invalid glob {glob}")))
                .collect::<Result<Vec<_>>>()?;
            Ok(CompiledRule {
                id: rule.id.clone(),
                patterns,
                classification: rule.classification.clone(),
                action: rule.action.clone(),
                canonical: rule.canonical,
                owner: rule.owner.clone(),
                rationale: rule.rationale.clone(),
            })
        })
        .collect()
}

fn build_inventory(repo_root: &Path, policy: &RetentionPolicyRoot) -> Result<InventoryRoot> {
    let rules = compile_rules(policy)?;
    let mut files = list_python_files(repo_root)?;
    files.sort();

    let mut rows = Vec::with_capacity(files.len());
    let mut unmatched = 0usize;
    let mut counts = BTreeMap::<String, usize>::new();

    for rel in files {
        let matched = rules.iter().find(|rule| {
            rule.patterns
                .iter()
                .any(|pattern| pattern.matches_path(Path::new(&rel)))
        });
        if let Some(rule) = matched {
            *counts.entry(rule.classification.clone()).or_default() += 1;
            rows.push(InventoryRow {
                path: rel,
                matched_rule: rule.id.clone(),
                classification: rule.classification.clone(),
                action: rule.action.clone(),
                canonical: rule.canonical,
                owner: rule.owner.clone(),
                rationale: rule.rationale.clone(),
            });
        } else {
            unmatched += 1;
            rows.push(InventoryRow {
                path: rel,
                matched_rule: "<unmatched>".to_string(),
                classification: "unmatched".to_string(),
                action: "extend_policy".to_string(),
                canonical: false,
                owner: "unclassified".to_string(),
                rationale: "File was not matched by any retention rule.".to_string(),
            });
        }
    }

    Ok(InventoryRoot {
        python_retention_inventory: InventoryMeta {
            version: policy.python_retention.version,
            policy: policy.python_retention.policy.clone(),
            file_count: rows.len(),
            classified_count: rows.len().saturating_sub(unmatched),
            unmatched_count: unmatched,
            counts_by_classification: counts,
        },
        file: rows,
    })
}

fn list_python_files(repo_root: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let skip_dirs: BTreeSet<&str> = BTreeSet::from([
        ".git",
        "target",
        ".cache",
        "venv",
        ".venv",
        ".venv_ingest",
        "wow_analysis_venv",
        "build",
        "__pycache__",
        "site-packages",
    ]);
    let walker = WalkDir::new(repo_root).into_iter().filter_entry(|entry| {
        let path = entry.path();
        if path
            .strip_prefix(repo_root)
            .ok()
            .map(|rel| rel.to_string_lossy().replace('\\', "/"))
            .is_some_and(|rel| rel.starts_with("crates/gororoba_py/venv/"))
        {
            return false;
        }
        if entry.file_type().is_dir()
            && let Some(name) = path.file_name().and_then(|value| value.to_str())
        {
            return !skip_dirs.contains(name);
        }
        true
    });
    for entry in walker.filter_map(Result::ok) {
        let path = entry.path();
        if !entry.file_type().is_file() || path.extension().and_then(|v| v.to_str()) != Some("py") {
            continue;
        }
        let rel = path
            .strip_prefix(repo_root)
            .with_context(|| format!("strip prefix for {}", path.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        out.push(rel);
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn write_inventory(path: &Path, inventory: &InventoryRoot) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create parent {}", parent.display()))?;
    }
    let body = toml::to_string_pretty(inventory).context("serialize inventory")?;
    fs::write(path, format!("{body}\n")).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}
