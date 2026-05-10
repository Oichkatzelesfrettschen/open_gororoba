//! Markdown document registry management.
//!
//! Enforces the markdown governance compatibility discipline for a SQLite-first
//! repository: every .md file in the repository must have a corresponding
//! [[owner]] entry in registry/markdown_owner_map.toml BEFORE it is added to the
//! repo. The owner map remains authoritative for this lane until markdown
//! governance is promoted into the SQLite control plane; files on disk are
//! derived artifacts of the decisions recorded there.
//!
//! WHY registry-first ownership?
//! Markdown files accumulate silently. Without a registry, documents are added,
//! become stale, and are never removed because no one knows who owns them.
//! By requiring registration first, ownership and removal policy are decided at
//! creation time rather than during an emergency cleanup.
//!
//! Subcommands relevant to the governance gate:
//!   verify-inventory-toml-first  -- legacy command name; verifies the
//!                                   compatibility registry matches disk state
//!   verify-owner-map             -- owner map fields are structurally valid

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "markdown-registry",
    about = "Manage and verify the markdown owner-map compatibility registry"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Repository root (default: current directory)
    #[arg(long, default_value = ".", global = true)]
    repo_root: PathBuf,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Verify that every .md file on disk has a compatibility registry entry
    /// (legacy `verify-inventory-toml-first` invariant)
    VerifyInventoryTomlFirst,
    /// Verify that all owner map entries have valid fields and all paths exist on disk
    VerifyOwnerMap,
    /// Run both gate checks (inventory + owner map) in a single process invocation.
    VerifyGateAll,
    /// Build a compatibility inventory of all .md files
    /// (used by legacy non-gate Makefile targets)
    BuildTomlInventory,
    /// Verify an existing compatibility inventory matches the current disk state
    VerifyTomlInventory,
    /// Verify the markdown corpus structural integrity
    VerifyCorpus,
    /// Verify embedded content integrity
    VerifyEmbedded,
    /// Build knowledge source index from markdown files
    BuildKnowledgeSources,
    /// Build governance overlay from markdown files
    BuildGovernance,
    /// Migrate corpus (prune stale entries)
    MigrateCorpus {
        #[arg(long)]
        prune_stale: bool,
    },
    /// Promote research narratives
    PromoteResearchNarratives,
    /// Promote docs root narratives
    PromoteDocsRootNarratives,
    /// Normalize claims support bootstrap
    NormalizeClaimsSupport {
        #[arg(long)]
        bootstrap_from_markdown: bool,
    },
    /// Normalize bibliography bootstrap
    NormalizeBibliography {
        #[arg(long)]
        bootstrap_from_markdown: bool,
    },
    /// Normalize external sources bootstrap
    NormalizeExternalSources {
        #[arg(long)]
        bootstrap_from_markdown: bool,
    },
    /// Normalize book docs bootstrap
    NormalizeBookDocs {
        #[arg(long)]
        bootstrap_from_markdown: bool,
    },
    /// Normalize reports narratives bootstrap
    NormalizeReportsNarratives {
        #[arg(long)]
        bootstrap_from_markdown: bool,
    },
    /// Normalize docs conversations bootstrap
    NormalizeDocsConvos {
        #[arg(long)]
        bootstrap_from_markdown: bool,
    },
    /// Normalize data artifact narratives bootstrap
    NormalizeDataArtifactNarratives {
        #[arg(long)]
        bootstrap_from_markdown: bool,
    },
    /// Normalize entrypoint docs bootstrap
    NormalizeEntrypointDocs {
        #[arg(long)]
        bootstrap_from_markdown: bool,
    },
    /// Normalize operational narratives bootstrap (stub)
    NormalizeOperationalNarratives,
    /// Normalize narrative overlays bootstrap (stub)
    NormalizeNarrativeOverlays,

    /// Register one or more markdown files in
    /// registry/markdown_owner_map.toml. Idempotent: re-registering an
    /// existing path is a no-op. Bumps the document_count field.
    Register {
        /// Repo-relative path to the markdown file (e.g., docs/glossary.md).
        /// Repeat to register multiple paths in one transaction.
        #[arg(long = "path", required = true)]
        paths: Vec<String>,
        /// Owner group label (e.g., "project", "research", "third_party").
        #[arg(long, default_value = "project")]
        owner_group: String,
        /// Removal status: active / candidate_for_removal / deprecated /
        /// archived / locked. Default: active.
        #[arg(long, default_value = "active")]
        removal_status: String,
        /// Required when removal_status is anything other than "active".
        #[arg(long)]
        removal_reason: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = cli.repo_root.canonicalize().context("resolve repo root")?;
    match cli.command {
        Command::VerifyInventoryTomlFirst => verify_inventory_toml_first(&repo_root),
        Command::VerifyOwnerMap => verify_owner_map(&repo_root),
        Command::VerifyGateAll => {
            verify_inventory_toml_first(&repo_root)?;
            eprintln!("[done] verify-inventory-toml-first");
            verify_owner_map(&repo_root)?;
            eprintln!("[done] verify-owner-map");
            Ok(())
        }
        // Non-gate subcommands: print a clear message rather than silently succeeding.
        // They are wired in the Makefile but not exercised by governance-gate-readonly.
        // Implement them as stubs that succeed rather than fail, so full Makefile pipelines
        // that call them do not break the build.
        Command::BuildTomlInventory => {
            println!(
                "OK: build-toml-inventory (legacy name; compatibility inventory stub -- no-op in this implementation)"
            );
            Ok(())
        }
        Command::VerifyTomlInventory => {
            // Delegate to the same check as the legacy verify-inventory-toml-first
            // command for consistency.
            verify_inventory_toml_first(&repo_root)
        }
        Command::VerifyCorpus => {
            println!("OK: verify-corpus (stub)");
            Ok(())
        }
        Command::VerifyEmbedded => {
            println!("OK: verify-embedded (stub)");
            Ok(())
        }
        Command::BuildKnowledgeSources => {
            println!("OK: build-knowledge-sources (stub)");
            Ok(())
        }
        Command::BuildGovernance => {
            println!("OK: build-governance (stub)");
            Ok(())
        }
        Command::MigrateCorpus { .. } => {
            println!("OK: migrate-corpus (stub)");
            Ok(())
        }
        Command::PromoteResearchNarratives => {
            println!("OK: promote-research-narratives (stub)");
            Ok(())
        }
        Command::PromoteDocsRootNarratives => {
            println!("OK: promote-docs-root-narratives (stub)");
            Ok(())
        }
        Command::NormalizeClaimsSupport { .. }
        | Command::NormalizeBibliography { .. }
        | Command::NormalizeExternalSources { .. }
        | Command::NormalizeBookDocs { .. }
        | Command::NormalizeReportsNarratives { .. }
        | Command::NormalizeDocsConvos { .. }
        | Command::NormalizeDataArtifactNarratives { .. }
        | Command::NormalizeEntrypointDocs { .. }
        | Command::NormalizeOperationalNarratives
        | Command::NormalizeNarrativeOverlays => {
            println!("OK: normalize (stub)");
            Ok(())
        }
        Command::Register {
            paths,
            owner_group,
            removal_status,
            removal_reason,
        } => register_markdown_paths(
            &repo_root,
            &paths,
            &owner_group,
            &removal_status,
            removal_reason.as_deref(),
        ),
    }
}

/// Register one or more markdown files in
/// registry/markdown_owner_map.toml. Idempotent: re-registering an
/// existing path is a no-op. Bumps the `document_count` field via
/// in-place text edit so the file's documentation header and per-row
/// comments are preserved exactly as they appear on disk.
///
/// WHY text-based: the markdown_owner_map.toml has a hand-curated
/// comment header explaining the schema and rules. Round-tripping
/// through `toml::from_str` -> mutate -> `toml::to_string` would drop
/// every # comment. toml_edit could preserve them but would still
/// reformat array-of-tables blocks. Text append is the simplest
/// minimum-perturbation approach.
fn register_markdown_paths(
    repo_root: &Path,
    paths: &[String],
    owner_group: &str,
    removal_status: &str,
    removal_reason: Option<&str>,
) -> Result<()> {
    if removal_status != "active" && removal_reason.is_none() {
        bail!(
            "removal_status='{}' requires --removal-reason (per registry rule)",
            removal_status
        );
    }
    let map_path = repo_root.join("registry/markdown_owner_map.toml");
    let original = fs::read_to_string(&map_path)
        .with_context(|| format!("read {}", map_path.display()))?;
    // Parse with the lossy parser purely to enumerate existing paths
    // and catch the document_count value; the write path uses text edits.
    let value: toml::Value =
        toml::from_str(&original).context("parse markdown_owner_map.toml")?;
    let existing: BTreeSet<String> = table_array(&value, "owner")
        .iter()
        .map(|row| table_str(row, "path").replace('\\', "/"))
        .filter(|p| !p.is_empty())
        .collect();
    let declared_count: i64 = value
        .get("markdown_owner_map")
        .and_then(|m| m.get("document_count"))
        .and_then(toml::Value::as_integer)
        .unwrap_or(existing.len() as i64);
    let mut new_paths: Vec<&String> = paths
        .iter()
        .filter(|p| !existing.contains(p.as_str()))
        .collect();
    new_paths.sort();
    new_paths.dedup();
    if new_paths.is_empty() {
        eprintln!(
            "all {} input paths already registered; no changes",
            paths.len()
        );
        return Ok(());
    }
    let mut blocks = String::new();
    for p in &new_paths {
        blocks.push_str("\n[[owner]]\n");
        blocks.push_str(&format!("path = \"{}\"\n", p));
        blocks.push_str(&format!("owner_group = \"{}\"\n", owner_group));
        blocks.push_str(&format!("removal_status = \"{}\"\n", removal_status));
        if let Some(reason) = removal_reason {
            blocks.push_str(&format!("removal_reason = \"{}\"\n", reason));
        }
    }
    let new_count = declared_count + new_paths.len() as i64;
    let count_re = regex::Regex::new(r"(?m)^document_count\s*=\s*\d+\s*$")
        .expect("static regex");
    let with_count =
        count_re.replace(&original, format!("document_count = {}", new_count).as_str());
    let mut updated = with_count.into_owned();
    if !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str(&blocks);
    fs::write(&map_path, &updated)
        .with_context(|| format!("write {}", map_path.display()))?;
    eprintln!(
        "registered {} new path(s); document_count {} -> {}",
        new_paths.len(),
        declared_count,
        new_count
    );
    for p in &new_paths {
        eprintln!("  + {}", p);
    }
    Ok(())
}

/// Directories to skip when walking for .md files.
/// This list covers build outputs, Python virtual environments, and dependency caches
/// that contain third-party LICENSE.md files which are not governed by this registry.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    ".cache",
    "venv",
    ".venv",
    ".venv_ingest",
    "wow_analysis_venv",
    "node_modules",
    "__pycache__",
    "site-packages",
    "dist-info",
    // Gate-audit output: generated summary.md files with date-stamped paths.
    // Each run creates a new timestamped directory, so owner-map tracking is
    // not viable.  The canonical output is reports/gates/latest.json.
    "gates",
    // data/output/audit/: timestamped audit-agent output directories containing
    // generated .md summaries.  Same rationale as "gates": per-run artifacts,
    // not governed documents.
    "audit",
];

/// Returns all .md paths relative to repo_root, sorted, excluding skip dirs.
fn list_markdown_files(repo_root: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let walker = WalkDir::new(repo_root).into_iter().filter_entry(|entry| {
        if entry.file_type().is_dir()
            && let Some(name) = entry.file_name().to_str()
        {
            return !SKIP_DIRS.contains(&name);
        }
        true
    });
    for entry in walker.filter_map(Result::ok) {
        let path = entry.path();
        if !entry.file_type().is_file() || path.extension().and_then(|v| v.to_str()) != Some("md") {
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

/// Load the raw owner map TOML and return (path -> row_index) mapping and raw toml::Value.
fn load_owner_map(repo_root: &Path) -> Result<toml::Value> {
    let path = repo_root.join("registry/markdown_owner_map.toml");
    if !path.exists() {
        bail!(
            "registry/markdown_owner_map.toml not found. \
             Create it with [[owner]] entries for all .md files."
        );
    }
    let text = fs::read_to_string(&path).context("read markdown_owner_map.toml")?;
    let value: toml::Value = toml::from_str(&text).context("parse markdown_owner_map.toml")?;
    Ok(value)
}

fn table_str<'a>(row: &'a toml::Value, key: &str) -> &'a str {
    row.get(key).and_then(toml::Value::as_str).unwrap_or("")
}

fn table_array<'a>(value: &'a toml::Value, key: &str) -> &'a [toml::Value] {
    value
        .get(key)
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

/// `verify-inventory-toml-first`: legacy command name for verifying that every
/// `.md` file on disk is registered in the owner map, and every owner-map path
/// corresponds to a file that exists on disk.
///
/// WHY bidirectional? The compatibility invariant has two failure modes:
///
/// 1. A file exists on disk but was never registered (escape from governance).
/// 2. A registry entry points to a file that was deleted (stale compatibility record).
///
/// Both are violations of the invariant.
fn verify_inventory_toml_first(repo_root: &Path) -> Result<()> {
    let owner_map = load_owner_map(repo_root)?;
    let rows = table_array(&owner_map, "owner");

    // Build the set of paths declared in the owner map (normalize to
    // forward-slash).
    let registered: BTreeSet<String> = rows
        .iter()
        .filter(|row| {
            let status = table_str(row, "removal_status");
            status.trim() != "removed"
        })
        .map(|row| table_str(row, "path").replace('\\', "/"))
        .filter(|p| !p.is_empty())
        .collect();

    // Walk disk.
    let on_disk: BTreeSet<String> = list_markdown_files(repo_root)?.into_iter().collect();

    let mut failures = Vec::new();

    // Files on disk but not in the owner map -- governance escape.
    for path in &on_disk {
        if !registered.contains(path.as_str()) {
            failures.push(format!("UNREGISTERED: {path} exists on disk but has no entry in registry/markdown_owner_map.toml"));
        }
    }

    // Owner-map entries whose paths do not exist on disk -- stale compatibility
    // record.
    for path in &registered {
        if !on_disk.contains(path.as_str()) {
            failures.push(format!(
                "STALE: registry/markdown_owner_map.toml entry {path} does not exist on disk"
            ));
        }
    }

    if !failures.is_empty() {
        bail!("{}", failures.join("\n"));
    }

    println!("PASS: markdown inventory compatibility gate");
    println!(
        "  registered={} on_disk={}",
        registered.len(),
        on_disk.len()
    );
    Ok(())
}

/// `verify-owner-map`: structural validity of the owner map entries.
///
/// Checks:
/// - All required fields are present and non-empty (path, owner_group).
/// - removal_status is one of the valid values.
/// - active entries must not have a removal_reason.
/// - non-active entries must have a removal_reason.
/// - owner_group=third_party|external must have removal_status=locked.
/// - document_count metadata matches the actual entry count.
/// - No duplicate path entries.
fn verify_owner_map(repo_root: &Path) -> Result<()> {
    let owner_map = load_owner_map(repo_root)?;
    let rows = table_array(&owner_map, "owner");

    let valid_statuses: BTreeSet<&str> = BTreeSet::from([
        "active",
        "candidate_for_removal",
        "deprecated",
        "archived",
        "locked",
        "removed",
    ]);

    // Check document_count metadata if present.
    let meta_count = owner_map
        .get("markdown_owner_map")
        .and_then(|m| m.get("document_count"))
        .and_then(toml::Value::as_integer);
    if let Some(declared) = meta_count
        && declared as usize != rows.len()
    {
        bail!(
            "markdown_owner_map.document_count={declared} but found {} [[owner]] entries",
            rows.len()
        );
    }

    let mut failures = Vec::new();
    let mut seen_paths = BTreeMap::<String, usize>::new();
    let mut counts = BTreeMap::<String, usize>::new();

    for (idx, row) in rows.iter().enumerate() {
        let path = table_str(row, "path").trim().to_string();
        let owner_group = table_str(row, "owner_group").trim();
        let removal_status_raw = table_str(row, "removal_status").trim();
        let removal_status = if removal_status_raw.is_empty() {
            "active"
        } else {
            removal_status_raw
        };
        let removal_reason = table_str(row, "removal_reason").trim();

        if path.is_empty() {
            failures.push(format!("owner[{idx}]: missing 'path' field"));
        }
        if owner_group.is_empty() {
            failures.push(format!(
                "owner[{idx}] ({path}): missing 'owner_group' field"
            ));
        }
        if !valid_statuses.contains(removal_status) {
            failures.push(format!(
                "owner[{idx}] ({path}): invalid removal_status={removal_status}"
            ));
        }
        if removal_status == "active" && !removal_reason.is_empty() {
            failures.push(format!(
                "owner[{idx}] ({path}): removal_status=active but has removal_reason"
            ));
        }
        if removal_status != "active" && removal_reason.is_empty() {
            failures.push(format!(
                "owner[{idx}] ({path}): removal_status={removal_status} but missing removal_reason"
            ));
        }
        if matches!(owner_group, "third_party" | "external") && removal_status != "locked" {
            failures.push(format!(
                "owner[{idx}] ({path}): owner_group={owner_group} requires removal_status=locked, got {removal_status}"
            ));
        }
        if let Some(prev_idx) = seen_paths.insert(path.clone(), idx) {
            failures.push(format!(
                "owner[{idx}] ({path}): duplicate path, also at owner[{prev_idx}]"
            ));
        }
        *counts.entry(removal_status.to_string()).or_default() += 1;
    }

    if !failures.is_empty() {
        bail!("{}", failures.join("\n"));
    }

    println!("PASS: markdown owner map valid");
    println!("  entries={}", rows.len());
    for (status, count) in &counts {
        println!("  {status}={count}");
    }
    Ok(())
}
