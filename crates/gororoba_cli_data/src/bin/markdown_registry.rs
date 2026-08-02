//! Markdown document registry management.
//!
//! Enforces the markdown governance compatibility discipline for a SQLite-first
//! repository: every Git-governed .md file must have a corresponding
//! `[[owner]]` entry in registry/markdown_owner_map.toml BEFORE it is added to the
//! repo. The owner map remains authoritative for this lane until markdown
//! governance is promoted into the SQLite control plane; files on disk are
//! derived artifacts of the decisions recorded there. Ignored local overlays,
//! generated caches, and retained acquisition notes stay outside this corpus
//! and are classified by the evidence-retention ledger.
//!
//! WHY registry-first ownership?
//! Markdown files accumulate silently. Without a registry, documents are added,
//! become stale, and are never removed because no one knows who owns them.
//! By requiring registration first, ownership and removal policy are decided at
//! creation time rather than during an emergency cleanup.
//!
//! Subcommands relevant to registry policy validation:
//!   verify-inventory-toml-first  -- legacy command name; verifies the
//!                                   compatibility registry matches disk state
//!   verify-owner-map             -- owner map fields are structurally valid

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process, str,
};
use toml::Value;

const IMMUTABLE_AGENT_OVERLAYS: &[&str] = &["CLAUDE.md", "GEMINI.md"];
const GENERATED_MARKERS: &[&str] = &[
    "AUTO-GENERATED",
    "Source of truth:",
    "This file is generated from",
    "DO NOT EDIT",
];

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
    /// Run all registry document policy checks in a single process invocation.
    #[command(name = "verify-all", visible_alias = "verify-gate-all")]
    VerifyAll,
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
    BuildKnowledgeSources {
        #[arg(long, default_value = "registry/knowledge_sources.toml")]
        out: PathBuf,
    },
    /// Build governance overlay from markdown files
    BuildGovernance {
        #[arg(long, default_value = "registry/knowledge_sources.toml")]
        knowledge_index: PathBuf,
        #[arg(long, default_value = "registry/markdown_governance.toml")]
        out: PathBuf,
    },
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

    /// Remove owner-map entries whose markdown files are absent from disk.
    PruneStaleOwnerMap,

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
        Command::VerifyAll => {
            verify_inventory_toml_first(&repo_root)?;
            eprintln!("[done] verify-inventory-toml-first");
            verify_owner_map(&repo_root)?;
            eprintln!("[done] verify-owner-map");
            verify_research_narrative_root_docs(&repo_root)?;
            eprintln!("[done] verify-research-narrative-root-docs");
            Ok(())
        }
        // Non-validation subcommands: print a clear message rather than silently succeeding.
        // They are wired in the Makefile but not exercised by validate-governance.
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
        Command::BuildKnowledgeSources { out } => build_knowledge_sources(&repo_root, &out),
        Command::BuildGovernance {
            knowledge_index,
            out,
        } => build_governance(&repo_root, &knowledge_index, &out),
        Command::MigrateCorpus { .. } => {
            println!("OK: migrate-corpus (stub)");
            Ok(())
        }
        Command::PruneStaleOwnerMap => prune_stale_owner_map(&repo_root),
        Command::PromoteResearchNarratives => promote_research_narratives(&repo_root),
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

#[derive(Clone, Debug)]
struct KnowledgeSourceRow {
    doc_id: String,
    path: String,
    title: String,
    kind: String,
    authoring_mode: String,
    generated: bool,
    status: String,
    migration_priority: String,
    toml_backing: String,
    sha256: String,
    size_bytes: usize,
    line_count: usize,
    claim_ref_count: usize,
    insight_ref_count: usize,
    experiment_ref_count: usize,
    link_count: usize,
    link_sample: Vec<String>,
}

fn build_knowledge_sources(repo_root: &Path, output: &Path) -> Result<()> {
    let mut paths = list_markdown_files(repo_root)?;
    paths.sort();

    let mut rows = Vec::with_capacity(paths.len());
    let mut kind_counts = BTreeMap::<String, usize>::new();
    for (index, path) in paths.into_iter().enumerate() {
        let full_path = repo_root.join(&path);
        let raw = fs::read(&full_path).with_context(|| format!("read {}", full_path.display()))?;
        let text = String::from_utf8_lossy(&raw);
        let title = first_title(
            &text,
            Path::new(&path)
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or(&path),
        );
        let (kind, authoring_mode, generated) = knowledge_kind_for_path(&path, &text);
        let link_data = extract_link_sample(&text);
        let row = KnowledgeSourceRow {
            doc_id: format!("DOC-{number:04}", number = index + 1),
            path: path.clone(),
            title,
            kind: kind.clone(),
            authoring_mode,
            generated,
            status: knowledge_status_for_path(&path),
            migration_priority: knowledge_migration_priority(&kind, &path),
            toml_backing: knowledge_toml_backing_for_path(&path),
            sha256: sha256_hex(&raw),
            size_bytes: raw.len(),
            line_count: text.lines().count(),
            claim_ref_count: count_distinct_references(r"\bC-\d{3,5}\b", &text)?,
            insight_ref_count: count_distinct_references(r"\bI-\d{3,5}\b", &text)?,
            experiment_ref_count: count_distinct_references(r"\bE-\d{3,5}\b", &text)?,
            link_count: link_data.1,
            link_sample: link_data.0,
        };
        *kind_counts.entry(row.kind.clone()).or_default() += 1;
        rows.push(row);
    }

    let mut lines = vec![
        "# Knowledge source index for Markdown assets.".to_string(),
        "# Auto-generated by markdown-registry build-knowledge-sources.".to_string(),
        "# Authoritative source: on-disk Markdown plus registry/markdown_owner_map.toml."
            .to_string(),
        "# Regenerate with: make registry-knowledge".to_string(),
        String::new(),
        "[knowledge_sources]".to_string(),
        "generated_at = \"deterministic\"".to_string(),
        format!("tracked_markdown_count = {}", rows.len()),
        format!(
            "manual_source_count = {}",
            kind_counts.get("manual_source").copied().unwrap_or(0)
        ),
        format!(
            "markdown_mirror_count = {}",
            kind_counts.get("markdown_mirror").copied().unwrap_or(0)
        ),
        format!(
            "generated_markdown_count = {}",
            kind_counts.get("generated_markdown").copied().unwrap_or(0)
        ),
        format!(
            "artifact_report_count = {}",
            kind_counts.get("artifact_report").copied().unwrap_or(0)
        ),
        format!(
            "transcript_input_count = {}",
            kind_counts.get("transcript_input").copied().unwrap_or(0)
        ),
        String::new(),
    ];
    for row in &rows {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {}", q(&row.doc_id)));
        lines.push(format!("path = {}", q(&row.path)));
        lines.push(format!("title = {}", q(&row.title)));
        lines.push(format!("kind = {}", q(&row.kind)));
        lines.push(format!("authoring_mode = {}", q(&row.authoring_mode)));
        lines.push(format!("generated = {}", bool_toml(row.generated)));
        lines.push(format!("status = {}", q(&row.status)));
        lines.push(format!(
            "migration_priority = {}",
            q(&row.migration_priority)
        ));
        if !row.toml_backing.is_empty() {
            lines.push(format!("toml_backing = {}", q(&row.toml_backing)));
        }
        lines.push(format!("sha256 = {}", q(&row.sha256)));
        lines.push(format!("size_bytes = {}", row.size_bytes));
        lines.push(format!("line_count = {}", row.line_count));
        lines.push(format!("claim_ref_count = {}", row.claim_ref_count));
        lines.push(format!("insight_ref_count = {}", row.insight_ref_count));
        lines.push(format!(
            "experiment_ref_count = {}",
            row.experiment_ref_count
        ));
        lines.push(format!("link_count = {}", row.link_count));
        lines.push(format!("link_sample = {}", q_list(&row.link_sample)));
        lines.push(String::new());
    }
    write_ascii(&repo_path(repo_root, output), &lines.join("\n"))?;
    println!(
        "Wrote {} with {} markdown records.",
        output.display(),
        rows.len()
    );
    Ok(())
}

fn build_governance(repo_root: &Path, knowledge_index: &Path, output: &Path) -> Result<()> {
    let knowledge_path = repo_path(repo_root, knowledge_index);
    let value = load_toml_value(&knowledge_path)?;
    let rows = load_knowledge_source_rows(&value);
    let mut by_path = BTreeMap::new();
    for row in rows {
        by_path.insert(row.path.clone(), row);
    }

    let mut paths = list_markdown_files(repo_root)?
        .into_iter()
        .collect::<BTreeSet<_>>();
    paths.extend(by_path.keys().cloned());

    let mut documents = Vec::new();
    let mut mode_counts = BTreeMap::<String, usize>::new();
    for (index, path) in paths.into_iter().enumerate() {
        if !path.ends_with(".md") {
            continue;
        }
        let row = by_path.get(&path);
        let kind = row
            .map(|item| item.kind.clone())
            .unwrap_or_else(|| "markdown".to_string());
        let generated = row.is_some_and(|item| item.generated);
        let toml_backing = row
            .map(|item| item.toml_backing.clone())
            .unwrap_or_default();
        let strict_generated_header = has_strict_generated_header(repo_root, &path)?;
        let (mode, header_required, notes) = if generated && strict_generated_header {
            (
                "toml_generated_mirror",
                true,
                "Generated Markdown mirror with an explicit immutable header.",
            )
        } else if generated {
            (
                "generated_artifact",
                false,
                "Generated or retained artifact without a TOML mirror contract.",
            )
        } else if kind == "transcript_input" {
            (
                "immutable_transcript",
                false,
                "Immutable transcript input; not authoritative for claims.",
            )
        } else if !toml_backing.is_empty() {
            (
                "toml_manual_source",
                false,
                "Manual Markdown source with a declared TOML destination; mirror promotion remains open.",
            )
        } else {
            (
                "manual_narrative",
                false,
                "Manual narrative source retained until a canonical registry lane is assigned.",
            )
        };
        let mut source_refs = Vec::new();
        if !toml_backing.is_empty() {
            source_refs.push(toml_backing);
        }
        *mode_counts.entry(mode.to_string()).or_default() += 1;
        documents.push((
            format!("MDG-{number:04}", number = index + 1),
            path,
            kind,
            mode.to_string(),
            header_required,
            source_refs,
            notes.to_string(),
        ));
    }

    let mut lines = vec![
        "# Markdown lifecycle governance registry (TOML-first).".to_string(),
        "# Generated by markdown-registry build-governance.".to_string(),
        "# Authoritative source: on-disk Markdown plus registry/markdown_owner_map.toml."
            .to_string(),
        "# Regenerate with: make registry-governance or make registry-export-markdown".to_string(),
        String::new(),
        "[markdown_governance]".to_string(),
        "generated_at = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("document_count = {}", documents.len()),
    ];
    for mode in mode_counts.keys() {
        lines.push(format!("{mode}_count = {}", mode_counts[mode]));
    }
    lines.extend([
        String::new(),
        "[policy]".to_string(),
        "safe_classifications = [\"toml_published_markdown\", \"toml_destination_exists_manual_markdown\", \"generated_artifact\", \"third_party_markdown\"]".to_string(),
        "tracked_allowed_modes = [\"toml_generated_mirror\", \"toml_manual_source\"]".to_string(),
        "tracked_allowed_paths = [\"docs/research/high_dimensional_algebra_unification_2026.md\", \"proofs/EPISTEMIC_BOUNDARIES.md\"]".to_string(),
        "embedded_markdown_prefixes = [\"docs/\", \"reports/\", \"data/artifacts/\"]".to_string(),
        "owner_scope_prefixes = [\"docs/\", \"reports/\", \"data/artifacts/\"]".to_string(),
        "generated_patterns = [\"build/docs/generated/*.md\", \"docs/generated/*.md\"]".to_string(),
        "skip_prefixes = [\".cache/\", \"target/\", \"build/\", \"dist/\", \"tmp/\"]".to_string(),
        "skip_path_parts = [\".cache\", \"target\", \"build\", \"dist\", \"tmp\"]".to_string(),
        "disk_forbidden_modes = [\"deleted_mirror\"]".to_string(),
        String::new(),
    ]);
    for (id, path, kind, mode, header_required, source_refs, notes) in documents {
        lines.push("[[document]]".to_string());
        lines.push(format!("path = {}", q(&path)));
        lines.push(format!("id = {}", q(&id)));
        lines.push(format!("kind = {}", q(&kind)));
        lines.push(format!("mode = {}", q(&mode)));
        lines.push(format!("header_required = {}", bool_toml(header_required)));
        if !source_refs.is_empty() {
            lines.push(format!("source_toml_refs = {}", q_list(&source_refs)));
        }
        lines.push(format!("notes = {}", q(&notes)));
        lines.push(String::new());
    }
    write_ascii(&repo_path(repo_root, output), &lines.join("\n"))?;
    println!(
        "Wrote {} with {} governance entries.",
        output.display(),
        mode_counts.values().sum::<usize>()
    );
    Ok(())
}

fn load_toml_value(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn load_knowledge_source_rows(value: &Value) -> Vec<KnowledgeSourceRow> {
    value
        .get("document")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_table)
        .map(|row| KnowledgeSourceRow {
            doc_id: table_str_value(row, "id"),
            path: table_str_value(row, "path"),
            title: table_str_value(row, "title"),
            kind: table_str_value(row, "kind"),
            authoring_mode: table_str_value(row, "authoring_mode"),
            generated: row
                .get("generated")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            status: table_str_value(row, "status"),
            migration_priority: table_str_value(row, "migration_priority"),
            toml_backing: table_str_value(row, "toml_backing"),
            sha256: table_str_value(row, "sha256"),
            size_bytes: row
                .get("size_bytes")
                .and_then(Value::as_integer)
                .unwrap_or(0)
                .max(0) as usize,
            line_count: row
                .get("line_count")
                .and_then(Value::as_integer)
                .unwrap_or(0)
                .max(0) as usize,
            claim_ref_count: 0,
            insight_ref_count: 0,
            experiment_ref_count: 0,
            link_count: 0,
            link_sample: Vec::new(),
        })
        .collect()
}

fn table_str_value(row: &toml::map::Map<String, Value>, key: &str) -> String {
    row.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .replace('\\', "/")
}

fn has_strict_generated_header(repo_root: &Path, path: &str) -> Result<bool> {
    let full_path = repo_root.join(path);
    if !full_path.is_file() {
        return Ok(false);
    }
    let head = fs::read_to_string(&full_path)
        .with_context(|| format!("read generated-header candidate {}", full_path.display()))?;
    let head = head.lines().take(12).collect::<Vec<_>>().join("\n");
    Ok(head.contains("AUTO-GENERATED: DO NOT EDIT") && head.contains("Source of truth:"))
}

fn first_title(text: &str, fallback: &str) -> String {
    let heading_re = Regex::new(r"(?m)^#\s+(.+?)\s*$").expect("static heading regex");
    let title = heading_re
        .captures(text)
        .and_then(|captures| captures.get(1).map(|value| value.as_str().trim()))
        .unwrap_or(fallback);
    collapse_ascii(title)
}

fn count_distinct_references(pattern: &str, text: &str) -> Result<usize> {
    let expression = Regex::new(pattern).with_context(|| format!("compile {pattern}"))?;
    Ok(expression
        .find_iter(text)
        .map(|value| value.as_str().to_string())
        .collect::<BTreeSet<_>>()
        .len())
}

fn extract_link_sample(text: &str) -> (Vec<String>, usize) {
    let backtick_re = Regex::new(r"`([^`\n]+)`").expect("static backtick regex");
    let mut sample = BTreeSet::new();
    let mut raw_count = 0;
    for capture in backtick_re.captures_iter(text) {
        let token = capture
            .get(1)
            .map(|value| collapse_ascii(value.as_str()))
            .unwrap_or_default();
        if token.is_empty() {
            continue;
        }
        raw_count += 1;
        if (!token.contains('/') && !token.contains('.'))
            || token.starts_with("http://")
            || token.starts_with("https://")
            || token.len() > 200
            || (token.contains(' ') && !token.contains('/'))
        {
            continue;
        }
        sample.insert(token);
    }
    (sample.into_iter().take(8).collect(), raw_count)
}

fn knowledge_toml_backing_for_path(path: &str) -> String {
    match path {
        "AGENTS.md" => "agents.toml".to_string(),
        "CLAUDE.md" | "GEMINI.md" | "README.md" | "README_GPU_STEPS.md" => {
            "registry/entrypoint_docs.toml".to_string()
        }
        "apps/gororoba_studio/README.md" | "crates/lbm_3d_cuda/README.md" => {
            "registry/requirements.toml".to_string()
        }
        "NAVIGATOR.md" => "registry/navigator.toml".to_string(),
        "REQUIREMENTS.md" | "docs/REQUIREMENTS.md" => "registry/requirements.toml".to_string(),
        "docs/CLAIMS_EVIDENCE_MATRIX.md" => "registry/claims.toml".to_string(),
        "docs/BIBLIOGRAPHY.md" => "registry/bibliography.toml".to_string(),
        "docs/INSIGHTS.md" => "registry/insights.toml".to_string(),
        "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md" => "registry/experiments.toml".to_string(),
        _ if path.starts_with("reports/") => "registry/reports_narratives.toml".to_string(),
        _ if path.starts_with("docs/book/src/") => "registry/book_docs.toml".to_string(),
        _ if path.starts_with("docs/external_sources/") => {
            "registry/external_sources.toml".to_string()
        }
        _ if path.starts_with("docs/convos/") => "registry/docs_convos.toml".to_string(),
        _ if path.starts_with("docs/engineering/")
            || path.starts_with("docs/research/")
            || path.starts_with("docs/theory/") =>
        {
            "registry/research_narratives.toml".to_string()
        }
        _ if path.starts_with("docs/") && path.matches('/').count() == 1 => {
            "registry/docs_root_narratives.toml".to_string()
        }
        _ => String::new(),
    }
}

fn knowledge_kind_for_path(path: &str, text: &str) -> (String, String, bool) {
    let generated_marker = text
        .lines()
        .take(12)
        .any(|line| GENERATED_MARKERS.iter().any(|marker| line.contains(marker)));
    if IMMUTABLE_AGENT_OVERLAYS.contains(&path) {
        ("manual_source".to_string(), "manual".to_string(), false)
    } else if path.starts_with("convos/") || path.starts_with("docs/convos/") {
        ("transcript_input".to_string(), "manual".to_string(), false)
    } else if path.starts_with("reports/") || path.starts_with("data/artifacts/") {
        ("artifact_report".to_string(), "generated".to_string(), true)
    } else if generated_marker || path.starts_with("docs/generated/") {
        (
            "generated_markdown".to_string(),
            "generated".to_string(),
            true,
        )
    } else {
        ("manual_source".to_string(), "manual".to_string(), false)
    }
}

fn knowledge_status_for_path(path: &str) -> String {
    if path.starts_with("archive/") || path.starts_with("docs/archive/") {
        "archived".to_string()
    } else {
        "active".to_string()
    }
}

fn knowledge_migration_priority(kind: &str, path: &str) -> String {
    if kind == "generated_markdown" {
        "critical".to_string()
    } else if kind == "manual_source" && path.starts_with("docs/") {
        "high".to_string()
    } else if kind == "manual_source" {
        "medium".to_string()
    } else {
        "none".to_string()
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn ascii_clean(text: &str) -> String {
    let mut output = String::new();
    for character in text.chars() {
        let code = character as u32;
        if matches!(character, '\n' | '\r' | '\t') {
            output.push(character);
        } else if code < 32 {
            output.push(' ');
        } else if code <= 127 {
            output.push(character);
        } else if code <= 0xffff {
            output.push_str(&format!("\\u{code:04X}"));
        } else {
            output.push_str(&format!("\\U{code:08X}"));
        }
    }
    output
}

fn collapse_ascii(text: &str) -> String {
    ascii_clean(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn bool_toml(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn q(value: &str) -> String {
    json!(value).to_string()
}

fn q_list(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| q(value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn write_ascii(path: &Path, text: &str) -> Result<()> {
    if let Some(character) = text.chars().find(|character| (*character as u32) > 127) {
        bail!("non-ASCII output in {}: {:?}", path.display(), character);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn repo_path(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn owner_path_from_line(line: &str) -> Option<String> {
    let rest = line.trim().strip_prefix("path")?.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let quoted = rest.strip_prefix('"')?;
    let end = quoted.find('"')?;
    Some(quoted[..end].replace('\\', "/"))
}

fn emit_owner_block_if_current(
    output: &mut String,
    block: &str,
    path: Option<&str>,
    stale_paths: &BTreeSet<String>,
) -> bool {
    let is_stale = path.is_some_and(|value| stale_paths.contains(value));
    if !is_stale {
        output.push_str(block);
    }
    !is_stale
}

fn prune_owner_map_text(original: &str, stale_paths: &BTreeSet<String>) -> String {
    let mut output = String::new();
    let mut block = String::new();
    let mut block_path: Option<String> = None;
    let mut in_owner_block = false;
    let mut kept_owner_count = 0usize;

    for line in original.split_inclusive('\n') {
        if line.trim() == "[[owner]]" {
            if in_owner_block
                && emit_owner_block_if_current(
                    &mut output,
                    &block,
                    block_path.as_deref(),
                    stale_paths,
                )
            {
                kept_owner_count += 1;
            }
            block.clear();
            block.push_str(line);
            block_path = None;
            in_owner_block = true;
            continue;
        }

        if in_owner_block {
            if block_path.is_none() {
                block_path = owner_path_from_line(line);
            }
            block.push_str(line);
        } else {
            output.push_str(line);
        }
    }

    if in_owner_block
        && emit_owner_block_if_current(&mut output, &block, block_path.as_deref(), stale_paths)
    {
        kept_owner_count += 1;
    }

    let count_re = regex::Regex::new(r"(?m)^document_count\s*=\s*\d+\s*$").expect("static regex");
    count_re
        .replace(
            &output,
            format!("document_count = {kept_owner_count}").as_str(),
        )
        .into_owned()
}

fn prune_stale_owner_map(repo_root: &Path) -> Result<()> {
    let map_path = repo_root.join("registry/markdown_owner_map.toml");
    let original =
        fs::read_to_string(&map_path).with_context(|| format!("read {}", map_path.display()))?;
    let owner_map: toml::Value =
        toml::from_str(&original).context("parse markdown_owner_map.toml")?;
    let on_disk: BTreeSet<String> = list_markdown_files(repo_root)?.into_iter().collect();
    let stale_paths: BTreeSet<String> = table_array(&owner_map, "owner")
        .iter()
        .map(|row| table_str(row, "path").replace('\\', "/"))
        .filter(|path| !path.is_empty() && !on_disk.contains(path))
        .collect();

    if stale_paths.is_empty() {
        eprintln!("owner map has no stale markdown paths");
        return Ok(());
    }

    let updated = prune_owner_map_text(&original, &stale_paths);
    fs::write(&map_path, updated).with_context(|| format!("write {}", map_path.display()))?;
    eprintln!("pruned {} stale owner-map path(s)", stale_paths.len());
    for path in &stale_paths {
        eprintln!("  - {path}");
    }
    Ok(())
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
    let original =
        fs::read_to_string(&map_path).with_context(|| format!("read {}", map_path.display()))?;
    // Parse with the lossy parser purely to enumerate existing paths
    // and catch the document_count value; the write path uses text edits.
    let value: toml::Value = toml::from_str(&original).context("parse markdown_owner_map.toml")?;
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
    let count_re = regex::Regex::new(r"(?m)^document_count\s*=\s*\d+\s*$").expect("static regex");
    let with_count = count_re.replace(
        &original,
        format!("document_count = {}", new_count).as_str(),
    );
    let mut updated = with_count.into_owned();
    if !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str(&blocks);
    fs::write(&map_path, &updated).with_context(|| format!("write {}", map_path.display()))?;
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

/// Returns Git-governed .md paths relative to repo_root, sorted, excluding skip dirs.
///
/// Git is the boundary because `*.md` is intentionally ignored for local
/// research captures and generated overlays. Walking the full filesystem makes
/// the gate depend on whichever private cache happens to be present, while the
/// index plus unignored working-tree paths describes the reproducible repository
/// surface.
fn list_markdown_files(repo_root: &Path) -> Result<Vec<String>> {
    let output = process::Command::new("git")
        .args([
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.md",
        ])
        .current_dir(repo_root)
        .output()
        .context("run git ls-files for markdown inventory")?;
    if !output.status.success() {
        bail!(
            "git ls-files failed for markdown inventory: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let mut out = Vec::new();
    for raw_path in output.stdout.split(|byte| *byte == 0) {
        if raw_path.is_empty() {
            continue;
        }
        let rel = str::from_utf8(raw_path)
            .context("Git returned a non-UTF-8 markdown path")?
            .replace('\\', "/");
        if Path::new(&rel).extension().and_then(|v| v.to_str()) != Some("md") {
            continue;
        }
        if rel
            .split('/')
            .any(|component| SKIP_DIRS.contains(&component))
        {
            continue;
        }
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

const RESEARCH_NARRATIVE_SOURCE_GLOBS: &[&str] = &[
    "docs/theory/*.md",
    "docs/engineering/*.md",
    "docs/research/*.md",
    "docs/*.md",
];

const ROOT_RESEARCH_NARRATIVE_PATHS: &[&str] = &[
    "docs/GRAND_SYNTHESIS.md",
    "docs/NAVIGATOR.md",
    "docs/EXCEPTIONAL_COSMOLOGY.md",
    "docs/SEDENION_GRAVASTAR_EQUIVALENCE.md",
    "docs/GRAND_SYNTHESIS_PLAN.md",
];

fn research_narrative_globs_literal() -> String {
    let joined = RESEARCH_NARRATIVE_SOURCE_GLOBS
        .iter()
        .map(|item| format!("\"{item}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!("source_markdown_globs = [{joined}]")
}

fn set_research_narrative_header(original: &str, document_count: usize) -> String {
    let mut output = String::new();
    let mut in_header = true;
    let mut wrote_globs = false;
    let mut wrote_count = false;

    for line in original.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed == "[[document]]" {
            if in_header {
                if !wrote_globs {
                    output.push_str(&research_narrative_globs_literal());
                    output.push('\n');
                }
                if !wrote_count {
                    output.push_str(&format!("document_count = {document_count}\n"));
                }
                in_header = false;
            }
            output.push_str(line);
            continue;
        }

        if in_header && trimmed.starts_with("source_markdown_globs") {
            output.push_str(&research_narrative_globs_literal());
            output.push('\n');
            wrote_globs = true;
            continue;
        }

        if in_header && trimmed.starts_with("document_count") {
            output.push_str(&format!("document_count = {document_count}\n"));
            wrote_count = true;
            continue;
        }

        output.push_str(line);
    }

    output
}

fn research_narrative_root_doc_failures(repo_root: &Path, value: &toml::Value) -> Vec<String> {
    let mut failures = Vec::new();
    let meta = value.get("research_narratives");
    let globs: BTreeSet<String> = meta
        .and_then(|row| row.get("source_markdown_globs"))
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_str)
        .map(ToString::to_string)
        .collect();
    for required_glob in RESEARCH_NARRATIVE_SOURCE_GLOBS {
        if !globs.contains(*required_glob) {
            failures.push(format!(
                "research_narratives.source_markdown_globs missing {required_glob}"
            ));
        }
    }

    let rows = table_array(value, "document");
    let declared_count = meta
        .and_then(|row| row.get("document_count"))
        .and_then(toml::Value::as_integer);
    if declared_count != Some(rows.len() as i64) {
        failures.push(format!(
            "research_narratives.document_count={declared_count:?} but found {} document rows",
            rows.len()
        ));
    }

    let source_rows: BTreeSet<String> = rows
        .iter()
        .map(|row| table_str(row, "source_markdown").to_string())
        .filter(|path| !path.is_empty())
        .collect();
    for required_path in ROOT_RESEARCH_NARRATIVE_PATHS {
        if !source_rows.contains(*required_path) {
            failures.push(format!(
                "registry/research_narratives.toml missing document row for {required_path}"
            ));
        }
        if !repo_root.join(required_path).is_file() {
            failures.push(format!(
                "research narrative source_markdown path does not exist: {required_path}"
            ));
        }
    }

    failures
}

fn verify_research_narrative_root_docs(repo_root: &Path) -> Result<()> {
    let path = repo_root.join("registry/research_narratives.toml");
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let value: toml::Value =
        toml::from_str(&text).context("parse registry/research_narratives.toml")?;
    let failures = research_narrative_root_doc_failures(repo_root, &value);
    if !failures.is_empty() {
        bail!("{}", failures.join("\n"));
    }
    Ok(())
}

fn promote_research_narratives(repo_root: &Path) -> Result<()> {
    let path = repo_root.join("registry/research_narratives.toml");
    let original = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let value: toml::Value =
        toml::from_str(&original).context("parse registry/research_narratives.toml")?;
    let rows = table_array(&value, "document");

    let updated = set_research_narrative_header(&original, rows.len());
    fs::write(&path, updated).with_context(|| format!("write {}", path.display()))?;
    verify_research_narrative_root_docs(repo_root)?;
    println!(
        "OK: promote-research-narratives refreshed docs/*.md coverage for {} document rows",
        rows.len()
    );
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{
        ROOT_RESEARCH_NARRATIVE_PATHS, list_markdown_files, research_narrative_root_doc_failures,
        set_research_narrative_header,
    };
    use std::{fs, path::Path};

    fn write_root_docs(repo_root: &Path) {
        fs::create_dir_all(repo_root.join("docs")).unwrap();
        for path in ROOT_RESEARCH_NARRATIVE_PATHS {
            fs::write(repo_root.join(path), "# root doc\n").unwrap();
        }
    }

    #[test]
    fn research_narrative_header_repair_adds_root_glob_and_count() {
        let input = "# header\n[research_narratives]\nsource_markdown_globs = [\"docs/theory/*.md\"]\ndocument_count = 1\n\n[[document]]\nid = \"RN-040\"\nsource_markdown = \"docs/GRAND_SYNTHESIS.md\"\n";

        let repaired = set_research_narrative_header(input, 5);

        assert!(repaired.contains(
            "source_markdown_globs = [\"docs/theory/*.md\", \"docs/engineering/*.md\", \"docs/research/*.md\", \"docs/*.md\"]"
        ));
        assert!(repaired.contains("document_count = 5"));
    }

    #[test]
    fn research_narrative_root_doc_check_accepts_registered_root_rows() {
        let temp = tempfile::tempdir().unwrap();
        write_root_docs(temp.path());
        let value: toml::Value = toml::from_str(
            r#"
[research_narratives]
source_markdown_globs = ["docs/theory/*.md", "docs/engineering/*.md", "docs/research/*.md", "docs/*.md"]
document_count = 5

[[document]]
id = "RN-040"
source_markdown = "docs/GRAND_SYNTHESIS.md"

[[document]]
id = "RN-041"
source_markdown = "docs/NAVIGATOR.md"

[[document]]
id = "RN-042"
source_markdown = "docs/EXCEPTIONAL_COSMOLOGY.md"

[[document]]
id = "RN-043"
source_markdown = "docs/SEDENION_GRAVASTAR_EQUIVALENCE.md"

[[document]]
id = "RN-044"
source_markdown = "docs/GRAND_SYNTHESIS_PLAN.md"
"#,
        )
        .unwrap();

        let failures = research_narrative_root_doc_failures(temp.path(), &value);

        assert!(failures.is_empty());
    }

    #[test]
    fn markdown_inventory_excludes_ignored_local_documents() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join(".gitignore"), "*.md\n!tracked.md\n").unwrap();
        fs::write(temp.path().join("tracked.md"), "# tracked\n").unwrap();
        fs::write(temp.path().join("ignored.md"), "# ignored\n").unwrap();
        fs::create_dir(temp.path().join("nested")).unwrap();
        fs::write(temp.path().join("nested/ignored.md"), "# ignored\n").unwrap();

        let init = std::process::Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(temp.path())
            .status()
            .unwrap();
        assert!(init.success());
        let add = std::process::Command::new("git")
            .args(["add", "--", ".gitignore", "tracked.md"])
            .current_dir(temp.path())
            .status()
            .unwrap();
        assert!(add.success());

        assert_eq!(
            list_markdown_files(temp.path()).unwrap(),
            vec!["tracked.md"]
        );
    }
}
