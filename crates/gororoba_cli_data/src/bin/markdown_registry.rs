use anyhow::{Context, Result, bail};
use chrono::Local;
use clap::{Parser, Subcommand};
use glob::Pattern;
use gororoba_cli_data::source_provenance;
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use toml::Value;
use walkdir::{DirEntry, WalkDir};

const DEFAULT_IGNORED_PREFIXES: &[&str] = &[
    ".cache/",
    ".pytest_cache/",
    "venv/",
    ".venv/",
    ".venv_ingest/",
    ".horusec/",
    ".claude/",
    ".gemini/",
    ".playwright-mcp/",
    ".mamba/",
    "target/",
    "logs/",
    "build/",
    "dist/",
    "temp/",
    "tmp/",
];

const DEFAULT_IGNORED_PARTS: &[&str] = &[
    ".cache",
    "cargo-home",
    ".pytest_cache",
    "venv",
    ".venv",
    "target",
    "logs",
    "build",
    "dist",
    "temp",
    "tmp",
];

const IN_SCOPE_PREFIXES: &[&str] = &["docs/", "reports/", "data/artifacts/"];
const GENERATED_MARKERS: &[&str] = &[
    "AUTO-GENERATED",
    "Source of truth:",
    "This file is generated from",
    "DO NOT EDIT",
];
const THIRD_PARTY_PATTERNS: &[&str] = &[
    ".pytest_cache/README.md",
    "*/site-packages/*/LICENSE.md",
    "*/site-packages/*/licenses/*.md",
    "data/external/intake/*",
];
const GENERATED_PATTERNS: &[&str] = &["build/docs/generated/*.md", "docs/generated/*.md"];
const MANUAL_EXCEPTIONS: &[&str] = &[];
const NON_DESTINATION_REGISTRIES: &[&str] = &[
    "registry/knowledge_migration_plan.toml",
    "registry/markdown_inventory.toml",
    "registry/markdown_corpus_registry.toml",
    "registry/toml_inventory.toml",
    "registry/control_plane_roadmap.toml",
    "registry/markdown_origin_audit.toml",
    "registry/markdown_owner_map.toml",
];
const ARCHIVAL_NON_DESTINATION_REGISTRIES: &[&str] = &["registry/wave4_roadmap.toml"];
const IMMUTABLE_AGENT_OVERLAYS: &[&str] = &["CLAUDE.md", "GEMINI.md"];
const KNOWLEDGE_FORCE_CAPTURE_EXACT: &[&str] = &[
    "docs/CLAIMS_EVIDENCE_MATRIX.md",
    "docs/generated/CLAIMS_REGISTRY_MIRROR.md",
    "docs/generated/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md",
    "docs/generated/ROADMAP_REGISTRY_MIRROR.md",
];
const KNOWLEDGE_FORCE_CAPTURE_PREFIXES: &[&str] = &["reports/", "docs/convos/"];
const KNOWLEDGE_FORCE_CAPTURE_SUFFIXES: &[&str] = &["_REGISTRY_MIRROR.md"];

#[derive(Parser, Debug)]
#[command(
    name = "markdown-registry",
    about = "Rust markdown control-plane builder/verifier"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    BuildKnowledgeSources(BuildKnowledgeSourcesArgs),
    BuildGovernance(BuildGovernanceArgs),
    MigrateCorpus(MigrateCorpusArgs),
    BuildInventory(BuildInventoryArgs),
    BuildCorpus(BuildCorpusArgs),
    BuildTomlInventory(BuildTomlInventoryArgs),
    BuildEmbedded(BuildEmbeddedArgs),
    BuildOriginAudit(BuildOriginAuditArgs),
    BuildOwnerMap(BuildOwnerMapArgs),
    BuildPayloads(BuildPayloadsArgs),
    NormalizeNarrativeOverlays(NormalizeNarrativeOverlaysArgs),
    NormalizeOperationalNarratives(NormalizeOperationalNarrativesArgs),
    PromoteDocsRootNarratives(PromoteDocsRootNarrativesArgs),
    PromoteResearchNarratives(PromoteResearchNarrativesArgs),
    VerifyCorpus(VerifyCorpusArgs),
    VerifyTomlInventory(VerifyTomlInventoryArgs),
    VerifyEmbedded(VerifyEmbeddedArgs),
    VerifyInventoryTomlFirst(VerifyInventoryArgs),
    VerifyOriginAudit(VerifyOriginArgs),
    VerifyOwnerMap(VerifyOwnerArgs),
}

#[derive(Parser, Debug)]
struct BuildKnowledgeSourcesArgs {
    #[arg(long, default_value = "registry/knowledge_sources.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct BuildGovernanceArgs {
    #[arg(long, default_value = "registry/knowledge_sources.toml")]
    knowledge_index: PathBuf,

    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/markdown_governance.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct MigrateCorpusArgs {
    #[arg(long, default_value = "registry/knowledge_sources.toml")]
    index: PathBuf,

    #[arg(long, default_value = "registry/knowledge/docs")]
    out_dir: PathBuf,

    #[arg(long, default_value = "registry/knowledge/documents.toml")]
    manifest: PathBuf,

    #[arg(long, default_value_t = false)]
    prune_stale: bool,
}

#[derive(Parser, Debug)]
struct BuildInventoryArgs {
    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct BuildCorpusArgs {
    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/markdown_corpus_registry.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct BuildTomlInventoryArgs {
    #[arg(long, default_value = "registry/toml_inventory.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct BuildEmbeddedArgs {
    #[arg(long, default_value = "registry/embedded_markdown_payloads.toml")]
    payload_out: PathBuf,

    #[arg(long, default_value = "registry/embedded_markdown_chunks.toml")]
    chunks_out: PathBuf,
}

#[derive(Parser, Debug)]
struct BuildOriginAuditArgs {
    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/markdown_origin_audit.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct BuildOwnerMapArgs {
    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/markdown_origin_audit.toml")]
    origin_audit: PathBuf,

    #[arg(long, default_value = "registry/markdown_governance.toml")]
    governance: PathBuf,

    #[arg(long, default_value = "registry/markdown_owner_map.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct BuildPayloadsArgs {
    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    inventory_path: PathBuf,

    #[arg(long, default_value = "registry/markdown_owner_map.toml")]
    owner_map_path: PathBuf,

    #[arg(long, default_value = "registry/markdown_payloads.toml")]
    payload_out: PathBuf,

    #[arg(long, default_value = "registry/markdown_payload_chunks.toml")]
    chunks_out: PathBuf,
}

#[derive(Parser, Debug)]
struct NormalizeNarrativeOverlaysArgs {
    #[arg(long, default_value_t = false)]
    bootstrap_from_markdown: bool,
}

#[derive(Parser, Debug)]
struct NormalizeOperationalNarrativesArgs {
    #[arg(long, default_value_t = false)]
    bootstrap_from_markdown: bool,
}

#[derive(Parser, Debug)]
struct PromoteDocsRootNarrativesArgs {
    #[arg(long, default_value = "registry/docs_root_narratives.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct PromoteResearchNarrativesArgs {
    #[arg(long, default_value = "registry/research_narratives.toml")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyInventoryArgs {
    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/markdown_governance.toml")]
    governance: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyCorpusArgs {
    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/markdown_corpus_registry.toml")]
    corpus: PathBuf,

    #[arg(long, default_value = "registry/markdown_governance.toml")]
    governance: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyTomlInventoryArgs {
    #[arg(long, default_value = "registry/toml_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    markdown_inventory: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyEmbeddedArgs {
    #[arg(long, default_value = "registry/embedded_markdown_payloads.toml")]
    payload_path: PathBuf,

    #[arg(long, default_value = "registry/embedded_markdown_chunks.toml")]
    chunks_path: PathBuf,

    #[arg(long, default_value = "registry/markdown_governance.toml")]
    governance: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyOriginArgs {
    #[arg(long, default_value = "registry/markdown_origin_audit.toml")]
    audit: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyOwnerArgs {
    #[arg(long, default_value = "registry/markdown_inventory.toml")]
    inventory: PathBuf,

    #[arg(long, default_value = "registry/markdown_owner_map.toml")]
    owner_map: PathBuf,

    #[arg(long, default_value = "registry/markdown_governance.toml")]
    governance: PathBuf,
}

#[derive(Clone, Debug)]
struct GovernancePolicy {
    safe_classifications: HashSet<String>,
    tracked_allowed_paths: HashSet<String>,
    owner_scope_prefixes: HashSet<String>,
    owner_scope_paths: HashSet<String>,
    skip_prefixes: Vec<String>,
    skip_path_parts: HashSet<String>,
}

#[derive(Clone, Debug)]
struct GovernanceDoc {
    path: String,
    mode: String,
    header_required: bool,
    source_toml_refs: Vec<String>,
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

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct InventoryRow {
    title: String,
    path: String,
    git_status: String,
    classification: String,
    toml_destination: String,
    generated_declared: bool,
    generated_pattern: bool,
    third_party: bool,
    generated: bool,
    manual_exception: bool,
    size_bytes: i64,
    line_count: i64,
    sha256: String,
    claim_ref_count: i64,
    insight_ref_count: i64,
    experiment_ref_count: i64,
    migration_action: String,
    migration_priority: String,
    rationale: String,
    archived: bool,
}

#[derive(Clone, Debug)]
struct OriginAuditRow {
    path: String,
    scope: String,
    classification: String,
    git_status: String,
    line_count: i64,
    toml_destination: String,
    destination_exists: bool,
    header_auto_generated: bool,
    header_source_of_truth: bool,
    source_of_truth_raw: String,
    source_of_truth_paths: Vec<String>,
    origin_process: String,
    origin_status: String,
    consolidation_action: String,
}

#[derive(Clone, Debug)]
struct OwnerRow {
    path: String,
    canonical_toml: String,
    requires_generated_header: bool,
}

#[derive(Clone, Debug)]
struct MarkdownUnit {
    kind: String,
    text_ascii: String,
    line_start: usize,
    line_end: usize,
    heading_level: usize,
}

#[derive(Clone, Debug)]
struct PayloadDocument {
    id: String,
    path: String,
    git_status: String,
    origin_class: String,
    generated: bool,
    third_party: bool,
    canonical_toml_owner: String,
    size_bytes: usize,
    line_count: usize,
    content_sha256: String,
    chunk_count: usize,
    heading_count: usize,
    paragraph_count: usize,
    list_item_count: usize,
    table_row_count: usize,
    code_block_count: usize,
    chunk_ids: Vec<String>,
}

#[derive(Clone, Debug)]
struct PayloadChunk {
    id: String,
    document_id: String,
    chunk_index: usize,
    kind: String,
    line_start: usize,
    line_end: usize,
    heading_level: usize,
    text_ascii: String,
    text_sha256: String,
}

#[derive(Clone, Debug)]
struct InventoryDoc {
    path: String,
    title: String,
    git_status: String,
    archived: bool,
    generated_declared: bool,
    generated_pattern: bool,
    generated: bool,
    manual_exception: bool,
    third_party: bool,
    classification: String,
    migration_action: String,
    migration_priority: String,
    toml_destination: String,
    rationale: String,
    size_bytes: usize,
    line_count: usize,
    sha256: String,
    claim_ref_count: usize,
    insight_ref_count: usize,
    experiment_ref_count: usize,
}

#[derive(Clone, Debug)]
struct EmbeddedCandidate {
    body: String,
    source_registry: String,
    source_document_id: String,
    source_title: String,
}

#[derive(Clone, Debug)]
struct EmbeddedDocument {
    id: String,
    path: String,
    scope: String,
    exists_on_disk: bool,
    source_registry: String,
    source_document_id: String,
    source_title: String,
    line_count: usize,
    size_bytes: usize,
    content_sha256: String,
    chunk_count: usize,
    heading_count: usize,
    paragraph_count: usize,
    list_item_count: usize,
    table_row_count: usize,
    code_block_count: usize,
    chunk_ids: Vec<String>,
}

#[derive(Clone, Debug)]
struct TomlInventoryRow {
    path: String,
    git_status: String,
    role: String,
    zone: String,
    parse_ok: bool,
    parse_error: String,
    line_count: usize,
    size_bytes: usize,
    sha256: String,
    table_count: usize,
    markdown_ref_count: usize,
    has_authoritative: bool,
}

#[derive(Clone, Debug)]
struct NarrativeDocRow {
    id: String,
    source_markdown: String,
    domain: String,
    slug: String,
    title: String,
    status_token: String,
    content_kind: String,
    verification_level: String,
    claim_refs: Vec<String>,
    url_refs: Vec<String>,
    path_refs: Vec<String>,
    line_count: usize,
    body_markdown: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = if cli.repo_root == Path::new(".") {
        source_provenance::default_repo_root()
    } else {
        cli.repo_root
    };
    match cli.command {
        Commands::BuildKnowledgeSources(args) => build_knowledge_sources(&repo_root, &args),
        Commands::BuildGovernance(args) => build_governance(&repo_root, &args),
        Commands::MigrateCorpus(args) => migrate_corpus(&repo_root, &args),
        Commands::BuildInventory(args) => build_inventory(&repo_root, &args),
        Commands::BuildCorpus(args) => build_corpus(&repo_root, &args),
        Commands::BuildTomlInventory(args) => build_toml_inventory(&repo_root, &args),
        Commands::BuildEmbedded(args) => build_embedded(&repo_root, &args),
        Commands::BuildOriginAudit(args) => build_origin_audit(&repo_root, &args),
        Commands::BuildOwnerMap(args) => build_owner_map(&repo_root, &args),
        Commands::BuildPayloads(args) => build_payloads(&repo_root, &args),
        Commands::NormalizeNarrativeOverlays(args) => {
            normalize_narrative_overlays(&repo_root, &args)
        }
        Commands::NormalizeOperationalNarratives(args) => {
            normalize_operational_narratives(&repo_root, &args)
        }
        Commands::PromoteDocsRootNarratives(args) => promote_docs_root_narratives(&repo_root, &args),
        Commands::PromoteResearchNarratives(args) => {
            promote_research_narratives(&repo_root, &args)
        }
        Commands::VerifyCorpus(args) => verify_corpus(&repo_root, &args),
        Commands::VerifyTomlInventory(args) => verify_toml_inventory(&repo_root, &args),
        Commands::VerifyEmbedded(args) => verify_embedded(&repo_root, &args),
        Commands::VerifyInventoryTomlFirst(args) => verify_inventory_toml_first(&repo_root, &args),
        Commands::VerifyOriginAudit(args) => verify_origin_audit(&repo_root, &args),
        Commands::VerifyOwnerMap(args) => verify_owner_map(&repo_root, &args),
    }
}

fn build_knowledge_sources(repo_root: &Path, args: &BuildKnowledgeSourcesArgs) -> Result<()> {
    let tracked = git_paths(repo_root, &["ls-files"], "*.md")?;
    let mut files = tracked
        .into_iter()
        .filter(|path| repo_root.join(path).is_file())
        .collect::<Vec<_>>();
    files.sort();

    let mut docs = Vec::new();
    let mut kind_counts = BTreeMap::<String, usize>::new();
    for (idx, rel_path) in files.iter().enumerate() {
        let full = repo_root.join(rel_path);
        let raw = fs::read(&full).with_context(|| format!("read {}", full.display()))?;
        let text = String::from_utf8_lossy(&raw).to_string();
        let title = first_title(
            &text,
            Path::new(rel_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(rel_path),
        );
        let (kind, authoring_mode, generated) = knowledge_kind_for_path(rel_path, &text);
        let row = KnowledgeSourceRow {
            doc_id: format!("DOC-{idx:04}", idx = idx + 1),
            path: rel_path.clone(),
            title,
            kind: kind.clone(),
            authoring_mode,
            generated,
            status: knowledge_status_for_path(rel_path),
            migration_priority: knowledge_migration_priority(&kind, rel_path),
            toml_backing: knowledge_toml_backing_for_path(rel_path),
            sha256: sha256_hex(text.as_bytes()),
            size_bytes: raw.len(),
            line_count: text.lines().count() + usize::from(!text.is_empty() && !text.ends_with('\n')),
            claim_ref_count: count_regex(r"\bC-\d{3}\b", &text)?,
            insight_ref_count: count_regex(r"\bI-\d{3}\b", &text)?,
            experiment_ref_count: count_regex(r"\bE-\d{3}\b", &text)?,
            link_count: extract_link_sample(&text).1,
            link_sample: extract_link_sample(&text).0,
        };
        *kind_counts.entry(row.kind.clone()).or_insert(0) += 1;
        docs.push(row);
    }

    let mut lines = vec![
        "# Knowledge source index for markdown assets.".to_string(),
        "# Auto-generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-knowledge-sources".to_string(),
        "# Regenerate with: cargo run -p gororoba_cli_data --bin markdown-registry -- build-knowledge-sources".to_string(),
        String::new(),
        "[knowledge_sources]".to_string(),
        "generated_at = \"deterministic\"".to_string(),
        format!("tracked_markdown_count = {}", docs.len()),
        format!("manual_source_count = {}", kind_counts.get("manual_source").copied().unwrap_or(0)),
        format!("markdown_mirror_count = {}", kind_counts.get("markdown_mirror").copied().unwrap_or(0)),
        format!("generated_markdown_count = {}", kind_counts.get("generated_markdown").copied().unwrap_or(0)),
        format!("artifact_report_count = {}", kind_counts.get("artifact_report").copied().unwrap_or(0)),
        format!("transcript_input_count = {}", kind_counts.get("transcript_input").copied().unwrap_or(0)),
        String::new(),
    ];
    for row in &docs {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {}", q(&row.doc_id)));
        lines.push(format!("path = {}", q(&row.path)));
        lines.push(format!("title = {}", q(&row.title)));
        lines.push(format!("kind = {}", q(&row.kind)));
        lines.push(format!("authoring_mode = {}", q(&row.authoring_mode)));
        lines.push(format!("generated = {}", bool_toml(row.generated)));
        lines.push(format!("status = {}", q(&row.status)));
        lines.push(format!("migration_priority = {}", q(&row.migration_priority)));
        if !row.toml_backing.is_empty() {
            lines.push(format!("toml_backing = {}", q(&row.toml_backing)));
        }
        lines.push(format!("sha256 = {}", q(&row.sha256)));
        lines.push(format!("size_bytes = {}", row.size_bytes));
        lines.push(format!("line_count = {}", row.line_count));
        lines.push(format!("claim_ref_count = {}", row.claim_ref_count));
        lines.push(format!("insight_ref_count = {}", row.insight_ref_count));
        lines.push(format!("experiment_ref_count = {}", row.experiment_ref_count));
        lines.push(format!("link_count = {}", row.link_count));
        lines.push(format!("link_sample = {}", q_list(&row.link_sample)));
        lines.push(String::new());
    }
    write_ascii(&repo_path(repo_root, &args.out), &lines.join("\n"))?;
    println!(
        "Wrote {} with {} markdown records.",
        args.out.display(),
        docs.len()
    );
    Ok(())
}

fn build_governance(repo_root: &Path, args: &BuildGovernanceArgs) -> Result<()> {
    let knowledge = load_toml(&repo_path(repo_root, &args.knowledge_index))?;
    let inventory = load_toml(&repo_path(repo_root, &args.inventory))?;
    let refs = iter_registry_markdown_refs(repo_root)?;
    let tracked_markdown = git_markdown_paths(repo_root, &["ls-files"])?;
    let inventory_by_path = load_inventory_rows(&inventory)
        .into_iter()
        .map(|row| (row.path.clone(), row))
        .collect::<HashMap<_, _>>();
    let knowledge_by_path = load_knowledge_source_rows(&knowledge)
        .into_iter()
        .map(|row| (row.path.clone(), row))
        .collect::<HashMap<_, _>>();
    let mut governed_paths = knowledge_by_path
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    governed_paths.extend(refs.keys().cloned());
    governed_paths.extend(tracked_markdown);

    let mut rows = Vec::<(String, String, String, bool, Vec<String>, String)>::new();
    for (idx, path) in governed_paths.into_iter().enumerate() {
        if !path.ends_with(".md") {
            continue;
        }
        let row = knowledge_by_path.get(&path);
        let inv_row = inventory_by_path.get(&path);
        let classification = inv_row
            .map(|item| item.classification.clone())
            .unwrap_or_default();
        let git_status = inv_row.map(|item| item.git_status.clone()).unwrap_or_default();
        if !git_status.is_empty() && git_status != "tracked" && classification != "generated_artifact"
        {
            continue;
        }
        if !classification.is_empty() && !knowledge_safe_classification(&classification) {
            continue;
        }
        let kind = row
            .map(|item| item.kind.clone())
            .filter(|item| !item.is_empty())
            .unwrap_or_else(|| {
                if classification.is_empty() {
                    "markdown".to_string()
                } else {
                    classification.clone()
                }
            });
        let mut source_refs = refs
            .get(&path)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        source_refs.sort();
        let toml_backing = row
            .map(|item| item.toml_backing.clone())
            .filter(|item| !item.is_empty())
            .or_else(|| {
                inv_row
                    .map(|item| item.toml_destination.clone())
                    .filter(|item| !item.is_empty())
            })
            .unwrap_or_default();
        if !toml_backing.is_empty() {
            source_refs.retain(|item| item != &toml_backing);
            source_refs.insert(0, toml_backing);
        }
        let (mode, header_required, notes) =
            governance_mode_for_path(&path, &kind, &classification, &source_refs);
        rows.push((
            format!("MDG-{idx:04}", idx = idx + 1),
            path,
            kind,
            header_required,
            source_refs,
            format!("{mode}\n{notes}"),
        ));
    }

    let mut mode_counts = BTreeMap::<String, usize>::new();
    for (_, _, _, _, _, mode_notes) in &rows {
        let mode = mode_notes
            .split('\n')
            .next()
            .unwrap_or_default()
            .to_string();
        *mode_counts.entry(mode).or_insert(0) += 1;
    }

    let mut lines = vec![
        "# Markdown lifecycle governance registry (TOML-first).".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-governance".to_string(),
        String::new(),
        "[markdown_governance]".to_string(),
        "generated_at = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("document_count = {}", rows.len()),
    ];
    for key in rows
        .iter()
        .map(|row| row.5.split('\n').next().unwrap_or_default().to_string())
        .collect::<BTreeSet<_>>()
    {
        lines.push(format!(
            "{}_count = {}",
            key,
            mode_counts.get(&key).copied().unwrap_or(0)
        ));
    }
    lines.push(String::new());
    lines.push("[policy]".to_string());
    lines.push(format!(
        "safe_classifications = {}",
        q_list(
            &[
                "toml_published_markdown".to_string(),
                "toml_destination_exists_manual_markdown".to_string(),
                "generated_artifact".to_string(),
                "third_party_markdown".to_string(),
            ]
        )
    ));
    lines.push(format!(
        "tracked_allowed_modes = {}",
        q_list(&["toml_generated_mirror".to_string(), "toml_manual_source".to_string()])
    ));
    lines.push(format!(
        "tracked_allowed_paths = {}",
        q_list(&[
            "docs/research/high_dimensional_algebra_unification_2026.md".to_string(),
            "proofs/EPISTEMIC_BOUNDARIES.md".to_string(),
        ])
    ));
    lines.push(format!(
        "embedded_markdown_prefixes = {}",
        q_list(
            &IN_SCOPE_PREFIXES
                .iter()
                .map(|value| (*value).to_string())
                .collect::<Vec<_>>()
        )
    ));
    lines.push(format!(
        "embedded_markdown_root_paths = {}",
        q_list(&embedded_markdown_root_paths())
    ));
    lines.push(format!(
        "owner_scope_prefixes = {}",
        q_list(
            &["docs/".to_string(), "reports/".to_string(), "data/artifacts/".to_string()]
        )
    ));
    lines.push(format!(
        "owner_scope_paths = {}",
        q_list(&owner_scope_paths())
    ));
    lines.push(format!(
        "generated_patterns = {}",
        q_list(
            &GENERATED_PATTERNS
                .iter()
                .map(|value| (*value).to_string())
                .collect::<Vec<_>>()
        )
    ));
    lines.push(format!(
        "skip_prefixes = {}",
        q_list(
            &DEFAULT_IGNORED_PREFIXES
                .iter()
                .map(|value| (*value).to_string())
                .collect::<Vec<_>>()
        )
    ));
    lines.push(format!(
        "skip_path_parts = {}",
        q_list(
            &DEFAULT_IGNORED_PARTS
                .iter()
                .map(|value| (*value).to_string())
                .collect::<Vec<_>>()
        )
    ));
    lines.push("disk_forbidden_modes = [\"deleted_mirror\"]".to_string());
    lines.push(String::new());
    for (id, path, kind, header_required, source_refs, mode_notes) in rows {
        let mut parts = mode_notes.splitn(2, '\n');
        let mode = parts.next().unwrap_or_default().to_string();
        let notes = parts.next().unwrap_or_default().to_string();
        lines.push("[[document]]".to_string());
        lines.push(format!("path = {}", q(&path)));
        lines.push(format!("id = {}", q(&id)));
        lines.push(format!("kind = {}", q(&kind)));
        lines.push(format!("mode = {}", q(&mode)));
        lines.push(format!("header_required = {}", bool_toml(header_required)));
        if !source_refs.is_empty() {
            lines.push(format!("source_toml_refs = {}", q_list(&source_refs)));
        }
        if !notes.is_empty() {
            lines.push(format!("notes = {}", q(&notes)));
        }
        lines.push(String::new());
    }
    write_ascii(&repo_path(repo_root, &args.out), &lines.join("\n"))?;
    println!("Wrote {} with {} entries.", args.out.display(), mode_counts.values().sum::<usize>());
    Ok(())
}

fn migrate_corpus(repo_root: &Path, args: &MigrateCorpusArgs) -> Result<()> {
    let docs = load_knowledge_source_rows(&load_toml(&repo_path(repo_root, &args.index))?);
    let out_dir = repo_path(repo_root, &args.out_dir);
    let manifest_path = repo_path(repo_root, &args.manifest);
    fs::create_dir_all(&out_dir).with_context(|| format!("mkdir {}", out_dir.display()))?;
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }

    let mut ingested = Vec::<(KnowledgeSourceRow, String)>::new();
    let mut skipped = Vec::<KnowledgeSourceRow>::new();
    let mut expected = BTreeSet::<PathBuf>::new();

    for doc in docs {
        if doc.generated && !knowledge_force_capture(&doc.path) {
            skipped.push(doc);
            continue;
        }
        let source_path = repo_root.join(&doc.path);
        if !source_path.exists() {
            continue;
        }
        let content = fs::read_to_string(&source_path)
            .with_context(|| format!("read {}", source_path.display()))?;
        let rel_out = format!("registry/knowledge/docs/{}.toml", doc.doc_id);
        let out_path = repo_root.join(&rel_out);
        expected.insert(out_path.clone());
        write_ascii(&out_path, &render_raw_capture_doc(&doc, &content))?;
        ingested.push((doc, rel_out));
    }

    let mut stale_count = 0usize;
    for existing in fs::read_dir(&out_dir)
        .with_context(|| format!("read_dir {}", out_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.file_name().and_then(|s| s.to_str()).unwrap_or_default().starts_with("DOC-"))
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("toml"))
    {
        if expected.contains(&existing) {
            continue;
        }
        stale_count += 1;
        if args.prune_stale {
            fs::remove_file(&existing).with_context(|| format!("remove {}", existing.display()))?;
        }
    }

    write_ascii(&manifest_path, &render_raw_capture_manifest(&ingested, &skipped))?;
    println!(
        "Raw-captured {} markdown docs into TOML; skipped {} generated docs; stale captures {}={}.",
        ingested.len(),
        skipped.len(),
        if args.prune_stale { "pruned" } else { "retained" },
        stale_count
    );
    Ok(())
}

fn normalize_narrative_overlays(
    repo_root: &Path,
    args: &NormalizeNarrativeOverlaysArgs,
) -> Result<()> {
    let insights_out = repo_root.join("registry/insights_narrative.toml");
    let experiments_out = repo_root.join("registry/experiments_narrative.toml");
    if !args.bootstrap_from_markdown {
        for path in [&insights_out, &experiments_out] {
            if !path.exists() {
                bail!(
                    "missing {}. Run with --bootstrap-from-markdown once to seed TOML overlays.",
                    path.display()
                );
            }
            let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
            assert_ascii(&raw, &path.display().to_string())?;
        }
        println!(
            "TOML-first mode: narrative overlay bootstrap skipped. Use --bootstrap-from-markdown to ingest markdown sources."
        );
        return Ok(());
    }

    let insights_src = repo_root.join("docs/INSIGHTS.md");
    let experiments_src = repo_root.join("docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md");
    if !insights_src.exists() {
        bail!("missing docs/INSIGHTS.md for bootstrap mode");
    }
    if !experiments_src.exists() {
        bail!("missing docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md for bootstrap mode");
    }
    let insight_heading_re = Regex::new(r"^##\s+(I-\d{3})\s*:\s*(.+?)\s*$")?;
    let experiment_heading_re = Regex::new(r"^##\s+(E-\d{3})\s*:\s*(.+?)\s*$")?;
    let (insights_preamble, insight_entries) =
        parse_overlay_sections(&insights_src, &insight_heading_re)?;
    let (experiments_preamble, experiment_entries) =
        parse_overlay_sections(&experiments_src, &experiment_heading_re)?;

    write_ascii(
        &insights_out,
        &render_narrative_overlay(
            "Insights narrative overlay registry (TOML-first).",
            "Captures long-form narrative previously maintained in docs/INSIGHTS.md.",
            "insights_narrative",
            "docs/INSIGHTS.md",
            &insights_preamble,
            &insight_entries,
        ),
    )?;
    write_ascii(
        &experiments_out,
        &render_narrative_overlay(
            "Experiments narrative overlay registry (TOML-first).",
            "Captures long-form narrative previously maintained in docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md.",
            "experiments_narrative",
            "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md",
            &experiments_preamble,
            &experiment_entries,
        ),
    )?;
    println!(
        "Normalized narrative overlays: registry/insights_narrative.toml, registry/experiments_narrative.toml."
    );
    Ok(())
}

fn normalize_operational_narratives(
    repo_root: &Path,
    args: &NormalizeOperationalNarrativesArgs,
) -> Result<()> {
    let outputs = [
        repo_root.join("registry/roadmap_narrative.toml"),
        repo_root.join("registry/todo_narrative.toml"),
        repo_root.join("registry/next_actions_narrative.toml"),
        repo_root.join("registry/requirements_narrative.toml"),
    ];
    if !args.bootstrap_from_markdown {
        for path in &outputs {
            if !path.exists() {
                bail!(
                    "missing {}. Run with --bootstrap-from-markdown once to seed TOML overlays.",
                    path.display()
                );
            }
            let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
            assert_ascii(&raw, &path.display().to_string())?;
        }
        println!(
            "TOML-first mode: operational overlay bootstrap skipped. Use --bootstrap-from-markdown to ingest markdown sources."
        );
        return Ok(());
    }

    let roadmap_body = read_overlay_markdown(&repo_root.join("docs/ROADMAP.md"))?;
    let todo_body = read_overlay_markdown(&repo_root.join("docs/TODO.md"))?;
    let next_actions_body = read_overlay_markdown(&repo_root.join("docs/NEXT_ACTIONS.md"))?;
    write_ascii(
        &outputs[0],
        &render_single_overlay(
            "Roadmap Narrative Overlay Registry (TOML-first).",
            "roadmap_narrative",
            "docs/ROADMAP.md",
            &roadmap_body,
        ),
    )?;
    write_ascii(
        &outputs[1],
        &render_single_overlay(
            "Todo Narrative Overlay Registry (TOML-first).",
            "todo_narrative",
            "docs/TODO.md",
            &todo_body,
        ),
    )?;
    write_ascii(
        &outputs[2],
        &render_single_overlay(
            "Next Actions Narrative Overlay Registry (TOML-first).",
            "next_actions_narrative",
            "docs/NEXT_ACTIONS.md",
            &next_actions_body,
        ),
    )?;
    let requirement_files = [
        repo_root.join("REQUIREMENTS.md"),
        repo_root.join("docs/REQUIREMENTS.md"),
    ];
    let mut requirement_docs = Vec::<(String, String, String)>::new();
    for file in requirement_files
        .into_iter()
        .chain(
            fs::read_dir(repo_root.join("docs/requirements"))
                .with_context(|| "read_dir docs/requirements".to_string())?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("md"))
                .collect::<Vec<_>>()
                .into_iter(),
        )
    {
        if !file.exists() {
            bail!(
                "missing {} for bootstrap mode",
                repo_rel(repo_root, &file)
            );
        }
        let body = read_overlay_markdown(&file)?;
        requirement_docs.push((
            repo_rel(repo_root, &file),
            title_from_markdown(&file, &body),
            body,
        ));
    }
    requirement_docs.sort_by(|a, b| a.0.cmp(&b.0));
    write_ascii(&outputs[3], &render_requirements_overlay(&requirement_docs))?;
    println!(
        "Normalized operational narrative overlays: registry/roadmap_narrative.toml, registry/todo_narrative.toml, registry/next_actions_narrative.toml, registry/requirements_narrative.toml."
    );
    Ok(())
}

fn build_inventory(repo_root: &Path, args: &BuildInventoryArgs) -> Result<()> {
    let governance = load_toml(&repo_path(
        repo_root,
        Path::new("registry/markdown_governance.toml"),
    ))?;
    let policy = load_governance_policy(&governance);
    let generated_allowlist = load_governance_docs(&governance)
        .into_iter()
        .filter(|row| row.mode == "toml_generated_mirror")
        .map(|row| row.path)
        .collect::<HashSet<_>>();
    let tracked = git_paths(repo_root, &["ls-files"], "*.md")?;
    let untracked = git_paths(repo_root, &["ls-files", "--others", "--exclude-standard"], "*.md")?;
    let ignored = git_paths(
        repo_root,
        &["ls-files", "--others", "--ignored", "--exclude-standard"],
        "*.md",
    )?;
    let mut all_paths = discover_markdown_files(repo_root, &policy)?
        .into_iter()
        .collect::<BTreeSet<_>>();
    all_paths.extend(tracked.iter().cloned());
    all_paths.extend(untracked.iter().cloned());
    all_paths.extend(ignored.iter().cloned());
    let registry_refs = iter_registry_markdown_refs(repo_root)?;

    let mut docs = Vec::new();
    for path in all_paths {
        if should_skip_path(&path, &policy) {
            continue;
        }
        let full = repo_root.join(&path);
        if !full.is_file() {
            continue;
        }
        let git_status = if tracked.contains(&path) {
            "tracked"
        } else if untracked.contains(&path) {
            "untracked"
        } else if ignored.contains(&path) {
            "ignored"
        } else {
            "filesystem_only"
        }
        .to_string();
        docs.push(build_inventory_doc(
            repo_root,
            &path,
            &git_status,
            &registry_refs,
            &generated_allowlist,
        )?);
    }
    docs.sort_by(|a, b| {
        priority_rank(&a.migration_priority)
            .cmp(&priority_rank(&b.migration_priority))
            .then_with(|| a.path.cmp(&b.path))
    });

    let tracked_count = docs.iter().filter(|doc| doc.git_status == "tracked").count();
    let untracked_count = docs.iter().filter(|doc| doc.git_status == "untracked").count();
    let ignored_count = docs.iter().filter(|doc| doc.git_status == "ignored").count();
    let filesystem_only_count = docs
        .iter()
        .filter(|doc| doc.git_status == "filesystem_only")
        .count();
    let generated_count = docs.iter().filter(|doc| doc.generated).count();
    let archived_count = docs.iter().filter(|doc| doc.archived).count();
    let third_party_count = docs.iter().filter(|doc| doc.third_party).count();
    let manual_exception_count = docs.iter().filter(|doc| doc.manual_exception).count();
    let unbacked_manual_count = docs
        .iter()
        .filter(|doc| doc.classification == "unbacked_manual_markdown")
        .count();
    let toml_backed_manual_count = docs
        .iter()
        .filter(|doc| doc.classification == "toml_destination_exists_manual_markdown")
        .count();

    let queue = docs
        .iter()
        .filter(|doc| {
            matches!(
                doc.migration_action.as_str(),
                "migrate_to_new_registry" | "port_body_to_toml_and_lock_mirror"
            )
        })
        .cloned()
        .collect::<Vec<_>>();

    let mut lines = vec![
        "# Full markdown inventory registry (TOML-first governance support).".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-inventory"
            .to_string(),
        String::new(),
        "[markdown_inventory]".to_string(),
        "generated_at = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("document_count = {}", docs.len()),
        format!("tracked_count = {}", tracked_count),
        format!("untracked_count = {}", untracked_count),
        format!("ignored_count = {}", ignored_count),
        format!("filesystem_only_count = {}", filesystem_only_count),
        format!("generated_count = {}", generated_count),
        format!("non_generated_count = {}", docs.len().saturating_sub(generated_count)),
        format!("archived_count = {}", archived_count),
        format!("third_party_count = {}", third_party_count),
        format!("manual_exception_count = {}", manual_exception_count),
        format!("unbacked_manual_count = {}", unbacked_manual_count),
        format!("toml_backed_manual_count = {}", toml_backed_manual_count),
        String::new(),
    ];

    for (idx, doc) in queue.iter().take(60).enumerate() {
        lines.push("[[migration_queue]]".to_string());
        lines.push(format!("rank = {}", idx + 1));
        lines.push(format!("path = {}", q(&doc.path)));
        lines.push(format!(
            "migration_priority = {}",
            q(&doc.migration_priority)
        ));
        lines.push(format!("migration_action = {}", q(&doc.migration_action)));
        if !doc.toml_destination.is_empty() {
            lines.push(format!("toml_destination = {}", q(&doc.toml_destination)));
        }
        lines.push(format!("line_count = {}", doc.line_count));
        lines.push(format!("rationale = {}", q(&doc.rationale)));
        lines.push(String::new());
    }

    for (idx, doc) in docs.iter().enumerate() {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {}", q(&format!("MDI-{:04}", idx + 1))));
        lines.push(format!("path = {}", q(&doc.path)));
        lines.push(format!("title = {}", q(&doc.title)));
        lines.push(format!("git_status = {}", q(&doc.git_status)));
        lines.push(format!("archived = {}", bool_toml(doc.archived)));
        lines.push(format!(
            "generated_declared = {}",
            bool_toml(doc.generated_declared)
        ));
        lines.push(format!(
            "generated_pattern = {}",
            bool_toml(doc.generated_pattern)
        ));
        lines.push(format!("generated = {}", bool_toml(doc.generated)));
        lines.push(format!(
            "manual_exception = {}",
            bool_toml(doc.manual_exception)
        ));
        lines.push(format!("third_party = {}", bool_toml(doc.third_party)));
        lines.push(format!("classification = {}", q(&doc.classification)));
        lines.push(format!("migration_action = {}", q(&doc.migration_action)));
        lines.push(format!(
            "migration_priority = {}",
            q(&doc.migration_priority)
        ));
        if !doc.toml_destination.is_empty() {
            lines.push(format!("toml_destination = {}", q(&doc.toml_destination)));
        }
        lines.push(format!("rationale = {}", q(&doc.rationale)));
        lines.push(format!("size_bytes = {}", doc.size_bytes));
        lines.push(format!("line_count = {}", doc.line_count));
        lines.push(format!("sha256 = {}", q(&doc.sha256)));
        lines.push(format!("claim_ref_count = {}", doc.claim_ref_count));
        lines.push(format!("insight_ref_count = {}", doc.insight_ref_count));
        lines.push(format!(
            "experiment_ref_count = {}",
            doc.experiment_ref_count
        ));
        lines.push(String::new());
    }

    write_ascii(&repo_path(repo_root, &args.out), &lines.join("\n"))?;
    println!(
        "Wrote {} with {} markdown records.",
        args.out.display(),
        docs.len()
    );
    Ok(())
}

fn build_corpus(repo_root: &Path, args: &BuildCorpusArgs) -> Result<()> {
    let inventory = load_toml(&repo_path(repo_root, &args.inventory))?;
    let governance = load_toml(&repo_path(
        repo_root,
        Path::new("registry/markdown_governance.toml"),
    ))?;
    let docs = load_inventory_rows(&inventory);
    let policy = load_governance_policy(&governance);
    let mut git_status_counts = BTreeMap::<String, usize>::new();
    let mut classification_counts = BTreeMap::<String, usize>::new();
    let mut lifecycle_counts = BTreeMap::<String, usize>::new();
    let mut tracked_violations = Vec::<String>::new();
    let mut classification_violations = Vec::<String>::new();
    let mut destination_missing = Vec::<String>::new();
    let mut risk_rows = Vec::<(i64, i64, String, InventoryRow, String, bool)>::new();

    for row in docs.iter() {
        let destination_exists =
            !row.toml_destination.is_empty() && repo_root.join(&row.toml_destination).is_file();
        let lifecycle = lifecycle_for_path(&row.path, row);
        let risk = risk_score(&row.path, row, destination_exists, &policy);
        *git_status_counts.entry(row.git_status.clone()).or_insert(0) += 1;
        *classification_counts
            .entry(row.classification.clone())
            .or_insert(0) += 1;
        *lifecycle_counts.entry(lifecycle.clone()).or_insert(0) += 1;

        if !row.path.is_empty()
            && row.git_status == "tracked"
            && !policy.tracked_allowed_paths.contains(&row.path)
        {
            tracked_violations.push(row.path.clone());
        }
        if !row.path.is_empty() && !policy.safe_classifications.contains(&row.classification) {
            classification_violations.push(row.path.clone());
        }
        if !row.path.is_empty()
            && row.classification == "toml_published_markdown"
            && !destination_exists
        {
            destination_missing.push(row.path.clone());
        }
        if risk > 0 {
            risk_rows.push((
                risk,
                row.line_count,
                row.path.clone(),
                row.clone(),
                lifecycle,
                destination_exists,
            ));
        }
    }
    risk_rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));

    let mut lines = vec![
        "# Wave 4 markdown corpus control-plane registry (TOML-first).".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-corpus"
            .to_string(),
        String::new(),
        "[markdown_corpus_registry]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("source_inventory = {}", q(args.inventory.to_string_lossy().as_ref())),
        format!("document_count = {}", docs.len()),
        format!("tracked_violation_count = {}", tracked_violations.len()),
        format!(
            "classification_violation_count = {}",
            classification_violations.len()
        ),
        format!("destination_missing_count = {}", destination_missing.len()),
        format!("risk_item_count = {}", risk_rows.len()),
        String::new(),
        "[policy]".to_string(),
        "toml_first_required = true".to_string(),
        "allow_tracked_markdown_entrypoints_only = true".to_string(),
        format!(
            "safe_classifications = {}",
            q_list(
                &policy
                    .safe_classifications
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
            )
        ),
        String::new(),
        "allowed_tracked_markdown = [".to_string(),
    ];
    for path in policy
        .tracked_allowed_paths
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
    {
        lines.push(format!("  {},", q(&path)));
    }
    lines.push("]".to_string());
    lines.push(String::new());
    lines.push("[git_status_counts]".to_string());
    for (key, value) in &git_status_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    lines.push("[classification_counts]".to_string());
    for (key, value) in &classification_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    lines.push("[lifecycle_counts]".to_string());
    for (key, value) in &lifecycle_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());

    for path in tracked_violations.iter().collect::<BTreeSet<_>>() {
        lines.push("[[policy_violation]]".to_string());
        lines.push("kind = \"tracked_markdown_outside_allowlist\"".to_string());
        lines.push(format!("path = {}", q(path)));
        lines.push(String::new());
    }
    for path in classification_violations.iter().collect::<BTreeSet<_>>() {
        lines.push("[[policy_violation]]".to_string());
        lines.push("kind = \"classification_outside_safe_set\"".to_string());
        lines.push(format!("path = {}", q(path)));
        lines.push(String::new());
    }
    for path in destination_missing.iter().collect::<BTreeSet<_>>() {
        lines.push("[[policy_violation]]".to_string());
        lines.push("kind = \"missing_toml_destination\"".to_string());
        lines.push(format!("path = {}", q(path)));
        lines.push(String::new());
    }
    for (idx, (risk, _, path, row, lifecycle, destination_exists)) in
        risk_rows.iter().take(120).enumerate()
    {
        lines.push("[[risk_queue]]".to_string());
        lines.push(format!("rank = {}", idx + 1));
        lines.push(format!("path = {}", q(path)));
        lines.push(format!("risk_score = {}", risk));
        lines.push(format!("classification = {}", q(&row.classification)));
        lines.push(format!("migration_action = {}", q(&row.migration_action)));
        lines.push(format!("migration_priority = {}", q(&row.migration_priority)));
        lines.push(format!("lifecycle = {}", q(lifecycle)));
        lines.push(format!("destination_exists = {}", bool_toml(*destination_exists)));
        if !row.toml_destination.is_empty() {
            lines.push(format!("toml_destination = {}", q(&row.toml_destination)));
        }
        lines.push(format!("line_count = {}", row.line_count));
        lines.push(format!("rationale = {}", q(&row.rationale)));
        lines.push(String::new());
    }
    let mut docs_sorted = docs.clone();
    docs_sorted.sort_by(|a, b| a.path.cmp(&b.path));
    for row in &docs_sorted {
        let lifecycle = lifecycle_for_path(&row.path, row);
        let destination_exists =
            !row.toml_destination.is_empty() && repo_root.join(&row.toml_destination).is_file();
        let tracked_allowed = !(row.git_status == "tracked"
            && !policy.tracked_allowed_paths.contains(&row.path));
        lines.push("[[document]]".to_string());
        lines.push(format!("path = {}", q(&row.path)));
        lines.push(format!("git_status = {}", q(&row.git_status)));
        lines.push(format!("classification = {}", q(&row.classification)));
        lines.push(format!("lifecycle = {}", q(&lifecycle)));
        lines.push(format!("generated = {}", bool_toml(row.generated)));
        lines.push(format!("third_party = {}", bool_toml(row.third_party)));
        lines.push(format!("tracked_allowed = {}", bool_toml(tracked_allowed)));
        lines.push(format!("destination_exists = {}", bool_toml(destination_exists)));
        lines.push(format!(
            "risk_score = {}",
            risk_score(&row.path, row, destination_exists, &policy)
        ));
        lines.push(format!("size_bytes = {}", row.size_bytes));
        lines.push(format!("line_count = {}", row.line_count));
        if !row.toml_destination.is_empty() {
            lines.push(format!("toml_destination = {}", q(&row.toml_destination)));
        }
        lines.push(String::new());
    }

    write_ascii(&repo_path(repo_root, &args.out), &lines.join("\n"))?;
    println!(
        "Wrote {} with {} markdown corpus records.",
        args.out.display(),
        docs.len()
    );
    Ok(())
}

fn build_toml_inventory(repo_root: &Path, args: &BuildTomlInventoryArgs) -> Result<()> {
    let tracked = git_paths(repo_root, &["ls-files"], "*.toml")?;
    let untracked = git_paths(repo_root, &["ls-files", "--others", "--exclude-standard"], "*.toml")?;
    let ignored = git_paths(
        repo_root,
        &["ls-files", "--others", "--ignored", "--exclude-standard"],
        "*.toml",
    )?;
    let mut all_paths = tracked
        .union(&untracked)
        .cloned()
        .collect::<BTreeSet<_>>();
    all_paths.extend(ignored.iter().cloned());
    let mut rows = Vec::new();
    let mut status_counts = BTreeMap::<String, usize>::new();
    let mut role_counts = BTreeMap::<String, usize>::new();
    let mut zone_counts = BTreeMap::<String, usize>::new();
    let mut parse_error_paths = Vec::<String>::new();

    for path in all_paths {
        if should_skip_toml_path(&path) {
            continue;
        }
        if !repo_root.join(&path).is_file() {
            continue;
        }
        let git_status = if tracked.contains(&path) {
            "tracked"
        } else if untracked.contains(&path) {
            "untracked"
        } else {
            "ignored"
        }
        .to_string();
        let row = scan_toml_document(repo_root, &path, &git_status)?;
        if !row.parse_ok {
            parse_error_paths.push(row.path.clone());
        }
        *status_counts.entry(row.git_status.clone()).or_insert(0) += 1;
        *role_counts.entry(row.role.clone()).or_insert(0) += 1;
        *zone_counts.entry(row.zone.clone()).or_insert(0) += 1;
        rows.push(row);
    }
    rows.sort_by(|a, b| a.path.cmp(&b.path));

    let mut lines = vec![
        "# TOML inventory registry (Wave 4 control plane).".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-toml-inventory".to_string(),
        String::new(),
        "[toml_inventory]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("document_count = {}", rows.len()),
        format!(
            "tracked_count = {}",
            status_counts.get("tracked").copied().unwrap_or(0)
        ),
        format!(
            "untracked_count = {}",
            status_counts.get("untracked").copied().unwrap_or(0)
        ),
        format!(
            "ignored_count = {}",
            status_counts.get("ignored").copied().unwrap_or(0)
        ),
        format!("parse_error_count = {}", parse_error_paths.len()),
        String::new(),
        "[git_status_counts]".to_string(),
    ];
    for (key, value) in &status_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    lines.push("[role_counts]".to_string());
    for (key, value) in &role_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    lines.push("[zone_counts]".to_string());
    for (key, value) in &zone_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    if !parse_error_paths.is_empty() {
        lines.push("parse_error_paths = [".to_string());
        for path in &parse_error_paths {
            lines.push(format!("  {},", q(path)));
        }
        lines.push("]".to_string());
        lines.push(String::new());
    }
    for (idx, row) in rows.iter().enumerate() {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {}", q(&format!("TOML-{:04}", idx + 1))));
        lines.push(format!("path = {}", q(&row.path)));
        lines.push(format!("git_status = {}", q(&row.git_status)));
        lines.push(format!("role = {}", q(&row.role)));
        lines.push(format!("zone = {}", q(&row.zone)));
        lines.push(format!("parse_ok = {}", bool_toml(row.parse_ok)));
        if !row.parse_error.is_empty() {
            lines.push(format!("parse_error = {}", q(&row.parse_error)));
        }
        lines.push(format!(
            "has_authoritative = {}",
            bool_toml(row.has_authoritative)
        ));
        lines.push(format!("table_count = {}", row.table_count));
        lines.push(format!("markdown_ref_count = {}", row.markdown_ref_count));
        lines.push(format!("size_bytes = {}", row.size_bytes));
        lines.push(format!("line_count = {}", row.line_count));
        lines.push(format!("sha256 = {}", q(&row.sha256)));
        lines.push(String::new());
    }
    write_ascii(&repo_path(repo_root, &args.out), &lines.join("\n"))?;
    println!(
        "Wrote {} with {} TOML records.",
        args.out.display(),
        rows.len()
    );
    Ok(())
}

fn build_embedded(repo_root: &Path, args: &BuildEmbeddedArgs) -> Result<()> {
    let governance = load_toml(&repo_path(
        repo_root,
        Path::new("registry/markdown_governance.toml"),
    ))?;
    let policy = load_governance_policy(&governance);
    let candidates = collect_embedded_candidates(repo_root, &policy)?;
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+?)\s*$")?;
    let list_re = Regex::new(r"^(?:[-*+]\s+|\d+\.\s+)(.+)$")?;
    let mut documents = Vec::<EmbeddedDocument>::new();
    let mut chunks = Vec::<PayloadChunk>::new();
    let mut kind_counts = BTreeMap::<String, usize>::new();

    for (doc_index, (path, candidate)) in candidates.iter().enumerate() {
        let doc_id = format!("EMB-{:05}", doc_index + 1);
        let body = ascii_clean(&candidate.body);
        let body_bytes = body.as_bytes();
        let units = parse_markdown_units(&body, &heading_re, &list_re);
        let mut chunk_ids = Vec::new();
        let mut heading_count = 0usize;
        let mut paragraph_count = 0usize;
        let mut list_item_count = 0usize;
        let mut table_row_count = 0usize;
        let mut code_block_count = 0usize;
        for (part_index, unit) in units.iter().enumerate() {
            let chunk_id = format!("{doc_id}-U{:04}", part_index + 1);
            chunk_ids.push(chunk_id.clone());
            chunks.push(PayloadChunk {
                id: chunk_id,
                document_id: doc_id.clone(),
                chunk_index: part_index + 1,
                kind: unit.kind.clone(),
                line_start: unit.line_start,
                line_end: unit.line_end,
                heading_level: unit.heading_level,
                text_ascii: unit.text_ascii.clone(),
                text_sha256: sha256_hex(unit.text_ascii.as_bytes()),
            });
            match unit.kind.as_str() {
                "heading" => heading_count += 1,
                "paragraph" => paragraph_count += 1,
                "list_item" => list_item_count += 1,
                "table_row" => table_row_count += 1,
                "code_block" => code_block_count += 1,
                _ => {}
            }
            *kind_counts.entry(unit.kind.clone()).or_insert(0) += 1;
        }
        let scope = scope_for_path(path);
        documents.push(EmbeddedDocument {
            id: doc_id,
            path: path.clone(),
            scope,
            exists_on_disk: repo_root.join(path).exists(),
            source_registry: candidate.source_registry.clone(),
            source_document_id: candidate.source_document_id.clone(),
            source_title: candidate.source_title.clone(),
            line_count: body.lines().count() + usize::from(!body.is_empty() && !body.ends_with('\n')),
            size_bytes: body_bytes.len(),
            content_sha256: sha256_hex(body_bytes),
            chunk_count: chunk_ids.len(),
            heading_count,
            paragraph_count,
            list_item_count,
            table_row_count,
            code_block_count,
            chunk_ids,
        });
    }
    chunks.sort_by(|a, b| a.document_id.cmp(&b.document_id).then_with(|| a.chunk_index.cmp(&b.chunk_index)));

    let mut payload_lines = vec![
        "# Embedded markdown structured payload registry (pure TOML units).".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-embedded".to_string(),
        String::new(),
        "[embedded_markdown_payloads]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "representation = \"structured_toml_units\"".to_string(),
        format!("document_count = {}", documents.len()),
        format!("heading_count = {}", kind_counts.get("heading").copied().unwrap_or(0)),
        format!("paragraph_count = {}", kind_counts.get("paragraph").copied().unwrap_or(0)),
        format!("list_item_count = {}", kind_counts.get("list_item").copied().unwrap_or(0)),
        format!("table_row_count = {}", kind_counts.get("table_row").copied().unwrap_or(0)),
        format!("code_block_count = {}", kind_counts.get("code_block").copied().unwrap_or(0)),
        String::new(),
    ];
    for row in &documents {
        payload_lines.push("[[document]]".to_string());
        payload_lines.push(format!("id = {}", q(&row.id)));
        payload_lines.push(format!("path = {}", q(&row.path)));
        payload_lines.push(format!("scope = {}", q(&row.scope)));
        payload_lines.push(format!("exists_on_disk = {}", bool_toml(row.exists_on_disk)));
        payload_lines.push(format!("source_registry = {}", q(&row.source_registry)));
        payload_lines.push(format!("source_document_id = {}", q(&row.source_document_id)));
        payload_lines.push(format!("source_title = {}", q(&row.source_title)));
        payload_lines.push(format!("line_count = {}", row.line_count));
        payload_lines.push(format!("size_bytes = {}", row.size_bytes));
        payload_lines.push(format!("content_sha256 = {}", q(&row.content_sha256)));
        payload_lines.push("content_encoding = \"structured_toml_units\"".to_string());
        payload_lines.push(format!("chunk_count = {}", row.chunk_count));
        payload_lines.push(format!("heading_count = {}", row.heading_count));
        payload_lines.push(format!("paragraph_count = {}", row.paragraph_count));
        payload_lines.push(format!("list_item_count = {}", row.list_item_count));
        payload_lines.push(format!("table_row_count = {}", row.table_row_count));
        payload_lines.push(format!("code_block_count = {}", row.code_block_count));
        payload_lines.push(format!("chunk_ids = {}", q_list(&row.chunk_ids)));
        payload_lines.push(String::new());
    }

    let mut chunk_lines = vec![
        "# Embedded markdown structured chunks (pure TOML units).".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-embedded".to_string(),
        String::new(),
        "[embedded_markdown_chunks]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "representation = \"structured_toml_units\"".to_string(),
        format!("chunk_count = {}", chunks.len()),
        format!("document_count = {}", documents.len()),
        String::new(),
    ];
    for row in &chunks {
        chunk_lines.push("[[chunk]]".to_string());
        chunk_lines.push(format!("id = {}", q(&row.id)));
        chunk_lines.push(format!("document_id = {}", q(&row.document_id)));
        chunk_lines.push(format!("chunk_index = {}", row.chunk_index));
        chunk_lines.push(format!("kind = {}", q(&row.kind)));
        chunk_lines.push(format!("line_start = {}", row.line_start));
        chunk_lines.push(format!("line_end = {}", row.line_end));
        chunk_lines.push(format!("heading_level = {}", row.heading_level));
        chunk_lines.push(format!("text_ascii = {}", q(&row.text_ascii)));
        chunk_lines.push(format!("text_sha256 = {}", q(&row.text_sha256)));
        chunk_lines.push(String::new());
    }

    write_ascii(&repo_path(repo_root, &args.payload_out), &payload_lines.join("\n"))?;
    write_ascii(&repo_path(repo_root, &args.chunks_out), &chunk_lines.join("\n"))?;
    println!(
        "Wrote embedded markdown structured registries: documents={} chunks={}",
        documents.len(),
        chunks.len()
    );
    Ok(())
}

fn build_origin_audit(repo_root: &Path, args: &BuildOriginAuditArgs) -> Result<()> {
    let inventory = load_toml(&repo_path(repo_root, &args.inventory))?;
    let docs = load_inventory_rows(&inventory);
    let source_re = Regex::new(r"Source of truth:\s*(.+?)\s*-->")?;
    let registry_path_re = Regex::new(r"(?:[A-Za-z0-9_./-]+/)?[A-Za-z0-9_.-]+\.toml")?;
    let mut origin_status_counts = BTreeMap::<String, usize>::new();
    let mut scope_counts = BTreeMap::<String, usize>::new();
    let mut queue = Vec::<OriginAuditRow>::new();
    let mut rows = Vec::<OriginAuditRow>::new();

    for row in docs.into_iter().filter(|row| {
        IN_SCOPE_PREFIXES
            .iter()
            .any(|prefix| row.path.starts_with(prefix))
    }) {
        let full = repo_root.join(&row.path);
        let text = fs::read_to_string(&full)
            .with_context(|| format!("read markdown {}", full.display()))?;
        let head = text.lines().take(80).collect::<Vec<_>>().join("\n");
        let destination_exists =
            !row.toml_destination.is_empty() && repo_root.join(&row.toml_destination).is_file();
        let has_auto = head.contains("AUTO-GENERATED");
        let has_source = head.contains("Source of truth:");
        let source_raw = source_re
            .captures(&head)
            .and_then(|caps| caps.get(1).map(|m| m.as_str().trim().to_string()))
            .unwrap_or_default();
        let mut source_paths = registry_path_re
            .find_iter(&source_raw)
            .map(|m| m.as_str().to_string())
            .collect::<Vec<_>>();
        source_paths.sort();
        source_paths.dedup();

        let generated_class = row.classification == "toml_published_markdown";
        let (origin_status, action) = if !generated_class {
            (
                "non_generated_needs_consolidation",
                "migrate_to_toml_registry_and_regenerate",
            )
        } else if !destination_exists {
            (
                "missing_destination_registry",
                "repair_toml_destination_mapping",
            )
        } else if !(has_auto && has_source) {
            ("missing_origin_headers", "regenerate_from_registry")
        } else if source_raw.is_empty() {
            (
                "missing_source_of_truth_value",
                "regenerate_with_source_header",
            )
        } else {
            ("generated_from_repo_process", "none")
        };

        let scope = scope_for_path(&row.path);
        *origin_status_counts
            .entry(origin_status.to_string())
            .or_insert(0) += 1;
        *scope_counts.entry(scope.clone()).or_insert(0) += 1;

        let entry = OriginAuditRow {
            path: row.path.clone(),
            scope,
            classification: row.classification.clone(),
            git_status: row.git_status.clone(),
            line_count: row.line_count,
            toml_destination: row.toml_destination.clone(),
            destination_exists,
            header_auto_generated: has_auto,
            header_source_of_truth: has_source,
            source_of_truth_raw: source_raw,
            source_of_truth_paths: source_paths,
            origin_process: origin_process(&row.path),
            origin_status: origin_status.to_string(),
            consolidation_action: action.to_string(),
        };
        if entry.consolidation_action != "none" {
            queue.push(entry.clone());
        }
        rows.push(entry);
    }
    rows.sort_by(|a, b| a.path.cmp(&b.path));
    queue.sort_by(|a, b| {
        b.line_count
            .cmp(&a.line_count)
            .then_with(|| a.path.cmp(&b.path))
    });

    let mut lines = vec![
        "# Markdown origin audit registry (docs/reports/data-artifacts).".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-origin-audit"
            .to_string(),
        String::new(),
        "[markdown_origin_audit]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("source_inventory = {}", q(args.inventory.to_string_lossy().as_ref())),
        format!("document_count = {}", rows.len()),
        format!(
            "generated_verified_count = {}",
            origin_status_counts
                .get("generated_from_repo_process")
                .copied()
                .unwrap_or(0)
        ),
        format!("needs_consolidation_count = {}", queue.len()),
        String::new(),
        "[scope_counts]".to_string(),
    ];
    for (key, value) in &scope_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    lines.push("[origin_status_counts]".to_string());
    for (key, value) in &origin_status_counts {
        lines.push(format!("{key} = {value}"));
    }
    lines.push(String::new());
    for (idx, item) in queue.iter().enumerate() {
        lines.push("[[consolidation_queue]]".to_string());
        lines.push(format!("rank = {}", idx + 1));
        lines.push(format!("path = {}", q(&item.path)));
        lines.push(format!("origin_status = {}", q(&item.origin_status)));
        lines.push(format!("classification = {}", q(&item.classification)));
        lines.push(format!(
            "consolidation_action = {}",
            q(&item.consolidation_action)
        ));
        if !item.toml_destination.is_empty() {
            lines.push(format!("toml_destination = {}", q(&item.toml_destination)));
        }
        lines.push(format!("line_count = {}", item.line_count));
        lines.push(String::new());
    }
    for item in &rows {
        lines.push("[[document]]".to_string());
        lines.push(format!("path = {}", q(&item.path)));
        lines.push(format!("scope = {}", q(&item.scope)));
        lines.push(format!("classification = {}", q(&item.classification)));
        lines.push(format!("git_status = {}", q(&item.git_status)));
        lines.push(format!("origin_status = {}", q(&item.origin_status)));
        lines.push(format!("origin_process = {}", q(&item.origin_process)));
        lines.push(format!(
            "destination_exists = {}",
            if item.destination_exists {
                "true"
            } else {
                "false"
            }
        ));
        lines.push(format!(
            "header_auto_generated = {}",
            if item.header_auto_generated {
                "true"
            } else {
                "false"
            }
        ));
        lines.push(format!(
            "header_source_of_truth = {}",
            if item.header_source_of_truth {
                "true"
            } else {
                "false"
            }
        ));
        if !item.toml_destination.is_empty() {
            lines.push(format!("toml_destination = {}", q(&item.toml_destination)));
        }
        if !item.source_of_truth_raw.is_empty() {
            lines.push(format!(
                "source_of_truth_raw = {}",
                q(&item.source_of_truth_raw)
            ));
        }
        lines.push(format!(
            "source_of_truth_paths = {}",
            q_list(&item.source_of_truth_paths)
        ));
        lines.push(format!(
            "consolidation_action = {}",
            q(&item.consolidation_action)
        ));
        lines.push(format!("line_count = {}", item.line_count));
        lines.push(String::new());
    }
    write_ascii(&repo_path(repo_root, &args.out), &lines.join("\n"))?;
    println!(
        "Wrote {} with {} markdown origin records.",
        args.out.display(),
        rows.len()
    );
    Ok(())
}

fn build_owner_map(repo_root: &Path, args: &BuildOwnerMapArgs) -> Result<()> {
    let inventory = load_toml(&repo_path(repo_root, &args.inventory))?;
    let origin = load_toml(&repo_path(repo_root, &args.origin_audit))?;
    let governance = load_toml(&repo_path(repo_root, &args.governance))?;
    let docs = load_inventory_rows(&inventory);
    let origin_by_path = load_origin_rows(&origin)
        .into_iter()
        .map(|row| (row.path.clone(), row))
        .collect::<HashMap<_, _>>();
    let governance_by_path = load_governance_docs(&governance)
        .into_iter()
        .filter(|row| {
            matches!(
                row.mode.as_str(),
                "toml_generated_mirror"
                    | "toml_manual_source"
                    | "immutable_transcript"
                    | "manual_narrative"
            )
        })
        .map(|row| (row.path.clone(), row))
        .collect::<HashMap<_, _>>();
    let policy = load_governance_policy(&governance);

    let mut scoped_docs = docs
        .into_iter()
        .filter(|row| {
            policy.owner_scope_paths.contains(&row.path)
                || policy
                    .owner_scope_prefixes
                    .iter()
                    .any(|prefix| row.path.starts_with(prefix))
        })
        .collect::<Vec<_>>();
    scoped_docs.sort_by(|a, b| a.path.cmp(&b.path));

    let mut lines = vec![
        "# Explicit markdown owner map for in-scope markdown files.".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-owner-map"
            .to_string(),
        String::new(),
        "[markdown_owner_map]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "scope = \"docs reports data_artifacts\"".to_string(),
        format!("document_count = {}", scoped_docs.len()),
        String::new(),
    ];

    for (idx, row) in scoped_docs.iter().enumerate() {
        let mut destination = row.toml_destination.clone();
        if let Some(origin_row) = origin_by_path.get(&row.path) {
            if let Some(first) = origin_row.source_of_truth_paths.first() {
                destination = first.clone();
            }
        }
        if destination.is_empty() {
            if let Some(gov) = governance_by_path.get(&row.path) {
                if let Some(first) = gov.source_toml_refs.first() {
                    destination = first.clone();
                }
            }
        }
        let gov_row = governance_by_path.get(&row.path);
        let requires_generated_header = gov_row.map(|row| row.header_required).unwrap_or(false);
        lines.push("[[owner]]".to_string());
        lines.push(format!("id = {}", q(&format!("MOWN-{:04}", idx + 1))));
        lines.push(format!("path = {}", q(&row.path)));
        lines.push(format!("scope = {}", q(&scope_for_path(&row.path))));
        lines.push(format!("canonical_toml = {}", q(&destination)));
        lines.push(format!(
            "owner_group = {}",
            q(&owner_group(&row.path, &destination))
        ));
        lines.push(format!(
            "requires_generated_header = {}",
            if requires_generated_header {
                "true"
            } else {
                "false"
            }
        ));
        lines.push(format!(
            "conversion_hint = {}",
            q(&conversion_hint(&row.path, &destination))
        ));
        lines.push(String::new());
    }

    write_ascii(&repo_path(repo_root, &args.out), &lines.join("\n"))?;
    println!(
        "Wrote {} with {} owner mappings.",
        args.out.display(),
        scoped_docs.len()
    );
    Ok(())
}

fn build_payloads(repo_root: &Path, args: &BuildPayloadsArgs) -> Result<()> {
    let inventory = load_toml(&repo_path(repo_root, &args.inventory_path))?;
    let owner_map = load_toml(&repo_path(repo_root, &args.owner_map_path))?;
    let governance = load_toml(&repo_path(
        repo_root,
        Path::new("registry/markdown_governance.toml"),
    ))?;
    let policy = load_governance_policy(&governance);
    let inventory_by_path = load_inventory_rows(&inventory)
        .into_iter()
        .map(|row| (row.path.clone(), row))
        .collect::<HashMap<_, _>>();
    let owner_by_path = load_owner_rows(&owner_map)
        .into_iter()
        .map(|row| (row.path.clone(), row))
        .collect::<HashMap<_, _>>();

    let tracked = git_markdown_paths(repo_root, &["ls-files"])?;
    let untracked = git_markdown_paths(repo_root, &["ls-files", "--others", "--exclude-standard"])?;
    let ignored = git_markdown_paths(
        repo_root,
        &["ls-files", "--others", "--ignored", "--exclude-standard"],
    )?;

    let markdown_paths = discover_markdown_files(repo_root, &policy)?;
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+?)\s*$")?;
    let list_re = Regex::new(r"^(?:[-*+]\s+|\d+\.\s+)(.+)$")?;
    let mut documents = Vec::<PayloadDocument>::new();
    let mut chunks = Vec::<PayloadChunk>::new();
    let mut status_counts = BTreeMap::<String, usize>::new();
    let mut origin_counts = BTreeMap::<String, usize>::new();
    let mut kind_counts = BTreeMap::<String, usize>::new();

    for (idx, rel_path) in markdown_paths.iter().enumerate() {
        let abs_path = repo_root.join(rel_path);
        let raw = fs::read(&abs_path).with_context(|| format!("read {}", abs_path.display()))?;
        let decoded = String::from_utf8_lossy(&raw).to_string();
        let sha256 = sha256_hex(&raw);
        let size_bytes = raw.len();
        let line_count =
            decoded.lines().count() + usize::from(!decoded.is_empty() && !decoded.ends_with('\n'));

        let inv_row = inventory_by_path.get(rel_path);
        let owner_row = owner_by_path.get(rel_path);
        let generated = inv_row.map(|row| row.generated).unwrap_or(false);
        let third_party = inv_row.map(|row| row.third_party).unwrap_or(false);
        let origin_class = origin_class(rel_path, generated, third_party);
        let git_status = if tracked.contains(rel_path) {
            "tracked"
        } else if untracked.contains(rel_path) {
            "untracked"
        } else if ignored.contains(rel_path) {
            "ignored"
        } else {
            "filesystem_only"
        }
        .to_string();
        let units = parse_markdown_units(&decoded, &heading_re, &list_re);
        let doc_id = format!("MPY-{:05}", idx + 1);
        let mut chunk_ids = Vec::new();
        let mut heading_count = 0usize;
        let mut paragraph_count = 0usize;
        let mut list_item_count = 0usize;
        let mut table_row_count = 0usize;
        let mut code_block_count = 0usize;
        for (part_idx, unit) in units.iter().enumerate() {
            let chunk_id = format!("{doc_id}-C{:04}", part_idx + 1);
            let text_sha256 = sha256_hex(unit.text_ascii.as_bytes());
            chunk_ids.push(chunk_id.clone());
            chunks.push(PayloadChunk {
                id: chunk_id,
                document_id: doc_id.clone(),
                chunk_index: part_idx + 1,
                kind: unit.kind.clone(),
                line_start: unit.line_start,
                line_end: unit.line_end,
                heading_level: unit.heading_level,
                text_ascii: unit.text_ascii.clone(),
                text_sha256,
            });
            match unit.kind.as_str() {
                "heading" => heading_count += 1,
                "paragraph" => paragraph_count += 1,
                "list_item" => list_item_count += 1,
                "table_row" => table_row_count += 1,
                "code_block" => code_block_count += 1,
                _ => {}
            }
            *kind_counts.entry(unit.kind.clone()).or_insert(0) += 1;
        }
        *status_counts.entry(git_status.clone()).or_insert(0) += 1;
        *origin_counts.entry(origin_class.clone()).or_insert(0) += 1;
        documents.push(PayloadDocument {
            id: doc_id,
            path: rel_path.clone(),
            git_status,
            origin_class,
            generated,
            third_party,
            canonical_toml_owner: owner_row
                .map(|row| row.canonical_toml.clone())
                .unwrap_or_default(),
            size_bytes,
            line_count,
            content_sha256: sha256,
            chunk_count: chunk_ids.len(),
            heading_count,
            paragraph_count,
            list_item_count,
            table_row_count,
            code_block_count,
            chunk_ids,
        });
    }

    let mut payload_lines =
        vec![
        "# Structured markdown payload registry (pure TOML textual units).".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-payloads"
            .to_string(),
        String::new(),
        "[markdown_payloads]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "representation = \"structured_toml_units\"".to_string(),
        format!("document_count = {}", documents.len()),
        format!(
            "tracked_count = {}",
            status_counts.get("tracked").copied().unwrap_or(0)
        ),
        format!(
            "untracked_count = {}",
            status_counts.get("untracked").copied().unwrap_or(0)
        ),
        format!(
            "ignored_count = {}",
            status_counts.get("ignored").copied().unwrap_or(0)
        ),
        format!(
            "filesystem_only_count = {}",
            status_counts.get("filesystem_only").copied().unwrap_or(0)
        ),
        format!(
            "project_manual_count = {}",
            origin_counts.get("project_manual").copied().unwrap_or(0)
        ),
        format!(
            "project_generated_count = {}",
            origin_counts.get("project_generated").copied().unwrap_or(0)
        ),
        format!(
            "third_party_cache_count = {}",
            origin_counts.get("third_party_cache").copied().unwrap_or(0)
        ),
        format!("heading_count = {}", kind_counts.get("heading").copied().unwrap_or(0)),
        format!(
            "paragraph_count = {}",
            kind_counts.get("paragraph").copied().unwrap_or(0)
        ),
        format!(
            "list_item_count = {}",
            kind_counts.get("list_item").copied().unwrap_or(0)
        ),
        format!(
            "table_row_count = {}",
            kind_counts.get("table_row").copied().unwrap_or(0)
        ),
        format!(
            "code_block_count = {}",
            kind_counts.get("code_block").copied().unwrap_or(0)
        ),
        String::new(),
    ];
    for row in &documents {
        payload_lines.push("[[document]]".to_string());
        payload_lines.push(format!("id = {}", q(&row.id)));
        payload_lines.push(format!("path = {}", q(&row.path)));
        payload_lines.push(format!("git_status = {}", q(&row.git_status)));
        payload_lines.push(format!("origin_class = {}", q(&row.origin_class)));
        payload_lines.push(format!(
            "generated = {}",
            if row.generated { "true" } else { "false" }
        ));
        payload_lines.push(format!(
            "third_party = {}",
            if row.third_party { "true" } else { "false" }
        ));
        payload_lines.push(format!(
            "canonical_toml_owner = {}",
            q(&row.canonical_toml_owner)
        ));
        payload_lines.push(format!("size_bytes = {}", row.size_bytes));
        payload_lines.push(format!("line_count = {}", row.line_count));
        payload_lines.push(format!("content_sha256 = {}", q(&row.content_sha256)));
        payload_lines.push("content_encoding = \"structured_toml_units\"".to_string());
        payload_lines.push(format!("chunk_count = {}", row.chunk_count));
        payload_lines.push(format!("heading_count = {}", row.heading_count));
        payload_lines.push(format!("paragraph_count = {}", row.paragraph_count));
        payload_lines.push(format!("list_item_count = {}", row.list_item_count));
        payload_lines.push(format!("table_row_count = {}", row.table_row_count));
        payload_lines.push(format!("code_block_count = {}", row.code_block_count));
        payload_lines.push(format!("chunk_ids = {}", q_list(&row.chunk_ids)));
        payload_lines.push(String::new());
    }

    let mut chunk_lines = vec![
        "# Structured markdown units (pure TOML textual chunks).".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- build-payloads"
            .to_string(),
        String::new(),
        "[markdown_payload_chunks]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "representation = \"structured_toml_units\"".to_string(),
        format!("chunk_count = {}", chunks.len()),
        format!("document_count = {}", documents.len()),
        String::new(),
    ];
    chunks.sort_by(|a, b| {
        a.document_id
            .cmp(&b.document_id)
            .then_with(|| a.chunk_index.cmp(&b.chunk_index))
    });
    for row in &chunks {
        chunk_lines.push("[[chunk]]".to_string());
        chunk_lines.push(format!("id = {}", q(&row.id)));
        chunk_lines.push(format!("document_id = {}", q(&row.document_id)));
        chunk_lines.push(format!("chunk_index = {}", row.chunk_index));
        chunk_lines.push(format!("kind = {}", q(&row.kind)));
        chunk_lines.push(format!("line_start = {}", row.line_start));
        chunk_lines.push(format!("line_end = {}", row.line_end));
        chunk_lines.push(format!("heading_level = {}", row.heading_level));
        chunk_lines.push(format!("text_ascii = {}", q(&row.text_ascii)));
        chunk_lines.push(format!("text_sha256 = {}", q(&row.text_sha256)));
        chunk_lines.push(String::new());
    }

    write_ascii(
        &repo_path(repo_root, &args.payload_out),
        &payload_lines.join("\n"),
    )?;
    write_ascii(
        &repo_path(repo_root, &args.chunks_out),
        &chunk_lines.join("\n"),
    )?;
    println!(
        "Wrote structured markdown payload registries: documents={} chunks={} third_party_cache={}",
        documents.len(),
        chunks.len(),
        origin_counts.get("third_party_cache").copied().unwrap_or(0)
    );
    Ok(())
}

fn promote_docs_root_narratives(repo_root: &Path, args: &PromoteDocsRootNarrativesArgs) -> Result<()> {
    const EXCLUDED_ROOT_DOCS: &[&str] = &[
        "BIBLIOGRAPHY.md",
        "CLAIMS_EVIDENCE_MATRIX.md",
        "CLAIMS_TASKS.md",
        "DATASET_MANIFEST.md",
        "EXPERIMENTS_PORTFOLIO_SHORTLIST.md",
        "INSIGHTS.md",
        "NEXT_ACTIONS.md",
        "REQUIREMENTS.md",
        "ROADMAP.md",
        "TODO.md",
    ];

    let existing_ids = load_existing_document_ids(&repo_path(repo_root, &args.out))?;
    let docs_dir = repo_root.join("docs");
    let mut paths = fs::read_dir(&docs_dir)
        .with_context(|| format!("read {}", docs_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("md"))
        .filter(|path| {
            !EXCLUDED_ROOT_DOCS.contains(&path.file_name().and_then(|name| name.to_str()).unwrap_or(""))
        })
        .map(|path| repo_rel(repo_root, &path))
        .collect::<Vec<_>>();
    paths.sort();

    let mut docs = Vec::new();
    let mut assigned = existing_ids.clone();
    let mut next_id = next_id_seed(&existing_ids, "DRN");
    for path in paths {
        let row = parse_root_narrative(repo_root, &path, &mut assigned, &mut next_id)?;
        docs.push(row);
    }
    let rendered = render_docs_root_narratives(&docs);
    let out_path = repo_path(repo_root, &args.out);
    write_ascii(&out_path, &rendered)?;
    println!("Wrote {} with {} documents.", out_path.display(), docs.len());
    Ok(())
}

fn promote_research_narratives(repo_root: &Path, args: &PromoteResearchNarrativesArgs) -> Result<()> {
    let existing_ids = load_existing_document_ids(&repo_path(repo_root, &args.out))?;
    let mut paths = Vec::new();
    for folder in ["docs/theory", "docs/engineering", "docs/research"] {
        let dir = repo_root.join(folder);
        if !dir.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
                continue;
            }
            if path.file_name().and_then(|name| name.to_str()) == Some("INDEX.md") {
                continue;
            }
            paths.push(repo_rel(repo_root, &path));
        }
    }
    paths.sort();

    let mut docs = Vec::new();
    let mut assigned = existing_ids.clone();
    let mut next_id = next_id_seed(&existing_ids, "RN");
    for path in paths {
        let row = parse_research_narrative(repo_root, &path, &mut assigned, &mut next_id)?;
        docs.push(row);
    }
    let rendered = render_research_narratives(&docs);
    let out_path = repo_path(repo_root, &args.out);
    write_ascii(&out_path, &rendered)?;
    println!("Wrote {} with {} documents.", out_path.display(), docs.len());
    Ok(())
}

fn verify_corpus(repo_root: &Path, args: &VerifyCorpusArgs) -> Result<()> {
    let inventory = load_toml(&repo_path(repo_root, &args.inventory))?;
    let corpus = load_toml(&repo_path(repo_root, &args.corpus))?;
    let governance = load_toml(&repo_path(repo_root, &args.governance))?;
    let docs = load_inventory_rows(&inventory);
    let policy = load_governance_policy(&governance);
    let corpus_policy = corpus
        .get("policy")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let safe_from_corpus = string_array_field(&corpus_policy, "safe_classifications")
        .into_iter()
        .collect::<HashSet<_>>();
    let allowed_from_corpus = string_array_field(&corpus_policy, "allowed_tracked_markdown")
        .into_iter()
        .collect::<HashSet<_>>();
    let mut failures = Vec::new();
    if !safe_from_corpus.is_empty() && safe_from_corpus != policy.safe_classifications {
        failures.push(
            "markdown_corpus_registry policy safe_classifications drift from markdown_governance"
                .to_string(),
        );
    }
    if !allowed_from_corpus.is_empty() && allowed_from_corpus != policy.tracked_allowed_paths {
        failures.push(
            "markdown_corpus_registry allowed_tracked_markdown drift from markdown_governance"
                .to_string(),
        );
    }
    for row in &docs {
        if !row.path.is_empty() && !policy.safe_classifications.contains(&row.classification) {
            failures.push(format!(
                "{}: classification={} is outside safe set",
                row.path, row.classification
            ));
        }
        if !row.path.is_empty()
            && row.git_status == "tracked"
            && !policy.tracked_allowed_paths.contains(&row.path)
        {
            failures.push(format!("{}: tracked markdown is outside allowlist", row.path));
        }
        if row.classification == "toml_published_markdown" {
            if row.toml_destination.is_empty() {
                failures.push(format!("{}: missing toml_destination", row.path));
            } else if !repo_root.join(&row.toml_destination).is_file() {
                failures.push(format!(
                    "{}: toml_destination not found -> {}",
                    row.path, row.toml_destination
                ));
            }
        }
    }
    if !failures.is_empty() {
        bail!(
            "Wave 4 markdown corpus policy verification failed.\n- {}",
            failures.join("\n- ")
        );
    }
    println!("OK: Wave 4 markdown corpus policy matches markdown_governance and markdown_inventory.");
    Ok(())
}

fn load_existing_document_ids(path: &Path) -> Result<HashMap<String, String>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let value = load_toml(path)?;
    let mut ids = HashMap::new();
    if let Some(rows) = value.get("document").and_then(Value::as_array) {
        for row in rows.iter().filter_map(Value::as_table) {
            let source_markdown = string_field(row, "source_markdown");
            let id = string_field(row, "id");
            if !source_markdown.is_empty() && !id.is_empty() {
                ids.insert(source_markdown, id);
            }
        }
    }
    Ok(ids)
}

fn next_id_seed(existing_ids: &HashMap<String, String>, prefix: &str) -> usize {
    existing_ids
        .values()
        .filter_map(|id| id.strip_prefix(&format!("{prefix}-")))
        .filter_map(|suffix| suffix.parse::<usize>().ok())
        .max()
        .unwrap_or(0)
        + 1
}

fn narrative_id_for_path(
    path: &str,
    prefix: &str,
    assigned: &mut HashMap<String, String>,
    next_id: &mut usize,
) -> String {
    if let Some(id) = assigned.get(path) {
        return id.clone();
    }
    let id = format!("{prefix}-{:03}", *next_id);
    *next_id += 1;
    assigned.insert(path.to_string(), id.clone());
    id
}

fn parse_root_narrative(
    repo_root: &Path,
    path: &str,
    assigned: &mut HashMap<String, String>,
    next_id: &mut usize,
) -> Result<NarrativeDocRow> {
    let full = repo_root.join(path);
    let raw = fs::read_to_string(&full).with_context(|| format!("read {}", full.display()))?;
    let body = normalize_promoted_body(&raw);
    let fallback = Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(path);
    let file_name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path);
    Ok(NarrativeDocRow {
        id: narrative_id_for_path(path, "DRN", assigned, next_id),
        source_markdown: path.to_string(),
        domain: String::new(),
        slug: fallback.to_ascii_lowercase().replace(' ', "_"),
        title: first_title(&body, fallback),
        status_token: root_status_token(file_name),
        content_kind: root_content_kind(file_name),
        verification_level: String::new(),
        claim_refs: extract_claim_like_refs(&body),
        url_refs: Vec::new(),
        path_refs: extract_backtick_paths(&body),
        line_count: body.lines().count() + usize::from(!body.is_empty() && !body.ends_with('\n')),
        body_markdown: body,
    })
}

fn parse_research_narrative(
    repo_root: &Path,
    path: &str,
    assigned: &mut HashMap<String, String>,
    next_id: &mut usize,
) -> Result<NarrativeDocRow> {
    let full = repo_root.join(path);
    let raw = fs::read_to_string(&full).with_context(|| format!("read {}", full.display()))?;
    let body = normalize_promoted_body(&raw);
    let rel_path = Path::new(path);
    let fallback = rel_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(path);
    let file_name = rel_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path);
    let domain = rel_path
        .components()
        .nth(1)
        .map(|part| part.as_os_str().to_string_lossy().to_string())
        .unwrap_or_else(|| "docs".to_string());
    Ok(NarrativeDocRow {
        id: narrative_id_for_path(path, "RN", assigned, next_id),
        source_markdown: path.to_string(),
        domain: domain.clone(),
        slug: fallback.to_ascii_lowercase().replace(' ', "_"),
        title: first_title(&body, fallback),
        status_token: research_status_token(file_name),
        content_kind: research_content_kind(file_name),
        verification_level: research_verification_level(&domain, file_name),
        claim_refs: extract_claim_like_refs(&body),
        url_refs: extract_urls(&body),
        path_refs: extract_backtick_paths(&body),
        line_count: body.lines().count() + usize::from(!body.is_empty() && !body.ends_with('\n')),
        body_markdown: body,
    })
}

fn normalize_promoted_body(text: &str) -> String {
    let cleaned = strip_generated_preamble(text);
    ascii_clean(cleaned.trim_matches('\n')).trim_end().to_string()
}

fn strip_generated_preamble(text: &str) -> String {
    let mut lines = text.lines();
    let first = lines.next().unwrap_or_default().trim();
    let second = lines.next().unwrap_or_default().trim();
    if first.contains("AUTO-GENERATED") && second.contains("Source of truth:") {
        let remaining = lines.collect::<Vec<_>>().join("\n");
        return remaining.trim_start_matches('\n').to_string();
    }
    text.to_string()
}

fn root_status_token(filename: &str) -> String {
    let upper = filename.to_ascii_uppercase();
    if upper.contains("AUDIT") {
        "AUDIT".to_string()
    } else if upper.contains("REPORT") {
        "REPORT".to_string()
    } else if upper.contains("STATUS") {
        "STATUS".to_string()
    } else if upper.contains("ANALYSIS") {
        "ANALYSIS".to_string()
    } else if upper.contains("GLOSSARY") {
        "GLOSSARY".to_string()
    } else {
        "NARRATIVE".to_string()
    }
}

fn root_content_kind(filename: &str) -> String {
    let upper = filename.to_ascii_uppercase();
    if upper.contains("TAXONOMY") || upper.contains("GLOSSARY") {
        "taxonomy_or_glossary".to_string()
    } else if upper.contains("REPLICATION") || upper.contains("VALIDATION") {
        "validation_note".to_string()
    } else if upper.contains("PROVENANCE") {
        "provenance_note".to_string()
    } else if upper.contains("ROADMAP") || upper.contains("STATUS") {
        "status_note".to_string()
    } else if upper.contains("ANALYSIS") || upper.contains("SYNTHESIS") {
        "analysis_note".to_string()
    } else {
        "research_note".to_string()
    }
}

fn research_status_token(filename: &str) -> String {
    let upper = filename.to_ascii_uppercase();
    if upper.starts_with("PHASE_") {
        "PHASE_REPORT".to_string()
    } else if upper.contains("AUDIT") {
        "AUDIT".to_string()
    } else if upper.contains("REPORT") {
        "REPORT".to_string()
    } else if upper.contains("SPEC") || upper.contains("PROTOCOL") {
        "SPECIFICATION".to_string()
    } else if upper.contains("METHODOLOGY") {
        "METHODOLOGY".to_string()
    } else if upper.contains("RECONCILIATION") {
        "RECONCILIATION".to_string()
    } else {
        "NARRATIVE".to_string()
    }
}

fn research_content_kind(filename: &str) -> String {
    let upper = filename.to_ascii_uppercase();
    if upper.starts_with("PHASE_") {
        "phase_execution_report".to_string()
    } else if upper.contains("AUDIT") {
        "audit_note".to_string()
    } else if upper.contains("REPORT") {
        "engineering_report".to_string()
    } else if upper.contains("SPEC") || upper.contains("PROTOCOL") {
        "specification".to_string()
    } else if upper.contains("THEORY")
        || upper.contains("RECONCILIATION")
        || upper.contains("METHODOLOGY")
    {
        "theory_note".to_string()
    } else {
        "research_note".to_string()
    }
}

fn research_verification_level(domain: &str, filename: &str) -> String {
    let upper = filename.to_ascii_uppercase();
    if upper.contains("VALIDATION") || upper.contains("VERIFICATION") {
        "validation_summary".to_string()
    } else if domain == "theory" {
        "theoretical".to_string()
    } else {
        "engineering_narrative".to_string()
    }
}

fn extract_claim_like_refs(text: &str) -> Vec<String> {
    let re = Regex::new(r"\b[A-Z]{1,3}-\d{3,4}\b").expect("claim-like regex");
    let mut seen = BTreeSet::new();
    for matched in re.find_iter(text) {
        seen.insert(matched.as_str().to_string());
    }
    seen.into_iter().collect()
}

fn extract_urls(text: &str) -> Vec<String> {
    let re = Regex::new(r#"https?://[^\s)>\"]+"#).expect("url regex");
    let mut seen = BTreeSet::new();
    for matched in re.find_iter(text) {
        let token = matched.as_str().trim_end_matches(&['.', ',', ';'][..]).to_string();
        if !token.is_empty() {
            seen.insert(token);
        }
    }
    seen.into_iter().collect()
}

fn extract_backtick_paths(text: &str) -> Vec<String> {
    let re = Regex::new(r"`([^`\n]+)`").expect("backtick regex");
    let mut seen = BTreeSet::new();
    for caps in re.captures_iter(text) {
        let token = caps.get(1).map(|m| m.as_str().trim()).unwrap_or_default();
        if token.is_empty()
            || token.starts_with("http://")
            || token.starts_with("https://")
            || (!token.contains('/') && !token.contains('.'))
        {
            continue;
        }
        seen.insert(normalize_path(token));
    }
    seen.into_iter().collect()
}

fn toml_string(value: &str) -> String {
    q(value)
}

fn toml_list(values: &[String]) -> String {
    q_list(values)
}

fn toml_multiline(value: &str) -> String {
    if !value.contains("'''") {
        format!("'''\n{}\n'''", value)
    } else {
        toml_string(value)
    }
}

fn render_docs_root_narratives(docs: &[NarrativeDocRow]) -> String {
    let updated = Local::now().format("%Y-%m-%d").to_string();
    let mut lines = vec![
        "# Root docs narratives registry (TOML-first).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/markdown_registry.rs".to_string(),
        String::new(),
        "[docs_root_narratives]".to_string(),
        format!("updated = {}", toml_string(&updated)),
        "authoritative = true".to_string(),
        r#"source_markdown_glob = "docs/*.md""#.to_string(),
        format!("document_count = {}", docs.len()),
        String::new(),
    ];
    for doc in docs {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {}", toml_string(&doc.id)));
        lines.push(format!(
            "source_markdown = {}",
            toml_string(&doc.source_markdown)
        ));
        lines.push(format!("slug = {}", toml_string(&doc.slug)));
        lines.push(format!("title = {}", toml_string(&doc.title)));
        lines.push(format!(
            "status_token = {}",
            toml_string(&doc.status_token)
        ));
        lines.push(format!(
            "content_kind = {}",
            toml_string(&doc.content_kind)
        ));
        lines.push(format!("claim_refs = {}", toml_list(&doc.claim_refs)));
        lines.push(format!("path_refs = {}", toml_list(&doc.path_refs)));
        lines.push(format!("line_count = {}", doc.line_count));
        lines.push(format!(
            "body_markdown = {}",
            toml_multiline(&doc.body_markdown)
        ));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_research_narratives(docs: &[NarrativeDocRow]) -> String {
    let updated = Local::now().format("%Y-%m-%d").to_string();
    let mut lines = vec![
        "# Research narrative registry (TOML-first).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/markdown_registry.rs".to_string(),
        String::new(),
        "[research_narratives]".to_string(),
        format!("updated = {}", toml_string(&updated)),
        "authoritative = true".to_string(),
        r#"source_markdown_globs = ["docs/theory/*.md", "docs/engineering/*.md", "docs/research/*.md"]"#.to_string(),
        format!("document_count = {}", docs.len()),
        String::new(),
    ];
    for doc in docs {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {}", toml_string(&doc.id)));
        lines.push(format!(
            "source_markdown = {}",
            toml_string(&doc.source_markdown)
        ));
        lines.push(format!("domain = {}", toml_string(&doc.domain)));
        lines.push(format!("slug = {}", toml_string(&doc.slug)));
        lines.push(format!("title = {}", toml_string(&doc.title)));
        lines.push(format!(
            "status_token = {}",
            toml_string(&doc.status_token)
        ));
        lines.push(format!(
            "content_kind = {}",
            toml_string(&doc.content_kind)
        ));
        lines.push(format!(
            "verification_level = {}",
            toml_string(&doc.verification_level)
        ));
        lines.push(format!("claim_refs = {}", toml_list(&doc.claim_refs)));
        lines.push(format!("url_refs = {}", toml_list(&doc.url_refs)));
        lines.push(format!("path_refs = {}", toml_list(&doc.path_refs)));
        lines.push(format!("line_count = {}", doc.line_count));
        lines.push(format!(
            "body_markdown = {}",
            toml_multiline(&doc.body_markdown)
        ));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn verify_toml_inventory(repo_root: &Path, args: &VerifyTomlInventoryArgs) -> Result<()> {
    let inventory = load_toml(&repo_path(repo_root, &args.inventory))?;
    let markdown_inventory = load_toml(&repo_path(repo_root, &args.markdown_inventory))?;
    let summary = inventory
        .get("toml_inventory")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let docs = load_toml_inventory_rows(&inventory);
    let doc_paths = docs.iter().map(|row| row.path.clone()).collect::<Vec<_>>();
    let doc_path_set = doc_paths.iter().cloned().collect::<HashSet<_>>();
    let mut failures = Vec::new();
    if integer_field(&summary, "document_count") != docs.len() as i64 {
        failures.push(format!(
            "document_count mismatch: {} vs {}",
            integer_field(&summary, "document_count"),
            docs.len()
        ));
    }
    if integer_field(&summary, "parse_error_count") != 0 {
        failures.push(format!(
            "parse_error_count={} (expected 0)",
            integer_field(&summary, "parse_error_count")
        ));
    }
    if doc_paths.len() != doc_path_set.len() {
        failures.push("duplicate TOML paths detected in registry/toml_inventory.toml".to_string());
    }
    for row in &docs {
        if !repo_root.join(&row.path).is_file() {
            failures.push(format!("{}: file missing on disk", row.path));
        }
        if !row.parse_ok {
            failures.push(format!("{}: parse_ok=false", row.path));
        }
        let (expected_role, _) = classify_toml_path(&row.path);
        if row.role != expected_role {
            failures.push(format!(
                "{}: role={} expected={}",
                row.path, row.role, expected_role
            ));
        }
    }
    let required_core = HashSet::from([
        "registry/claims.toml",
        "registry/insights.toml",
        "registry/experiments.toml",
        "registry/bibliography.toml",
        "registry/roadmap.toml",
        "registry/todo.toml",
        "registry/next_actions.toml",
        "registry/markdown_inventory.toml",
        "registry/markdown_governance.toml",
        "registry/markdown_corpus_registry.toml",
        "registry/toml_inventory.toml",
        "registry/csv_inventory.toml",
        "registry/csv_migration_scope.toml",
        "registry/control_plane_roadmap.toml",
    ]);
    for path in required_core {
        if !doc_path_set.contains(path) {
            failures.push(format!("missing required TOML inventory path: {path}"));
        }
    }
    let archival_compat = "registry/wave4_roadmap.toml";
    if repo_root.join(archival_compat).is_file() && !doc_path_set.contains(archival_compat) {
        failures.push(format!(
            "missing archival compatibility TOML inventory path: {archival_compat}"
        ));
    }
    for row in load_inventory_rows(&markdown_inventory) {
        if row.classification == "toml_published_markdown"
            && !row.toml_destination.is_empty()
            && !doc_path_set.contains(&row.toml_destination)
        {
            failures.push(format!(
                "markdown destination missing from TOML inventory: {}",
                row.toml_destination
            ));
        }
    }
    if !failures.is_empty() {
        bail!(
            "Control-plane TOML inventory verification failed.\n- {}",
            failures.join("\n- ")
        );
    }
    println!("OK: Control-plane TOML inventory coverage and role checks passed.");
    Ok(())
}

fn verify_embedded(repo_root: &Path, args: &VerifyEmbeddedArgs) -> Result<()> {
    let governance = load_toml(&repo_path(repo_root, &args.governance))?;
    let payload_raw = load_toml(&repo_path(repo_root, &args.payload_path))?;
    let chunks_raw = load_toml(&repo_path(repo_root, &args.chunks_path))?;
    let payload_meta = payload_raw
        .get("embedded_markdown_payloads")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let chunks_meta = chunks_raw
        .get("embedded_markdown_chunks")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let docs = payload_raw
        .get("document")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let chunks = chunks_raw
        .get("chunk")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let governance_docs = load_governance_docs(&governance)
        .into_iter()
        .map(|row| (row.path.clone(), row.mode))
        .collect::<HashMap<_, _>>();
    let governed_forbidden = governance
        .get("policy")
        .and_then(Value::as_table)
        .map(|policy| string_array_field(policy, "disk_forbidden_modes"))
        .unwrap_or_default()
        .into_iter()
        .collect::<HashSet<_>>();
    let policy = load_governance_policy(&governance);
    let expected_paths = collect_embedded_candidates(repo_root, &policy)?
        .keys()
        .cloned()
        .collect::<HashSet<_>>();
    let mut failures = Vec::new();
    if integer_field(&payload_meta, "document_count") != docs.len() as i64 {
        failures.push("payload document_count metadata mismatch".to_string());
    }
    if integer_field(&chunks_meta, "chunk_count") != chunks.len() as i64 {
        failures.push("chunks chunk_count metadata mismatch".to_string());
    }
    if string_field(&payload_meta, "representation") != "structured_toml_units" {
        failures.push("payload representation must be structured_toml_units".to_string());
    }
    if string_field(&chunks_meta, "representation") != "structured_toml_units" {
        failures.push("chunks representation must be structured_toml_units".to_string());
    }
    let mut chunk_ids_seen = HashSet::new();
    let mut chunk_by_id = HashMap::<String, Value>::new();
    for chunk in &chunks {
        if let Some(table) = chunk.as_table() {
            let id = string_field(table, "id");
            if !chunk_ids_seen.insert(id.clone()) {
                failures.push(format!("duplicate chunk id: {id}"));
            }
            chunk_by_id.insert(id, chunk.clone());
        }
    }
    let mut payload_paths = HashSet::new();
    for doc in &docs {
        if let Some(table) = doc.as_table() {
            let path = string_field(table, "path");
            payload_paths.insert(path.clone());
            if string_field(table, "content_encoding") != "structured_toml_units" {
                failures.push(format!("document not structured_toml_units: {path}"));
            }
            let chunk_ids = string_array_field(table, "chunk_ids");
            if chunk_ids.len() as i64 != integer_field(table, "chunk_count") {
                failures.push(format!("chunk_count mismatch: {path}"));
            }
            for chunk_id in chunk_ids {
                if !chunk_by_id.contains_key(&chunk_id) {
                    failures.push(format!("missing chunk for document {path}: {chunk_id}"));
                }
            }
            if repo_root.join(&path).exists() {
                let mode = governance_docs.get(&path).cloned().unwrap_or_default();
                if governed_forbidden.contains(&mode) {
                    failures.push(format!(
                        "markdown exists on disk despite forbidden governance mode: {path}"
                    ));
                }
            }
        }
    }
    let missing = expected_paths
        .difference(&payload_paths)
        .cloned()
        .collect::<Vec<_>>();
    let extra = payload_paths
        .difference(&expected_paths)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        failures.push(format!(
            "missing structured docs for embedded markdown paths: {}",
            missing.len()
        ));
        for item in missing.iter().take(25) {
            failures.push(format!("  missing: {item}"));
        }
    }
    if !extra.is_empty() {
        failures.push(format!("unexpected extra payload paths: {}", extra.len()));
        for item in extra.iter().take(25) {
            failures.push(format!("  extra: {item}"));
        }
    }
    if !failures.is_empty() {
        bail!(
            "embedded markdown structured registry verification failed:\n- {}",
            failures.join("\n- ")
        );
    }
    println!(
        "OK: embedded markdown structured registries verified. documents={} chunks={}",
        docs.len(),
        chunks.len()
    );
    Ok(())
}

fn verify_inventory_toml_first(repo_root: &Path, args: &VerifyInventoryArgs) -> Result<()> {
    let inventory = load_toml(&repo_path(repo_root, &args.inventory))?;
    let governance = load_toml(&repo_path(repo_root, &args.governance))?;
    let rows = load_inventory_rows(&inventory);
    let inventory_summary = inventory
        .get("markdown_inventory")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let policy = load_governance_policy(&governance);
    let governance_docs = load_governance_docs(&governance)
        .into_iter()
        .map(|row| (row.path.clone(), row))
        .collect::<HashMap<_, _>>();
    let allowed_modes = HashSet::from([
        "toml_generated_mirror".to_string(),
        "toml_manual_source".to_string(),
        "immutable_transcript".to_string(),
        "manual_narrative".to_string(),
        "generated_artifact".to_string(),
        "third_party_markdown".to_string(),
    ]);
    let tracked_count = integer_field(&inventory_summary, "tracked_count");
    let mut failures = Vec::new();
    let mut disallowed_tracked_count = 0usize;
    for row in &rows {
        let tracked_allowed = policy.tracked_allowed_paths.contains(&row.path);
        let governance_mode = governance_docs
            .get(&row.path)
            .map(|row| row.mode.clone())
            .unwrap_or_default();
        if row.git_status == "tracked"
            && row.classification != "third_party_markdown"
            && !tracked_allowed
        {
            disallowed_tracked_count += 1;
            failures.push(format!(
                "{}: tracked markdown is disallowed in strict TOML-only mode",
                row.path
            ));
        }
        if !policy.safe_classifications.contains(&row.classification) {
            failures.push(format!(
                "{}: disallowed classification={}",
                row.path, row.classification
            ));
        }
        if row.classification == "generated_artifact" {
            if !row.path.starts_with("build/docs/generated/") {
                failures.push(format!(
                    "{}: generated_artifact allowed only under build/docs/generated/",
                    row.path
                ));
            }
            if row.git_status == "tracked" && row.classification != "third_party_markdown" {
                failures.push(format!(
                    "{}: generated_artifact must not be tracked",
                    row.path
                ));
            }
            continue;
        }
        if row.classification == "toml_destination_exists_manual_markdown" {
            if row.git_status == "tracked" && !tracked_allowed {
                failures.push(format!(
                    "{}: manual markdown with TOML destination must not be tracked",
                    row.path
                ));
            }
            if governance_mode == "toml_generated_mirror" {
                failures.push(format!(
                    "{}: governance expects generated mirror but inventory marks manual markdown",
                    row.path
                ));
            }
            if row.toml_destination.is_empty() {
                failures.push(format!(
                    "{}: missing toml_destination for manual markdown",
                    row.path
                ));
            } else if !repo_root.join(&row.toml_destination).is_file() {
                failures.push(format!(
                    "{}: missing toml_destination file {}",
                    row.path, row.toml_destination
                ));
            }
            continue;
        }
        if row.classification == "toml_published_markdown" {
            if !governance_mode.is_empty() && !allowed_modes.contains(&governance_mode) {
                failures.push(format!(
                    "{}: governance mode {} is not publishable",
                    row.path, governance_mode
                ));
            }
            if !governance_mode.is_empty() && governance_mode != "toml_generated_mirror" {
                failures.push(format!(
                    "{}: governance mode {} conflicts with published classification",
                    row.path, governance_mode
                ));
            }
            if !row.generated_declared && !row.path.starts_with("build/docs/generated/") {
                failures.push(format!(
                    "{}: toml_published_markdown without explicit generated marker header",
                    row.path
                ));
            }
            if row.toml_destination.is_empty() {
                failures.push(format!(
                    "{}: toml_published_markdown without toml_destination",
                    row.path
                ));
            } else if !repo_root.join(&row.toml_destination).is_file() {
                failures.push(format!(
                    "{}: missing toml_destination file {}",
                    row.path, row.toml_destination
                ));
            }
        }
    }
    if tracked_count != 0 && disallowed_tracked_count != 0 {
        failures.push(format!(
            "disallowed tracked markdown count={} (tracked_count={})",
            disallowed_tracked_count, tracked_count
        ));
    }
    if !failures.is_empty() {
        bail!(
            "markdown inventory violates strict TOML-only policy:\n- {}",
            failures.join("\n- ")
        );
    }
    println!("OK: markdown inventory is TOML-first with governance-backed tracked exceptions.");
    Ok(())
}

fn verify_origin_audit(repo_root: &Path, args: &VerifyOriginArgs) -> Result<()> {
    let audit = load_toml(&repo_path(repo_root, &args.audit))?;
    let summary = audit
        .get("markdown_origin_audit")
        .and_then(Value::as_table)
        .context("markdown_origin_audit table missing")?;
    let docs = load_origin_rows(&audit);
    let queue_len = audit
        .get("consolidation_queue")
        .and_then(Value::as_array)
        .map(|rows| rows.len())
        .unwrap_or(0);
    let mut failures = Vec::new();
    if integer_field(summary, "document_count") != docs.len() as i64 {
        failures.push(format!(
            "document_count mismatch: {} vs {}",
            integer_field(summary, "document_count"),
            docs.len()
        ));
    }
    if integer_field(summary, "needs_consolidation_count") != queue_len as i64 {
        failures.push(format!(
            "needs_consolidation_count mismatch: {} vs {}",
            integer_field(summary, "needs_consolidation_count"),
            queue_len
        ));
    }
    if queue_len != 0 {
        failures.push(format!("consolidation_queue not empty ({queue_len} items)"));
    }
    for row in &docs {
        if !matches!(row.scope.as_str(), "docs" | "reports" | "data_artifacts") {
            failures.push(format!("{}: invalid scope={}", row.path, row.scope));
        }
        if row.origin_status != "generated_from_repo_process" {
            failures.push(format!("{}: origin_status={}", row.path, row.origin_status));
        }
        if row.toml_destination.is_empty() {
            failures.push(format!("{}: missing toml_destination", row.path));
        } else if !row.destination_exists {
            failures.push(format!(
                "{}: toml_destination missing on disk ({})",
                row.path, row.toml_destination
            ));
        }
        if !row.header_auto_generated || !row.header_source_of_truth {
            failures.push(format!(
                "{}: missing headers auto={} source={}",
                row.path, row.header_auto_generated, row.header_source_of_truth
            ));
        }
    }
    if !failures.is_empty() {
        bail!(
            "markdown origin audit verification failed:\n- {}",
            failures.join("\n- ")
        );
    }
    println!("OK: markdown origin audit verified (all in-scope markdown is TOML-generated).");
    Ok(())
}

fn verify_owner_map(repo_root: &Path, args: &VerifyOwnerArgs) -> Result<()> {
    let inventory = load_toml(&repo_path(repo_root, &args.inventory))?;
    let owner_map = load_toml(&repo_path(repo_root, &args.owner_map))?;
    let governance = load_toml(&repo_path(repo_root, &args.governance))?;
    let inv_rows = load_inventory_rows(&inventory);
    let owner_rows = load_owner_rows(&owner_map);
    let policy = load_governance_policy(&governance);
    let governance_by_path = load_governance_docs(&governance)
        .into_iter()
        .map(|row| (row.path.clone(), row))
        .collect::<HashMap<_, _>>();

    let in_scope = inv_rows
        .into_iter()
        .filter(|row| {
            policy.owner_scope_paths.contains(&row.path)
                || policy
                    .owner_scope_prefixes
                    .iter()
                    .any(|prefix| row.path.starts_with(prefix))
        })
        .collect::<Vec<_>>();
    let mut owner_by_path = HashMap::<String, OwnerRow>::new();
    let mut duplicate_owner_paths = BTreeSet::new();
    for row in owner_rows {
        if owner_by_path.contains_key(&row.path) {
            duplicate_owner_paths.insert(row.path.clone());
        }
        owner_by_path.insert(row.path.clone(), row);
    }

    let mut failures = Vec::new();
    for path in duplicate_owner_paths {
        failures.push(format!("duplicate owner mapping entry: {path}"));
    }
    let inventory_paths = in_scope
        .iter()
        .map(|row| row.path.clone())
        .collect::<BTreeSet<_>>();
    let owner_paths = owner_by_path.keys().cloned().collect::<BTreeSet<_>>();
    for path in inventory_paths.difference(&owner_paths) {
        failures.push(format!(
            "{}: missing explicit owner mapping; conversion path: {}",
            path,
            conversion_hint(path, "")
        ));
    }
    for path in owner_paths.difference(&inventory_paths) {
        failures.push(format!(
            "{}: stale owner mapping (file not present in in-scope markdown)",
            path
        ));
    }

    for row in &in_scope {
        let owner = owner_by_path.get(&row.path);
        let gov_row = governance_by_path.get(&row.path);
        let gov_refs = gov_row
            .map(|row| row.source_toml_refs.clone())
            .unwrap_or_default();
        let canonical = owner
            .map(|row| row.canonical_toml.clone())
            .or_else(|| {
                if row.toml_destination.is_empty() {
                    gov_refs.first().cloned()
                } else {
                    Some(row.toml_destination.clone())
                }
            })
            .unwrap_or_default();
        if canonical.is_empty() {
            failures.push(format!("{}: owner map canonical_toml is empty", row.path));
            continue;
        }
        if !repo_root.join(&canonical).is_file() {
            failures.push(format!(
                "{}: canonical_toml missing on disk: {}",
                row.path, canonical
            ));
        }
        if owner.is_some() && !row.toml_destination.is_empty() && row.toml_destination != canonical
        {
            failures.push(format!(
                "{}: owner canonical_toml mismatch inventory destination ({} != {})",
                row.path, canonical, row.toml_destination
            ));
        }
        if !policy.safe_classifications.contains(&row.classification) {
            failures.push(format!(
                "{}: disallowed classification={}",
                row.path, row.classification
            ));
        }
        if row.classification == "toml_published_markdown"
            && owner
                .map(|row| row.requires_generated_header)
                .unwrap_or(false)
        {
            let text = fs::read_to_string(repo_root.join(&row.path))
                .with_context(|| format!("read {}", row.path))?;
            let head = text.lines().take(80).collect::<Vec<_>>().join("\n");
            if !head.contains("AUTO-GENERATED") {
                failures.push(format!("{}: missing AUTO-GENERATED header", row.path));
            }
            if !head.contains("Source of truth:") {
                failures.push(format!("{}: missing Source of truth header", row.path));
            }
            if !head.contains(&canonical) {
                failures.push(format!(
                    "{}: Source of truth header does not reference canonical_toml ({})",
                    row.path, canonical
                ));
            }
        }
    }
    if !failures.is_empty() {
        bail!(
            "markdown owner map verification failed:\n- {}",
            failures.join("\n- ")
        );
    }
    println!(
        "OK: markdown owner map verified. All in-scope markdown has explicit canonical TOML ownership."
    );
    Ok(())
}

fn build_inventory_doc(
    repo_root: &Path,
    path: &str,
    git_status: &str,
    refs: &HashMap<String, HashSet<String>>,
    generated_allowlist: &HashSet<String>,
) -> Result<InventoryDoc> {
    let full = repo_root.join(path);
    let raw = fs::read(&full).with_context(|| format!("read {}", full.display()))?;
    let text = String::from_utf8_lossy(&raw).to_string();
    let title = first_title(&text, Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or(path));
    let generated_declared = declared_generated(&text);
    let generated_pattern = is_generated_pattern(path);
    let mut toml_destination = markdown_destination_override(path).unwrap_or_default();
    if toml_destination.is_empty() {
        toml_destination = destination_by_scope(path);
    }
    if toml_destination.is_empty() {
        toml_destination = choose_destination(refs.get(path));
    }
    let generated = is_pipeline_generated(
        path,
        generated_declared,
        generated_pattern,
        &toml_destination,
        generated_allowlist,
    );
    let third_party = is_third_party(path);
    let manual_exception = MANUAL_EXCEPTIONS.contains(&path);
    let (classification, migration_action, migration_priority, rationale) =
        classify_markdown(path, generated, third_party, manual_exception, &toml_destination);
    Ok(InventoryDoc {
        path: path.to_string(),
        title: collapse(&title),
        git_status: git_status.to_string(),
        archived: is_archived_markdown(path),
        generated_declared,
        generated_pattern,
        generated,
        manual_exception,
        third_party,
        classification,
        migration_action,
        migration_priority,
        toml_destination,
        rationale,
        size_bytes: raw.len(),
        line_count: text.lines().count() + usize::from(!text.is_empty() && !text.ends_with('\n')),
        sha256: sha256_hex(text.as_bytes()),
        claim_ref_count: count_regex(r"\bC-\d{3}\b", &text)?,
        insight_ref_count: count_regex(r"\bI-\d{3}\b", &text)?,
        experiment_ref_count: count_regex(r"\bE-\d{3}\b", &text)?,
    })
}

fn markdown_destination_override(path: &str) -> Option<String> {
    Some(match path {
        "proofs/README.md" => "registry/roadmap.toml",
        "proofs/EPISTEMIC_BOUNDARIES.md" => "registry/entrypoint_docs.toml",
        "PANTHEON_PHYSICSFORGE_90_POINT_MIGRATION_PLAN.md"
        | "PHASE10_11_ULTIMATE_ROADMAP.md"
        | "PYTHON_REFACTORING_ROADMAP.md"
        | "SYNTHESIS_PIPELINE_PROGRESS.md"
        | "crates/sign_imbalance/IMPLEMENTATION_NOTES.md" => {
            "registry/legacy_markdown_interfaces.toml"
        }
        "data/artifacts/ALGEBRAIC_FOUNDATIONS.md"
        | "data/artifacts/BIBLIOGRAPHY.md"
        | "data/artifacts/FINAL_REPORT.md"
        | "data/artifacts/QUANTUM_REPORT.md"
        | "data/artifacts/SIMULATION_REPORT.md"
        | "data/artifacts/WARP_RING_REPORT.md"
        | "data/artifacts/extracted_equations.md"
        | "data/artifacts/reality_check_and_synthesis.md" => "registry/artifact_scrolls.toml",
        "docs/generated/BIBLIOGRAPHY_REGISTRY_MIRROR.md" => "registry/bibliography.toml",
        "docs/generated/BOOK_DOCS_REGISTRY_MIRROR.md" => "registry/book_docs.toml",
        "docs/generated/CLAIMS_DOMAINS_REGISTRY_MIRROR.md" => "registry/claims_domains.toml",
        "docs/generated/CLAIMS_REGISTRY_MIRROR.md" => "registry/claims.toml",
        "docs/generated/CLAIMS_TASKS_REGISTRY_MIRROR.md" => "registry/claims_tasks.toml",
        "docs/generated/CLAIM_TICKETS_REGISTRY_MIRROR.md" => "registry/claim_tickets.toml",
        "docs/generated/DATA_ARTIFACT_NARRATIVES_REGISTRY_MIRROR.md" => "registry/artifact_scrolls.toml",
        "docs/generated/DOCS_CONVOS_REGISTRY_MIRROR.md" => "registry/docs_convos.toml",
        "docs/generated/DOCS_ROOT_NARRATIVES_REGISTRY_MIRROR.md" => "registry/docs_root_narratives.toml",
        "docs/generated/ENTRYPOINT_DOCS_REGISTRY_MIRROR.md" => "registry/entrypoint_docs.toml",
        "docs/generated/EXPERIMENTS_REGISTRY_MIRROR.md" => "registry/experiments.toml",
        "docs/generated/EXTERNAL_SOURCES_REGISTRY_MIRROR.md" => "registry/external_sources.toml",
        "docs/generated/INSIGHTS_REGISTRY_MIRROR.md" => "registry/insights.toml",
        "docs/generated/KNOWLEDGE_MIGRATION_PLAN_REGISTRY_MIRROR.md" => "registry/knowledge_migration_plan.toml",
        "docs/generated/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md" => "registry/markdown_governance.toml",
        "docs/generated/NAVIGATOR_REGISTRY_MIRROR.md" => "registry/navigator.toml",
        "docs/generated/NEXT_ACTIONS_REGISTRY_MIRROR.md" => "registry/next_actions.toml",
        "docs/generated/REPORTS_NARRATIVES_REGISTRY_MIRROR.md" => "registry/reports_narratives.toml",
        "docs/generated/REQUIREMENTS_REGISTRY_MIRROR.md" => "registry/requirements.toml",
        "docs/generated/RESEARCH_NARRATIVES_REGISTRY_MIRROR.md" => "registry/research_narratives.toml",
        "docs/generated/ROADMAP_REGISTRY_MIRROR.md" => "registry/roadmap.toml",
        "docs/generated/TODO_REGISTRY_MIRROR.md" => "registry/todo.toml",
        "docs/claims/INDEX.md" => "registry/claims_domains.toml",
        "docs/DATASET_MANIFEST.md" => "registry/external_sources.toml",
        "crates/lbm_3d_cuda/README.md" => "registry/entrypoint_docs.toml",
        _ => return None,
    }
    .to_string())
}

fn destination_by_scope(path: &str) -> String {
    match path {
        "AGENTS.md" => "agents.toml".to_string(),
        _ if path.starts_with("build/docs/generated/") => "registry/markdown_payloads.toml".to_string(),
        _ if path.starts_with("docs/generated/") => "registry/markdown_payloads.toml".to_string(),
        _ if path.starts_with("docs/book/src/") => "registry/book_docs.toml".to_string(),
        _ if path.starts_with("docs/external_sources/") => "registry/external_sources.toml".to_string(),
        _ if path.starts_with("docs/claims/by_domain/") => "registry/claims_domains.toml".to_string(),
        _ if path.starts_with("docs/tickets/") => "registry/claim_tickets.toml".to_string(),
        _ if path.starts_with("docs/convos/") => "registry/docs_convos.toml".to_string(),
        _ if path.starts_with("docs/engineering/")
            || path.starts_with("docs/research/")
            || path.starts_with("docs/theory/") =>
        {
            "registry/research_narratives.toml".to_string()
        }
        _ if path.starts_with("docs/monograph/") => "registry/monograph.toml".to_string(),
        _ if path.starts_with("docs/requirements/") => "registry/requirements.toml".to_string(),
        _ if path.starts_with("reports/") => "registry/reports_narratives.toml".to_string(),
        _ if path.starts_with("data/artifacts/") && path != "data/artifacts/README.md" => {
            "registry/artifact_scrolls.toml".to_string()
        }
        "CLAUDE.md" | "GEMINI.md" | "README.md" | "curated/README.md"
        | "curated/01_theory_frameworks/README_COQ.md" | "data/csv/README.md"
        | "data/artifacts/README.md" => "registry/entrypoint_docs.toml".to_string(),
        "NAVIGATOR.md" => "registry/navigator.toml".to_string(),
        "REQUIREMENTS.md" | "docs/REQUIREMENTS.md" => "registry/requirements.toml".to_string(),
        "docs/CLAIMS_EVIDENCE_MATRIX.md" => "registry/claims.toml".to_string(),
        "docs/INSIGHTS.md" => "registry/insights.toml".to_string(),
        "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md" => "registry/experiments.toml".to_string(),
        "docs/ROADMAP.md" => "registry/roadmap.toml".to_string(),
        "docs/TODO.md" => "registry/todo.toml".to_string(),
        "docs/NEXT_ACTIONS.md" => "registry/next_actions.toml".to_string(),
        "docs/CLAIMS_TASKS.md" => "registry/claims_tasks.toml".to_string(),
        _ => String::new(),
    }
}

fn choose_destination(candidates: Option<&HashSet<String>>) -> String {
    let Some(candidates) = candidates else {
        return String::new();
    };
    if candidates.is_empty() {
        return String::new();
    }
    let mut sorted = candidates.iter().cloned().collect::<Vec<_>>();
    sorted.sort();
    let preferred = sorted
        .iter()
        .filter(|item| *item != "registry/knowledge_migration_plan.toml")
        .cloned()
        .collect::<Vec<_>>();
    if let Some(canonical) = preferred
        .iter()
        .find(|item| !item.ends_with("_narrative.toml"))
        .cloned()
    {
        canonical
    } else {
        preferred
            .first()
            .cloned()
            .or_else(|| sorted.first().cloned())
            .unwrap_or_default()
    }
}

fn is_archived_markdown(path: &str) -> bool {
    path.starts_with("archive/") || path.starts_with("docs/archive/")
}

fn is_generated_pattern(path: &str) -> bool {
    GENERATED_PATTERNS
        .iter()
        .filter_map(|pattern| Pattern::new(pattern).ok())
        .any(|pattern| pattern.matches(path))
}

fn first_title(text: &str, fallback: &str) -> String {
    let heading_re = Regex::new(r"(?m)^#\s+(.+?)\s*$").expect("heading regex");
    collapse(&heading_re
        .captures(text)
        .and_then(|caps| caps.get(1).map(|m| m.as_str().trim().to_string()))
        .unwrap_or_else(|| fallback.to_string()))
}

fn declared_generated(text: &str) -> bool {
    text.lines()
        .take(80)
        .any(|line| GENERATED_MARKERS.iter().any(|marker| line.contains(marker)))
}

fn is_pipeline_generated(
    path: &str,
    generated_declared: bool,
    generated_pattern: bool,
    toml_destination: &str,
    generated_allowlist: &HashSet<String>,
) -> bool {
    generated_pattern
        || (generated_declared
            && !toml_destination.is_empty()
            && IN_SCOPE_PREFIXES.iter().any(|prefix| path.starts_with(prefix)))
        || (generated_allowlist.contains(path) && generated_declared && !toml_destination.is_empty())
}

fn is_third_party(path: &str) -> bool {
    THIRD_PARTY_PATTERNS
        .iter()
        .filter_map(|pattern| Pattern::new(pattern).ok())
        .any(|pattern| pattern.matches(path))
}

fn classify_markdown(
    path: &str,
    generated: bool,
    third_party: bool,
    manual_exception: bool,
    toml_destination: &str,
) -> (String, String, String, String) {
    let high_information = path.starts_with("docs/")
        || path.starts_with("data/artifacts/")
        || path.starts_with("reports/")
        || path.starts_with("NAVIGATOR.md")
        || path.starts_with("REQUIREMENTS.md");
    if third_party {
        return (
            "third_party_markdown".to_string(),
            "ignore_vendor".to_string(),
            "none".to_string(),
            "Third-party or tool cache markdown; not a project knowledge source.".to_string(),
        );
    }
    if manual_exception {
        return (
            "manual_exception".to_string(),
            "keep_manual_exception".to_string(),
            "none".to_string(),
            "Explicitly retained manual entrypoint/readme exception.".to_string(),
        );
    }
    if generated && !toml_destination.is_empty() {
        return (
            "toml_published_markdown".to_string(),
            "keep_generated_mirror".to_string(),
            "low".to_string(),
            "Published markdown generated from TOML destination.".to_string(),
        );
    }
    if generated {
        return (
            "generated_artifact".to_string(),
            "keep_generated_artifact".to_string(),
            "low".to_string(),
            "Generated artifact markdown with no migration requirement.".to_string(),
        );
    }
    if !toml_destination.is_empty() {
        return (
            "toml_destination_exists_manual_markdown".to_string(),
            "port_body_to_toml_and_lock_mirror".to_string(),
            if high_information { "high" } else { "medium" }.to_string(),
            "Markdown has TOML destination but still carries manual content; lock to TOML flow."
                .to_string(),
        );
    }
    (
        "unbacked_manual_markdown".to_string(),
        "migrate_to_new_registry".to_string(),
        if high_information { "critical" } else { "high" }.to_string(),
        "Manual markdown without TOML destination; migrate aggressively.".to_string(),
    )
}

fn priority_rank(priority: &str) -> usize {
    match priority {
        "critical" => 0,
        "high" => 1,
        "medium" => 2,
        "low" => 3,
        "none" => 4,
        _ => 5,
    }
}

fn count_regex(pattern: &str, text: &str) -> Result<usize> {
    let re = Regex::new(pattern)?;
    Ok(re
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect::<HashSet<_>>()
        .len())
}

fn iter_registry_markdown_refs(repo_root: &Path) -> Result<HashMap<String, HashSet<String>>> {
    let mut refs = HashMap::<String, HashSet<String>>::new();
    for reg in fs::read_dir(repo_root.join("registry"))
        .with_context(|| format!("read {}", repo_root.join("registry").display()))?
    {
        let reg = reg?;
        let path = reg.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }
        let rel = repo_rel(repo_root, &path);
        if NON_DESTINATION_REGISTRIES.contains(&rel.as_str())
            || ARCHIVAL_NON_DESTINATION_REGISTRIES.contains(&rel.as_str())
        {
            continue;
        }
        let data = load_toml(&path)?;
        walk_markdown_refs(repo_root, &rel, &data, &mut refs);
    }
    Ok(refs)
}

fn walk_markdown_refs(
    repo_root: &Path,
    source: &str,
    value: &Value,
    refs: &mut HashMap<String, HashSet<String>>,
) {
    match value {
        Value::Table(table) => {
            for (key, entry) in table {
                let key = key.to_ascii_lowercase();
                match key.as_str() {
                    "source_markdown" | "markdown" | "output_markdown" | "path"
                    | "primary_markdown" | "generated_mirror" => {
                        add_markdown_ref(repo_root, source, entry, refs);
                    }
                    "source_markdown_glob" | "source_markdown_globs" => {
                        add_markdown_glob(repo_root, source, entry, refs);
                    }
                    _ => walk_markdown_refs(repo_root, source, entry, refs),
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                walk_markdown_refs(repo_root, source, item, refs);
            }
        }
        _ => {}
    }
}

fn add_markdown_ref(
    repo_root: &Path,
    source: &str,
    value: &Value,
    refs: &mut HashMap<String, HashSet<String>>,
) {
    match value {
        Value::String(item) => maybe_add_markdown_ref(repo_root, source, item, refs),
        Value::Array(items) => {
            for item in items {
                if let Some(text) = item.as_str() {
                    maybe_add_markdown_ref(repo_root, source, text, refs);
                }
            }
        }
        _ => {}
    }
}

fn maybe_add_markdown_ref(
    repo_root: &Path,
    source: &str,
    item: &str,
    refs: &mut HashMap<String, HashSet<String>>,
) {
    let path = normalize_path(item);
    if !path.ends_with(".md") {
        return;
    }
    if repo_root.join(&path).is_file() || path.ends_with(".md") {
        refs.entry(path).or_default().insert(source.to_string());
    }
}

fn add_markdown_glob(
    repo_root: &Path,
    source: &str,
    value: &Value,
    refs: &mut HashMap<String, HashSet<String>>,
) {
    let globs = match value {
        Value::String(item) => vec![item.to_string()],
        Value::Array(items) => items
            .iter()
            .filter_map(Value::as_str)
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    for pattern in globs {
        if let Ok(paths) = glob::glob(&repo_root.join(&pattern).to_string_lossy()) {
            for entry in paths.flatten() {
                if entry.is_file() {
                    let rel = repo_rel(repo_root, &entry);
                    if rel.ends_with(".md") {
                        refs.entry(rel).or_default().insert(source.to_string());
                    }
                }
            }
        }
    }
}

fn lifecycle_for_path(path: &str, row: &InventoryRow) -> String {
    if row.third_party {
        "third_party_cache".to_string()
    } else if path.starts_with("docs/generated/") {
        "generated_publish_mirror".to_string()
    } else if path.starts_with("docs/book/src/") {
        "generated_book_source".to_string()
    } else if path.starts_with("docs/convos/") {
        "generated_conversation_extract".to_string()
    } else if path.starts_with("docs/") {
        "generated_docs_tree".to_string()
    } else if path.starts_with("data/artifacts/") {
        "generated_artifact_report".to_string()
    } else if path.starts_with("reports/") {
        "generated_report_output".to_string()
    } else if matches!(path, "NAVIGATOR.md" | "REQUIREMENTS.md") {
        "generated_root_overlay".to_string()
    } else {
        "generated_other".to_string()
    }
}

fn risk_score(path: &str, row: &InventoryRow, destination_exists: bool, policy: &GovernancePolicy) -> i64 {
    let mut score = 0i64;
    if !policy.safe_classifications.contains(&row.classification) {
        score += 100;
    }
    if row.git_status == "tracked" && !policy.tracked_allowed_paths.contains(path) {
        score += 80;
    }
    if !row.generated && !row.third_party {
        score += 80;
    }
    if matches!(
        row.migration_action.as_str(),
        "migrate_to_new_registry" | "port_body_to_toml_and_lock_mirror"
    ) {
        score += 50;
    }
    if row.classification == "toml_published_markdown" && !destination_exists {
        score += 50;
    }
    score + (row.line_count / 250).min(20)
}

fn should_skip_toml_path(path: &str) -> bool {
    const SKIP_PREFIXES: &[&str] = &[
        ".cache/",
        ".horusec/",
        ".pytest_cache/",
        "target/",
        "venv/",
        ".venv/",
        ".mamba/",
    ];
    const SKIP_PARTS: &[&str] = &[
        ".cache",
        "cargo-home",
        "target",
        ".pytest_cache",
        "venv",
        ".venv",
        ".mamba",
    ];
    SKIP_PREFIXES.iter().any(|prefix| path.starts_with(prefix))
        || path.split('/').any(|part| SKIP_PARTS.contains(&part))
}

fn classify_toml_path(path: &str) -> (String, String) {
    if path.starts_with("registry/data/") {
        ("dataset_scroll".to_string(), "registry_dataset".to_string())
    } else if path.starts_with("registry/") {
        ("registry_control_plane".to_string(), "registry_control".to_string())
    } else if path == "Cargo.toml" {
        ("cargo_workspace_manifest".to_string(), "workspace_manifest".to_string())
    } else if path.ends_with("/Cargo.toml") {
        ("cargo_crate_manifest".to_string(), "crate_manifest".to_string())
    } else if path == ".cargo/config.toml" {
        ("cargo_toolchain_config".to_string(), "toolchain_config".to_string())
    } else if path == "pyproject.toml" {
        ("python_project_config".to_string(), "python_config".to_string())
    } else if path.starts_with("papers/") {
        ("papers_registry".to_string(), "papers".to_string())
    } else {
        ("toml_other".to_string(), "other".to_string())
    }
}

fn scan_toml_document(repo_root: &Path, path: &str, git_status: &str) -> Result<TomlInventoryRow> {
    let full = repo_root.join(path);
    let raw = fs::read(&full).with_context(|| format!("read {}", full.display()))?;
    let text = String::from_utf8_lossy(&raw).to_string();
    let (role, zone) = classify_toml_path(path);
    let parsed = toml::from_str::<Value>(&text);
    let (parse_ok, parse_error, table_count, markdown_ref_count, has_authoritative) = match parsed {
        Ok(value) => (
            true,
            String::new(),
            value.as_table().map(|table| table.len()).unwrap_or(0),
            count_markdown_refs_value(&value),
            has_authoritative_value(&value),
        ),
        Err(err) => (false, err.to_string(), 0usize, 0usize, false),
    };
    Ok(TomlInventoryRow {
        path: path.to_string(),
        git_status: git_status.to_string(),
        role,
        zone,
        parse_ok,
        parse_error,
        line_count: text.lines().count() + usize::from(!text.is_empty() && !text.ends_with('\n')),
        size_bytes: raw.len(),
        sha256: sha256_hex(text.as_bytes()),
        table_count,
        markdown_ref_count,
        has_authoritative,
    })
}

fn count_markdown_refs_value(value: &Value) -> usize {
    match value {
        Value::Table(table) => table
            .iter()
            .map(|(key, entry)| {
                let key = key.to_ascii_lowercase();
                let this = if matches!(
                    key.as_str(),
                    "source_markdown"
                        | "source_markdown_glob"
                        | "source_markdown_globs"
                        | "generated_mirror"
                        | "output_markdown"
                        | "markdown"
                        | "primary_markdown"
                ) {
                    match entry {
                        Value::String(item) => usize::from(item.contains(".md")),
                        Value::Array(items) => items
                            .iter()
                            .filter_map(Value::as_str)
                            .filter(|item| item.contains(".md"))
                            .count(),
                        _ => 0,
                    }
                } else {
                    0
                };
                this + count_markdown_refs_value(entry)
            })
            .sum(),
        Value::Array(items) => items.iter().map(count_markdown_refs_value).sum(),
        _ => 0,
    }
}

fn has_authoritative_value(value: &Value) -> bool {
    match value {
        Value::Table(table) => table.iter().any(|(key, entry)| {
            (key == "authoritative" && entry.as_bool() == Some(true))
                || has_authoritative_value(entry)
        }),
        Value::Array(items) => items.iter().any(has_authoritative_value),
        _ => false,
    }
}

fn collect_embedded_candidates(
    repo_root: &Path,
    policy: &GovernancePolicy,
) -> Result<BTreeMap<String, EmbeddedCandidate>> {
    let target_prefixes = governance_string_set(repo_root, "embedded_markdown_prefixes")?;
    let root_targets = governance_string_set(repo_root, "embedded_markdown_root_paths")?;
    let mut best_by_path = BTreeMap::<String, EmbeddedCandidate>::new();
    for reg in fs::read_dir(repo_root.join("registry"))
        .with_context(|| format!("read {}", repo_root.join("registry").display()))?
    {
        let reg = reg?;
        let path = reg.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }
        let rel_registry = repo_rel(repo_root, &path);
        let value = load_toml(&path)?;
        let docs = value
            .get("document")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for row in docs {
            let Some(table) = row.as_table() else {
                continue;
            };
            let body = string_field(table, "body_markdown");
            if body.trim().is_empty() {
                continue;
            }
            let candidate_path = pick_embedded_path(table);
            if !is_target_embedded_path(&candidate_path, &target_prefixes, &root_targets) {
                continue;
            }
            if should_skip_path(&candidate_path, policy) {
                continue;
            }
            let candidate = EmbeddedCandidate {
                body,
                source_registry: rel_registry.clone(),
                source_document_id: string_field(table, "id"),
                source_title: string_field(table, "title"),
            };
            match best_by_path.get(&candidate_path) {
                Some(existing) if existing.body.len() >= candidate.body.len() => {}
                _ => {
                    best_by_path.insert(candidate_path, candidate);
                }
            }
        }
    }
    Ok(best_by_path)
}

fn governance_string_set(repo_root: &Path, key: &str) -> Result<HashSet<String>> {
    let governance = load_toml(&repo_root.join("registry/markdown_governance.toml"))?;
    let policy = governance
        .get("policy")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    Ok(string_array_field(&policy, key).into_iter().collect())
}

fn pick_embedded_path(row: &toml::map::Map<String, Value>) -> String {
    ["source_markdown", "path", "markdown"]
        .iter()
        .find_map(|key| {
            let value = string_field(row, key);
            if value.ends_with(".md") {
                Some(value)
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn is_target_embedded_path(path: &str, prefixes: &HashSet<String>, roots: &HashSet<String>) -> bool {
    path.ends_with(".md")
        && (roots.contains(path) || prefixes.iter().any(|prefix| path.starts_with(prefix)))
}

fn bool_toml(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn load_toml(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn load_inventory_rows(value: &Value) -> Vec<InventoryRow> {
    value
        .get("document")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_table)
        .map(|row| InventoryRow {
            title: string_field(row, "title"),
            path: string_field(row, "path"),
            git_status: string_field(row, "git_status"),
            classification: string_field(row, "classification"),
            toml_destination: string_field(row, "toml_destination"),
            generated_declared: bool_field(row, "generated_declared"),
            generated_pattern: bool_field(row, "generated_pattern"),
            third_party: bool_field(row, "third_party"),
            generated: bool_field(row, "generated"),
            manual_exception: bool_field(row, "manual_exception"),
            size_bytes: integer_field(row, "size_bytes"),
            line_count: integer_field(row, "line_count"),
            sha256: string_field(row, "sha256"),
            claim_ref_count: integer_field(row, "claim_ref_count"),
            insight_ref_count: integer_field(row, "insight_ref_count"),
            experiment_ref_count: integer_field(row, "experiment_ref_count"),
            migration_action: string_field(row, "migration_action"),
            migration_priority: string_field(row, "migration_priority"),
            rationale: string_field(row, "rationale"),
            archived: bool_field(row, "archived"),
        })
        .collect()
}

fn load_knowledge_source_rows(value: &Value) -> Vec<KnowledgeSourceRow> {
    value
        .get("document")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_table)
        .map(|row| KnowledgeSourceRow {
            doc_id: string_field(row, "id"),
            path: string_field(row, "path"),
            title: string_field(row, "title"),
            kind: string_field(row, "kind"),
            authoring_mode: string_field(row, "authoring_mode"),
            generated: bool_field(row, "generated"),
            status: string_field(row, "status"),
            migration_priority: string_field(row, "migration_priority"),
            toml_backing: string_field(row, "toml_backing"),
            sha256: string_field(row, "sha256"),
            size_bytes: integer_field(row, "size_bytes").max(0) as usize,
            line_count: integer_field(row, "line_count").max(0) as usize,
            claim_ref_count: integer_field(row, "claim_ref_count").max(0) as usize,
            insight_ref_count: integer_field(row, "insight_ref_count").max(0) as usize,
            experiment_ref_count: integer_field(row, "experiment_ref_count").max(0) as usize,
            link_count: integer_field(row, "link_count").max(0) as usize,
            link_sample: string_array_field(row, "link_sample"),
        })
        .collect()
}

fn load_toml_inventory_rows(value: &Value) -> Vec<TomlInventoryRow> {
    value
        .get("document")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_table)
        .map(|row| TomlInventoryRow {
            path: string_field(row, "path"),
            git_status: string_field(row, "git_status"),
            role: string_field(row, "role"),
            zone: string_field(row, "zone"),
            parse_ok: bool_field(row, "parse_ok"),
            parse_error: string_field(row, "parse_error"),
            line_count: integer_field(row, "line_count").max(0) as usize,
            size_bytes: integer_field(row, "size_bytes").max(0) as usize,
            sha256: string_field(row, "sha256"),
            table_count: integer_field(row, "table_count").max(0) as usize,
            markdown_ref_count: integer_field(row, "markdown_ref_count").max(0) as usize,
            has_authoritative: bool_field(row, "has_authoritative"),
        })
        .collect()
}

fn load_origin_rows(value: &Value) -> Vec<OriginAuditRow> {
    value
        .get("document")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_table)
        .map(|row| OriginAuditRow {
            path: string_field(row, "path"),
            scope: string_field(row, "scope"),
            classification: string_field(row, "classification"),
            git_status: string_field(row, "git_status"),
            line_count: integer_field(row, "line_count"),
            toml_destination: string_field(row, "toml_destination"),
            destination_exists: bool_field(row, "destination_exists"),
            header_auto_generated: bool_field(row, "header_auto_generated"),
            header_source_of_truth: bool_field(row, "header_source_of_truth"),
            source_of_truth_raw: string_field(row, "source_of_truth_raw"),
            source_of_truth_paths: string_array_field(row, "source_of_truth_paths"),
            origin_process: string_field(row, "origin_process"),
            origin_status: string_field(row, "origin_status"),
            consolidation_action: string_field(row, "consolidation_action"),
        })
        .collect()
}

fn load_owner_rows(value: &Value) -> Vec<OwnerRow> {
    value
        .get("owner")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_table)
        .map(|row| OwnerRow {
            path: string_field(row, "path"),
            canonical_toml: string_field(row, "canonical_toml"),
            requires_generated_header: bool_field(row, "requires_generated_header"),
        })
        .collect()
}

fn load_governance_docs(value: &Value) -> Vec<GovernanceDoc> {
    value
        .get("document")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_table)
        .map(|row| GovernanceDoc {
            path: string_field(row, "path").replace('\\', "/"),
            mode: string_field(row, "mode"),
            header_required: bool_field(row, "header_required"),
            source_toml_refs: string_array_field(row, "source_toml_refs"),
        })
        .collect()
}

fn load_governance_policy(value: &Value) -> GovernancePolicy {
    let policy = value
        .get("policy")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let mut tracked_allowed_paths = string_array_field(&policy, "tracked_allowed_paths")
        .into_iter()
        .map(|value| normalize_path(&value))
        .collect::<HashSet<_>>();
    let tracked_allowed_modes = string_array_field(&policy, "tracked_allowed_modes")
        .into_iter()
        .collect::<HashSet<_>>();
    let governance_docs = load_governance_docs(value);
    for doc in governance_docs {
        if tracked_allowed_modes.contains(&doc.mode) {
            tracked_allowed_paths.insert(doc.path);
        }
    }
    let safe_classifications = string_array_field(&policy, "safe_classifications")
        .into_iter()
        .collect::<HashSet<_>>();
    let owner_scope_prefixes = string_array_field(&policy, "owner_scope_prefixes")
        .into_iter()
        .collect::<HashSet<_>>();
    let owner_scope_paths = string_array_field(&policy, "owner_scope_paths")
        .into_iter()
        .map(|value| normalize_path(&value))
        .collect::<HashSet<_>>();
    let skip_prefixes = {
        let configured = string_array_field(&policy, "skip_prefixes");
        if configured.is_empty() {
            DEFAULT_IGNORED_PREFIXES
                .iter()
                .map(|value| (*value).to_string())
                .collect()
        } else {
            configured
        }
    };
    let skip_path_parts = {
        let configured = string_array_field(&policy, "skip_path_parts");
        if configured.is_empty() {
            DEFAULT_IGNORED_PARTS
                .iter()
                .map(|value| (*value).to_string())
                .collect()
        } else {
            configured
        }
    }
    .into_iter()
    .collect::<HashSet<_>>();

    GovernancePolicy {
        safe_classifications,
        tracked_allowed_paths,
        owner_scope_prefixes,
        owner_scope_paths,
        skip_prefixes,
        skip_path_parts,
    }
}

fn discover_markdown_files(repo_root: &Path, policy: &GovernancePolicy) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let walker = WalkDir::new(repo_root)
        .into_iter()
        .filter_entry(|entry| keep_entry(repo_root, entry, policy));
    for entry in walker {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|ext| ext.to_str()) != Some("md") {
            continue;
        }
        let rel = repo_rel(repo_root, entry.path());
        if should_skip_path(&rel, policy) {
            continue;
        }
        out.push(rel);
    }
    out.sort();
    Ok(out)
}

fn git_paths(repo_root: &Path, args: &[&str], pattern: &str) -> Result<HashSet<String>> {
    let output = Command::new("git")
        .args(args)
        .arg("--")
        .arg(pattern)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(normalize_path)
        .filter(|line| !line.is_empty())
        .collect())
}

fn keep_entry(repo_root: &Path, entry: &DirEntry, policy: &GovernancePolicy) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    let rel = repo_rel(repo_root, entry.path());
    !should_skip_path(&rel, policy)
}

fn should_skip_path(path: &str, policy: &GovernancePolicy) -> bool {
    if policy
        .skip_prefixes
        .iter()
        .any(|prefix| path.starts_with(prefix))
    {
        return true;
    }
    path.split('/')
        .any(|part| policy.skip_path_parts.contains(part))
}

fn git_markdown_paths(repo_root: &Path, args: &[&str]) -> Result<HashSet<String>> {
    git_paths(repo_root, args, "*.md")
}

fn parse_markdown_units(text: &str, heading_re: &Regex, list_re: &Regex) -> Vec<MarkdownUnit> {
    let lines = ascii_clean(text)
        .lines()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let mut units = Vec::new();
    let mut paragraph_lines = Vec::<String>::new();
    let mut paragraph_start = 0usize;
    let mut in_code = false;
    let mut code_start = 0usize;
    let mut code_lines = Vec::<String>::new();

    let flush_paragraph = |units: &mut Vec<MarkdownUnit>,
                           paragraph_lines: &mut Vec<String>,
                           paragraph_start: &mut usize,
                           line_end: usize| {
        if paragraph_lines.is_empty() {
            return;
        }
        let payload = collapse(&paragraph_lines.join(" "));
        if !payload.is_empty() {
            units.push(MarkdownUnit {
                kind: "paragraph".to_string(),
                text_ascii: payload,
                line_start: *paragraph_start,
                line_end: (*paragraph_start).max(line_end),
                heading_level: 0,
            });
        }
        paragraph_lines.clear();
        *paragraph_start = 0;
    };

    for (idx, raw) in lines.iter().enumerate() {
        let line_no = idx + 1;
        let stripped = raw.trim();
        if in_code {
            if stripped.starts_with("```") {
                units.push(MarkdownUnit {
                    kind: "code_block".to_string(),
                    text_ascii: ascii_clean(&code_lines.join("\n")).trim_end().to_string(),
                    line_start: code_start,
                    line_end: line_no,
                    heading_level: 0,
                });
                in_code = false;
                code_start = 0;
                code_lines.clear();
            } else {
                code_lines.push(raw.clone());
            }
            continue;
        }
        if stripped.starts_with("```") {
            flush_paragraph(
                &mut units,
                &mut paragraph_lines,
                &mut paragraph_start,
                line_no - 1,
            );
            in_code = true;
            code_start = line_no;
            continue;
        }
        if let Some(caps) = heading_re.captures(stripped) {
            flush_paragraph(
                &mut units,
                &mut paragraph_lines,
                &mut paragraph_start,
                line_no - 1,
            );
            let level = caps.get(1).map(|m| m.as_str().len()).unwrap_or(0);
            let text_ascii = collapse(caps.get(2).map(|m| m.as_str()).unwrap_or(""));
            units.push(MarkdownUnit {
                kind: "heading".to_string(),
                text_ascii,
                line_start: line_no,
                line_end: line_no,
                heading_level: level,
            });
            continue;
        }
        if stripped.starts_with('|') && stripped.matches('|').count() >= 2 {
            flush_paragraph(
                &mut units,
                &mut paragraph_lines,
                &mut paragraph_start,
                line_no - 1,
            );
            if !is_table_separator(stripped) {
                let payload = collapse(
                    &stripped
                        .trim_matches('|')
                        .split('|')
                        .map(|cell| cell.trim())
                        .collect::<Vec<_>>()
                        .join(" | "),
                );
                if !payload.is_empty() {
                    units.push(MarkdownUnit {
                        kind: "table_row".to_string(),
                        text_ascii: payload,
                        line_start: line_no,
                        line_end: line_no,
                        heading_level: 0,
                    });
                }
            }
            continue;
        }
        if let Some(caps) = list_re.captures(stripped) {
            flush_paragraph(
                &mut units,
                &mut paragraph_lines,
                &mut paragraph_start,
                line_no - 1,
            );
            let payload = collapse(caps.get(1).map(|m| m.as_str()).unwrap_or(""));
            if !payload.is_empty() {
                units.push(MarkdownUnit {
                    kind: "list_item".to_string(),
                    text_ascii: payload,
                    line_start: line_no,
                    line_end: line_no,
                    heading_level: 0,
                });
            }
            continue;
        }
        if stripped.is_empty() {
            flush_paragraph(
                &mut units,
                &mut paragraph_lines,
                &mut paragraph_start,
                line_no - 1,
            );
            continue;
        }
        if paragraph_lines.is_empty() {
            paragraph_start = line_no;
        }
        paragraph_lines.push(raw.clone());
    }
    if in_code {
        units.push(MarkdownUnit {
            kind: "code_block".to_string(),
            text_ascii: ascii_clean(&code_lines.join("\n")).trim_end().to_string(),
            line_start: code_start,
            line_end: code_start.max(lines.len()),
            heading_level: 0,
        });
    }
    flush_paragraph(
        &mut units,
        &mut paragraph_lines,
        &mut paragraph_start,
        lines.len(),
    );
    units
}

fn is_table_separator(stripped: &str) -> bool {
    let core = stripped.trim_matches('|').trim();
    !core.is_empty() && core.chars().all(|ch| matches!(ch, '-' | ':' | ' '))
}

fn origin_class(path: &str, generated: bool, third_party: bool) -> String {
    let lowered = path.to_ascii_lowercase();
    if third_party
        || lowered.contains("/venv/")
        || lowered.contains("/site-packages/")
        || lowered.starts_with(".pytest_cache/")
        || lowered.contains("/.pytest_cache/")
    {
        "third_party_cache".to_string()
    } else if generated {
        "project_generated".to_string()
    } else {
        "project_manual".to_string()
    }
}

fn origin_process(path: &str) -> String {
    if path.starts_with("docs/generated/") {
        "cargo run -p gororoba_cli_data --bin registry-emit".to_string()
    } else {
        "cargo run -p gororoba_cli_data --bin markdown-registry".to_string()
    }
}

fn owner_group(path: &str, destination: &str) -> String {
    if destination == "agents.toml" {
        "agents_policy".to_string()
    } else if destination == "registry/monograph.toml" {
        "monograph".to_string()
    } else if path.starts_with("docs/book/src/") {
        "book_docs".to_string()
    } else if path.starts_with("docs/external_sources/") {
        "external_sources".to_string()
    } else if path.starts_with("docs/tickets/") {
        "claim_tickets".to_string()
    } else if path.starts_with("docs/claims/by_domain/") {
        "claims_domains".to_string()
    } else if path.starts_with("docs/convos/") {
        "docs_convos".to_string()
    } else if path.starts_with("docs/theory/")
        || path.starts_with("docs/engineering/")
        || path.starts_with("docs/research/")
    {
        "research_narratives".to_string()
    } else if path.starts_with("docs/") && destination == "registry/docs_root_narratives.toml" {
        "docs_root_narratives".to_string()
    } else if path.starts_with("reports/") {
        "reports_narratives".to_string()
    } else if path.starts_with("data/artifacts/") && path != "data/artifacts/README.md" {
        if destination == "registry/artifact_scrolls.toml" {
            "artifact_scrolls".to_string()
        } else {
            "data_artifact_narratives".to_string()
        }
    } else {
        "general".to_string()
    }
}

fn conversion_hint(path: &str, destination: &str) -> String {
    if path == "AGENTS.md" {
        "Run: cargo run -p gororoba_cli_data --bin agents-render -- --check".to_string()
    } else if destination == "registry/monograph.toml" {
        "Edit registry/monograph.toml and regenerate the published markdown mirror.".to_string()
    } else if path.starts_with("docs/")
        || path.starts_with("reports/")
        || path.starts_with("data/artifacts/")
    {
        format!(
            "Edit the canonical TOML source{} and regenerate markdown mirrors with the Rust registry pipeline.",
            if destination.is_empty() {
                "".to_string()
            } else {
                format!(" ({destination})")
            }
        )
    } else {
        "Assign canonical TOML ownership and regenerate via the Rust markdown registry pipeline."
            .to_string()
    }
}

fn knowledge_toml_backing_for_path(path: &str) -> String {
    match path {
        "AGENTS.md" => "agents.toml".to_string(),
        "CLAUDE.md" | "GEMINI.md" | "README.md" | "curated/README.md"
        | "curated/01_theory_frameworks/README_COQ.md" | "data/csv/README.md"
        | "data/artifacts/README.md" => "registry/entrypoint_docs.toml".to_string(),
        "NAVIGATOR.md" => "registry/navigator.toml".to_string(),
        "REQUIREMENTS.md" | "docs/REQUIREMENTS.md" => "registry/requirements.toml".to_string(),
        "docs/CLAIMS_EVIDENCE_MATRIX.md" | "docs/book/src/registry/claims.md" => {
            "registry/claims.toml".to_string()
        }
        "docs/BIBLIOGRAPHY.md" => "registry/bibliography.toml".to_string(),
        "docs/INSIGHTS.md" | "docs/book/src/registry/insights.md" => {
            "registry/insights.toml".to_string()
        }
        "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md"
        | "docs/book/src/registry/experiments.md" => "registry/experiments.toml".to_string(),
        "docs/generated/REPORTS_NARRATIVES_REGISTRY_MIRROR.md" => {
            "registry/reports_narratives.toml".to_string()
        }
        "docs/generated/DOCS_CONVOS_REGISTRY_MIRROR.md" => "registry/docs_convos.toml".to_string(),
        "docs/generated/DATA_ARTIFACT_NARRATIVES_REGISTRY_MIRROR.md" => {
            "registry/data_artifact_narratives.toml".to_string()
        }
        "data/artifacts/ALGEBRAIC_FOUNDATIONS.md"
        | "data/artifacts/BIBLIOGRAPHY.md"
        | "data/artifacts/FINAL_REPORT.md"
        | "data/artifacts/QUANTUM_REPORT.md"
        | "data/artifacts/SIMULATION_REPORT.md"
        | "data/artifacts/extracted_equations.md"
        | "data/artifacts/reality_check_and_synthesis.md" => {
            "registry/data_artifact_narratives.toml".to_string()
        }
        _ if path.starts_with("reports/") => "registry/reports_narratives.toml".to_string(),
        _ if path.starts_with("docs/convos/") => "registry/docs_convos.toml".to_string(),
        _ if path.starts_with("docs/") && path.matches('/').count() == 1 => {
            "registry/docs_root_narratives.toml".to_string()
        }
        _ if path.starts_with("docs/book/src/") => "registry/book_docs.toml".to_string(),
        _ if path.starts_with("docs/external_sources/") => "registry/external_sources.toml".to_string(),
        _ if path.starts_with("docs/theory/") || path.starts_with("docs/engineering/") => {
            "registry/research_narratives.toml".to_string()
        }
        _ => String::new(),
    }
}

fn knowledge_kind_for_path(path: &str, text: &str) -> (String, String, bool) {
    if IMMUTABLE_AGENT_OVERLAYS.contains(&path) {
        ("manual_source".to_string(), "manual".to_string(), false)
    } else if knowledge_toml_backing_for_path(path) == "registry/data_artifact_narratives.toml"
        && path.starts_with("data/artifacts/")
    {
        ("markdown_mirror".to_string(), "generated".to_string(), true)
    } else if path.starts_with("reports/")
        || path.starts_with("docs/convos/")
        || (path.starts_with("docs/") && path.matches('/').count() == 1)
        || path.starts_with("docs/book/src/")
        || path.starts_with("docs/theory/")
        || path.starts_with("docs/engineering/")
        || path.starts_with("docs/external_sources/")
        || !knowledge_toml_backing_for_path(path).is_empty()
    {
        ("markdown_mirror".to_string(), "generated".to_string(), true)
    } else if path.starts_with("convos/") {
        ("transcript_input".to_string(), "manual".to_string(), false)
    } else if path.starts_with("data/artifacts/") {
        ("artifact_report".to_string(), "generated".to_string(), true)
    } else if path.starts_with("docs/claims/by_domain/")
        || (path.starts_with("docs/tickets/") && path.ends_with("_claims_audit.md"))
        || path.starts_with("docs/book/src/registry/")
    {
        ("generated_markdown".to_string(), "generated".to_string(), true)
    } else if text.to_ascii_lowercase().contains("auto-generated")
        || text.to_ascii_lowercase().contains("do not edit")
    {
        ("generated_markdown".to_string(), "generated".to_string(), true)
    } else {
        ("manual_source".to_string(), "manual".to_string(), false)
    }
}

fn knowledge_status_for_path(path: &str) -> String {
    if path.starts_with("docs/archive/") || path.starts_with("archive/") {
        "archived".to_string()
    } else {
        "active".to_string()
    }
}

fn knowledge_migration_priority(kind: &str, path: &str) -> String {
    if kind == "markdown_mirror" {
        "critical".to_string()
    } else if kind == "manual_source" && path.starts_with("docs/") {
        "high".to_string()
    } else if kind == "manual_source" {
        "medium".to_string()
    } else {
        "none".to_string()
    }
}

fn extract_link_sample(text: &str) -> (Vec<String>, usize) {
    let backtick_re = Regex::new(r"`([^`\n]+)`").expect("backtick regex");
    let mut sample = BTreeSet::<String>::new();
    let mut raw_count = 0usize;
    for raw in backtick_re.captures_iter(text) {
        let token = raw.get(1).map(|m| m.as_str().trim()).unwrap_or_default();
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
        sample.insert(token.to_string());
    }
    (sample.into_iter().take(8).collect::<Vec<_>>(), raw_count)
}

fn knowledge_safe_classification(value: &str) -> bool {
    matches!(
        value,
        "toml_published_markdown"
            | "toml_destination_exists_manual_markdown"
            | "generated_artifact"
            | "third_party_markdown"
            | ""
    )
}

fn embedded_markdown_root_paths() -> Vec<String> {
    vec![
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        "README.md",
        "PANTHEON_PHYSICSFORGE_90_POINT_MIGRATION_PLAN.md",
        "PHASE10_11_ULTIMATE_ROADMAP.md",
        "PYTHON_REFACTORING_ROADMAP.md",
        "SYNTHESIS_PIPELINE_PROGRESS.md",
        "crates/sign_imbalance/IMPLEMENTATION_NOTES.md",
        "curated/README.md",
        "curated/01_theory_frameworks/README_COQ.md",
        "data/csv/README.md",
        "data/artifacts/README.md",
        "NAVIGATOR.md",
        "REQUIREMENTS.md",
        "docs/REQUIREMENTS.md",
    ]
    .into_iter()
    .map(|item| item.to_string())
    .collect()
}

fn owner_scope_paths() -> Vec<String> {
    vec![
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        "README.md",
        "apps/gororoba_studio/README.md",
        "crates/lbm_3d_cuda/README.md",
        "curated/README.md",
        "data/csv/README.md",
        "data/external/README.md",
        "proofs/EPISTEMIC_BOUNDARIES.md",
        "proofs/README.md",
    ]
    .into_iter()
    .map(|item| item.to_string())
    .collect()
}

fn governance_mode_for_path(
    path: &str,
    kind: &str,
    classification: &str,
    source_refs: &[String],
) -> (String, bool, String) {
    if classification == "third_party_markdown" {
        (
            "third_party_markdown".to_string(),
            false,
            "Third-party or cache markdown; allowed on disk but not authoritative.".to_string(),
        )
    } else if classification == "generated_artifact" {
        (
            "generated_artifact".to_string(),
            false,
            "Generated artifact/report; preserve reproducibility.".to_string(),
        )
    } else if IMMUTABLE_AGENT_OVERLAYS.contains(&path) {
        (
            "toml_manual_source".to_string(),
            false,
            "Manual compatibility stub; TOML pipelines must not rewrite this file.".to_string(),
        )
    } else if classification == "toml_published_markdown"
        || (path.starts_with("docs/") && path.matches('/').count() == 1)
        || generated_mirror_pattern(path)
    {
        (
            "toml_generated_mirror".to_string(),
            true,
            "Generated from TOML registries and overlays.".to_string(),
        )
    } else if kind == "transcript_input" {
        (
            "immutable_transcript".to_string(),
            false,
            "Immutable transcript input; not authoritative for claims.".to_string(),
        )
    } else if !source_refs.is_empty() {
        (
            "toml_manual_source".to_string(),
            false,
            "Manual source consumed by TOML normalizers.".to_string(),
        )
    } else if matches!(kind, "generated_markdown" | "artifact_report") {
        (
            "generated_artifact".to_string(),
            false,
            "Generated artifact/report; preserve reproducibility.".to_string(),
        )
    } else {
        (
            "manual_narrative".to_string(),
            false,
            "Manual narrative source; raw-captured in registry/knowledge/docs.".to_string(),
        )
    }
}

fn generated_mirror_pattern(path: &str) -> bool {
    [
        "AGENTS.md",
        "README.md",
        "curated/README.md",
        "curated/01_theory_frameworks/README_COQ.md",
        "data/csv/README.md",
        "data/artifacts/README.md",
        "NAVIGATOR.md",
        "docs/generated/*.md",
        "docs/CLAIMS_EVIDENCE_MATRIX.md",
        "docs/BIBLIOGRAPHY.md",
        "docs/INSIGHTS.md",
        "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md",
        "docs/ROADMAP.md",
        "docs/TODO.md",
        "docs/NEXT_ACTIONS.md",
        "docs/CLAIMS_TASKS.md",
        "docs/claims/INDEX.md",
        "docs/claims/by_domain/*.md",
        "docs/tickets/*.md",
        "docs/tickets/INDEX.md",
        "docs/book/src/*.md",
        "docs/book/src/*/*.md",
        "docs/book/src/*/*/*.md",
        "docs/external_sources/*.md",
        "docs/external_sources/INDEX.md",
        "docs/theory/*.md",
        "docs/theory/INDEX.md",
        "docs/engineering/*.md",
        "docs/engineering/INDEX.md",
        "data/artifacts/ALGEBRAIC_FOUNDATIONS.md",
        "data/artifacts/BIBLIOGRAPHY.md",
        "data/artifacts/FINAL_REPORT.md",
        "data/artifacts/QUANTUM_REPORT.md",
        "data/artifacts/SIMULATION_REPORT.md",
        "data/artifacts/extracted_equations.md",
        "data/artifacts/reality_check_and_synthesis.md",
        "reports/*.md",
        "docs/convos/*.md",
        "REQUIREMENTS.md",
        "docs/REQUIREMENTS.md",
        "docs/requirements/*.md",
    ]
    .into_iter()
    .filter_map(|pattern| Pattern::new(pattern).ok())
    .any(|pattern| pattern.matches(path))
}

fn knowledge_force_capture(path: &str) -> bool {
    KNOWLEDGE_FORCE_CAPTURE_EXACT.contains(&path)
        || KNOWLEDGE_FORCE_CAPTURE_PREFIXES
            .iter()
            .any(|prefix| path.starts_with(prefix))
        || KNOWLEDGE_FORCE_CAPTURE_SUFFIXES
            .iter()
            .any(|suffix| path.ends_with(suffix))
}

fn render_raw_capture_doc(source: &KnowledgeSourceRow, content: &str) -> String {
    let lines = vec![
        "# Markdown source migrated to TOML document store.".to_string(),
        "# This file is generated by cargo run -p gororoba_cli_data --bin markdown-registry -- migrate-corpus".to_string(),
        String::new(),
        "[document]".to_string(),
        format!("id = {}", q(&source.doc_id)),
        format!("source_path = {}", q(&source.path)),
        format!("title = {}", q(&source.title)),
        format!("kind = {}", q(&source.kind)),
        format!("status = {}", q(&source.status)),
        format!("migration_priority = {}", q(&source.migration_priority)),
        format!("generated = {}", bool_toml(source.generated)),
        format!("source_sha256 = {}", q(&source.sha256)),
        format!("source_line_count = {}", source.line_count),
        format!("source_size_bytes = {}", source.size_bytes),
        format!("content_sha256 = {}", q(&sha256_hex(content.as_bytes()))),
        "migrated_at = \"deterministic\"".to_string(),
        "capture_mode = \"raw_markdown_capture\"".to_string(),
        "authoritative = false".to_string(),
        "content_format = \"markdown\"".to_string(),
        format!("content_markdown = {}", toml_multiline_ascii(content)),
        String::new(),
    ];
    lines.join("\n")
}

fn render_raw_capture_manifest(
    ingested: &[(KnowledgeSourceRow, String)],
    skipped: &[KnowledgeSourceRow],
) -> String {
    let mut lines = vec![
        "# Central document manifest for TOML-backed markdown knowledge corpus.".to_string(),
        "# Generated by cargo run -p gororoba_cli_data --bin markdown-registry -- migrate-corpus".to_string(),
        "# This is a raw capture layer, not a normalized authoritative schema.".to_string(),
        String::new(),
        "[knowledge_documents]".to_string(),
        "generated_at = \"deterministic\"".to_string(),
        format!("raw_capture_count = {}", ingested.len()),
        format!("skipped_generated_count = {}", skipped.len()),
        "authoritative = false".to_string(),
        "normalization_required = true".to_string(),
        String::new(),
    ];
    for (source, out_rel) in ingested {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {}", q(&source.doc_id)));
        lines.push(format!("source_path = {}", q(&source.path)));
        lines.push(format!("title = {}", q(&source.title)));
        lines.push(format!("kind = {}", q(&source.kind)));
        lines.push(format!(
            "migration_priority = {}",
            q(&source.migration_priority)
        ));
        lines.push("raw_captured = true".to_string());
        lines.push("authoritative = false".to_string());
        lines.push("normalization_status = \"pending\"".to_string());
        lines.push(format!("document_toml = {}", q(out_rel)));
        lines.push(String::new());
    }
    for source in skipped {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {}", q(&source.doc_id)));
        lines.push(format!("source_path = {}", q(&source.path)));
        lines.push(format!("title = {}", q(&source.title)));
        lines.push(format!("kind = {}", q(&source.kind)));
        lines.push("raw_captured = false".to_string());
        lines.push("authoritative = false".to_string());
        lines.push("skip_reason = \"generated_markdown_or_artifact\"".to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

fn parse_overlay_sections(
    source_path: &Path,
    heading_re: &Regex,
) -> Result<(String, Vec<(String, String, String)>)> {
    let raw_lines = fs::read_to_string(source_path)
        .with_context(|| format!("read {}", source_path.display()))?
        .lines()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let lines = strip_generated_header_lines(raw_lines);
    let mut preamble = Vec::<String>::new();
    let mut entries = Vec::<(String, String, Vec<String>)>::new();
    let mut current_id = String::new();
    let mut current_title = String::new();
    let mut current_body = Vec::<String>::new();
    let mut seen_first = false;

    for line in lines {
        if let Some(caps) = heading_re.captures(&line) {
            if !current_id.is_empty() {
                entries.push((current_id.clone(), current_title.clone(), current_body.clone()));
            }
            current_id = caps.get(1).map(|m| m.as_str()).unwrap_or_default().to_string();
            current_title = caps.get(2).map(|m| m.as_str().trim()).unwrap_or_default().to_string();
            current_body.clear();
            seen_first = true;
            continue;
        }
        if !seen_first {
            preamble.push(line);
        } else {
            current_body.push(line);
        }
    }
    if !current_id.is_empty() {
        entries.push((current_id, current_title, current_body));
    }

    Ok((
        preamble.join("\n").trim().to_string(),
        entries
            .into_iter()
            .map(|(id, title, body)| (id, title, clean_entry_body(&body.join("\n"))))
            .collect(),
    ))
}

fn strip_generated_header_lines(lines: Vec<String>) -> Vec<String> {
    let mut idx = 0usize;
    while idx < lines.len() {
        let stripped = lines[idx].trim();
        if stripped.is_empty() {
            idx += 1;
            continue;
        }
        if GENERATED_MARKERS.iter().any(|prefix| stripped.contains(prefix)) {
            idx += 1;
            continue;
        }
        if stripped.starts_with("<!--") && stripped.ends_with("-->") {
            idx += 1;
            continue;
        }
        break;
    }
    lines.into_iter().skip(idx).collect()
}

fn clean_entry_body(text: &str) -> String {
    let mut lines = text.lines().map(ToString::to_string).collect::<Vec<_>>();
    while lines.first().map(|line| line.trim().is_empty()).unwrap_or(false) {
        lines.remove(0);
    }
    while lines.first().map(|line| line.trim() == "---").unwrap_or(false) {
        lines.remove(0);
        while lines.first().map(|line| line.trim().is_empty()).unwrap_or(false) {
            lines.remove(0);
        }
    }
    let mut out = lines
        .into_iter()
        .filter(|line| line.trim() != "---")
        .collect::<Vec<_>>();
    while out.last().map(|line| line.trim().is_empty()).unwrap_or(false) {
        out.pop();
    }
    out.join("\n").trim().to_string()
}

fn render_narrative_overlay(
    title: &str,
    subtitle: &str,
    section: &str,
    source_markdown: &str,
    preamble: &str,
    entries: &[(String, String, String)],
) -> String {
    let mut lines = vec![
        format!("# {title}"),
        format!("# {subtitle}"),
        String::new(),
        format!("[{section}]"),
        "authoritative = true".to_string(),
        "updated = \"2026-02-09\"".to_string(),
        format!("source_markdown = {}", q(source_markdown)),
        format!("entry_count = {}", entries.len()),
        format!("preamble_markdown = {}", toml_multiline_ascii(preamble)),
        String::new(),
    ];
    for (id, entry_title, body) in entries {
        lines.push("[[entry]]".to_string());
        lines.push(format!("id = {}", q(id)));
        lines.push(format!("title = {}", q(entry_title)));
        lines.push(format!("body_markdown = {}", toml_multiline_ascii(body)));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn read_overlay_markdown(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(strip_generated_header_lines(
        raw.lines().map(ToString::to_string).collect::<Vec<_>>(),
    )
    .join("\n")
    .trim()
    .to_string())
}

fn render_single_overlay(title: &str, section: &str, source_markdown: &str, body: &str) -> String {
    vec![
        format!("# {title}"),
        String::new(),
        format!("[{section}]"),
        "authoritative = true".to_string(),
        "updated = \"2026-02-09\"".to_string(),
        format!("source_markdown = {}", q(source_markdown)),
        format!("body_markdown = {}", toml_multiline_ascii(body.trim())),
        String::new(),
    ]
    .join("\n")
}

fn render_requirements_overlay(documents: &[(String, String, String)]) -> String {
    let mut lines = vec![
        "# Requirements narrative overlay registry (TOML-first).".to_string(),
        String::new(),
        "[requirements_narrative]".to_string(),
        "authoritative = true".to_string(),
        "updated = \"2026-02-09\"".to_string(),
        "source_markdown_glob = \"docs/requirements/*.md\"".to_string(),
        format!("document_count = {}", documents.len()),
        String::new(),
    ];
    for (path, title, body) in documents {
        lines.push("[[document]]".to_string());
        lines.push(format!("path = {}", q(path)));
        lines.push(format!("title = {}", q(title)));
        lines.push(format!("body_markdown = {}", toml_multiline_ascii(body.trim())));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn title_from_markdown(path: &Path, body: &str) -> String {
    body.lines()
        .find_map(|line| line.strip_prefix("# ").map(|value| value.trim().to_string()))
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or_default()
                .to_string()
        })
}

fn scope_for_path(path: &str) -> String {
    if path.starts_with("docs/") {
        "docs".to_string()
    } else if path.starts_with("reports/") {
        "reports".to_string()
    } else if path.starts_with("data/artifacts/") {
        "data_artifacts".to_string()
    } else {
        "root".to_string()
    }
}

fn q(value: &str) -> String {
    json!(value).to_string()
}

fn q_list(values: &[String]) -> String {
    let rendered = values.iter().map(|value| q(value)).collect::<Vec<_>>();
    format!("[{}]", rendered.join(", "))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn ascii_clean(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        let code = ch as u32;
        if matches!(ch, '\n' | '\r' | '\t') {
            out.push(ch);
        } else if code < 32 {
            out.push(' ');
        } else if code <= 127 {
            out.push(ch);
        } else if code <= 0xFFFF {
            out.push_str(&format!("\\u{code:04X}"));
        } else {
            out.push_str(&format!("\\U{code:08X}"));
        }
    }
    out
}

fn collapse(text: &str) -> String {
    ascii_clean(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn toml_multiline_ascii(content: &str) -> String {
    let mut out = String::new();
    for ch in content.chars() {
        let code = ch as u32;
        if ch == '\\' {
            out.push_str("\\\\");
        } else if ch == '"' {
            out.push_str("\\\"");
        } else if ch == '\t' {
            out.push_str("\\t");
        } else if ch == '\r' {
            out.push_str("\\r");
        } else if ch == '\n' {
            out.push('\n');
        } else if code < 32 {
            out.push_str(&format!("\\u{code:04X}"));
        } else if code <= 127 {
            out.push(ch);
        } else if code <= 0xFFFF {
            out.push_str(&format!("\\u{code:04X}"));
        } else {
            out.push_str(&format!("\\U{code:08X}"));
        }
    }
    format!("\"\"\"\n{}\n\"\"\"", out)
}

fn assert_ascii(text: &str, context: &str) -> Result<()> {
    if let Some(ch) = text.chars().find(|ch| (*ch as u32) > 127) {
        bail!("non-ASCII content in {}: {:?}", context, ch);
    }
    Ok(())
}

fn write_ascii(path: &Path, text: &str) -> Result<()> {
    let bad = text.chars().find(|ch| (*ch as u32) > 127);
    if let Some(ch) = bad {
        bail!("non-ASCII output in {}: {:?}", path.display(), ch);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn string_field(table: &toml::map::Map<String, Value>, key: &str) -> String {
    table
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .replace('\\', "/")
}

fn string_array_field(table: &toml::map::Map<String, Value>, key: &str) -> Vec<String> {
    table
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(normalize_path)
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn bool_field(table: &toml::map::Map<String, Value>, key: &str) -> bool {
    table.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn integer_field(table: &toml::map::Map<String, Value>, key: &str) -> i64 {
    table.get(key).and_then(Value::as_integer).unwrap_or(0)
}

fn normalize_path(value: &str) -> String {
    value.trim().replace('\\', "/")
}

fn repo_path(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn repo_rel(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
