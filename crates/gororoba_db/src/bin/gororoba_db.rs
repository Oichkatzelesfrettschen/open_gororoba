//! `gororoba-db` — unified Rust-native CLI for the SQLite source-of-truth.
//!
//! This binary consolidates all database interaction into a single
//! entrypoint: schema introspection, statistics, querying, import/export
//! of knowledge-base and planning tables, full-text search, audit of
//! legacy TOML files, and notebook-session management for evcxr/Jupyter
//! integration.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use provenance_store::ProvenanceStore;
use std::{
    fs,
    path::{Path, PathBuf},
};
use toml::Value;

// ─── CLI definition ────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "gororoba-db",
    about = "Rust-native CLI for the SQLite source-of-truth database",
    long_about = "Unified entrypoint for querying, importing, exporting, auditing, \
                  and managing the canonical SQLite control-plane database. \
                  Replaces scattered TOML-file projections with direct database interaction."
)]
struct Cli {
    /// Repository root.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// SQLite database path.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Show database statistics: table row counts, migration status, and source-of-truth manifest.
    Stats,

    /// Print full schema introspection (tables, columns, row counts).
    Schema,

    /// Import knowledge-base TOML files (equation atoms, proofs, derivations) into SQLite.
    ImportKnowledge(ImportKnowledgeArgs),

    /// Import planning TOML files (roadmap, todo, next-actions) into SQLite.
    ImportPlanning(ImportPlanningArgs),

    /// Import research narrative TOML into SQLite.
    ImportNarratives(ImportNarrativesArgs),

    /// Export planning tables to TOML-compatible output (stdout or file).
    ExportPlanning(ExportPlanningArgs),

    /// Full-text search across research narratives.
    Search(SearchArgs),

    /// Query rows from any table by name with optional status filter.
    Query(QueryArgs),

    /// Audit: compare database source-of-truth against legacy TOML files.
    Audit,

    /// Show legacy TOML files that should be archived.
    ArchiveLegacy,

    /// Show evcxr/Jupyter notebook integration status and capabilities.
    NotebookInfo,

    /// List or manage notebook sessions stored in the database.
    Notebooks(NotebookArgs),
}

#[derive(Parser, Debug)]
struct ImportKnowledgeArgs {
    /// Path to equation atoms TOML (v3 preferred).
    #[arg(long, default_value = "registry/knowledge/equation_atoms_v3.toml")]
    equation_atoms: PathBuf,

    /// Path to derivation steps TOML.
    #[arg(long, default_value = "registry/knowledge/derivation_steps.toml")]
    derivation_steps: PathBuf,

    /// Path to proof skeletons TOML.
    #[arg(long, default_value = "registry/knowledge/proof_skeletons.toml")]
    proof_skeletons: PathBuf,
}

#[derive(Parser, Debug)]
struct ImportPlanningArgs {
    /// Path to roadmap TOML.
    #[arg(long, default_value = "registry/roadmap.toml")]
    roadmap: PathBuf,

    /// Path to todo TOML.
    #[arg(long, default_value = "registry/todo.toml")]
    todo: PathBuf,

    /// Path to next-actions TOML.
    #[arg(long, default_value = "registry/next_actions.toml")]
    next_actions: PathBuf,
}

#[derive(Parser, Debug)]
struct ImportNarrativesArgs {
    /// Path to research narratives TOML.
    #[arg(long, default_value = "registry/research_narratives.toml")]
    narratives: PathBuf,
}

#[derive(Parser, Debug)]
struct ExportPlanningArgs {
    /// Output format.
    #[arg(long, default_value = "toml")]
    format: OutputFormat,

    /// Optional output file (defaults to stdout).
    #[arg(long)]
    out: Option<PathBuf>,

    /// Which planning table to export.
    #[arg(long, default_value = "roadmap")]
    table: PlanningTable,
}

#[derive(Clone, Debug, ValueEnum)]
enum OutputFormat {
    Toml,
    Json,
    Text,
}

#[derive(Clone, Debug, ValueEnum)]
enum PlanningTable {
    Roadmap,
    Todo,
    NextActions,
}

#[derive(Parser, Debug)]
struct SearchArgs {
    /// Search query (FTS5 syntax supported).
    query: String,

    /// Maximum results.
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

#[derive(Parser, Debug)]
struct QueryArgs {
    /// Table name to query.
    table: String,

    /// Optional status filter.
    #[arg(long)]
    status: Option<String>,

    /// Maximum rows.
    #[arg(long, default_value_t = 50)]
    limit: usize,
}

#[derive(Parser, Debug)]
struct NotebookArgs {
    #[command(subcommand)]
    action: NotebookAction,
}

#[derive(Subcommand, Debug)]
enum NotebookAction {
    /// List existing notebook sessions.
    List,
    /// Create a new notebook session.
    Create {
        /// Session title.
        #[arg(long)]
        title: String,
        /// Optional description.
        #[arg(long, default_value = "")]
        description: String,
    },
}

// ─── Main ──────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    let db_path = cli.repo_root.join(&cli.db);
    let store =
        ProvenanceStore::open(&db_path).with_context(|| format!("open database {}", db_path.display()))?;

    match cli.command {
        Commands::Stats => cmd_stats(&store),
        Commands::Schema => cmd_schema(&store),
        Commands::ImportKnowledge(args) => cmd_import_knowledge(&store, &cli.repo_root, &args),
        Commands::ImportPlanning(args) => cmd_import_planning(&store, &cli.repo_root, &args),
        Commands::ImportNarratives(args) => cmd_import_narratives(&store, &cli.repo_root, &args),
        Commands::ExportPlanning(args) => cmd_export_planning(&store, &args),
        Commands::Search(args) => cmd_search(&store, &args),
        Commands::Query(args) => cmd_query(&store, &args),
        Commands::Audit => cmd_audit(&store, &cli.repo_root),
        Commands::ArchiveLegacy => cmd_archive_legacy(&cli.repo_root),
        Commands::NotebookInfo => cmd_notebook_info(),
        Commands::Notebooks(args) => cmd_notebooks(&store, &args),
    }
}

// ─── Stats ─────────────────────────────────────────────────────────

fn cmd_stats(store: &ProvenanceStore) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        gororoba-db  •  Source-of-Truth Statistics           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let manifest = store.source_of_truth_stats()?;
    let mut current_cat = String::new();
    let mut total_rows: i64 = 0;

    for (table, category, count, meta) in &manifest {
        if *category != current_cat {
            if !current_cat.is_empty() {
                println!();
            }
            println!("  ── {} ──", category.to_uppercase());
            current_cat.clone_from(category);
        }
        let parts: Vec<&str> = meta.splitn(2, '|').collect();
        let status = parts.get(1).unwrap_or(&"");
        println!("    {:<35} {:>8} rows  [{}]", table, count, status);
        total_rows += count;
    }

    println!();
    println!("  Total rows across all tables: {total_rows}");
    println!();
    Ok(())
}

// ─── Schema ────────────────────────────────────────────────────────

fn cmd_schema(store: &ProvenanceStore) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        gororoba-db  •  Schema Introspection                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let tables = store.schema_summary()?;
    for (name, count, cols) in &tables {
        println!("  TABLE {name}  ({count} rows)");
        for col in cols {
            println!("    • {col}");
        }
        println!();
    }

    println!("  {} tables total", tables.len());
    Ok(())
}

// ─── Import knowledge ──────────────────────────────────────────────

fn cmd_import_knowledge(
    store: &ProvenanceStore,
    repo_root: &Path,
    args: &ImportKnowledgeArgs,
) -> Result<()> {
    println!("Importing knowledge base into SQLite...");

    // Import equation atoms
    let eq_path = repo_root.join(&args.equation_atoms);
    if eq_path.exists() {
        let text = fs::read_to_string(&eq_path)
            .with_context(|| format!("read {}", eq_path.display()))?;
        let val: Value = toml::from_str(&text)
            .with_context(|| format!("parse {}", eq_path.display()))?;
        let mut count = 0u64;
        if let Some(atoms) = val.get("atom").and_then(|v| v.as_array()) {
            for atom in atoms {
                let id = atom.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let expression = atom.get("expression").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let normalized = atom.get("normalized_expression").and_then(|v| v.as_str()).unwrap_or("");
                    let relation = atom.get("relation_operator").and_then(|v| v.as_str()).unwrap_or("implicit");
                    let kind = atom.get("equation_kind").and_then(|v| v.as_str()).unwrap_or("");
                    let confidence = atom.get("extraction_confidence").and_then(|v| v.as_str()).unwrap_or("medium");
                    let domain = atom.get("domain_applicability").and_then(|v| v.as_str()).unwrap_or("");
                    let source_uid = atom.get("source_uid").and_then(|v| v.as_str()).unwrap_or("");
                    let source_path = atom.get("source_path").and_then(|v| v.as_str()).unwrap_or("");
                    let section = atom.get("section_title").and_then(|v| v.as_str()).unwrap_or("");
                    let assumptions = json_array_field(atom, "assumptions");
                    let derivation_links = json_array_field(atom, "derivation_links");
                    let depends_on = json_array_field(atom, "depends_on_equations");
                    let sweep = atom.get("parameter_sweep").map(|v| serde_json::to_string(v).unwrap_or_default()).unwrap_or_default();

                    store.conn_exec(
                        "INSERT OR REPLACE INTO equation_atoms
                         (id, expression, normalized_expression, relation_operator, equation_kind,
                          extraction_confidence, domain_applicability, source_uid, source_path,
                          section_title, assumptions_json, parameter_sweep_json,
                          derivation_links_json, depends_on_json)
                         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
                        [id, expression, normalized, relation, kind, confidence,
                          domain, source_uid, source_path, section,
                          &assumptions, &sweep, &derivation_links, &depends_on],
                    )?;
                    count += 1;
                }
            }
        }
        println!("  ✓ Imported {count} equation atoms from {}", eq_path.display());
    } else {
        println!("  ⚠ Equation atoms file not found: {}", eq_path.display());
    }

    // Import proof skeletons
    let ps_path = repo_root.join(&args.proof_skeletons);
    if ps_path.exists() {
        let text = fs::read_to_string(&ps_path)
            .with_context(|| format!("read {}", ps_path.display()))?;
        let val: Value = toml::from_str(&text)
            .with_context(|| format!("parse {}", ps_path.display()))?;
        let mut count = 0u64;
        if let Some(skels) = val.get("skeleton").and_then(|v| v.as_array()) {
            for skel in skels {
                let id = skel.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let kind = skel.get("skeleton_kind").and_then(|v| v.as_str()).unwrap_or("");
                    let source_path = skel.get("source_path").and_then(|v| v.as_str()).unwrap_or("");
                    let source_uid = skel.get("source_uid").and_then(|v| v.as_str()).unwrap_or("");
                    let claim_id = skel.get("claim_id").and_then(|v| v.as_str()).unwrap_or("");
                    let claim_refs = json_array_field(skel, "claim_refs");
                    let title = skel.get("title").and_then(|v| v.as_str()).unwrap_or("");
                    let status = skel.get("status").and_then(|v| v.as_str()).unwrap_or("draft");
                    let step_count = skel.get("step_count").and_then(|v| v.as_integer()).unwrap_or(0);

                    store.conn_exec(
                        "INSERT OR REPLACE INTO proof_skeletons
                         (id, skeleton_kind, source_path, source_uid, claim_id,
                          claim_refs_json, title, status, step_count)
                         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                        [id, kind, source_path, source_uid, claim_id,
                          &claim_refs, title, status, &step_count.to_string()],
                    )?;
                    count += 1;
                }
            }
        }
        println!("  ✓ Imported {count} proof skeletons from {}", ps_path.display());
    } else {
        println!("  ⚠ Proof skeletons file not found: {}", ps_path.display());
    }

    // Import derivation steps
    let ds_path = repo_root.join(&args.derivation_steps);
    if ds_path.exists() {
        let text = fs::read_to_string(&ds_path)
            .with_context(|| format!("read {}", ds_path.display()))?;
        let val: Value = toml::from_str(&text)
            .with_context(|| format!("parse {}", ds_path.display()))?;
        let mut count = 0u64;
        if let Some(steps) = val.get("step").and_then(|v| v.as_array()) {
            for step in steps {
                let id = step.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let skeleton_id = step.get("skeleton_id").and_then(|v| v.as_str()).unwrap_or("");
                    let skeleton_kind = step.get("skeleton_kind").and_then(|v| v.as_str()).unwrap_or("");
                    let source_path = step.get("source_path").and_then(|v| v.as_str()).unwrap_or("");
                    let source_uid = step.get("source_uid").and_then(|v| v.as_str()).unwrap_or("");
                    let claim_id = step.get("claim_id").and_then(|v| v.as_str()).unwrap_or("");
                    let claim_refs = json_array_field(step, "claim_refs");
                    let step_index = step.get("step_index").and_then(|v| v.as_integer()).unwrap_or(0);
                    let step_kind = step.get("step_kind").and_then(|v| v.as_str()).unwrap_or("derivation_step");
                    let text_val = step.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    let text_sha256 = step.get("text_sha256").and_then(|v| v.as_str()).unwrap_or("");
                    let equation_refs = json_array_field(step, "equation_refs");
                    let symbol_refs = json_array_field(step, "symbol_refs");
                    let numeric_constants = json_array_field(step, "numeric_constants");
                    let key_tokens = json_array_field(step, "key_tokens");
                    let depends_on = json_array_field(step, "depends_on_step_ids");
                    let line_start = step.get("line_start").and_then(|v| v.as_integer()).unwrap_or(0);
                    let line_end = step.get("line_end").and_then(|v| v.as_integer()).unwrap_or(0);

                    store.conn_exec(
                        "INSERT OR REPLACE INTO derivation_steps
                         (id, skeleton_id, skeleton_kind, source_path, source_uid, claim_id,
                          claim_refs_json, step_index, step_kind, text, text_sha256,
                          equation_refs_json, symbol_refs_json, numeric_constants_json,
                          key_tokens_json, depends_on_step_ids_json, line_start, line_end)
                         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18)",
                        [id, skeleton_id, skeleton_kind, source_path, source_uid, claim_id,
                          &claim_refs, &step_index.to_string(), step_kind, text_val,
                          text_sha256, &equation_refs, &symbol_refs, &numeric_constants,
                          &key_tokens, &depends_on, &line_start.to_string(), &line_end.to_string()],
                    )?;
                    count += 1;
                }
            }
        }
        println!("  ✓ Imported {count} derivation steps from {}", ds_path.display());
    } else {
        println!("  ⚠ Derivation steps file not found: {}", ds_path.display());
    }

    println!("Knowledge import complete.");
    Ok(())
}

// ─── Import planning ───────────────────────────────────────────────

fn cmd_import_planning(
    store: &ProvenanceStore,
    repo_root: &Path,
    args: &ImportPlanningArgs,
) -> Result<()> {
    println!("Importing planning data into SQLite...");

    // Import roadmap
    let rm_path = repo_root.join(&args.roadmap);
    if rm_path.exists() {
        let text = fs::read_to_string(&rm_path)
            .with_context(|| format!("read {}", rm_path.display()))?;
        let val: Value = toml::from_str(&text)
            .with_context(|| format!("parse {}", rm_path.display()))?;
        let mut count = 0u64;
        if let Some(items) = val.get("workstream").and_then(|v| v.as_array()) {
            for item in items {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let deps = json_array_field(item, "dependencies");
                    let ac = json_array_field(item, "acceptance_criteria");
                    let po = json_array_field(item, "primary_outputs");
                    let er = json_array_field(item, "evidence_refs");
                    let lac = json_array_field(item, "lacunae");
                    store.upsert_roadmap_item(&provenance_store::RoadmapItem {
                        id,
                        name: item.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        priority: item.get("priority").and_then(|v| v.as_str()).unwrap_or("medium"),
                        status: item.get("status").and_then(|v| v.as_str()).unwrap_or("planned"),
                        status_token: item.get("status_token").and_then(|v| v.as_str()).unwrap_or("PLANNED"),
                        description: item.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                        sprint: item.get("sprint").and_then(|v| v.as_str()).unwrap_or(""),
                        dependencies_json: &deps,
                        acceptance_criteria_json: &ac,
                        primary_outputs_json: &po,
                        evidence_refs_json: &er,
                        lacunae_json: &lac,
                    })?;
                    count += 1;
                }
            }
        }
        println!("  ✓ Imported {count} roadmap workstreams from {}", rm_path.display());
    } else {
        println!("  ⚠ Roadmap file not found: {}", rm_path.display());
    }

    // Import todo
    let td_path = repo_root.join(&args.todo);
    if td_path.exists() {
        let text = fs::read_to_string(&td_path)
            .with_context(|| format!("read {}", td_path.display()))?;
        let val: Value = toml::from_str(&text)
            .with_context(|| format!("parse {}", td_path.display()))?;
        let mut count = 0u64;
        if let Some(items) = val.get("item").and_then(|v| v.as_array()) {
            for item in items {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let deps = json_array_field(item, "dependencies");
                    let ac = json_array_field(item, "acceptance_criteria");
                    store.upsert_todo_item(&provenance_store::ActionItem {
                        id,
                        area: item.get("area").and_then(|v| v.as_str()).unwrap_or(""),
                        title: item.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                        description: item.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                        priority: item.get("priority").and_then(|v| v.as_str()).unwrap_or("medium"),
                        status: item.get("status").and_then(|v| v.as_str()).unwrap_or("open"),
                        status_token: item.get("status_token").and_then(|v| v.as_str()).unwrap_or("OPEN"),
                        dependencies_json: &deps,
                        acceptance_criteria_json: &ac,
                    })?;
                    count += 1;
                }
            }
        }
        println!("  ✓ Imported {count} todo items from {}", td_path.display());
    } else {
        println!("  ⚠ Todo file not found: {}", td_path.display());
    }

    // Import next actions
    let na_path = repo_root.join(&args.next_actions);
    if na_path.exists() {
        let text = fs::read_to_string(&na_path)
            .with_context(|| format!("read {}", na_path.display()))?;
        let val: Value = toml::from_str(&text)
            .with_context(|| format!("parse {}", na_path.display()))?;
        let mut count = 0u64;
        if let Some(items) = val.get("action").and_then(|v| v.as_array()) {
            for item in items {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let deps = json_array_field(item, "dependencies");
                    let ac = json_array_field(item, "acceptance_criteria");
                    store.upsert_next_action(&provenance_store::ActionItem {
                        id,
                        area: item.get("area").and_then(|v| v.as_str()).unwrap_or(""),
                        title: item.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                        description: item.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                        priority: item.get("priority").and_then(|v| v.as_str()).unwrap_or("medium"),
                        status: item.get("status").and_then(|v| v.as_str()).unwrap_or("open"),
                        status_token: item.get("status_token").and_then(|v| v.as_str()).unwrap_or("OPEN"),
                        dependencies_json: &deps,
                        acceptance_criteria_json: &ac,
                    })?;
                    count += 1;
                }
            }
        }
        println!("  ✓ Imported {count} next actions from {}", na_path.display());
    } else {
        println!("  ⚠ Next actions file not found: {}", na_path.display());
    }

    println!("Planning import complete.");
    Ok(())
}

// ─── Import narratives ─────────────────────────────────────────────

fn cmd_import_narratives(
    store: &ProvenanceStore,
    repo_root: &Path,
    args: &ImportNarrativesArgs,
) -> Result<()> {
    println!("Importing research narratives into SQLite...");

    let path = repo_root.join(&args.narratives);
    if !path.exists() {
        println!("  ⚠ Narratives file not found: {}", path.display());
        return Ok(());
    }

    let text = fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    let val: Value = toml::from_str(&text)
        .with_context(|| format!("parse {}", path.display()))?;
    let mut count = 0u64;

    if let Some(docs) = val.get("document").and_then(|v| v.as_array()) {
        for doc in docs {
            let id = doc.get("id").and_then(|v| v.as_str()).unwrap_or("");
            if !id.is_empty() {
                let cr = json_array_field(doc, "claim_refs");
                let ur = json_array_field(doc, "url_refs");
                let pr = json_array_field(doc, "path_refs");
                store.upsert_research_narrative(&provenance_store::ResearchNarrativeRow {
                    id,
                    source_markdown: doc.get("source_markdown").and_then(|v| v.as_str()).unwrap_or(""),
                    domain: doc.get("domain").and_then(|v| v.as_str()).unwrap_or(""),
                    slug: doc.get("slug").and_then(|v| v.as_str()).unwrap_or(""),
                    title: doc.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                    status_token: doc.get("status_token").and_then(|v| v.as_str()).unwrap_or("NARRATIVE"),
                    content_kind: doc.get("content_kind").and_then(|v| v.as_str()).unwrap_or("research_note"),
                    verification_level: doc.get("verification_level").and_then(|v| v.as_str()).unwrap_or(""),
                    claim_refs_json: &cr,
                    url_refs_json: &ur,
                    path_refs_json: &pr,
                    body_markdown: doc.get("body_markdown").and_then(|v| v.as_str()).unwrap_or(""),
                    line_count: doc.get("line_count").and_then(|v| v.as_integer()).unwrap_or(0),
                })?;
                count += 1;
            }
        }
    }

    println!("  ✓ Imported {count} research narratives from {}", path.display());
    println!("Narrative import complete.");
    Ok(())
}

// ─── Export planning ───────────────────────────────────────────────

fn cmd_export_planning(store: &ProvenanceStore, args: &ExportPlanningArgs) -> Result<()> {
    let (label, items) = match args.table {
        PlanningTable::Roadmap => {
            let items = store.list_roadmap_items(None)?;
            ("roadmap", items)
        }
        PlanningTable::Todo => {
            let items = store.list_todo_items(None)?;
            ("todo", items)
        }
        PlanningTable::NextActions => {
            let items = store.list_next_actions(None)?;
            ("next_actions", items)
        }
    };

    let output = match args.format {
        OutputFormat::Json => {
            let entries: Vec<serde_json::Value> = items
                .iter()
                .map(|(id, name, priority, status)| {
                    serde_json::json!({
                        "id": id,
                        "name": name,
                        "priority": priority,
                        "status": status,
                    })
                })
                .collect();
            serde_json::to_string_pretty(&entries)?
        }
        OutputFormat::Toml => {
            let mut lines = vec![format!("# Exported from SQLite source-of-truth ({label})")];
            lines.push(format!("# Item count: {}", items.len()));
            lines.push(String::new());
            for (id, name, priority, status) in &items {
                lines.push("[[item]]".to_string());
                lines.push(format!("id = {}", toml_quote(id)));
                lines.push(format!("name = {}", toml_quote(name)));
                lines.push(format!("priority = {}", toml_quote(priority)));
                lines.push(format!("status = {}", toml_quote(status)));
                lines.push(String::new());
            }
            lines.join("\n")
        }
        OutputFormat::Text => {
            let mut lines = vec![format!("{label} ({} items):", items.len())];
            for (id, name, priority, status) in &items {
                lines.push(format!("  {id:<30} {priority:<8} {status:<12} {name}"));
            }
            lines.join("\n")
        }
    };

    if let Some(out_path) = &args.out {
        fs::write(out_path, format!("{output}\n"))
            .with_context(|| format!("write {}", out_path.display()))?;
        println!("Exported to {}", out_path.display());
    } else {
        println!("{output}");
    }
    Ok(())
}

// ─── Search ────────────────────────────────────────────────────────

fn cmd_search(store: &ProvenanceStore, args: &SearchArgs) -> Result<()> {
    let results = store.search_narratives(&args.query, args.limit)?;
    if results.is_empty() {
        println!("No results for query: {}", args.query);
        return Ok(());
    }
    println!("Search results for \"{}\" ({} hits):", args.query, results.len());
    for (id, title, rank) in &results {
        println!("  {id:<12} (rank: {rank:.4})  {title}");
    }
    Ok(())
}

// ─── Query ─────────────────────────────────────────────────────────

fn cmd_query(store: &ProvenanceStore, args: &QueryArgs) -> Result<()> {
    // Delegate to the appropriate typed list method where available
    match args.table.as_str() {
        "roadmap_items" | "roadmap" => {
            let items = store.list_roadmap_items(args.status.as_deref())?;
            println!("roadmap_items ({} rows):", items.len());
            for (id, name, priority, status) in items.iter().take(args.limit) {
                println!("  {id:<30} {priority:<8} {status:<12} {name}");
            }
        }
        "todo_items" | "todo" => {
            let items = store.list_todo_items(args.status.as_deref())?;
            println!("todo_items ({} rows):", items.len());
            for (id, title, priority, status) in items.iter().take(args.limit) {
                println!("  {id:<30} {priority:<8} {status:<12} {title}");
            }
        }
        "next_action_items" | "next_actions" => {
            let items = store.list_next_actions(args.status.as_deref())?;
            println!("next_action_items ({} rows):", items.len());
            for (id, title, priority, status) in items.iter().take(args.limit) {
                println!("  {id:<30} {priority:<8} {status:<12} {title}");
            }
        }
        "notebook_sessions" | "notebooks" => {
            let items = store.list_notebook_sessions()?;
            println!("notebook_sessions ({} rows):", items.len());
            for s in items.iter().take(args.limit) {
                println!("  {:<20} [{}] {:<10} cells={}  {}", s.id, s.kernel, s.status, s.cell_count, s.title);
            }
        }
        other => {
            // Generic table query
            let count = store.table_row_count(other)?;
            println!("{other} ({count} rows)");
            println!("  (Use a specific table name for detailed output.)");
            println!("  Supported tables: roadmap_items, todo_items, next_action_items, notebook_sessions");
        }
    }
    Ok(())
}

// ─── Audit ─────────────────────────────────────────────────────────

fn cmd_audit(store: &ProvenanceStore, repo_root: &Path) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        gororoba-db  •  Source-of-Truth Audit               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let manifest = store.source_of_truth_manifest()?;

    println!("  ── AUTHORITATIVE (database is source of truth) ──");
    println!();
    for row in &manifest {
        if row.authoritative {
            let (table, category, legacy, migration_status) = (&row.table_name, &row.category, &row.legacy_toml_path, &row.migration_status);
            let toml_exists = if legacy.is_empty() {
                "n/a".to_string()
            } else {
                let p = repo_root.join(legacy);
                if p.exists() { "✓ exists".to_string() } else { "✗ missing".to_string() }
            };
            let count = store.table_row_count(table).unwrap_or(0);
            println!(
                "    {table:<35} [{category}] {count:>6} rows  legacy={toml_exists:<12} status={migration_status}"
            );
        }
    }

    println!();
    println!("  ── ANALYSIS ──");
    println!();
    println!("  Items that MUST be in the database (authoritative source of truth):");
    println!("    • claims, insights, experiments, binaries, theorems (control plane)");
    println!("    • artifacts, documents, citations (provenance index)");
    println!("    • download_jobs, download_attempts, download_campaigns (pipeline)");
    println!("    • external_source_contracts, external_source_dossiers (governance)");
    println!("    • equation_atoms, proof_skeletons, derivation_steps (knowledge)");
    println!("    • roadmap_items, todo_items, next_action_items (planning)");
    println!("    • research_narratives (narrative documents)");
    println!("    • notebook_sessions (interactive analysis)");
    println!();
    println!("  Items that MUST NOT be in the database:");
    println!("    • Raw data files (CSV, HDF5, FITS) → remain on filesystem under data/");
    println!("    • Compiled artifacts (target/, *.vo) → build outputs, never versioned");
    println!("    • Credentials, secrets, API keys → .env files, never committed");
    println!("    • Binary executables → compiled from source, not stored");
    println!();
    println!("  Legacy items for archive/:");
    println!("    • Wave 4-6 phase plans → already in archive/registry/wave_phase_plans/");
    println!("    • Pantheon/PhysicsForge migration → already in archive/registry/pantheon_physicsforge/");
    println!("    • 8086 legacy CSVs → already in archive/8086_legacy/");
    println!("    • Superseded registry TOMLs → candidates for archive once DB migration complete");
    println!();
    println!("  Legitimate items handled differently:");
    println!("    • Formal proofs (*.v, *.lean) → remain in proofs/ with DB references");
    println!("    • LaTeX sources → remain in docs/latex/ with DB cross-references");
    println!("    • Makefile targets → orchestration layer, references DB commands");
    println!("    • Python scripts → legacy generators, being replaced by Rust binaries");
    println!();
    Ok(())
}

// ─── Archive legacy ────────────────────────────────────────────────

fn cmd_archive_legacy(repo_root: &Path) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        gororoba-db  •  Legacy Archive Candidates           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // TOML files that are auto-generated read-only compatibility exports
    let compat_exports = [
        "registry/claims.toml",
        "registry/experiments.toml",
        "registry/insights.toml",
        "registry/external_sources.toml",
    ];

    println!("  ── Auto-generated TOML exports (read-only, generated from DB) ──");
    println!("  These are compatibility layers; the database is the source of truth.");
    println!("  They can be regenerated at any time and do not need to be committed.");
    println!();
    for path in &compat_exports {
        let full = repo_root.join(path);
        let status = if full.exists() { "present" } else { "absent" };
        println!("    {path:<55} [{status}]");
    }

    println!();
    println!("  ── Superseded planning TOMLs (candidates for DB-only) ──");
    println!("  Once import-planning has been run, these become read-only exports.");
    println!();
    let planning_tomls = [
        "registry/roadmap.toml",
        "registry/todo.toml",
        "registry/next_actions.toml",
        "registry/research_narratives.toml",
    ];
    for path in &planning_tomls {
        let full = repo_root.join(path);
        let status = if full.exists() { "present" } else { "absent" };
        println!("    {path:<55} [{status}]");
    }

    println!();
    println!("  ── Already archived ──");
    let archive_dirs = [
        "archive/registry/pantheon_physicsforge/",
        "archive/registry/wave_phase_plans/",
        "archive/8086_legacy/",
        "archive/external_legacy_placeholders/",
        "archive/external_nonreproducible_snapshots/",
    ];
    for path in &archive_dirs {
        let full = repo_root.join(path);
        let status = if full.exists() { "✓" } else { "✗" };
        println!("    {status} {path}");
    }

    println!();
    println!("  ── Python scripts being replaced by Rust binaries ──");
    let legacy_scripts = [
        "src/scripts/analysis/build_wave5_batch4_registries.py",
        "src/scripts/analysis/build_registry_execution_planning.py",
    ];
    for path in &legacy_scripts {
        let full = repo_root.join(path);
        let status = if full.exists() { "present (legacy)" } else { "absent" };
        println!("    {path:<55} [{status}]");
    }

    println!();
    Ok(())
}

// ─── Notebook info ─────────────────────────────────────────────────

fn cmd_notebook_info() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        gororoba-db  •  Notebook Integration Status         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Rust Jupyter Notebook Support: evcxr_jupyter");
    println!();
    println!("  ── What is evcxr? ──");
    println!("  evcxr is an evaluation context for Rust that provides:");
    println!("    • A Jupyter kernel for interactive Rust notebooks");
    println!("    • A standalone Rust REPL");
    println!("    • Rich display support (HTML, images, plots)");
    println!("    • On-the-fly crate dependency management via :dep");
    println!();
    println!("  ── Installation ──");
    println!("    cargo install --locked evcxr_jupyter");
    println!("    evcxr_jupyter --install");
    println!("    jupyter notebook  # then select 'Rust' kernel");
    println!();
    println!("  ── Integration with open_gororoba ──");
    println!("  In an evcxr notebook, you can load workspace crates:");
    println!("    :dep provenance_store = {{ path = \"crates/provenance_store\" }}");
    println!("    :dep provenance_core = {{ path = \"crates/provenance_core\" }}");
    println!("    :dep cd_kernel = {{ path = \"crates/cd_kernel\" }}");
    println!();
    println!("  Then interact with the database directly:");
    println!("    use provenance_store::ProvenanceStore;");
    println!("    let store = ProvenanceStore::open(");
    println!("        std::path::Path::new(\"registry/canonical/control_plane.sqlite3\")");
    println!("    ).unwrap();");
    println!("    let stats = store.source_of_truth_stats().unwrap();");
    println!();
    println!("  ── Notebook sessions ──");
    println!("  The database tracks notebook sessions in the `notebook_sessions` table.");
    println!("  Use `gororoba-db notebooks list` to view sessions.");
    println!("  Use `gororoba-db notebooks create --title <name>` to create one.");
    println!();
    println!("  ── Capabilities ──");
    println!("  With evcxr + open_gororoba crates you can:");
    println!("    • Query and visualize claims, experiments, insights");
    println!("    • Run Cayley-Dickson algebra computations interactively");
    println!("    • Prototype data analysis pipelines");
    println!("    • Generate plots with the plotters crate");
    println!("    • Inspect provenance and download audit trails");
    println!();
    Ok(())
}

// ─── Notebooks ─────────────────────────────────────────────────────

fn cmd_notebooks(store: &ProvenanceStore, args: &NotebookArgs) -> Result<()> {
    match &args.action {
        NotebookAction::List => {
            let sessions = store.list_notebook_sessions()?;
            if sessions.is_empty() {
                println!("No notebook sessions found. Create one with:");
                println!("  gororoba-db notebooks create --title \"My Analysis\"");
                return Ok(());
            }
            println!("Notebook sessions ({}):", sessions.len());
            for s in &sessions {
                println!("  {:<20} [{}] {:<10} cells={}  {}", s.id, s.kernel, s.status, s.cell_count, s.title);
            }
        }
        NotebookAction::Create { title, description } => {
            let id = format!(
                "NB-{:04}",
                store.table_row_count("notebook_sessions").unwrap_or(0) + 1
            );
            store.upsert_notebook_session(&provenance_store::NotebookSessionRow {
                id: &id,
                title,
                description,
                kernel: "evcxr",
                status: "draft",
                cell_count: 0,
                cells_json: "[]",
            })?;
            println!("Created notebook session: {id} — \"{title}\"");
        }
    }
    Ok(())
}

// ─── Helpers ───────────────────────────────────────────────────────

/// Extract a TOML array field as a JSON array string.
fn json_array_field(val: &Value, key: &str) -> String {
    match val.get(key) {
        Some(Value::Array(arr)) => {
            let items: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| format!("\"{}\"", s.replace('"', "\\\""))))
                .collect();
            format!("[{}]", items.join(","))
        }
        _ => "[]".to_string(),
    }
}

/// Quote a string for TOML output.
fn toml_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
