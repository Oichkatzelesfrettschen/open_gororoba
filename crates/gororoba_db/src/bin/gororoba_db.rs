//! `gororoba-db` -- unified Rust-native CLI for the SQLite source-of-truth.
//!
//! This binary consolidates all database interaction into a single
//! entrypoint: schema introspection, statistics, querying, import/export
//! of knowledge-base, planning, and requirements tables, full-text search, audit of
//! legacy TOML compatibility layer, and notebook-session management for evcxr/Jupyter
//! integration.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use provenance_store::ProvenanceStore;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use toml::Value;

// ─── CLI definition ────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "gororoba-db",
    about = "Three-layer registry CLI: SQLite source -> compatibility exports -> query",
    long_about = "Unified entrypoint for building, querying, and auditing the registry.\n\n\
                  Layer 1 (Canonical Source): .cache/registry.sqlite3.\n\
                  Layer 2 (Compatibility): registry/*.toml (legacy, generated/validated).\n\
                  Layer 3 (Query):  This CLI."
)]
struct Cli {
    /// Repository root.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// SQLite database path.
    #[arg(long, default_value = ".cache/registry.sqlite3")]
    db: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Rebuild .cache/registry.sqlite3 from compatibility inputs and refresh compatibility artifacts.
    Build(BuildArgs),

    /// Show database statistics: table row counts, migration status, and source-of-truth manifest.
    Stats,

    /// Print full schema introspection (tables, columns, row counts).
    Schema,

    /// List, show, or search claims.
    Claims(ClaimsArgs),

    /// List or search insights.
    Insights(InsightsArgs),

    /// List experiments.
    Experiments(ExperimentsArgs),

    /// Full-text search across claims, insights, narratives, and bibliography.
    Search(SearchArgs),

    /// Cross-reference queries (dangling refs, unlinked claims, coverage).
    Xref(XrefArgs),

    /// Audit: verify signatures, crossrefs, and labels.
    AuditCmd(AuditArgs),

    /// Import knowledge-base TOML files (equation atoms, proofs, derivations) into SQLite.
    ImportKnowledge(ImportKnowledgeArgs),

    /// Import planning TOML files (roadmap, todo, next-actions) into SQLite.
    ImportPlanning(ImportPlanningArgs),

    /// Import requirements TOML into SQLite.
    ImportRequirements(ImportRequirementsArgs),

    /// Import research narrative TOML into SQLite.
    ImportNarratives(ImportNarrativesArgs),

    /// Export planning tables to TOML-compatible output (stdout or file).
    ExportPlanning(ExportPlanningArgs),

    /// Export requirements tables to TOML-compatible output (stdout or file).
    ExportRequirements(ExportRequirementsArgs),

    /// Query rows from any table by name with optional status filter.
    Query(QueryArgs),

    /// Show legacy TOML files that should be archived.
    ArchiveLegacy,

    /// Show evcxr/Jupyter notebook integration status and capabilities.
    NotebookInfo,

    /// List or manage notebook sessions stored in the database.
    Notebooks(NotebookArgs),
}

#[derive(Parser, Debug)]
struct BuildArgs {
    /// Verify crossrefs and signatures after building (exit non-zero on errors).
    #[arg(long)]
    verify: bool,
}

#[derive(Parser, Debug)]
struct ClaimsArgs {
    #[command(subcommand)]
    action: ClaimsAction,
}

#[derive(Subcommand, Debug)]
enum ClaimsAction {
    /// List claims with optional status filter.
    List {
        #[arg(long)]
        status: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Show a single claim by ID.
    Show { id: String },
    /// Full-text search claims.
    Search {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// List claims with no linked experiments or insights.
    Unlinked,
}

#[derive(Parser, Debug)]
struct InsightsArgs {
    #[command(subcommand)]
    action: InsightsAction,
}

#[derive(Subcommand, Debug)]
enum InsightsAction {
    /// List insights.
    List {
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Full-text search insights.
    Search {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
}

#[derive(Parser, Debug)]
struct ExperimentsArgs {
    #[command(subcommand)]
    action: ExperimentsAction,
}

#[derive(Subcommand, Debug)]
enum ExperimentsAction {
    /// List experiments with optional status filter.
    List {
        #[arg(long)]
        status: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
}

#[derive(Parser, Debug)]
struct XrefArgs {
    #[command(subcommand)]
    action: XrefAction,
}

#[derive(Subcommand, Debug)]
enum XrefAction {
    /// Find dangling crossrefs (references to non-existent claims).
    Dangling,
    /// Find claims with no linked experiments or insights.
    Unlinked,
    /// Show crossref coverage summary.
    Coverage,
}

#[derive(Parser, Debug)]
struct AuditArgs {
    #[command(subcommand)]
    action: AuditAction,
}

#[derive(Subcommand, Debug)]
enum AuditAction {
    /// Verify schema signatures match TOML content hashes.
    Signatures,
    /// Check for dangling crossrefs.
    Crossrefs,
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
struct ImportRequirementsArgs {
    /// Path to requirements TOML.
    #[arg(long, default_value = "registry/requirements.toml")]
    requirements: PathBuf,
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
enum RequirementsOutputFormat {
    Toml,
    Json,
    Text,
}

#[derive(Parser, Debug)]
struct ExportRequirementsArgs {
    /// Output format.
    #[arg(long, default_value = "toml")]
    format: RequirementsOutputFormat,

    /// Optional output file (defaults to stdout).
    #[arg(long)]
    out: Option<PathBuf>,
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

    // Build command creates DB from scratch; all others open existing.
    if let Commands::Build(ref args) = cli.command {
        return cmd_build(&cli.repo_root, &db_path, args);
    }

    let mut store = ProvenanceStore::open(&db_path)
        .with_context(|| format!("open database {}", db_path.display()))?;

    match cli.command {
        Commands::Build(_) => unreachable!(),
        Commands::Stats => cmd_stats(&store),
        Commands::Schema => cmd_schema(&store),
        Commands::Claims(args) => cmd_claims(&store, &args),
        Commands::Insights(args) => cmd_insights(&store, &args),
        Commands::Experiments(args) => cmd_experiments(&store, &args),
        Commands::Search(args) => cmd_search(&store, &args),
        Commands::Xref(args) => cmd_xref(&store, &args),
        Commands::AuditCmd(args) => cmd_audit_cmd(&store, &cli.repo_root, &args),
        Commands::ImportKnowledge(args) => cmd_import_knowledge(&store, &cli.repo_root, &args),
        Commands::ImportPlanning(args) => cmd_import_planning(&mut store, &cli.repo_root, &args),
        Commands::ImportRequirements(args) => {
            cmd_import_requirements(&mut store, &cli.repo_root, &args)
        }
        Commands::ImportNarratives(args) => cmd_import_narratives(&store, &cli.repo_root, &args),
        Commands::ExportPlanning(args) => cmd_export_planning(&store, &args),
        Commands::ExportRequirements(args) => cmd_export_requirements(&store, &args),
        Commands::Query(args) => cmd_query(&store, &args),
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
        let text =
            fs::read_to_string(&eq_path).with_context(|| format!("read {}", eq_path.display()))?;
        let val: Value =
            toml::from_str(&text).with_context(|| format!("parse {}", eq_path.display()))?;
        let mut count = 0u64;
        if let Some(atoms) = val.get("atom").and_then(|v| v.as_array()) {
            for atom in atoms {
                let id = atom.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let expression = atom
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !id.is_empty() {
                    let normalized = atom
                        .get("normalized_expression")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let relation = atom
                        .get("relation_operator")
                        .and_then(|v| v.as_str())
                        .unwrap_or("implicit");
                    let kind = atom
                        .get("equation_kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let confidence = atom
                        .get("extraction_confidence")
                        .and_then(|v| v.as_str())
                        .unwrap_or("medium");
                    let domain = atom
                        .get("domain_applicability")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let source_uid = atom
                        .get("source_uid")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let source_path = atom
                        .get("source_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let section = atom
                        .get("section_title")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let assumptions = json_array_field(atom, "assumptions");
                    let derivation_links = json_array_field(atom, "derivation_links");
                    let depends_on = json_array_field(atom, "depends_on_equations");
                    let sweep = atom
                        .get("parameter_sweep")
                        .map(|v| serde_json::to_string(v).unwrap_or_default())
                        .unwrap_or_default();

                    store.conn_exec(
                        "INSERT OR REPLACE INTO equation_atoms
                         (id, expression, normalized_expression, relation_operator, equation_kind,
                          extraction_confidence, domain_applicability, source_uid, source_path,
                          section_title, assumptions_json, parameter_sweep_json,
                          derivation_links_json, depends_on_json)
                         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
                        [
                            id,
                            expression,
                            normalized,
                            relation,
                            kind,
                            confidence,
                            domain,
                            source_uid,
                            source_path,
                            section,
                            &assumptions,
                            &sweep,
                            &derivation_links,
                            &depends_on,
                        ],
                    )?;
                    count += 1;
                }
            }
        }
        println!(
            "  <EMOJI+2713> Imported {count} equation atoms from {}",
            eq_path.display()
        );
    } else {
        println!(
            "  <EMOJI+26A0> Equation atoms file not found: {}",
            eq_path.display()
        );
    }

    // Import proof skeletons
    let ps_path = repo_root.join(&args.proof_skeletons);
    if ps_path.exists() {
        let text =
            fs::read_to_string(&ps_path).with_context(|| format!("read {}", ps_path.display()))?;
        let val: Value =
            toml::from_str(&text).with_context(|| format!("parse {}", ps_path.display()))?;
        let mut count = 0u64;
        if let Some(skels) = val.get("skeleton").and_then(|v| v.as_array()) {
            for skel in skels {
                let id = skel.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let kind = skel
                        .get("skeleton_kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let source_path = skel
                        .get("source_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let source_uid = skel
                        .get("source_uid")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let claim_id = skel.get("claim_id").and_then(|v| v.as_str()).unwrap_or("");
                    let claim_refs = json_array_field(skel, "claim_refs");
                    let title = skel.get("title").and_then(|v| v.as_str()).unwrap_or("");
                    let status = skel
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("draft");
                    let step_count = skel
                        .get("step_count")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0);

                    store.conn_exec(
                        "INSERT OR REPLACE INTO proof_skeletons
                         (id, skeleton_kind, source_path, source_uid, claim_id,
                          claim_refs_json, title, status, step_count)
                         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                        [
                            id,
                            kind,
                            source_path,
                            source_uid,
                            claim_id,
                            &claim_refs,
                            title,
                            status,
                            &step_count.to_string(),
                        ],
                    )?;
                    count += 1;
                }
            }
        }
        println!(
            "  <EMOJI+2713> Imported {count} proof skeletons from {}",
            ps_path.display()
        );
    } else {
        println!(
            "  <EMOJI+26A0> Proof skeletons file not found: {}",
            ps_path.display()
        );
    }

    // Import derivation steps
    let ds_path = repo_root.join(&args.derivation_steps);
    if ds_path.exists() {
        let text =
            fs::read_to_string(&ds_path).with_context(|| format!("read {}", ds_path.display()))?;
        let val: Value =
            toml::from_str(&text).with_context(|| format!("parse {}", ds_path.display()))?;
        let mut count = 0u64;
        if let Some(steps) = val.get("step").and_then(|v| v.as_array()) {
            for step in steps {
                let id = step.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let skeleton_id = step
                        .get("skeleton_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let skeleton_kind = step
                        .get("skeleton_kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let source_path = step
                        .get("source_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let source_uid = step
                        .get("source_uid")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let claim_id = step.get("claim_id").and_then(|v| v.as_str()).unwrap_or("");
                    let claim_refs = json_array_field(step, "claim_refs");
                    let step_index = step
                        .get("step_index")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0);
                    let step_kind = step
                        .get("step_kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("derivation_step");
                    let text_val = step.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    let text_sha256 = step
                        .get("text_sha256")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let equation_refs = json_array_field(step, "equation_refs");
                    let symbol_refs = json_array_field(step, "symbol_refs");
                    let numeric_constants = json_array_field(step, "numeric_constants");
                    let key_tokens = json_array_field(step, "key_tokens");
                    let depends_on = json_array_field(step, "depends_on_step_ids");
                    let line_start = step
                        .get("line_start")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0);
                    let line_end = step
                        .get("line_end")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0);

                    store.conn_exec(
                        "INSERT OR REPLACE INTO derivation_steps
                         (id, skeleton_id, skeleton_kind, source_path, source_uid, claim_id,
                          claim_refs_json, step_index, step_kind, text, text_sha256,
                          equation_refs_json, symbol_refs_json, numeric_constants_json,
                          key_tokens_json, depends_on_step_ids_json, line_start, line_end)
                         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18)",
                        [
                            id,
                            skeleton_id,
                            skeleton_kind,
                            source_path,
                            source_uid,
                            claim_id,
                            &claim_refs,
                            &step_index.to_string(),
                            step_kind,
                            text_val,
                            text_sha256,
                            &equation_refs,
                            &symbol_refs,
                            &numeric_constants,
                            &key_tokens,
                            &depends_on,
                            &line_start.to_string(),
                            &line_end.to_string(),
                        ],
                    )?;
                    count += 1;
                }
            }
        }
        println!(
            "  <EMOJI+2713> Imported {count} derivation steps from {}",
            ds_path.display()
        );
    } else {
        println!(
            "  <EMOJI+26A0> Derivation steps file not found: {}",
            ds_path.display()
        );
    }

    println!("Knowledge import complete.");
    Ok(())
}

// ─── Import planning ───────────────────────────────────────────────

fn cmd_import_planning(
    store: &mut ProvenanceStore,
    repo_root: &Path,
    args: &ImportPlanningArgs,
) -> Result<()> {
    println!("Importing planning data into SQLite...");

    // Import roadmap
    let rm_path = repo_root.join(&args.roadmap);
    if rm_path.exists() {
        let text =
            fs::read_to_string(&rm_path).with_context(|| format!("read {}", rm_path.display()))?;
        let val: Value =
            toml::from_str(&text).with_context(|| format!("parse {}", rm_path.display()))?;
        store.record_registry_snapshot(repo_root, "roadmap", &rm_path, &text)?;
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
                    let claims = json_array_field(item, "claims");
                    store.upsert_roadmap_item(&provenance_store::RoadmapItem {
                        id,
                        name: item.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        priority: item
                            .get("priority")
                            .and_then(|v| v.as_str())
                            .unwrap_or("medium"),
                        status: item
                            .get("status")
                            .and_then(|v| v.as_str())
                            .unwrap_or("planned"),
                        status_token: item
                            .get("status_token")
                            .and_then(|v| v.as_str())
                            .unwrap_or("PLANNED"),
                        description: item
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                        sprint: item.get("sprint").and_then(|v| v.as_str()).unwrap_or(""),
                        dependencies_json: &deps,
                        acceptance_criteria_json: &ac,
                        primary_outputs_json: &po,
                        evidence_refs_json: &er,
                        lacunae_json: &lac,
                        claims_json: &claims,
                        insight: item.get("insight").and_then(|v| v.as_str()).unwrap_or(""),
                    })?;
                    count += 1;
                }
            }
        }
        println!(
            "  <EMOJI+2713> Imported {count} roadmap workstreams from {}",
            rm_path.display()
        );
    } else {
        println!(
            "  <EMOJI+26A0> Roadmap file not found: {}",
            rm_path.display()
        );
    }

    // Import todo
    let td_path = repo_root.join(&args.todo);
    if td_path.exists() {
        let text =
            fs::read_to_string(&td_path).with_context(|| format!("read {}", td_path.display()))?;
        let val: Value =
            toml::from_str(&text).with_context(|| format!("parse {}", td_path.display()))?;
        store.record_registry_snapshot(repo_root, "todo", &td_path, &text)?;
        let mut count = 0u64;
        if let Some(items) = val.get("item").and_then(|v| v.as_array()) {
            for item in items {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let deps = json_array_field(item, "dependencies");
                    let ac = json_array_field(item, "acceptance_criteria");
                    let er = json_array_field(item, "evidence_refs");
                    store.upsert_todo_item(&provenance_store::ActionItem {
                        id,
                        area: item.get("area").and_then(|v| v.as_str()).unwrap_or(""),
                        title: item.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                        description: item
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                        priority: item
                            .get("priority")
                            .and_then(|v| v.as_str())
                            .unwrap_or("medium"),
                        status: item
                            .get("status")
                            .and_then(|v| v.as_str())
                            .unwrap_or("open"),
                        status_token: item
                            .get("status_token")
                            .and_then(|v| v.as_str())
                            .unwrap_or("OPEN"),
                        dependencies_json: &deps,
                        acceptance_criteria_json: &ac,
                        evidence_refs_json: &er,
                    })?;
                    count += 1;
                }
            }
        }
        println!(
            "  <EMOJI+2713> Imported {count} todo items from {}",
            td_path.display()
        );
    } else {
        println!("  <EMOJI+26A0> Todo file not found: {}", td_path.display());
    }

    // Import next actions
    let na_path = repo_root.join(&args.next_actions);
    if na_path.exists() {
        let text =
            fs::read_to_string(&na_path).with_context(|| format!("read {}", na_path.display()))?;
        let val: Value =
            toml::from_str(&text).with_context(|| format!("parse {}", na_path.display()))?;
        store.record_registry_snapshot(repo_root, "next_actions", &na_path, &text)?;
        let mut count = 0u64;
        if let Some(items) = val.get("action").and_then(|v| v.as_array()) {
            for item in items {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if !id.is_empty() {
                    let deps = json_array_field(item, "dependencies");
                    let ac = json_array_field(item, "acceptance_criteria");
                    let er = json_array_field(item, "evidence_refs");
                    store.upsert_next_action(&provenance_store::ActionItem {
                        id,
                        area: item.get("area").and_then(|v| v.as_str()).unwrap_or(""),
                        title: item.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                        description: item
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                        priority: item
                            .get("priority")
                            .and_then(|v| v.as_str())
                            .unwrap_or("medium"),
                        status: item
                            .get("status")
                            .and_then(|v| v.as_str())
                            .unwrap_or("open"),
                        status_token: item
                            .get("status_token")
                            .and_then(|v| v.as_str())
                            .unwrap_or("OPEN"),
                        dependencies_json: &deps,
                        acceptance_criteria_json: &ac,
                        evidence_refs_json: &er,
                    })?;
                    count += 1;
                }
            }
        }
        println!(
            "  <EMOJI+2713> Imported {count} next actions from {}",
            na_path.display()
        );
    } else {
        println!(
            "  <EMOJI+26A0> Next actions file not found: {}",
            na_path.display()
        );
    }

    println!("Planning import complete.");
    Ok(())
}

// ─── Import requirements ───────────────────────────────────────────

fn cmd_import_requirements(
    store: &mut ProvenanceStore,
    repo_root: &Path,
    args: &ImportRequirementsArgs,
) -> Result<()> {
    println!("Importing requirements data into SQLite...");

    let requirements_path = repo_root.join(&args.requirements);
    if !requirements_path.exists() {
        println!(
            "  <EMOJI+26A0> Requirements file not found: {}",
            requirements_path.display()
        );
        return Ok(());
    }

    let text = fs::read_to_string(&requirements_path)
        .with_context(|| format!("read {}", requirements_path.display()))?;
    let value: Value =
        toml::from_str(&text).with_context(|| format!("parse {}", requirements_path.display()))?;
    store.record_registry_snapshot(repo_root, "requirements", &requirements_path, &text)?;

    let requirements_table = root_table(&value, "requirements");
    let schema_table = child_table(requirements_table, "schema");

    let status_allowlist = serde_json::to_string(&table_array(
        requirements_table,
        "status_allowlist",
        &["active", "deprecated", "planned", "blocked"],
    ))?;
    let runtime_stack_allowlist = serde_json::to_string(&table_array(
        requirements_table,
        "runtime_stack_allowlist",
        &[
            "mixed",
            "rust",
            "python",
            "docker_python",
            "rocq",
            "latex",
            "cpp",
        ],
    ))?;
    let required_module_fields = serde_json::to_string(&table_array(
        schema_table,
        "required_module_fields",
        &[
            "id",
            "name",
            "status",
            "status_token",
            "runtime_stack",
            "requires_modules",
            "install_targets",
            "verify_targets",
            "acceptance_criteria",
        ],
    ))?;
    let required_gap_fields = serde_json::to_string(&table_array(
        schema_table,
        "required_gap_fields",
        &[
            "id",
            "area",
            "status",
            "status_token",
            "description",
            "proposed_resolution",
            "related_module_ids",
        ],
    ))?;
    let meta_status = table_string(requirements_table, "status", "active");
    let meta_status_token = table_string(requirements_table, "status_token", "ACTIVE");
    let meta_updated = table_string(requirements_table, "updated", "2026-02-10");
    let python_recommended = table_string(
        requirements_table,
        "python_recommended",
        "3.11-3.12",
    );
    let python_allowed = table_string(
        requirements_table,
        "python_allowed",
        "3.13+ (with optional extras caveats)",
    );
    let primary_markdown = table_string(
        requirements_table,
        "primary_markdown",
        "docs/REQUIREMENTS.md",
    );

    store.conn_exec(
        "DELETE FROM requirements_modules",
        std::iter::empty::<&str>(),
    )?;
    store.conn_exec(
        "DELETE FROM requirements_coverage_gaps",
        std::iter::empty::<&str>(),
    )?;
    store.conn_exec(
        "DELETE FROM requirements_registry_meta",
        std::iter::empty::<&str>(),
    )?;

    store.upsert_requirements_meta(&provenance_store::RequirementsMeta {
        authoritative: table_bool(requirements_table, "authoritative", true),
        status: &meta_status,
        status_token: &meta_status_token,
        updated: &meta_updated,
        python_recommended: &python_recommended,
        python_allowed: &python_allowed,
        primary_markdown: &primary_markdown,
        status_allowlist_json: &status_allowlist,
        runtime_stack_allowlist_json: &runtime_stack_allowlist,
        required_module_fields_json: &required_module_fields,
        required_gap_fields_json: &required_gap_fields,
    })?;

    let mut module_count = 0u64;
    if let Some(items) = value.get("module").and_then(Value::as_array) {
        for item in items {
            let id = item.get("id").and_then(Value::as_str).unwrap_or("");
            if id.is_empty() {
                continue;
            }
            let requires_modules = json_array_field(item, "requires_modules");
            let install_targets = json_array_field(item, "install_targets");
            let verify_targets = json_array_field(item, "verify_targets");
            let acceptance_criteria = json_array_field(item, "acceptance_criteria");
            store.upsert_requirement_module(&provenance_store::RequirementModuleItem {
                id,
                name: item.get("name").and_then(Value::as_str).unwrap_or(""),
                markdown: item.get("markdown").and_then(Value::as_str).unwrap_or(""),
                status: item.get("status").and_then(Value::as_str).unwrap_or("active"),
                status_token: item
                    .get("status_token")
                    .and_then(Value::as_str)
                    .unwrap_or("ACTIVE"),
                runtime_stack: item
                    .get("runtime_stack")
                    .and_then(Value::as_str)
                    .unwrap_or("mixed"),
                requires_modules_json: &requires_modules,
                install_targets_json: &install_targets,
                verify_targets_json: &verify_targets,
                acceptance_criteria_json: &acceptance_criteria,
            })?;
            module_count += 1;
        }
    }

    let mut gap_count = 0u64;
    if let Some(items) = value.get("coverage_gap").and_then(Value::as_array) {
        for item in items {
            let id = item.get("id").and_then(Value::as_str).unwrap_or("");
            if id.is_empty() {
                continue;
            }
            let related_module_ids = json_array_field(item, "related_module_ids");
            store.upsert_requirement_coverage_gap(
                &provenance_store::RequirementCoverageGapItem {
                    id,
                    area: item.get("area").and_then(Value::as_str).unwrap_or(""),
                    status: item.get("status").and_then(Value::as_str).unwrap_or("open"),
                    status_token: item
                        .get("status_token")
                        .and_then(Value::as_str)
                        .unwrap_or("OPEN"),
                    description: item
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                    proposed_resolution: item
                        .get("proposed_resolution")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                    related_module_ids_json: &related_module_ids,
                },
            )?;
            gap_count += 1;
        }
    }

    println!(
        "  <EMOJI+2713> Imported {module_count} requirement modules and {gap_count} coverage gaps from {}",
        requirements_path.display()
    );
    println!("Requirements import complete.");
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
        println!(
            "  <EMOJI+26A0> Narratives file not found: {}",
            path.display()
        );
        return Ok(());
    }

    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let val: Value = toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
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
                    source_markdown: doc
                        .get("source_markdown")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                    domain: doc.get("domain").and_then(|v| v.as_str()).unwrap_or(""),
                    slug: doc.get("slug").and_then(|v| v.as_str()).unwrap_or(""),
                    title: doc.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                    status_token: doc
                        .get("status_token")
                        .and_then(|v| v.as_str())
                        .unwrap_or("NARRATIVE"),
                    content_kind: doc
                        .get("content_kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("research_note"),
                    verification_level: doc
                        .get("verification_level")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                    claim_refs_json: &cr,
                    url_refs_json: &ur,
                    path_refs_json: &pr,
                    body_markdown: doc
                        .get("body_markdown")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                    line_count: doc
                        .get("line_count")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0),
                })?;
                count += 1;
            }
        }
    }

    println!(
        "  <EMOJI+2713> Imported {count} research narratives from {}",
        path.display()
    );
    println!("Narrative import complete.");
    Ok(())
}

// ─── Export planning ───────────────────────────────────────────────

fn cmd_export_planning(store: &ProvenanceStore, args: &ExportPlanningArgs) -> Result<()> {
    let label = match args.table {
        PlanningTable::Roadmap => "roadmap",
        PlanningTable::Todo => "todo",
        PlanningTable::NextActions => "next_actions",
    };

    let output = match args.format {
        OutputFormat::Json => match args.table {
            PlanningTable::Roadmap => serde_json::to_string_pretty(
                &store
                    .planning_roadmap_rows()?
                    .into_iter()
                    .map(|row| {
                        serde_json::json!({
                            "id": row.id,
                            "name": row.name,
                            "priority": row.priority,
                            "status": row.status,
                            "status_token": row.status_token,
                            "description": row.description,
                            "sprint": row.sprint,
                            "dependencies": serde_json::from_str::<Vec<String>>(&row.dependencies_json).unwrap_or_default(),
                            "acceptance_criteria": serde_json::from_str::<Vec<String>>(&row.acceptance_criteria_json).unwrap_or_default(),
                            "primary_outputs": serde_json::from_str::<Vec<String>>(&row.primary_outputs_json).unwrap_or_default(),
                            "evidence_refs": serde_json::from_str::<Vec<String>>(&row.evidence_refs_json).unwrap_or_default(),
                            "lacunae": serde_json::from_str::<Vec<String>>(&row.lacunae_json).unwrap_or_default(),
                            "claims": serde_json::from_str::<Vec<String>>(&row.claims_json).unwrap_or_default(),
                            "insight": row.insight,
                        })
                    })
                    .collect::<Vec<_>>(),
            )?,
            PlanningTable::Todo => serde_json::to_string_pretty(
                &store
                    .planning_todo_rows()?
                    .into_iter()
                    .map(|row| {
                        serde_json::json!({
                            "id": row.id,
                            "area": row.area,
                            "title": row.title,
                            "description": row.description,
                            "priority": row.priority,
                            "status": row.status,
                            "status_token": row.status_token,
                            "dependencies": serde_json::from_str::<Vec<String>>(&row.dependencies_json).unwrap_or_default(),
                            "acceptance_criteria": serde_json::from_str::<Vec<String>>(&row.acceptance_criteria_json).unwrap_or_default(),
                            "evidence_refs": serde_json::from_str::<Vec<String>>(&row.evidence_refs_json).unwrap_or_default(),
                        })
                    })
                    .collect::<Vec<_>>(),
            )?,
            PlanningTable::NextActions => serde_json::to_string_pretty(
                &store
                    .planning_next_action_rows()?
                    .into_iter()
                    .map(|row| {
                        serde_json::json!({
                            "id": row.id,
                            "area": row.area,
                            "title": row.title,
                            "description": row.description,
                            "priority": row.priority,
                            "status": row.status,
                            "status_token": row.status_token,
                            "dependencies": serde_json::from_str::<Vec<String>>(&row.dependencies_json).unwrap_or_default(),
                            "acceptance_criteria": serde_json::from_str::<Vec<String>>(&row.acceptance_criteria_json).unwrap_or_default(),
                            "evidence_refs": serde_json::from_str::<Vec<String>>(&row.evidence_refs_json).unwrap_or_default(),
                        })
                    })
                    .collect::<Vec<_>>(),
            )?,
        },
        OutputFormat::Toml => render_planning_toml(store, &args.table)?,
        OutputFormat::Text => {
            let items = match args.table {
                PlanningTable::Roadmap => store.list_roadmap_items(None)?,
                PlanningTable::Todo => store.list_todo_items(None)?,
                PlanningTable::NextActions => store.list_next_actions(None)?,
            };
            let mut lines = vec![format!("{label} ({} items):", items.len())];
            for (id, name, priority, status) in items {
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
    println!(
        "Search results for \"{}\" ({} hits):",
        args.query,
        results.len()
    );
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
                println!(
                    "  {:<20} [{}] {:<10} cells={}  {}",
                    s.id, s.kernel, s.status, s.cell_count, s.title
                );
            }
        }
        other => {
            // Generic table query
            let count = store.table_row_count(other)?;
            println!("{other} ({count} rows)");
            println!("  (Use a specific table name for detailed output.)");
            println!(
                "  Supported tables: roadmap_items, todo_items, next_action_items, notebook_sessions"
            );
        }
    }
    Ok(())
}

// ─── Audit ─────────────────────────────────────────────────────────

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
        let status = if full.exists() {
            "<EMOJI+2713>"
        } else {
            "<EMOJI+2717>"
        };
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
        let status = if full.exists() {
            "present (legacy)"
        } else {
            "absent"
        };
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
                println!(
                    "  {:<20} [{}] {:<10} cells={}  {}",
                    s.id, s.kernel, s.status, s.cell_count, s.title
                );
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
            println!("Created notebook session: {id} -- \"{title}\"");
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
                .filter_map(|v| {
                    v.as_str()
                        .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
                })
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

fn json_string_array(raw: &str) -> Result<Vec<String>> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(raw).with_context(|| format!("parse JSON string array from {raw}"))
}

fn toml_string_array(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| toml_quote(value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn root_table<'a>(value: &'a Value, key: &str) -> Option<&'a toml::value::Table> {
    value.get(key).and_then(Value::as_table)
}

fn child_table<'a>(
    table: Option<&'a toml::value::Table>,
    key: &str,
) -> Option<&'a toml::value::Table> {
    table
        .and_then(|table| table.get(key))
        .and_then(Value::as_table)
}

fn table_string(table: Option<&toml::value::Table>, key: &str, default: &str) -> String {
    table
        .and_then(|table| table.get(key))
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

fn table_bool(table: Option<&toml::value::Table>, key: &str, default: bool) -> bool {
    table
        .and_then(|table| table.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

fn table_array(table: Option<&toml::value::Table>, key: &str, default: &[&str]) -> Vec<String> {
    match table
        .and_then(|table| table.get(key))
        .and_then(Value::as_array)
    {
        Some(values) => values
            .iter()
            .filter_map(Value::as_str)
            .map(ToOwned::to_owned)
            .collect(),
        None => default.iter().map(|value| (*value).to_string()).collect(),
    }
}

fn parse_snapshot(store: &ProvenanceStore, kind: &str) -> Result<Option<Value>> {
    store
        .registry_snapshot(kind)?
        .map(|raw| {
            toml::from_str::<Value>(&raw).with_context(|| format!("parse {kind} registry snapshot"))
        })
        .transpose()
}

fn render_roadmap_toml(store: &ProvenanceStore) -> Result<String> {
    let snapshot = parse_snapshot(store, "roadmap")?;
    let roadmap_table = snapshot
        .as_ref()
        .and_then(|value| root_table(value, "roadmap"));
    let schema_table = child_table(roadmap_table, "schema");
    let sections_table = child_table(roadmap_table, "sections");
    let rows = store.planning_roadmap_rows()?;

    let status_allowlist = table_array(
        roadmap_table,
        "status_allowlist",
        &[
            "planned",
            "active",
            "in_progress",
            "done",
            "paused",
            "blocked",
        ],
    );
    let priority_allowlist = table_array(
        roadmap_table,
        "priority_allowlist",
        &["high", "medium", "low"],
    );
    let required_fields = table_array(
        schema_table,
        "required_fields",
        &[
            "id",
            "name",
            "priority",
            "status",
            "status_token",
            "description",
            "dependencies",
            "acceptance_criteria",
        ],
    );

    let mut lines = vec![
        "# Operational roadmap registry (SQLite compatibility export from canonical registry.sqlite3).".to_string(),
        "# Generated by `gororoba-db build` / `gororoba-db export-planning --table roadmap`.".to_string(),
        String::new(),
        "[roadmap]".to_string(),
        format!(
            "source_markdown = {}",
            toml_quote(&table_string(roadmap_table, "source_markdown", "docs/ROADMAP.md"))
        ),
        format!(
            "consolidated_date = {}",
            toml_quote(&table_string(roadmap_table, "consolidated_date", "2026-02-10"))
        ),
        format!(
            "supersedes = {}",
            toml_string_array(&table_array(roadmap_table, "supersedes", &[]))
        ),
        format!(
            "companion_docs = {}",
            toml_string_array(&table_array(roadmap_table, "companion_docs", &[]))
        ),
        format!(
            "status = {}",
            toml_quote(&table_string(roadmap_table, "status", "active"))
        ),
        format!(
            "status_token = {}",
            toml_quote(&table_string(roadmap_table, "status_token", "ACTIVE"))
        ),
        format!(
            "authoritative = {}",
            if table_bool(roadmap_table, "authoritative", true) {
                "true"
            } else {
                "false"
            }
        ),
        format!("workstream_count = {}", rows.len()),
        format!("status_allowlist = {}", toml_string_array(&status_allowlist)),
        format!("priority_allowlist = {}", toml_string_array(&priority_allowlist)),
        String::new(),
        "[roadmap.schema]".to_string(),
        format!("required_fields = {}", toml_string_array(&required_fields)),
        format!(
            "dependency_id_pattern = {}",
            toml_quote(&table_string(
                schema_table,
                "dependency_id_pattern",
                "WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*",
            ))
        ),
        String::new(),
        "[roadmap.sections]".to_string(),
        format!(
            "architectural_evolution = {}",
            toml_quote(&table_string(
                sections_table,
                "architectural_evolution",
                "## 1. Architectural Evolution",
            ))
        ),
        format!(
            "crate_ecosystem = {}",
            toml_quote(&table_string(
                sections_table,
                "crate_ecosystem",
                "## 2. Crate Ecosystem",
            ))
        ),
        format!(
            "documentation_registry = {}",
            toml_quote(&table_string(
                sections_table,
                "documentation_registry",
                "### 1.5 Documentation Registry Evolution (Sprint 13)",
            ))
        ),
        format!(
            "long_term_vision = {}",
            toml_quote(&table_string(
                sections_table,
                "long_term_vision",
                "## 8. Long-Term Vision",
            ))
        ),
        format!(
            "remaining_workstreams = {}",
            toml_quote(&table_string(
                sections_table,
                "remaining_workstreams",
                "## 7. Remaining Workstreams from ULTRA_ROADMAP.md",
            ))
        ),
        String::new(),
    ];

    for row in rows {
        lines.push("[[workstream]]".to_string());
        lines.push(format!("id = {}", toml_quote(&row.id)));
        lines.push(format!("name = {}", toml_quote(&row.name)));
        lines.push(format!("priority = {}", toml_quote(&row.priority)));
        lines.push(format!("status = {}", toml_quote(&row.status)));
        lines.push(format!("status_token = {}", toml_quote(&row.status_token)));
        lines.push(format!("description = {}", toml_quote(&row.description)));
        if !row.sprint.trim().is_empty() {
            lines.push(format!("sprint = {}", toml_quote(&row.sprint)));
        }
        lines.push(format!(
            "primary_outputs = {}",
            toml_string_array(&json_string_array(&row.primary_outputs_json)?)
        ));
        let claims = json_string_array(&row.claims_json)?;
        if !claims.is_empty() {
            lines.push(format!("claims = {}", toml_string_array(&claims)));
        }
        if !row.insight.trim().is_empty() {
            lines.push(format!("insight = {}", toml_quote(&row.insight)));
        }
        lines.push(format!(
            "dependencies = {}",
            toml_string_array(&json_string_array(&row.dependencies_json)?)
        ));
        lines.push(format!(
            "acceptance_criteria = {}",
            toml_string_array(&json_string_array(&row.acceptance_criteria_json)?)
        ));
        lines.push(format!(
            "evidence_refs = {}",
            toml_string_array(&json_string_array(&row.evidence_refs_json)?)
        ));
        let lacunae = json_string_array(&row.lacunae_json)?;
        if !lacunae.is_empty() {
            lines.push(format!("lacunae = {}", toml_string_array(&lacunae)));
        }
        lines.push(String::new());
    }

    while matches!(lines.last(), Some(line) if line.is_empty()) {
        lines.pop();
    }
    Ok(lines.join("\n"))
}

fn render_todo_toml(store: &ProvenanceStore) -> Result<String> {
    let snapshot = parse_snapshot(store, "todo")?;
    let todo_table = snapshot
        .as_ref()
        .and_then(|value| root_table(value, "todo"));
    let schema_table = child_table(todo_table, "schema");
    let rows = store.planning_todo_rows()?;

    let status_allowlist = table_array(
        todo_table,
        "status_allowlist",
        &["open", "in_progress", "done", "blocked", "deferred"],
    );
    let priority_allowlist =
        table_array(todo_table, "priority_allowlist", &["high", "medium", "low"]);
    let required_fields = table_array(
        schema_table,
        "required_fields",
        &[
            "id",
            "area",
            "title",
            "description",
            "priority",
            "status",
            "status_token",
            "dependencies",
            "acceptance_criteria",
        ],
    );

    let mut lines = vec![
        "# To-Do Registry (SQLite compatibility export from canonical registry.sqlite3)."
            .to_string(),
        "# Generated by `gororoba-db build` / `gororoba-db export-planning --table todo`."
            .to_string(),
        String::new(),
        "[todo]".to_string(),
        format!(
            "updated = {}",
            toml_quote(&table_string(todo_table, "updated", "2026-02-10"))
        ),
        format!(
            "status = {}",
            toml_quote(&table_string(todo_table, "status", "active"))
        ),
        format!(
            "status_token = {}",
            toml_quote(&table_string(todo_table, "status_token", "ACTIVE"))
        ),
        format!("item_count = {}", rows.len()),
        format!(
            "status_allowlist = {}",
            toml_string_array(&status_allowlist)
        ),
        format!(
            "priority_allowlist = {}",
            toml_string_array(&priority_allowlist)
        ),
        String::new(),
        "[todo.schema]".to_string(),
        format!("required_fields = {}", toml_string_array(&required_fields)),
        format!(
            "dependency_id_pattern = {}",
            toml_quote(&table_string(
                schema_table,
                "dependency_id_pattern",
                "WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*",
            ))
        ),
        String::new(),
    ];

    for row in rows {
        lines.push("[[item]]".to_string());
        lines.push(format!("id = {}", toml_quote(&row.id)));
        lines.push(format!("area = {}", toml_quote(&row.area)));
        lines.push(format!("title = {}", toml_quote(&row.title)));
        lines.push(format!("description = {}", toml_quote(&row.description)));
        lines.push(format!("priority = {}", toml_quote(&row.priority)));
        lines.push(format!("status = {}", toml_quote(&row.status)));
        lines.push(format!("status_token = {}", toml_quote(&row.status_token)));
        lines.push(format!(
            "dependencies = {}",
            toml_string_array(&json_string_array(&row.dependencies_json)?)
        ));
        lines.push(format!(
            "acceptance_criteria = {}",
            toml_string_array(&json_string_array(&row.acceptance_criteria_json)?)
        ));
        lines.push(format!(
            "evidence_refs = {}",
            toml_string_array(&json_string_array(&row.evidence_refs_json)?)
        ));
        lines.push(String::new());
    }

    while matches!(lines.last(), Some(line) if line.is_empty()) {
        lines.pop();
    }
    Ok(lines.join("\n"))
}

fn render_next_actions_toml(store: &ProvenanceStore) -> Result<String> {
    let snapshot = parse_snapshot(store, "next_actions")?;
    let meta_table = snapshot
        .as_ref()
        .and_then(|value| root_table(value, "meta"));
    let next_actions_table = snapshot
        .as_ref()
        .and_then(|value| root_table(value, "next_actions"));
    let schema_table = child_table(next_actions_table, "schema");
    let rows = store.planning_next_action_rows()?;

    let status_allowlist = table_array(
        meta_table,
        "status_allowlist",
        &["todo", "in_progress", "done", "blocked", "deferred"],
    );
    let priority_allowlist =
        table_array(meta_table, "priority_allowlist", &["high", "medium", "low"]);
    let required_fields = table_array(
        schema_table,
        "required_fields",
        &[
            "id",
            "area",
            "title",
            "description",
            "priority",
            "status",
            "status_token",
            "dependencies",
            "acceptance_criteria",
        ],
    );

    let mut lines = vec![
        "# Next Actions Registry (SQLite compatibility export from canonical registry.sqlite3)."
            .to_string(),
        "# Generated by `gororoba-db build` / `gororoba-db export-planning --table next-actions`."
            .to_string(),
        String::new(),
        "[meta]".to_string(),
        format!(
            "updated = {}",
            toml_quote(&table_string(meta_table, "updated", "2026-02-10"))
        ),
        format!(
            "status = {}",
            toml_quote(&table_string(meta_table, "status", "active"))
        ),
        format!(
            "status_token = {}",
            toml_quote(&table_string(meta_table, "status_token", "ACTIVE"))
        ),
        format!("action_count = {}", rows.len()),
        format!(
            "status_allowlist = {}",
            toml_string_array(&status_allowlist)
        ),
        format!(
            "priority_allowlist = {}",
            toml_string_array(&priority_allowlist)
        ),
        String::new(),
        "[next_actions.schema]".to_string(),
        format!("required_fields = {}", toml_string_array(&required_fields)),
        format!(
            "dependency_id_pattern = {}",
            toml_quote(&table_string(
                schema_table,
                "dependency_id_pattern",
                "WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*",
            ))
        ),
        String::new(),
    ];

    for row in rows {
        lines.push("[[action]]".to_string());
        lines.push(format!("id = {}", toml_quote(&row.id)));
        lines.push(format!("area = {}", toml_quote(&row.area)));
        lines.push(format!("title = {}", toml_quote(&row.title)));
        lines.push(format!("description = {}", toml_quote(&row.description)));
        lines.push(format!("priority = {}", toml_quote(&row.priority)));
        lines.push(format!("status = {}", toml_quote(&row.status)));
        lines.push(format!("status_token = {}", toml_quote(&row.status_token)));
        lines.push(format!(
            "dependencies = {}",
            toml_string_array(&json_string_array(&row.dependencies_json)?)
        ));
        lines.push(format!(
            "acceptance_criteria = {}",
            toml_string_array(&json_string_array(&row.acceptance_criteria_json)?)
        ));
        lines.push(format!(
            "evidence_refs = {}",
            toml_string_array(&json_string_array(&row.evidence_refs_json)?)
        ));
        lines.push(String::new());
    }

    while matches!(lines.last(), Some(line) if line.is_empty()) {
        lines.pop();
    }
    Ok(lines.join("\n"))
}

fn render_planning_toml(store: &ProvenanceStore, table: &PlanningTable) -> Result<String> {
    match table {
        PlanningTable::Roadmap => render_roadmap_toml(store),
        PlanningTable::Todo => render_todo_toml(store),
        PlanningTable::NextActions => render_next_actions_toml(store),
    }
}

fn export_planning_compat_files(store: &ProvenanceStore, repo_root: &Path) -> Result<()> {
    let roadmap_path = repo_root.join("registry/roadmap.toml");
    let todo_path = repo_root.join("registry/todo.toml");
    let next_actions_path = repo_root.join("registry/next_actions.toml");

    fs::write(&roadmap_path, format!("{}\n", render_roadmap_toml(store)?))
        .with_context(|| format!("write {}", roadmap_path.display()))?;
    fs::write(&todo_path, format!("{}\n", render_todo_toml(store)?))
        .with_context(|| format!("write {}", todo_path.display()))?;
    fs::write(
        &next_actions_path,
        format!("{}\n", render_next_actions_toml(store)?),
    )
    .with_context(|| format!("write {}", next_actions_path.display()))?;

    println!("  Planning compatibility exports refreshed:");
    println!("    {}", roadmap_path.display());
    println!("    {}", todo_path.display());
    println!("    {}", next_actions_path.display());
    Ok(())
}

// ── Build (Layer 2) ─────────────────────────────────────────────

#[derive(Deserialize)]
struct SourceManifest {
    source: Vec<SourceEntry>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct SourceEntry {
    path: String,
    role: String,
    table: String,
    description: String,
}

fn cmd_build(repo_root: &Path, db_path: &Path, args: &BuildArgs) -> Result<()> {
    println!("Building derived registry database...");
    println!("  DB path: {}", db_path.display());

    // Read source manifest.
    let manifest_path = repo_root.join("registry/source_manifest.toml");
    let manifest_text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: SourceManifest = toml::from_str(&manifest_text)
        .with_context(|| format!("parse {}", manifest_path.display()))?;

    println!(
        "  {} source files declared in manifest",
        manifest.source.len()
    );

    // Create fresh DB (deletes any existing file).
    let mut store = ProvenanceStore::build_fresh(db_path)?;
    store.record_build_metadata("builder", "gororoba-db build")?;
    store.record_build_metadata("built_at", &chrono::Utc::now().to_rfc3339())?;
    store.record_build_metadata("source_count", &manifest.source.len().to_string())?;

    // Ingest core registries via existing reindex methods.
    let claims_path = repo_root.join("registry/claims.toml");
    let insights_path = repo_root.join("registry/insights.toml");
    let experiments_path = repo_root.join("registry/experiments.toml");
    let binaries_path = repo_root.join("registry/binaries.toml");
    let proofs_project_path = repo_root.join("proofs/_CoqProject");

    if claims_path.exists()
        && insights_path.exists()
        && experiments_path.exists()
        && binaries_path.exists()
    {
        let stats = store.reindex_control_plane_from_registries(
            repo_root,
            &claims_path,
            &insights_path,
            &experiments_path,
            &binaries_path,
            &proofs_project_path,
        )?;
        println!(
            "  Control plane: {} claims, {} insights, {} experiments, {} binaries, {} theorems",
            stats.claim_count,
            stats.insight_count,
            stats.experiment_count,
            stats.binary_count,
            stats.theorem_count
        );
    }

    // Ingest artifacts and documents.
    let artifact_path = repo_root.join("registry/artifact_source_of_truth.toml");
    let knowledge_docs_path = repo_root.join("registry/knowledge/documents.toml");
    let lane_dir = repo_root.join("registry/source_lanes");
    if artifact_path.exists() {
        let stats = store.reindex_from_registries(
            repo_root,
            &artifact_path,
            &knowledge_docs_path,
            &lane_dir,
        )?;
        println!(
            "  Provenance: {} artifacts, {} documents, {} lanes",
            stats.artifact_count, stats.document_count, stats.lane_assignment_count
        );
    }

    // Ingest bibliography.
    let bib_path = repo_root.join("registry/bibliography.toml");
    if bib_path.exists() {
        let text = fs::read_to_string(&bib_path)?;
        let count = store.ingest_bibliography(&text)?;
        println!("  Bibliography: {count} entries");
    }

    // Ingest evidence edges.
    let edges_path = repo_root.join("registry/claims_evidence_edges.toml");
    if edges_path.exists() {
        let text = fs::read_to_string(&edges_path)?;
        let count = store.ingest_evidence_edges(&text)?;
        println!("  Evidence edges: {count}");
    }

    // Ingest lacunae.
    let lacunae_path = repo_root.join("registry/lacunae.toml");
    if lacunae_path.exists() {
        let text = fs::read_to_string(&lacunae_path)?;
        let count = store.ingest_lacunae(&text)?;
        println!("  Lacunae: {count}");
    }

    let planning_args = ImportPlanningArgs {
        roadmap: PathBuf::from("registry/roadmap.toml"),
        todo: PathBuf::from("registry/todo.toml"),
        next_actions: PathBuf::from("registry/next_actions.toml"),
    };
    cmd_import_planning(&mut store, repo_root, &planning_args)?;
    export_planning_compat_files(&store, repo_root)?;

    // Build crossref join tables.
    let (ce, ci) = store.build_crossrefs()?;
    println!("  Crossrefs: {ce} claim-experiment, {ci} claim-insight");

    println!("  Build complete.");

    if args.verify {
        println!();
        println!("Verifying...");
        let dangling = store.dangling_crossrefs()?;
        if dangling.is_empty() {
            println!("  Crossrefs: OK (no dangling references)");
        } else {
            println!("  Crossrefs: {} dangling references:", dangling.len());
            for (source, target, kind) in &dangling {
                println!("    {kind} {source} -> {target} (missing)");
            }
        }
        println!("  Verify complete.");
    }

    Ok(())
}

// ── Claims (Layer 3) ────────────────────────────────────────────

fn cmd_claims(store: &ProvenanceStore, args: &ClaimsArgs) -> Result<()> {
    match &args.action {
        ClaimsAction::List { status, limit } => {
            let claims = store.list_claims_filtered(status.as_deref(), *limit)?;
            for c in &claims {
                let proof_marker = if c.formal_proof.is_some() {
                    " [proved]"
                } else {
                    ""
                };
                println!(
                    "{:<8} [{:<12}] {}{}",
                    c.id, c.status, c.statement, proof_marker
                );
            }
            println!("\n{} claims shown.", claims.len());
        }
        ClaimsAction::Show { id } => match store.claim_by_id(id)? {
            Some(c) => {
                println!("ID:            {}", c.id);
                println!("Status:        {}", c.status);
                println!("Statement:     {}", c.statement);
                println!("Where stated:  {}", c.where_stated);
                println!("Last verified: {}", c.last_verified);
                if let Some(ref proof) = c.formal_proof {
                    println!("Formal proof:  {proof}");
                }
                if let Some(ref note) = c.status_note {
                    println!("Status note:   {note}");
                }
            }
            None => {
                println!("Claim {id} not found.");
            }
        },
        ClaimsAction::Search { query, limit } => {
            let results = store.search_claims(query, *limit)?;
            for (id, statement, status, _rank) in &results {
                println!("{:<8} [{:<12}] {}", id, status, statement);
            }
            println!("\n{} results.", results.len());
        }
        ClaimsAction::Unlinked => {
            let unlinked = store.unlinked_claims()?;
            for (id, statement) in &unlinked {
                println!("{:<8} {}", id, statement);
            }
            println!("\n{} unlinked claims.", unlinked.len());
        }
    }
    Ok(())
}

// ── Insights (Layer 3) ──────────────────────────────────────────

fn cmd_insights(store: &ProvenanceStore, args: &InsightsArgs) -> Result<()> {
    match &args.action {
        InsightsAction::List { limit } => {
            let insights = store.list_insights()?;
            for (i, ins) in insights.iter().enumerate() {
                if i >= *limit {
                    break;
                }
                println!("{:<8} [{:<12}] {}", ins.id, ins.status, ins.title);
            }
            println!("\n{} insights shown.", insights.len().min(*limit));
        }
        InsightsAction::Search { query, limit } => {
            let results = store.search_insights(query, *limit)?;
            for (id, title, status, _rank) in &results {
                println!("{:<8} [{:<12}] {}", id, status, title);
            }
            println!("\n{} results.", results.len());
        }
    }
    Ok(())
}

// ── Experiments (Layer 3) ───────────────────────────────────────

fn cmd_experiments(store: &ProvenanceStore, args: &ExperimentsArgs) -> Result<()> {
    match &args.action {
        ExperimentsAction::List { status, limit } => {
            let experiments = store.list_experiments_filtered(status.as_deref(), *limit)?;
            for e in &experiments {
                let bin = if e.binary.is_some() {
                    format!(" ({})", e.binary.as_deref().unwrap_or(""))
                } else {
                    String::new()
                };
                println!("{:<8} [{:<12}] {}{}", e.id, e.status, e.title, bin);
            }
            println!("\n{} experiments shown.", experiments.len());
        }
    }
    Ok(())
}

// ── Xref (Layer 3) ─────────────────────────────────────────────

fn cmd_xref(store: &ProvenanceStore, args: &XrefArgs) -> Result<()> {
    match &args.action {
        XrefAction::Dangling => {
            let dangling = store.dangling_crossrefs()?;
            if dangling.is_empty() {
                println!("No dangling crossrefs found.");
            } else {
                for (source, target, kind) in &dangling {
                    println!("{kind}: {source} -> {target} (missing)");
                }
                println!("\n{} dangling crossrefs.", dangling.len());
            }
        }
        XrefAction::Unlinked => {
            let unlinked = store.unlinked_claims()?;
            for (id, statement) in &unlinked {
                println!("{:<8} {}", id, statement);
            }
            println!("\n{} unlinked claims.", unlinked.len());
        }
        XrefAction::Coverage => {
            let total_claims = store.table_row_count("claims")?;
            let unlinked = store.unlinked_claims()?;
            let linked = total_claims - unlinked.len() as i64;
            let pct = if total_claims > 0 {
                (linked as f64 / total_claims as f64) * 100.0
            } else {
                0.0
            };
            println!("Claims:   {total_claims} total");
            println!("Linked:   {linked} ({pct:.1}%)");
            println!("Unlinked: {}", unlinked.len());
        }
    }
    Ok(())
}

// ── Audit (Layer 3) ─────────────────────────────────────────────

fn cmd_audit_cmd(store: &ProvenanceStore, repo_root: &Path, args: &AuditArgs) -> Result<()> {
    match &args.action {
        AuditAction::Signatures => {
            println!("Verifying schema signatures...");
            store.verify_invariants(repo_root)?;
            println!("OK: all signatures valid.");
        }
        AuditAction::Crossrefs => {
            let dangling = store.dangling_crossrefs()?;
            if dangling.is_empty() {
                println!("OK: no dangling crossrefs.");
            } else {
                for (source, target, kind) in &dangling {
                    println!("DANGLING: {kind} {source} -> {target}");
                }
                anyhow::bail!("{} dangling crossrefs found", dangling.len());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_parses_stats_subcommand() {
        let cli = Cli::try_parse_from(["gororoba-db", "stats"]);
        assert!(cli.is_ok(), "stats subcommand should parse: {cli:?}");
    }
}
