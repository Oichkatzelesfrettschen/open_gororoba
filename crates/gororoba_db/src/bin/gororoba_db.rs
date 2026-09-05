//! `gororoba-db` -- unified Rust-native CLI for the SQLite source-of-truth.
//!
//! This binary consolidates all database interaction into a single
//! entrypoint: schema introspection, statistics, querying, import/export
//! of knowledge-base, planning, and requirements tables, full-text search, audit of
//! legacy TOML compatibility layer, and notebook-session management for evcxr/Jupyter
//! integration.

use anyhow::{Context, Result, bail};
use clap::Parser;
use provenance_store::{
    ExecutionTargetRetarget, PlanningCompatTable, ProvenanceStore, SourcePathRetarget,
    parse_theorem_identity_spec,
};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use toml::Value;

// Cli + all subcommand/Args enum and struct definitions live in the
// `types` submodule (~650 lines of declarative clap definitions).
// Uses `#[path]` because this binary has explicit Cargo.toml path.
#[path = "gororoba_db/types.rs"]
mod types;
use types::*;

// ─── Main ──────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    let db_path = cli.repo_root.join(&cli.db);

    // Build command creates DB from scratch; all others open existing.
    if let Commands::Build(ref args) = cli.command {
        return cmd_build(&cli.repo_root, &db_path, args);
    }

    if let Commands::Claim(args) = &cli.command
        && matches!(
            &args.action,
            ClaimMutationAction::Transition(transition_args)
                if matches!(
                    &transition_args.action,
                    ClaimTransitionAction::Plan { .. }
                        | ClaimTransitionAction::Show { .. }
                        | ClaimTransitionAction::Fingerprint { .. }
                )
        )
    {
        let store = ProvenanceStore::open_read_only(&db_path)
            .with_context(|| format!("open database read-only {}", db_path.display()))?;
        return cmd_claim_transition_read_only(&store, &cli.repo_root, args);
    }

    if let Commands::Theorem(args) = &cli.command
        && matches!(
            &args.action,
            TheoremAction::Identity(identity_args)
                if matches!(&identity_args.action, TheoremIdentityAction::Validate)
        )
    {
        let store = ProvenanceStore::open_read_only(&db_path)
            .with_context(|| format!("open database read-only {}", db_path.display()))?;
        return cmd_theorem_identity_read_only(&store, &cli.repo_root);
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
        Commands::Planning(args) => cmd_planning_mutation(&mut store, &cli.repo_root, &args),
        Commands::Claim(args) => cmd_claim_mutation(&mut store, &cli.repo_root, &args),
        Commands::Insight(args) => cmd_insight_mutation(&mut store, &args),
        Commands::Experiment(args) => cmd_experiment_mutation(&mut store, &cli.repo_root, &args),
        Commands::Binaries(args) => cmd_binaries_mutation(&mut store, &cli.repo_root, &args),
        Commands::Artifact(args) => cmd_artifact_mutation(&mut store, &cli.repo_root, &args),
        Commands::Theorem(args) => cmd_theorem_mutation(&mut store, &cli.repo_root, &args),
        Commands::Requirements(args) => {
            cmd_requirements_mutation(&mut store, &cli.repo_root, &args)
        }
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
                    store.upsert_roadmap_item_with_links(
                        &provenance_store::RoadmapItemWithLinks {
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
                        },
                    )?;
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
                    store.upsert_todo_item_with_evidence(
                        &provenance_store::ActionItemWithEvidence {
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
                        },
                    )?;
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
                    store.upsert_next_action_with_evidence(
                        &provenance_store::ActionItemWithEvidence {
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
                        },
                    )?;
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
    let python_recommended = table_string(requirements_table, "python_recommended", "3.11-3.12");
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
                status: item
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("active"),
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

fn cmd_export_requirements(store: &ProvenanceStore, args: &ExportRequirementsArgs) -> Result<()> {
    let output = match args.format {
        RequirementsOutputFormat::Json => {
            let meta = store.requirements_meta_row()?;
            let modules = store.requirements_module_rows()?;
            let gaps = store.requirements_coverage_gap_rows()?;

            serde_json::to_string_pretty(&serde_json::json!({
                "requirements": meta.as_ref().map(|row| {
                    serde_json::json!({
                        "authoritative": row.authoritative,
                        "status": row.status,
                        "status_token": row.status_token,
                        "updated": row.updated,
                        "python_recommended": row.python_recommended,
                        "python_allowed": row.python_allowed,
                        "primary_markdown": row.primary_markdown,
                        "status_allowlist": serde_json::from_str::<Vec<String>>(&row.status_allowlist_json).unwrap_or_default(),
                        "runtime_stack_allowlist": serde_json::from_str::<Vec<String>>(&row.runtime_stack_allowlist_json).unwrap_or_default(),
                        "required_module_fields": serde_json::from_str::<Vec<String>>(&row.required_module_fields_json).unwrap_or_default(),
                        "required_gap_fields": serde_json::from_str::<Vec<String>>(&row.required_gap_fields_json).unwrap_or_default(),
                    })
                }).unwrap_or(serde_json::Value::Null),
                "module": modules.into_iter().map(|row| {
                    serde_json::json!({
                        "id": row.id,
                        "name": row.name,
                        "markdown": row.markdown,
                        "status": row.status,
                        "status_token": row.status_token,
                        "runtime_stack": row.runtime_stack,
                        "requires_modules": serde_json::from_str::<Vec<String>>(&row.requires_modules_json).unwrap_or_default(),
                        "install_targets": serde_json::from_str::<Vec<String>>(&row.install_targets_json).unwrap_or_default(),
                        "verify_targets": serde_json::from_str::<Vec<String>>(&row.verify_targets_json).unwrap_or_default(),
                        "acceptance_criteria": serde_json::from_str::<Vec<String>>(&row.acceptance_criteria_json).unwrap_or_default(),
                    })
                }).collect::<Vec<_>>(),
                "coverage_gap": gaps.into_iter().map(|row| {
                    serde_json::json!({
                        "id": row.id,
                        "area": row.area,
                        "status": row.status,
                        "status_token": row.status_token,
                        "description": row.description,
                        "proposed_resolution": row.proposed_resolution,
                        "related_module_ids": serde_json::from_str::<Vec<String>>(&row.related_module_ids_json).unwrap_or_default(),
                    })
                }).collect::<Vec<_>>(),
            }))?
        }
        RequirementsOutputFormat::Toml => store.render_requirements_compat_toml()?,
        RequirementsOutputFormat::Text => {
            let meta = store.requirements_meta_row()?;
            let modules = store.requirements_module_rows()?;
            let gaps = store.requirements_coverage_gap_rows()?;

            let mut lines = vec![format!(
                "requirements ({} modules, {} coverage gaps):",
                modules.len(),
                gaps.len()
            )];
            if let Some(meta_row) = meta {
                lines.push(format!(
                    "  status={} updated={} primary_markdown={}",
                    meta_row.status, meta_row.updated, meta_row.primary_markdown
                ));
            }
            for row in modules {
                lines.push(format!(
                    "  module {:<18} {:<10} {:<14} {}",
                    row.id, row.status, row.runtime_stack, row.name
                ));
            }
            for row in gaps {
                lines.push(format!(
                    "  gap {:<18} {:<10} {}",
                    row.id, row.status, row.area
                ));
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

fn status_token_or_default(status: &str, explicit: &str) -> String {
    if explicit.trim().is_empty() {
        status.trim().to_ascii_uppercase()
    } else {
        explicit.trim().to_string()
    }
}

fn json_array(values: &[String]) -> Result<String> {
    serde_json::to_string(values).context("serialize JSON array")
}

fn load_requirements_narrative_paths(
    repo_root: &Path,
) -> Result<std::collections::BTreeSet<String>> {
    let path = repo_root.join("registry/requirements_narrative.toml");
    if !path.exists() {
        bail!("requirements narrative file is missing: {}", path.display());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let value: Value =
        toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let mut out = std::collections::BTreeSet::new();
    if let Some(documents) = value.get("document").and_then(Value::as_array) {
        for document in documents {
            let path = document
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if !path.is_empty() {
                out.insert(path.to_string());
            }
        }
    }
    Ok(out)
}

fn ensure_requirements_narrative_path(
    repo_root: &Path,
    narrative_paths: &std::collections::BTreeSet<String>,
    field_name: &str,
    path: &str,
) -> Result<()> {
    if narrative_paths.contains(path) {
        return Ok(());
    }
    bail!(
        "{field_name} `{path}` is not declared in {}; add a matching [[document]] path before updating the canonical DB",
        repo_root
            .join("registry/requirements_narrative.toml")
            .display()
    )
}

fn default_requirements_meta() -> provenance_store::RequirementsMeta<'static> {
    provenance_store::RequirementsMeta {
        authoritative: true,
        status: "active",
        status_token: "ACTIVE",
        updated: "2026-02-10",
        python_recommended: "3.11-3.12",
        python_allowed: "3.13+ (with optional extras caveats)",
        primary_markdown: "docs/REQUIREMENTS.md",
        status_allowlist_json: "[\"active\",\"deprecated\",\"planned\",\"blocked\"]",
        runtime_stack_allowlist_json: "[\"mixed\",\"rust\",\"python\",\"docker_python\",\"rocq\",\"latex\",\"cpp\"]",
        required_module_fields_json: "[\"id\",\"name\",\"status\",\"status_token\",\"runtime_stack\",\"requires_modules\",\"install_targets\",\"verify_targets\",\"acceptance_criteria\"]",
        required_gap_fields_json: "[\"id\",\"area\",\"status\",\"status_token\",\"description\",\"proposed_resolution\",\"related_module_ids\"]",
    }
}

fn cmd_planning_mutation(
    store: &mut ProvenanceStore,
    repo_root: &Path,
    args: &PlanningMutationArgs,
) -> Result<()> {
    match &args.action {
        PlanningMutationAction::UpsertRoadmapItem {
            id,
            name,
            priority,
            status,
            status_token,
            description,
            sprint,
            dependencies,
            acceptance_criteria,
            primary_outputs,
            evidence_refs,
            lacunae,
            claims,
            insight,
        } => {
            let dependencies_json = json_array(dependencies)?;
            let acceptance_criteria_json = json_array(acceptance_criteria)?;
            let primary_outputs_json = json_array(primary_outputs)?;
            let evidence_refs_json = json_array(evidence_refs)?;
            let lacunae_json = json_array(lacunae)?;
            let claims_json = json_array(claims)?;
            let status_token = status_token_or_default(status, status_token);
            store.upsert_roadmap_item_with_links(&provenance_store::RoadmapItemWithLinks {
                id,
                name,
                priority,
                status,
                status_token: &status_token,
                description,
                sprint,
                dependencies_json: &dependencies_json,
                acceptance_criteria_json: &acceptance_criteria_json,
                primary_outputs_json: &primary_outputs_json,
                evidence_refs_json: &evidence_refs_json,
                lacunae_json: &lacunae_json,
                claims_json: &claims_json,
                insight,
            })?;
            println!("Updated roadmap item: {id}");
        }
        PlanningMutationAction::DeleteRoadmapItem { id } => {
            store.delete_roadmap_item(id)?;
            println!("Deleted roadmap item: {id}");
        }
        PlanningMutationAction::UpsertTodoItem {
            id,
            area,
            title,
            description,
            priority,
            status,
            status_token,
            dependencies,
            acceptance_criteria,
            evidence_refs,
        } => {
            let dependencies_json = json_array(dependencies)?;
            let acceptance_criteria_json = json_array(acceptance_criteria)?;
            let evidence_refs_json = json_array(evidence_refs)?;
            let status_token = status_token_or_default(status, status_token);
            store.upsert_todo_item_with_evidence(&provenance_store::ActionItemWithEvidence {
                id,
                area,
                title,
                description,
                priority,
                status,
                status_token: &status_token,
                dependencies_json: &dependencies_json,
                acceptance_criteria_json: &acceptance_criteria_json,
                evidence_refs_json: &evidence_refs_json,
            })?;
            println!("Updated todo item: {id}");
        }
        PlanningMutationAction::DeleteTodoItem { id } => {
            store.delete_todo_item(id)?;
            println!("Deleted todo item: {id}");
        }
        PlanningMutationAction::UpsertNextAction {
            id,
            area,
            title,
            description,
            priority,
            status,
            status_token,
            dependencies,
            acceptance_criteria,
            evidence_refs,
        } => {
            let dependencies_json = json_array(dependencies)?;
            let acceptance_criteria_json = json_array(acceptance_criteria)?;
            let evidence_refs_json = json_array(evidence_refs)?;
            let status_token = status_token_or_default(status, status_token);
            store.upsert_next_action_with_evidence(&provenance_store::ActionItemWithEvidence {
                id,
                area,
                title,
                description,
                priority,
                status,
                status_token: &status_token,
                dependencies_json: &dependencies_json,
                acceptance_criteria_json: &acceptance_criteria_json,
                evidence_refs_json: &evidence_refs_json,
            })?;
            println!("Updated next action: {id}");
        }
        PlanningMutationAction::DeleteNextAction { id } => {
            store.delete_next_action(id)?;
            println!("Deleted next action: {id}");
        }
    }

    export_planning_compat_files(store, repo_root)?;
    Ok(())
}

/// Pretty-print a status_note revision audit row to stdout.
fn print_revision_summary(entity_kind: &str, revision: &provenance_store::StatusNoteRevision) {
    println!(
        "{} {} {}: revision {} actor={} prev_sha256={} new_sha256={}",
        entity_kind,
        revision.entity_id,
        revision.field_name,
        revision.revision_id,
        revision.actor,
        revision.prev_value_sha256.as_deref().unwrap_or("(none)"),
        revision.new_value_sha256
    );
}

/// If `regen_toml` is true, spawn `provenance export-control-plane` to
/// regenerate the compatibility-export TOMLs and downstream mirror files.
/// `gororoba_cli_provenance` owns the CLI entrypoint and delegates the
/// implementation to `provenance_ops`. Subprocess execution keeps those
/// implementation dependencies outside `gororoba_db`. Errors propagate after
/// the caller's mutation has committed.
fn maybe_regen_toml(regen_toml: bool) -> Result<()> {
    if !regen_toml {
        eprintln!(
            "skipped TOML regen (--regen-toml false); run `make registry-export-markdown` later."
        );
        return Ok(());
    }
    eprintln!("regenerating compatibility-export TOMLs ...");
    let status = std::process::Command::new("cargo")
        .args([
            "run",
            "--release",
            "-p",
            "gororoba_cli_provenance",
            "--bin",
            "provenance",
            "--",
            "export-control-plane",
        ])
        .status()
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to spawn `cargo run -p gororoba_cli_provenance --bin provenance`: {}",
                e
            )
        })?;
    if !status.success() {
        return Err(anyhow::anyhow!(
            "TOML regen subprocess exited with {}; the SQLite mutation already \
             committed -- you may need to re-run `make registry-export-markdown` \
             manually.",
            status
        ));
    }
    eprintln!(
        "regen complete; remember to run `make registry-integrity` to refresh schema_signatures.toml."
    );
    Ok(())
}

fn cmd_claim_mutation(
    store: &mut ProvenanceStore,
    repo_root: &Path,
    args: &ClaimMutationArgs,
) -> Result<()> {
    match &args.action {
        ClaimMutationAction::SetEvidence {
            spec,
            actor,
            reason,
        } => {
            let path = resolve_cli_path(repo_root, spec);
            let text = fs::read_to_string(&path)
                .with_context(|| format!("read claim evidence {}", path.display()))?;
            let spec = ProvenanceStore::parse_claim_evidence_spec(&text)?;
            let revision = store.set_claim_evidence(repo_root, &spec, actor, reason)?;
            println!("claim={} evidence_revision={revision}", spec.claim_id);
        }
        ClaimMutationAction::Transition(transition_args) => {
            let ClaimTransitionAction::Apply { spec, regen_toml } = &transition_args.action else {
                unreachable!(
                    "read-only claim transitions are dispatched before opening a write handle"
                )
            };
            let spec_path = resolve_cli_path(repo_root, spec);
            let raw_spec = fs::read_to_string(&spec_path)
                .with_context(|| format!("read claim transition spec {}", spec_path.display()))?;
            let parsed = ProvenanceStore::parse_claim_transition_spec(&raw_spec)?;
            let result = store.apply_claim_transition(&parsed, &raw_spec)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            maybe_regen_toml(*regen_toml)?;
        }
        ClaimMutationAction::UpdateStatusNote {
            id,
            status_note,
            actor,
            reason,
            regen_toml,
        } => {
            let actor = actor
                .clone()
                .or_else(|| std::env::var("USER").ok())
                .unwrap_or_else(|| "unknown".to_string());
            let revision =
                store.claim_update_status_note(id, status_note, &actor, reason.as_deref())?;
            print_revision_summary("claim", &revision);
            maybe_regen_toml(*regen_toml)?;
        }
        ClaimMutationAction::ShowStatusNote { id } => {
            let note = store.claim_status_note(id)?;
            match note {
                Some(text) => println!("{}: {}", id, text),
                None => println!("{}: (status_note is NULL)", id),
            }
        }
        ClaimMutationAction::UpdateFormalProof {
            id,
            formal_proof,
            actor,
            reason,
            regen_toml,
        } => {
            let actor = actor
                .clone()
                .or_else(|| std::env::var("USER").ok())
                .unwrap_or_else(|| "unknown".to_string());
            let revision =
                store.claim_update_formal_proof(id, formal_proof, &actor, reason.as_deref())?;
            print_revision_summary("claim formal_proof", &revision);
            maybe_regen_toml(*regen_toml)?;
        }
        ClaimMutationAction::ShowFormalProof { id } => {
            let note = store.claim_formal_proof(id)?;
            match note {
                Some(text) if text.is_empty() => {
                    println!("{}: (formal_proof is empty)", id)
                }
                Some(text) => println!("{}: {}", id, text),
                None => println!("{}: (formal_proof is NULL)", id),
            }
        }
    }
    Ok(())
}

fn cmd_claim_transition_read_only(
    store: &ProvenanceStore,
    repo_root: &Path,
    args: &ClaimMutationArgs,
) -> Result<()> {
    let ClaimMutationAction::Transition(transition_args) = &args.action else {
        unreachable!("claim transition read-only dispatch received another claim action")
    };
    match &transition_args.action {
        ClaimTransitionAction::Plan { spec } => {
            let spec_path = resolve_cli_path(repo_root, spec);
            let raw_spec = fs::read_to_string(&spec_path)
                .with_context(|| format!("read claim transition spec {}", spec_path.display()))?;
            let parsed = ProvenanceStore::parse_claim_transition_spec(&raw_spec)?;
            let plan = store.plan_claim_transition(&parsed, &raw_spec)?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        ClaimTransitionAction::Show { key } => {
            let event = store
                .claim_transition_by_key(key)?
                .with_context(|| format!("transition event {key} not found"))?;
            println!("{}", serde_json::to_string_pretty(&event)?);
        }
        ClaimTransitionAction::Fingerprint { id } => {
            let fingerprint = serde_json::json!({
                "claim_id": id,
                "source_state_sha256": store.claim_transition_source_state_sha256(id)?,
                "expected_claim_id_max": store.claim_transition_expected_claim_id_max()?,
            });
            println!("{}", serde_json::to_string_pretty(&fingerprint)?);
        }
        ClaimTransitionAction::Apply { .. } => {
            unreachable!("apply is dispatched through the mutable claim command")
        }
    }
    Ok(())
}

fn resolve_cli_path(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn cmd_insight_mutation(store: &mut ProvenanceStore, args: &InsightMutationArgs) -> Result<()> {
    match &args.action {
        InsightMutationAction::UpdateStatusNote {
            id,
            status_note,
            actor,
            reason,
            regen_toml,
        } => {
            let actor = actor
                .clone()
                .or_else(|| std::env::var("USER").ok())
                .unwrap_or_else(|| "unknown".to_string());
            let revision =
                store.insight_update_status_note(id, status_note, &actor, reason.as_deref())?;
            print_revision_summary("insight", &revision);
            maybe_regen_toml(*regen_toml)?;
        }
        InsightMutationAction::ShowStatusNote { id } => {
            let note = store.insight_status_note(id)?;
            match note {
                Some(text) => println!("{}: {}", id, text),
                None => println!("{}: (status_note is NULL)", id),
            }
        }
        InsightMutationAction::UpdateSummary {
            id,
            summary,
            actor,
            reason,
            regen_toml,
        } => {
            let actor = actor
                .clone()
                .or_else(|| std::env::var("USER").ok())
                .unwrap_or_else(|| "unknown".to_string());
            let revision = store.insight_update_summary(id, summary, &actor, reason.as_deref())?;
            print_revision_summary("insight", &revision);
            maybe_regen_toml(*regen_toml)?;
        }
        InsightMutationAction::ShowSummary { id } => match store.insight_summary(id)? {
            Some(text) => println!("{}: {}", id, text),
            None => println!("{}: (summary is absent)", id),
        },
    }
    Ok(())
}

fn cmd_experiment_mutation(
    store: &mut ProvenanceStore,
    repo_root: &Path,
    args: &ExperimentMutationArgs,
) -> Result<()> {
    match &args.action {
        ExperimentMutationAction::UpdateStatusNote {
            id,
            status_note,
            actor,
            reason,
            regen_toml,
        } => {
            let actor = actor
                .clone()
                .or_else(|| std::env::var("USER").ok())
                .unwrap_or_else(|| "unknown".to_string());
            let revision =
                store.experiment_update_status_note(id, status_note, &actor, reason.as_deref())?;
            print_revision_summary("experiment", &revision);
            maybe_regen_toml(*regen_toml)?;
        }
        ExperimentMutationAction::ShowStatusNote { id } => {
            let note = store.experiment_status_note(id)?;
            match note {
                Some(text) => println!("{}: {}", id, text),
                None => println!("{}: (status_note is NULL)", id),
            }
        }
        ExperimentMutationAction::UpsertFromToml { spec, regen_toml } => {
            let spec_path = resolve_cli_path(repo_root, spec);
            let raw = fs::read_to_string(&spec_path)
                .with_context(|| format!("read experiment spec {}", spec_path.display()))?;
            let ids = store.upsert_experiments_from_registry_text(repo_root, &spec_path, &raw)?;
            println!("upserted experiments: {}", ids.join(", "));
            maybe_regen_toml(*regen_toml)?;
        }
        ExperimentMutationAction::RetargetExecutionTarget {
            from,
            to,
            actor,
            reason,
            dry_run,
            regen_toml,
        } => {
            if *dry_run {
                let preview = store.preview_execution_target_retarget(from, to)?;
                for (table, field, id) in &preview {
                    println!("would update {table}.{field} for {id}");
                }
                println!("{} row(s) would change", preview.len());
                return Ok(());
            }
            let actor = resolve_actor(actor.clone());
            let summary = store.retarget_execution_target(ExecutionTargetRetarget {
                from,
                to,
                actor: &actor,
                reason: reason.as_deref(),
            })?;
            for revision in &summary.revisions {
                print_revision_summary("execution-target", revision);
            }
            println!(
                "Retargeted {from} -> {to} across {} row(s).",
                summary.revisions.len()
            );
            maybe_regen_toml(*regen_toml)?;
        }
        ExperimentMutationAction::RetargetSourcePath {
            from,
            to,
            actor,
            reason,
            regen_toml,
        } => {
            let actor = resolve_actor(actor.clone());
            let summary = store.retarget_source_path(SourcePathRetarget {
                from,
                to,
                actor: &actor,
                reason: reason.as_deref(),
            })?;
            for revision in &summary.revisions {
                print_revision_summary("source-path", revision);
            }
            println!(
                "Moved {from} -> {to} across {} entity row(s) and {} contract path(s).",
                summary.revisions.len(),
                summary.contract_paths_updated
            );
            maybe_regen_toml(*regen_toml)?;
        }
    }
    Ok(())
}

/// Actor recorded on a revision row: the explicit `--actor`, otherwise the
/// invoking account, otherwise a marker that says the account was unreadable
/// rather than leaving the audit row blank.
fn resolve_actor(actor: Option<String>) -> String {
    actor
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Binary targets cargo declares for the workspace, read from `cargo metadata`
/// so the inventory follows the manifests rather than a second hand-kept list.
fn declared_binary_targets(repo_root: &Path) -> Result<Vec<provenance_store::BinaryRecord>> {
    let output = std::process::Command::new(std::env::var("CARGO").as_deref().unwrap_or("cargo"))
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(repo_root)
        .output()
        .context("run cargo metadata")?;
    if !output.status.success() {
        anyhow::bail!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("parse cargo metadata")?;
    let packages = metadata["packages"]
        .as_array()
        .context("cargo metadata has no packages array")?;
    let mut records = Vec::new();
    for package in packages {
        let crate_name = package["name"].as_str().unwrap_or_default().to_string();
        let Some(targets) = package["targets"].as_array() else {
            continue;
        };
        for target in targets {
            let is_bin = target["kind"]
                .as_array()
                .is_some_and(|kinds| kinds.iter().any(|kind| kind == "bin"));
            if !is_bin {
                continue;
            }
            let name = target["name"].as_str().unwrap_or_default().to_string();
            records.push(provenance_store::BinaryRecord {
                description: format!(
                    "Workspace binary discovered from cargo metadata in crate {crate_name}; registry metadata pending."
                ),
                name,
                crate_name: crate_name.clone(),
                experiment: None,
                source: "cargo-metadata".to_string(),
            });
        }
    }
    Ok(records)
}

fn cmd_binaries_mutation(
    store: &mut ProvenanceStore,
    repo_root: &Path,
    args: &BinariesMutationArgs,
) -> Result<()> {
    match &args.action {
        BinariesMutationAction::Sync {
            actor,
            reason,
            dry_run,
            regen_toml,
        } => {
            let declared = declared_binary_targets(repo_root)?;
            if *dry_run {
                let summary = store.preview_binaries_sync(&declared)?;
                for name in &summary.removed {
                    println!("would remove {name}");
                }
                for name in &summary.added {
                    println!("would add {name}");
                }
                for change in &summary.owner_changes {
                    println!(
                        "would change owner {}: {} -> {}",
                        change.name, change.previous_crate, change.declared_crate
                    );
                }
                println!(
                    "{} add(s), {} removal(s), {} owner change(s)",
                    summary.added.len(),
                    summary.removed.len(),
                    summary.owner_changes.len()
                );
                return Ok(());
            }
            let actor = resolve_actor(actor.clone());
            let summary = store.binaries_sync(&declared)?;
            for name in &summary.removed {
                println!("removed {name}");
            }
            for name in &summary.added {
                println!("added {name}");
            }
            for change in &summary.owner_changes {
                println!(
                    "changed owner {}: {} -> {}",
                    change.name, change.previous_crate, change.declared_crate
                );
            }
            println!(
                "Synced binaries_cp by {actor}{}: {} retained, {} added, {} removed, {} owner changes.",
                reason
                    .as_deref()
                    .map(|r| format!(" ({r})"))
                    .unwrap_or_default(),
                summary.retained,
                summary.added.len(),
                summary.removed.len(),
                summary.owner_changes.len()
            );
            maybe_regen_toml(*regen_toml)?;
        }
    }
    Ok(())
}

fn cmd_artifact_mutation(
    store: &mut ProvenanceStore,
    repo_root: &Path,
    args: &ArtifactArgs,
) -> Result<()> {
    match &args.action {
        ArtifactAction::RepairPaths { spec } => {
            let spec = repo_root.join(spec);
            let specification: provenance_store::ArtifactPathRepairSpec =
                toml::from_str(&fs::read_to_string(&spec)?)?;
            let report = store.repair_artifact_paths(repo_root, &specification)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        ArtifactAction::RecordRetrieval { spec } => {
            let specification: provenance_store::ArtifactRetrievalSpec =
                toml::from_str(&fs::read_to_string(repo_root.join(spec))?)?;
            let report = store.record_artifact_retrieval(repo_root, &specification)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        ArtifactAction::AssignLane {
            ids,
            lane,
            actor,
            reason,
        } => {
            let count =
                store.assign_artifact_lane(ids, lane, actor.as_deref(), reason.as_deref())?;
            println!("Added {count} artifact lane assignments in {lane}.");
        }
        ArtifactAction::RegisterLocal {
            id,
            key,
            title,
            citation,
            paths,
            lane,
            source_refs,
            actor,
            reason,
        } => {
            let registration = provenance_store::LocalArtifactRegistration {
                id,
                key,
                title,
                citation,
                paths,
                lane_name: lane,
                source_refs,
                actor: actor.as_deref(),
                reason: reason.as_deref(),
            };
            let path_count = store.register_local_artifact(repo_root, &registration)?;
            println!("Registered local artifact {id} with {path_count} retained paths in {lane}.");
        }
    }
    Ok(())
}

fn cmd_theorem_mutation(
    store: &mut ProvenanceStore,
    repo_root: &Path,
    args: &TheoremArgs,
) -> Result<()> {
    let TheoremAction::Identity(identity_args) = &args.action;
    match &identity_args.action {
        TheoremIdentityAction::Bind { spec, regen_toml } => {
            let spec_path = resolve_cli_path(repo_root, spec);
            let raw_spec = fs::read_to_string(&spec_path)
                .with_context(|| format!("read theorem identity spec {}", spec_path.display()))?;
            let parsed = parse_theorem_identity_spec(&raw_spec)?;
            let result = store.bind_theorem_identities(repo_root, &parsed, &raw_spec)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            maybe_regen_toml(*regen_toml)?;
        }
        TheoremIdentityAction::Validate => {
            unreachable!("theorem validation is dispatched through a read-only handle")
        }
    }
    Ok(())
}

fn cmd_theorem_identity_read_only(store: &ProvenanceStore, repo_root: &Path) -> Result<()> {
    store.verify_control_plane_invariants(repo_root)?;
    println!("theorem identity validation passed");
    Ok(())
}

fn cmd_requirements_mutation(
    store: &mut ProvenanceStore,
    repo_root: &Path,
    args: &RequirementsMutationArgs,
) -> Result<()> {
    let narrative_paths = load_requirements_narrative_paths(repo_root)?;
    match &args.action {
        RequirementsMutationAction::SetMeta {
            authoritative,
            status,
            status_token,
            updated,
            python_recommended,
            python_allowed,
            primary_markdown,
            status_allowlist,
            runtime_stack_allowlist,
            required_module_fields,
            required_gap_fields,
        } => {
            let existing = store.requirements_meta_row()?;
            let defaults = default_requirements_meta();
            let status_value = status
                .as_deref()
                .or_else(|| existing.as_ref().map(|row| row.status.as_str()))
                .unwrap_or(defaults.status);
            let status_token_value = status_token
                .as_deref()
                .map(|value| value.to_string())
                .unwrap_or_else(|| {
                    existing
                        .as_ref()
                        .map(|row| row.status_token.clone())
                        .unwrap_or_else(|| {
                            status_token_or_default(status_value, defaults.status_token)
                        })
                });
            let updated_value = updated
                .as_deref()
                .or_else(|| existing.as_ref().map(|row| row.updated.as_str()))
                .unwrap_or(defaults.updated)
                .to_string();
            let python_recommended_value = python_recommended
                .as_deref()
                .or_else(|| existing.as_ref().map(|row| row.python_recommended.as_str()))
                .unwrap_or(defaults.python_recommended)
                .to_string();
            let python_allowed_value = python_allowed
                .as_deref()
                .or_else(|| existing.as_ref().map(|row| row.python_allowed.as_str()))
                .unwrap_or(defaults.python_allowed)
                .to_string();
            let primary_markdown_value = primary_markdown
                .as_deref()
                .or_else(|| existing.as_ref().map(|row| row.primary_markdown.as_str()))
                .unwrap_or(defaults.primary_markdown)
                .to_string();
            ensure_requirements_narrative_path(
                repo_root,
                &narrative_paths,
                "primary_markdown",
                &primary_markdown_value,
            )?;

            let status_allowlist_json = if status_allowlist.is_empty() {
                existing
                    .as_ref()
                    .map(|row| row.status_allowlist_json.clone())
                    .unwrap_or_else(|| defaults.status_allowlist_json.to_string())
            } else {
                json_array(status_allowlist)?
            };
            let runtime_stack_allowlist_json = if runtime_stack_allowlist.is_empty() {
                existing
                    .as_ref()
                    .map(|row| row.runtime_stack_allowlist_json.clone())
                    .unwrap_or_else(|| defaults.runtime_stack_allowlist_json.to_string())
            } else {
                json_array(runtime_stack_allowlist)?
            };
            let required_module_fields_json = if required_module_fields.is_empty() {
                existing
                    .as_ref()
                    .map(|row| row.required_module_fields_json.clone())
                    .unwrap_or_else(|| defaults.required_module_fields_json.to_string())
            } else {
                json_array(required_module_fields)?
            };
            let required_gap_fields_json = if required_gap_fields.is_empty() {
                existing
                    .as_ref()
                    .map(|row| row.required_gap_fields_json.clone())
                    .unwrap_or_else(|| defaults.required_gap_fields_json.to_string())
            } else {
                json_array(required_gap_fields)?
            };

            store.upsert_requirements_meta(&provenance_store::RequirementsMeta {
                authoritative: authoritative
                    .or_else(|| existing.as_ref().map(|row| row.authoritative))
                    .unwrap_or(defaults.authoritative),
                status: status_value,
                status_token: &status_token_value,
                updated: &updated_value,
                python_recommended: &python_recommended_value,
                python_allowed: &python_allowed_value,
                primary_markdown: &primary_markdown_value,
                status_allowlist_json: &status_allowlist_json,
                runtime_stack_allowlist_json: &runtime_stack_allowlist_json,
                required_module_fields_json: &required_module_fields_json,
                required_gap_fields_json: &required_gap_fields_json,
            })?;
            println!("Updated requirements metadata.");
        }
        RequirementsMutationAction::UpsertModule {
            id,
            name,
            markdown,
            status,
            status_token,
            runtime_stack,
            requires_modules,
            install_targets,
            verify_targets,
            acceptance_criteria,
        } => {
            ensure_requirements_narrative_path(
                repo_root,
                &narrative_paths,
                "module.markdown",
                markdown,
            )?;
            let requires_modules_json = json_array(requires_modules)?;
            let install_targets_json = json_array(install_targets)?;
            let verify_targets_json = json_array(verify_targets)?;
            let acceptance_criteria_json = json_array(acceptance_criteria)?;
            let status_token = status_token_or_default(status, status_token);
            store.upsert_requirement_module(&provenance_store::RequirementModuleItem {
                id,
                name,
                markdown,
                status,
                status_token: &status_token,
                runtime_stack,
                requires_modules_json: &requires_modules_json,
                install_targets_json: &install_targets_json,
                verify_targets_json: &verify_targets_json,
                acceptance_criteria_json: &acceptance_criteria_json,
            })?;
            println!("Updated requirements module: {id}");
        }
        RequirementsMutationAction::DeleteModule { id } => {
            store.delete_requirement_module(id)?;
            println!("Deleted requirements module: {id}");
        }
        RequirementsMutationAction::UpsertGap {
            id,
            area,
            status,
            status_token,
            description,
            proposed_resolution,
            related_module_ids,
        } => {
            let related_module_ids_json = json_array(related_module_ids)?;
            let status_token = status_token_or_default(status, status_token);
            store.upsert_requirement_coverage_gap(
                &provenance_store::RequirementCoverageGapItem {
                    id,
                    area,
                    status,
                    status_token: &status_token,
                    description,
                    proposed_resolution,
                    related_module_ids_json: &related_module_ids_json,
                },
            )?;
            println!("Updated requirements coverage gap: {id}");
        }
        RequirementsMutationAction::DeleteGap { id } => {
            store.delete_requirement_coverage_gap(id)?;
            println!("Deleted requirements coverage gap: {id}");
        }
    }

    export_requirements_compat_file(store, repo_root)?;
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
    println!("  ── Generated planning / requirements views (DB-backed compatibility layer) ──");
    println!(
        "  These structured TOMLs are generated views. Update the canonical SQLite DB via `gororoba-db`, then regenerate exports."
    );
    println!();
    let planning_tomls = [
        "registry/roadmap.toml",
        "registry/todo.toml",
        "registry/next_actions.toml",
        "registry/requirements.toml",
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

fn render_roadmap_toml(store: &ProvenanceStore) -> Result<String> {
    store.render_planning_compat_toml(PlanningCompatTable::Roadmap)
}

fn render_todo_toml(store: &ProvenanceStore) -> Result<String> {
    store.render_planning_compat_toml(PlanningCompatTable::Todo)
}

fn render_next_actions_toml(store: &ProvenanceStore) -> Result<String> {
    store.render_planning_compat_toml(PlanningCompatTable::NextActions)
}

fn render_planning_toml(store: &ProvenanceStore, table: &PlanningTable) -> Result<String> {
    match table {
        PlanningTable::Roadmap => store.render_planning_compat_toml(PlanningCompatTable::Roadmap),
        PlanningTable::Todo => store.render_planning_compat_toml(PlanningCompatTable::Todo),
        PlanningTable::NextActions => {
            store.render_planning_compat_toml(PlanningCompatTable::NextActions)
        }
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

fn export_requirements_compat_file(store: &ProvenanceStore, repo_root: &Path) -> Result<()> {
    let requirements_path = repo_root.join("registry/requirements.toml");
    fs::write(
        &requirements_path,
        format!("{}\n", store.render_requirements_compat_toml()?),
    )
    .with_context(|| format!("write {}", requirements_path.display()))?;
    println!("  Requirements compatibility export refreshed:");
    println!("    {}", requirements_path.display());
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

    // build_fresh deletes the database file, and the importer below reads only the
    // lanes named in registry/source_manifest.toml. Claim transition events, the
    // relations they allocate, and standalone claim revisions have no compatibility
    // TOML in that manifest. A rebuild recreates those tables empty. The append-only
    // triggers cannot catch file removal because it issues no DELETE. Refuse rather
    // than destroy, and require an explicit acknowledgement of that loss.
    if db_path.exists() {
        let existing = ProvenanceStore::open(db_path)
            .with_context(|| format!("open existing db {}", db_path.display()))?;
        let event_count = existing.list_claim_transition_events()?.len();
        let revision_count = existing.table_row_count("claim_revisions")?;
        if claim_history_requires_loss_acknowledgement(event_count, revision_count)
            && !args.allow_transition_history_loss
        {
            bail!(
                "refusing to rebuild {}: it holds {event_count} claim transition events and \
                 {revision_count} claim revisions that no compatibility TOML can restore. \
                 Rebuilding recreates the event, relation and revision tables empty. Export \
                 instead of rebuilding, or pass \
                 --allow-transition-history-loss if discarding the adjudication history is \
                 intended.",
                db_path.display()
            );
        }
    }

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
            provenance_store::RegistryImportPaths {
                claims: &claims_path,
                insights: &insights_path,
                experiments: &experiments_path,
                binaries: &binaries_path,
                rocq_project: &proofs_project_path,
            },
            // build_fresh deleted the file, so the tables are empty and the
            // bootstrap importer is the correct mode here.
            provenance_store::ReimportOptions::bootstrap(),
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

    // Ingest external-source contracts and dossiers. build_fresh deletes the
    // prior database file, so the external-source control plane must be
    // reimported from its committed compatibility exports on every build;
    // otherwise a routine rebuild silently erases the canonical contract and
    // dossier tables.
    let source_contracts_path = repo_root.join("data/external/SOURCES.toml");
    let dossiers_path = repo_root.join("registry/external_sources.toml");
    if source_contracts_path.exists() && dossiers_path.exists() {
        let (contract_count, dossier_count) = store.reindex_external_sources_from_compat(
            repo_root,
            &source_contracts_path,
            &dossiers_path,
        )?;
        println!("  External sources: {contract_count} contracts, {dossier_count} dossiers");
        if contract_count == 0 {
            anyhow::bail!(
                "external-source reimport produced zero contracts while {} exists",
                source_contracts_path.display()
            );
        }
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
    let requirements_args = ImportRequirementsArgs {
        requirements: PathBuf::from("registry/requirements.toml"),
    };
    cmd_import_planning(&mut store, repo_root, &planning_args)?;
    cmd_import_requirements(&mut store, repo_root, &requirements_args)?;
    export_planning_compat_files(&store, repo_root)?;
    export_requirements_compat_file(&store, repo_root)?;

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

fn claim_history_requires_loss_acknowledgement(event_count: usize, revision_count: i64) -> bool {
    event_count != 0 || revision_count != 0
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

    #[test]
    fn revision_only_history_requires_loss_acknowledgement() {
        assert!(claim_history_requires_loss_acknowledgement(0, 1));
        assert!(claim_history_requires_loss_acknowledgement(1, 0));
        assert!(!claim_history_requires_loss_acknowledgement(0, 0));
    }
}
