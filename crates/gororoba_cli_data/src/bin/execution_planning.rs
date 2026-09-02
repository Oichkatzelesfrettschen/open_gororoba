use anyhow::{Context, Result, bail};
use clap::Parser;
use provenance_store::{ControlPlaneCompatKind, PlanningCompatTable, ProvenanceStore};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
// PathBuf is only used by the cfg(test) test workspace below;
// gating the import keeps the non-test build warning-free.
#[cfg(test)]
use std::path::PathBuf;
use toml::Value;

// Type definitions live in the `types` submodule (~165 lines).
// Uses `#[path]` because this binary has an explicit Cargo.toml path.
#[path = "execution_planning/types.rs"]
mod types;
use types::*;

fn experiment_binary_overrides() -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "E-021".to_string(),
            "tessarines-mixed-quaternions-census".to_string(),
        ),
        (
            "E-022".to_string(),
            "albert-algebra-structure-census".to_string(),
        ),
        (
            "E-023".to_string(),
            "composition-algebra-taxonomy".to_string(),
        ),
        ("E-024".to_string(), "registry-event-tracker".to_string()),
        (
            "E-026".to_string(),
            "third-party-source-verifier".to_string(),
        ),
    ])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let repo_root = args.repo_root.canonicalize().context("resolve repo root")?;
    if args.verify {
        return verify_execution_planning(&repo_root, &args);
    }
    build_execution_planning(&repo_root, &args)
}

fn build_execution_planning(repo_root: &Path, args: &Args) -> Result<()> {
    let claims = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Claims,
            "registry/claims.toml",
        )?,
        "claim",
    )?;
    let claim_ids = claims
        .iter()
        .map(|row| string_field(row, "id"))
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>();

    let binaries_rows = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Binaries,
            "registry/binaries.toml",
        )?,
        "binary",
    )?;
    let mut binaries = BTreeMap::new();
    for row in binaries_rows {
        let name = string_field(&row, "name");
        if !name.is_empty() {
            binaries.insert(name, row);
        }
    }
    let bench_targets = load_workspace_bench_targets(repo_root)?;

    let dataset_path_index = load_dataset_path_index(repo_root)?;
    let dataset_label_aliases = load_dataset_label_aliases(repo_root)?;
    let dataset_ids = dataset_path_index
        .values()
        .cloned()
        .collect::<BTreeSet<_>>();
    let experiments_input = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Experiments,
            "registry/experiments.toml",
        )?,
        "experiment",
    )?;

    let experiment_rows = build_experiment_rows(
        &experiments_input,
        &binaries,
        &bench_targets,
        &dataset_path_index,
        &dataset_label_aliases,
    )?;
    let (lineage_rows, lineage_edges) =
        build_experiment_lineage(&experiment_rows, &claim_ids, &dataset_ids);

    let roadmap_raw = load_toml(&repo_root.join("registry/roadmap.toml"))?;
    let mut roadmap_meta = table_value(&roadmap_raw, "roadmap");
    let sections = if let Some(roadmap) = roadmap_raw.get("roadmap").and_then(Value::as_table) {
        roadmap
            .get("sections")
            .and_then(Value::as_table)
            .cloned()
            .unwrap_or_default()
    } else {
        roadmap_raw
            .get("sections")
            .and_then(Value::as_table)
            .cloned()
            .unwrap_or_default()
    };
    roadmap_meta.insert("sections".to_string(), Value::Table(sections));
    let roadmap_rows = build_hardened_planning_rows(
        &table_array(&roadmap_raw, "workstream")?,
        "workstream",
        "id",
        ROADMAP_STATUS_ALLOWLIST,
        PLANNING_PRIORITY_ALLOWLIST,
    );

    let todo_raw = load_toml(&repo_root.join("registry/todo.toml"))?;
    let todo_rows = build_hardened_planning_rows(
        &table_array(&todo_raw, "item")?,
        "todo_item",
        "id",
        TODO_STATUS_ALLOWLIST,
        PLANNING_PRIORITY_ALLOWLIST,
    );

    let actions_raw = load_toml(&repo_root.join("registry/next_actions.toml"))?;
    let action_rows = build_hardened_planning_rows(
        &table_array(&actions_raw, "action")?,
        "next_action",
        "id",
        ACTION_STATUS_ALLOWLIST,
        PLANNING_PRIORITY_ALLOWLIST,
    );

    let requirements_raw = load_toml(&repo_root.join("registry/requirements.toml"))?;
    let req_meta = table_value(&requirements_raw, "requirements");
    let (req_modules, req_gaps) = build_requirements_hardened(
        &table_array(&requirements_raw, "module")?,
        &table_array(&requirements_raw, "coverage_gap")?,
    );
    let pyproject = load_toml(&repo_root.join("pyproject.toml"))?;
    let cargo_toml = load_toml(&repo_root.join("Cargo.toml"))?;
    let (module_rows, package_rows, command_rows) =
        build_module_requirements(&req_modules, &pyproject, &cargo_toml)?;
    let experiments_text = render_experiments(&experiment_rows)? + "\n";

    let roadmap_text =
        load_planning_compat_export(repo_root, &args.db, PlanningCompatTable::Roadmap)?
            .unwrap_or(render_roadmap(&roadmap_meta, &roadmap_rows)?);
    let todo_text = load_planning_compat_export(repo_root, &args.db, PlanningCompatTable::Todo)?
        .unwrap_or(render_todo(&todo_rows)?);
    let next_actions_text =
        load_planning_compat_export(repo_root, &args.db, PlanningCompatTable::NextActions)?
            .unwrap_or(render_next_actions(&action_rows)?);
    let requirements_text = load_requirements_compat_export(repo_root, &args.db)?
        .unwrap_or(render_requirements(&req_modules, &req_gaps, &req_meta)?);

    sync_or_write_experiments_registry(repo_root, args, &experiments_text)?;
    write_ascii(
        &repo_root.join(&args.lineage_out),
        &(render_experiment_lineage(&lineage_rows, &lineage_edges)? + "\n"),
    )?;
    write_ascii(&repo_root.join(&args.roadmap_out), &(roadmap_text + "\n"))?;
    write_ascii(&repo_root.join(&args.todo_out), &(todo_text + "\n"))?;
    write_ascii(
        &repo_root.join(&args.next_actions_out),
        &(next_actions_text + "\n"),
    )?;
    write_ascii(
        &repo_root.join(&args.requirements_out),
        &(requirements_text + "\n"),
    )?;
    write_ascii(
        &repo_root.join(&args.module_requirements_out),
        &(render_module_requirements(&module_rows, &package_rows, &command_rows)? + "\n"),
    )?;

    println!(
        "Wrote execution-planning registry lane artifacts (canonical registry-build-execution-planning script): experiments={} lineages={} lineage_edges={} roadmap={} todo={} next_actions={} req_modules={} module_packages={} module_commands={}",
        experiment_rows.len(),
        lineage_rows.len(),
        lineage_edges.len(),
        roadmap_rows.len(),
        todo_rows.len(),
        action_rows.len(),
        req_modules.len(),
        package_rows.len(),
        command_rows.len()
    );
    Ok(())
}

fn verify_execution_planning(repo_root: &Path, args: &Args) -> Result<()> {
    let required = [
        repo_root.join(&args.experiments_out),
        repo_root.join(&args.lineage_out),
        repo_root.join(&args.roadmap_out),
        repo_root.join(&args.todo_out),
        repo_root.join(&args.next_actions_out),
        repo_root.join(&args.requirements_out),
        repo_root.join(&args.module_requirements_out),
        repo_root.join("registry/external_sources.toml"),
        repo_root.join("registry/dataset_label_aliases.toml"),
        repo_root.join("data/external/SOURCES.toml"),
    ];
    for path in &required {
        if !path.exists() {
            bail!("ERROR: missing required registry {}", path.display());
        }
    }
    for path in &required {
        assert_ascii_file(path)?;
    }

    let claims = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Claims,
            "registry/claims.toml",
        )?,
        "claim",
    )?;
    let insights = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Insights,
            "registry/insights.toml",
        )?,
        "insight",
    )?;
    let binaries = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Binaries,
            "registry/binaries.toml",
        )?,
        "binary",
    )?;
    let experiments_raw = load_toml(&repo_root.join(&args.experiments_out))?;
    let lineage_raw = load_toml(&repo_root.join(&args.lineage_out))?;
    let roadmap_raw = load_toml(&repo_root.join(&args.roadmap_out))?;
    let todo_raw = load_toml(&repo_root.join(&args.todo_out))?;
    let actions_raw = load_toml(&repo_root.join(&args.next_actions_out))?;
    let requirements_raw = load_toml(&repo_root.join(&args.requirements_out))?;
    let module_requirements_raw = load_toml(&repo_root.join(&args.module_requirements_out))?;
    let dataset_label_aliases_raw =
        load_toml(&repo_root.join("registry/dataset_label_aliases.toml"))?;
    let _ = load_toml(&repo_root.join("registry/external_sources.toml"))?;

    let claim_ids = ids_from_rows(&claims);
    let insight_ids = ids_from_rows(&insights);
    let dataset_ids = collect_dataset_ids(repo_root)?;
    let source_ids = collect_source_ids(repo_root)?;
    let mut dataset_label_aliases = BTreeMap::new();
    for row in table_array(&dataset_label_aliases_raw, "alias")? {
        let mut label = normalize_dataset_label(&string_field(&row, "label_normalized"));
        if label.is_empty() {
            label = normalize_dataset_label(&string_field(&row, "label"));
        }
        let canonical_dataset_id = string_field(&row, "canonical_dataset_id");
        if !label.is_empty() && !canonical_dataset_id.is_empty() {
            dataset_label_aliases.insert(label, canonical_dataset_id);
        }
    }

    let binary_names = binaries
        .iter()
        .map(|row| string_field(row, "name"))
        .filter(|name| !name.is_empty())
        .collect::<BTreeSet<_>>();
    let bench_target_names = load_workspace_bench_targets(repo_root)?;
    let execution_target_names = binary_names
        .iter()
        .cloned()
        .chain(bench_target_names.iter().cloned())
        .collect::<BTreeSet<_>>();
    let mut binary_experiment = BTreeMap::new();
    for row in &binaries {
        let name = string_field(row, "name");
        if !name.is_empty() {
            binary_experiment.insert(name, string_field(row, "experiment"));
        }
    }
    // Experiments sharing one execution target, keyed by its head token. A
    // dispatcher answers to `turboquant bench` and `turboquant validate` alike,
    // so the head is what the binaries registry knows and the peer set is what
    // an experiment's declared attribution is checked against.
    let mut experiments_by_execution_target: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    let experiments_meta = table_value(&experiments_raw, "experiments");
    let experiments = table_array(&experiments_raw, "experiment")?;
    for row in &experiments {
        let binary = string_field(row, "binary");
        if !binary.is_empty() {
            experiments_by_execution_target
                .entry(execution_target_head(&binary).to_string())
                .or_default()
                .insert(string_field(row, "id"));
        }
    }
    let lineages_meta = table_value(&lineage_raw, "experiment_lineage");
    let lineages = table_array(&lineage_raw, "lineage")?;
    let edges = table_array(&lineage_raw, "edge")?;
    let workstreams = table_array(&roadmap_raw, "workstream")?;
    let todo_items = table_array(&todo_raw, "item")?;
    let actions = table_array(&actions_raw, "action")?;
    let req_modules = table_array(&requirements_raw, "module")?;
    let req_gaps = table_array(&requirements_raw, "coverage_gap")?;
    let mr_meta = table_value(&module_requirements_raw, "module_requirements");
    let mr_modules = table_array(&module_requirements_raw, "module")?;
    let mr_commands = table_array(&module_requirements_raw, "command")?;
    let mr_packages = table_array(&module_requirements_raw, "package")?;

    let workstream_ids = ids_from_rows(&workstreams);
    let todo_ids = ids_from_rows(&todo_items);
    let action_ids = ids_from_rows(&actions);
    let req_ids = ids_from_rows(&req_modules);

    let mut failures = Vec::new();

    if integer_field(&experiments_meta, "experiment_count", -1) != experiments.len() as i64 {
        failures.push("experiments experiment_count metadata mismatch".to_string());
    }
    if integer_field(&experiments_meta, "deterministic_count", -1)
        != experiments
            .iter()
            .filter(|row| bool_field(row, "deterministic"))
            .count() as i64
    {
        failures.push("experiments deterministic_count metadata mismatch".to_string());
    }
    if integer_field(&experiments_meta, "gpu_count", -1)
        != experiments
            .iter()
            .filter(|row| bool_field(row, "gpu"))
            .count() as i64
    {
        failures.push("experiments gpu_count metadata mismatch".to_string());
    }
    if integer_field(&experiments_meta, "seeded_count", -1)
        != experiments
            .iter()
            .filter(|row| row.get("seed").is_some())
            .count() as i64
    {
        failures.push("experiments seeded_count metadata mismatch".to_string());
    }

    let exp_status_allow = string_list_from_table(&experiments_meta, "status_allowlist")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let exp_ids = ids_from_rows(&experiments);
    let exp_by_id = rows_by_id(&experiments);
    let lineage_ids = ids_from_rows(&lineages);
    let mut seen_exp_ids = BTreeSet::new();
    for row in &experiments {
        let eid = string_field(row, "id");
        if !seen_exp_ids.insert(eid.clone()) {
            failures.push(format!("duplicate experiment id: {eid}"));
            continue;
        }
        let status = string_field(row, "status");
        if !exp_status_allow.contains(&status) {
            failures.push(format!(
                "experiment[{eid}] status outside allowlist: {status}"
            ));
        }
        if string_field(row, "status_token") != verify_status_token(&status) {
            failures.push(format!("experiment[{eid}] status_token mismatch"));
        }
        if !lineage_ids.contains(&string_field(row, "lineage_id")) {
            failures.push(format!("experiment[{eid}] unknown lineage_id"));
        }
        let run_cmd = string_field(row, "run");
        let run_sha = string_field(row, "run_command_sha256");
        let expected_sha = sha256_hex(run_cmd.as_bytes());
        if run_sha != expected_sha {
            failures.push(format!("experiment[{eid}] run_command_sha256 mismatch"));
        }

        let binary = string_field(row, "binary");
        let binary_registered = bool_field(row, "binary_registered");
        if binary_registered && !execution_target_registered(&binary, &execution_target_names) {
            failures.push(format!(
                "experiment[{eid}] binary marked registered but missing: {binary}"
            ));
        }
        if !binary_registered && execution_target_registered(&binary, &execution_target_names) {
            failures.push(format!(
                "experiment[{eid}] binary marked unregistered but exists: {binary}"
            ));
        }
        let declared = string_field(row, "binary_experiment_declared");
        let registered_experiment = binary_experiment
            .get(&binary)
            .or_else(|| binary_experiment.get(execution_target_head(&binary)))
            .cloned()
            .unwrap_or_default();
        if !declared.is_empty() {
            if !registered_experiment.is_empty() {
                if declared != registered_experiment {
                    failures.push(format!(
                        "experiment[{eid}] binary_experiment_declared mismatch: {declared}"
                    ));
                }
            } else if declared != eid
                && !experiments_by_execution_target
                    .get(execution_target_head(&binary))
                    .is_some_and(|peers| peers.contains(&declared))
            {
                // The binaries registry attributes no experiment to this target,
                // so the declaration stands if the experiment credits itself or
                // credits another experiment running the same target. A
                // dispatcher serves many lanes, and demanding self-declaration
                // there would reject every lane that credits the experiment
                // which first registered the target.
                failures.push(format!(
                    "experiment[{eid}] binary_experiment_declared names no experiment on {binary}: {declared}"
                ));
            }
        }
        let claims_refs = string_list_field(row, "claim_refs")
            .into_iter()
            .collect::<BTreeSet<_>>();
        let claims_legacy = string_list_field(row, "claims")
            .into_iter()
            .collect::<BTreeSet<_>>();
        if claims_refs != claims_legacy {
            failures.push(format!("experiment[{eid}] claim_refs and claims diverge"));
        }
        for cid in claims_refs {
            if !claim_ids.contains(&cid) {
                failures.push(format!("experiment[{eid}] unknown claim ref: {cid}"));
            }
        }
        for did in string_list_field(row, "dataset_refs") {
            if !did.is_empty() && !dataset_ids.contains(&did) {
                failures.push(format!("experiment[{eid}] unknown dataset ref: {did}"));
            }
        }
        for label in string_list_field(row, "dataset_label_refs") {
            if !dataset_label_aliases.contains_key(&normalize_dataset_label(&label)) {
                failures.push(format!(
                    "experiment[{eid}] unknown dataset label ref: {label}"
                ));
            }
        }
        for xid in string_list_field(row, "external_source_refs") {
            if !xid.is_empty() && !source_ids.contains(&xid) {
                failures.push(format!(
                    "experiment[{eid}] unknown external source ref: {xid}"
                ));
            }
        }
        for surface in string_list_field(row, "truth_surface_consumption") {
            if !TRUTH_SURFACE_ALLOWLIST.contains(&surface.as_str()) {
                failures.push(format!(
                    "experiment[{eid}] invalid truth_surface_consumption: {surface}"
                ));
            }
        }
    }

    if integer_field(&lineages_meta, "lineage_count", -1) != lineages.len() as i64 {
        failures.push("experiment_lineage lineage_count metadata mismatch".to_string());
    }
    if integer_field(&lineages_meta, "edge_count", -1) != edges.len() as i64 {
        failures.push("experiment_lineage edge_count metadata mismatch".to_string());
    }

    let mut by_experiment = BTreeMap::new();
    for row in &lineages {
        by_experiment.insert(string_field(row, "experiment_id"), row.clone());
    }
    for eid in &exp_ids {
        if !by_experiment.contains_key(eid) {
            failures.push(format!("missing lineage row for experiment: {eid}"));
        }
    }
    let mut seen_lineage_ids = BTreeSet::new();
    let mut lineage_to_experiment = BTreeMap::new();
    for row in &lineages {
        let lid = string_field(row, "id");
        let eid = string_field(row, "experiment_id");
        let exp_row = exp_by_id.get(&eid);
        let expected_binary = exp_row
            .map(|row| string_field(row, "binary"))
            .unwrap_or_default();
        let lineage_binary = string_field(row, "binary");
        if !seen_lineage_ids.insert(lid.clone()) {
            failures.push(format!("duplicate lineage id: {lid}"));
            continue;
        }
        lineage_to_experiment.insert(lid.clone(), eid.clone());
        if !exp_ids.contains(&eid) {
            failures.push(format!("lineage[{lid}] unknown experiment_id: {eid}"));
        }
        if expected_binary != lineage_binary {
            failures.push(format!(
                "lineage[{lid}] binary mismatch: expected {}",
                if expected_binary.is_empty() {
                    "<none>"
                } else {
                    &expected_binary
                }
            ));
        }
        if !lineage_binary.is_empty()
            && !execution_target_registered(&lineage_binary, &execution_target_names)
        {
            failures.push(format!("lineage[{lid}] unknown binary"));
        }
        if string_field(row, "run_command_sha256")
            != sha256_hex(string_field(row, "run_command").as_bytes())
        {
            failures.push(format!("lineage[{lid}] run_command_sha256 mismatch"));
        }
        for cid in string_list_field(row, "claim_refs") {
            if !claim_ids.contains(&cid) {
                failures.push(format!("lineage[{lid}] unknown claim ref: {cid}"));
            }
        }
        for did in string_list_field(row, "dataset_refs") {
            if !dataset_ids.contains(&did) {
                failures.push(format!("lineage[{lid}] unknown dataset ref: {did}"));
            }
        }
        let exp_dataset_labels = exp_row
            .map(|row| string_list_field(row, "dataset_label_refs"))
            .unwrap_or_default();
        if string_list_field(row, "dataset_label_refs") != exp_dataset_labels {
            failures.push(format!("lineage[{lid}] dataset_label_refs mismatch"));
        }
        for label in string_list_field(row, "dataset_label_refs") {
            if !dataset_label_aliases.contains_key(&normalize_dataset_label(&label)) {
                failures.push(format!("lineage[{lid}] unknown dataset label ref: {label}"));
            }
        }
        let exp_source_refs = exp_row
            .map(|row| string_list_field(row, "external_source_refs"))
            .unwrap_or_default();
        if string_list_field(row, "external_source_refs") != exp_source_refs {
            failures.push(format!("lineage[{lid}] external_source_refs mismatch"));
        }
        for xid in string_list_field(row, "external_source_refs") {
            if !source_ids.contains(&xid) {
                failures.push(format!("lineage[{lid}] unknown external source ref: {xid}"));
            }
        }
        let exp_truth_surfaces = exp_row
            .map(|row| string_list_field(row, "truth_surface_consumption"))
            .unwrap_or_default();
        if string_list_field(row, "truth_surface_consumption") != exp_truth_surfaces {
            failures.push(format!("lineage[{lid}] truth_surface_consumption mismatch"));
        }
        for surface in string_list_field(row, "truth_surface_consumption") {
            if !TRUTH_SURFACE_ALLOWLIST.contains(&surface.as_str()) {
                failures.push(format!(
                    "lineage[{lid}] invalid truth_surface_consumption: {surface}"
                ));
            }
        }
    }

    let edge_kinds = BTreeSet::from([
        "implemented_by_binary".to_string(),
        "supports_claim".to_string(),
        "touches_dataset".to_string(),
        "consumes_path".to_string(),
        "produces_path".to_string(),
        "consumes_source".to_string(),
        "consumes_truth_surface".to_string(),
    ]);
    let to_kinds = BTreeSet::from([
        "binary".to_string(),
        "claim".to_string(),
        "dataset".to_string(),
        "path".to_string(),
        "source".to_string(),
        "truth_surface".to_string(),
    ]);
    let mut edge_id_seen = BTreeSet::new();
    let mut binary_edge_lineages = BTreeSet::new();
    let mut source_edge_refs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut truth_surface_edge_refs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for row in &edges {
        let edge_id = string_field(row, "id");
        if !edge_id_seen.insert(edge_id.clone()) {
            failures.push(format!("duplicate lineage edge id: {edge_id}"));
            continue;
        }
        let lid = string_field(row, "lineage_id");
        let eid = string_field(row, "from_id");
        let to_ref = string_field(row, "to_ref");
        let to_kind = string_field(row, "to_kind");
        let edge_kind = string_field(row, "edge_kind");
        if !lineage_to_experiment.contains_key(&lid) {
            failures.push(format!("lineage edge[{edge_id}] unknown lineage_id: {lid}"));
        }
        if !exp_ids.contains(&eid) {
            failures.push(format!("lineage edge[{edge_id}] unknown from_id: {eid}"));
        } else if lineage_to_experiment.get(&lid).cloned().unwrap_or_default() != eid {
            failures.push(format!(
                "lineage edge[{edge_id}] from_id does not match lineage experiment: {lid}"
            ));
        }
        if !to_kinds.contains(&to_kind) {
            failures.push(format!(
                "lineage edge[{edge_id}] invalid to_kind: {to_kind}"
            ));
        }
        if !edge_kinds.contains(&edge_kind) {
            failures.push(format!(
                "lineage edge[{edge_id}] invalid edge_kind: {edge_kind}"
            ));
        }
        match to_kind.as_str() {
            "binary" => {
                let expected_binary = exp_by_id
                    .get(&eid)
                    .map(|row| string_field(row, "binary"))
                    .unwrap_or_default();
                if to_ref.is_empty() {
                    if !expected_binary.is_empty() {
                        failures.push(format!("lineage edge[{edge_id}] empty binary ref"));
                    }
                } else {
                    if !execution_target_registered(&to_ref, &execution_target_names) {
                        failures.push(format!(
                            "lineage edge[{edge_id}] unknown binary ref: {to_ref}"
                        ));
                    }
                    if !expected_binary.is_empty() && to_ref != expected_binary {
                        failures.push(format!(
                            "lineage edge[{edge_id}] binary ref mismatch: expected {expected_binary}"
                        ));
                    }
                    binary_edge_lineages.insert(lid.clone());
                }
            }
            "claim" if !claim_ids.contains(&to_ref) => {
                failures.push(format!(
                    "lineage edge[{edge_id}] unknown claim ref: {to_ref}"
                ));
            }
            "dataset" if !dataset_ids.contains(&to_ref) => {
                failures.push(format!(
                    "lineage edge[{edge_id}] unknown dataset ref: {to_ref}"
                ));
            }
            "path" if to_ref.is_empty() => {
                failures.push(format!("lineage edge[{edge_id}] empty path ref"));
            }
            "claim" | "dataset" | "path" => {}
            "source" => {
                if !source_ids.contains(&to_ref) {
                    failures.push(format!(
                        "lineage edge[{edge_id}] unknown source ref: {to_ref}"
                    ));
                }
                source_edge_refs
                    .entry(lid.clone())
                    .or_default()
                    .insert(to_ref);
            }
            "truth_surface" => {
                if !TRUTH_SURFACE_ALLOWLIST.contains(&to_ref.as_str()) {
                    failures.push(format!(
                        "lineage edge[{edge_id}] invalid truth surface ref: {to_ref}"
                    ));
                }
                truth_surface_edge_refs
                    .entry(lid.clone())
                    .or_default()
                    .insert(to_ref);
            }
            _ => {}
        }
    }
    for lid in &seen_lineage_ids {
        let exp_row = lineage_to_experiment
            .get(lid)
            .and_then(|eid| exp_by_id.get(eid));
        let exp_binary = exp_row
            .map(|row| string_field(row, "binary"))
            .unwrap_or_default();
        let exp_sources = exp_row
            .map(|row| string_list_field(row, "external_source_refs"))
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let exp_truth_surfaces = exp_row
            .map(|row| string_list_field(row, "truth_surface_consumption"))
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        if !exp_binary.is_empty() && !binary_edge_lineages.contains(lid) {
            failures.push(format!("lineage[{lid}] missing binary edge"));
        }
        if exp_sources != source_edge_refs.get(lid).cloned().unwrap_or_default() {
            failures.push(format!(
                "lineage[{lid}] source edges do not match experiment refs"
            ));
        }
        if exp_truth_surfaces
            != truth_surface_edge_refs
                .get(lid)
                .cloned()
                .unwrap_or_default()
        {
            failures.push(format!(
                "lineage[{lid}] truth-surface edges do not match experiment refs"
            ));
        }
    }

    let roadmap_meta = table_value(&roadmap_raw, "roadmap");
    let roadmap_status_allow = string_list_from_table(&roadmap_meta, "status_allowlist")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let roadmap_priority_allow = string_list_from_table(&roadmap_meta, "priority_allowlist")
        .into_iter()
        .collect::<BTreeSet<_>>();
    if integer_field(&roadmap_meta, "workstream_count", -1) != workstreams.len() as i64 {
        failures.push("roadmap workstream_count metadata mismatch".to_string());
    }
    for row in &workstreams {
        let wid = string_field(row, "id");
        let status = string_field(row, "status");
        let priority = string_field(row, "priority");
        let deps = string_list_field(row, "dependencies");
        let acceptance = list_field(row, "acceptance_criteria");
        if !roadmap_status_allow.contains(&status) {
            failures.push(format!(
                "workstream[{wid}] status outside allowlist: {status}"
            ));
        }
        if !roadmap_priority_allow.contains(&priority) {
            failures.push(format!(
                "workstream[{wid}] priority outside allowlist: {priority}"
            ));
        }
        if string_field(row, "status_token") != verify_status_token(&status) {
            failures.push(format!("workstream[{wid}] status_token mismatch"));
        }
        if acceptance.is_empty() {
            failures.push(format!("workstream[{wid}] missing acceptance_criteria"));
        }
        verify_dependencies(
            &mut failures,
            &deps,
            &format!("workstream[{wid}].dependencies"),
            &DependencyIdSets {
                claim_ids: &claim_ids,
                insight_ids: &insight_ids,
                experiment_ids: &exp_ids,
                workstream_ids: &workstream_ids,
                todo_ids: &todo_ids,
                action_ids: &action_ids,
                req_ids: &req_ids,
            },
        );
    }

    let todo_meta = table_value(&todo_raw, "todo");
    let todo_status_allow = string_list_from_table(&todo_meta, "status_allowlist")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let todo_priority_allow = string_list_from_table(&todo_meta, "priority_allowlist")
        .into_iter()
        .collect::<BTreeSet<_>>();
    if integer_field(&todo_meta, "item_count", -1) != todo_items.len() as i64 {
        failures.push("todo item_count metadata mismatch".to_string());
    }
    for row in &todo_items {
        let tid = string_field(row, "id");
        let status = string_field(row, "status");
        let priority = string_field(row, "priority");
        let deps = string_list_field(row, "dependencies");
        let acceptance = list_field(row, "acceptance_criteria");
        if !todo_status_allow.contains(&status) {
            failures.push(format!("todo[{tid}] status outside allowlist: {status}"));
        }
        if !todo_priority_allow.contains(&priority) {
            failures.push(format!(
                "todo[{tid}] priority outside allowlist: {priority}"
            ));
        }
        if string_field(row, "status_token") != verify_status_token(&status) {
            failures.push(format!("todo[{tid}] status_token mismatch"));
        }
        if acceptance.is_empty() {
            failures.push(format!("todo[{tid}] missing acceptance_criteria"));
        }
        verify_dependencies(
            &mut failures,
            &deps,
            &format!("todo[{tid}].dependencies"),
            &DependencyIdSets {
                claim_ids: &claim_ids,
                insight_ids: &insight_ids,
                experiment_ids: &exp_ids,
                workstream_ids: &workstream_ids,
                todo_ids: &todo_ids,
                action_ids: &action_ids,
                req_ids: &req_ids,
            },
        );
    }

    let actions_meta = table_value(&actions_raw, "meta");
    let actions_status_allow = string_list_from_table(&actions_meta, "status_allowlist")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let actions_priority_allow = string_list_from_table(&actions_meta, "priority_allowlist")
        .into_iter()
        .collect::<BTreeSet<_>>();
    if integer_field(&actions_meta, "action_count", -1) != actions.len() as i64 {
        failures.push("next_actions action_count metadata mismatch".to_string());
    }
    for row in &actions {
        let aid = string_field(row, "id");
        let status = string_field(row, "status");
        let priority = string_field(row, "priority");
        let deps = string_list_field(row, "dependencies");
        let acceptance = list_field(row, "acceptance_criteria");
        if !actions_status_allow.contains(&status) {
            failures.push(format!(
                "next_actions[{aid}] status outside allowlist: {status}"
            ));
        }
        if !actions_priority_allow.contains(&priority) {
            failures.push(format!(
                "next_actions[{aid}] priority outside allowlist: {priority}"
            ));
        }
        if string_field(row, "status_token") != verify_status_token(&status) {
            failures.push(format!("next_actions[{aid}] status_token mismatch"));
        }
        if acceptance.is_empty() {
            failures.push(format!("next_actions[{aid}] missing acceptance_criteria"));
        }
        verify_dependencies(
            &mut failures,
            &deps,
            &format!("next_actions[{aid}].dependencies"),
            &DependencyIdSets {
                claim_ids: &claim_ids,
                insight_ids: &insight_ids,
                experiment_ids: &exp_ids,
                workstream_ids: &workstream_ids,
                todo_ids: &todo_ids,
                action_ids: &action_ids,
                req_ids: &req_ids,
            },
        );
    }

    let requirements_meta = table_value(&requirements_raw, "requirements");
    let req_status_allow = string_list_from_table(&requirements_meta, "status_allowlist")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let runtime_allow = string_list_from_table(&requirements_meta, "runtime_stack_allowlist")
        .into_iter()
        .collect::<BTreeSet<_>>();
    if integer_field(&requirements_meta, "module_count", -1) != req_modules.len() as i64 {
        failures.push("requirements module_count metadata mismatch".to_string());
    }
    if integer_field(&requirements_meta, "coverage_gap_count", -1) != req_gaps.len() as i64 {
        failures.push("requirements coverage_gap_count metadata mismatch".to_string());
    }
    for row in &req_modules {
        let rid = string_field(row, "id");
        let status = string_field(row, "status");
        let runtime_stack = string_field(row, "runtime_stack");
        if !req_status_allow.contains(&status) {
            failures.push(format!(
                "requirements.module[{rid}] invalid status: {status}"
            ));
        }
        if string_field(row, "status_token") != verify_status_token(&status) {
            failures.push(format!("requirements.module[{rid}] status_token mismatch"));
        }
        if !runtime_allow.contains(&runtime_stack) {
            failures.push(format!(
                "requirements.module[{rid}] invalid runtime_stack: {runtime_stack}"
            ));
        }
        for dep_id in string_list_field(row, "requires_modules") {
            if !req_ids.contains(&dep_id) {
                failures.push(format!(
                    "requirements.module[{rid}] unknown requires_modules ref: {dep_id}"
                ));
            }
        }
        for field in ["install_targets", "verify_targets", "acceptance_criteria"] {
            if !matches!(row.get(field), Some(Value::Array(_))) {
                failures.push(format!("requirements.module[{rid}] {field} must be list"));
            }
        }
    }
    for row in &req_gaps {
        let gid = string_field(row, "id");
        let gap_status = string_field(row, "status");
        if !["open", "in_progress", "resolved", "blocked", "deferred"]
            .contains(&gap_status.as_str())
        {
            failures.push(format!(
                "requirements.coverage_gap[{gid}] invalid status: {gap_status}"
            ));
        }
        if string_field(row, "status_token") != verify_status_token(&gap_status) {
            failures.push(format!(
                "requirements.coverage_gap[{gid}] status_token mismatch"
            ));
        }
        for module_id in string_list_field(row, "related_module_ids") {
            if !module_id.is_empty() && !req_ids.contains(&module_id) {
                failures.push(format!(
                    "requirements.coverage_gap[{gid}] unknown related_module_id: {module_id}"
                ));
            }
        }
    }

    if integer_field(&mr_meta, "module_count", -1) != mr_modules.len() as i64 {
        failures.push("module_requirements module_count metadata mismatch".to_string());
    }
    if integer_field(&mr_meta, "package_count", -1) != mr_packages.len() as i64 {
        failures.push("module_requirements package_count metadata mismatch".to_string());
    }
    if integer_field(&mr_meta, "command_count", -1) != mr_commands.len() as i64 {
        failures.push("module_requirements command_count metadata mismatch".to_string());
    }
    if integer_field(&mr_meta, "python_package_count", -1)
        != mr_packages
            .iter()
            .filter(|row| string_field(row, "manager") == "pip")
            .count() as i64
    {
        failures.push("module_requirements python_package_count metadata mismatch".to_string());
    }
    if integer_field(&mr_meta, "rust_package_count", -1)
        != mr_packages
            .iter()
            .filter(|row| string_field(row, "manager") == "cargo")
            .count() as i64
    {
        failures.push("module_requirements rust_package_count metadata mismatch".to_string());
    }
    let mr_module_ids = ids_from_rows(&mr_modules);
    if mr_module_ids != req_ids {
        failures.push(
            "module_requirements module id set differs from requirements module set".to_string(),
        );
    }
    let command_ids = ids_from_rows(&mr_commands);
    let package_ids = ids_from_rows(&mr_packages);
    for row in &mr_modules {
        let mid = string_field(row, "id");
        let status = string_field(row, "status");
        if !req_status_allow.contains(&status) {
            failures.push(format!(
                "module_requirements.module[{mid}] invalid status: {status}"
            ));
        }
        if string_field(row, "status_token") != verify_status_token(&status) {
            failures.push(format!(
                "module_requirements.module[{mid}] status_token mismatch"
            ));
        }
        for dep_id in string_list_field(row, "requires_modules") {
            if !mr_module_ids.contains(&dep_id) {
                failures.push(format!(
                    "module_requirements.module[{mid}] unknown requires_modules ref: {dep_id}"
                ));
            }
        }
        for cmd_id in string_list_field(row, "command_refs") {
            if !command_ids.contains(&cmd_id) {
                failures.push(format!(
                    "module_requirements.module[{mid}] unknown command_ref: {cmd_id}"
                ));
            }
        }
        for pkg_id in string_list_field(row, "package_refs") {
            if !package_ids.contains(&pkg_id) {
                failures.push(format!(
                    "module_requirements.module[{mid}] unknown package_ref: {pkg_id}"
                ));
            }
        }
    }
    for row in &mr_commands {
        let cid = string_field(row, "id");
        let mid = string_field(row, "module_id");
        let kind = string_field(row, "kind");
        let cmd = string_field(row, "command");
        if !mr_module_ids.contains(&mid) {
            failures.push(format!(
                "module_requirements.command[{cid}] unknown module_id: {mid}"
            ));
        }
        if !["install", "verify"].contains(&kind.as_str()) {
            failures.push(format!(
                "module_requirements.command[{cid}] invalid kind: {kind}"
            ));
        }
        if cmd.is_empty() {
            failures.push(format!("module_requirements.command[{cid}] empty command"));
        }
    }
    for row in &mr_packages {
        let pid = string_field(row, "id");
        let mid = string_field(row, "module_id");
        let manager = string_field(row, "manager");
        let name = string_field(row, "name");
        if !mr_module_ids.contains(&mid) {
            failures.push(format!(
                "module_requirements.package[{pid}] unknown module_id: {mid}"
            ));
        }
        if !["pip", "cargo"].contains(&manager.as_str()) {
            failures.push(format!(
                "module_requirements.package[{pid}] invalid manager: {manager}"
            ));
        }
        if name.is_empty() {
            failures.push(format!("module_requirements.package[{pid}] empty name"));
        }
    }

    if !failures.is_empty() {
        println!(
            "ERROR: execution-planning registry lane verification failed (canonical registry-verify-execution-planning script)."
        );
        for item in failures.iter().take(300) {
            println!("- {item}");
        }
        if failures.len() > 300 {
            println!("- ... and {} more failures", failures.len() - 300);
        }
        std::process::exit(1);
    }

    println!(
        "OK: execution-planning registry lane verified (canonical registry-verify-execution-planning script). experiments={} lineages={} edges={} workstreams={} todo={} actions={} req_modules={} mr_packages={}",
        experiments.len(),
        lineages.len(),
        edges.len(),
        workstreams.len(),
        todo_items.len(),
        actions.len(),
        req_modules.len(),
        mr_packages.len()
    );
    Ok(())
}

fn build_experiment_rows(
    experiments: &[Table],
    binaries: &BTreeMap<String, Table>,
    bench_targets: &BTreeSet<String>,
    dataset_path_index: &BTreeMap<String, String>,
    dataset_label_aliases: &BTreeMap<String, String>,
) -> Result<Vec<ExperimentRow>> {
    let mut rows = experiments.to_vec();
    rows.sort_by_key(|row| string_field(row, "id"));
    let overrides = experiment_binary_overrides();
    let mut out = Vec::new();
    let mut used_lineage_ids = BTreeSet::new();
    let mut next_lineage_seq = 1usize;
    for row in rows {
        let eid = collapse(&string_field(&row, "id"));
        let mut binary = normalize_binary_name(&string_field(&row, "binary"));
        if binary.is_empty() {
            binary = overrides.get(&eid).cloned().unwrap_or_default();
        }
        let method = collapse(&string_field(&row, "method"));
        let input_text = collapse(&string_field(&row, "input"));
        let outputs = string_list_field(&row, "output");
        let output_text = outputs.join(" ");
        let claims = dedup_sorted(
            string_list_field(&row, "claims")
                .into_iter()
                .map(|value| collapse(&value))
                .filter(|value| !value.is_empty())
                .collect(),
        );
        let seed = row.get("seed").and_then(Value::as_integer);
        let deterministic = bool_field(&row, "deterministic");
        let gpu = bool_field(&row, "gpu");
        let mut status = collapse(&string_field(&row, "status"));
        status.make_ascii_lowercase();
        if !["active", "deprecated", "planned", "blocked"].contains(&status.as_str()) {
            status = "active".to_string();
        }
        let run_cmd = collapse(&string_field(&row, "run"));
        let existing_lineage = collapse(&string_field(&row, "lineage_id"));
        let lineage_id = choose_lineage_id(
            &existing_lineage,
            &mut used_lineage_ids,
            &mut next_lineage_seq,
        );
        let input_path_refs = dedup_sorted(
            extract_paths(&input_text)
                .into_iter()
                .chain(normalize_string_list(row.get("input_path_refs")))
                .collect(),
        );
        let output_path_refs = dedup_sorted(
            extract_paths(&output_text)
                .into_iter()
                .chain(normalize_string_list(row.get("output_path_refs")))
                .collect(),
        );
        let (dataset_refs, dataset_label_refs) = normalize_dataset_links(
            &normalize_string_list(row.get("dataset_refs")),
            &input_path_refs,
            &output_path_refs,
            dataset_path_index,
            dataset_label_aliases,
        );
        let reproducibility_class = if deterministic {
            "deterministic_replay".to_string()
        } else if seed.is_some() {
            "seeded_stochastic_replay".to_string()
        } else {
            "non_deterministic".to_string()
        };
        let binary_row = binaries
            .get(&binary)
            .or_else(|| binaries.get(execution_target_head(&binary)));
        let binary_registered = binary_row.is_some()
            || bench_targets.contains(&binary)
            || bench_targets.contains(execution_target_head(&binary));
        let binary_experiment_declared = {
            let registered = binary_row
                .map(|row| string_field(row, "experiment"))
                .unwrap_or_default();
            let declared = collapse(&string_field(&row, "binary_experiment_declared"));
            if !registered.is_empty() {
                registered
            } else if !declared.is_empty() {
                declared
            } else if !binary.is_empty() {
                eid.clone()
            } else {
                String::new()
            }
        };
        let external_source_refs = normalize_string_list(row.get("external_source_refs"));
        let truth_surface_consumption = normalize_string_list(row.get("truth_surface_consumption"));
        out.push(ExperimentRow {
            id: eid,
            title: collapse(&string_field(&row, "title")),
            binary,
            binary_registered,
            binary_experiment_declared,
            method,
            input: input_text,
            output: outputs.into_iter().map(|value| collapse(&value)).collect(),
            run: run_cmd.clone(),
            run_command_sha256: sha256_hex(run_cmd.as_bytes()),
            claims: claims.clone(),
            claim_refs: claims,
            deterministic,
            seed,
            gpu,
            status: status.clone(),
            status_token: build_status_token(&status),
            lineage_id,
            input_path_refs,
            output_path_refs,
            dataset_refs,
            dataset_label_refs,
            external_source_refs,
            truth_surface_consumption,
            reproducibility_class,
        });
    }
    Ok(out)
}

fn build_experiment_lineage(
    experiment_rows: &[ExperimentRow],
    claim_ids: &BTreeSet<String>,
    dataset_ids: &BTreeSet<String>,
) -> (Vec<LineageRow>, Vec<LineageEdge>) {
    let mut lineages = Vec::new();
    let mut edges = Vec::new();
    let mut edge_seq = 0usize;
    for row in experiment_rows {
        lineages.push(LineageRow {
            id: row.lineage_id.clone(),
            experiment_id: row.id.clone(),
            binary: row.binary.clone(),
            deterministic: row.deterministic,
            seed: row.seed,
            gpu: row.gpu,
            run_command: row.run.clone(),
            run_command_sha256: row.run_command_sha256.clone(),
            claim_refs: row.claim_refs.clone(),
            input_path_refs: row.input_path_refs.clone(),
            output_path_refs: row.output_path_refs.clone(),
            dataset_refs: row.dataset_refs.clone(),
            dataset_label_refs: row.dataset_label_refs.clone(),
            external_source_refs: row.external_source_refs.clone(),
            truth_surface_consumption: row.truth_surface_consumption.clone(),
            replay_steps: vec![
                "Confirm required input paths are available.".to_string(),
                row.run.clone(),
                "Verify expected outputs or stdout artifacts are produced.".to_string(),
            ],
            acceptance_criteria: vec![
                "Claim references resolve in registry/claims.toml.".to_string(),
                if row.binary.is_empty() {
                    "Execution command is explicitly declared.".to_string()
                } else {
                    "Execution target is registered in registry/binaries.toml or the workspace bench catalog.".to_string()
                },
                "Reproducibility class is explicitly declared.".to_string(),
            ],
        });
        if !row.binary.is_empty() {
            edge_seq += 1;
            edges.push(LineageEdge {
                id: format!("XLE-{edge_seq:05}"),
                lineage_id: row.lineage_id.clone(),
                from_id: row.id.clone(),
                to_ref: row.binary.clone(),
                to_kind: "binary".to_string(),
                edge_kind: "implemented_by_binary".to_string(),
                verified: row.binary_registered,
            });
        }
        for cid in &row.claim_refs {
            edge_seq += 1;
            edges.push(LineageEdge {
                id: format!("XLE-{edge_seq:05}"),
                lineage_id: row.lineage_id.clone(),
                from_id: row.id.clone(),
                to_ref: cid.clone(),
                to_kind: "claim".to_string(),
                edge_kind: "supports_claim".to_string(),
                verified: claim_ids.contains(cid),
            });
        }
        for path in &row.input_path_refs {
            edge_seq += 1;
            edges.push(LineageEdge {
                id: format!("XLE-{edge_seq:05}"),
                lineage_id: row.lineage_id.clone(),
                from_id: row.id.clone(),
                to_ref: path.clone(),
                to_kind: "path".to_string(),
                edge_kind: "consumes_path".to_string(),
                verified: true,
            });
        }
        for path in &row.output_path_refs {
            edge_seq += 1;
            edges.push(LineageEdge {
                id: format!("XLE-{edge_seq:05}"),
                lineage_id: row.lineage_id.clone(),
                from_id: row.id.clone(),
                to_ref: path.clone(),
                to_kind: "path".to_string(),
                edge_kind: "produces_path".to_string(),
                verified: true,
            });
        }
        for did in &row.dataset_refs {
            edge_seq += 1;
            edges.push(LineageEdge {
                id: format!("XLE-{edge_seq:05}"),
                lineage_id: row.lineage_id.clone(),
                from_id: row.id.clone(),
                to_ref: did.clone(),
                to_kind: "dataset".to_string(),
                edge_kind: "touches_dataset".to_string(),
                verified: dataset_ids.contains(did),
            });
        }
        for xid in &row.external_source_refs {
            edge_seq += 1;
            edges.push(LineageEdge {
                id: format!("XLE-{edge_seq:05}"),
                lineage_id: row.lineage_id.clone(),
                from_id: row.id.clone(),
                to_ref: xid.clone(),
                to_kind: "source".to_string(),
                edge_kind: "consumes_source".to_string(),
                verified: true,
            });
        }
        for surface in &row.truth_surface_consumption {
            edge_seq += 1;
            edges.push(LineageEdge {
                id: format!("XLE-{edge_seq:05}"),
                lineage_id: row.lineage_id.clone(),
                from_id: row.id.clone(),
                to_ref: surface.clone(),
                to_kind: "truth_surface".to_string(),
                edge_kind: "consumes_truth_surface".to_string(),
                verified: true,
            });
        }
    }
    (lineages, edges)
}

fn build_hardened_planning_rows(
    rows: &[Table],
    row_kind: &str,
    id_key: &str,
    status_allowlist: &[&str],
    priority_allowlist: &[&str],
) -> Vec<PlanningRow> {
    let known_ids = rows
        .iter()
        .map(|row| string_field(row, id_key))
        .collect::<BTreeSet<_>>();
    let mut out = Vec::new();
    for row in rows {
        let rid = collapse(&string_field(row, id_key));
        let mut status = collapse(&string_field(row, "status"));
        status.make_ascii_lowercase();
        if !status_allowlist.contains(&status.as_str()) {
            status = status_allowlist[0].to_string();
        }
        let mut priority = collapse(&string_field(row, "priority"));
        priority.make_ascii_lowercase();
        if !priority_allowlist.contains(&priority.as_str()) {
            priority = "medium".to_string();
        }
        let text_blob = [
            collapse(&string_field(row, "description")),
            collapse(&string_field(row, "title")),
            string_list_field(row, "claims").join(" "),
            collapse(&string_field(row, "insight")),
        ]
        .join(" ");
        let refs = extract_id_refs(&text_blob)
            .into_iter()
            .filter(|value| value != &rid)
            .collect::<Vec<_>>();
        let deps = dedup_sorted(
            refs.into_iter()
                .filter(|value| {
                    known_ids.contains(value)
                        || value.starts_with("C-")
                        || value.starts_with("I-")
                        || value.starts_with("E-")
                        || value.starts_with("REQ-")
                })
                .collect(),
        );
        let mut evidence_refs = extract_paths(&text_blob);
        if matches!(row.get("primary_outputs"), Some(Value::Array(_))) {
            evidence_refs.extend(
                string_list_field(row, "primary_outputs")
                    .into_iter()
                    .filter(|value| !value.is_empty()),
            );
        }
        evidence_refs = dedup_sorted(evidence_refs);
        let mut acceptance = vec![
            format!("{row_kind} status is constrained to declared enum values."),
            format!("{row_kind} dependencies are explicit and machine-parseable."),
        ];
        if !string_list_field(row, "claims").is_empty() {
            acceptance
                .push("Claim references remain resolvable in registry/claims.toml.".to_string());
        }
        if !evidence_refs.is_empty() {
            acceptance.push("Evidence references point to maintained canonical paths.".to_string());
        }
        let mut hardened = row.clone();
        hardened.insert("status".to_string(), Value::String(status.clone()));
        hardened.insert(
            "status_token".to_string(),
            Value::String(build_status_token(&status)),
        );
        if hardened.contains_key("priority") {
            hardened.insert("priority".to_string(), Value::String(priority));
        }
        hardened.insert(
            "dependencies".to_string(),
            Value::Array(deps.into_iter().map(Value::String).collect()),
        );
        hardened.insert(
            "acceptance_criteria".to_string(),
            Value::Array(acceptance.into_iter().map(Value::String).collect()),
        );
        hardened.insert(
            "evidence_refs".to_string(),
            Value::Array(evidence_refs.into_iter().map(Value::String).collect()),
        );
        out.push(PlanningRow { raw: hardened });
    }
    out.sort_by_key(|row| string_field(&row.raw, id_key));
    out
}

fn build_requirements_hardened(
    rows: &[Table],
    gaps: &[Table],
) -> (Vec<RequirementModule>, Vec<CoverageGap>) {
    let known_module_ids = rows
        .iter()
        .map(|row| string_field(row, "id"))
        .collect::<BTreeSet<_>>();
    let mut modules = Vec::new();
    let mut sorted_rows = rows.to_vec();
    sorted_rows.sort_by_key(|row| string_field(row, "id"));
    for row in sorted_rows {
        let mid = collapse(&string_field(&row, "id"));
        let mut status = collapse(&string_field(&row, "status"));
        status.make_ascii_lowercase();
        if !REQUIREMENT_STATUS_ALLOWLIST.contains(&status.as_str()) {
            status = "active".to_string();
        }
        let install_targets = normalize_string_list(row.get("install_targets"));
        let verify_targets = dedup_sorted(install_targets.clone());
        let runtime_stack = infer_runtime_stack(&collapse(&string_field(&row, "name")));
        let requires_modules = module_dependency_defaults(&mid)
            .into_iter()
            .filter(|dep| known_module_ids.contains(dep))
            .collect::<Vec<_>>();
        let acceptance = vec![
            "Runtime stack classification is explicit.".to_string(),
            "Install and verify commands are reproducible.".to_string(),
            "Module dependencies are fully declared in TOML.".to_string(),
        ];
        let mut module_row = row.clone();
        module_row.insert("status".to_string(), Value::String(status.clone()));
        module_row.insert(
            "status_token".to_string(),
            Value::String(build_status_token(&status)),
        );
        module_row.insert("runtime_stack".to_string(), Value::String(runtime_stack));
        module_row.insert(
            "requires_modules".to_string(),
            Value::Array(requires_modules.into_iter().map(Value::String).collect()),
        );
        module_row.insert(
            "install_targets".to_string(),
            Value::Array(install_targets.into_iter().map(Value::String).collect()),
        );
        module_row.insert(
            "verify_targets".to_string(),
            Value::Array(verify_targets.into_iter().map(Value::String).collect()),
        );
        module_row.insert(
            "acceptance_criteria".to_string(),
            Value::Array(acceptance.into_iter().map(Value::String).collect()),
        );
        modules.push(RequirementModule { raw: module_row });
    }

    let mut coverage_gaps = Vec::new();
    let mut sorted_gaps = gaps.to_vec();
    sorted_gaps.sort_by_key(|row| string_field(row, "id"));
    for row in sorted_gaps {
        let mut status = collapse(&string_field(&row, "status"));
        status.make_ascii_lowercase();
        if !["open", "in_progress", "done"].contains(&status.as_str()) {
            status = "open".to_string();
        }
        let blob = format!(
            "{} {}",
            collapse(&string_field(&row, "description")),
            collapse(&string_field(&row, "proposed_resolution"))
        );
        let related_module_ids = dedup_sorted(
            extract_id_refs(&blob)
                .into_iter()
                .filter(|value| value.starts_with("REQ-"))
                .collect(),
        );
        let mut gap = row.clone();
        gap.insert("status".to_string(), Value::String(status.clone()));
        gap.insert(
            "status_token".to_string(),
            Value::String(build_status_token(&status)),
        );
        gap.insert(
            "related_module_ids".to_string(),
            Value::Array(related_module_ids.into_iter().map(Value::String).collect()),
        );
        coverage_gaps.push(CoverageGap { raw: gap });
    }
    (modules, coverage_gaps)
}

fn build_module_requirements(
    requirements_modules: &[RequirementModule],
    pyproject: &Table,
    cargo_toml: &Table,
) -> Result<(
    Vec<ModuleRequirement>,
    Vec<ModulePackage>,
    Vec<ModuleCommand>,
)> {
    let mut modules = Vec::new();
    let mut packages = Vec::new();
    let mut commands = Vec::new();
    let module_ids = requirements_modules
        .iter()
        .map(|row| string_field(&row.raw, "id"))
        .collect::<BTreeSet<_>>();
    for row in requirements_modules {
        let mid = collapse(&string_field(&row.raw, "id"));
        let install_targets = normalize_string_list(row.raw.get("install_targets"));
        let verify_targets = normalize_string_list(row.raw.get("verify_targets"));
        let mut command_refs = Vec::new();
        for cmd in &install_targets {
            let cid = format!("CMD-{:04}", commands.len() + 1);
            commands.push(ModuleCommand {
                id: cid.clone(),
                module_id: mid.clone(),
                kind: "install".to_string(),
                command: cmd.clone(),
            });
            command_refs.push(cid);
        }
        for cmd in &verify_targets {
            let cid = format!("CMD-{:04}", commands.len() + 1);
            commands.push(ModuleCommand {
                id: cid.clone(),
                module_id: mid.clone(),
                kind: "verify".to_string(),
                command: cmd.clone(),
            });
            command_refs.push(cid);
        }
        modules.push(ModuleRequirement {
            id: mid,
            name: collapse(&string_field(&row.raw, "name")),
            runtime_stack: collapse(&string_field(&row.raw, "runtime_stack")),
            status: collapse(&string_field(&row.raw, "status")),
            status_token: collapse(&string_field(&row.raw, "status_token")),
            source_markdown: collapse(&string_field(&row.raw, "markdown")),
            requires_modules: string_list_field(&row.raw, "requires_modules")
                .into_iter()
                .filter(|dep| module_ids.contains(dep))
                .collect(),
            command_refs,
            package_refs: Vec::new(),
        });
    }
    let mut module_index = BTreeMap::new();
    for (idx, module) in modules.iter().enumerate() {
        module_index.insert(module.id.clone(), idx);
    }

    let project = table_value(pyproject, "project");
    let base_deps = project
        .get("dependencies")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let optional = project
        .get("optional-dependencies")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let extra_module_map = BTreeMap::from([
        ("analysis".to_string(), "REQ-ANALYSIS".to_string()),
        ("astro".to_string(), "REQ-ASTRO".to_string()),
        ("particle".to_string(), "REQ-PARTICLE".to_string()),
        ("quantum".to_string(), "REQ-QUANTUM".to_string()),
        ("dev".to_string(), "REQ-CORE".to_string()),
    ]);

    let add_package = |packages: &mut Vec<ModulePackage>,
                       modules: &mut Vec<ModuleRequirement>,
                       module_index: &BTreeMap<String, usize>,
                       module_id: &str,
                       manager: &str,
                       spec: &str,
                       source: &str,
                       group: &str,
                       optional_flag: bool| {
        let target_module = if module_index.contains_key(module_id) {
            module_id.to_string()
        } else {
            "REQ-CORE".to_string()
        };
        let (name, constraint) = parse_dependency_spec(spec);
        let pid = format!("PKG-{:05}", packages.len() + 1);
        packages.push(ModulePackage {
            id: pid.clone(),
            module_id: target_module.clone(),
            manager: manager.to_string(),
            name,
            constraint,
            spec: collapse(spec),
            group: group.to_string(),
            optional: optional_flag,
            source: source.to_string(),
        });
        if let Some(index) = module_index.get(&target_module) {
            modules[*index].package_refs.push(pid);
        }
    };

    for spec in base_deps {
        add_package(
            &mut packages,
            &mut modules,
            &module_index,
            "REQ-CORE",
            "pip",
            &value_to_string(&spec),
            "pyproject.toml",
            "base",
            false,
        );
    }
    for (extra, deps) in optional {
        let module_id = extra_module_map
            .get(&extra)
            .cloned()
            .unwrap_or_else(|| "REQ-CORE".to_string());
        if let Some(dep_array) = deps.as_array() {
            for spec in dep_array {
                add_package(
                    &mut packages,
                    &mut modules,
                    &module_index,
                    &module_id,
                    "pip",
                    &value_to_string(spec),
                    "pyproject.toml",
                    &format!("extra:{extra}"),
                    true,
                );
            }
        }
    }

    let workspace = table_value(cargo_toml, "workspace");
    let ws_deps = workspace
        .get("dependencies")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let quantum_rust = BTreeSet::from([
        "quantum".to_string(),
        "qua_ten_net".to_string(),
        "cudarc".to_string(),
        "numpy".to_string(),
        "pyo3".to_string(),
    ]);
    for (dep_name, dep_spec) in ws_deps {
        let module_id = if quantum_rust.contains(&dep_name) {
            "REQ-QUANTUM"
        } else {
            "REQ-CORE"
        };
        let spec = match dep_spec {
            Value::String(text) => format!("{dep_name} {text}"),
            other => format!(
                "{dep_name} {}",
                serde_json::to_string(&toml_to_json(&other))
                    .unwrap_or_else(|_| "\"<unserializable>\"".to_string())
            ),
        };
        add_package(
            &mut packages,
            &mut modules,
            &module_index,
            module_id,
            "cargo",
            &spec,
            "Cargo.toml",
            "workspace.dependencies",
            false,
        );
    }

    for module in &mut modules {
        module.package_refs.sort();
        module.command_refs.sort();
    }
    modules.sort_by_key(|row| row.id.clone());
    packages.sort_by_key(|row| row.id.clone());
    commands.sort_by_key(|row| row.id.clone());
    Ok((modules, packages, commands))
}

fn render_experiments(rows: &[ExperimentRow]) -> Result<String> {
    let mut lines = vec![
        "# Experiments registry (execution-planning lane strict schema; legacy Wave 5 Batch 4 compatibility).".to_string(),
        "# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.".to_string(),
        String::new(),
        "[experiments]".to_string(),
        "updated = \"2026-02-10\"".to_string(),
        "authoritative = true".to_string(),
        format!("experiment_count = {}", rows.len()),
        format!("deterministic_count = {}", rows.iter().filter(|row| row.deterministic).count()),
        format!("gpu_count = {}", rows.iter().filter(|row| row.gpu).count()),
        format!("seeded_count = {}", rows.iter().filter(|row| row.seed.is_some()).count()),
        "status_allowlist = [\"active\", \"deprecated\", \"planned\", \"blocked\"]".to_string(),
        String::new(),
    ];
    for row in rows {
        lines.push("[[experiment]]".to_string());
        lines.push(format!("id = {}", q(&row.id)));
        lines.push(format!("title = {}", q(&row.title)));
        lines.push(format!("binary = {}", q(&row.binary)));
        lines.push(format!("binary_registered = {}", row.binary_registered));
        lines.push(format!(
            "binary_experiment_declared = {}",
            q(&row.binary_experiment_declared)
        ));
        lines.push(format!("method = {}", q(&row.method)));
        lines.push(format!("input = {}", q(&row.input)));
        lines.push(format!("output = {}", render_list(&row.output)));
        lines.push(format!("run = {}", q(&row.run)));
        lines.push(format!(
            "run_command_sha256 = {}",
            q(&row.run_command_sha256)
        ));
        lines.push(format!("claims = {}", render_list(&row.claims)));
        lines.push(format!("claim_refs = {}", render_list(&row.claim_refs)));
        lines.push(format!("deterministic = {}", row.deterministic));
        if let Some(seed) = row.seed {
            lines.push(format!("seed = {seed}"));
        }
        lines.push(format!("gpu = {}", row.gpu));
        lines.push(format!("status = {}", q(&row.status)));
        lines.push(format!("status_token = {}", q(&row.status_token)));
        lines.push(format!("lineage_id = {}", q(&row.lineage_id)));
        lines.push(format!(
            "input_path_refs = {}",
            render_list(&row.input_path_refs)
        ));
        lines.push(format!(
            "output_path_refs = {}",
            render_list(&row.output_path_refs)
        ));
        lines.push(format!("dataset_refs = {}", render_list(&row.dataset_refs)));
        lines.push(format!(
            "dataset_label_refs = {}",
            render_list(&row.dataset_label_refs)
        ));
        lines.push(format!(
            "external_source_refs = {}",
            render_list(&row.external_source_refs)
        ));
        lines.push(format!(
            "truth_surface_consumption = {}",
            render_list(&row.truth_surface_consumption)
        ));
        lines.push(format!(
            "reproducibility_class = {}",
            q(&row.reproducibility_class)
        ));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_experiment_lineage(lineages: &[LineageRow], edges: &[LineageEdge]) -> Result<String> {
    let mut lines = vec![
        "# Experiment lineage registry (execution-planning lane strict schema; legacy Wave 5 Batch 4 compatibility).".to_string(),
        "# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.".to_string(),
        String::new(),
        "[experiment_lineage]".to_string(),
        "updated = \"2026-02-10\"".to_string(),
        "authoritative = true".to_string(),
        "source_registry = \"registry/experiments.toml\"".to_string(),
        format!("lineage_count = {}", lineages.len()),
        format!("edge_count = {}", edges.len()),
        String::new(),
    ];
    for row in lineages {
        lines.push("[[lineage]]".to_string());
        lines.push(format!("id = {}", q(&row.id)));
        lines.push(format!("experiment_id = {}", q(&row.experiment_id)));
        lines.push(format!("binary = {}", q(&row.binary)));
        lines.push(format!("deterministic = {}", row.deterministic));
        if let Some(seed) = row.seed {
            lines.push(format!("seed = {seed}"));
        }
        lines.push(format!("gpu = {}", row.gpu));
        lines.push(format!("run_command = {}", q(&row.run_command)));
        lines.push(format!(
            "run_command_sha256 = {}",
            q(&row.run_command_sha256)
        ));
        lines.push(format!("claim_refs = {}", render_list(&row.claim_refs)));
        lines.push(format!(
            "input_path_refs = {}",
            render_list(&row.input_path_refs)
        ));
        lines.push(format!(
            "output_path_refs = {}",
            render_list(&row.output_path_refs)
        ));
        lines.push(format!("dataset_refs = {}", render_list(&row.dataset_refs)));
        lines.push(format!(
            "dataset_label_refs = {}",
            render_list(&row.dataset_label_refs)
        ));
        lines.push(format!(
            "external_source_refs = {}",
            render_list(&row.external_source_refs)
        ));
        lines.push(format!(
            "truth_surface_consumption = {}",
            render_list(&row.truth_surface_consumption)
        ));
        lines.push(format!("replay_steps = {}", render_list(&row.replay_steps)));
        lines.push(format!(
            "acceptance_criteria = {}",
            render_list(&row.acceptance_criteria)
        ));
        lines.push(String::new());
    }
    for row in edges {
        lines.push("[[edge]]".to_string());
        lines.push(format!("id = {}", q(&row.id)));
        lines.push(format!("lineage_id = {}", q(&row.lineage_id)));
        lines.push(format!("from_id = {}", q(&row.from_id)));
        lines.push(format!("to_ref = {}", q(&row.to_ref)));
        lines.push(format!("to_kind = {}", q(&row.to_kind)));
        lines.push(format!("edge_kind = {}", q(&row.edge_kind)));
        lines.push(format!("verified = {}", row.verified));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_roadmap(meta: &Table, rows: &[PlanningRow]) -> Result<String> {
    let source_markdown = q(&collapse(&string_field(meta, "source_markdown")));
    let supersedes = normalize_string_list(meta.get("supersedes"));
    let companion_docs = normalize_string_list(meta.get("companion_docs"));
    let mut lines = vec![
        "# GENERATED VIEW: DO NOT EDIT.".to_string(),
        "# Update via `gororoba-db planning ...` against `registry/canonical/control_plane.sqlite3`.".to_string(),
        "# Operational roadmap registry (SQLite compatibility export from canonical control_plane.sqlite3).".to_string(),
        "# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.".to_string(),
        String::new(),
        "[roadmap]".to_string(),
        format!(
            "source_markdown = {}",
            if source_markdown == "\"\"" {
                "\"docs/ROADMAP.md\"".to_string()
            } else {
                source_markdown
            }
        ),
        "consolidated_date = \"2026-02-10\"".to_string(),
        format!("supersedes = {}", render_list(&supersedes)),
        format!("companion_docs = {}", render_list(&companion_docs)),
        "status = \"active\"".to_string(),
        "status_token = \"ACTIVE\"".to_string(),
        "authoritative = true".to_string(),
        format!("workstream_count = {}", rows.len()),
        format!(
            "status_allowlist = {}",
            render_list(
                &ROADMAP_STATUS_ALLOWLIST
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect::<Vec<_>>()
            )
        ),
        format!(
            "priority_allowlist = {}",
            render_list(
                &PLANNING_PRIORITY_ALLOWLIST
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect::<Vec<_>>()
            )
        ),
        String::new(),
        "[roadmap.schema]".to_string(),
        "required_fields = [\"id\", \"name\", \"priority\", \"status\", \"status_token\", \"description\", \"dependencies\", \"acceptance_criteria\"]".to_string(),
        "dependency_id_pattern = \"WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*\"".to_string(),
        String::new(),
    ];
    if let Some(sections) = meta.get("sections").and_then(Value::as_table)
        && !sections.is_empty()
    {
        lines.push("[roadmap.sections]".to_string());
        for (key, value) in sections {
            lines.push(format!("{key} = {}", q(&collapse(&value_to_string(value)))));
        }
        lines.push(String::new());
    }
    for row in rows {
        lines.push("[[workstream]]".to_string());
        for key in [
            "id",
            "name",
            "priority",
            "status",
            "status_token",
            "description",
        ] {
            lines.push(format!(
                "{key} = {}",
                q(&collapse(&string_field(&row.raw, key)))
            ));
        }
        let sprint = collapse(&string_field(&row.raw, "sprint"));
        if !sprint.is_empty() {
            lines.push(format!("sprint = {}", q(&sprint)));
        }
        let primary_outputs = normalize_string_list(row.raw.get("primary_outputs"));
        if !primary_outputs.is_empty() {
            lines.push(format!(
                "primary_outputs = {}",
                render_list(&primary_outputs)
            ));
        }
        let claims = normalize_string_list(row.raw.get("claims"));
        if !claims.is_empty() {
            lines.push(format!("claims = {}", render_list(&claims)));
        }
        let insight = collapse(&string_field(&row.raw, "insight"));
        if !insight.is_empty() {
            lines.push(format!("insight = {}", q(&insight)));
        }
        let lacunae = normalize_string_list(row.raw.get("lacunae"));
        if !lacunae.is_empty() {
            lines.push(format!("lacunae = {}", render_list(&lacunae)));
        }
        lines.push(format!(
            "dependencies = {}",
            render_list(&string_list_field(&row.raw, "dependencies"))
        ));
        lines.push(format!(
            "acceptance_criteria = {}",
            render_list(&string_list_field(&row.raw, "acceptance_criteria"))
        ));
        lines.push(format!(
            "evidence_refs = {}",
            render_list(&string_list_field(&row.raw, "evidence_refs"))
        ));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_todo(rows: &[PlanningRow]) -> Result<String> {
    let mut lines = vec![
        "# GENERATED VIEW: DO NOT EDIT.".to_string(),
        "# Update via `gororoba-db planning ...` against `registry/canonical/control_plane.sqlite3`.".to_string(),
        "# To-Do Registry (SQLite compatibility export from canonical control_plane.sqlite3).".to_string(),
        "# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.".to_string(),
        String::new(),
        "[todo]".to_string(),
        "updated = \"2026-02-10\"".to_string(),
        "status = \"active\"".to_string(),
        "status_token = \"ACTIVE\"".to_string(),
        format!("item_count = {}", rows.len()),
        format!(
            "status_allowlist = {}",
            render_list(
                &TODO_STATUS_ALLOWLIST
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect::<Vec<_>>()
            )
        ),
        format!(
            "priority_allowlist = {}",
            render_list(
                &PLANNING_PRIORITY_ALLOWLIST
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect::<Vec<_>>()
            )
        ),
        String::new(),
        "[todo.schema]".to_string(),
        "required_fields = [\"id\", \"area\", \"title\", \"description\", \"priority\", \"status\", \"status_token\", \"dependencies\", \"acceptance_criteria\"]".to_string(),
        "dependency_id_pattern = \"WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*\"".to_string(),
        String::new(),
    ];
    for row in rows {
        lines.push("[[item]]".to_string());
        for key in [
            "id",
            "area",
            "title",
            "description",
            "priority",
            "status",
            "status_token",
        ] {
            lines.push(format!(
                "{key} = {}",
                q(&collapse(&string_field(&row.raw, key)))
            ));
        }
        lines.push(format!(
            "dependencies = {}",
            render_list(&string_list_field(&row.raw, "dependencies"))
        ));
        lines.push(format!(
            "acceptance_criteria = {}",
            render_list(&string_list_field(&row.raw, "acceptance_criteria"))
        ));
        lines.push(format!(
            "evidence_refs = {}",
            render_list(&string_list_field(&row.raw, "evidence_refs"))
        ));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_next_actions(rows: &[PlanningRow]) -> Result<String> {
    let mut lines = vec![
        "# GENERATED VIEW: DO NOT EDIT.".to_string(),
        "# Update via `gororoba-db planning ...` against `registry/canonical/control_plane.sqlite3`.".to_string(),
        "# Next Actions Registry (SQLite compatibility export from canonical control_plane.sqlite3).".to_string(),
        "# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.".to_string(),
        String::new(),
        "[meta]".to_string(),
        "updated = \"2026-02-10\"".to_string(),
        "status = \"active\"".to_string(),
        "status_token = \"ACTIVE\"".to_string(),
        format!("action_count = {}", rows.len()),
        format!(
            "status_allowlist = {}",
            render_list(
                &ACTION_STATUS_ALLOWLIST
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect::<Vec<_>>()
            )
        ),
        format!(
            "priority_allowlist = {}",
            render_list(
                &PLANNING_PRIORITY_ALLOWLIST
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect::<Vec<_>>()
            )
        ),
        String::new(),
        "[next_actions.schema]".to_string(),
        "required_fields = [\"id\", \"area\", \"title\", \"description\", \"priority\", \"status\", \"status_token\", \"dependencies\", \"acceptance_criteria\"]".to_string(),
        "dependency_id_pattern = \"WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*\"".to_string(),
        String::new(),
    ];
    for row in rows {
        lines.push("[[action]]".to_string());
        for key in [
            "id",
            "area",
            "title",
            "description",
            "priority",
            "status",
            "status_token",
        ] {
            lines.push(format!(
                "{key} = {}",
                q(&collapse(&string_field(&row.raw, key)))
            ));
        }
        lines.push(format!(
            "dependencies = {}",
            render_list(&string_list_field(&row.raw, "dependencies"))
        ));
        lines.push(format!(
            "acceptance_criteria = {}",
            render_list(&string_list_field(&row.raw, "acceptance_criteria"))
        ));
        lines.push(format!(
            "evidence_refs = {}",
            render_list(&string_list_field(&row.raw, "evidence_refs"))
        ));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_requirements(
    modules: &[RequirementModule],
    gaps: &[CoverageGap],
    meta: &Table,
) -> Result<String> {
    let python_recommended = collapse(&string_field(meta, "python_recommended"));
    let python_allowed = collapse(&string_field(meta, "python_allowed"));
    let primary_markdown = collapse(&string_field(meta, "primary_markdown"));
    let mut lines = vec![
        "# GENERATED VIEW: DO NOT EDIT.".to_string(),
        "# Update via `gororoba-db requirements ...` against `registry/canonical/control_plane.sqlite3`.".to_string(),
        "# Requirements registry (SQLite compatibility export from canonical control_plane.sqlite3).".to_string(),
        "# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.".to_string(),
        String::new(),
        "[requirements]".to_string(),
        "authoritative = true".to_string(),
        "status = \"active\"".to_string(),
        "status_token = \"ACTIVE\"".to_string(),
        "updated = \"2026-02-10\"".to_string(),
        format!(
            "python_recommended = {}",
            q(if python_recommended.is_empty() {
                "3.11-3.12"
            } else {
                &python_recommended
            })
        ),
        format!(
            "python_allowed = {}",
            q(if python_allowed.is_empty() {
                "3.13+ (with optional extras caveats)"
            } else {
                &python_allowed
            })
        ),
        format!(
            "primary_markdown = {}",
            q(if primary_markdown.is_empty() {
                "docs/REQUIREMENTS.md"
            } else {
                &primary_markdown
            })
        ),
        format!("module_count = {}", modules.len()),
        format!("coverage_gap_count = {}", gaps.len()),
        format!(
            "status_allowlist = {}",
            render_list(
                &REQUIREMENT_STATUS_ALLOWLIST
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect::<Vec<_>>()
            )
        ),
        format!(
            "runtime_stack_allowlist = {}",
            render_list(
                &RUNTIME_STACK_ALLOWLIST
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect::<Vec<_>>()
            )
        ),
        String::new(),
        "[requirements.schema]".to_string(),
        "required_module_fields = [\"id\", \"name\", \"status\", \"status_token\", \"runtime_stack\", \"requires_modules\", \"install_targets\", \"verify_targets\", \"acceptance_criteria\"]".to_string(),
        "required_gap_fields = [\"id\", \"area\", \"status\", \"status_token\", \"description\", \"proposed_resolution\", \"related_module_ids\"]".to_string(),
        String::new(),
    ];
    for row in modules {
        lines.push("[[module]]".to_string());
        lines.push(format!(
            "id = {}",
            q(&collapse(&string_field(&row.raw, "id")))
        ));
        lines.push(format!(
            "name = {}",
            q(&collapse(&string_field(&row.raw, "name")))
        ));
        lines.push(format!(
            "markdown = {}",
            q(&collapse(&string_field(&row.raw, "markdown")))
        ));
        lines.push(format!(
            "status = {}",
            q(&collapse(&string_field(&row.raw, "status")))
        ));
        lines.push(format!(
            "status_token = {}",
            q(&collapse(&string_field(&row.raw, "status_token")))
        ));
        lines.push(format!(
            "runtime_stack = {}",
            q(&collapse(&string_field(&row.raw, "runtime_stack")))
        ));
        lines.push(format!(
            "requires_modules = {}",
            render_list(&string_list_field(&row.raw, "requires_modules"))
        ));
        lines.push(format!(
            "install_targets = {}",
            render_list(&string_list_field(&row.raw, "install_targets"))
        ));
        lines.push(format!(
            "verify_targets = {}",
            render_list(&string_list_field(&row.raw, "verify_targets"))
        ));
        lines.push(format!(
            "acceptance_criteria = {}",
            render_list(&string_list_field(&row.raw, "acceptance_criteria"))
        ));
        lines.push(String::new());
    }
    for row in gaps {
        lines.push("[[coverage_gap]]".to_string());
        lines.push(format!(
            "id = {}",
            q(&collapse(&string_field(&row.raw, "id")))
        ));
        lines.push(format!(
            "area = {}",
            q(&collapse(&string_field(&row.raw, "area")))
        ));
        lines.push(format!(
            "status = {}",
            q(&collapse(&string_field(&row.raw, "status")))
        ));
        lines.push(format!(
            "status_token = {}",
            q(&collapse(&string_field(&row.raw, "status_token")))
        ));
        lines.push(format!(
            "description = {}",
            q(&collapse(&string_field(&row.raw, "description")))
        ));
        lines.push(format!(
            "proposed_resolution = {}",
            q(&collapse(&string_field(&row.raw, "proposed_resolution")))
        ));
        lines.push(format!(
            "related_module_ids = {}",
            render_list(&string_list_field(&row.raw, "related_module_ids"))
        ));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn render_module_requirements(
    modules: &[ModuleRequirement],
    packages: &[ModulePackage],
    commands: &[ModuleCommand],
) -> Result<String> {
    let mut lines = vec![
        "# Module requirements decomposition registry (execution-planning lane strict schema; legacy Wave 5 Batch 4 compatibility).".to_string(),
        "# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.".to_string(),
        String::new(),
        "[module_requirements]".to_string(),
        "updated = \"2026-02-10\"".to_string(),
        "authoritative = true".to_string(),
        "source_registries = [\"registry/requirements.toml\", \"pyproject.toml\", \"Cargo.toml\"]".to_string(),
        format!("module_count = {}", modules.len()),
        format!("package_count = {}", packages.len()),
        format!("command_count = {}", commands.len()),
        format!(
            "python_package_count = {}",
            packages.iter().filter(|pkg| pkg.manager == "pip").count()
        ),
        format!(
            "rust_package_count = {}",
            packages.iter().filter(|pkg| pkg.manager == "cargo").count()
        ),
        String::new(),
    ];
    for row in modules {
        lines.push("[[module]]".to_string());
        lines.push(format!("id = {}", q(&row.id)));
        lines.push(format!("name = {}", q(&row.name)));
        lines.push(format!("runtime_stack = {}", q(&row.runtime_stack)));
        lines.push(format!("status = {}", q(&row.status)));
        lines.push(format!("status_token = {}", q(&row.status_token)));
        lines.push(format!("source_markdown = {}", q(&row.source_markdown)));
        lines.push(format!(
            "requires_modules = {}",
            render_list(&row.requires_modules)
        ));
        lines.push(format!("command_refs = {}", render_list(&row.command_refs)));
        lines.push(format!("package_refs = {}", render_list(&row.package_refs)));
        lines.push(String::new());
    }
    for row in commands {
        lines.push("[[command]]".to_string());
        lines.push(format!("id = {}", q(&row.id)));
        lines.push(format!("module_id = {}", q(&row.module_id)));
        lines.push(format!("kind = {}", q(&row.kind)));
        lines.push(format!("command = {}", q(&row.command)));
        lines.push(String::new());
    }
    for row in packages {
        lines.push("[[package]]".to_string());
        lines.push(format!("id = {}", q(&row.id)));
        lines.push(format!("module_id = {}", q(&row.module_id)));
        lines.push(format!("manager = {}", q(&row.manager)));
        lines.push(format!("name = {}", q(&row.name)));
        lines.push(format!("constraint = {}", q(&row.constraint)));
        lines.push(format!("spec = {}", q(&row.spec)));
        lines.push(format!("group = {}", q(&row.group)));
        lines.push(format!("optional = {}", row.optional));
        lines.push(format!("source = {}", q(&row.source)));
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

fn load_dataset_path_index(repo_root: &Path) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for rel in [
        "registry/project_csv_canonical_datasets.toml",
        "registry/project_csv_generated_artifacts.toml",
        "registry/project_csv_generated_datasets.toml",
        "registry/external_csv_datasets.toml",
        "registry/archive_csv_datasets.toml",
        "registry/curated_csv_datasets.toml",
    ] {
        let path = repo_root.join(rel);
        if !path.exists() {
            continue;
        }
        let raw = load_toml(&path)?;
        for value in raw.values() {
            if let Some(rows) = value.as_array() {
                for row in rows {
                    let Some(table) = row.as_table() else {
                        continue;
                    };
                    let dataset_id = collapse(&string_field(table, "id"));
                    if !dataset_id_regex().is_match(&dataset_id) {
                        continue;
                    }
                    for key in ["source_csv", "canonical_toml"] {
                        let item = collapse(&string_field(table, key));
                        if !item.is_empty() {
                            out.insert(item, dataset_id.clone());
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn load_dataset_label_aliases(repo_root: &Path) -> Result<BTreeMap<String, String>> {
    let path = repo_root.join("registry/dataset_label_aliases.toml");
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = load_toml(&path)?;
    let mut aliases = BTreeMap::new();
    for row in table_array(&raw, "alias")? {
        let mut label = normalize_dataset_label(&string_field(&row, "label_normalized"));
        if label.is_empty() {
            label = normalize_dataset_label(&string_field(&row, "label"));
        }
        let canonical_dataset_id = collapse(&string_field(&row, "canonical_dataset_id"));
        if !label.is_empty() && !canonical_dataset_id.is_empty() {
            aliases.insert(label, canonical_dataset_id);
        }
    }
    Ok(aliases)
}

fn normalize_dataset_links(
    explicit_dataset_refs: &[String],
    input_path_refs: &[String],
    output_path_refs: &[String],
    dataset_path_index: &BTreeMap<String, String>,
    dataset_label_aliases: &BTreeMap<String, String>,
) -> (Vec<String>, Vec<String>) {
    let mut dataset_ids = Vec::new();
    let mut dataset_labels = Vec::new();
    let mut seen_ids = BTreeSet::new();
    let mut seen_labels = BTreeSet::new();

    let mut add_dataset_id = |value: String| {
        if !value.is_empty() && seen_ids.insert(value.clone()) {
            dataset_ids.push(value);
        }
    };
    let mut add_dataset_label = |value: String| {
        if !value.is_empty() && seen_labels.insert(value.clone()) {
            dataset_labels.push(value);
        }
    };

    for path in input_path_refs.iter().chain(output_path_refs.iter()) {
        if let Some(dataset_id) = dataset_path_index.get(path) {
            add_dataset_id(dataset_id.clone());
        }
    }
    let joined = input_path_refs
        .iter()
        .chain(output_path_refs.iter())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    for dataset_id in dataset_id_regex().find_iter(&joined) {
        add_dataset_id(dataset_id.as_str().to_string());
    }
    for reference in explicit_dataset_refs {
        if dataset_id_regex().is_match(reference) {
            add_dataset_id(reference.clone());
        } else if let Some(dataset_id) = dataset_path_index.get(reference) {
            add_dataset_id(dataset_id.clone());
        } else if let Some(dataset_id) =
            dataset_label_aliases.get(&normalize_dataset_label(reference))
        {
            add_dataset_id(dataset_id.clone());
            add_dataset_label(reference.clone());
        } else {
            add_dataset_label(reference.clone());
        }
    }
    (dataset_ids, dataset_labels)
}

fn choose_lineage_id(existing: &str, used: &mut BTreeSet<String>, sequence: &mut usize) -> String {
    if lineage_id_regex().is_match(existing) && !used.contains(existing) {
        used.insert(existing.to_string());
        return existing.to_string();
    }
    loop {
        let candidate = format!("XL-{sequence:03}");
        *sequence += 1;
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
}

/// Bundle of cross-reference ID sets used by `verify_dependencies` to
/// validate dependency strings. Seven `BTreeSet<String>` references that
/// always travel together. Bundling keeps the verifier under
/// clippy::too_many_arguments without `#[allow]` suppression.
struct DependencyIdSets<'a> {
    claim_ids: &'a BTreeSet<String>,
    insight_ids: &'a BTreeSet<String>,
    experiment_ids: &'a BTreeSet<String>,
    workstream_ids: &'a BTreeSet<String>,
    todo_ids: &'a BTreeSet<String>,
    action_ids: &'a BTreeSet<String>,
    req_ids: &'a BTreeSet<String>,
}

fn verify_dependencies(
    failures: &mut Vec<String>,
    deps: &[String],
    where_label: &str,
    ids: &DependencyIdSets<'_>,
) {
    let claim_ids = ids.claim_ids;
    let insight_ids = ids.insight_ids;
    let experiment_ids = ids.experiment_ids;
    let workstream_ids = ids.workstream_ids;
    let todo_ids = ids.todo_ids;
    let action_ids = ids.action_ids;
    let req_ids = ids.req_ids;
    for dep in deps {
        let dep_id = dep.trim();
        if dep_id.is_empty() {
            continue;
        }
        if claim_regex().is_match(dep_id) {
            if !claim_ids.contains(dep_id) {
                failures.push(format!("{where_label} unknown claim dependency: {dep_id}"));
            }
            continue;
        }
        if insight_regex().is_match(dep_id) {
            if !insight_ids.contains(dep_id) {
                failures.push(format!(
                    "{where_label} unknown insight dependency: {dep_id}"
                ));
            }
            continue;
        }
        if experiment_regex().is_match(dep_id) {
            if !experiment_ids.contains(dep_id) {
                failures.push(format!(
                    "{where_label} unknown experiment dependency: {dep_id}"
                ));
            }
            continue;
        }
        if workstream_regex().is_match(dep_id) {
            if !workstream_ids.contains(dep_id) {
                failures.push(format!(
                    "{where_label} unknown workstream dependency: {dep_id}"
                ));
            }
            continue;
        }
        if todo_regex().is_match(dep_id) {
            if !todo_ids.contains(dep_id) {
                failures.push(format!("{where_label} unknown todo dependency: {dep_id}"));
            }
            continue;
        }
        if action_regex().is_match(dep_id) {
            if !action_ids.contains(dep_id) {
                failures.push(format!("{where_label} unknown action dependency: {dep_id}"));
            }
            continue;
        }
        if req_regex().is_match(dep_id) {
            if !req_ids.contains(dep_id) {
                failures.push(format!(
                    "{where_label} unknown requirements dependency: {dep_id}"
                ));
            }
            continue;
        }
        failures.push(format!("{where_label} malformed dependency id: {dep_id}"));
    }
}

fn collect_dataset_ids(repo_root: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    for rel in [
        "registry/project_csv_canonical_datasets.toml",
        "registry/project_csv_generated_artifacts.toml",
        "registry/project_csv_generated_datasets.toml",
        "registry/external_csv_datasets.toml",
        "registry/archive_csv_datasets.toml",
        "registry/curated_csv_datasets.toml",
    ] {
        let path = repo_root.join(rel);
        if !path.exists() {
            continue;
        }
        let raw = load_toml(&path)?;
        for value in raw.values() {
            if let Some(rows) = value.as_array() {
                for row in rows {
                    let Some(table) = row.as_table() else {
                        continue;
                    };
                    let rid = string_field(table, "id");
                    if dataset_id_regex().is_match(&rid) {
                        out.insert(rid);
                    }
                }
            }
        }
    }
    Ok(out)
}

fn collect_source_ids(repo_root: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    let external_sources_path = repo_root.join("registry/external_sources.toml");
    if external_sources_path.exists() {
        let raw = load_toml(&external_sources_path)?;
        for row in table_array(&raw, "document")? {
            let id = string_field(&row, "id");
            if !id.is_empty() {
                out.insert(id);
            }
        }
    }
    let source_contract_path = repo_root.join("data/external/SOURCES.toml");
    if source_contract_path.exists() {
        let raw = load_toml(&source_contract_path)?;
        for row in table_array(&raw, "source")? {
            let id = string_field(&row, "id");
            if !id.is_empty() {
                out.insert(id);
            }
        }
    }
    Ok(out)
}

fn infer_runtime_stack(module_name: &str) -> String {
    let name = module_name.to_lowercase();
    BTreeMap::from([
        ("core".to_string(), "mixed".to_string()),
        ("algebra".to_string(), "rust".to_string()),
        ("analysis".to_string(), "python".to_string()),
        ("astro".to_string(), "python".to_string()),
        ("materials".to_string(), "mixed".to_string()),
        ("particle".to_string(), "python".to_string()),
        ("quantum_docker".to_string(), "docker_python".to_string()),
        ("rocq".to_string(), "rocq".to_string()),
        ("latex".to_string(), "latex".to_string()),
        ("cpp".to_string(), "cpp".to_string()),
    ])
    .get(&name)
    .cloned()
    .unwrap_or_else(|| "mixed".to_string())
}

fn module_dependency_defaults(module_id: &str) -> Vec<String> {
    if module_id == "REQ-CORE" {
        Vec::new()
    } else {
        vec!["REQ-CORE".to_string()]
    }
}

fn load_toml(path: &Path) -> Result<Table> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    // Use toml::from_str rather than .parse::<Value>(): in toml 1.1 the FromStr
    // implementation uses a stricter parser that rejects valid [table] headers.
    let value =
        toml::from_str::<Value>(&text).with_context(|| format!("parse TOML {}", path.display()))?;
    let table = value
        .as_table()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("root TOML document is not a table: {}", path.display()))?;
    Ok(table)
}

fn load_control_plane_registry(
    repo_root: &Path,
    db_rel_path: &Path,
    kind: ControlPlaneCompatKind,
    fallback_rel_path: &str,
) -> Result<Table> {
    let db_path = repo_root.join(db_rel_path);
    if db_path.exists() {
        let mut store = ProvenanceStore::open(&db_path)
            .with_context(|| format!("open canonical control-plane DB {}", db_path.display()))?;
        let text = store.control_plane_compat_text(kind).with_context(|| {
            format!(
                "render {:?} compatibility text from {}",
                kind,
                db_path.display()
            )
        })?;
        // Use toml::from_str rather than .parse::<Value>(): in toml 1.1 the FromStr
        // implementation uses a stricter parser that rejects valid [[table]] headers.
        let value = toml::from_str::<Value>(&text)
            .with_context(|| format!("parse {:?} compatibility TOML", kind))?;
        let table = value.as_table().cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "root TOML document is not a table for {:?} compatibility text",
                kind
            )
        })?;
        return Ok(table);
    }
    load_toml(&repo_root.join(fallback_rel_path))
}

fn load_planning_compat_export(
    repo_root: &Path,
    db_rel_path: &Path,
    table: PlanningCompatTable,
) -> Result<Option<String>> {
    let db_path = repo_root.join(db_rel_path);
    if !db_path.exists() {
        return Ok(None);
    }
    let store = ProvenanceStore::open(&db_path)
        .with_context(|| format!("open canonical control-plane DB {}", db_path.display()))?;
    let text = store.render_planning_compat_toml(table).with_context(|| {
        format!(
            "render {:?} planning compatibility text from {}",
            table,
            db_path.display()
        )
    })?;
    Ok(Some(text))
}

fn load_requirements_compat_export(repo_root: &Path, db_rel_path: &Path) -> Result<Option<String>> {
    let db_path = repo_root.join(db_rel_path);
    if !db_path.exists() {
        return Ok(None);
    }
    let store = ProvenanceStore::open(&db_path)
        .with_context(|| format!("open canonical control-plane DB {}", db_path.display()))?;
    let text = store.render_requirements_compat_toml().with_context(|| {
        format!(
            "render requirements compatibility text from {}",
            db_path.display()
        )
    })?;
    Ok(Some(text))
}

fn load_workspace_bench_targets(repo_root: &Path) -> Result<BTreeSet<String>> {
    let root_manifest = load_toml(&repo_root.join("Cargo.toml"))?;
    let members = root_manifest
        .get("workspace")
        .and_then(Value::as_table)
        .and_then(|workspace| workspace.get("members"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut targets = BTreeSet::new();
    for member in members {
        let Some(member_rel) = member.as_str() else {
            continue;
        };
        let manifest_path = if member_rel.ends_with("Cargo.toml") {
            repo_root.join(member_rel)
        } else {
            repo_root.join(member_rel).join("Cargo.toml")
        };
        if !manifest_path.exists() {
            continue;
        }
        let manifest = load_toml(&manifest_path)?;
        for bench in table_array(&manifest, "bench")? {
            let name = string_field(&bench, "name");
            if !name.is_empty() {
                targets.insert(name);
            }
        }
    }
    Ok(targets)
}

fn sync_or_write_experiments_registry(repo_root: &Path, args: &Args, content: &str) -> Result<()> {
    let experiments_out_path = repo_root.join(&args.experiments_out);
    let canonical_experiments_path = repo_root.join("registry/experiments.toml");
    let db_path = repo_root.join(&args.db);
    if experiments_out_path == canonical_experiments_path && db_path.exists() {
        let mut store = ProvenanceStore::open(&db_path)
            .with_context(|| format!("open canonical control-plane DB {}", db_path.display()))?;
        store.replace_control_plane_experiments_from_registry_text(
            repo_root,
            &canonical_experiments_path,
            content,
        )?;
        let claims = repo_root.join("registry/claims.toml");
        let insights = repo_root.join("registry/insights.toml");
        let binaries = repo_root.join("registry/binaries.toml");
        let theorems = repo_root.join("docs/THEOREMS.md");
        let theorems_mirror = repo_root.join("docs/generated/THEOREMS_REGISTRY_MIRROR.md");
        store.export_control_plane_compat_paths(
            repo_root,
            provenance_store::CompatExportPaths {
                claims: &claims,
                insights: &insights,
                experiments: &canonical_experiments_path,
                binaries: &binaries,
                theorems: &theorems,
                theorems_mirror: &theorems_mirror,
            },
        )?;
        return Ok(());
    }
    write_ascii(&experiments_out_path, content)
}

fn write_ascii(path: &Path, content: &str) -> Result<()> {
    assert_ascii_text(content, &path.display().to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create parent {}", parent.display()))?;
    }
    fs::write(path, content).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn assert_ascii_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    assert_ascii_text(&text, &path.display().to_string())
}

fn assert_ascii_text(text: &str, context: &str) -> Result<()> {
    let mut bad = BTreeSet::new();
    for ch in text.chars() {
        if (ch != '\n' && ch != '\r' && ch != '\t') && ch as u32 > 127 {
            bad.insert(ch);
        }
    }
    if !bad.is_empty() {
        let sample = bad.iter().take(20).collect::<String>();
        bail!("ERROR: non-ASCII output in {context}: {sample:?}");
    }
    Ok(())
}

fn ascii_clean(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        let code = ch as u32;
        if ch == '\n' || ch == '\r' || ch == '\t' {
            out.push(ch);
        } else if code < 32 {
            out.push(' ');
        } else if code <= 127 {
            out.push(ch);
        } else {
            out.push_str(&format!("\\u{code:04X}"));
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

#[cfg(test)]
// The `mod tests` block is followed by helper fns that ARE referenced
// by tests but are placed at file scope so they can be reused across
// test modules; clippy::items_after_test_module is overly strict here.
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestWorkspace {
        root: PathBuf,
        db: PathBuf,
    }

    impl Drop for TestWorkspace {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn load_workspace_bench_targets_includes_named_bench() -> Result<()> {
        let fixture = make_test_workspace("bench_targets")?;
        let benches = load_workspace_bench_targets(&fixture.root)?;
        assert!(benches.contains("x87_bench"));
        Ok(())
    }

    #[test]
    fn sync_or_write_experiments_registry_updates_db_and_export() -> Result<()> {
        let fixture = make_test_workspace("sync_experiments")?;
        let args = Args {
            repo_root: fixture.root.clone(),
            db: PathBuf::from("registry/canonical/control_plane.sqlite3"),
            verify: false,
            experiments_out: PathBuf::from("registry/experiments.toml"),
            lineage_out: PathBuf::from("registry/experiment_lineage.toml"),
            roadmap_out: PathBuf::from("registry/roadmap.toml"),
            todo_out: PathBuf::from("registry/todo.toml"),
            next_actions_out: PathBuf::from("registry/next_actions.toml"),
            requirements_out: PathBuf::from("registry/requirements.toml"),
            module_requirements_out: PathBuf::from("registry/module_requirements.toml"),
        };
        let replacement = r#"[experiments]
authoritative = true
status_allowlist = ["active", "planned", "blocked", "deprecated"]

[[experiment]]
id = "E-002"
title = "Synced replacement experiment"
status = "active"
binary = "mini-bin"
claim_refs = ["C-001"]
deterministic = true
"#;

        sync_or_write_experiments_registry(&fixture.root, &args, replacement)?;

        let mut store = ProvenanceStore::open(&fixture.db)?;
        let rendered = store.control_plane_compat_text(ControlPlaneCompatKind::Experiments)?;
        let exported = fs::read_to_string(fixture.root.join("registry/experiments.toml"))?;
        assert!(rendered.contains("id = \"E-002\""));
        assert!(!rendered.contains("id = \"E-001\""));
        assert!(exported.contains("id = \"E-002\""));
        store.verify_control_plane_invariants(&fixture.root)?;
        let claims = fixture.root.join("registry/claims.toml");
        let insights = fixture.root.join("registry/insights.toml");
        let experiments = fixture.root.join("registry/experiments.toml");
        let binaries = fixture.root.join("registry/binaries.toml");
        let theorems = fixture.root.join("docs/THEOREMS.md");
        let theorems_mirror = fixture
            .root
            .join("docs/generated/THEOREMS_REGISTRY_MIRROR.md");
        store.verify_control_plane_compat_exports_paths(
            &fixture.root,
            provenance_store::CompatExportPaths {
                claims: &claims,
                insights: &insights,
                experiments: &experiments,
                binaries: &binaries,
                theorems: &theorems,
                theorems_mirror: &theorems_mirror,
            },
        )?;
        Ok(())
    }

    #[test]
    fn extract_paths_normalizes_bare_registry_claims_file() {
        let paths = extract_paths("None (hardcoded stream statuses from claims.toml)");
        assert_eq!(paths, vec!["registry/claims.toml"]);
    }

    #[test]
    fn extract_paths_normalizes_bare_registry_basename_before_sentence_period() {
        let paths = extract_paths(r#"input = "Compare against claims.toml.""#);
        assert_eq!(paths, vec!["registry/claims.toml"]);
    }

    #[test]
    fn extract_paths_does_not_normalize_basename_inside_real_path() {
        let paths = extract_paths("Read data/archive/claims.toml before registry export.");
        assert_eq!(paths, vec!["data/archive/claims.toml"]);
    }

    #[test]
    fn extract_paths_does_not_normalize_registry_basename_with_suffix() {
        let paths = extract_paths("Read claims.toml.bak before registry export.");
        assert!(paths.is_empty());
    }

    #[test]
    fn experiment_lineage_surfaces_bare_claims_registry_input() -> Result<()> {
        let raw: Value = toml::from_str(
            r#"
[[experiment]]
id = "E-079"
title = "Sterile-Neutrino Null-Result Audit"
status = "active"
binary = "sterile-neutrino-audit"
claims = ["C-703"]
input = "None (hardcoded stream statuses from claims.toml)"
output = ["reports/sterile_neutrino_audit.toml"]
deterministic = true
"#,
        )?;
        let experiment_rows = build_experiment_rows(
            &table_array(
                raw.as_table()
                    .context("experiment fixture root must be table")?,
                "experiment",
            )?,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
        )?;
        assert_eq!(
            experiment_rows[0].input_path_refs,
            vec!["registry/claims.toml"]
        );

        let claim_ids = BTreeSet::from(["C-703".to_string()]);
        let (lineages, edges) =
            build_experiment_lineage(&experiment_rows, &claim_ids, &BTreeSet::new());

        assert_eq!(lineages[0].input_path_refs, vec!["registry/claims.toml"]);
        assert!(edges.iter().any(|edge| {
            edge.from_id == "E-079"
                && edge.to_ref == "registry/claims.toml"
                && edge.edge_kind == "consumes_path"
        }));
        Ok(())
    }

    fn make_test_workspace(label: &str) -> Result<TestWorkspace> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gororoba_execution_planning_{label}_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root)?;

        write_ascii(
            &root.join("Cargo.toml"),
            r#"[workspace]
resolver = "3"
members = ["crates/test_cli"]
"#,
        )?;
        write_ascii(
            &root.join("crates/test_cli/Cargo.toml"),
            r#"[package]
name = "test_cli"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "mini-bin"
path = "src/main.rs"

[[bench]]
name = "x87_bench"
path = "benches/x87_bench.rs"
"#,
        )?;
        write_ascii(
            &root.join("crates/test_cli/src/main.rs"),
            "fn main() { println!(\"mini-bin\"); }\n",
        )?;
        write_ascii(
            &root.join("crates/test_cli/benches/x87_bench.rs"),
            "fn main() {}\n",
        )?;

        write_ascii(
            &root.join("registry/claims.toml"),
            r#"[[claim]]
id = "C-001"
statement = "Mini claim"
status = "Verified"
where_stated = "`crates/test_cli/src/main.rs`"
last_verified = "2026-03-13"
formal_proof = "proofs/verified/C001_Test.v"
status_note = "Mini proof"
"#,
        )?;
        write_ascii(
            &root.join("registry/insights.toml"),
            r#"[[insight]]
id = "I-001"
title = "Mini insight"
status = "verified"
claims = ["C-001"]
"#,
        )?;
        write_ascii(
            &root.join("registry/experiments.toml"),
            r#"[experiments]
authoritative = true
status_allowlist = ["active", "planned", "blocked", "deprecated"]

[[experiment]]
id = "E-001"
title = "Original experiment"
status = "active"
binary = "mini-bin"
claim_refs = ["C-001"]
deterministic = true
"#,
        )?;
        write_ascii(
            &root.join("registry/binaries.toml"),
            r#"[[binary]]
name = "mini-bin"
crate = "test_cli"
description = "Mini binary"
experiment = "E-001"
"#,
        )?;
        write_ascii(&root.join("proofs/_RocqProject"), "verified/C001_Test.v\n")?;
        write_ascii(
            &root.join("proofs/verified/C001_Test.v"),
            "(* proof placeholder *)\n",
        )?;

        let db = root.join("registry/canonical/control_plane.sqlite3");
        let mut store = ProvenanceStore::open(&db)?;
        store.reindex_control_plane_from_registries(
            &root,
            provenance_store::RegistryImportPaths {
                claims: &root.join("registry/claims.toml"),
                insights: &root.join("registry/insights.toml"),
                experiments: &root.join("registry/experiments.toml"),
                binaries: &root.join("registry/binaries.toml"),
                rocq_project: &root.join("proofs/_RocqProject"),
            },
            provenance_store::ReimportOptions::bootstrap(),
        )?;
        let claims = root.join("registry/claims.toml");
        let insights = root.join("registry/insights.toml");
        let experiments = root.join("registry/experiments.toml");
        let binaries = root.join("registry/binaries.toml");
        let theorems = root.join("docs/THEOREMS.md");
        let theorems_mirror = root.join("docs/generated/THEOREMS_REGISTRY_MIRROR.md");
        store.export_control_plane_compat_paths(
            &root,
            provenance_store::CompatExportPaths {
                claims: &claims,
                insights: &insights,
                experiments: &experiments,
                binaries: &binaries,
                theorems: &theorems,
                theorems_mirror: &theorems_mirror,
            },
        )?;

        Ok(TestWorkspace { root, db })
    }
}

fn build_status_token(status: &str) -> String {
    let mut token = collapse(status)
        .to_uppercase()
        .replace(['/', '-', ' '], "_");
    token = token
        .chars()
        .map(|ch| {
            if ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    while token.contains("__") {
        token = token.replace("__", "_");
    }
    token.trim_matches('_').to_string().if_empty("UNSPECIFIED")
}

fn verify_status_token(status: &str) -> String {
    status.trim().to_uppercase().replace('-', "_")
}

fn extract_id_refs(text: &str) -> Vec<String> {
    dedup_sorted(
        id_ref_regex()
            .find_iter(&ascii_clean(text))
            .map(|m| m.as_str().to_string())
            .collect(),
    )
}

fn extract_paths(text: &str) -> Vec<String> {
    let cleaned = ascii_clean(text);
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut push_path = |item: String| {
        if !item.is_empty() && seen.insert(item.clone()) {
            out.push(item);
        }
    };
    for found in path_regex().find_iter(&cleaned) {
        let item = collapse(found.as_str())
            .trim_end_matches(|ch: char| ['.', ',', ';', ':', ')'].contains(&ch))
            .to_string();
        push_path(item);
    }
    for found in registry_basename_regex().find_iter(&cleaned) {
        if !registry_basename_match_is_bare(&cleaned, found.start(), found.end()) {
            continue;
        }
        if let Some(path) = registry_basename_path(found.as_str()) {
            push_path(path);
        }
    }
    out
}

fn registry_basename_match_is_bare(text: &str, start: usize, end: usize) -> bool {
    fn is_previous_path_continuation(ch: char) -> bool {
        matches!(ch, '/' | '\\' | '.' | '-' | '_') || ch.is_ascii_alphanumeric()
    }
    fn is_next_path_continuation(after_match: &str) -> bool {
        let mut chars = after_match.chars();
        match chars.next() {
            Some('.') => chars
                .next()
                .is_some_and(|ch| ch == '_' || ch == '-' || ch.is_ascii_alphanumeric()),
            Some(ch) => matches!(ch, '/' | '\\' | '-' | '_') || ch.is_ascii_alphanumeric(),
            None => false,
        }
    }

    let previous_is_clear = match text[..start].chars().next_back() {
        Some(ch) => !is_previous_path_continuation(ch),
        None => true,
    };
    let next_is_clear = !is_next_path_continuation(&text[end..]);
    previous_is_clear && next_is_clear
}

fn registry_basename_path(name: &str) -> Option<String> {
    match name {
        "binaries.toml" | "claims.toml" | "experiments.toml" | "insights.toml" | "todo.toml" => {
            Some(format!("registry/{name}"))
        }
        _ => None,
    }
}

fn normalize_binary_name(value: &str) -> String {
    let binary = collapse(value);
    if ["N/A", "NA", "NONE"].contains(&binary.to_uppercase().as_str()) {
        return String::new();
    }
    if binary.contains('/')
        || binary.ends_with(".py")
        || binary.ends_with(".sh")
        || binary.ends_with(".bash")
    {
        return String::new();
    }
    binary
}

fn normalize_dataset_label(value: &str) -> String {
    collapse(value).to_lowercase()
}

fn normalize_string_list(value: Option<&Value>) -> Vec<String> {
    let Some(Value::Array(values)) = value else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for value in values {
        let item = collapse(&value_to_string(value));
        if !item.is_empty() && seen.insert(item.clone()) {
            out.push(item);
        }
    }
    out
}

fn parse_dependency_spec(spec: &str) -> (String, String) {
    let clean = collapse(spec);
    let captures = dep_spec_regex().captures(&clean);
    if let Some(caps) = captures {
        let name = collapse(caps.get(1).map(|m| m.as_str()).unwrap_or(""));
        let constraint = collapse(caps.get(2).map(|m| m.as_str()).unwrap_or(""));
        (name, constraint)
    } else {
        (clean, String::new())
    }
}

fn dedup_sorted(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn table_array(root: &Table, key: &str) -> Result<Vec<Table>> {
    let Some(value) = root.get(key) else {
        return Ok(Vec::new());
    };
    let Some(values) = value.as_array() else {
        bail!("expected array for key {key}");
    };
    Ok(values
        .iter()
        .filter_map(|value| value.as_table().cloned())
        .collect())
}

fn table_value(root: &Table, key: &str) -> Table {
    root.get(key)
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default()
}

/// The registered target inside an execution-target citation.
///
/// A cargo bin or bench target is one word. A dispatcher that collapsed a
/// cluster of lanes into subcommands is cited with the lane appended, as in
/// `turboquant bench` or `heliosphere predictive-eval`, and only the first word
/// names a target cargo can build. Splitting on whitespace therefore recovers
/// the buildable name while `registry/experiments.toml` keeps the lane in its
/// `binary` field, which is what distinguishes one lane's evidence from
/// another's.
fn execution_target_head(target: &str) -> &str {
    target.split_whitespace().next().unwrap_or(target)
}

/// Whether an execution-target citation names a registered target, accepting
/// both the bare target and the dispatcher-plus-lane form.
fn execution_target_registered(target: &str, registered: &BTreeSet<String>) -> bool {
    registered.contains(target) || registered.contains(execution_target_head(target))
}

fn string_field(table: &Table, key: &str) -> String {
    table
        .get(key)
        .map(value_to_string)
        .map(|value| collapse(&value))
        .unwrap_or_default()
}

fn integer_field(table: &Table, key: &str, default: i64) -> i64 {
    table
        .get(key)
        .and_then(Value::as_integer)
        .unwrap_or(default)
}

fn bool_field(table: &Table, key: &str) -> bool {
    table.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn string_list_field(table: &Table, key: &str) -> Vec<String> {
    normalize_string_list(table.get(key))
}

fn string_list_from_table(table: &Table, key: &str) -> Vec<String> {
    normalize_string_list(table.get(key))
}

fn list_field(table: &Table, key: &str) -> Vec<Value> {
    table
        .get(key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn ids_from_rows(rows: &[Table]) -> BTreeSet<String> {
    rows.iter()
        .map(|row| string_field(row, "id"))
        .filter(|value| !value.is_empty())
        .collect()
}

fn rows_by_id(rows: &[Table]) -> BTreeMap<String, Table> {
    let mut out = BTreeMap::new();
    for row in rows {
        let id = string_field(row, "id");
        if !id.is_empty() {
            out.insert(id, row.clone());
        }
    }
    out
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Integer(v) => v.to_string(),
        Value::Float(v) => v.to_string(),
        Value::Boolean(v) => v.to_string(),
        Value::Datetime(v) => v.to_string(),
        other => other.to_string(),
    }
}

fn toml_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::String(v) => serde_json::Value::String(v.clone()),
        Value::Integer(v) => serde_json::Value::Number((*v).into()),
        Value::Float(v) => serde_json::Number::from_f64(*v)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::Boolean(v) => serde_json::Value::Bool(*v),
        Value::Datetime(v) => serde_json::Value::String(v.to_string()),
        Value::Array(values) => serde_json::Value::Array(values.iter().map(toml_to_json).collect()),
        Value::Table(table) => serde_json::Value::Object(
            table
                .iter()
                .map(|(key, value)| (key.clone(), toml_to_json(value)))
                .collect(),
        ),
    }
}

fn q(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn render_list(values: &[String]) -> String {
    if values.is_empty() {
        "[]".to_string()
    } else {
        format!(
            "[{}]",
            values
                .iter()
                .map(|value| q(value))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

trait IfEmpty {
    fn if_empty(self, fallback: &str) -> String;
}

impl IfEmpty for String {
    fn if_empty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
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

fn id_ref_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        Regex::new(
            r"\b(?:WS-[A-Z0-9-]+|T-\d{3,}|NA-\d{3,}|C-\d{3,}|I-\d{3,}|E-\d{3,}|REQ-[A-Z0-9-]+)\b",
        )
        .unwrap()
    })
}

fn path_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        Regex::new(r"(?:data|registry|docs|crates|src|tests)/[A-Za-z0-9_./{}:+-]+").unwrap()
    })
}

fn registry_basename_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        Regex::new(r"\b(?:binaries|claims|experiments|insights|todo)\.toml\b").unwrap()
    })
}

fn dataset_id_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^(?:PC|PG|EX|AR|CU)-\d{4}$").unwrap())
}

fn dep_spec_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^\s*([A-Za-z0-9_.-]+)\s*(.*)$").unwrap())
}

fn lineage_id_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^XL-\d{3}$").unwrap())
}

fn claim_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^C-\d{3,}$").unwrap())
}

fn insight_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^I-\d{3,}$").unwrap())
}

fn experiment_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^E-\d{3,}$").unwrap())
}

fn workstream_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^WS-[A-Z0-9-]+$").unwrap())
}

fn todo_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^T-\d{3,}$").unwrap())
}

fn action_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^NA-\d{3,}$").unwrap())
}

fn req_regex() -> &'static Regex {
    static ONCE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| Regex::new(r"^REQ-[A-Z0-9-]+$").unwrap())
}
