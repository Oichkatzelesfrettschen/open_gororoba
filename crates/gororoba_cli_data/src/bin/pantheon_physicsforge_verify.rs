use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use regex::Regex;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const REQUIRED_LICENSE: &str = "GPL-2.0-only";
const MATRIX_PATH: &str =
    "archive/registry/pantheon_physicsforge/pantheon_physicsforge_migration_matrix.toml";
const TODO_PATH: &str =
    "archive/registry/pantheon_physicsforge/pantheon_physicsforge_migration_todo.toml";
const PORTED_FILES_PATH: &str =
    "archive/registry/pantheon_physicsforge/pantheon_physicsforge_ported_files.toml";
const ALIGNMENT_PATH: &str =
    "archive/registry/pantheon_physicsforge/pantheon_physicsforge_license_alignment.toml";
const TRACKER_PATH: &str =
    "archive/registry/pantheon_physicsforge/pantheon_physicsforge_overflow_tracker.toml";

#[derive(Parser, Debug)]
#[command(
    name = "pantheon-physicsforge-verify",
    about = "Verify Pantheon/PhysicsForge migration governance and licensing contracts"
)]
struct Cli {
    #[command(subcommand)]
    command: CommandSet,
}

#[derive(Subcommand, Debug)]
enum CommandSet {
    All,
    License,
    Provenance,
    Mapping,
    LicenseHeaders,
    Overflow,
}

fn repo_root() -> PathBuf {
    repo_root::resolve!()
}

fn load_toml(path: &Path) -> Result<toml::Value> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn table<'a>(value: &'a toml::Value, key: &str) -> Option<&'a toml::value::Table> {
    value.get(key).and_then(toml::Value::as_table)
}

fn array<'a>(value: &'a toml::Value, key: &str) -> Option<&'a Vec<toml::Value>> {
    value.get(key).and_then(toml::Value::as_array)
}

fn string_field(table: &toml::value::Table, key: &str) -> String {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn bool_field(table: &toml::value::Table, key: &str) -> bool {
    table
        .get(key)
        .and_then(toml::Value::as_bool)
        .unwrap_or(false)
}

fn int_field(table: &toml::value::Table, key: &str) -> i64 {
    table
        .get(key)
        .and_then(toml::Value::as_integer)
        .unwrap_or(-1)
}

fn string_array_field(table: &toml::value::Table, key: &str) -> Vec<String> {
    table
        .get(key)
        .and_then(toml::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(toml::Value::as_str)
                .map(|value| value.trim().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn check_license_file(path: &Path, failures: &mut Vec<String>) {
    if !path.is_file() {
        failures.push(format!("missing file: {}", path.display()));
        return;
    }
    let text = fs::read_to_string(path).unwrap_or_default();
    if !text.contains("GNU GENERAL PUBLIC LICENSE") || !text.contains("Version 2, June 1991") {
        failures.push(format!("{}: expected GPLv2 canonical text", path.display()));
    }
}

fn check_pyproject(path: &Path, failures: &mut Vec<String>) {
    if !path.is_file() {
        failures.push(format!("missing file: {}", path.display()));
        return;
    }
    let Ok(value) = load_toml(path) else {
        failures.push(format!("failed to parse {}", path.display()));
        return;
    };
    let text = value
        .get("project")
        .and_then(toml::Value::as_table)
        .and_then(|project| project.get("license"))
        .and_then(toml::Value::as_table)
        .and_then(|license| license.get("text"))
        .and_then(toml::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if text != REQUIRED_LICENSE {
        failures.push(format!(
            "{}: project.license.text={text:?}, expected {REQUIRED_LICENSE:?}",
            path.display()
        ));
    }
}

fn check_readme(path: &Path, failures: &mut Vec<String>) {
    if !path.is_file() {
        failures.push(format!("missing file: {}", path.display()));
        return;
    }
    let text = fs::read_to_string(path).unwrap_or_default();
    if !text.contains(REQUIRED_LICENSE) && !text.contains("GPL v2 only") {
        failures.push(format!(
            "{}: missing explicit GPL-2.0-only/GPL v2 only statement",
            path.display()
        ));
    }
}

fn check_no_fallback_license_mentions(path: &Path, failures: &mut Vec<String>) {
    if !path.is_file() {
        failures.push(format!("missing file: {}", path.display()));
        return;
    }
    let text = fs::read_to_string(path).unwrap_or_default();
    let patterns = [r"CC-BY-4\.0", r"\bMIT\b", r"\bApache\b", r"\bproprietary\b"];
    for pattern in patterns {
        let re = Regex::new(pattern).expect("valid regex");
        if re.is_match(&text) {
            failures.push(format!(
                "{}: contains forbidden fallback license token {pattern:?}",
                path.display()
            ));
        }
    }
}

fn run_license(root: &Path) -> Result<()> {
    let github_root = root
        .parent()
        .ok_or_else(|| anyhow::anyhow!("repo root has no parent"))?;
    let pantheon = github_root.join("pantheon");
    let physicsforge = github_root.join("PhysicsForge");
    let mut failures = Vec::new();

    check_license_file(&pantheon.join("LICENSE"), &mut failures);
    check_pyproject(&pantheon.join("pyproject.toml"), &mut failures);
    check_readme(&pantheon.join("README.md"), &mut failures);

    check_license_file(&physicsforge.join("LICENSE"), &mut failures);
    check_pyproject(&physicsforge.join("pyproject.toml"), &mut failures);
    check_readme(&physicsforge.join("README.md"), &mut failures);
    check_no_fallback_license_mentions(
        &physicsforge.join("docs").join("RELEASE_NOTES_v1.0.md"),
        &mut failures,
    );

    if !failures.is_empty() {
        println!("ERROR: Pantheon/PhysicsForge license consistency verification failed.");
        for failure in failures {
            println!("- {failure}");
        }
        bail!("pantheon/physicsforge license verification failed");
    }

    println!("OK: Pantheon and PhysicsForge are aligned to GPL-2.0-only policy.");
    Ok(())
}

fn run_provenance(root: &Path) -> Result<()> {
    let registry_path = root.join(PORTED_FILES_PATH);
    let data = load_toml(&registry_path)?;
    let gate = table(&data, "provenance_gate").cloned().unwrap_or_default();
    let required_prefixes = string_array_field(&gate, "required_prefixes");
    let reject_verbatim = gate
        .get("reject_verbatim_copy")
        .and_then(toml::Value::as_bool)
        .unwrap_or(true);
    let rows = array(&data, "ported_file").cloned().unwrap_or_default();

    let mut failures = Vec::new();
    let mut by_path = BTreeMap::new();

    for row in rows {
        let Some(row_table) = row.as_table() else {
            failures.push("ported_file row is not a table".to_string());
            continue;
        };
        let path = string_field(row_table, "path");
        if path.is_empty() {
            failures.push("ported_file row missing path".to_string());
            continue;
        }
        if by_path.contains_key(&path) {
            failures.push(format!("duplicate ported_file row for {path}"));
            continue;
        }
        by_path.insert(path.clone(), row_table.clone());
        let file_path = root.join(&path);
        if !file_path.is_file() {
            failures.push(format!("ported_file path does not exist: {path}"));
        }

        let copy_mode = string_field(row_table, "copy_mode");
        if reject_verbatim && copy_mode == "verbatim" {
            failures.push(format!("{path}: forbidden copy_mode=verbatim"));
        }

        let origin = string_field(row_table, "origin").to_lowercase();
        let mapping_ids = string_array_field(row_table, "source_mapping_ids");
        if matches!(origin.as_str(), "pantheon" | "physicsforge") && mapping_ids.is_empty() {
            failures.push(format!(
                "{path}: origin={origin} requires source_mapping_ids"
            ));
        }
        if !bool_field(row_table, "license_checked") {
            failures.push(format!("{path}: license_checked must be true"));
        }
    }

    for prefix in required_prefixes {
        let prefix_path = root.join(&prefix);
        if !prefix_path.exists() {
            failures.push(format!("required prefix does not exist: {prefix}"));
            continue;
        }
        for entry in walkdir::WalkDir::new(&prefix_path)
            .into_iter()
            .filter_map(std::result::Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            if !by_path.contains_key(&rel) {
                failures.push(format!("missing provenance row for migrated file: {rel}"));
            }
        }
    }

    if !failures.is_empty() {
        println!("ERROR: provenance gate failed for Pantheon/PhysicsForge migration.");
        for failure in failures {
            println!("- {failure}");
        }
        bail!("pantheon/physicsforge provenance verification failed");
    }

    println!(
        "OK: provenance gate passed (all required migrated files have canonical provenance rows)."
    );
    Ok(())
}

fn run_mapping(root: &Path) -> Result<()> {
    let matrix = load_toml(&root.join(MATRIX_PATH))?;
    let todo = load_toml(&root.join(TODO_PATH))?;

    let matrix_meta = table(&matrix, "migration_matrix")
        .cloned()
        .unwrap_or_default();
    let task_completion = table(&matrix, "task_completion")
        .cloned()
        .unwrap_or_default();
    let boundary_rules = array(&matrix, "boundary_rule").cloned().unwrap_or_default();
    let module_rows = array(&matrix, "module_mapping")
        .cloned()
        .unwrap_or_default();

    let todo_meta = table(&todo, "migration_todo").cloned().unwrap_or_default();
    let todo_rows = array(&todo, "task").cloned().unwrap_or_default();

    let mut failures = Vec::new();
    let mut todo_by_number = BTreeMap::new();
    for row in &todo_rows {
        let Some(row_table) = row.as_table() else {
            failures.push("todo task row is not a table".to_string());
            continue;
        };
        let number = int_field(row_table, "number");
        if todo_by_number.contains_key(&number) {
            failures.push(format!("duplicate migration todo number: {number}"));
            continue;
        }
        todo_by_number.insert(number, row_table.clone());
    }

    if int_field(&todo_meta, "task_count") as usize != todo_rows.len() {
        failures.push(format!(
            "migration_todo.task_count metadata mismatch ({} != {})",
            int_field(&todo_meta, "task_count"),
            todo_rows.len()
        ));
    }

    let counts = [
        ("done", "done_count"),
        ("in_progress", "in_progress_count"),
        ("blocked", "blocked_count"),
        ("todo", "todo_count"),
    ];
    for (status, meta_key) in counts {
        let actual = todo_rows
            .iter()
            .filter_map(toml::Value::as_table)
            .filter(|row| string_field(row, "status") == status)
            .count() as i64;
        if int_field(&todo_meta, meta_key) != actual {
            failures.push(format!(
                "migration_todo.{meta_key} metadata mismatch ({} != {})",
                int_field(&todo_meta, meta_key),
                actual
            ));
        }
    }

    let allowed_actions: BTreeSet<String> = boundary_rules
        .iter()
        .filter_map(toml::Value::as_table)
        .map(|row| string_field(row, "category"))
        .filter(|value| !value.is_empty())
        .collect();
    if allowed_actions.is_empty() {
        failures.push("no boundary_rule categories found in migration matrix".to_string());
    }

    let mut seen_mapping_ids = BTreeSet::new();
    for row in &module_rows {
        let Some(row_table) = row.as_table() else {
            failures.push("module_mapping row is not a table".to_string());
            continue;
        };
        let mapping_id = string_field(row_table, "id");
        let source_path = string_field(row_table, "source_path");
        let action = string_field(row_table, "action");
        let status = string_field(row_table, "status");
        let target_crate = string_field(row_table, "target_crate");
        let target_module = string_field(row_table, "target_module");

        if mapping_id.is_empty() {
            failures.push("module_mapping row missing id".to_string());
            continue;
        }
        if !seen_mapping_ids.insert(mapping_id.clone()) {
            failures.push(format!("duplicate module_mapping id: {mapping_id}"));
        }
        if source_path.is_empty() {
            failures.push(format!("module_mapping[{mapping_id}] missing source_path"));
        }
        if action.is_empty() {
            failures.push(format!("module_mapping[{mapping_id}] missing action"));
        } else if !allowed_actions.contains(&action) {
            failures.push(format!(
                "module_mapping[{mapping_id}] action not in boundary rules: {action}"
            ));
        }
        if status != "mapped" {
            failures.push(format!(
                "module_mapping[{mapping_id}] status must be 'mapped', got {status:?}"
            ));
        }
        if matches!(action.as_str(), "port" | "rewrite") {
            if matches!(target_crate.as_str(), "" | "none") {
                failures.push(format!(
                    "module_mapping[{mapping_id}] action={action} requires target_crate"
                ));
            }
            if matches!(target_module.as_str(), "" | "none") {
                failures.push(format!(
                    "module_mapping[{mapping_id}] action={action} requires target_module"
                ));
            }
        }
    }

    if module_rows.is_empty() {
        failures.push("migration matrix has zero module_mapping rows".to_string());
    }

    let scope_tasks = string_array_field(&matrix_meta, "scope_tasks");
    if scope_tasks.is_empty() {
        failures.push("migration_matrix.scope_tasks is empty".to_string());
    }

    let status_to_completion = BTreeMap::from([
        ("done", "completed"),
        ("in_progress", "in_progress"),
        ("blocked", "blocked"),
        ("todo", "pending"),
    ]);

    for task_num_str in &scope_tasks {
        let Ok(task_num) = task_num_str.parse::<i64>() else {
            failures.push(format!(
                "migration_matrix.scope_tasks contains non-numeric task: {task_num_str:?}"
            ));
            continue;
        };
        let Some(todo_row) = todo_by_number.get(&task_num) else {
            failures.push(format!("scope task missing in todo registry: {task_num}"));
            continue;
        };
        let completion_key = format!("task_{task_num}");
        let completion_state = task_completion
            .get(&completion_key)
            .and_then(toml::Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if completion_state.is_empty() {
            failures.push(format!("task_completion missing key: {completion_key}"));
            continue;
        }
        let todo_status = string_field(todo_row, "status");
        let Some(expected_completion) = status_to_completion.get(todo_status.as_str()) else {
            failures.push(format!(
                "todo task {task_num} has unknown status token: {todo_status:?}"
            ));
            continue;
        };
        if completion_state != *expected_completion {
            failures.push(format!(
                "task_completion[{completion_key}]={completion_state:?} does not match todo status {todo_status:?}"
            ));
        }
    }

    for task_num in 10..19 {
        let Some(row) = todo_by_number.get(&task_num) else {
            failures.push(format!("missing phase-2 mapping task row: {task_num}"));
            continue;
        };
        let evidence_refs = string_array_field(row, "evidence_refs");
        if !evidence_refs.iter().any(|value| value == MATRIX_PATH) {
            failures.push(format!(
                "todo task {task_num} evidence_refs must include {MATRIX_PATH}"
            ));
        }
    }

    if !failures.is_empty() {
        println!("ERROR: Pantheon/PhysicsForge mapping completeness verification failed.");
        for failure in failures {
            println!("- {failure}");
        }
        bail!("pantheon/physicsforge mapping verification failed");
    }

    println!(
        "OK: migration matrix and todo registries are consistent (module_mappings={}, scope_tasks={}).",
        module_rows.len(),
        scope_tasks.len()
    );
    Ok(())
}

fn scan_header(path: &Path, scan_lines: usize) -> Result<(bool, Vec<String>)> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read migrated file {}", path.display()))?;
    let head = text.lines().take(scan_lines).collect::<Vec<_>>().join("\n");
    let mut failures = Vec::new();
    let mut has_header_signal = false;

    let spdx_re =
        Regex::new(r"SPDX-License-Identifier\s*:\s*([^\s*#]+)").expect("valid SPDX regex");
    if let Some(captures) = spdx_re.captures(&head) {
        has_header_signal = true;
        let value = captures
            .get(1)
            .map(|m| m.as_str())
            .unwrap_or_default()
            .trim();
        if value != REQUIRED_LICENSE {
            failures.push(format!(
                "{}: SPDX header must be GPL-2.0-only, found {value:?}",
                path.display()
            ));
        }
    }

    let lower = head.to_lowercase();
    if lower.contains("gpl-2.0-only") || lower.contains("gpl v2 only") {
        has_header_signal = true;
    }

    let forbidden_patterns = [
        r"\bGPL-3\.0\b",
        r"\bGPL-3\.0-only\b",
        r"\bGPL-3\.0-or-later\b",
        r"\bLGPL\b",
        r"\bAGPL\b",
        r"\bMIT\b",
        r"\bApache\b",
        r"\bBSD\b",
        r"\bCC-BY\b",
        r"\bproprietary\b",
    ];
    for pattern in forbidden_patterns {
        let re = Regex::new(pattern).expect("valid forbidden license regex");
        if re.is_match(&head) {
            failures.push(format!(
                "{}: forbidden license token in header region: {pattern}",
                path.display()
            ));
        }
    }

    Ok((has_header_signal, failures))
}

fn run_license_headers(root: &Path) -> Result<()> {
    let ported = load_toml(&root.join(PORTED_FILES_PATH))?;
    let alignment = load_toml(&root.join(ALIGNMENT_PATH))?;
    let alignment_table = table(&alignment, "license_alignment")
        .cloned()
        .unwrap_or_default();
    let required_license = string_field(&alignment_table, "required_license");
    if required_license != REQUIRED_LICENSE {
        println!(
            "ERROR: license alignment policy drift detected. required_license={required_license:?}"
        );
        bail!("pantheon/physicsforge license alignment drift");
    }

    let rows = array(&ported, "ported_file").cloned().unwrap_or_default();
    let mut migrated_rows = Vec::new();
    for row in rows {
        let Some(row_table) = row.as_table() else {
            continue;
        };
        let origin = string_field(row_table, "origin").to_lowercase();
        let rel_path = string_field(row_table, "path");
        if !matches!(origin.as_str(), "pantheon" | "physicsforge") {
            continue;
        }
        if !(rel_path.ends_with(".rs") || rel_path.ends_with(".py")) {
            continue;
        }
        migrated_rows.push(rel_path);
    }

    let mut failures = Vec::new();
    let mut header_present_paths = Vec::new();
    let mut scanned_paths = Vec::new();
    for rel_path in migrated_rows {
        let file_path = root.join(&rel_path);
        if !file_path.is_file() {
            failures.push(format!("ported migrated file missing on disk: {rel_path}"));
            continue;
        }
        let (has_header, scan_failures) = scan_header(&file_path, 12)?;
        if has_header {
            header_present_paths.push(rel_path.clone());
        }
        scanned_paths.push(rel_path);
        failures.extend(scan_failures);
    }

    if !scanned_paths.is_empty()
        && !header_present_paths.is_empty()
        && header_present_paths.len() < scanned_paths.len()
    {
        failures.push(format!(
            "mixed license header style in migrated files: {} of {} have explicit headers",
            header_present_paths.len(),
            scanned_paths.len()
        ));
    }

    if !failures.is_empty() {
        println!("ERROR: Pantheon/PhysicsForge migrated license header verification failed.");
        for failure in failures {
            println!("- {failure}");
        }
        bail!("pantheon/physicsforge license header verification failed");
    }

    let header_mode =
        if !scanned_paths.is_empty() && header_present_paths.len() == scanned_paths.len() {
            "explicit_gpl2_headers"
        } else {
            "headerless_repo_style"
        };
    println!(
        "OK: migrated Rust/Python license header consistency verified (files={}, mode={}, required_license={}).",
        scanned_paths.len(),
        header_mode,
        required_license
    );
    Ok(())
}

fn run_overflow(root: &Path) -> Result<()> {
    let tracker = load_toml(&root.join(TRACKER_PATH))?;
    let meta = table(&tracker, "overflow_tracker")
        .cloned()
        .unwrap_or_default();
    let rows = array(&tracker, "overflow_task")
        .cloned()
        .unwrap_or_default();
    let max_active = int_field(&meta, "max_active_tasks");
    let active_statuses: BTreeSet<String> = meta
        .get("active_statuses")
        .and_then(toml::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(toml::Value::as_str)
                .map(|value| value.trim().to_string())
                .collect()
        })
        .unwrap_or_else(|| {
            ["open", "in_progress", "blocked"]
                .into_iter()
                .map(str::to_string)
                .collect()
        });
    let id_re = Regex::new(r"^OF-(\d{1,2})-(\d{3})$").expect("valid overflow id regex");

    let mut failures = Vec::new();
    let mut seen_ids = BTreeSet::new();
    let mut active_count = 0_i64;

    for row in rows.iter().filter_map(toml::Value::as_table) {
        let overflow_id = string_field(row, "id");
        let phase = int_field(row, "phase");
        let status = string_field(row, "status");
        let owner = string_field(row, "owner");
        let eta = string_field(row, "eta");
        let rationale = string_field(row, "rationale");
        let deferral_rationale = string_field(row, "deferral_rationale");

        if overflow_id.is_empty() {
            failures.push("overflow_task row missing id".to_string());
            continue;
        }
        if !seen_ids.insert(overflow_id.clone()) {
            failures.push(format!("duplicate overflow id: {overflow_id}"));
            continue;
        }

        if let Some(captures) = id_re.captures(&overflow_id) {
            let id_phase = captures
                .get(1)
                .and_then(|m| m.as_str().parse::<i64>().ok())
                .unwrap_or(-1);
            if id_phase != phase {
                failures.push(format!(
                    "overflow_task[{overflow_id}] phase mismatch (id encodes {id_phase}, row has {phase})"
                ));
            }
        } else {
            failures.push(format!("invalid overflow id format: {overflow_id}"));
        }

        if !(1..=10).contains(&phase) {
            failures.push(format!(
                "overflow_task[{overflow_id}] invalid phase: {phase}"
            ));
        }

        if active_statuses.contains(&status) {
            active_count += 1;
            if owner.is_empty() {
                failures.push(format!(
                    "overflow_task[{overflow_id}] active status requires owner"
                ));
            }
            if eta.is_empty() {
                failures.push(format!(
                    "overflow_task[{overflow_id}] active status requires eta"
                ));
            }
            if rationale.is_empty() {
                failures.push(format!(
                    "overflow_task[{overflow_id}] active status requires rationale"
                ));
            }
        }

        if status == "deferred" && deferral_rationale.is_empty() {
            failures.push(format!(
                "overflow_task[{overflow_id}] deferred status requires deferral_rationale"
            ));
        }
    }

    if active_count > max_active {
        failures.push(format!(
            "active overflow task limit exceeded ({active_count} > {max_active})"
        ));
    }
    if int_field(&meta, "active_count") != active_count {
        failures.push(format!(
            "overflow_tracker.active_count metadata mismatch ({} != {})",
            int_field(&meta, "active_count"),
            active_count
        ));
    }
    if int_field(&meta, "total_count") as usize != rows.len() {
        failures.push(format!(
            "overflow_tracker.total_count metadata mismatch ({} != {})",
            int_field(&meta, "total_count"),
            rows.len()
        ));
    }

    if !failures.is_empty() {
        println!("ERROR: Pantheon/PhysicsForge overflow tracker verification failed.");
        for failure in failures {
            println!("- {failure}");
        }
        bail!("pantheon/physicsforge overflow verification failed");
    }

    println!(
        "OK: overflow tracker verified (total={}, active={}, max_active={}).",
        rows.len(),
        active_count,
        max_active
    );
    Ok(())
}

fn run_all(root: &Path) -> Result<()> {
    run_license(root)?;
    run_provenance(root)?;
    run_mapping(root)?;
    run_license_headers(root)?;
    run_overflow(root)?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = repo_root();
    match cli.command {
        CommandSet::All => run_all(&root),
        CommandSet::License => run_license(&root),
        CommandSet::Provenance => run_provenance(&root),
        CommandSet::Mapping => run_mapping(&root),
        CommandSet::LicenseHeaders => run_license_headers(&root),
        CommandSet::Overflow => run_overflow(&root),
    }
}
