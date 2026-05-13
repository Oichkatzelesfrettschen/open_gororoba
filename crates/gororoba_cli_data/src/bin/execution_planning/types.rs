//! Type definitions for the `execution-planning` binary: clap Args,
//! row-shaped data carriers (ExperimentRow, LineageRow, LineageEdge,
//! PlanningRow, RequirementModule, CoverageGap, ModuleRequirement,
//! ModuleCommand, ModulePackage) plus the `Table` type alias and the
//! REQUIREMENT_LANES constant list.
//!
//! Fields are `pub(crate)` so the bin root can construct and read.
//! Uses `#[path]` in the parent because the binary has an explicit
//! Cargo.toml path.

use clap::Parser;
use std::path::PathBuf;
use toml::Value;

pub(crate) type Table = toml::map::Map<String, Value>;

#[derive(Parser, Debug)]
#[command(
    name = "execution-planning",
    about = "Build or verify canonical execution-planning lane registries"
)]
pub(crate) struct Args {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) db: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) verify: bool,
    #[arg(long, default_value = "registry/experiments.toml")]
    pub(crate) experiments_out: PathBuf,
    #[arg(long, default_value = "registry/experiment_lineage.toml")]
    pub(crate) lineage_out: PathBuf,
    #[arg(long, default_value = "registry/roadmap.toml")]
    pub(crate) roadmap_out: PathBuf,
    #[arg(long, default_value = "registry/todo.toml")]
    pub(crate) todo_out: PathBuf,
    #[arg(long, default_value = "registry/next_actions.toml")]
    pub(crate) next_actions_out: PathBuf,
    #[arg(long, default_value = "registry/requirements.toml")]
    pub(crate) requirements_out: PathBuf,
    #[arg(long, default_value = "registry/module_requirements.toml")]
    pub(crate) module_requirements_out: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ExperimentRow {
    pub(crate) id: String,
    pub(crate) title: String,
    pub(crate) binary: String,
    pub(crate) binary_registered: bool,
    pub(crate) binary_experiment_declared: String,
    pub(crate) method: String,
    pub(crate) input: String,
    pub(crate) output: Vec<String>,
    pub(crate) run: String,
    pub(crate) run_command_sha256: String,
    pub(crate) claims: Vec<String>,
    pub(crate) claim_refs: Vec<String>,
    pub(crate) deterministic: bool,
    pub(crate) seed: Option<i64>,
    pub(crate) gpu: bool,
    pub(crate) status: String,
    pub(crate) status_token: String,
    pub(crate) lineage_id: String,
    pub(crate) input_path_refs: Vec<String>,
    pub(crate) output_path_refs: Vec<String>,
    pub(crate) dataset_refs: Vec<String>,
    pub(crate) dataset_label_refs: Vec<String>,
    pub(crate) external_source_refs: Vec<String>,
    pub(crate) truth_surface_consumption: Vec<String>,
    pub(crate) reproducibility_class: String,
}

#[derive(Debug, Clone)]
pub(crate) struct LineageRow {
    pub(crate) id: String,
    pub(crate) experiment_id: String,
    pub(crate) binary: String,
    pub(crate) deterministic: bool,
    pub(crate) seed: Option<i64>,
    pub(crate) gpu: bool,
    pub(crate) run_command: String,
    pub(crate) run_command_sha256: String,
    pub(crate) claim_refs: Vec<String>,
    pub(crate) input_path_refs: Vec<String>,
    pub(crate) output_path_refs: Vec<String>,
    pub(crate) dataset_refs: Vec<String>,
    pub(crate) dataset_label_refs: Vec<String>,
    pub(crate) external_source_refs: Vec<String>,
    pub(crate) truth_surface_consumption: Vec<String>,
    pub(crate) replay_steps: Vec<String>,
    pub(crate) acceptance_criteria: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct LineageEdge {
    pub(crate) id: String,
    pub(crate) lineage_id: String,
    pub(crate) from_id: String,
    pub(crate) to_ref: String,
    pub(crate) to_kind: String,
    pub(crate) edge_kind: String,
    pub(crate) verified: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct PlanningRow {
    pub(crate) raw: Table,
}

#[derive(Debug, Clone)]
pub(crate) struct RequirementModule {
    pub(crate) raw: Table,
}

#[derive(Debug, Clone)]
pub(crate) struct CoverageGap {
    pub(crate) raw: Table,
}

#[derive(Debug, Clone)]
pub(crate) struct ModuleRequirement {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) runtime_stack: String,
    pub(crate) status: String,
    pub(crate) status_token: String,
    pub(crate) source_markdown: String,
    pub(crate) requires_modules: Vec<String>,
    pub(crate) command_refs: Vec<String>,
    pub(crate) package_refs: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ModuleCommand {
    pub(crate) id: String,
    pub(crate) module_id: String,
    pub(crate) kind: String,
    pub(crate) command: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ModulePackage {
    pub(crate) id: String,
    pub(crate) module_id: String,
    pub(crate) manager: String,
    pub(crate) name: String,
    pub(crate) constraint: String,
    pub(crate) spec: String,
    pub(crate) group: String,
    pub(crate) optional: bool,
    pub(crate) source: String,
}

pub(crate) const ROADMAP_STATUS_ALLOWLIST: &[&str] = &[
    "planned",
    "active",
    "in_progress",
    "done",
    "paused",
    "blocked",
];
pub(crate) const TODO_STATUS_ALLOWLIST: &[&str] = &["open", "in_progress", "done", "blocked", "deferred"];
pub(crate) const ACTION_STATUS_ALLOWLIST: &[&str] = &["todo", "in_progress", "done", "blocked", "deferred"];
pub(crate) const PLANNING_PRIORITY_ALLOWLIST: &[&str] = &["high", "medium", "low"];
pub(crate) const REQUIREMENT_STATUS_ALLOWLIST: &[&str] = &["active", "deprecated", "planned", "blocked"];
pub(crate) const RUNTIME_STACK_ALLOWLIST: &[&str] = &[
    "mixed",
    "rust",
    "python",
    "docker_python",
    "rocq",
    "latex",
    "cpp",
];
pub(crate) const TRUTH_SURFACE_ALLOWLIST: &[&str] = &[
    "chronology_control",
    "environment_context",
    "lineage_transition",
    "observation_benchmark",
];
