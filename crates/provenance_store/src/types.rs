//! Public row, table, and revision types for the `ProvenanceStore`
//! SQLite control plane.
//!
//! These types describe:
//! - Discriminators (`ControlPlaneCompatKind`, `PlanningCompatTable`).
//! - Borrow-style upsert rows used by the mutator API
//!   (`RoadmapItem<'a>`, `ActionItem<'a>`, requirements rows,
//!   `ResearchNarrativeRow<'a>`, `NotebookSessionRow<'a>`).
//! - Owned compat-row types used for read-back into TOML compatibility
//!   exports (`RoadmapCompatRow`, `ActionCompatRow`, the requirements
//!   *CompatRow* family).
//! - Read-back summaries (`ManifestRow`, `NotebookSessionSummary`).
//! - Bundles for argument cleanup (`CompatExportPaths<'a>`,
//!   `EntityFieldTarget<'a>`) that fold related identifiers into a
//!   single parameter.
//! - The append-only audit record (`StatusNoteRevision`) returned by
//!   the status_note mutators.

use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlPlaneCompatKind {
    Claims,
    Insights,
    Experiments,
    Binaries,
    Theorems,
    TheoremsMirror,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanningCompatTable {
    Roadmap,
    Todo,
    NextActions,
}

/// Row struct for roadmap item upserts.
pub struct RoadmapItem<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub priority: &'a str,
    pub status: &'a str,
    pub status_token: &'a str,
    pub description: &'a str,
    pub sprint: &'a str,
    pub dependencies_json: &'a str,
    pub acceptance_criteria_json: &'a str,
    pub primary_outputs_json: &'a str,
    pub evidence_refs_json: &'a str,
    pub lacunae_json: &'a str,
    pub claims_json: &'a str,
    pub insight: &'a str,
}

/// Row struct for todo / next-action item upserts.
pub struct ActionItem<'a> {
    pub id: &'a str,
    pub area: &'a str,
    pub title: &'a str,
    pub description: &'a str,
    pub priority: &'a str,
    pub status: &'a str,
    pub status_token: &'a str,
    pub dependencies_json: &'a str,
    pub acceptance_criteria_json: &'a str,
    pub evidence_refs_json: &'a str,
}

/// Singleton requirements registry metadata row.
pub struct RequirementsMeta<'a> {
    pub authoritative: bool,
    pub status: &'a str,
    pub status_token: &'a str,
    pub updated: &'a str,
    pub python_recommended: &'a str,
    pub python_allowed: &'a str,
    pub primary_markdown: &'a str,
    pub status_allowlist_json: &'a str,
    pub runtime_stack_allowlist_json: &'a str,
    pub required_module_fields_json: &'a str,
    pub required_gap_fields_json: &'a str,
}

/// Row struct for requirements module upserts.
pub struct RequirementModuleItem<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub markdown: &'a str,
    pub status: &'a str,
    pub status_token: &'a str,
    pub runtime_stack: &'a str,
    pub requires_modules_json: &'a str,
    pub install_targets_json: &'a str,
    pub verify_targets_json: &'a str,
    pub acceptance_criteria_json: &'a str,
}

/// Row struct for requirements coverage-gap upserts.
pub struct RequirementCoverageGapItem<'a> {
    pub id: &'a str,
    pub area: &'a str,
    pub status: &'a str,
    pub status_token: &'a str,
    pub description: &'a str,
    pub proposed_resolution: &'a str,
    pub related_module_ids_json: &'a str,
}

pub struct RoadmapCompatRow {
    pub id: String,
    pub name: String,
    pub priority: String,
    pub status: String,
    pub status_token: String,
    pub description: String,
    pub sprint: String,
    pub dependencies_json: String,
    pub acceptance_criteria_json: String,
    pub primary_outputs_json: String,
    pub evidence_refs_json: String,
    pub lacunae_json: String,
    pub claims_json: String,
    pub insight: String,
}

pub struct ActionCompatRow {
    pub id: String,
    pub area: String,
    pub title: String,
    pub description: String,
    pub priority: String,
    pub status: String,
    pub status_token: String,
    pub dependencies_json: String,
    pub acceptance_criteria_json: String,
    pub evidence_refs_json: String,
}

pub struct RequirementsMetaCompatRow {
    pub authoritative: bool,
    pub status: String,
    pub status_token: String,
    pub updated: String,
    pub python_recommended: String,
    pub python_allowed: String,
    pub primary_markdown: String,
    pub status_allowlist_json: String,
    pub runtime_stack_allowlist_json: String,
    pub required_module_fields_json: String,
    pub required_gap_fields_json: String,
}

pub struct RequirementModuleCompatRow {
    pub id: String,
    pub name: String,
    pub markdown: String,
    pub status: String,
    pub status_token: String,
    pub runtime_stack: String,
    pub requires_modules_json: String,
    pub install_targets_json: String,
    pub verify_targets_json: String,
    pub acceptance_criteria_json: String,
}

pub struct RequirementCoverageGapCompatRow {
    pub id: String,
    pub area: String,
    pub status: String,
    pub status_token: String,
    pub description: String,
    pub proposed_resolution: String,
    pub related_module_ids_json: String,
}

/// Row struct for research narrative upserts.
pub struct ResearchNarrativeRow<'a> {
    pub id: &'a str,
    pub source_markdown: &'a str,
    pub domain: &'a str,
    pub slug: &'a str,
    pub title: &'a str,
    pub status_token: &'a str,
    pub content_kind: &'a str,
    pub verification_level: &'a str,
    pub claim_refs_json: &'a str,
    pub url_refs_json: &'a str,
    pub path_refs_json: &'a str,
    pub body_markdown: &'a str,
    pub line_count: i64,
}

/// Row struct for notebook session upserts.
pub struct NotebookSessionRow<'a> {
    pub id: &'a str,
    pub title: &'a str,
    pub description: &'a str,
    pub kernel: &'a str,
    pub status: &'a str,
    pub cell_count: i64,
    pub cells_json: &'a str,
}

/// Row returned from `source_of_truth_manifest`.
pub struct ManifestRow {
    pub table_name: String,
    pub category: String,
    pub authoritative: bool,
    pub legacy_toml_path: String,
    pub description: String,
    pub migration_status: String,
}

/// Row returned from `list_notebook_sessions`.
pub struct NotebookSessionSummary {
    pub id: String,
    pub title: String,
    pub kernel: String,
    pub status: String,
    pub cell_count: i64,
}

/// Bundle of compat-export TOML target paths. The 6 paths form a logical
/// unit (the canonical compat-export surface: claims, insights,
/// experiments, binaries, theorems, theorems_mirror) so passing them as
/// individual `&Path` arguments would trip clippy::too_many_arguments and
/// scatter related state. Borrowed so callers can pass references without
/// allocation.
#[derive(Debug, Clone, Copy)]
pub struct CompatExportPaths<'a> {
    pub claims: &'a Path,
    pub insights: &'a Path,
    pub experiments: &'a Path,
    pub binaries: &'a Path,
    pub theorems: &'a Path,
    pub theorems_mirror: &'a Path,
}

/// SQL-identifier bundle for the generic per-column updater. Bundling the
/// four `&str` identifiers into one parameter keeps `entity_update_field`
/// under clippy::too_many_arguments without resorting to #[allow]; each
/// field MUST be a trusted constant from the call site (never user input)
/// because they are interpolated directly into SQL.
#[derive(Debug, Clone, Copy)]
pub struct EntityFieldTarget<'a> {
    pub table: &'a str,
    pub revisions_table: &'a str,
    pub fk_col: &'a str,
    pub field: &'a str,
}

/// Append-only audit record returned by claim/insight/experiment
/// status_note mutators. Useful for callers that want to print a
/// confirmation including the prev/new content hashes.
#[derive(Debug, Clone)]
pub struct StatusNoteRevision {
    pub entity_id: String,
    pub field_name: String,
    pub prev_value_sha256: Option<String>,
    pub new_value_sha256: String,
    pub actor: String,
    pub reason: Option<String>,
    pub revision_id: i64,
}
