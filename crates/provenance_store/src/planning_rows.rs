//! SQLite row-fetchers backing the planning + requirements
//! compatibility renderers.
//!
//! Methods added to `ProvenanceStore` via a second impl block:
//!   * `planning_roadmap_rows`     -> `Vec<RoadmapCompatRow>`
//!   * `planning_todo_rows`        -> `Vec<ActionCompatRow>`
//!   * `planning_next_action_rows` -> `Vec<ActionCompatRow>`
//!   * `requirements_meta_row`     -> `Option<RequirementsMetaCompatRow>`
//!   * `requirements_module_rows`  -> `Vec<RequirementModuleCompatRow>`
//!   * `requirements_coverage_gap_rows` -> `Vec<RequirementCoverageGapCompatRow>`
//!
//! Each consults parent's private `self.conn: Connection`. Methods
//! are pub so the planning_render submodule and the parent
//! render_requirements_compat_toml can call them via `self.X()`.

use anyhow::Result;
use rusqlite::OptionalExtension;

use crate::ProvenanceStore;
use crate::types::{
    ActionCompatRow, RequirementCoverageGapCompatRow, RequirementModuleCompatRow,
    RequirementsMetaCompatRow, RoadmapCompatRow,
};

impl ProvenanceStore {
    pub fn planning_roadmap_rows(&self) -> Result<Vec<RoadmapCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, priority, status, status_token, description, sprint,
                    dependencies_json, acceptance_criteria_json, primary_outputs_json,
                    evidence_refs_json, lacunae_json, claims_json, insight
             FROM roadmap_items
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(RoadmapCompatRow {
                id: row.get(0)?,
                name: row.get(1)?,
                priority: row.get(2)?,
                status: row.get(3)?,
                status_token: row.get(4)?,
                description: row.get(5)?,
                sprint: row.get(6)?,
                dependencies_json: row.get(7)?,
                acceptance_criteria_json: row.get(8)?,
                primary_outputs_json: row.get(9)?,
                evidence_refs_json: row.get(10)?,
                lacunae_json: row.get(11)?,
                claims_json: row.get(12)?,
                insight: row.get(13)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn planning_todo_rows(&self) -> Result<Vec<ActionCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, area, title, description, priority, status, status_token,
                    dependencies_json, acceptance_criteria_json, evidence_refs_json
             FROM todo_items
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ActionCompatRow {
                id: row.get(0)?,
                area: row.get(1)?,
                title: row.get(2)?,
                description: row.get(3)?,
                priority: row.get(4)?,
                status: row.get(5)?,
                status_token: row.get(6)?,
                dependencies_json: row.get(7)?,
                acceptance_criteria_json: row.get(8)?,
                evidence_refs_json: row.get(9)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn planning_next_action_rows(&self) -> Result<Vec<ActionCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, area, title, description, priority, status, status_token,
                    dependencies_json, acceptance_criteria_json, evidence_refs_json
             FROM next_action_items
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ActionCompatRow {
                id: row.get(0)?,
                area: row.get(1)?,
                title: row.get(2)?,
                description: row.get(3)?,
                priority: row.get(4)?,
                status: row.get(5)?,
                status_token: row.get(6)?,
                dependencies_json: row.get(7)?,
                acceptance_criteria_json: row.get(8)?,
                evidence_refs_json: row.get(9)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn requirements_meta_row(&self) -> Result<Option<RequirementsMetaCompatRow>> {
        self.conn
            .query_row(
                "SELECT authoritative, status, status_token, updated, python_recommended,
                        python_allowed, primary_markdown, status_allowlist_json,
                        runtime_stack_allowlist_json, required_module_fields_json,
                        required_gap_fields_json
                 FROM requirements_registry_meta
                 WHERE kind = 'requirements'",
                [],
                |row| {
                    Ok(RequirementsMetaCompatRow {
                        authoritative: row.get::<_, i64>(0)? != 0,
                        status: row.get(1)?,
                        status_token: row.get(2)?,
                        updated: row.get(3)?,
                        python_recommended: row.get(4)?,
                        python_allowed: row.get(5)?,
                        primary_markdown: row.get(6)?,
                        status_allowlist_json: row.get(7)?,
                        runtime_stack_allowlist_json: row.get(8)?,
                        required_module_fields_json: row.get(9)?,
                        required_gap_fields_json: row.get(10)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn requirements_module_rows(&self) -> Result<Vec<RequirementModuleCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, markdown, status, status_token, runtime_stack,
                    requires_modules_json, install_targets_json, verify_targets_json,
                    acceptance_criteria_json
             FROM requirements_modules
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(RequirementModuleCompatRow {
                id: row.get(0)?,
                name: row.get(1)?,
                markdown: row.get(2)?,
                status: row.get(3)?,
                status_token: row.get(4)?,
                runtime_stack: row.get(5)?,
                requires_modules_json: row.get(6)?,
                install_targets_json: row.get(7)?,
                verify_targets_json: row.get(8)?,
                acceptance_criteria_json: row.get(9)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn requirements_coverage_gap_rows(&self) -> Result<Vec<RequirementCoverageGapCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, area, status, status_token, description, proposed_resolution,
                    related_module_ids_json
             FROM requirements_coverage_gaps
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(RequirementCoverageGapCompatRow {
                id: row.get(0)?,
                area: row.get(1)?,
                status: row.get(2)?,
                status_token: row.get(3)?,
                description: row.get(4)?,
                proposed_resolution: row.get(5)?,
                related_module_ids_json: row.get(6)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}
