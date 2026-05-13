//! Mutator methods for the requirements registry: meta-row +
//! per-module + per-coverage-gap upsert/delete operations.
//!
//! Methods added to `ProvenanceStore` via a second impl block:
//!   * `upsert_requirements_meta`
//!   * `upsert_requirement_module`
//!   * `delete_requirement_module`
//!   * `upsert_requirement_coverage_gap`
//!   * `delete_requirement_coverage_gap`
//!
//! All are pub on ProvenanceStore and access self.conn directly.

use anyhow::Result;
use rusqlite::params;

use crate::ProvenanceStore;
use crate::types::{RequirementCoverageGapItem, RequirementModuleItem, RequirementsMeta};

impl ProvenanceStore {
    pub fn upsert_requirements_meta(&self, meta: &RequirementsMeta<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT INTO requirements_registry_meta
             (kind, authoritative, status, status_token, updated, python_recommended,
              python_allowed, primary_markdown, status_allowlist_json,
              runtime_stack_allowlist_json, required_module_fields_json,
              required_gap_fields_json, updated_at)
             VALUES ('requirements', ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
             ON CONFLICT(kind) DO UPDATE SET
                 authoritative               = excluded.authoritative,
                 status                      = excluded.status,
                 status_token                = excluded.status_token,
                 updated                     = excluded.updated,
                 python_recommended          = excluded.python_recommended,
                 python_allowed              = excluded.python_allowed,
                 primary_markdown            = excluded.primary_markdown,
                 status_allowlist_json       = excluded.status_allowlist_json,
                 runtime_stack_allowlist_json = excluded.runtime_stack_allowlist_json,
                 required_module_fields_json = excluded.required_module_fields_json,
                 required_gap_fields_json    = excluded.required_gap_fields_json,
                 updated_at                  = excluded.updated_at",
            params![
                if meta.authoritative { 1 } else { 0 },
                meta.status,
                meta.status_token,
                meta.updated,
                meta.python_recommended,
                meta.python_allowed,
                meta.primary_markdown,
                meta.status_allowlist_json,
                meta.runtime_stack_allowlist_json,
                meta.required_module_fields_json,
                meta.required_gap_fields_json,
            ],
        )?;
        Ok(())
    }

    pub fn upsert_requirement_module(&self, item: &RequirementModuleItem<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO requirements_modules
             (id, name, markdown, status, status_token, runtime_stack,
              requires_modules_json, install_targets_json, verify_targets_json,
              acceptance_criteria_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10, datetime('now'))",
            params![
                item.id,
                item.name,
                item.markdown,
                item.status,
                item.status_token,
                item.runtime_stack,
                item.requires_modules_json,
                item.install_targets_json,
                item.verify_targets_json,
                item.acceptance_criteria_json,
            ],
        )?;
        Ok(())
    }

    pub fn delete_requirement_module(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM requirements_modules WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    pub fn upsert_requirement_coverage_gap(
        &self,
        item: &RequirementCoverageGapItem<'_>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO requirements_coverage_gaps
             (id, area, status, status_token, description, proposed_resolution,
              related_module_ids_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7, datetime('now'))",
            params![
                item.id,
                item.area,
                item.status,
                item.status_token,
                item.description,
                item.proposed_resolution,
                item.related_module_ids_json,
            ],
        )?;
        Ok(())
    }

    pub fn delete_requirement_coverage_gap(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM requirements_coverage_gaps WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }
}
