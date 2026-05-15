//! One-shot ingestion seeder for the Pantheon/PhysicsForge migration.
//!
//! `seed_pantheon_physicsforge_migration` is a single large method
//! that consumes two hand-curated TOML files (findings + overflow)
//! and bulk-inserts their contents into the canonical control-plane
//! tables in a single atomic transaction, returning a
//! `PantheonSeedSummary` of counts.
//!
//! Lives in a sibling submodule (second impl ProvenanceStore block)
//! because (a) it's a 256-line one-shot rarely-used method and
//! (b) keeping it out of the main impl block makes the rest of the
//! ProvenanceStore API easier to navigate.
//!
//! Accesses parent's private self.conn directly (Rust child modules
//! see private fields of parent's types).

use std::path::Path;

use anyhow::{Context, Result, bail};
use provenance_core::PantheonSeedSummary;
use rusqlite::params;
use toml::Value;

use crate::{
    ProvenanceStore,
    sql_helpers::scalar_count,
    toml_helpers::{
        join_refs, load_toml_value, optional_integer_field, string_array_field, string_field,
    },
};

impl ProvenanceStore {
    pub fn seed_pantheon_physicsforge_migration(
        &mut self,
        findings_path: &Path,
        overflow_path: &Path,
    ) -> Result<PantheonSeedSummary> {
        let findings_raw = load_toml_value(findings_path)?;
        let overflow_raw = load_toml_value(overflow_path)?;

        let findings_meta = findings_raw
            .get("migration_findings")
            .and_then(Value::as_table)
            .context("migration_findings table missing")?;
        let findings = findings_raw
            .get("finding")
            .and_then(Value::as_array)
            .context("finding table missing")?;
        let risks = findings_raw
            .get("risk")
            .and_then(Value::as_array)
            .context("risk table missing")?;

        let overflow_meta = overflow_raw
            .get("overflow_tracker")
            .and_then(Value::as_table)
            .context("overflow_tracker table missing")?;
        let overflow_rows = overflow_raw
            .get("overflow_task")
            .and_then(Value::as_array)
            .context("overflow_task table missing")?;

        let max_active_overflow = overflow_meta
            .get("max_active_tasks")
            .and_then(Value::as_integer)
            .unwrap_or(5)
            .max(0) as usize;
        let active_statuses = string_array_field(overflow_meta, "active_statuses");
        let active_count = overflow_rows
            .iter()
            .filter(|row| {
                row.as_table()
                    .and_then(|table| table.get("status"))
                    .and_then(Value::as_str)
                    .map(|status| {
                        active_statuses
                            .iter()
                            .any(|allowed| allowed == status.trim())
                    })
                    .unwrap_or(false)
            })
            .count();
        if active_count > max_active_overflow {
            bail!(
                "overflow tracker violates max active policy before sqlite seed: active={} max_active={}",
                active_count,
                max_active_overflow
            );
        }

        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS migration_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS migration_findings (
                finding_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                owner TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS unresolved_risks (
                risk_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                mitigation TEXT NOT NULL,
                owner TEXT NOT NULL,
                eta TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS overflow_tasks (
                overflow_id TEXT PRIMARY KEY,
                source_task_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                status TEXT NOT NULL,
                owner TEXT NOT NULL,
                eta TEXT NOT NULL,
                rationale TEXT NOT NULL,
                deferral_rationale TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                updated TEXT NOT NULL
            );
            ",
        )?;

        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![
                "source_findings_toml",
                findings_path.to_string_lossy().to_string()
            ],
        )?;
        tx.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![
                "source_overflow_toml",
                overflow_path.to_string_lossy().to_string()
            ],
        )?;
        tx.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params!["max_active_overflow", max_active_overflow.to_string()],
        )?;

        let findings_updated = findings_meta
            .get("updated")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        for row in findings {
            let table = row.as_table().context("finding row must be a table")?;
            tx.execute(
                "
                INSERT INTO migration_findings(
                    finding_id, task_id, phase, severity, status, summary, owner, evidence_refs, updated
                ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                ON CONFLICT(finding_id) DO UPDATE SET
                    task_id=excluded.task_id,
                    phase=excluded.phase,
                    severity=excluded.severity,
                    status=excluded.status,
                    summary=excluded.summary,
                    owner=excluded.owner,
                    evidence_refs=excluded.evidence_refs,
                    updated=excluded.updated
                ",
                params![
                    string_field(table, "id"),
                    string_field(table, "task_id"),
                    optional_integer_field(table, "phase").unwrap_or(0),
                    string_field(table, "severity"),
                    string_field(table, "status"),
                    string_field(table, "summary"),
                    string_field(table, "owner"),
                    join_refs(&string_array_field(table, "evidence_refs")),
                    findings_updated,
                ],
            )?;
        }

        for row in risks {
            let table = row.as_table().context("risk row must be a table")?;
            tx.execute(
                "
                INSERT INTO unresolved_risks(
                    risk_id, task_id, phase, risk_level, status, summary, mitigation, owner, eta, evidence_refs, updated
                ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                ON CONFLICT(risk_id) DO UPDATE SET
                    task_id=excluded.task_id,
                    phase=excluded.phase,
                    risk_level=excluded.risk_level,
                    status=excluded.status,
                    summary=excluded.summary,
                    mitigation=excluded.mitigation,
                    owner=excluded.owner,
                    eta=excluded.eta,
                    evidence_refs=excluded.evidence_refs,
                    updated=excluded.updated
                ",
                params![
                    string_field(table, "id"),
                    string_field(table, "task_id"),
                    optional_integer_field(table, "phase").unwrap_or(0),
                    string_field(table, "risk_level"),
                    string_field(table, "status"),
                    string_field(table, "summary"),
                    string_field(table, "mitigation"),
                    string_field(table, "owner"),
                    string_field(table, "eta"),
                    join_refs(&string_array_field(table, "evidence_refs")),
                    findings_updated,
                ],
            )?;
        }

        let overflow_updated = overflow_meta
            .get("updated")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        for row in overflow_rows {
            let table = row.as_table().context("overflow row must be a table")?;
            tx.execute(
                "
                INSERT INTO overflow_tasks(
                    overflow_id, source_task_id, phase, status, owner, eta, rationale, deferral_rationale, evidence_refs, updated
                ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                ON CONFLICT(overflow_id) DO UPDATE SET
                    source_task_id=excluded.source_task_id,
                    phase=excluded.phase,
                    status=excluded.status,
                    owner=excluded.owner,
                    eta=excluded.eta,
                    rationale=excluded.rationale,
                    deferral_rationale=excluded.deferral_rationale,
                    evidence_refs=excluded.evidence_refs,
                    updated=excluded.updated
                ",
                params![
                    string_field(table, "id"),
                    string_field(table, "source_task_id"),
                    optional_integer_field(table, "phase").unwrap_or(0),
                    string_field(table, "status"),
                    string_field(table, "owner"),
                    string_field(table, "eta"),
                    string_field(table, "rationale"),
                    string_field(table, "deferral_rationale"),
                    join_refs(&string_array_field(table, "evidence_refs")),
                    overflow_updated,
                ],
            )?;
        }
        tx.commit()?;

        let findings_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM migration_findings")?;
        let risk_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM unresolved_risks")?;
        let overflow_task_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM overflow_tasks")?;
        let db_path = self
            .conn
            .pragma_query_value(None, "database_list", |row| row.get::<_, String>(2))
            .unwrap_or_else(|_| String::new());
        Ok(PantheonSeedSummary {
            db_path,
            findings_count,
            risk_count,
            overflow_task_count,
            max_active_overflow,
        })
    }
}
