// SPDX-License-Identifier: GPL-2.0-or-later
//
// Bootstrap-only guard and value-preservation snapshot for the legacy
// compatibility-TOML importer.

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, params};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Import policy handed to `ProvenanceStore::reindex_control_plane_from_registries`.
///
/// The importer reads `registry/*.toml` and writes the canonical SQLite. The
/// compatibility mirrors omit columns the canonical store owns -- `insights`
/// carries no `status_note` key at all -- so a run against a populated database
/// resets those columns to NULL. `ReimportOptions::bootstrap` is the safe
/// default: it refuses on a populated database. `ReimportOptions::destructive`
/// names the database file so the run can back it up and record what it
/// changed before it changes it.
#[derive(Clone, Copy, Debug)]
pub struct ReimportOptions<'a> {
    pub allow_destructive_reimport: bool,
    pub db_path: Option<&'a Path>,
}

impl<'a> ReimportOptions<'a> {
    /// Bootstrap import. Refuses when the control-plane tables hold rows.
    #[must_use]
    pub const fn bootstrap() -> Self {
        Self {
            allow_destructive_reimport: false,
            db_path: None,
        }
    }

    /// Acknowledged destructive re-import of `db_path`, which is backed up and
    /// diffed before the transaction opens.
    #[must_use]
    pub const fn destructive(db_path: &'a Path) -> Self {
        Self {
            allow_destructive_reimport: true,
            db_path: Some(db_path),
        }
    }
}

/// Row counts of the four tables whose canonical content the importer overwrites.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControlPlanePopulation {
    pub claims: i64,
    pub insights: i64,
    pub experiments: i64,
    pub transition_events: i64,
}

impl ControlPlanePopulation {
    #[must_use]
    pub const fn is_populated(&self) -> bool {
        self.claims > 0 || self.insights > 0 || self.experiments > 0 || self.transition_events > 0
    }
}

pub(crate) fn measure_population(conn: &Connection) -> Result<ControlPlanePopulation> {
    let count = |table: &str| -> Result<i64> {
        Ok(conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })?)
    };
    Ok(ControlPlanePopulation {
        claims: count("claims")?,
        insights: count("insights")?,
        experiments: count("experiments_cp")?,
        transition_events: count("claim_transition_events")?,
    })
}

/// Refusal text for a populated database, matching the shape of the
/// `--allow-transition-history-loss` guard on `gororoba-db build`.
#[must_use]
pub fn refusal_message(db_path: Option<&Path>, population: &ControlPlanePopulation) -> String {
    let target = db_path.map_or_else(
        || "the canonical control plane".to_string(),
        |path| path.display().to_string(),
    );
    format!(
        "refusing to re-import compatibility registries into {target}: it holds {} claims, {} \
         insights, {} experiments and {} claim transition events. index-control-plane is a \
         bootstrap importer: it reads registry/*.toml, which omit columns the canonical store \
         owns (insight status_note has no compatibility key at all), so a run here overwrites \
         canonical values with mirror values. Export instead of importing \
         (provenance export-control-plane), or pass --allow-destructive-reimport to back the \
         database up, record a semantic diff, and proceed.",
        population.claims, population.insights, population.experiments, population.transition_events
    )
}

/// Canonical-only column values captured before the importer clears its tables.
///
/// `clear_control_plane_tables` issues `DELETE FROM insights`, so an
/// `ON CONFLICT DO UPDATE ... COALESCE` on the insert cannot see the prior row.
/// The values are read out first and reapplied after the inserts.
#[derive(Debug, Default)]
pub(crate) struct PreservedValues {
    pub claim_formal_proof: BTreeMap<String, String>,
    pub claim_status_note: BTreeMap<String, String>,
    pub insight_status_note: BTreeMap<String, String>,
    pub insight_claim_refs_json: BTreeMap<String, String>,
    pub experiment_status_note: BTreeMap<String, String>,
    pub binary_description: BTreeMap<String, String>,
    pub binary_experiment_id: BTreeMap<String, String>,
}

fn collect(conn: &Connection, sql: &str) -> Result<BTreeMap<String, String>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut out = BTreeMap::new();
    for row in rows {
        let (key, value) = row?;
        if !value.is_empty() {
            out.insert(key, value);
        }
    }
    Ok(out)
}

pub(crate) fn capture_preserved_values(conn: &Connection) -> Result<PreservedValues> {
    Ok(PreservedValues {
        claim_formal_proof: collect(
            conn,
            "SELECT id, formal_proof FROM claims WHERE formal_proof IS NOT NULL",
        )?,
        claim_status_note: collect(
            conn,
            "SELECT id, status_note FROM claims WHERE status_note IS NOT NULL",
        )?,
        insight_status_note: collect(
            conn,
            "SELECT id, status_note FROM insights WHERE status_note IS NOT NULL",
        )?,
        insight_claim_refs_json: collect(
            conn,
            "SELECT id, claim_refs_json FROM insights WHERE claim_refs_json NOT IN ('', '[]')",
        )?,
        experiment_status_note: collect(
            conn,
            "SELECT id, status_note FROM experiments_cp WHERE status_note IS NOT NULL",
        )?,
        binary_description: collect(
            conn,
            "SELECT name, description FROM binaries_cp WHERE description IS NOT NULL",
        )?,
        binary_experiment_id: collect(
            conn,
            "SELECT name, experiment_id FROM binaries_cp WHERE experiment_id IS NOT NULL",
        )?,
    })
}

/// Reapply captured canonical values wherever the freshly imported row left the
/// column NULL (or, for `claim_refs_json`, an empty array). A mirror that
/// carries its own value wins; a mirror that omits the column loses to SQLite.
pub(crate) fn restore_preserved_values(
    conn: &Connection,
    preserved: &PreservedValues,
) -> Result<usize> {
    let mut restored = 0usize;
    let text_lanes: [(&str, &str, &str, &BTreeMap<String, String>); 6] = [
        (
            "claims",
            "id",
            "formal_proof",
            &preserved.claim_formal_proof,
        ),
        ("claims", "id", "status_note", &preserved.claim_status_note),
        (
            "insights",
            "id",
            "status_note",
            &preserved.insight_status_note,
        ),
        (
            "experiments_cp",
            "id",
            "status_note",
            &preserved.experiment_status_note,
        ),
        (
            "binaries_cp",
            "name",
            "description",
            &preserved.binary_description,
        ),
        (
            "binaries_cp",
            "name",
            "experiment_id",
            &preserved.binary_experiment_id,
        ),
    ];
    for (table, key, column, values) in text_lanes {
        for (id, value) in values {
            restored += conn.execute(
                &format!(
                    "UPDATE {table} SET {column} = ?2
                     WHERE {key} = ?1 AND ({column} IS NULL OR {column} = '')"
                ),
                params![id, value],
            )?;
        }
    }
    for (id, value) in &preserved.insight_claim_refs_json {
        restored += conn.execute(
            "UPDATE insights SET claim_refs_json = ?2
             WHERE id = ?1 AND claim_refs_json IN ('', '[]')",
            params![id, value],
        )?;
    }
    Ok(restored)
}

/// Every column the importer is responsible for, per table. A column present in
/// the live schema but absent here is an ambiguous mapping: the importer neither
/// writes it nor preserves it, so the run bails before touching a row.
const MAPPED_COLUMNS: [(&str, &[&str]); 4] = [
    (
        "claims",
        &[
            "id",
            "statement",
            "status",
            "where_stated",
            "last_verified",
            "formal_proof",
            "status_note",
            "compat_toml_text",
        ],
    ),
    (
        "insights",
        &[
            "id",
            "title",
            "status",
            "claim_refs_json",
            "status_note",
            "compat_toml_text",
        ],
    ),
    (
        "experiments_cp",
        &[
            "id",
            "title",
            "status",
            "binary_name",
            "claim_refs_json",
            "status_note",
            "compat_toml_text",
        ],
    ),
    (
        "binaries_cp",
        &[
            "name",
            "crate_name",
            "description",
            "experiment_id",
            "source",
        ],
    ),
];

/// Compare the live schema against `MAPPED_COLUMNS` and bail on any column the
/// importer would silently drop. Called before the transaction opens.
pub(crate) fn assert_column_mapping_total(conn: &Connection) -> Result<()> {
    for (table, mapped) in MAPPED_COLUMNS {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let live: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<_, _>>()?;
        let unmapped: Vec<&String> = live.iter().filter(|c| !mapped.contains(&c.as_str())).collect();
        if !unmapped.is_empty() {
            bail!(
                "refusing to import: table {table} has columns the compatibility importer neither \
                 writes nor preserves: {}. Add each to MAPPED_COLUMNS in \
                 provenance_store::reimport_guard and to the write or preserve path before \
                 running index-control-plane.",
                unmapped
                    .iter()
                    .map(|c| c.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    Ok(())
}

/// One table's row-level divergence between the canonical store and the mirrors.
#[derive(Debug, Default)]
pub struct TableDiff {
    pub table: String,
    pub sqlite_only_ids: Vec<String>,
    pub toml_only_ids: Vec<String>,
    pub nulled_field_ids: Vec<String>,
    /// Rows where the mirror carries its own value and it differs from the
    /// canonical one. The mirror wins, so these are the values the re-import
    /// overwrites.
    pub overwritten_field_ids: Vec<String>,
}

/// The full pre-import semantic diff, printed and written beside the backup.
#[derive(Debug, Default)]
pub struct ReimportDiff {
    pub generated_at: String,
    pub backup_path: String,
    pub tables: Vec<TableDiff>,
}

impl ReimportDiff {
    /// TOML rendering. One `[[table]]` array entry per control-plane table, each
    /// listing the identifiers on both sides of the divergence.
    #[must_use]
    pub fn to_toml(&self) -> String {
        let mut out = String::new();
        out.push_str("# AUTO-GENERATED: index-control-plane destructive re-import diff.\n");
        out.push_str("[meta]\n");
        out.push_str(&format!("generated_at = \"{}\"\n", self.generated_at));
        out.push_str(&format!("backup_path = \"{}\"\n", self.backup_path));
        for table in &self.tables {
            out.push_str("\n[[table]]\n");
            out.push_str(&format!("name = \"{}\"\n", table.table));
            for (key, ids) in [
                ("sqlite_only_ids", &table.sqlite_only_ids),
                ("toml_only_ids", &table.toml_only_ids),
                ("nulled_field_ids", &table.nulled_field_ids),
                ("overwritten_field_ids", &table.overwritten_field_ids),
            ] {
                out.push_str(&format!("{key}_count = {}\n", ids.len()));
                out.push_str(&format!(
                    "{key} = [{}]\n",
                    ids.iter()
                        .map(|id| format!("\"{id}\""))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        out
    }

    /// Human-readable summary printed to stdout before the import proceeds.
    #[must_use]
    pub fn to_summary(&self) -> String {
        let mut out = format!("Destructive re-import diff (backup {}):\n", self.backup_path);
        for table in &self.tables {
            out.push_str(&format!(
                "  {}: sqlite_only={} toml_only={} fields_nulled={} fields_overwritten={}\n",
                table.table,
                table.sqlite_only_ids.len(),
                table.toml_only_ids.len(),
                table.nulled_field_ids.len(),
                table.overwritten_field_ids.len()
            ));
        }
        out
    }
}

/// One diffable table: SQL selecting `(id, canonical-only field)` pairs, and the
/// incoming mirror ids with the mirror's value for that same field.
pub(crate) struct DiffLane<'a> {
    pub table: &'a str,
    pub select_sql: &'a str,
    pub incoming: Vec<(String, Option<String>)>,
}

pub(crate) fn build_diff(
    conn: &Connection,
    backup_path: &str,
    lanes: Vec<DiffLane<'_>>,
) -> Result<ReimportDiff> {
    let mut tables = Vec::new();
    for lane in lanes {
        let mut stmt = conn.prepare(lane.select_sql)?;
        let existing: BTreeMap<String, Option<String>> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            })?
            .collect::<std::result::Result<_, _>>()?;
        let incoming: BTreeMap<String, Option<String>> = lane.incoming.into_iter().collect();
        let sqlite_only_ids = existing
            .keys()
            .filter(|id| !incoming.contains_key(*id))
            .cloned()
            .collect();
        let toml_only_ids = incoming
            .keys()
            .filter(|id| !existing.contains_key(*id))
            .cloned()
            .collect();
        let nulled_field_ids = existing
            .iter()
            .filter(|(id, value)| {
                value.is_some()
                    && incoming
                        .get(*id)
                        .is_some_and(|incoming_value| incoming_value.is_none())
            })
            .map(|(id, _)| id.clone())
            .collect();
        let overwritten_field_ids = existing
            .iter()
            .filter(|(id, value)| {
                value.is_some()
                    && incoming.get(*id).is_some_and(|incoming_value| {
                        incoming_value.is_some() && incoming_value != *value
                    })
            })
            .map(|(id, _)| id.clone())
            .collect();
        tables.push(TableDiff {
            table: lane.table.to_string(),
            sqlite_only_ids,
            toml_only_ids,
            nulled_field_ids,
            overwritten_field_ids,
        });
    }
    Ok(ReimportDiff {
        generated_at: chrono::Utc::now().to_rfc3339(),
        backup_path: backup_path.to_string(),
        tables,
    })
}

/// Copy the canonical database to `registry/canonical/backups/` under a UTC
/// timestamp. `VACUUM INTO` writes one consistent file regardless of journal
/// mode, which a plain file copy does not guarantee under WAL.
pub(crate) fn backup_database(conn: &Connection, db_path: &Path) -> Result<PathBuf> {
    let dir = db_path
        .parent()
        .context("canonical database has no parent directory")?
        .join("backups");
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let stem = db_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("control_plane");
    let target = dir.join(format!("{stem}.{stamp}.sqlite3"));
    if target.exists() {
        fs::remove_file(&target).with_context(|| format!("replace {}", target.display()))?;
    }
    conn.execute("VACUUM INTO ?1", params![target.to_string_lossy()])
        .with_context(|| format!("VACUUM INTO {}", target.display()))?;
    Ok(target)
}

/// Path of the diff TOML written beside a backup file.
#[must_use]
pub fn diff_path_for_backup(backup: &Path) -> PathBuf {
    backup.with_extension("diff.toml")
}
