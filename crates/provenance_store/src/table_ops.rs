//! SQL table-clearing and ranked-value upsert/load helpers shared
//! across the ProvenanceStore mutators.
//!
//! `clear_tables` / `clear_control_plane_tables` / `clear_external_source_tables`
//! drop every row from the three logical table groups before a fresh
//! bootstrap from the registry sources. `insert_ranked_values`,
//! `replace_ranked_values`, and `load_ranked_values` implement the
//! shared (owner_id, relation, ord, value) projection used by the
//! claim/insight/experiment relation tables.

use anyhow::{Result, bail};
use rusqlite::{Connection, params};

pub(crate) fn refuse_artifact_path_history_loss(conn: &Connection) -> Result<()> {
    let table_exists = |table: &str| -> Result<bool> {
        Ok(conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
            [table],
            |row| row.get(0),
        )?)
    };
    let events: i64 = if table_exists("export_runs")? {
        conn.query_row(
            "SELECT count(*) FROM export_runs WHERE action='repair-artifact-paths'",
            [],
            |row| row.get(0),
        )?
    } else {
        0
    };
    let relations: i64 = if table_exists("artifact_paths")? {
        conn.query_row(
            "SELECT count(*) FROM artifact_paths WHERE relation IN ('referenced','historical_download','transformed_copy')",
            [],
            |row| row.get(0),
        )?
    } else {
        0
    };
    if events > 0 || relations > 0 {
        bail!(
            "refusing to discard artifact path repair history: {events} repair events and {relations} canonical-only path relations cannot be restored from the artifact compatibility inventory. Preserve the canonical database; bootstrap a separate empty database for an import."
        );
    }
    Ok(())
}

pub(crate) fn clear_tables(conn: &Connection) -> Result<()> {
    refuse_artifact_path_history_loss(conn)?;
    conn.execute_batch(
        "
        INSERT INTO document_search(document_search) VALUES('delete-all');
        DELETE FROM record_sources;
        DELETE FROM citations
         WHERE artifact_id NOT IN (
             SELECT artifact_id FROM claim_transition_evidence
             UNION
             SELECT artifact_id FROM claim_transition_successor_evidence
         );
        DELETE FROM mirror_observations
         WHERE artifact_id NOT IN (
             SELECT artifact_id FROM claim_transition_evidence
             UNION
             SELECT artifact_id FROM claim_transition_successor_evidence
         );
        DELETE FROM artifact_links
         WHERE artifact_id NOT IN (
             SELECT artifact_id FROM claim_transition_evidence
             UNION
             SELECT artifact_id FROM claim_transition_successor_evidence
         );
        DELETE FROM artifact_paths
         WHERE artifact_id NOT IN (
             SELECT artifact_id FROM claim_transition_evidence
             UNION
             SELECT artifact_id FROM claim_transition_successor_evidence
         );
        DELETE FROM lane_assignments
         WHERE artifact_id NOT IN (
             SELECT artifact_id FROM claim_transition_evidence
             UNION
             SELECT artifact_id FROM claim_transition_successor_evidence
         );
        DELETE FROM links
         WHERE url NOT IN (
             SELECT url FROM artifact_links
             UNION
             SELECT url FROM mirror_observations
         );
        DELETE FROM artifacts
         WHERE id NOT IN (
             SELECT artifact_id FROM claim_transition_evidence
             UNION
             SELECT artifact_id FROM claim_transition_successor_evidence
         );
        DELETE FROM documents;
        DELETE FROM ingest_fingerprints;
        ",
    )?;
    Ok(())
}

/// A control-plane reindex rewrites exactly these snapshot kinds. Every
/// other kind (roadmap, todo, next_actions, requirements, the external
/// source lanes) is recorded by its own command and survives the reindex,
/// so `render_roadmap_compat_toml` keeps its `supersedes` and
/// `companion_docs` arrays across `provenance index-control-plane`.
pub(crate) const CONTROL_PLANE_SNAPSHOT_KINDS: [&str; 5] =
    ["claims", "insights", "experiments", "binaries", "rocq_project"];

pub(crate) fn clear_control_plane_tables(conn: &Connection) -> Result<()> {
    let kinds = CONTROL_PLANE_SNAPSHOT_KINDS
        .iter()
        .map(|kind| format!("'{kind}'"))
        .collect::<Vec<_>>()
        .join(", ");
    conn.execute(
        &format!("DELETE FROM registry_snapshots WHERE registry_kind IN ({kinds})"),
        [],
    )?;
    conn.execute_batch(
        "
        DELETE FROM insights;
        DELETE FROM experiments_cp
         WHERE id NOT IN (
             SELECT experiment_id FROM claim_transition_experiments
         );
        DELETE FROM binaries_cp;
        DELETE FROM theorem_claim_links;
        DELETE FROM theorems;
        DELETE FROM control_plane_meta;
        ",
    )?;
    Ok(())
}

pub(crate) fn clear_external_source_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        DELETE FROM external_source_contract_values;
        DELETE FROM external_source_contracts;
        DELETE FROM external_source_contracts_meta;
        DELETE FROM external_source_dossier_values;
        DELETE FROM external_source_dossiers;
        DELETE FROM external_source_dossiers_meta;
        ",
    )?;
    Ok(())
}

pub(crate) fn insert_ranked_values(
    conn: &Connection,
    table: &str,
    owner_column: &str,
    owner_id: &str,
    relation: &str,
    values: &[String],
) -> Result<()> {
    let sql =
        format!("INSERT INTO {table}({owner_column}, relation, ord, value) VALUES(?1, ?2, ?3, ?4)");
    for (ord, value) in values.iter().enumerate() {
        conn.execute(&sql, params![owner_id, relation, ord as i64, value])?;
    }
    Ok(())
}

pub(crate) fn replace_ranked_values(
    conn: &Connection,
    table: &str,
    owner_column: &str,
    owner_id: &str,
    relation: &str,
    values: &[String],
) -> Result<()> {
    let delete_sql = format!("DELETE FROM {table} WHERE {owner_column} = ?1 AND relation = ?2");
    conn.execute(&delete_sql, params![owner_id, relation])?;
    insert_ranked_values(conn, table, owner_column, owner_id, relation, values)
}

pub(crate) fn load_ranked_values(
    conn: &Connection,
    table: &str,
    owner_column: &str,
    owner_id: &str,
    relation: &str,
) -> Result<Vec<String>> {
    let sql = format!(
        "SELECT value FROM {table} WHERE {owner_column} = ?1 AND relation = ?2 ORDER BY ord"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![owner_id, relation], |row| row.get::<_, String>(0))?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}
