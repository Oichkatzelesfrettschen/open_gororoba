//! SQL table-clearing and ranked-value upsert/load helpers shared
//! across the ProvenanceStore mutators.
//!
//! `clear_tables` / `clear_control_plane_tables` / `clear_external_source_tables`
//! drop every row from the three logical table groups before a fresh
//! bootstrap from the registry sources. `insert_ranked_values`,
//! `replace_ranked_values`, and `load_ranked_values` implement the
//! shared (owner_id, relation, ord, value) projection used by the
//! claim/insight/experiment relation tables.

use anyhow::Result;
use rusqlite::{Connection, params};

pub(crate) fn clear_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        INSERT INTO document_search(document_search) VALUES('delete-all');
        DELETE FROM record_sources;
        DELETE FROM citations;
        DELETE FROM mirror_observations;
        DELETE FROM artifact_links;
        DELETE FROM artifact_paths;
        DELETE FROM lane_assignments;
        DELETE FROM links;
        DELETE FROM artifacts;
        DELETE FROM documents;
        DELETE FROM ingest_fingerprints;
        ",
    )?;
    Ok(())
}

pub(crate) fn clear_control_plane_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        DELETE FROM registry_snapshots;
        DELETE FROM claims;
        DELETE FROM insights;
        DELETE FROM experiments_cp;
        DELETE FROM binaries_cp;
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
