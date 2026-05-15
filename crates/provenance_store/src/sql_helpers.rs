//! Small SQL helpers used across the ProvenanceStore impl block.
//!
//! Covers four concerns:
//! - Fingerprint and registry-snapshot writers
//!   (`write_fingerprint`, `write_registry_snapshot`) that compute the
//!   blake3 digest and UPSERT it alongside the source path.
//! - Cardinality and grouped-count read helpers (`scalar_count`,
//!   `query_count_summaries`, `query_backend_health`).
//! - String-vector loaders (`load_string_vec`, `load_record_sources`).
//! - Path normalization (`to_repo_rel`) used by every fingerprint and
//!   snapshot writer to record paths relative to the repo root.

use std::{fs, path::Path};

use anyhow::{Context, Result};
use blake3::Hasher;
use provenance_core::{BackendHealthSummary, CountSummary};
use rusqlite::{Connection, params};

pub(crate) fn write_fingerprint(
    conn: &Connection,
    repo_root: &Path,
    path: &Path,
    indexed_at: &str,
) -> Result<()> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Hasher::new();
    hasher.update(&raw);
    let rel = to_repo_rel(repo_root, path);
    conn.execute(
        "INSERT INTO ingest_fingerprints (path, blake3_hex, size_bytes, indexed_at)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(path) DO UPDATE SET
            blake3_hex=excluded.blake3_hex,
            size_bytes=excluded.size_bytes,
            indexed_at=excluded.indexed_at",
        params![
            rel,
            hasher.finalize().to_hex().to_string(),
            raw.len() as i64,
            indexed_at
        ],
    )?;
    Ok(())
}

pub(crate) fn write_registry_snapshot(
    conn: &Connection,
    repo_root: &Path,
    kind: &str,
    path: &Path,
    body: &str,
    indexed_at: &str,
) -> Result<()> {
    let mut hasher = Hasher::new();
    hasher.update(body.as_bytes());
    conn.execute(
        "INSERT INTO registry_snapshots(registry_kind, source_path, content_text, blake3_hex, indexed_at)
         VALUES(?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(registry_kind) DO UPDATE SET
            source_path=excluded.source_path,
            content_text=excluded.content_text,
            blake3_hex=excluded.blake3_hex,
            indexed_at=excluded.indexed_at",
        params![
            kind,
            to_repo_rel(repo_root, path),
            body,
            hasher.finalize().to_hex().to_string(),
            indexed_at
        ],
    )?;
    Ok(())
}

pub(crate) fn scalar_count(conn: &Connection, sql: &str) -> Result<usize> {
    Ok(conn.query_row(sql, [], |row| row.get::<_, i64>(0))? as usize)
}

pub(crate) fn query_count_summaries(conn: &Connection, sql: &str) -> Result<Vec<CountSummary>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(CountSummary {
            key: row.get(0)?,
            count: row.get::<_, i64>(1)? as usize,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

pub(crate) fn query_backend_health(conn: &Connection) -> Result<Vec<BackendHealthSummary>> {
    let mut stmt = conn.prepare(
        "SELECT backend,
                SUM(CASE WHEN succeeded != 0 THEN 1 ELSE 0 END) AS success_count,
                SUM(CASE WHEN succeeded = 0 THEN 1 ELSE 0 END) AS failure_count,
                SUM(bytes) AS total_bytes
         FROM download_attempts
         GROUP BY backend
         ORDER BY (success_count + failure_count) DESC, backend ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(BackendHealthSummary {
            backend: row.get(0)?,
            success_count: row.get::<_, i64>(1)? as usize,
            failure_count: row.get::<_, i64>(2)? as usize,
            total_bytes: row.get::<_, i64>(3)?.max(0) as u64,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

pub(crate) fn load_string_vec(conn: &Connection, sql: &str, id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params![id], |row| row.get::<_, String>(0))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

pub(crate) fn load_record_sources(
    conn: &Connection,
    entity_kind: &str,
    entity_id: &str,
) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT source_ref FROM record_sources
         WHERE entity_kind = ?1 AND entity_id = ?2
         ORDER BY source_ref",
    )?;
    let rows = stmt.query_map(params![entity_kind, entity_id], |row| {
        row.get::<_, String>(0)
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

pub(crate) fn to_repo_rel(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
