//! Download-attempt observation pipeline.
//!
//! Aggregates link observations from two sources:
//!   1. Intake-tree TSV tables (`fetch_results*_normalized.tsv`,
//!      `mirror_retry_results*`, `link_audit_results*`)
//!   2. The canonical SQLite control plane's `download_attempts`
//!      join with `download_jobs`
//!
//! Functions:
//!   * `collect_link_observations` -- gather both sources into a
//!     URL -> Vec<LinkObservation> map plus the list of source tables
//!   * `merge_sqlite_download_observations` -- internal SQLite step
//!   * `derive_attempt_status` -- normalize SQLite attempt fields
//!     into the same status vocabulary used by the TSV path
//!   * `collect_download_map` -- URL -> on-disk PDF path map
//!     assembled from intake `pdf_success_added.tsv` files plus the
//!     hand-maintained `cayley_dickson_canonical_sources.toml` and
//!     `cayley_dickson_source_recovery_2026_02_15.toml` registries
//!   * `provenance_intake_roots` -- the two well-known intake roots
//!   * `extend_download_map_from_local_artifacts` -- promote local
//!     artifact paths into the URL->path map
//!   * `register_download_aliases` -- insert under all arxiv-pair
//!     aliases (or the bare URL if none)
//!
//! Public type:
//!   * `LinkMap` = `(HashMap<String, Vec<LinkObservation>>, Vec<String>)`
//!
//! All items `pub(super)`. Depends on parent's `LinkObservation`,
//! `UnifiedArtifact`, `dedupe`, plus submodule fns
//! (`file_io::{read_tsv_rows, derive_status, load_toml_value}`,
//! `url_helpers::normalize_url`, `text_helpers::url_re`,
//! `identity_aliases::arxiv_equivalent_urls`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::Connection;
use toml::Value;
use walkdir::WalkDir;

use super::file_io::{derive_status, load_toml_value, read_tsv_rows};
use super::identity_aliases::arxiv_equivalent_urls;
use super::text_helpers::url_re;
use super::url_helpers::normalize_url;
use super::{LinkObservation, UnifiedArtifact, dedupe};

pub(super) type LinkMap = (HashMap<String, Vec<LinkObservation>>, Vec<String>);

pub(super) fn collect_link_observations(repo_root: &Path) -> Result<LinkMap> {
    let mut table_paths = Vec::new();
    for intake_root in provenance_intake_roots(repo_root) {
        if intake_root.exists() {
            for entry in WalkDir::new(&intake_root)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                if (name.starts_with("fetch_results") && name.ends_with("_normalized.tsv"))
                    || name.starts_with("mirror_retry_results")
                    || name.starts_with("link_audit_results")
                {
                    table_paths.push(path.to_path_buf());
                }
            }
        }
    }
    table_paths.sort();
    table_paths.dedup();

    let mut observations: HashMap<String, Vec<LinkObservation>> = HashMap::new();
    let mut source_tables = Vec::new();
    for path in table_paths {
        let rel = path
            .strip_prefix(repo_root)
            .unwrap_or(path.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        source_tables.push(rel.clone());
        for row in read_tsv_rows(&path)? {
            let url = normalize_url(row.get("url").map(String::as_str).unwrap_or_default());
            if !url_re().is_match(&url) {
                continue;
            }
            let status = derive_status(&row);
            observations
                .entry(url.clone())
                .or_default()
                .push(LinkObservation {
                    status: status.clone(),
                });

            let effective = normalize_url(
                row.get("url_effective")
                    .map(String::as_str)
                    .unwrap_or_default(),
            );
            if url_re().is_match(&effective) && effective != url {
                observations
                    .entry(effective)
                    .or_default()
                    .push(LinkObservation {
                        status: status.clone(),
                    });
            }
        }
    }
    let sqlite_path = repo_root.join("registry/canonical/control_plane.sqlite3");
    if sqlite_path.exists() {
        source_tables
            .push("registry/canonical/control_plane.sqlite3::download_attempts".to_string());
        merge_sqlite_download_observations(&sqlite_path, &mut observations)?;
    }
    Ok((observations, source_tables))
}

fn merge_sqlite_download_observations(
    sqlite_path: &Path,
    observations: &mut HashMap<String, Vec<LinkObservation>>,
) -> Result<()> {
    let conn = Connection::open(sqlite_path)
        .with_context(|| format!("open sqlite {}", sqlite_path.display()))?;
    let mut stmt = conn.prepare(
        "SELECT j.requested_url, j.final_url, a.http_code, a.is_pdf, a.succeeded, a.failure_class
         FROM download_attempts a
         JOIN download_jobs j ON j.id = a.job_id",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<i64>>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, Option<String>>(5)?,
        ))
    })?;
    for row in rows {
        let (requested_url, final_url, http_code, is_pdf, succeeded, failure_class) = row?;
        let status = derive_attempt_status(http_code, is_pdf != 0, succeeded != 0, failure_class);
        let requested_url = normalize_url(&requested_url);
        if url_re().is_match(&requested_url) {
            observations
                .entry(requested_url)
                .or_default()
                .push(LinkObservation {
                    status: status.clone(),
                });
        }
        if let Some(final_url) = final_url {
            let final_url = normalize_url(&final_url);
            if url_re().is_match(&final_url) {
                observations
                    .entry(final_url)
                    .or_default()
                    .push(LinkObservation {
                        status: status.clone(),
                    });
            }
        }
    }
    Ok(())
}

fn derive_attempt_status(
    http_code: Option<i64>,
    is_pdf: bool,
    succeeded: bool,
    failure_class: Option<String>,
) -> String {
    if let Some(code) = http_code {
        let code_str = code.to_string();
        if code_str.starts_with('2') && is_pdf {
            return "pdf_ok".to_string();
        }
        if code_str.starts_with('2') {
            return "ok_nonpdf".to_string();
        }
        return format!("http_{code}");
    }
    if succeeded {
        return if is_pdf {
            "pdf_ok".to_string()
        } else {
            "ok_nonpdf".to_string()
        };
    }
    if let Some(failure_class) = failure_class
        && !failure_class.trim().is_empty()
    {
        return "failed".to_string();
    }
    "unknown".to_string()
}

pub(super) fn collect_download_map(repo_root: &Path) -> Result<HashMap<String, Vec<String>>> {
    let mut url_to_paths: HashMap<String, Vec<String>> = HashMap::new();
    for intake_root in provenance_intake_roots(repo_root) {
        if intake_root.exists() {
            for entry in WalkDir::new(&intake_root)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if !path.is_file()
                    || path.file_name().and_then(|n| n.to_str()) != Some("pdf_success_added.tsv")
                {
                    continue;
                }
                let pdf_dir = path.parent().unwrap_or(path).join("pdf_success");
                for row in read_tsv_rows(path)? {
                    let source_url = normalize_url(
                        row.get("source_url")
                            .map(String::as_str)
                            .unwrap_or_default(),
                    );
                    let name = row.get("canonical_pdf_name").cloned().unwrap_or_default();
                    if !url_re().is_match(&source_url) || name.is_empty() {
                        continue;
                    }
                    let candidate = pdf_dir.join(&name);
                    if candidate.exists() {
                        let rel = candidate
                            .strip_prefix(repo_root)
                            .unwrap_or(candidate.as_path())
                            .to_string_lossy()
                            .replace('\\', "/");
                        register_download_aliases(&mut url_to_paths, &source_url, &rel);
                    }
                }
            }
        }
    }

    let cdcs_path = repo_root.join("registry/cayley_dickson_canonical_sources.toml");
    if cdcs_path.exists() {
        let data = load_toml_value(&cdcs_path)?;
        if let Some(papers) = data.get("paper").and_then(Value::as_array) {
            for paper in papers {
                let Some(table) = paper.as_table() else {
                    continue;
                };
                let path = table
                    .get("canonical_pdf_path")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                let url = normalize_url(
                    table
                        .get("canonical_functional_url")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
                if path.is_empty() {
                    continue;
                }
                let candidate = repo_root.join(path);
                if !candidate.exists() {
                    continue;
                }
                let rel = candidate
                    .strip_prefix(repo_root)
                    .unwrap_or(candidate.as_path())
                    .to_string_lossy()
                    .replace('\\', "/");
                if url_re().is_match(&url) {
                    register_download_aliases(&mut url_to_paths, &url, &rel);
                }
                if let Some(mirrors) = table.get("working_pdf_mirrors").and_then(Value::as_array) {
                    for mirror in mirrors.iter().filter_map(Value::as_str) {
                        let mirror_url = normalize_url(mirror);
                        if url_re().is_match(&mirror_url) {
                            register_download_aliases(&mut url_to_paths, &mirror_url, &rel);
                        }
                    }
                }
            }
        }
    }

    let brown_report = repo_root.join("reports/cayley_dickson_source_recovery_2026_02_15.toml");
    if brown_report.exists() {
        let data = load_toml_value(&brown_report)?;
        if let Some(table) = data.get("brown_1972").and_then(Value::as_table) {
            let path = table
                .get("canonical_pdf_path")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let url = normalize_url(
                table
                    .get("core_download_url")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
            if !path.is_empty() && url_re().is_match(&url) {
                let candidate = repo_root.join(path);
                if candidate.exists() {
                    let rel = candidate
                        .strip_prefix(repo_root)
                        .unwrap_or(candidate.as_path())
                        .to_string_lossy()
                        .replace('\\', "/");
                    url_to_paths.entry(url).or_default().push(rel);
                }
            }
        }
    }

    for paths in url_to_paths.values_mut() {
        *paths = dedupe(std::mem::take(paths));
    }
    Ok(url_to_paths)
}

pub(super) fn provenance_intake_roots(repo_root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![
        repo_root.join("data/external/intake"),
        repo_root.join("data/papers/intake"),
    ];
    roots.sort();
    roots.dedup();
    roots
}

pub(super) fn extend_download_map_from_local_artifacts(
    download_map: &mut HashMap<String, Vec<String>>,
    artifacts: &[UnifiedArtifact],
) {
    for artifact in artifacts {
        if artifact.local_paths.is_empty() {
            continue;
        }
        for rel in &artifact.local_paths {
            for link in &artifact.links {
                register_download_aliases(download_map, link, rel);
            }
        }
    }
}

fn register_download_aliases(
    url_to_paths: &mut HashMap<String, Vec<String>>,
    source_url: &str,
    rel: &str,
) {
    let aliases = arxiv_equivalent_urls(source_url);
    if aliases.is_empty() {
        url_to_paths
            .entry(source_url.to_string())
            .or_default()
            .push(rel.to_string());
        return;
    }
    for alias in aliases {
        url_to_paths.entry(alias).or_default().push(rel.to_string());
    }
}
