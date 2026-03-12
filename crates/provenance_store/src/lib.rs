use anyhow::{Context, Result, bail};
use blake3::Hasher;
use camino::Utf8PathBuf;
use chrono::Utc;
use provenance_core::{
    ArtifactQueryResult, ArtifactRecord, ArtifactStatus, DoctorReport, DocumentQueryResult,
    DocumentRecord, IndexStats, LaneAssignment, MirrorKind, MirrorObservationRecord,
    PantheonSeedSummary,
};
use rusqlite::{Connection, OptionalExtension, params};
use rusqlite_migration::{M, Migrations};
use std::fs;
use std::path::Path;
use toml::Value;

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![M::up(
        "
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            kind TEXT NOT NULL,
            authoring_mode TEXT NOT NULL,
            generated INTEGER NOT NULL,
            status TEXT NOT NULL,
            toml_backing TEXT,
            sha256 TEXT,
            size_bytes INTEGER,
            line_count INTEGER
        );
        CREATE TABLE record_sources (
            entity_kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            PRIMARY KEY(entity_kind, entity_id, source_ref)
        );
        CREATE TABLE citations (
            id INTEGER PRIMARY KEY,
            artifact_id TEXT,
            citation_text TEXT NOT NULL,
            doi TEXT,
            canonical_url TEXT,
            FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
        );
        CREATE TABLE links (
            url TEXT PRIMARY KEY,
            host TEXT
        );
        CREATE TABLE artifacts (
            id TEXT PRIMARY KEY,
            key TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            citation TEXT NOT NULL,
            status TEXT NOT NULL,
            minimum_requirement_met INTEGER NOT NULL,
            canonical_functional_url TEXT,
            canonical_download_path TEXT
        );
        CREATE TABLE artifact_links (
            artifact_id TEXT NOT NULL,
            url TEXT NOT NULL,
            relation TEXT NOT NULL,
            PRIMARY KEY(artifact_id, url, relation),
            FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE,
            FOREIGN KEY(url) REFERENCES links(url) ON DELETE CASCADE
        );
        CREATE TABLE artifact_paths (
            artifact_id TEXT NOT NULL,
            path TEXT NOT NULL,
            relation TEXT NOT NULL,
            PRIMARY KEY(artifact_id, path, relation),
            FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
        );
        CREATE TABLE mirror_observations (
            artifact_id TEXT NOT NULL,
            url TEXT NOT NULL,
            mirror_kind TEXT NOT NULL,
            PRIMARY KEY(artifact_id, url, mirror_kind),
            FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE,
            FOREIGN KEY(url) REFERENCES links(url) ON DELETE CASCADE
        );
        CREATE TABLE lane_assignments (
            artifact_id TEXT NOT NULL,
            lane_name TEXT NOT NULL,
            PRIMARY KEY(artifact_id, lane_name),
            FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
        );
        CREATE TABLE export_runs (
            id INTEGER PRIMARY KEY,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL,
            artifact_count INTEGER NOT NULL,
            document_count INTEGER NOT NULL,
            details_json TEXT NOT NULL
        );
        CREATE TABLE ingest_fingerprints (
            path TEXT PRIMARY KEY,
            blake3_hex TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            indexed_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE document_search USING fts5(document_id, path, title, kind, content='');
        ",
    )])
}

pub struct ProvenanceStore {
    conn: Connection,
}

impl ProvenanceStore {
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create db directory {}", parent.display()))?;
        }
        let mut conn = Connection::open(db_path)
            .with_context(|| format!("open sqlite database {}", db_path.display()))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        migrations().to_latest(&mut conn)?;
        Ok(Self { conn })
    }

    pub fn reindex_from_registries(
        &mut self,
        repo_root: &Path,
        artifact_registry_path: &Path,
        knowledge_sources_path: &Path,
        lane_dir: &Path,
    ) -> Result<IndexStats> {
        let artifacts = load_artifacts(artifact_registry_path)?;
        let artifact_id_set = artifacts
            .iter()
            .map(|artifact| artifact.id.clone())
            .collect::<std::collections::HashSet<_>>();
        let documents = load_documents(knowledge_sources_path)?;
        let lanes = load_lane_assignments(lane_dir)?;
        let mirror_observations = build_mirror_observations(artifact_registry_path)?;
        let indexed_at = Utc::now().to_rfc3339();

        let tx = self.conn.transaction()?;
        clear_tables(&tx)?;

        for path in [artifact_registry_path, knowledge_sources_path] {
            write_fingerprint(&tx, repo_root, path, &indexed_at)?;
        }
        for lane in ["datasets.toml", "papers_pdf.toml", "slides_artifacts.toml", "web_references.toml"] {
            let path = lane_dir.join(lane);
            if path.exists() {
                write_fingerprint(&tx, repo_root, &path, &indexed_at)?;
            }
        }

        for document in &documents {
            tx.execute(
                "INSERT INTO documents (
                    id, path, title, kind, authoring_mode, generated, status,
                    toml_backing, sha256, size_bytes, line_count
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    document.id,
                    document.path.as_str(),
                    document.title,
                    document.kind,
                    document.authoring_mode,
                    i64::from(document.generated),
                    document.status,
                    document.toml_backing.as_ref().map(|v| v.as_str()),
                    document.sha256.as_deref(),
                    document.size_bytes,
                    document.line_count
                ],
            )?;
            if let Some(backing) = &document.toml_backing {
                tx.execute(
                    "INSERT INTO record_sources (entity_kind, entity_id, source_ref) VALUES ('document', ?1, ?2)",
                    params![document.id, backing.as_str()],
                )?;
            }
            tx.execute(
                "INSERT INTO document_search (document_id, path, title, kind) VALUES (?1, ?2, ?3, ?4)",
                params![document.id, document.path.as_str(), document.title, document.kind],
            )?;
        }

        for artifact in &artifacts {
            tx.execute(
                "INSERT INTO artifacts (
                    id, key, title, citation, status,
                    minimum_requirement_met, canonical_functional_url, canonical_download_path
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    artifact.id,
                    artifact.key,
                    artifact.title,
                    artifact.citation,
                    artifact.status.as_str(),
                    i64::from(artifact.minimum_requirement_met),
                    artifact.canonical_functional_url.as_deref(),
                    artifact.canonical_download_path.as_ref().map(|v| v.as_str())
                ],
            )?;
            for source_ref in &artifact.source_refs {
                tx.execute(
                    "INSERT INTO record_sources (entity_kind, entity_id, source_ref) VALUES ('artifact', ?1, ?2)
                     ON CONFLICT(entity_kind, entity_id, source_ref) DO NOTHING",
                    params![artifact.id, source_ref],
                )?;
            }
            for doi in &artifact.doi_list {
                tx.execute(
                    "INSERT INTO citations (artifact_id, citation_text, doi, canonical_url) VALUES (?1, ?2, ?3, ?4)",
                    params![
                        artifact.id,
                        artifact.citation,
                        doi,
                        artifact.canonical_functional_url.as_deref()
                    ],
                )?;
            }
            for url in &artifact.all_links {
                let host = host_for_url(url);
                tx.execute(
                    "INSERT INTO links (url, host) VALUES (?1, ?2)
                     ON CONFLICT(url) DO UPDATE SET host=excluded.host",
                    params![url, host],
                )?;
                tx.execute(
                    "INSERT INTO artifact_links (artifact_id, url, relation) VALUES (?1, ?2, 'all_links')
                     ON CONFLICT(artifact_id, url, relation) DO NOTHING",
                    params![artifact.id, url],
                )?;
            }
            for path in &artifact.downloaded_paths {
                tx.execute(
                    "INSERT INTO artifact_paths (artifact_id, path, relation) VALUES (?1, ?2, 'downloaded')
                     ON CONFLICT(artifact_id, path, relation) DO NOTHING",
                    params![artifact.id, path.as_str()],
                )?;
            }
        }

        for observation in &mirror_observations {
            tx.execute(
                "INSERT OR IGNORE INTO links (url, host) VALUES (?1, ?2)",
                params![observation.url, host_for_url(&observation.url)],
            )?;
            tx.execute(
                "INSERT INTO mirror_observations (artifact_id, url, mirror_kind) VALUES (?1, ?2, ?3)
                 ON CONFLICT(artifact_id, url, mirror_kind) DO NOTHING",
                params![
                    observation.artifact_id,
                    observation.url,
                    observation.mirror_kind.as_str()
                ],
            )?;
        }

        let mut inserted_lane_count = 0usize;
        for lane in &lanes {
            if !artifact_id_set.contains(&lane.artifact_id) {
                continue;
            }
            tx.execute(
                "INSERT INTO lane_assignments (artifact_id, lane_name) VALUES (?1, ?2)
                 ON CONFLICT(artifact_id, lane_name) DO NOTHING",
                params![lane.artifact_id, lane.lane_name],
            )?;
            inserted_lane_count += 1;
        }

        let details = serde_json::json!({
            "artifact_registry": to_repo_rel(repo_root, artifact_registry_path),
            "knowledge_sources": to_repo_rel(repo_root, knowledge_sources_path),
            "lane_dir": to_repo_rel(repo_root, lane_dir),
        });
        tx.execute(
            "INSERT INTO export_runs (action, created_at, artifact_count, document_count, details_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                "index",
                indexed_at,
                artifacts.len() as i64,
                documents.len() as i64,
                details.to_string()
            ],
        )?;

        tx.commit()?;

        Ok(IndexStats {
            indexed_at,
            artifact_count: artifacts.len(),
            document_count: documents.len(),
            lane_assignment_count: inserted_lane_count,
            mirror_observation_count: mirror_observations.len(),
        })
    }

    pub fn record_export_run(
        &mut self,
        action: &str,
        artifact_count: usize,
        document_count: usize,
        details_json: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO export_runs (action, created_at, artifact_count, document_count, details_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                action,
                Utc::now().to_rfc3339(),
                artifact_count as i64,
                document_count as i64,
                details_json
            ],
        )?;
        Ok(())
    }

    pub fn artifact_by_needle(&self, needle: &str) -> Result<Option<ArtifactQueryResult>> {
        let row = self
            .conn
            .query_row(
                "SELECT id, key, title, citation, status, minimum_requirement_met,
                        canonical_functional_url, canonical_download_path
                 FROM artifacts
                 WHERE id = ?1 OR key = ?1
                    OR lower(title) LIKE '%' || lower(?1) || '%'
                 ORDER BY CASE WHEN id = ?1 OR key = ?1 THEN 0 ELSE 1 END, id
                 LIMIT 1",
                params![needle],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, i64>(5)?,
                        row.get::<_, Option<String>>(6)?,
                        row.get::<_, Option<String>>(7)?,
                    ))
                },
            )
            .optional()?;

        let Some((id, key, title, citation, status_raw, minimum_requirement_met, canonical_functional_url, canonical_download_path)) = row else {
            return Ok(None);
        };
        let artifact = ArtifactRecord {
            id: id.clone(),
            key,
            title,
            citation,
            status: ArtifactStatus::parse(&status_raw)
                .with_context(|| format!("invalid artifact status {status_raw}"))?,
            minimum_requirement_met: minimum_requirement_met != 0,
            canonical_functional_url,
            canonical_download_path: canonical_download_path.map(Utf8PathBuf::from),
            source_refs: load_record_sources(&self.conn, "artifact", &id)?,
            all_links: load_string_vec(
                &self.conn,
                "SELECT url FROM artifact_links WHERE artifact_id = ?1 ORDER BY url",
                &id,
            )?,
            downloaded_paths: load_string_vec(
                &self.conn,
                "SELECT path FROM artifact_paths WHERE artifact_id = ?1 AND relation = 'downloaded' ORDER BY path",
                &id,
            )?
            .into_iter()
            .map(Utf8PathBuf::from)
            .collect(),
            doi_list: load_string_vec(
                &self.conn,
                "SELECT doi FROM citations WHERE artifact_id = ?1 AND doi IS NOT NULL ORDER BY doi",
                &id,
            )?,
            notes: Vec::new(),
        };

        let lanes = load_string_vec(
            &self.conn,
            "SELECT lane_name FROM lane_assignments WHERE artifact_id = ?1 ORDER BY lane_name",
            &artifact.id,
        )?;
        let mirror_observations = self.load_mirrors(&artifact.id)?;
        Ok(Some(ArtifactQueryResult {
            artifact,
            lanes,
            mirror_observations,
        }))
    }

    pub fn document_by_needle(&self, needle: &str) -> Result<Option<DocumentQueryResult>> {
        let document = self
            .conn
            .query_row(
                "SELECT id, path, title, kind, authoring_mode, generated, status,
                        toml_backing, sha256, size_bytes, line_count
                 FROM documents
                 WHERE id = ?1 OR path = ?1
                    OR lower(title) LIKE '%' || lower(?1) || '%'
                 ORDER BY CASE WHEN id = ?1 OR path = ?1 THEN 0 ELSE 1 END, id
                 LIMIT 1",
                params![needle],
                |row| {
                    Ok(DocumentRecord {
                        id: row.get(0)?,
                        path: Utf8PathBuf::from(row.get::<_, String>(1)?),
                        title: row.get(2)?,
                        kind: row.get(3)?,
                        authoring_mode: row.get(4)?,
                        generated: row.get::<_, i64>(5)? != 0,
                        status: row.get(6)?,
                        toml_backing: row.get::<_, Option<String>>(7)?.map(Utf8PathBuf::from),
                        sha256: row.get(8)?,
                        size_bytes: row.get(9)?,
                        line_count: row.get(10)?,
                    })
                },
            )
            .optional()?;

        let Some(document) = document else {
            return Ok(None);
        };
        let source_refs = load_record_sources(&self.conn, "document", &document.id)?;
        Ok(Some(DocumentQueryResult { document, source_refs }))
    }

    pub fn doctor_report(&self) -> Result<DoctorReport> {
        let artifact_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM artifacts")?;
        let document_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM documents")?;
        let missing_minimum_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM artifacts WHERE minimum_requirement_met = 0",
        )?;
        let blocked_count =
            scalar_count(&self.conn, "SELECT COUNT(*) FROM artifacts WHERE status = 'blocked'")?;
        let unverified_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM artifacts WHERE status = 'unverified'",
        )?;
        let citation_only_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM artifacts WHERE status = 'citation_only_no_link'",
        )?;
        let missing_lane_assignment_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM artifacts a
             WHERE NOT EXISTS (
                 SELECT 1 FROM lane_assignments l WHERE l.artifact_id = a.id
             )",
        )?;
        let documents_without_backing_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM documents WHERE toml_backing IS NULL OR toml_backing = ''",
        )?;
        let last_indexed_at = self
            .conn
            .query_row(
                "SELECT created_at FROM export_runs WHERE action = 'index' ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let last_exported_at = self
            .conn
            .query_row(
                "SELECT created_at FROM export_runs WHERE action = 'export' ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(DoctorReport {
            generated_at: Utc::now().to_rfc3339(),
            artifact_count,
            document_count,
            missing_minimum_count,
            blocked_count,
            unverified_count,
            citation_only_count,
            missing_lane_assignment_count,
            documents_without_backing_count,
            last_indexed_at,
            last_exported_at,
        })
    }

    pub fn verify_invariants(&self, repo_root: &Path) -> Result<()> {
        let mut failures = Vec::new();
        let invalid_status_count = self.conn.query_row(
            "SELECT COUNT(*) FROM artifacts
             WHERE status NOT IN ('downloaded','downloadable','blocked','citation_only_no_link','unverified')",
            [],
            |row| row.get::<_, i64>(0),
        )? as usize;
        if invalid_status_count != 0 {
            failures.push(format!("found {invalid_status_count} artifacts with invalid status"));
        }
        let missing_lane_assignment_count = self.conn.query_row(
            "SELECT COUNT(*) FROM artifacts a
             WHERE NOT EXISTS (SELECT 1 FROM lane_assignments l WHERE l.artifact_id = a.id)",
            [],
            |row| row.get::<_, i64>(0),
        )? as usize;
        if missing_lane_assignment_count != 0 {
            failures.push(format!(
                "{missing_lane_assignment_count} artifacts missing lane assignments"
            ));
        }
        let mut stmt = self
            .conn
            .prepare("SELECT canonical_download_path FROM artifacts WHERE canonical_download_path IS NOT NULL AND canonical_download_path != ''")?;
        let paths = stmt.query_map([], |row| row.get::<_, String>(0))?;
        for path in paths {
            let path = path?;
            if !repo_root.join(&path).exists() {
                failures.push(format!("missing canonical download path on disk: {path}"));
            }
        }
        if !failures.is_empty() {
            bail!("provenance store invariants failed:\n- {}", failures.join("\n- "));
        }
        Ok(())
    }

    pub fn recovery_candidates(&self, limit: usize) -> Result<Vec<ArtifactRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, key, title, citation, status, minimum_requirement_met,
                    canonical_functional_url, canonical_download_path
             FROM artifacts
             WHERE minimum_requirement_met = 0
             ORDER BY status, id
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, key, title, citation, status_raw, minimum_requirement_met, canonical_functional_url, canonical_download_path) = row?;
            out.push(ArtifactRecord {
                id,
                key,
                title,
                citation,
                status: ArtifactStatus::parse(&status_raw)
                    .with_context(|| format!("invalid artifact status {status_raw}"))?,
                minimum_requirement_met: minimum_requirement_met != 0,
                canonical_functional_url,
                canonical_download_path: canonical_download_path.map(Utf8PathBuf::from),
                source_refs: Vec::new(),
                all_links: Vec::new(),
                downloaded_paths: Vec::new(),
                doi_list: Vec::new(),
                notes: Vec::new(),
            });
        }
        Ok(out)
    }

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
                    .map(|status| active_statuses.iter().any(|allowed| allowed == status.trim()))
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
            params!["source_findings_toml", findings_path.to_string_lossy().to_string()],
        )?;
        tx.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params!["source_overflow_toml", overflow_path.to_string_lossy().to_string()],
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

    fn load_mirrors(&self, artifact_id: &str) -> Result<Vec<MirrorObservationRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT artifact_id, url, mirror_kind
             FROM mirror_observations
             WHERE artifact_id = ?1
             ORDER BY mirror_kind, url",
        )?;
        let rows = stmt.query_map(params![artifact_id], |row| {
            let mirror_kind = match row.get::<_, String>(2)?.as_str() {
                "working" => MirrorKind::Working,
                "working_pdf" => MirrorKind::WorkingPdf,
                "nonworking" => MirrorKind::Nonworking,
                _ => MirrorKind::Unverified,
            };
            Ok(MirrorObservationRecord {
                artifact_id: row.get(0)?,
                url: row.get(1)?,
                mirror_kind,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }
}

fn clear_tables(conn: &Connection) -> Result<()> {
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

fn write_fingerprint(conn: &Connection, repo_root: &Path, path: &Path, indexed_at: &str) -> Result<()> {
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
        params![rel, hasher.finalize().to_hex().to_string(), raw.len() as i64, indexed_at],
    )?;
    Ok(())
}

fn scalar_count(conn: &Connection, sql: &str) -> Result<usize> {
    Ok(conn.query_row(sql, [], |row| row.get::<_, i64>(0))? as usize)
}

fn load_string_vec(conn: &Connection, sql: &str, id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params![id], |row| row.get::<_, String>(0))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn load_record_sources(conn: &Connection, entity_kind: &str, entity_id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT source_ref FROM record_sources
         WHERE entity_kind = ?1 AND entity_id = ?2
         ORDER BY source_ref",
    )?;
    let rows = stmt.query_map(params![entity_kind, entity_id], |row| row.get::<_, String>(0))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn to_repo_rel(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn load_artifacts(path: &Path) -> Result<Vec<ArtifactRecord>> {
    let value = load_toml_value(path)?;
    let artifacts = value
        .get("artifact")
        .and_then(Value::as_array)
        .context("artifact table missing")?;
    let mut out = Vec::new();
    for artifact in artifacts {
        let table = artifact.as_table().context("artifact row must be a table")?;
        let status_raw = table
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unverified");
        let status = ArtifactStatus::parse(status_raw)
            .with_context(|| format!("invalid artifact status {status_raw}"))?;
        out.push(ArtifactRecord {
            id: string_field(table, "id"),
            key: string_field(table, "key"),
            title: string_field(table, "title"),
            citation: string_field(table, "citation"),
            status,
            minimum_requirement_met: bool_field(table, "minimum_requirement_met"),
            canonical_functional_url: optional_string_field(table, "canonical_functional_url"),
            canonical_download_path: optional_string_field(table, "canonical_download_path")
                .map(Utf8PathBuf::from),
            source_refs: string_array_field(table, "source_refs"),
            all_links: string_array_field(table, "all_links"),
            downloaded_paths: string_array_field(table, "downloaded_paths")
                .into_iter()
                .map(Utf8PathBuf::from)
                .collect(),
            doi_list: string_array_field(table, "doi_list"),
            notes: string_array_field(table, "notes"),
        });
    }
    Ok(out)
}

fn load_documents(path: &Path) -> Result<Vec<DocumentRecord>> {
    let value = load_toml_value(path)?;
    let documents = value
        .get("document")
        .and_then(Value::as_array)
        .context("document table missing")?;
    let mut out = Vec::new();
    for document in documents {
        let table = document.as_table().context("document row must be a table")?;
        out.push(DocumentRecord {
            id: string_field(table, "id"),
            path: Utf8PathBuf::from(string_field(table, "path")),
            title: string_field(table, "title"),
            kind: string_field(table, "kind"),
            authoring_mode: string_field(table, "authoring_mode"),
            generated: bool_field(table, "generated"),
            status: string_field(table, "status"),
            toml_backing: optional_string_field(table, "toml_backing").map(Utf8PathBuf::from),
            sha256: optional_string_field(table, "sha256"),
            size_bytes: optional_integer_field(table, "size_bytes"),
            line_count: optional_integer_field(table, "line_count"),
        });
    }
    Ok(out)
}

fn load_lane_assignments(lane_dir: &Path) -> Result<Vec<LaneAssignment>> {
    let mut out = Vec::new();
    for lane_name in ["datasets", "slides_artifacts", "papers_pdf", "web_references"] {
        let path = lane_dir.join(format!("{lane_name}.toml"));
        if !path.exists() {
            continue;
        }
        let value = load_toml_value(&path)?;
        let refs = value
            .get("artifact_ref")
            .and_then(Value::as_array)
            .context("artifact_ref table missing")?;
        for artifact_ref in refs {
            let table = artifact_ref
                .as_table()
                .context("artifact_ref row must be a table")?;
            out.push(LaneAssignment {
                artifact_id: string_field(table, "id"),
                lane_name: lane_name.to_string(),
            });
        }
    }
    Ok(out)
}

fn build_mirror_observations(path: &Path) -> Result<Vec<MirrorObservationRecord>> {
    let value = load_toml_value(path)?;
    let artifacts = value
        .get("artifact")
        .and_then(Value::as_array)
        .context("artifact table missing")?;
    let mut out = Vec::new();
    for artifact in artifacts {
        let table = artifact.as_table().context("artifact row must be a table")?;
        let artifact_id = string_field(table, "id");
        for (field, kind) in [
            ("working_mirrors", MirrorKind::Working),
            ("working_pdf_mirrors", MirrorKind::WorkingPdf),
            ("nonworking_mirrors", MirrorKind::Nonworking),
            ("unverified_mirrors", MirrorKind::Unverified),
        ] {
            for url in string_array_field(table, field) {
                out.push(MirrorObservationRecord {
                    artifact_id: artifact_id.clone(),
                    url,
                    mirror_kind: kind.clone(),
                });
            }
        }
    }
    Ok(out)
}

fn load_toml_value(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn string_field(table: &toml::map::Map<String, Value>, key: &str) -> String {
    table.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

fn optional_string_field(table: &toml::map::Map<String, Value>, key: &str) -> Option<String> {
    let value = table.get(key).and_then(Value::as_str).unwrap_or("").trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn string_array_field(table: &toml::map::Map<String, Value>, key: &str) -> Vec<String> {
    table
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items.iter()
                .filter_map(Value::as_str)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn bool_field(table: &toml::map::Map<String, Value>, key: &str) -> bool {
    table.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn optional_integer_field(table: &toml::map::Map<String, Value>, key: &str) -> Option<i64> {
    table.get(key).and_then(Value::as_integer)
}

fn host_for_url(url: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(|host| host.to_string()))
}

fn join_refs(values: &[String]) -> String {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
}
