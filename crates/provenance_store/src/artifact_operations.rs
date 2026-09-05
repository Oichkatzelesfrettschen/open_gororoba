use super::*;

impl ProvenanceStore {
    /// Add lane membership atomically while retaining artifact metadata and evidence.
    pub fn assign_artifact_lane(
        &mut self,
        ids: &[String],
        lane_name: &str,
        actor: Option<&str>,
        reason: Option<&str>,
    ) -> Result<usize> {
        if !matches!(
            lane_name,
            "datasets" | "papers_pdf" | "slides_artifacts" | "web_references"
        ) {
            bail!("unsupported artifact lane {lane_name}");
        }
        if ids.is_empty() {
            bail!("artifact lane assignment requires at least one ID");
        }
        for (field, value) in ids
            .iter()
            .map(|id| ("id", id.as_str()))
            .chain(actor.map(|value| ("actor", value)))
            .chain(reason.map(|value| ("reason", value)))
        {
            if value.trim().is_empty() || !value.is_ascii() {
                bail!("artifact {field} must be a non-empty ASCII string");
            }
        }
        let distinct_ids: BTreeSet<&str> = ids.iter().map(String::as_str).collect();
        let tx = self.conn.transaction()?;
        for id in &distinct_ids {
            let exists: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM artifacts WHERE id = ?1)",
                params![id],
                |row| row.get(0),
            )?;
            if !exists {
                bail!("unknown artifact ID {id}");
            }
        }
        let mut added_ids = Vec::new();
        for id in distinct_ids {
            if tx.execute(
                "INSERT INTO lane_assignments (artifact_id, lane_name) VALUES (?1, ?2)
                 ON CONFLICT(artifact_id, lane_name) DO NOTHING",
                params![id, lane_name],
            )? != 0
            {
                added_ids.push(id);
            }
        }
        if !added_ids.is_empty() {
            let details = serde_json::json!({"artifact_ids":added_ids,"lane":lane_name,"actor":actor,"reason":reason});
            tx.execute(
                "INSERT INTO export_runs (action, created_at, artifact_count, document_count, details_json)
                 VALUES ('assign-artifact-lane', ?1, ?2, 0, ?3)",
                params![Utc::now().to_rfc3339(), added_ids.len() as i64, details.to_string()],
            )?;
        }
        tx.commit()?;
        Ok(added_ids.len())
    }

    pub fn register_local_artifact(
        &mut self,
        repo_root: &Path,
        registration: &LocalArtifactRegistration<'_>,
    ) -> Result<usize> {
        let LocalArtifactRegistration {
            id,
            key,
            title,
            citation,
            paths,
            lane_name,
            source_refs,
            actor,
            reason,
        } = *registration;
        for (field_name, value) in [
            ("id", id),
            ("key", key),
            ("title", title),
            ("citation", citation),
            ("lane", lane_name),
        ] {
            if value.trim().is_empty() {
                bail!("local artifact {field_name} must not be empty");
            }
            if !value.is_ascii() {
                bail!("local artifact {field_name} must contain ASCII only");
            }
        }
        if !matches!(
            lane_name,
            "datasets" | "papers_pdf" | "slides_artifacts" | "web_references"
        ) {
            bail!("unsupported artifact lane {lane_name}");
        }
        if paths.is_empty() {
            bail!("local artifact registration requires at least one path");
        }

        let mut relative_paths = BTreeSet::new();
        for raw_path in paths {
            if raw_path.trim().is_empty() || !raw_path.is_ascii() {
                bail!("artifact paths must be non-empty ASCII strings");
            }
            let relative_path = Path::new(raw_path);
            if relative_path.is_absolute()
                || relative_path.components().any(|component| {
                    matches!(
                        component,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
            {
                bail!("artifact path must be repository-relative: {raw_path}");
            }
            let full_path = repo_root.join(relative_path);
            let metadata = fs::symlink_metadata(&full_path)
                .with_context(|| format!("inspect artifact path {}", full_path.display()))?;
            if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                bail!("artifact path is not a regular file: {raw_path}");
            }
            relative_paths.insert(raw_path.clone());
        }
        if relative_paths.is_empty() {
            bail!("local artifact registration requires distinct paths");
        }
        for source_ref in source_refs {
            if source_ref.trim().is_empty() || !source_ref.is_ascii() {
                bail!("artifact source references must be non-empty ASCII strings");
            }
        }
        if let Some(actor) = actor
            && (actor.trim().is_empty() || !actor.is_ascii())
        {
            bail!("artifact actor must be a non-empty ASCII string");
        }
        if let Some(reason) = reason
            && (reason.trim().is_empty() || !reason.is_ascii())
        {
            bail!("artifact reason must be a non-empty ASCII string");
        }

        let relative_paths = relative_paths.into_iter().collect::<Vec<_>>();
        let canonical_path = relative_paths
            .first()
            .expect("non-empty artifact paths after validation");
        let tx = self.conn.transaction()?;
        let conflicting_id = tx
            .query_row(
                "SELECT id FROM artifacts WHERE key = ?1 AND id <> ?2",
                params![key, id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        if let Some(conflicting_id) = conflicting_id {
            bail!("artifact key {key} already belongs to {conflicting_id}");
        }

        tx.execute(
            "INSERT INTO artifacts (
                id, key, title, citation, status,
                minimum_requirement_met, canonical_functional_url, canonical_download_path
            ) VALUES (?1, ?2, ?3, ?4, 'downloaded', 1, NULL, ?5)
            ON CONFLICT(id) DO UPDATE SET
                key = excluded.key,
                title = excluded.title,
                citation = excluded.citation,
                status = excluded.status,
                minimum_requirement_met = excluded.minimum_requirement_met,
                canonical_functional_url = excluded.canonical_functional_url,
                canonical_download_path = excluded.canonical_download_path",
            params![id, key, title, citation, canonical_path],
        )?;
        tx.execute(
            "DELETE FROM artifact_paths WHERE artifact_id = ?1 AND relation = 'downloaded'",
            params![id],
        )?;
        tx.execute(
            "DELETE FROM lane_assignments WHERE artifact_id = ?1",
            params![id],
        )?;
        tx.execute(
            "DELETE FROM record_sources WHERE entity_kind = 'artifact' AND entity_id = ?1",
            params![id],
        )?;
        for path in &relative_paths {
            tx.execute(
                "INSERT INTO artifact_paths (artifact_id, path, relation) VALUES (?1, ?2, 'downloaded')",
                params![id, path],
            )?;
        }
        tx.execute(
            "INSERT INTO lane_assignments (artifact_id, lane_name) VALUES (?1, ?2)",
            params![id, lane_name],
        )?;
        for source_ref in source_refs {
            tx.execute(
                "INSERT INTO record_sources (entity_kind, entity_id, source_ref) VALUES ('artifact', ?1, ?2)",
                params![id, source_ref],
            )?;
        }
        let details = serde_json::json!({
            "artifact_id": id,
            "key": key,
            "paths": &relative_paths,
            "lane": lane_name,
            "source_refs": source_refs,
            "actor": actor,
            "reason": reason,
        });
        tx.execute(
            "INSERT INTO export_runs (
                action, created_at, artifact_count, document_count, details_json
            ) VALUES ('register-local-artifact', ?1, 1, 0, ?2)",
            params![Utc::now().to_rfc3339(), details.to_string()],
        )?;
        tx.commit()?;
        Ok(relative_paths.len())
    }

    pub fn record_download_result(
        &mut self,
        job: &DownloadJobRecord,
        attempt: &DownloadAttemptRecord,
    ) -> Result<i64> {
        self.record_download_trace(job, std::slice::from_ref(attempt))
    }

    pub fn record_download_trace(
        &mut self,
        job: &DownloadJobRecord,
        attempts: &[DownloadAttemptRecord],
    ) -> Result<i64> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO download_jobs (
                requested_url, transfer_kind, requested_backend, route_scheme, route_host,
                route_backends_json, note, status, final_url, output_path, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                job.requested_url,
                job.transfer_kind,
                job.requested_backend,
                job.route_scheme,
                job.route_host,
                serde_json::to_string(&job.route_backends)?,
                job.note,
                job.status,
                job.final_url,
                job.output_path,
                job.created_at,
            ],
        )?;
        let job_id = tx.last_insert_rowid();
        for attempt in attempts {
            tx.execute(
                "INSERT INTO download_attempts (
                    job_id, backend, succeeded, http_code, content_type, bytes, sha256, is_pdf,
                    final_url, note, error_message, recorded_at, failure_class
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    job_id,
                    attempt.backend,
                    if attempt.succeeded { 1_i64 } else { 0_i64 },
                    attempt.http_code,
                    attempt.content_type,
                    attempt.bytes,
                    attempt.sha256,
                    if attempt.is_pdf { 1_i64 } else { 0_i64 },
                    attempt.final_url,
                    attempt.note,
                    attempt.error_message,
                    attempt.recorded_at,
                    attempt.failure_class,
                ],
            )?;
        }
        tx.commit()?;
        Ok(job_id)
    }

    pub fn create_download_campaign(&mut self, campaign: &DownloadCampaignRecord) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO download_campaigns (
                name, command_kind, input_path, out_ledger_path, dest_dir, note, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                campaign.name,
                campaign.command_kind,
                campaign.input_path,
                campaign.out_ledger_path,
                campaign.dest_dir,
                campaign.note,
                campaign.created_at,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn link_download_job_to_campaign(&mut self, campaign_id: i64, job_id: i64) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO download_campaign_jobs (campaign_id, job_id) VALUES (?1, ?2)",
            params![campaign_id, job_id],
        )?;
        Ok(())
    }
}
