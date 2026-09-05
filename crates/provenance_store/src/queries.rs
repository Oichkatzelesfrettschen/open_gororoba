use super::*;

impl ProvenanceStore {
    pub fn list_claims(&self) -> Result<Vec<ClaimRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, statement, status, where_stated, last_verified, formal_proof, status_note, compat_toml_text
             FROM claims ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ClaimRecord {
                id: row.get(0)?,
                statement: row.get(1)?,
                status: row.get(2)?,
                where_stated: row.get(3)?,
                last_verified: row.get(4)?,
                formal_proof: row.get(5)?,
                status_note: row.get(6)?,
                compat_toml_text: row.get(7)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_insights(&self) -> Result<Vec<InsightRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, status, claim_refs_json, compat_toml_text
             FROM insights ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let claim_refs_json: String = row.get(3)?;
            Ok(InsightRecord {
                id: row.get(0)?,
                title: row.get(1)?,
                status: row.get(2)?,
                claim_refs: serde_json::from_str(&claim_refs_json).unwrap_or_default(),
                compat_toml_text: row.get(4)?,
            })
        })?;
        collect_rows(rows)
    }

    pub(super) fn list_insights_for_compat(&self) -> Result<Vec<types::InsightCompatRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, status, claim_refs_json, status_note, compat_toml_text
             FROM insights ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let claim_refs_json: String = row.get(3)?;
            Ok(types::InsightCompatRecord {
                id: row.get(0)?,
                title: row.get(1)?,
                status: row.get(2)?,
                claim_refs: serde_json::from_str(&claim_refs_json).unwrap_or_default(),
                status_note: row.get(4)?,
                compat_toml_text: row.get(5)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_experiments(&self) -> Result<Vec<ExperimentRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, status, binary_name, claim_refs_json, compat_toml_text
             FROM experiments_cp ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let claim_refs_json: String = row.get(4)?;
            Ok(ExperimentRecord {
                id: row.get(0)?,
                title: row.get(1)?,
                status: row.get(2)?,
                binary: row.get(3)?,
                claim_refs: serde_json::from_str(&claim_refs_json).unwrap_or_default(),
                compat_toml_text: row.get(5)?,
            })
        })?;
        collect_rows(rows)
    }

    pub(super) fn list_experiments_for_compat(&self) -> Result<Vec<types::ExperimentCompatRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, status, binary_name, claim_refs_json, status_note, compat_toml_text
             FROM experiments_cp ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let claim_refs_json: String = row.get(4)?;
            Ok(types::ExperimentCompatRecord {
                id: row.get(0)?,
                title: row.get(1)?,
                status: row.get(2)?,
                binary: row.get(3)?,
                claim_refs: serde_json::from_str(&claim_refs_json).unwrap_or_default(),
                status_note: row.get(5)?,
                compat_toml_text: row.get(6)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_theorems(&self) -> Result<Vec<TheoremRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT COALESCE(theorem_identities.stable_id, 'THM-LEGACY-' || theorems.id),
                    theorems.id, theorems.title, theorems.proof_path, theorems.status,
                    COALESCE(theorem_identities.identity_kind, 'unresolved'),
                    theorems.linked_claim_ids_json, theorems.source
             FROM theorems
             LEFT JOIN theorem_identities
               ON theorem_identities.legacy_name = theorems.id
             ORDER BY theorem_identities.stable_id, theorems.id",
        )?;
        let rows = stmt.query_map([], |row| {
            let links: String = row.get(6)?;
            Ok(TheoremRecord {
                id: row.get(0)?,
                legacy_name: row.get(1)?,
                title: row.get(2)?,
                proof_path: Utf8PathBuf::from(row.get::<_, String>(3)?),
                status: row.get(4)?,
                identity_kind: row.get(5)?,
                linked_claim_ids: serde_json::from_str(&links).unwrap_or_default(),
                source: row.get(7)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_binaries(&self) -> Result<Vec<BinaryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, crate_name, description, experiment_id, source
             FROM binaries_cp ORDER BY name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(BinaryRecord {
                name: row.get(0)?,
                crate_name: row.get(1)?,
                description: row.get(2)?,
                experiment: row.get(3)?,
                source: row.get(4)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_external_source_contracts(&self) -> Result<Vec<ExternalSourceContractRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path_glob, canonical_url, access_class, status, retrieval_method,
                    attempt_deadline_utc, resolution_deadline_utc, blocker_note
             FROM external_source_contracts
             ORDER BY id",
        )?;
        let base_rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut out = Vec::with_capacity(base_rows.len());
        for (
            id,
            path_glob,
            canonical_url,
            access_class,
            status,
            retrieval_method,
            attempt_deadline_utc,
            resolution_deadline_utc,
            blocker_note,
        ) in base_rows
        {
            out.push(ExternalSourceContractRecord {
                id: id.clone(),
                path_glob,
                canonical_url,
                mirror_urls: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "mirror_url",
                )?,
                access_class,
                status,
                retrieval_method,
                attempt_deadline_utc,
                resolution_deadline_utc,
                blocker_note,
                evidence_refs: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "evidence_ref",
                )?,
                manual_manifest_refs: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "manual_manifest_ref",
                )?,
                blocked_action_plan: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "blocked_action_plan",
                )?,
                scientific_validator_refs: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "scientific_validator_ref",
                )?,
            });
        }
        Ok(out)
    }

    pub fn list_external_source_dossiers(&self) -> Result<Vec<ExternalSourceDossierRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_markdown, slug, title, status_token, content_kind,
                    authority_level, verification_level, operational_role,
                    source_lineage_summary, has_full_transcript, line_count, notes, body_markdown
             FROM external_source_dossiers
             ORDER BY id",
        )?;
        let base_rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                    row.get::<_, String>(12)?,
                    row.get::<_, String>(13)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut out = Vec::with_capacity(base_rows.len());
        for (
            id,
            source_markdown,
            slug,
            title,
            status_token,
            content_kind,
            authority_level,
            verification_level,
            operational_role,
            source_lineage_summary,
            has_full_transcript,
            line_count,
            notes,
            body_markdown,
        ) in base_rows
        {
            out.push(ExternalSourceDossierRecord {
                id: id.clone(),
                source_markdown,
                slug,
                title,
                status_token,
                content_kind,
                authority_level,
                verification_level,
                operational_role,
                source_lineage_summary,
                truth_surfaces: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "truth_surface",
                )?,
                artifact_contract_paths: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "artifact_contract_path",
                )?,
                has_full_transcript: has_full_transcript != 0,
                claim_refs: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "claim_ref",
                )?,
                url_refs: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "url_ref",
                )?,
                path_refs: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "path_ref",
                )?,
                line_count: line_count as usize,
                notes,
                body_markdown,
            });
        }
        Ok(out)
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

        let Some((
            id,
            key,
            title,
            citation,
            status_raw,
            minimum_requirement_met,
            canonical_functional_url,
            canonical_download_path,
        )) = row
        else {
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
        Ok(Some(DocumentQueryResult {
            document,
            source_refs,
        }))
    }

    pub fn recent_download_jobs(&self, limit: usize) -> Result<Vec<DownloadQueryResult>> {
        self.query_download_jobs(limit, None, None, None, None)
    }

    pub fn query_download_jobs(
        &self,
        limit: usize,
        needle: Option<&str>,
        host: Option<&str>,
        status: Option<&str>,
        backend: Option<&str>,
    ) -> Result<Vec<DownloadQueryResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, requested_url, transfer_kind, requested_backend, route_scheme, route_host,
                    route_backends_json, note, status, final_url, output_path, created_at
             FROM download_jobs
             WHERE (?1 IS NULL OR requested_url LIKE '%' || ?1 || '%')
               AND (?2 IS NULL OR route_host = ?2)
               AND (?3 IS NULL OR status = ?3)
               AND (?4 IS NULL OR requested_backend = ?4
                    OR EXISTS (
                        SELECT 1 FROM download_attempts a
                        WHERE a.job_id = download_jobs.id AND a.backend = ?4
                    ))
             ORDER BY id DESC
             LIMIT ?5",
        )?;
        let mut rows = stmt.query(params![needle, host, status, backend, limit as i64,])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            let job_id = row.get::<_, i64>(0)?;
            let route_backends_json = row.get::<_, String>(6)?;
            let attempts = self.download_attempts_for_job(job_id)?;
            results.push(DownloadQueryResult {
                job: DownloadJobRecord {
                    id: Some(job_id),
                    requested_url: row.get(1)?,
                    transfer_kind: row.get(2)?,
                    requested_backend: row.get(3)?,
                    route_scheme: row.get(4)?,
                    route_host: row.get(5)?,
                    route_backends: serde_json::from_str(&route_backends_json).unwrap_or_default(),
                    note: row.get(7)?,
                    status: row.get(8)?,
                    final_url: row.get(9)?,
                    output_path: row.get(10)?,
                    created_at: row.get(11)?,
                },
                attempts,
            });
        }
        Ok(results)
    }

    pub fn download_attempts_for_job(&self, job_id: i64) -> Result<Vec<DownloadAttemptRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, backend, succeeded, failure_class, http_code, content_type, bytes, sha256,
                    is_pdf, final_url, note, error_message, recorded_at
             FROM download_attempts
             WHERE job_id = ?1
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![job_id], |row| {
            Ok(DownloadAttemptRecord {
                id: Some(row.get(0)?),
                job_id: Some(job_id),
                backend: row.get(1)?,
                succeeded: row.get::<_, i64>(2)? != 0,
                failure_class: row.get(3)?,
                http_code: row.get(4)?,
                content_type: row.get(5)?,
                bytes: row.get(6)?,
                sha256: row.get(7)?,
                is_pdf: row.get::<_, i64>(8)? != 0,
                final_url: row.get(9)?,
                note: row.get(10)?,
                error_message: row.get(11)?,
                recorded_at: row.get(12)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn recent_download_campaigns(
        &self,
        limit: usize,
    ) -> Result<Vec<DownloadCampaignQueryResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.name, c.command_kind, c.input_path, c.out_ledger_path, c.dest_dir, c.note, c.created_at,
                    COUNT(j.id) AS job_count,
                    SUM(CASE WHEN j.status = 'succeeded' THEN 1 ELSE 0 END) AS success_count,
                    SUM(CASE WHEN j.status = 'failed' THEN 1 ELSE 0 END) AS failure_count
             FROM download_campaigns c
             LEFT JOIN download_campaign_jobs cj ON cj.campaign_id = c.id
             LEFT JOIN download_jobs j ON j.id = cj.job_id
             GROUP BY c.id, c.name, c.command_kind, c.input_path, c.out_ledger_path, c.dest_dir, c.note, c.created_at
             ORDER BY c.id DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(DownloadCampaignQueryResult {
                campaign: DownloadCampaignRecord {
                    id: Some(row.get(0)?),
                    name: row.get(1)?,
                    command_kind: row.get(2)?,
                    input_path: row.get(3)?,
                    out_ledger_path: row.get(4)?,
                    dest_dir: row.get(5)?,
                    note: row.get(6)?,
                    created_at: row.get(7)?,
                },
                job_count: row.get::<_, i64>(8)?.max(0) as usize,
                success_count: row.get::<_, Option<i64>>(9)?.unwrap_or(0).max(0) as usize,
                failure_count: row.get::<_, Option<i64>>(10)?.unwrap_or(0).max(0) as usize,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn project_download_history_rows(
        &self,
        limit: usize,
        needle: Option<&str>,
        host: Option<&str>,
        status: Option<&str>,
        backend: Option<&str>,
    ) -> Result<Vec<DownloadLedgerProjectionRow>> {
        let jobs = self.query_download_jobs(limit, needle, host, status, backend)?;
        let mut rows = Vec::new();
        for result in jobs {
            let job_id = result.job.id.unwrap_or_default();
            for attempt in result.attempts {
                let attempt_id = attempt.id.unwrap_or_default();
                let id = format!("job_{job_id:06}_attempt_{attempt_id:06}");
                let note = match attempt.error_message.as_deref() {
                    Some(error) if !error.is_empty() => {
                        format!("{}; error={error}", attempt.note)
                    }
                    _ => attempt.note.clone(),
                };
                let note = match attempt.failure_class.as_deref() {
                    Some(failure_class) if !failure_class.is_empty() => {
                        format!("{note}; failure_class={failure_class}")
                    }
                    _ => note,
                };
                rows.push(DownloadLedgerProjectionRow {
                    id,
                    url: attempt
                        .final_url
                        .clone()
                        .or_else(|| result.job.final_url.clone())
                        .unwrap_or_else(|| result.job.requested_url.clone()),
                    http_code: attempt
                        .http_code
                        .map(|value| value.to_string())
                        .unwrap_or_default(),
                    content_type: attempt.content_type.clone().unwrap_or_default(),
                    bytes: attempt.bytes.max(0) as u64,
                    sha256: attempt.sha256.clone().unwrap_or_default(),
                    is_pdf: if attempt.is_pdf { "yes" } else { "no" }.to_string(),
                    note,
                    status: result.job.status.clone(),
                });
            }
        }
        Ok(rows)
    }

}
