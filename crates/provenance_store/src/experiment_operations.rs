use super::*;

impl ProvenanceStore {
    pub fn replace_control_plane_experiments_from_registry_text(
        &mut self,
        repo_root: &Path,
        source_path: &Path,
        raw: &str,
    ) -> Result<usize> {
        let indexed_at = Utc::now().to_rfc3339();
        let experiments = load_experiments_from_registry(raw)?;
        let experiments_meta_toml =
            load_registry_table_toml(raw, "experiments")?.unwrap_or_default();
        let tx = self.conn.transaction()?;
        let incoming_experiment_ids = experiments
            .iter()
            .map(|experiment| experiment.id.as_str())
            .collect::<BTreeSet<_>>();
        let protected_experiment_ids = {
            let mut statement = tx.prepare(
                "SELECT experiment_id FROM claim_transition_experiments
                 UNION
                 SELECT experiment_id FROM claim_evidence_revision_experiments
                 UNION
                 SELECT experiment_id FROM experiment_revisions",
            )?;
            statement
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<BTreeSet<_>, _>>()?
        };
        let missing_protected_ids = protected_experiment_ids
            .iter()
            .filter(|id| !incoming_experiment_ids.contains(id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if !missing_protected_ids.is_empty() {
            bail!(
                "experiment registry omits protected canonical experiments: {}",
                missing_protected_ids.join(", ")
            );
        }

        // The execution-planning build reaches this path with rendered registry
        // text, which is the same mirror-to-canonical direction the
        // index-control-plane guard covers. Capture the canonical-only column
        // values first so the delete-and-reinsert below cannot null them.
        let preserved = capture_preserved_values(&tx)?;

        // Rebuild the derived claim-to-experiment join before replacing the
        // experiment rows. Transition evidence and revision history retain
        // their direct foreign keys and require their referenced experiments
        // to remain present in the incoming registry.
        tx.execute("DELETE FROM claim_experiment_refs", [])?;
        tx.execute(
            "DELETE FROM experiments_cp
             WHERE id NOT IN (
                 SELECT experiment_id FROM claim_transition_experiments
                 UNION
                 SELECT experiment_id FROM claim_evidence_revision_experiments
                 UNION
                 SELECT experiment_id FROM experiment_revisions
             )",
            [],
        )?;
        tx.execute(
            "DELETE FROM control_plane_meta WHERE kind = 'experiments'",
            [],
        )?;
        write_registry_snapshot(&tx, repo_root, "experiments", source_path, raw, &indexed_at)?;
        for experiment in &experiments {
            tx.execute(
                "INSERT INTO experiments_cp(id, title, status, binary_name, claim_refs_json, status_note, compat_toml_text)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    status = excluded.status,
                    binary_name = excluded.binary_name,
                    claim_refs_json = excluded.claim_refs_json,
                    status_note = excluded.status_note,
                    compat_toml_text = excluded.compat_toml_text",
                params![
                    experiment.id,
                    experiment.title,
                    experiment.status,
                    experiment.binary,
                    serde_json::to_string(&experiment.claim_refs)?,
                    experiment.status_note,
                    experiment.compat_toml_text
                ],
            )?;
            for claim_id in &experiment.claim_refs {
                tx.execute(
                    "INSERT INTO claim_experiment_refs (claim_id, experiment_id)
                     VALUES (?1, ?2)
                     ON CONFLICT(claim_id, experiment_id) DO NOTHING",
                    params![claim_id, experiment.id],
                )?;
            }
        }
        tx.execute(
            "INSERT INTO control_plane_meta(kind, compat_toml_text)
             VALUES(?1, ?2)",
            params!["experiments", experiments_meta_toml],
        )?;
        restore_preserved_values(&tx, &preserved)?;
        tx.commit()?;
        self.record_control_plane_run(
            "replace_control_plane_experiments_from_registry_text",
            &serde_json::json!({
                "source_path": to_repo_rel(repo_root, source_path),
                "experiment_count": experiments.len(),
            })
            .to_string(),
        )?;
        Ok(experiments.len())
    }

    /// Insert or update the `[[experiment]]` rows in `raw` without deleting
    /// any other canonical experiment.
    /// An omitted status note preserves the existing canonical note.
    /// Changed fields and explicit notes append revisions atomically with a
    /// complete before/after audit record.
    pub fn upsert_experiments_from_registry_text(
        &mut self,
        repo_root: &Path,
        source_path: &Path,
        raw: &str,
    ) -> Result<Vec<String>> {
        let value: Value = toml::from_str(raw).context("parse experiment mutation")?;
        let rows = value
            .get("experiment")
            .and_then(Value::as_array)
            .context("experiment array missing")?;
        let mut seen_ids = BTreeSet::new();
        for row in rows {
            for field in ["id", "title", "status", "binary"] {
                let text = row
                    .get(field)
                    .and_then(Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                    .with_context(|| format!("experiment requires nonempty string {field}"))?;
                if field == "id" {
                    if !text.strip_prefix("E-").is_some_and(|suffix| {
                        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
                    }) {
                        bail!("invalid experiment ID {text:?}: expected E- followed by digits");
                    }
                    if !seen_ids.insert(text.to_owned()) {
                        bail!("duplicate experiment ID {text}");
                    }
                }
            }
        }
        let experiments = load_experiments_from_registry(raw)?;
        if experiments.is_empty() {
            bail!("experiment spec contains no [[experiment]] rows");
        }
        let note_reason = format!(
            "Apply explicit status_note from experiment fragment {}",
            to_repo_rel(repo_root, source_path)
        );
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut ids = Vec::new();
        let mut changes = Vec::new();
        let read_row = |tx: &rusqlite::Transaction<'_>,
                        id: &str|
         -> Result<Option<serde_json::Value>> {
            let mut record = tx.query_row(
                    "SELECT title, status, binary_name, claim_refs_json, status_note, compat_toml_text
                     FROM experiments_cp WHERE id = ?1",
                    params![id],
                    |row| {
                        Ok(serde_json::json!({
                            "id": id,
                            "title": row.get::<_, String>(0)?,
                            "status": row.get::<_, String>(1)?,
                            "binary_name": row.get::<_, Option<String>>(2)?,
                            "claim_refs_json": row.get::<_, String>(3)?,
                            "status_note": row.get::<_, Option<String>>(4)?,
                            "compat_toml_text": row.get::<_, String>(5)?,
                        }))
                    },
                ).optional()?;
            if let Some(record) = &mut record {
                let mut statement = tx.prepare(
                        "SELECT claim_id FROM claim_experiment_refs WHERE experiment_id = ?1 ORDER BY claim_id",
                    )?;
                let references = statement
                    .query_map(params![id], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                record["claim_experiment_refs"] = serde_json::json!(references);
            }
            Ok(record)
        };
        for experiment in &experiments {
            let before = read_row(&tx, &experiment.id)?;
            let claim_refs_json = serde_json::to_string(&experiment.claim_refs)?;
            tx.execute(
                "INSERT INTO experiments_cp(id, title, status, binary_name, claim_refs_json, status_note, compat_toml_text)
                 VALUES(?1, ?2, ?3, ?4, ?5, NULL, ?6)
                 ON CONFLICT(id) DO NOTHING",
                params![
                    experiment.id,
                    experiment.title,
                    experiment.status,
                    experiment.binary,
                    claim_refs_json,
                    experiment.compat_toml_text
                ],
            )?;
            if let Some(previous) = &before {
                for (field, new_value) in [
                    ("title", experiment.title.as_str()),
                    ("status", experiment.status.as_str()),
                    (
                        "binary_name",
                        experiment.binary.as_deref().expect("validated binary"),
                    ),
                    ("claim_refs_json", claim_refs_json.as_str()),
                    ("compat_toml_text", experiment.compat_toml_text.as_str()),
                ] {
                    if previous[field].as_str() != Some(new_value) {
                        Self::entity_update_field_in_transaction(
                            &tx,
                            &experiment.id,
                            new_value,
                            "experiment-registry-upsert",
                            Some("Apply validated experiment fragment"),
                            EntityFieldTarget {
                                table: "experiments_cp",
                                revisions_table: "experiment_revisions",
                                fk_col: "experiment_id",
                                field,
                            },
                        )?;
                    }
                }
            }
            if let Some(note) = &experiment.status_note {
                Self::entity_update_field_in_transaction(
                    &tx,
                    &experiment.id,
                    note,
                    "experiment-registry-upsert",
                    Some(&note_reason),
                    EntityFieldTarget {
                        table: "experiments_cp",
                        revisions_table: "experiment_revisions",
                        fk_col: "experiment_id",
                        field: "status_note",
                    },
                )?;
            }
            tx.execute(
                "DELETE FROM claim_experiment_refs WHERE experiment_id = ?1",
                params![experiment.id],
            )?;
            for claim_id in &experiment.claim_refs {
                tx.execute(
                    "INSERT INTO claim_experiment_refs (claim_id, experiment_id)
                     VALUES (?1, ?2)
                     ON CONFLICT(claim_id, experiment_id) DO NOTHING",
                    params![claim_id, experiment.id],
                )?;
            }
            ids.push(experiment.id.clone());
            changes.push(serde_json::json!({
                "before": before, "after": read_row(&tx, &experiment.id)?,
            }));
        }
        tx.execute(
            "INSERT INTO control_plane_runs(action, created_at, details_json) VALUES (?1, ?2, ?3)",
            params![
                "upsert_experiments_from_registry_text",
                Utc::now().to_rfc3339(),
                serde_json::json!({
                    "source_path": to_repo_rel(repo_root, source_path),
                    "actor": "experiment-registry-upsert",
                    "reason": "Apply validated experiment fragment",
                    "spec_sha256": sha256_hex(raw),
                    "experiment_ids": ids,
                    "changes": changes,
                })
                .to_string()
            ],
        )?;
        tx.commit()?;
        Ok(ids)
    }
}
