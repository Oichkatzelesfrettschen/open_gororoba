use super::*;

impl ProvenanceStore {
    /// Read-only accessor for the current status_note on a claim row.
    /// Returns Ok(None) if the row exists but the column is NULL,
    /// Err if the row does not exist.
    pub fn claim_status_note(&self, id: &str) -> Result<Option<String>> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT status_note FROM claims WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("claim {} not found in canonical DB: {}", id, e))?;
        Ok(row)
    }

    /// Update the status_note on a claim row inside a BEGIN IMMEDIATE
    /// transaction, append a row to claim_revisions, and return the
    /// audit record. The compat-export TOML must be regenerated
    /// afterwards via `make registry-export-markdown`.
    ///
    /// Idempotent: if the new note equals the current note, the
    /// function still records a `touch` revision so the actor + reason
    /// are preserved, but does not change the underlying row.
    pub fn claim_update_status_note(
        &mut self,
        id: &str,
        new_note: &str,
        actor: &str,
        reason: Option<&str>,
    ) -> Result<StatusNoteRevision> {
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let prev_note: Option<String> = tx
            .query_row(
                "SELECT status_note FROM claims WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "claim {} not found in canonical DB (or read failed): {}",
                    id,
                    e
                )
            })?;
        let prev_value_sha256 = prev_note.as_deref().map(sha256_hex);
        let new_value_sha256 = sha256_hex(new_note);
        let operation = if prev_note.as_deref() == Some(new_note) {
            "touch"
        } else {
            tx.execute(
                "UPDATE claims SET status_note = ?2 WHERE id = ?1",
                params![id, new_note],
            )?;
            "update"
        };
        tx.execute(
            "INSERT INTO claim_revisions
             (claim_id, field_name, prev_value_sha256, new_value_sha256,
              actor, reason, operation, application_id)
             VALUES (?1, 'status_note', ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                id,
                prev_value_sha256,
                new_value_sha256,
                actor,
                reason,
                operation,
                CLI_APPLICATION_ID
            ],
        )?;
        let revision_id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(StatusNoteRevision {
            entity_id: id.to_string(),
            field_name: "status_note".to_string(),
            prev_value_sha256,
            new_value_sha256,
            actor: actor.to_string(),
            reason: reason.map(str::to_string),
            revision_id,
        })
    }

    /// Read-only accessor for an insight's status_note column (added in
    /// migration 0016).
    pub fn insight_status_note(&self, id: &str) -> Result<Option<String>> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT status_note FROM insights WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("insight {} not found in canonical DB: {}", id, e))?;
        Ok(row)
    }

    /// Update the status_note on an insight row. Mirrors
    /// claim_update_status_note end-to-end.
    pub fn insight_update_status_note(
        &mut self,
        id: &str,
        new_note: &str,
        actor: &str,
        reason: Option<&str>,
    ) -> Result<StatusNoteRevision> {
        self.entity_update_status_note(
            id,
            new_note,
            actor,
            reason,
            EntityFieldTarget {
                table: "insights",
                revisions_table: "insight_revisions",
                fk_col: "insight_id",
                field: "status_note",
            },
        )
    }

    /// Read-only accessor for an insight's summary, which lives inside the
    /// cached compat TOML body because the insights table has no summary
    /// column.
    pub fn insight_summary(&self, id: &str) -> Result<Option<String>> {
        let compat: String = self
            .conn
            .query_row(
                "SELECT compat_toml_text FROM insights WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("insight {} not found in canonical DB: {}", id, e))?;
        let doc: toml_edit::DocumentMut = compat
            .parse()
            .with_context(|| format!("parse compat TOML body of insight {id}"))?;
        Ok(doc.get("summary").and_then(|v| v.as_str()).map(str::to_string))
    }

    /// Rewrite the summary inside an insight's cached compat TOML body in one
    /// BEGIN IMMEDIATE transaction and append an insight_revisions row with
    /// field_name 'summary'. toml_edit keeps every other key and its
    /// formatting, so the export projects only the summary change.
    pub fn insight_update_summary(
        &mut self,
        id: &str,
        new_summary: &str,
        actor: &str,
        reason: Option<&str>,
    ) -> Result<StatusNoteRevision> {
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let compat: String = tx
            .query_row(
                "SELECT compat_toml_text FROM insights WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("insight {} not found in canonical DB: {}", id, e))?;
        let mut doc: toml_edit::DocumentMut = compat
            .parse()
            .with_context(|| format!("parse compat TOML body of insight {id}"))?;
        let prev = doc.get("summary").and_then(|v| v.as_str()).map(str::to_string);
        let prev_value_sha256 = prev.as_deref().map(sha256_hex);
        let new_value_sha256 = sha256_hex(new_summary);
        let operation = if prev.as_deref() == Some(new_summary) {
            "touch"
        } else {
            doc["summary"] = toml_edit::value(new_summary);
            tx.execute(
                "UPDATE insights SET compat_toml_text = ?2 WHERE id = ?1",
                params![id, doc.to_string()],
            )?;
            "update"
        };
        tx.execute(
            "INSERT INTO insight_revisions
             (insight_id, field_name, prev_value_sha256, new_value_sha256,
              actor, reason, operation, application_id)
             VALUES (?1, 'summary', ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                id,
                prev_value_sha256,
                new_value_sha256,
                actor,
                reason,
                operation,
                CLI_APPLICATION_ID
            ],
        )?;
        let revision_id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(StatusNoteRevision {
            entity_id: id.to_string(),
            field_name: "summary".to_string(),
            prev_value_sha256,
            new_value_sha256,
            actor: actor.to_string(),
            reason: reason.map(str::to_string),
            revision_id,
        })
    }

    /// Read-only accessor for an experiment's status_note column.
    pub fn experiment_status_note(&self, id: &str) -> Result<Option<String>> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT status_note FROM experiments_cp WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("experiment {} not found in canonical DB: {}", id, e))?;
        Ok(row)
    }

    /// Update the status_note on an experiment row. Mirrors
    /// claim_update_status_note end-to-end.
    pub fn experiment_update_status_note(
        &mut self,
        id: &str,
        new_note: &str,
        actor: &str,
        reason: Option<&str>,
    ) -> Result<StatusNoteRevision> {
        self.entity_update_status_note(
            id,
            new_note,
            actor,
            reason,
            EntityFieldTarget {
                table: "experiments_cp",
                revisions_table: "experiment_revisions",
                fk_col: "experiment_id",
                field: "status_note",
            },
        )
    }

    /// Generic helper for status_note updates across claims, insights,
    /// experiments_cp. Caller passes the table, the revisions table, and
    /// the fk column name. All three call sites use this; it is the only
    /// place SQL is constructed for the entity-level update.
    pub(super) fn entity_update_status_note(
        &mut self,
        id: &str,
        new_note: &str,
        actor: &str,
        reason: Option<&str>,
        target: EntityFieldTarget<'_>,
    ) -> Result<StatusNoteRevision> {
        self.entity_update_field(
            id,
            new_note,
            actor,
            reason,
            EntityFieldTarget {
                table: target.table,
                revisions_table: target.revisions_table,
                fk_col: target.fk_col,
                field: "status_note",
            },
        )
    }

    /// Generic per-column updater used by status_note and formal_proof
    /// mutators. Wraps a single BEGIN IMMEDIATE transaction that reads
    /// the prior value, hashes prev/new, conditionally writes, and
    /// appends a revisions audit row. `target.field` must be a trusted
    /// identifier from the call site (never user input).
    pub fn entity_update_field(
        &mut self,
        id: &str,
        new_value: &str,
        actor: &str,
        reason: Option<&str>,
        target: EntityFieldTarget<'_>,
    ) -> Result<StatusNoteRevision> {
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let revision =
            Self::entity_update_field_in_transaction(&tx, id, new_value, actor, reason, target)?;
        tx.commit()?;
        Ok(revision)
    }

    pub(super) fn entity_update_field_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        id: &str,
        new_value: &str,
        actor: &str,
        reason: Option<&str>,
        target: EntityFieldTarget<'_>,
    ) -> Result<StatusNoteRevision> {
        let EntityFieldTarget {
            table,
            revisions_table,
            fk_col,
            field,
        } = target;
        let select_sql = format!("SELECT {field} FROM {table} WHERE id = ?1");
        let update_sql = format!("UPDATE {table} SET {field} = ?2 WHERE id = ?1");
        let insert_sql = format!(
            "INSERT INTO {revisions_table}
             ({fk_col}, field_name, prev_value_sha256, new_value_sha256,
              actor, reason, operation, application_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"
        );
        let prev: Option<String> = tx
            .query_row(&select_sql, params![id], |row| row.get(0))
            .map_err(|e| anyhow::anyhow!("{} {} not found in canonical DB: {}", table, id, e))?;
        let prev_value_sha256 = prev.as_deref().map(sha256_hex);
        let new_value_sha256 = sha256_hex(new_value);
        let operation = if prev.as_deref() == Some(new_value) {
            "touch"
        } else {
            tx.execute(&update_sql, params![id, new_value])?;
            "update"
        };
        tx.execute(
            &insert_sql,
            params![
                id,
                field,
                prev_value_sha256,
                new_value_sha256,
                actor,
                reason,
                operation,
                CLI_APPLICATION_ID
            ],
        )?;
        let revision_id = tx.last_insert_rowid();
        Ok(StatusNoteRevision {
            entity_id: id.to_string(),
            field_name: field.to_string(),
            prev_value_sha256,
            new_value_sha256,
            actor: actor.to_string(),
            reason: reason.map(str::to_string),
            revision_id,
        })
    }

    /// Read-only accessor for the current formal_proof on a claim row.
    pub fn claim_formal_proof(&self, id: &str) -> Result<Option<String>> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT formal_proof FROM claims WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("claim {} not found in canonical DB: {}", id, e))?;
        Ok(row)
    }

    /// Update the formal_proof on a claim row. Mirrors
    /// claim_update_status_note end-to-end via entity_update_field.
    pub fn claim_update_formal_proof(
        &mut self,
        id: &str,
        new_value: &str,
        actor: &str,
        reason: Option<&str>,
    ) -> Result<StatusNoteRevision> {
        self.entity_update_field(
            id,
            new_value,
            actor,
            reason,
            EntityFieldTarget {
                table: "claims",
                revisions_table: "claim_revisions",
                fk_col: "claim_id",
                field: "formal_proof",
            },
        )
    }

}
