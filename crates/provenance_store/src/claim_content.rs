//! Optimistic, atomic corrections retaining complete content in the audit record.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimContentSpec {
    pub claim_id: String,
    pub expected_statement: String,
    pub expected_where_stated: String,
    pub statement: String,
    pub where_stated: String,
}

impl ProvenanceStore {
    pub fn parse_claim_content_spec(text: &str) -> Result<ClaimContentSpec> {
        toml::from_str(text).context("parse claim content correction")
    }

    /// Correct both content columns and their compatibility representation atomically.
    /// The append-only reason envelope retains the complete expected and new values.
    pub fn correct_claim_content(
        &mut self,
        spec: &ClaimContentSpec,
        actor: &str,
        reason: &str,
    ) -> Result<Vec<StatusNoteRevision>> {
        for (name, value) in [
            ("claim_id", spec.claim_id.as_str()),
            ("expected_statement", spec.expected_statement.as_str()),
            ("statement", spec.statement.as_str()),
            ("where_stated", spec.where_stated.as_str()),
            ("actor", actor),
            ("reason", reason),
        ] {
            anyhow::ensure!(!value.trim().is_empty(), "{name} must contain text");
            anyhow::ensure!(
                !value
                    .chars()
                    .any(|character| character.is_control()
                        && !matches!(character, '\n' | '\r' | '\t')),
                "{name} contains a prohibited control character"
            );
        }
        let transaction = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let (statement, locator, compat): (String, String, String) = transaction
            .query_row(
                "SELECT statement,where_stated,compat_toml_text FROM claims WHERE id=?1",
                [&spec.claim_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .context("read claim content prestate")?;
        anyhow::ensure!(
            statement == spec.expected_statement && locator == spec.expected_where_stated,
            "stale claim content prestate for {}",
            spec.claim_id
        );
        let mut document: toml_edit::DocumentMut =
            compat.parse().context("parse claim compatibility body")?;
        document["statement"] = toml_edit::value(&spec.statement);
        document["where_stated"] = toml_edit::value(&spec.where_stated);
        let envelope = serde_json::to_string(&serde_json::json!({
            "schema": "claim_content_correction_v1", "reason": reason, "spec": spec
        }))?;
        transaction.execute(
            "UPDATE claims SET statement=?2,where_stated=?3,compat_toml_text=?4 WHERE id=?1",
            params![
                spec.claim_id,
                spec.statement,
                spec.where_stated,
                document.to_string()
            ],
        )?;
        let mut revisions = Vec::new();
        for (field_name, previous, next) in [
            ("statement", &statement, &spec.statement),
            ("where_stated", &locator, &spec.where_stated),
        ] {
            let previous_hash = sha256_hex(previous);
            let next_hash = sha256_hex(next);
            transaction.execute(
                "INSERT INTO claim_revisions (claim_id,field_name,prev_value_sha256,new_value_sha256,actor,reason,operation,application_id) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                params![spec.claim_id, field_name, previous_hash, next_hash, actor, envelope,
                    if previous == next { "touch" } else { "update" }, CLI_APPLICATION_ID],
            )?;
            revisions.push(StatusNoteRevision {
                entity_id: spec.claim_id.clone(),
                field_name: field_name.into(),
                prev_value_sha256: Some(previous_hash),
                new_value_sha256: next_hash,
                actor: actor.into(),
                reason: Some(envelope.clone()),
                revision_id: transaction.last_insert_rowid(),
            });
        }
        transaction.commit()?;
        Ok(revisions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Result<(ProvenanceStore, ClaimContentSpec)> {
        let store = ProvenanceStore::open(Path::new(":memory:"))?;
        store.conn.execute("INSERT INTO claims (id,statement,status,where_stated,last_verified,compat_toml_text) VALUES ('C-1','old statement','Provisional','old locator','','extra = 42')", [])?;
        Ok((
            store,
            ClaimContentSpec {
                claim_id: "C-1".into(),
                expected_statement: "old statement".into(),
                expected_where_stated: "old locator".into(),
                statement: "bounded correction".into(),
                where_stated: "tested locator".into(),
            },
        ))
    }

    #[test]
    fn correction_retains_history_and_exports_content() -> Result<()> {
        let (mut store, spec) = fixture()?;
        assert_eq!(
            ProvenanceStore::parse_claim_content_spec(&toml::to_string(&spec)?)?,
            spec
        );
        let revisions =
            store.correct_claim_content(&spec, "investigator", "measurement reconciliation")?;
        assert_eq!(revisions.len(), 2);
        let envelope: serde_json::Value =
            serde_json::from_str(revisions[0].reason.as_deref().unwrap())?;
        assert_eq!(
            serde_json::from_value::<ClaimContentSpec>(envelope["spec"].clone())?,
            spec
        );
        let rendered = store.render_control_plane_compat_outputs()?.claims;
        let exported: toml::Value = toml::from_str(&rendered)?;
        assert_eq!(
            exported["claim"][0]["statement"].as_str(),
            Some(spec.statement.as_str())
        );
        assert_eq!(
            exported["claim"][0]["where_stated"].as_str(),
            Some(spec.where_stated.as_str())
        );
        assert_eq!(exported["claim"][0]["status"].as_str(), Some("Provisional"));
        assert_eq!(exported["claim"][0]["extra"].as_integer(), Some(42));
        assert!(
            store
                .correct_claim_content(&spec, "investigator", "stale retry")
                .is_err()
        );
        assert_eq!(store.table_row_count("claim_revisions")?, 2);
        Ok(())
    }

    #[test]
    fn correction_rolls_back_both_fields_and_history_on_second_audit_failure() -> Result<()> {
        let (mut store, spec) = fixture()?;
        store.conn.execute_batch("CREATE TRIGGER reject_locator_revision BEFORE INSERT ON claim_revisions WHEN NEW.field_name='where_stated' BEGIN SELECT RAISE(ABORT,'injected audit failure'); END;")?;
        assert!(
            store
                .correct_claim_content(&spec, "investigator", "correction")
                .is_err()
        );
        let prestate: (String, String, String) = store.conn.query_row(
            "SELECT statement,where_stated,compat_toml_text FROM claims WHERE id='C-1'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        assert_eq!(
            prestate,
            (
                spec.expected_statement,
                spec.expected_where_stated,
                "extra = 42".into()
            )
        );
        assert_eq!(store.table_row_count("claim_revisions")?, 0);
        Ok(())
    }

    #[test]
    fn correction_rejects_empty_content_and_unknown_spec_fields() -> Result<()> {
        let (mut store, mut spec) = fixture()?;
        spec.where_stated = "  ".into();
        assert!(
            store
                .correct_claim_content(&spec, "investigator", "correction")
                .is_err()
        );
        assert_eq!(store.table_row_count("claim_revisions")?, 0);
        assert!(
            ProvenanceStore::parse_claim_content_spec(&format!(
                "{}\nunknown = 1",
                toml::to_string(&spec)?
            ))
            .is_err()
        );
        Ok(())
    }
}
