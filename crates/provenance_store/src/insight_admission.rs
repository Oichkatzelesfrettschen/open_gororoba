//! Atomic insight adjudications with predicate-scoped evidence and immutable history.
use super::*;
use crate::migrations::CANONICAL_INSIGHT_STATUSES;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsightAdmissionState {
    pub title: String,
    pub status: String,
    pub summary: Option<String>,
    pub confidence: Option<String>,
    pub status_note: Option<String>,
    pub claim_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsightPredicateKind {
    AssertedProposition,
    ReviewerCounterfactual,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsightOutcome {
    Supported,
    Refuted,
    Revised,
    Inconclusive,
    Untested,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsightExecutionStatus {
    SourceReview,
    RetainedResultReview,
    Replayed,
    NotExecuted,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsightReferenceRole {
    Supports,
    Refutes,
    Qualifies,
    Motivates,
    HistoricalContext,
    FollowupResult,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsightEvidenceBinding {
    pub path: String,
    pub json_pointer: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsightPredicate {
    pub id: String,
    pub proposition: String,
    pub kind: InsightPredicateKind,
    pub evidence_layer: EvidenceLayer,
    pub outcome: InsightOutcome,
    pub execution_status: InsightExecutionStatus,
    pub boundary: String,
    pub evidence_anchors: Vec<String>,
    pub evidence_bindings: Vec<InsightEvidenceBinding>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsightClaimRole {
    pub claim_id: String,
    pub predicate_id: String,
    pub role: InsightReferenceRole,
    pub provenance: String,
    pub boundary: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsightAdmissionSpec {
    pub admission_id: String,
    pub insight_id: String,
    pub expected: InsightAdmissionState,
    pub expected_previous_evidence_sha256: Option<String>,
    pub revised: InsightAdmissionState,
    pub evidence_files: Vec<RetrievalEvidenceFile>,
    pub predicates: Vec<InsightPredicate>,
    pub references: Vec<InsightClaimRole>,
    pub disposition: String,
    pub falsifier: String,
    pub successor: String,
    pub residual: String,
}

fn text_required(value: &str) -> Result<()> {
    anyhow::ensure!(
        !value.trim().is_empty(),
        "admission text must contain content"
    );
    anyhow::ensure!(
        !value
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t')),
        "prohibited control character"
    );
    Ok(())
}

impl InsightAdmissionSpec {
    pub fn validate(&self) -> Result<()> {
        for value in [
            &self.admission_id,
            &self.insight_id,
            &self.disposition,
            &self.falsifier,
            &self.successor,
            &self.residual,
            &self.revised.title,
        ] {
            text_required(value)?;
        }
        anyhow::ensure!(
            CANONICAL_INSIGHT_STATUSES.contains(&self.revised.status.as_str()),
            "unknown insight status"
        );
        anyhow::ensure!(
            !self.evidence_files.is_empty(),
            "hash-bound evidence required"
        );
        anyhow::ensure!(
            self.evidence_files
                .iter()
                .map(|file| &file.path)
                .collect::<BTreeSet<_>>()
                .len()
                == self.evidence_files.len(),
            "duplicate bound evidence path"
        );
        for state in [&self.expected, &self.revised] {
            anyhow::ensure!(
                state.claim_refs.iter().collect::<BTreeSet<_>>().len() == state.claim_refs.len(),
                "duplicate claim reference"
            );
        }
        anyhow::ensure!(
            self.expected
                .claim_refs
                .iter()
                .all(|claim| self.revised.claim_refs.contains(claim)),
            "historical claim references must be retained"
        );
        for value in [
            &self.revised.summary,
            &self.revised.status_note,
            &self.revised.confidence,
        ]
        .into_iter()
        .flatten()
        {
            text_required(value)?;
        }
        let mut predicates = BTreeSet::new();
        anyhow::ensure!(!self.predicates.is_empty(), "predicates required");
        for predicate in &self.predicates {
            for value in [&predicate.id, &predicate.proposition, &predicate.boundary] {
                text_required(value)?;
            }
            anyhow::ensure!(predicates.insert(&predicate.id), "duplicate predicate");
            anyhow::ensure!(
                !predicate.evidence_anchors.is_empty(),
                "predicate evidence anchors required"
            );
            anyhow::ensure!(
                !predicate.evidence_bindings.is_empty(),
                "predicate evidence bindings required"
            );
            for binding in &predicate.evidence_bindings {
                anyhow::ensure!(
                    self.evidence_files
                        .iter()
                        .any(|file| file.path == binding.path),
                    "predicate binding lacks hash-bound file"
                );
                anyhow::ensure!(
                    binding.json_pointer.starts_with('/'),
                    "predicate JSON pointer required"
                );
            }
            for anchor in &predicate.evidence_anchors {
                text_required(anchor)?;
            }
            anyhow::ensure!(
                predicate.execution_status != InsightExecutionStatus::NotExecuted
                    || matches!(
                        predicate.outcome,
                        InsightOutcome::Untested | InsightOutcome::Inconclusive
                    ),
                "unexecuted predicate cannot assert an adjudicated result"
            );
        }
        let mut pairs = BTreeSet::new();
        let mut covered = BTreeSet::new();
        for reference in &self.references {
            text_required(&reference.provenance)?;
            text_required(&reference.boundary)?;
            anyhow::ensure!(
                predicates.contains(&reference.predicate_id),
                "unknown role predicate"
            );
            anyhow::ensure!(
                pairs.insert((&reference.claim_id, &reference.predicate_id)),
                "duplicate claim/predicate role"
            );
            covered.insert(&reference.claim_id);
        }
        anyhow::ensure!(
            covered == self.revised.claim_refs.iter().collect(),
            "roles must cover exactly the resulting claim references"
        );
        Ok(())
    }
}

fn read_state(
    connection: &Connection,
    insight_id: &str,
) -> Result<(InsightAdmissionState, String)> {
    let (title,status,refs,note,compat):(String,String,String,Option<String>,String)=connection.query_row("SELECT title,status,claim_refs_json,status_note,compat_toml_text FROM insights WHERE id=?1",[insight_id],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?)))?;
    let document: toml::Value = toml::from_str(&compat)?;
    let confidence = document
        .get("confidence")
        .map(|value| {
            value
                .as_str()
                .context("confidence must be text")
                .map(str::to_owned)
        })
        .transpose()?;
    let summary = document
        .get("summary")
        .map(|value| {
            value
                .as_str()
                .context("summary must be text")
                .map(str::to_owned)
        })
        .transpose()?;
    Ok((
        InsightAdmissionState {
            title,
            status,
            summary,
            confidence,
            status_note: note,
            claim_refs: serde_json::from_str(&refs)?,
        },
        compat,
    ))
}

impl ProvenanceStore {
    pub fn parse_insight_admission_spec(text: &str) -> Result<InsightAdmissionSpec> {
        let spec: InsightAdmissionSpec = toml::from_str(text)?;
        spec.validate()?;
        Ok(spec)
    }
    pub fn insight_admission_state(&self, insight_id: &str) -> Result<InsightAdmissionState> {
        Ok(read_state(&self.conn, insight_id)?.0)
    }
    pub fn admit_insight(
        &mut self,
        repo_root: &Path,
        spec: &InsightAdmissionSpec,
        actor: &str,
        reason: &str,
    ) -> Result<i64> {
        spec.validate()?;
        anyhow::ensure!(
            !spec.evidence_files.is_empty(),
            "hash-bound evidence required"
        );
        for evidence in &spec.evidence_files {
            crate::artifact_paths::verified_file_bytes(
                repo_root,
                &evidence.path,
                &evidence.sha256,
            )?;
        }
        for predicate in &spec.predicates {
            for binding in &predicate.evidence_bindings {
                let evidence = spec
                    .evidence_files
                    .iter()
                    .find(|file| file.path == binding.path)
                    .context("bound file required")?;
                let bytes = crate::artifact_paths::verified_file_bytes(
                    repo_root,
                    &evidence.path,
                    &evidence.sha256,
                )?;
                let document: serde_json::Value = serde_json::from_slice(&bytes)?;
                anyhow::ensure!(
                    document.pointer(&binding.json_pointer).is_some(),
                    "unresolved predicate JSON pointer"
                );
            }
        }
        text_required(actor)?;
        text_required(reason)?;
        let spec_json = serde_json::to_string(spec)?;
        let transaction = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let (state, compat) = read_state(&transaction, &spec.insight_id)?;
        let replay:Option<(i64,String,String,String)>=transaction.query_row("SELECT id,spec_json,after_json,after_compat FROM insight_admissions WHERE admission_id=?1",[&spec.admission_id],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?))).optional()?;
        let normalized: BTreeSet<String> = {
            let mut statement = transaction
                .prepare("SELECT claim_id FROM claim_insight_refs WHERE insight_id=?1")?;
            statement
                .query_map([&spec.insight_id], |row| row.get(0))?
                .collect::<rusqlite::Result<_>>()?
        };
        anyhow::ensure!(
            normalized == state.claim_refs.iter().cloned().collect(),
            "normalized insight references drifted"
        );
        if let Some((revision, previous_spec, after, after_compat)) = replay {
            let evidence: Option<String> = transaction
                .query_row(
                    "SELECT spec_json FROM insight_evidence WHERE insight_id=?1",
                    [&spec.insight_id],
                    |row| row.get(0),
                )
                .optional()?;
            anyhow::ensure!(
                previous_spec == spec_json
                    && after == serde_json::to_string(&state)?
                    && after_compat == compat
                    && evidence.as_deref() == Some(spec_json.as_str()),
                "admission replay conflicts with retained contract or live state"
            );
            return Ok(revision);
        }
        anyhow::ensure!(
            state == spec.expected,
            "stale insight admission prestate for {}",
            spec.insight_id
        );
        for claim in &spec.revised.claim_refs {
            let exists: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM claims WHERE id=?1)",
                [claim],
                |row| row.get(0),
            )?;
            anyhow::ensure!(exists, "unknown claim {claim}");
        }
        let previous_evidence: Option<String> = transaction
            .query_row(
                "SELECT spec_json FROM insight_evidence WHERE insight_id=?1",
                [&spec.insight_id],
                |row| row.get(0),
            )
            .optional()?;
        anyhow::ensure!(
            previous_evidence.as_deref().map(sha256_hex) == spec.expected_previous_evidence_sha256,
            "stale prior insight evidence"
        );
        let mut document: toml_edit::DocumentMut = compat.parse()?;
        document["id"] = toml_edit::value(&spec.insight_id);
        document["title"] = toml_edit::value(&spec.revised.title);
        document["status"] = toml_edit::value(&spec.revised.status);
        document["claims"] =
            toml_edit::value(spec.revised.claim_refs.iter().collect::<toml_edit::Array>());
        for (field, value) in [
            ("confidence", &spec.revised.confidence),
            ("summary", &spec.revised.summary),
            ("status_note", &spec.revised.status_note),
        ] {
            if let Some(value) = value {
                document[field] = toml_edit::value(value);
            } else {
                document.remove(field);
            }
        }
        let after_compat = document.to_string();
        transaction.execute("UPDATE insights SET title=?2,status=?3,claim_refs_json=?4,status_note=?5,compat_toml_text=?6 WHERE id=?1",params![spec.insight_id,spec.revised.title,spec.revised.status,serde_json::to_string(&spec.revised.claim_refs)?,spec.revised.status_note,after_compat])?;
        for claim in &spec.revised.claim_refs {
            transaction.execute(
                "INSERT OR IGNORE INTO claim_insight_refs (claim_id,insight_id) VALUES (?1,?2)",
                params![claim, spec.insight_id],
            )?;
        }
        transaction.execute("INSERT INTO insight_evidence (insight_id,spec_json) VALUES (?1,?2) ON CONFLICT(insight_id) DO UPDATE SET spec_json=excluded.spec_json",params![spec.insight_id,spec_json])?;
        transaction.execute("INSERT INTO insight_admissions (admission_id,insight_id,spec_json,before_json,after_json,before_compat,after_compat,previous_evidence_json,actor,reason) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",params![spec.admission_id,spec.insight_id,spec_json,serde_json::to_string(&state)?,serde_json::to_string(&spec.revised)?,compat,after_compat,previous_evidence,actor,reason])?;
        let revision = transaction.last_insert_rowid();
        let before_fields = serde_json::to_value(&state)?;
        let after_fields = serde_json::to_value(&spec.revised)?;
        for (field, next) in after_fields.as_object().context("state object required")? {
            let previous = &before_fields[field];
            if previous != next {
                transaction.execute("INSERT INTO insight_revisions(insight_id,field_name,prev_value_sha256,new_value_sha256,actor,reason,operation,application_id) VALUES (?1,?2,?3,?4,?5,?6,'update',?7)",params![spec.insight_id,field,previous.as_str().map(sha256_hex).or_else(|| (!previous.is_null()).then(|| sha256_hex(&previous.to_string()))),sha256_hex(next.as_str().unwrap_or(&next.to_string())),actor,spec_json,CLI_APPLICATION_ID])?;
            }
        }
        for evidence in &spec.evidence_files {
            crate::artifact_paths::verified_file_bytes(
                repo_root,
                &evidence.path,
                &evidence.sha256,
            )?;
        }
        transaction.commit()?;
        Ok(revision)
    }
    pub(crate) fn overlay_insight_evidence(&self, rendered: String) -> Result<String> {
        let mut document: toml_edit::DocumentMut = rendered.parse()?;
        let mut expected: toml::Value = toml::from_str(&rendered)?;
        if let Some(insights) = document
            .get_mut("insight")
            .and_then(toml_edit::Item::as_array_of_tables_mut)
        {
            for (index, insight) in insights.iter_mut().enumerate() {
                let id = insight
                    .get("id")
                    .and_then(toml_edit::Item::as_str)
                    .context("insight ID required")?;
                let contract: Option<String> = self
                    .conn
                    .query_row(
                        "SELECT spec_json FROM insight_evidence WHERE insight_id=?1",
                        [id],
                        |row| row.get(0),
                    )
                    .optional()?;
                if let Some(contract) = contract {
                    let spec: InsightAdmissionSpec = serde_json::from_str(&contract)?;
                    spec.validate()?;
                    anyhow::ensure!(
                        self.insight_admission_state(id)? == spec.revised,
                        "insight contract drift for {id}"
                    );
                    let serialized = toml::to_string(&spec)?;
                    let fields: toml_edit::DocumentMut = serialized.parse()?;
                    let mut item = toml_edit::Item::Table(fields.into_table());
                    crate::claim_evidence::clear_transplanted_table_positions(&mut item);
                    insight.insert("evidence_admission", item);
                    expected["insight"][index]
                        .as_table_mut()
                        .context("insight table required")?
                        .insert("evidence_admission".into(), toml::Value::try_from(spec)?);
                }
            }
        }
        crate::claim_evidence::checked_semantic_render(&document, &expected)
    }
}

pub(crate) fn refuse_insight_admission_history_loss(connection: &Connection) -> Result<()> {
    let exists:bool=connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='insight_admissions')",[],|row|row.get(0))?;
    if exists {
        let count: i64 =
            connection.query_row("SELECT count(*) FROM insight_admissions", [], |row| {
                row.get(0)
            })?;
        anyhow::ensure!(
            count == 0,
            "refusing to discard canonical insight admission history"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_git(root: &Path, arguments: &[&str]) -> Result<()> {
        let mut command = std::process::Command::new("git");
        for (name, _) in std::env::vars_os() {
            if name.as_encoded_bytes().starts_with(b"GIT_") {
                command.env_remove(name);
            }
        }
        anyhow::ensure!(
            command
                .arg("-C")
                .arg(root)
                .args(arguments)
                .status()?
                .success(),
            "fixture git command failed"
        );
        Ok(())
    }
    #[test]
    fn admission_fixture_preserves_inherited_hook_repository() -> Result<()> {
        let (_store, _spec, root) = fixture()?;
        let paths = [".git/config", ".git/HEAD", ".git/index"];
        let before: Vec<_> = paths
            .iter()
            .map(|path| fs::read(root.join(path)))
            .collect::<std::io::Result<_>>()?;
        let result = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "insight_admission::tests::admission_roundtrip_replay_and_immutable_history",
            ])
            .env("GIT_DIR", root.join(".git"))
            .env("GIT_WORK_TREE", &root)
            .env("GIT_INDEX_FILE", root.join(".git/index"))
            .output()?;
        anyhow::ensure!(
            result.status.success(),
            "hook-environment child failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let after: Vec<_> = paths
            .iter()
            .map(|path| fs::read(root.join(path)))
            .collect::<std::io::Result<_>>()?;
        assert_eq!(before, after);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn fixture() -> Result<(ProvenanceStore, InsightAdmissionSpec, PathBuf)> {
        let root = std::env::temp_dir().join(format!(
            "insight-admission-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir_all(&root)?;
        fixture_git(&root, &["init", "--quiet"])?;
        fs::write(root.join("evidence.txt"), r#"{"lesson":"evidence"}"#)?;
        fixture_git(&root, &["add", "evidence.txt"])?;
        let store = ProvenanceStore::open(Path::new(":memory:"))?;
        store.conn.execute("INSERT INTO insights(id,title,status,claim_refs_json,compat_toml_text) VALUES ('I-1','old','open','[\"C-1\"]','extra = 42')",[])?;
        for claim in ["C-1", "C-2"] {
            store.conn.execute("INSERT INTO claims(id,statement,status,where_stated,last_verified,compat_toml_text) VALUES (?1,'statement','Verified','','','')",[claim])?;
        }
        store.conn.execute(
            "INSERT INTO claim_insight_refs(claim_id,insight_id) VALUES ('C-1','I-1')",
            [],
        )?;
        let expected = store.insight_admission_state("I-1")?;
        let mut revised = expected.clone();
        revised.title = "bounded lesson".into();
        revised.summary = Some("source-reviewed lesson".into());
        revised.claim_refs.push("C-2".into());
        let spec = InsightAdmissionSpec {
            admission_id: "admit-I1".into(),
            insight_id: "I-1".into(),
            expected,
            expected_previous_evidence_sha256: None,
            revised,
            evidence_files: vec![RetrievalEvidenceFile {
                path: "evidence.txt".into(),
                sha256: sha256_hex(r#"{"lesson":"evidence"}"#),
            }],
            predicates: vec![InsightPredicate {
                id: "lesson".into(),
                proposition: "bounded lesson".into(),
                kind: InsightPredicateKind::AssertedProposition,
                evidence_layer: EvidenceLayer::SourceProposition,
                outcome: InsightOutcome::Supported,
                execution_status: InsightExecutionStatus::SourceReview,
                boundary: "source review only".into(),
                evidence_anchors: vec!["evidence.txt".into()],
                evidence_bindings: vec![InsightEvidenceBinding {
                    path: "evidence.txt".into(),
                    json_pointer: "/lesson".into(),
                }],
            }],
            references: ["C-1", "C-2"]
                .into_iter()
                .map(|claim| InsightClaimRole {
                    claim_id: claim.into(),
                    predicate_id: "lesson".into(),
                    role: InsightReferenceRole::Qualifies,
                    provenance: "source review".into(),
                    boundary: "bounded".into(),
                })
                .collect(),
            disposition: "narrow".into(),
            falsifier: "contrary source text".into(),
            successor: "bounded successor".into(),
            residual: "physical validation".into(),
        };
        Ok((store, spec, root))
    }
    #[test]
    fn admission_roundtrip_replay_and_immutable_history() -> Result<()> {
        let (mut store, spec, root) = fixture()?;
        assert_eq!(
            ProvenanceStore::parse_insight_admission_spec(&toml::to_string(&spec)?)?,
            spec
        );
        let revision = store.admit_insight(&root, &spec, "tester", "source correction")?;
        assert_eq!(store.insight_admission_state("I-1")?, spec.revised);
        assert!(
            store
                .insight_update_summary("I-1", "untyped", "tester", Some("edit"))
                .unwrap_err()
                .to_string()
                .contains("typed admission")
        );
        assert!(
            store
                .insight_update_status_note("I-1", "untyped", "tester", Some("edit"))
                .unwrap_err()
                .to_string()
                .contains("typed admission")
        );
        assert_eq!(store.insight_admission_state("I-1")?, spec.revised);
        assert_eq!(
            store.admit_insight(&root, &spec, "tester", "replay")?,
            revision
        );
        assert_eq!(store.table_row_count("insight_admissions")?, 1);
        assert!(
            store
                .conn
                .execute("DELETE FROM insight_admissions", [])
                .is_err()
        );
        assert!(
            store
                .conn
                .execute("UPDATE insight_admissions SET reason='changed'", [])
                .is_err()
        );
        assert!(refuse_insight_admission_history_loss(&store.conn).is_err());
        let indexed: i64 = store.conn.query_row(
            "SELECT count(*) FROM insights_fts WHERE insights_fts MATCH 'bounded'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(indexed, 1);
        store.conn.execute(
            "INSERT INTO insights_fts(insights_fts, rank) VALUES('integrity-check', 1)",
            [],
        )?;
        let exported = store.render_control_plane_compat_outputs()?.insights;
        let value: toml::Value = toml::from_str(&exported)?;
        assert_eq!(
            value["insight"][0]["title"].as_str(),
            Some("bounded lesson")
        );
        assert_eq!(value["insight"][0]["extra"].as_integer(), Some(42));
        assert_eq!(
            value["insight"][0]["evidence_admission"]["predicates"][0]["outcome"].as_str(),
            Some("supported")
        );
        store
            .conn
            .execute("UPDATE insights SET title='drift' WHERE id='I-1'", [])?;
        assert!(
            store
                .admit_insight(&root, &spec, "tester", "replay")
                .is_err()
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn migration_repairs_populated_external_content_index_update() -> Result<()> {
        let (mut store, spec, root) = fixture()?;
        store.conn.execute_batch("DROP TRIGGER insights_fts_au; CREATE TRIGGER insights_fts_au AFTER UPDATE ON insights BEGIN DELETE FROM insights_fts WHERE rowid=old.rowid; INSERT INTO insights_fts(rowid,id,title,status) VALUES(new.rowid,new.id,new.title,new.status); END;")?;
        let indexed: i64 = store.conn.query_row(
            "SELECT count(*) FROM insights_fts WHERE insights_fts MATCH 'old'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(indexed, 1);
        store.conn.execute_batch(include_str!(
            "../../../db/migrations/0023_insight_admission.sql"
        ))?;
        store.admit_insight(&root, &spec, "tester", "migrated index correction")?;
        store.conn.execute(
            "INSERT INTO insights_fts(insights_fts,rank) VALUES('integrity-check',1)",
            [],
        )?;
        let old: i64 = store.conn.query_row(
            "SELECT count(*) FROM insights_fts WHERE insights_fts MATCH 'old'",
            [],
            |row| row.get(0),
        )?;
        let revised: i64 = store.conn.query_row(
            "SELECT count(*) FROM insights_fts WHERE insights_fts MATCH 'bounded'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!((old, revised), (0, 1));
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn admission_rejects_stale_missing_claim_and_role_coverage() -> Result<()> {
        let (mut store, spec, root) = fixture()?;
        let mut stale = spec.clone();
        stale.expected.title = "stale".into();
        assert!(
            store
                .admit_insight(&root, &stale, "tester", "correction")
                .unwrap_err()
                .to_string()
                .contains("stale insight admission prestate")
        );
        let mut missing = spec.clone();
        missing.revised.claim_refs.push("C-missing".into());
        let mut role = missing.references[0].clone();
        role.claim_id = "C-missing".into();
        missing.references.push(role);
        assert!(
            store
                .admit_insight(&root, &missing, "tester", "correction")
                .unwrap_err()
                .to_string()
                .contains("unknown claim C-missing")
        );
        let mut incomplete = spec.clone();
        incomplete.references.pop();
        assert!(incomplete.validate().is_err());
        let mut removed = spec.clone();
        removed.revised.claim_refs.remove(0);
        assert!(removed.validate().is_err());
        assert_eq!(store.table_row_count("insight_admissions")?, 0);
        assert_eq!(store.insight_admission_state("I-1")?, spec.expected);
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn admission_rejects_stale_contract_at_identical_public_state() -> Result<()> {
        let (mut store, spec, root) = fixture()?;
        store.admit_insight(&root, &spec, "tester", "first admission")?;
        let mut correction = spec.clone();
        correction.admission_id = "second-admission".into();
        correction.expected = spec.revised.clone();
        correction.residual = "revised research obligation".into();
        let error = store
            .admit_insight(&root, &correction, "tester", "contract-only correction")
            .unwrap_err();
        assert!(error.to_string().contains("stale prior insight evidence"));
        correction.expected_previous_evidence_sha256 =
            Some(sha256_hex(&serde_json::to_string(&spec)?));
        store.admit_insight(&root, &correction, "tester", "contract-only correction")?;
        assert_eq!(store.insight_admission_state("I-1")?, spec.revised);
        let mut stale = correction.clone();
        stale.admission_id = "stale-third-admission".into();
        assert!(
            store
                .admit_insight(&root, &stale, "tester", "stale contract")
                .unwrap_err()
                .to_string()
                .contains("stale prior insight evidence")
        );
        assert_eq!(store.table_row_count("insight_admissions")?, 2);
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn admission_rejects_unbound_and_unresolved_predicate_evidence() -> Result<()> {
        let (mut store, mut spec, root) = fixture()?;
        spec.predicates[0].evidence_bindings[0].path = "unrelated.json".into();
        assert!(
            spec.validate()
                .unwrap_err()
                .to_string()
                .contains("lacks hash-bound file")
        );
        spec.predicates[0].evidence_bindings[0].path = "evidence.txt".into();
        spec.predicates[0].evidence_bindings[0].json_pointer = "/missing".into();
        assert!(
            store
                .admit_insight(&root, &spec, "tester", "bad pointer")
                .unwrap_err()
                .to_string()
                .contains("unresolved predicate JSON pointer")
        );
        assert_eq!(store.table_row_count("insight_admissions")?, 0);
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn admission_rolls_back_and_rejects_changed_evidence() -> Result<()> {
        let (mut store, spec, root) = fixture()?;
        store.conn.execute_batch("CREATE TRIGGER fail_admission BEFORE INSERT ON insight_admissions BEGIN SELECT RAISE(ABORT,'injected failure'); END;")?;
        assert!(
            store
                .admit_insight(&root, &spec, "tester", "correction")
                .unwrap_err()
                .to_string()
                .contains("injected failure")
        );
        assert_eq!(store.insight_admission_state("I-1")?, spec.expected);
        assert_eq!(store.table_row_count("insight_evidence")?, 0);
        assert_eq!(store.table_row_count("claim_insight_refs")?, 1);
        store.conn.execute_batch("DROP TRIGGER fail_admission; CREATE TRIGGER fail_revision BEFORE INSERT ON insight_revisions BEGIN SELECT RAISE(ABORT,'injected revision failure'); END;")?;
        assert!(
            store
                .admit_insight(&root, &spec, "tester", "revision rollback")
                .unwrap_err()
                .to_string()
                .contains("injected revision failure")
        );
        assert_eq!(store.insight_admission_state("I-1")?, spec.expected);
        assert_eq!(store.table_row_count("insight_admissions")?, 0);
        assert_eq!(store.table_row_count("insight_evidence")?, 0);
        assert_eq!(store.table_row_count("insight_revisions")?, 0);
        assert_eq!(store.table_row_count("claim_insight_refs")?, 1);
        store.conn.execute_batch("DROP TRIGGER fail_revision")?;
        fs::write(root.join("evidence.txt"), "changed")?;
        assert!(
            store
                .admit_insight(&root, &spec, "tester", "correction")
                .unwrap_err()
                .to_string()
                .contains("SHA256 mismatch")
        );
        assert_eq!(store.table_row_count("insight_admissions")?, 0);
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
