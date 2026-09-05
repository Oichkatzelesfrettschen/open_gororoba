//! Typed claim transitions and successor-claim persistence.
//!
//! Experiment verdicts are observations about an experiment. Claim statuses
//! are canonical repository state. This module keeps those vocabularies and
//! their histories separate while applying one transition as one transaction.

use anyhow::{Context, Result, bail};
use chrono::DateTime;
use rusqlite::{OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    CLI_APPLICATION_ID, ProvenanceStore, migrations::CANONICAL_CLAIM_STATUSES, scalar_count,
    sha256_hex,
};

pub const CLAIM_EXPERIMENT_VERDICTS: &[&str] = &[
    "Falsifies",
    "MethodologyInvalid",
    "Inconclusive",
    "SurvivesChallenge",
    "Replicates",
];

pub const CLAIM_RELATION_KINDS: &[&str] = &[
    "source_split",
    "implementation_split",
    "narrows",
    "refines",
    "supersedes",
];

#[derive(Debug, Clone, Copy)]
pub struct ClaimTransitionCompatPaths<'a> {
    pub events: &'a std::path::Path,
    pub relations: &'a std::path::Path,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ClaimTransitionSpec {
    pub transition: ClaimTransitionRequest,
    #[serde(rename = "successor", default)]
    pub successors: Vec<SuccessorClaimSpec>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ClaimTransitionRequest {
    pub transition_key: String,
    pub source_claim_id: String,
    pub expected_prior_status: String,
    pub experiment_verdict: String,
    #[serde(default)]
    pub evidence_artifact_ids: Vec<String>,
    #[serde(default)]
    pub experiment_ids: Vec<String>,
    pub exercised_falsifier: String,
    pub proposed_claim_status: String,
    pub rationale: String,
    #[serde(default)]
    pub unresolved_assumptions: Vec<String>,
    pub actor: String,
    pub reason: String,
    pub transition_timestamp: String,
    pub expected_source_state_sha256: String,
    pub expected_claim_id_max: i64,
    #[serde(default)]
    pub reserved_claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SuccessorClaimSpec {
    pub proposal_key: String,
    pub statement: String,
    pub initial_status: String,
    pub source_or_implementation_boundary: String,
    pub required_falsifier: String,
    #[serde(default)]
    pub where_stated: Vec<String>,
    #[serde(default)]
    pub evidence_artifact_ids: Vec<String>,
    pub predecessor_relation_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AllocatedSuccessor {
    pub proposal_key: String,
    pub canonical_claim_id: String,
    pub relation_kind: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClaimTransitionPlan {
    pub transition_key: String,
    pub transition_spec_sha256: String,
    pub source_claim_id: String,
    pub current_status: String,
    pub expected_prior_status: String,
    pub proposed_claim_status: String,
    pub experiment_verdict: String,
    pub evidence_artifact_ids: Vec<String>,
    pub experiment_ids: Vec<String>,
    pub allocated_successors: Vec<AllocatedSuccessor>,
    pub skipped_reserved_claim_ids: Vec<String>,
    pub status_count_delta: BTreeMap<String, i64>,
    pub exact_replay: bool,
    pub replay_event_id: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClaimTransitionApplyResult {
    pub transition_key: String,
    pub transition_spec_sha256: String,
    pub event_id: Option<i64>,
    pub allocated_successors: Vec<AllocatedSuccessor>,
    pub skipped_reserved_claim_ids: Vec<String>,
    pub exact_replay: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClaimTransitionEventView {
    pub event_id: i64,
    pub transition_key: String,
    pub source_claim_id: String,
    pub expected_prior_status: String,
    pub experiment_verdict: String,
    pub proposed_claim_status: String,
    pub current_status: String,
    pub exercised_falsifier: String,
    pub rationale: String,
    pub unresolved_assumptions: Vec<String>,
    pub evidence_artifact_ids: Vec<String>,
    pub experiment_ids: Vec<String>,
    pub actor: String,
    pub reason: String,
    pub transition_timestamp: String,
    pub transition_spec_sha256: String,
    pub successors: Vec<AllocatedSuccessor>,
}

#[derive(Clone, Debug)]
struct ExistingEvent {
    event_id: i64,
    transition_key: String,
    source_claim_id: String,
    expected_prior_status: String,
    experiment_verdict: String,
    proposed_claim_status: String,
    exercised_falsifier: String,
    rationale: String,
    actor: String,
    reason: String,
    transition_timestamp: String,
    transition_spec_sha256: String,
}

impl ProvenanceStore {
    pub fn export_claim_transition_compat_paths(
        &mut self,
        repo_root: &std::path::Path,
        paths: ClaimTransitionCompatPaths<'_>,
    ) -> Result<()> {
        let events = self.list_claim_transition_events()?;
        let relations = self.list_claim_relations()?;
        let event_body = render_transition_events_compat(&events)?;
        let relation_body = render_claim_relations_compat(&relations)?;
        crate::compat_render::write_text(paths.events, &event_body)?;
        crate::compat_render::write_text(paths.relations, &relation_body)?;
        self.record_control_plane_run(
            "export_claim_transitions",
            &serde_json::json!({
                "events": repo_root
                    .join(paths.events)
                    .strip_prefix(repo_root)
                    .unwrap_or(paths.events)
                    .display()
                    .to_string(),
                "relations": repo_root
                    .join(paths.relations)
                    .strip_prefix(repo_root)
                    .unwrap_or(paths.relations)
                    .display()
                    .to_string(),
                "event_count": events.len(),
                "relation_count": relations.len(),
            })
            .to_string(),
        )?;
        Ok(())
    }

    pub fn claim_transition_compat_texts(&self) -> Result<(String, String)> {
        let events = self.list_claim_transition_events()?;
        let relations = self.list_claim_relations()?;
        Ok((
            render_transition_events_compat(&events)?,
            render_claim_relations_compat(&relations)?,
        ))
    }

    pub fn list_claim_relations(&self) -> Result<Vec<ClaimRelationView>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, predecessor_claim_id, successor_claim_id, relation_kind,
                    transition_event_id
             FROM claim_relations ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ClaimRelationView {
                relation_id: row.get(0)?,
                predecessor_claim_id: row.get(1)?,
                successor_claim_id: row.get(2)?,
                relation_kind: row.get(3)?,
                transition_event_id: row.get(4)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn verify_claim_transition_invariants(&self) -> Result<()> {
        let mut failures = Vec::new();
        self.verify_transition_event_history(&mut failures)?;
        self.verify_transition_relation_integrity(&mut failures)?;
        self.verify_transition_reference_integrity(&mut failures)?;

        if !failures.is_empty() {
            bail!(
                "claim transition invariants failed:\n- {}",
                failures.join("\n- ")
            );
        }
        Ok(())
    }

    fn verify_transition_event_history(&self, failures: &mut Vec<String>) -> Result<()> {
        let mut event_stmt = self.conn.prepare(
            "SELECT id, source_claim_id, expected_prior_status, proposed_claim_status
             FROM claim_transition_events ORDER BY id",
        )?;
        let event_rows = event_stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut by_source = BTreeMap::<String, Vec<(i64, String, String)>>::new();
        for row in event_rows {
            let (event_id, source_claim_id, expected_prior, proposed) = row?;
            let source_exists = self.conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM claims WHERE id = ?1)",
                params![source_claim_id],
                |value| value.get::<_, i64>(0),
            )?;
            if source_exists == 0 {
                failures.push(format!(
                    "transition event {event_id} source claim is missing"
                ));
            }
            by_source.entry(source_claim_id).or_default().push((
                event_id,
                expected_prior,
                proposed,
            ));
        }

        for (source_claim_id, mut events) in by_source {
            events.sort_by_key(|event| event.0);
            let Some(current_status) = self
                .claim_by_id(&source_claim_id)?
                .map(|claim| claim.status)
            else {
                continue;
            };
            let mut observed_status = current_status;
            for (event_id, expected_prior, proposed) in events.into_iter().rev() {
                if observed_status != proposed {
                    failures.push(format!(
                        "transition event {event_id} proposed {proposed} but historical current status is {observed_status}"
                    ));
                }
                observed_status = expected_prior;
            }
        }
        Ok(())
    }

    fn verify_transition_relation_integrity(&self, failures: &mut Vec<String>) -> Result<()> {
        let relation_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM claim_relations")?;
        let resolved_relation_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM claim_relations AS relation
             JOIN claims AS predecessor ON predecessor.id = relation.predecessor_claim_id
             JOIN claims AS successor ON successor.id = relation.successor_claim_id
             JOIN claim_transition_events AS event ON event.id = relation.transition_event_id",
        )?;
        if relation_count != resolved_relation_count {
            failures.push(format!(
                "claim relation resolution count {resolved_relation_count} differs from {relation_count}"
            ));
        }
        if let Err(error) = validate_relation_graph(&self.conn, "", &[]) {
            failures.push(error.to_string());
        }

        let invalid_relation_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM claim_relations
             WHERE relation_kind NOT IN (
                 'source_split', 'implementation_split', 'narrows', 'refines', 'supersedes'
             )",
        )?;
        if invalid_relation_count != 0 {
            failures.push(format!(
                "claim relation table contains {invalid_relation_count} invalid relation kinds"
            ));
        }
        Ok(())
    }

    fn verify_transition_reference_integrity(&self, failures: &mut Vec<String>) -> Result<()> {
        let missing_successor_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM claim_transition_successors AS successor
             LEFT JOIN claims ON claims.id = successor.successor_claim_id
             LEFT JOIN claim_transition_events AS event ON event.id = successor.transition_event_id
             WHERE claims.id IS NULL OR event.id IS NULL",
        )?;
        if missing_successor_count != 0 {
            failures.push(format!(
                "{missing_successor_count} successor rows have unresolved claim or event references"
            ));
        }

        let missing_evidence_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM claim_transition_evidence AS evidence
             LEFT JOIN artifacts ON artifacts.id = evidence.artifact_id
             WHERE artifacts.id IS NULL",
        )? + scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM claim_transition_successor_evidence AS evidence
             LEFT JOIN artifacts ON artifacts.id = evidence.artifact_id
             WHERE artifacts.id IS NULL",
        )?;
        if missing_evidence_count != 0 {
            failures.push(format!(
                "{missing_evidence_count} transition evidence references are unresolved"
            ));
        }
        Ok(())
    }

    pub fn parse_claim_transition_spec(raw: &str) -> Result<ClaimTransitionSpec> {
        if raw.is_empty() {
            bail!("claim transition specification is empty");
        }
        if !raw.is_ascii() {
            bail!("claim transition specification must contain ASCII only");
        }
        toml::from_str(raw).context("parse claim transition TOML specification")
    }

    pub fn claim_transition_source_state_sha256(&self, id: &str) -> Result<String> {
        let state = self.claim_transition_source_state(id)?;
        Ok(sha256_claim_state(&state))
    }

    pub fn claim_transition_expected_claim_id_max(&self) -> Result<i64> {
        self.max_numeric_claim_id()
    }

    pub fn plan_claim_transition(
        &self,
        spec: &ClaimTransitionSpec,
        raw_spec: &str,
    ) -> Result<ClaimTransitionPlan> {
        let spec_sha256 = crate::sha256_hex(raw_spec);
        validate_spec(spec)?;
        if let Some(existing) = self.transition_event_by_key(&spec.transition.transition_key)? {
            if existing.transition_spec_sha256 != spec_sha256 {
                bail!(
                    "transition key {} already exists with different content hash",
                    spec.transition.transition_key
                );
            }
            return self.plan_for_replay(spec, &existing);
        }

        let current = self.claim_transition_source_state(&spec.transition.source_claim_id)?;
        validate_source_state(spec, &current)?;
        let max_claim_id = self.max_numeric_claim_id()?;
        if max_claim_id != spec.transition.expected_claim_id_max {
            bail!(
                "expected claim id max {} is stale; canonical max is {}",
                spec.transition.expected_claim_id_max,
                max_claim_id
            );
        }
        let reserved_ids =
            reserved_numeric_claim_ids(&self.conn, &spec.transition.reserved_claim_ids)?;
        let (allocated_successors, skipped_reserved_claim_ids) =
            allocate_successors_with_reservations(&spec.successors, max_claim_id, &reserved_ids)?;
        validate_successor_references(self, spec, &allocated_successors)?;
        Ok(ClaimTransitionPlan {
            transition_key: spec.transition.transition_key.clone(),
            transition_spec_sha256: spec_sha256,
            source_claim_id: spec.transition.source_claim_id.clone(),
            current_status: current.status,
            expected_prior_status: spec.transition.expected_prior_status.clone(),
            proposed_claim_status: spec.transition.proposed_claim_status.clone(),
            experiment_verdict: spec.transition.experiment_verdict.clone(),
            evidence_artifact_ids: spec.transition.evidence_artifact_ids.clone(),
            experiment_ids: spec.transition.experiment_ids.clone(),
            allocated_successors,
            skipped_reserved_claim_ids,
            status_count_delta: status_count_delta_with_successors(
                &spec.transition.expected_prior_status,
                &spec.transition.proposed_claim_status,
                &spec.successors,
            ),
            exact_replay: false,
            replay_event_id: None,
        })
    }

    pub fn apply_claim_transition(
        &mut self,
        spec: &ClaimTransitionSpec,
        raw_spec: &str,
    ) -> Result<ClaimTransitionApplyResult> {
        let spec_sha256 = crate::sha256_hex(raw_spec);
        validate_spec(spec)?;
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let result = match transition_event_by_key_on_conn(&tx, &spec.transition.transition_key)? {
            Some(existing) => apply_transition_replay(&tx, existing, spec, &spec_sha256)?,
            None => apply_new_transition(&tx, spec, &spec_sha256)?,
        };
        tx.execute("DELETE FROM claim_status_write_context WHERE id = 1", [])?;
        tx.commit()?;
        Ok(result)
    }

    pub fn claim_transition_by_key(&self, key: &str) -> Result<Option<ClaimTransitionEventView>> {
        let Some(event) = self.transition_event_by_key(key)? else {
            return Ok(None);
        };
        let current_status = self
            .claim_by_id(&event.source_claim_id)?
            .map(|claim| claim.status)
            .context("transition source claim disappeared")?;
        let unresolved_assumptions = self.load_transition_assumptions(event.event_id)?;
        let evidence_artifact_ids = self.load_transition_ids(
            "SELECT artifact_id FROM claim_transition_evidence
             WHERE transition_event_id = ?1 ORDER BY artifact_id",
            event.event_id,
        )?;
        let experiment_ids = self.load_transition_ids(
            "SELECT experiment_id FROM claim_transition_experiments
             WHERE transition_event_id = ?1 ORDER BY experiment_id",
            event.event_id,
        )?;
        Ok(Some(ClaimTransitionEventView {
            event_id: event.event_id,
            transition_key: event.transition_key,
            source_claim_id: event.source_claim_id,
            expected_prior_status: event.expected_prior_status,
            experiment_verdict: event.experiment_verdict,
            proposed_claim_status: event.proposed_claim_status,
            current_status,
            exercised_falsifier: event.exercised_falsifier,
            rationale: event.rationale,
            unresolved_assumptions,
            evidence_artifact_ids,
            experiment_ids,
            actor: event.actor,
            reason: event.reason,
            transition_timestamp: event.transition_timestamp,
            transition_spec_sha256: event.transition_spec_sha256,
            successors: successors_for_event(&self.conn, event.event_id)?,
        }))
    }

    pub fn list_claim_transition_events(&self) -> Result<Vec<ClaimTransitionEventView>> {
        let keys = {
            let mut stmt = self
                .conn
                .prepare("SELECT transition_key FROM claim_transition_events ORDER BY id")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        keys.into_iter()
            .map(|key| {
                self.claim_transition_by_key(&key)?
                    .context("transition key disappeared during list")
            })
            .collect()
    }

    fn transition_event_by_key(&self, key: &str) -> Result<Option<ExistingEvent>> {
        transition_event_by_key_on_conn(&self.conn, key)
    }

    fn claim_transition_source_state(&self, id: &str) -> Result<ClaimSourceState> {
        claim_transition_source_state_on_conn(&self.conn, id)
    }

    fn max_numeric_claim_id(&self) -> Result<i64> {
        max_numeric_claim_id_on_conn(&self.conn)
    }

    fn load_transition_assumptions(&self, event_id: i64) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT assumption FROM claim_transition_assumptions
             WHERE transition_event_id = ?1 ORDER BY ordinal",
        )?;
        let rows = stmt.query_map(params![event_id], |row| row.get::<_, String>(0))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    fn load_transition_ids(&self, sql: &str, event_id: i64) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(params![event_id], |row| row.get::<_, String>(0))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    fn plan_for_replay(
        &self,
        spec: &ClaimTransitionSpec,
        existing: &ExistingEvent,
    ) -> Result<ClaimTransitionPlan> {
        if existing.source_claim_id != spec.transition.source_claim_id
            || existing.expected_prior_status != spec.transition.expected_prior_status
            || existing.experiment_verdict != spec.transition.experiment_verdict
            || existing.proposed_claim_status != spec.transition.proposed_claim_status
        {
            bail!(
                "transition key {} does not match its stored event",
                existing.transition_key
            );
        }
        let allocated_successors = successors_for_event(&self.conn, existing.event_id)?;
        let current_status = self
            .claim_by_id(&existing.source_claim_id)?
            .map(|claim| claim.status)
            .context("transition source claim disappeared")?;
        let successor_statuses = successor_statuses_for_event(&self.conn, existing.event_id)?;
        Ok(ClaimTransitionPlan {
            transition_key: existing.transition_key.clone(),
            transition_spec_sha256: existing.transition_spec_sha256.clone(),
            source_claim_id: existing.source_claim_id.clone(),
            current_status,
            expected_prior_status: existing.expected_prior_status.clone(),
            proposed_claim_status: existing.proposed_claim_status.clone(),
            experiment_verdict: existing.experiment_verdict.clone(),
            evidence_artifact_ids: self.load_transition_ids(
                "SELECT artifact_id FROM claim_transition_evidence
                 WHERE transition_event_id = ?1 ORDER BY artifact_id",
                existing.event_id,
            )?,
            experiment_ids: self.load_transition_ids(
                "SELECT experiment_id FROM claim_transition_experiments
                 WHERE transition_event_id = ?1 ORDER BY experiment_id",
                existing.event_id,
            )?,
            allocated_successors,
            skipped_reserved_claim_ids: Vec::new(),
            status_count_delta: status_count_delta_from_statuses(
                &existing.expected_prior_status,
                &existing.proposed_claim_status,
                &successor_statuses,
            ),
            exact_replay: true,
            replay_event_id: Some(existing.event_id),
        })
    }
}

fn apply_transition_replay(
    tx: &rusqlite::Transaction<'_>,
    existing: ExistingEvent,
    spec: &ClaimTransitionSpec,
    spec_sha256: &str,
) -> Result<ClaimTransitionApplyResult> {
    if existing.transition_spec_sha256 != spec_sha256 {
        bail!(
            "transition key {} already exists with different content hash",
            spec.transition.transition_key
        );
    }
    Ok(ClaimTransitionApplyResult {
        transition_key: existing.transition_key,
        transition_spec_sha256: existing.transition_spec_sha256,
        event_id: Some(existing.event_id),
        allocated_successors: successors_for_event(tx, existing.event_id)?,
        skipped_reserved_claim_ids: Vec::new(),
        exact_replay: true,
    })
}

fn apply_new_transition(
    tx: &rusqlite::Transaction<'_>,
    spec: &ClaimTransitionSpec,
    spec_sha256: &str,
) -> Result<ClaimTransitionApplyResult> {
    let current = claim_transition_source_state_on_conn(tx, &spec.transition.source_claim_id)?;
    validate_source_state(spec, &current)?;
    let (allocated_successors, skipped_reserved_claim_ids) =
        allocate_transition_successors(tx, spec)?;
    let event_id = insert_transition_event(tx, spec, spec_sha256)?;
    set_transition_write_context(tx, event_id, spec)?;
    insert_transition_metadata(tx, event_id, spec)?;
    insert_successor_claims(tx, event_id, spec, &allocated_successors)?;
    update_source_claim_status(tx, spec, &current.status)?;
    Ok(ClaimTransitionApplyResult {
        transition_key: spec.transition.transition_key.clone(),
        transition_spec_sha256: spec_sha256.to_string(),
        event_id: Some(event_id),
        allocated_successors,
        skipped_reserved_claim_ids,
        exact_replay: false,
    })
}

fn allocate_transition_successors(
    tx: &rusqlite::Transaction<'_>,
    spec: &ClaimTransitionSpec,
) -> Result<(Vec<AllocatedSuccessor>, Vec<String>)> {
    let max_claim_id = max_numeric_claim_id_on_conn(tx)?;
    if max_claim_id != spec.transition.expected_claim_id_max {
        bail!(
            "expected claim id max {} is stale; canonical max is {}",
            spec.transition.expected_claim_id_max,
            max_claim_id
        );
    }
    let reserved_ids = reserved_numeric_claim_ids(tx, &spec.transition.reserved_claim_ids)?;
    let (allocated_successors, skipped_reserved_claim_ids) =
        allocate_successors_with_reservations(&spec.successors, max_claim_id, &reserved_ids)?;
    validate_successor_references_on_conn(tx, spec, &allocated_successors)?;
    Ok((allocated_successors, skipped_reserved_claim_ids))
}

fn insert_transition_event(
    tx: &rusqlite::Transaction<'_>,
    spec: &ClaimTransitionSpec,
    spec_sha256: &str,
) -> Result<i64> {
    tx.execute(
        "INSERT INTO claim_transition_events (
            transition_key, source_claim_id, expected_prior_status,
            experiment_verdict, proposed_claim_status, exercised_falsifier,
            rationale, actor, reason, transition_ts_utc,
            transition_spec_sha256, expected_source_state_sha256,
            expected_claim_id_max
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            spec.transition.transition_key,
            spec.transition.source_claim_id,
            spec.transition.expected_prior_status,
            spec.transition.experiment_verdict,
            spec.transition.proposed_claim_status,
            spec.transition.exercised_falsifier,
            spec.transition.rationale,
            spec.transition.actor,
            spec.transition.reason,
            spec.transition.transition_timestamp,
            spec_sha256,
            spec.transition.expected_source_state_sha256,
            spec.transition.expected_claim_id_max,
        ],
    )?;
    Ok(tx.last_insert_rowid())
}

fn set_transition_write_context(
    tx: &rusqlite::Transaction<'_>,
    event_id: i64,
    spec: &ClaimTransitionSpec,
) -> Result<()> {
    tx.execute(
        "INSERT INTO claim_status_write_context (
             id, mode, transition_event_id, source_claim_id, proposed_status
         ) VALUES (1, 'transition_apply', ?1, ?2, ?3)
         ON CONFLICT(id) DO UPDATE SET
             mode = excluded.mode,
             transition_event_id = excluded.transition_event_id,
             source_claim_id = excluded.source_claim_id,
             proposed_status = excluded.proposed_status",
        params![
            event_id,
            spec.transition.source_claim_id,
            spec.transition.proposed_claim_status,
        ],
    )?;
    Ok(())
}

fn insert_transition_metadata(
    tx: &rusqlite::Transaction<'_>,
    event_id: i64,
    spec: &ClaimTransitionSpec,
) -> Result<()> {
    for artifact_id in &spec.transition.evidence_artifact_ids {
        tx.execute(
            "INSERT INTO claim_transition_evidence (transition_event_id, artifact_id)
             VALUES (?1, ?2)",
            params![event_id, artifact_id],
        )?;
    }
    for experiment_id in &spec.transition.experiment_ids {
        tx.execute(
            "INSERT INTO claim_transition_experiments (transition_event_id, experiment_id)
             VALUES (?1, ?2)",
            params![event_id, experiment_id],
        )?;
    }
    for (ordinal, assumption) in spec.transition.unresolved_assumptions.iter().enumerate() {
        tx.execute(
            "INSERT INTO claim_transition_assumptions (transition_event_id, ordinal, assumption)
             VALUES (?1, ?2, ?3)",
            params![event_id, ordinal as i64, assumption],
        )?;
    }
    Ok(())
}

fn insert_successor_claims(
    tx: &rusqlite::Transaction<'_>,
    event_id: i64,
    spec: &ClaimTransitionSpec,
    allocated_successors: &[AllocatedSuccessor],
) -> Result<()> {
    let source_date = spec
        .transition
        .transition_timestamp
        .get(..10)
        .context("transition timestamp must contain a YYYY-MM-DD date")?;
    let mut ordered_successors = spec.successors.iter().collect::<Vec<_>>();
    ordered_successors.sort_by(|left, right| left.proposal_key.cmp(&right.proposal_key));
    for (successor, allocated) in ordered_successors
        .into_iter()
        .zip(allocated_successors.iter())
    {
        insert_successor_claim(
            tx,
            event_id,
            successor,
            allocated,
            source_date,
            &spec.transition,
        )?;
    }
    Ok(())
}

fn insert_successor_claim(
    tx: &rusqlite::Transaction<'_>,
    event_id: i64,
    successor: &SuccessorClaimSpec,
    allocated: &AllocatedSuccessor,
    source_date: &str,
    transition: &ClaimTransitionRequest,
) -> Result<()> {
    let where_stated = successor.where_stated.join("; ");
    let compat_toml_text = render_new_claim_compat_text(
        &allocated.canonical_claim_id,
        &successor.statement,
        &successor.initial_status,
        &where_stated,
        source_date,
    );
    tx.execute(
        "INSERT INTO claims (
            id, statement, status, where_stated, last_verified,
            formal_proof, status_note, compat_toml_text
         ) VALUES (?1, ?2, ?3, ?4, ?5, NULL, NULL, ?6)",
        params![
            allocated.canonical_claim_id,
            successor.statement,
            successor.initial_status,
            where_stated,
            source_date,
            compat_toml_text,
        ],
    )?;
    tx.execute(
        "INSERT INTO claim_revisions (
            claim_id, field_name, prev_value_sha256, new_value_sha256,
            actor, reason, operation, application_id
         ) VALUES (?1, 'claim', NULL, ?2, ?3, ?4, 'create', ?5)",
        params![
            allocated.canonical_claim_id,
            sha256_claim_state(&ClaimSourceState {
                id: allocated.canonical_claim_id.clone(),
                statement: successor.statement.clone(),
                status: successor.initial_status.clone(),
                where_stated: where_stated.clone(),
                last_verified: source_date.to_string(),
                formal_proof: None,
                status_note: None,
                compat_toml_text: compat_toml_text.clone(),
                evidence_spec_json: None,
            }),
            transition.actor,
            transition.reason,
            CLI_APPLICATION_ID,
        ],
    )?;
    tx.execute(
        "INSERT INTO claim_transition_successors (
            transition_event_id, proposal_key, successor_claim_id,
            statement, initial_status, source_or_implementation_boundary,
            required_falsifier, predecessor_relation_kind
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            event_id,
            successor.proposal_key,
            allocated.canonical_claim_id,
            successor.statement,
            successor.initial_status,
            successor.source_or_implementation_boundary,
            successor.required_falsifier,
            successor.predecessor_relation_kind,
        ],
    )?;
    let successor_id = tx.last_insert_rowid();
    insert_successor_references(tx, successor_id, successor)?;
    tx.execute(
        "INSERT INTO claim_relations (
            predecessor_claim_id, successor_claim_id, relation_kind,
            transition_event_id
         ) VALUES (?1, ?2, ?3, ?4)",
        params![
            transition.source_claim_id,
            allocated.canonical_claim_id,
            successor.predecessor_relation_kind,
            event_id,
        ],
    )?;
    Ok(())
}

fn insert_successor_references(
    tx: &rusqlite::Transaction<'_>,
    successor_id: i64,
    successor: &SuccessorClaimSpec,
) -> Result<()> {
    for (ordinal, reference) in successor.where_stated.iter().enumerate() {
        tx.execute(
            "INSERT INTO claim_transition_successor_where_stated
             (successor_id, ordinal, reference) VALUES (?1, ?2, ?3)",
            params![successor_id, ordinal as i64, reference],
        )?;
    }
    for artifact_id in &successor.evidence_artifact_ids {
        tx.execute(
            "INSERT INTO claim_transition_successor_evidence
             (successor_id, artifact_id) VALUES (?1, ?2)",
            params![successor_id, artifact_id],
        )?;
    }
    Ok(())
}

fn update_source_claim_status(
    tx: &rusqlite::Transaction<'_>,
    spec: &ClaimTransitionSpec,
    previous_status: &str,
) -> Result<()> {
    tx.execute(
        "UPDATE claims SET status = ?2 WHERE id = ?1",
        params![
            spec.transition.source_claim_id,
            spec.transition.proposed_claim_status
        ],
    )?;
    tx.execute(
        "INSERT INTO claim_revisions (
            claim_id, field_name, prev_value_sha256, new_value_sha256,
            actor, reason, operation, application_id
         ) VALUES (?1, 'status', ?2, ?3, ?4, ?5, 'update', ?6)",
        params![
            spec.transition.source_claim_id,
            sha256_hex(previous_status),
            sha256_hex(&spec.transition.proposed_claim_status),
            spec.transition.actor,
            spec.transition.reason,
            CLI_APPLICATION_ID,
        ],
    )?;
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub struct ClaimRelationView {
    pub relation_id: i64,
    pub predecessor_claim_id: String,
    pub successor_claim_id: String,
    pub relation_kind: String,
    pub transition_event_id: i64,
}

#[derive(Serialize)]
struct TransitionCompatFile<'a> {
    meta: TransitionCompatMeta,
    #[serde(rename = "transition")]
    transitions: &'a [ClaimTransitionEventView],
}

#[derive(Serialize)]
struct TransitionCompatMeta {
    surface: &'static str,
    event_count: usize,
    history_policy: &'static str,
}

#[derive(Serialize)]
struct RelationCompatFile<'a> {
    meta: RelationCompatMeta,
    #[serde(rename = "relation")]
    relations: &'a [ClaimRelationView],
}

#[derive(Serialize)]
struct RelationCompatMeta {
    surface: &'static str,
    relation_count: usize,
    relation_kinds: &'static [&'static str],
}

fn render_transition_events_compat(events: &[ClaimTransitionEventView]) -> Result<String> {
    let body = toml::to_string(&TransitionCompatFile {
        meta: TransitionCompatMeta {
            surface: "canonical claim transition events",
            event_count: events.len(),
            history_policy: "append-only events retain prior status, verdict, evidence, assumptions, and successor IDs",
        },
        transitions: events,
    })?;
    Ok(format!(
        "{}{}",
        crate::compat_render::compat_toml_export_header("claim_transitions").join("\n"),
        body
    ))
}

fn render_claim_relations_compat(relations: &[ClaimRelationView]) -> Result<String> {
    let body = toml::to_string(&RelationCompatFile {
        meta: RelationCompatMeta {
            surface: "typed predecessor and successor claim relations",
            relation_count: relations.len(),
            relation_kinds: CLAIM_RELATION_KINDS,
        },
        relations,
    })?;
    Ok(format!(
        "{}{}",
        crate::compat_render::compat_toml_export_header("claim_relations").join("\n"),
        body
    ))
}

#[derive(Clone, Debug)]
struct ClaimSourceState {
    id: String,
    status: String,
    statement: String,
    where_stated: String,
    last_verified: String,
    formal_proof: Option<String>,
    status_note: Option<String>,
    compat_toml_text: String,
    evidence_spec_json: Option<String>,
}

fn claim_transition_source_state_on_conn(
    conn: &rusqlite::Connection,
    id: &str,
) -> Result<ClaimSourceState> {
    conn.query_row(
        "SELECT id, status, statement, where_stated, last_verified, formal_proof,
                status_note, compat_toml_text,
                (SELECT spec_json FROM claim_evidence WHERE claim_id = claims.id)
         FROM claims WHERE id = ?1",
        params![id],
        |row| {
            Ok(ClaimSourceState {
                id: row.get(0)?,
                status: row.get(1)?,
                statement: row.get(2)?,
                where_stated: row.get(3)?,
                last_verified: row.get(4)?,
                formal_proof: row.get(5)?,
                status_note: row.get(6)?,
                compat_toml_text: row.get(7)?,
                evidence_spec_json: row.get(8)?,
            })
        },
    )
    .optional()?
    .context("source claim not found in canonical DB")
}

fn transition_event_by_key_on_conn(
    conn: &rusqlite::Connection,
    key: &str,
) -> Result<Option<ExistingEvent>> {
    conn.query_row(
        "SELECT id, transition_key, source_claim_id, expected_prior_status,
                experiment_verdict, proposed_claim_status, exercised_falsifier,
                rationale, actor, reason, transition_ts_utc,
                transition_spec_sha256
         FROM claim_transition_events WHERE transition_key = ?1",
        params![key],
        existing_event_from_row,
    )
    .optional()
    .map_err(Into::into)
}

fn existing_event_from_row(row: &Row<'_>) -> rusqlite::Result<ExistingEvent> {
    Ok(ExistingEvent {
        event_id: row.get(0)?,
        transition_key: row.get(1)?,
        source_claim_id: row.get(2)?,
        expected_prior_status: row.get(3)?,
        experiment_verdict: row.get(4)?,
        proposed_claim_status: row.get(5)?,
        exercised_falsifier: row.get(6)?,
        rationale: row.get(7)?,
        actor: row.get(8)?,
        reason: row.get(9)?,
        transition_timestamp: row.get(10)?,
        transition_spec_sha256: row.get(11)?,
    })
}

fn validate_spec(spec: &ClaimTransitionSpec) -> Result<()> {
    let request = &spec.transition;
    validate_transition_request(request)?;
    validate_successor_specs(&spec.successors)?;
    Ok(())
}

fn validate_transition_request(request: &ClaimTransitionRequest) -> Result<()> {
    for (field, value) in [
        ("transition_key", request.transition_key.as_str()),
        ("source_claim_id", request.source_claim_id.as_str()),
        (
            "expected_prior_status",
            request.expected_prior_status.as_str(),
        ),
        ("experiment_verdict", request.experiment_verdict.as_str()),
        ("exercised_falsifier", request.exercised_falsifier.as_str()),
        (
            "proposed_claim_status",
            request.proposed_claim_status.as_str(),
        ),
        ("rationale", request.rationale.as_str()),
        ("actor", request.actor.as_str()),
        ("reason", request.reason.as_str()),
        (
            "transition_timestamp",
            request.transition_timestamp.as_str(),
        ),
        (
            "expected_source_state_sha256",
            request.expected_source_state_sha256.as_str(),
        ),
    ] {
        validate_nonempty_ascii(field, value)?;
    }
    if request.expected_claim_id_max < 0 {
        bail!("expected_claim_id_max must not be negative");
    }
    if request.expected_source_state_sha256.len() != 64
        || !request
            .expected_source_state_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("expected_source_state_sha256 must be a 64-character hex digest");
    }
    DateTime::parse_from_rfc3339(&request.transition_timestamp)
        .context("transition_timestamp must be RFC3339")?;
    validate_claim_status(&request.expected_prior_status, "expected_prior_status")?;
    validate_claim_status(&request.proposed_claim_status, "proposed_claim_status")?;
    if !CLAIM_EXPERIMENT_VERDICTS.contains(&request.experiment_verdict.as_str()) {
        bail!("invalid experiment verdict {}", request.experiment_verdict);
    }
    validate_distinct_nonempty_ascii(
        "evidence_artifact_ids",
        &request.evidence_artifact_ids,
        false,
    )?;
    validate_distinct_nonempty_ascii("experiment_ids", &request.experiment_ids, true)?;
    validate_distinct_nonempty_ascii(
        "unresolved_assumptions",
        &request.unresolved_assumptions,
        true,
    )?;
    validate_distinct_nonempty_ascii("reserved_claim_ids", &request.reserved_claim_ids, true)?;
    for claim_id in &request.reserved_claim_ids {
        parse_numeric_claim_id(claim_id)
            .with_context(|| format!("invalid reserved claim id {claim_id}"))?;
    }
    Ok(())
}

fn validate_successor_specs(successors: &[SuccessorClaimSpec]) -> Result<()> {
    let mut proposal_keys = BTreeSet::new();
    let mut statements = BTreeSet::new();
    for successor in successors {
        validate_successor_spec(successor)?;
        if !proposal_keys.insert(&successor.proposal_key) {
            bail!(
                "duplicate successor proposal key {}",
                successor.proposal_key
            );
        }
        if !statements.insert(&successor.statement) {
            bail!("duplicate successor statement");
        }
    }
    Ok(())
}

fn validate_successor_spec(successor: &SuccessorClaimSpec) -> Result<()> {
    validate_nonempty_ascii("successor.proposal_key", &successor.proposal_key)?;
    validate_nonempty_ascii("successor.statement", &successor.statement)?;
    validate_nonempty_ascii("successor.initial_status", &successor.initial_status)?;
    validate_nonempty_ascii(
        "successor.source_or_implementation_boundary",
        &successor.source_or_implementation_boundary,
    )?;
    validate_nonempty_ascii(
        "successor.required_falsifier",
        &successor.required_falsifier,
    )?;
    validate_nonempty_ascii(
        "successor.predecessor_relation_kind",
        &successor.predecessor_relation_kind,
    )?;
    validate_claim_status(&successor.initial_status, "successor.initial_status")?;
    if !CLAIM_RELATION_KINDS.contains(&successor.predecessor_relation_kind.as_str()) {
        bail!(
            "invalid predecessor relation kind {}",
            successor.predecessor_relation_kind
        );
    }
    validate_distinct_nonempty_ascii("successor.where_stated", &successor.where_stated, true)?;
    validate_distinct_nonempty_ascii(
        "successor.evidence_artifact_ids",
        &successor.evidence_artifact_ids,
        true,
    )?;
    Ok(())
}

fn validate_source_state(spec: &ClaimTransitionSpec, current: &ClaimSourceState) -> Result<()> {
    if current.status != spec.transition.expected_prior_status {
        bail!(
            "source claim {} has status {}, expected {}",
            spec.transition.source_claim_id,
            current.status,
            spec.transition.expected_prior_status
        );
    }
    let actual_hash = sha256_claim_state(current);
    if actual_hash != spec.transition.expected_source_state_sha256 {
        bail!(
            "source claim {} changed since the transition specification was prepared",
            spec.transition.source_claim_id
        );
    }
    Ok(())
}

fn validate_claim_status(status: &str, field: &str) -> Result<()> {
    if !CANONICAL_CLAIM_STATUSES.contains(&status) {
        bail!("invalid claim status for {field}: {status}");
    }
    Ok(())
}

fn validate_nonempty_ascii(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} must not be empty");
    }
    if !value.is_ascii() {
        bail!("{field} must contain ASCII only");
    }
    Ok(())
}

fn validate_distinct_nonempty_ascii(
    field: &str,
    values: &[String],
    allow_empty: bool,
) -> Result<()> {
    if !allow_empty && values.is_empty() {
        bail!("{field} must contain at least one value");
    }
    let mut seen = BTreeSet::new();
    for value in values {
        validate_nonempty_ascii(field, value)?;
        if !seen.insert(value) {
            bail!("{field} contains a duplicate value: {value}");
        }
    }
    Ok(())
}

fn allocate_successors_with_reservations(
    successors: &[SuccessorClaimSpec],
    max_claim_id: i64,
    reserved_ids: &BTreeSet<i64>,
) -> Result<(Vec<AllocatedSuccessor>, Vec<String>)> {
    let mut ordered = successors.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.proposal_key.cmp(&right.proposal_key));
    let mut next_id = max_claim_id;
    let mut skipped = Vec::new();
    let mut allocated = Vec::with_capacity(ordered.len());
    for successor in ordered {
        loop {
            next_id = next_id
                .checked_add(1)
                .context("successor claim id allocation overflow")?;
            if reserved_ids.contains(&next_id) {
                skipped.push(format!("C-{next_id}"));
                continue;
            }
            allocated.push(AllocatedSuccessor {
                proposal_key: successor.proposal_key.clone(),
                canonical_claim_id: format!("C-{next_id}"),
                relation_kind: successor.predecessor_relation_kind.clone(),
            });
            break;
        }
    }
    Ok((allocated, skipped))
}

fn reserved_numeric_claim_ids(
    conn: &rusqlite::Connection,
    requested_ids: &[String],
) -> Result<BTreeSet<i64>> {
    let mut reserved = BTreeSet::new();
    let mut statement = conn.prepare("SELECT legacy_name FROM theorem_identities")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    for row in rows {
        if let Some(numeric_id) = numeric_prefix_id(&row?) {
            reserved.insert(numeric_id);
        }
    }
    for claim_id in requested_ids {
        reserved.insert(parse_numeric_claim_id(claim_id)?);
    }
    Ok(reserved)
}

fn parse_numeric_claim_id(claim_id: &str) -> Result<i64> {
    let number = claim_id
        .strip_prefix("C-")
        .context("claim id must use C-<digits> form")?;
    if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("claim id must use C-<digits> form: {claim_id}");
    }
    number
        .parse::<i64>()
        .context("claim id number does not fit in i64")
}

fn numeric_prefix_id(value: &str) -> Option<i64> {
    let suffix = value.strip_prefix('C')?;
    let digits = suffix
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn validate_successor_references(
    store: &ProvenanceStore,
    spec: &ClaimTransitionSpec,
    allocated: &[AllocatedSuccessor],
) -> Result<()> {
    validate_successor_references_on_conn(&store.conn, spec, allocated)
}

fn validate_successor_references_on_conn(
    conn: &rusqlite::Connection,
    spec: &ClaimTransitionSpec,
    allocated: &[AllocatedSuccessor],
) -> Result<()> {
    validate_existing_artifact_ids(
        conn,
        &spec.transition.evidence_artifact_ids,
        "evidence artifact",
    )?;
    validate_existing_experiment_ids(conn, &spec.transition.experiment_ids)?;
    for successor in &spec.successors {
        validate_existing_artifact_ids(
            conn,
            &successor.evidence_artifact_ids,
            "successor evidence artifact",
        )?;
    }
    validate_allocated_successor_ids(conn, &spec.transition.source_claim_id, allocated)?;
    validate_relation_graph(conn, &spec.transition.source_claim_id, allocated)?;
    Ok(())
}

fn validate_existing_artifact_ids(
    conn: &rusqlite::Connection,
    artifact_ids: &[String],
    label: &str,
) -> Result<()> {
    for artifact_id in artifact_ids {
        let exists = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM artifacts WHERE id = ?1)",
            params![artifact_id],
            |row| row.get::<_, i64>(0),
        )?;
        if exists == 0 {
            bail!("unresolved {label} id {artifact_id}");
        }
    }
    Ok(())
}

fn validate_existing_experiment_ids(
    conn: &rusqlite::Connection,
    experiment_ids: &[String],
) -> Result<()> {
    for experiment_id in experiment_ids {
        let exists = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM experiments_cp WHERE id = ?1)",
            params![experiment_id],
            |row| row.get::<_, i64>(0),
        )?;
        if exists == 0 {
            bail!("unresolved experiment id {experiment_id}");
        }
    }
    Ok(())
}

fn validate_allocated_successor_ids(
    conn: &rusqlite::Connection,
    source_claim_id: &str,
    allocated: &[AllocatedSuccessor],
) -> Result<()> {
    for successor in allocated {
        if successor.canonical_claim_id == source_claim_id {
            bail!("successor relation cannot point back to its source claim");
        }
        let collision = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM claims WHERE id = ?1)",
            params![successor.canonical_claim_id],
            |row| row.get::<_, i64>(0),
        )?;
        if collision != 0 {
            bail!("successor id collision: {}", successor.canonical_claim_id);
        }
    }
    Ok(())
}

fn validate_relation_graph(
    conn: &rusqlite::Connection,
    source_claim_id: &str,
    allocated: &[AllocatedSuccessor],
) -> Result<()> {
    let mut graph = BTreeMap::<String, Vec<String>>::new();
    let mut stmt =
        conn.prepare("SELECT predecessor_claim_id, successor_claim_id FROM claim_relations")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (predecessor, successor) = row?;
        graph.entry(predecessor).or_default().push(successor);
    }
    for successor in allocated {
        graph
            .entry(source_claim_id.to_string())
            .or_default()
            .push(successor.canonical_claim_id.clone());
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for node in graph.keys() {
        if relation_graph_has_cycle(node, &graph, &mut visiting, &mut visited) {
            bail!("claim relation graph contains a forbidden cycle at {node}");
        }
    }
    Ok(())
}

fn relation_graph_has_cycle(
    node: &str,
    graph: &BTreeMap<String, Vec<String>>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> bool {
    if visiting.contains(node) {
        return true;
    }
    if !visited.insert(node.to_string()) {
        return false;
    }
    visiting.insert(node.to_string());
    if let Some(successors) = graph.get(node)
        && successors
            .iter()
            .any(|successor| relation_graph_has_cycle(successor, graph, visiting, visited))
    {
        return true;
    }
    visiting.remove(node);
    false
}

fn max_numeric_claim_id_on_conn(conn: &rusqlite::Connection) -> Result<i64> {
    let ids = {
        let mut stmt = conn.prepare("SELECT id FROM claims")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    Ok(ids
        .into_iter()
        .filter_map(|id| {
            id.strip_prefix("C-")
                .and_then(|value| value.parse::<i64>().ok())
        })
        .max()
        .unwrap_or(0))
}

fn successors_for_event(
    conn: &rusqlite::Connection,
    event_id: i64,
) -> Result<Vec<AllocatedSuccessor>> {
    let mut stmt = conn.prepare(
        "SELECT proposal_key, successor_claim_id, predecessor_relation_kind
         FROM claim_transition_successors
         WHERE transition_event_id = ?1 ORDER BY proposal_key",
    )?;
    let rows = stmt.query_map(params![event_id], |row| {
        Ok(AllocatedSuccessor {
            proposal_key: row.get(0)?,
            canonical_claim_id: row.get(1)?,
            relation_kind: row.get(2)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

fn successor_statuses_for_event(conn: &rusqlite::Connection, event_id: i64) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT initial_status FROM claim_transition_successors
         WHERE transition_event_id = ?1 ORDER BY proposal_key",
    )?;
    let rows = stmt.query_map(params![event_id], |row| row.get::<_, String>(0))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

fn status_count_delta_with_successors(
    prior: &str,
    proposed: &str,
    successors: &[SuccessorClaimSpec],
) -> BTreeMap<String, i64> {
    status_count_delta_from_statuses(
        prior,
        proposed,
        &successors
            .iter()
            .map(|successor| successor.initial_status.clone())
            .collect::<Vec<_>>(),
    )
}

fn status_count_delta_from_statuses(
    prior: &str,
    proposed: &str,
    successor_statuses: &[String],
) -> BTreeMap<String, i64> {
    let mut delta = BTreeMap::new();
    if prior != proposed {
        *delta.entry(prior.to_string()).or_default() -= 1;
        *delta.entry(proposed.to_string()).or_default() += 1;
    }
    for status in successor_statuses {
        *delta.entry(status.clone()).or_default() += 1;
    }
    delta
}

fn sha256_claim_state(state: &ClaimSourceState) -> String {
    let mut canonical = [
        state.id.as_str(),
        state.statement.as_str(),
        state.status.as_str(),
        state.where_stated.as_str(),
        state.last_verified.as_str(),
        state.formal_proof.as_deref().unwrap_or(""),
        state.status_note.as_deref().unwrap_or(""),
        state.compat_toml_text.as_str(),
    ]
    .into_iter()
    .map(|value| format!("{}:{}", value.len(), value))
    .collect::<Vec<_>>()
    .join("|");
    if let Some(evidence) = &state.evidence_spec_json {
        canonical.push_str(&format!("|claim_evidence:{}:{}", evidence.len(), evidence));
    }
    crate::sha256_hex(&canonical)
}

fn render_new_claim_compat_text(
    id: &str,
    statement: &str,
    status: &str,
    where_stated: &str,
    last_verified: &str,
) -> String {
    format!(
        "id = {id:?}\nstatement = {statement:?}\nstatus = {status:?}\nwhere_stated = {where_stated:?}\nlast_verified = {last_verified:?}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_successors_by_proposal_key() {
        let specs = vec![
            SuccessorClaimSpec {
                proposal_key: "z".to_string(),
                statement: "z statement".to_string(),
                initial_status: "Provisional".to_string(),
                source_or_implementation_boundary: "boundary".to_string(),
                required_falsifier: "falsifier".to_string(),
                where_stated: vec!["source".to_string()],
                evidence_artifact_ids: vec!["A".to_string()],
                predecessor_relation_kind: "narrows".to_string(),
            },
            SuccessorClaimSpec {
                proposal_key: "a".to_string(),
                statement: "a statement".to_string(),
                initial_status: "Theoretical".to_string(),
                source_or_implementation_boundary: "boundary".to_string(),
                required_falsifier: "falsifier".to_string(),
                where_stated: vec!["source".to_string()],
                evidence_artifact_ids: vec!["A".to_string()],
                predecessor_relation_kind: "refines".to_string(),
            },
        ];
        let (allocated, skipped) =
            allocate_successors_with_reservations(&specs, 10, &BTreeSet::new())
                .expect("allocation");
        assert_eq!(allocated[0].proposal_key, "a");
        assert_eq!(allocated[0].canonical_claim_id, "C-11");
        assert_eq!(allocated[1].canonical_claim_id, "C-12");
        assert!(skipped.is_empty());
    }

    #[test]
    fn allocation_skips_reserved_numeric_prefixes() {
        let specs = vec![SuccessorClaimSpec {
            proposal_key: "a".to_string(),
            statement: "a statement".to_string(),
            initial_status: "Provisional".to_string(),
            source_or_implementation_boundary: "boundary".to_string(),
            required_falsifier: "falsifier".to_string(),
            where_stated: vec![],
            evidence_artifact_ids: vec![],
            predecessor_relation_kind: "narrows".to_string(),
        }];
        let reserved = BTreeSet::from([11_i64, 12_i64]);
        let (allocated, skipped) =
            allocate_successors_with_reservations(&specs, 10, &reserved).expect("allocation");
        assert_eq!(allocated[0].canonical_claim_id, "C-13");
        assert_eq!(skipped, ["C-11", "C-12"]);
    }

    #[test]
    fn status_delta_is_explicit() {
        let successors = vec!["Theoretical".to_string()];
        let delta = status_count_delta_from_statuses("Verified", "Provisional", &successors);
        assert_eq!(delta.get("Verified"), Some(&-1));
        assert_eq!(delta.get("Provisional"), Some(&1));
        assert_eq!(delta.get("Theoretical"), Some(&1));
        assert!(status_count_delta_from_statuses("Verified", "Verified", &[]).is_empty());
    }

    fn transition_test_store() -> ProvenanceStore {
        let mut conn = rusqlite::Connection::open_in_memory().expect("open test database");
        conn.pragma_update(None, "foreign_keys", "ON")
            .expect("enable foreign keys");
        crate::migrations::migrations()
            .to_latest(&mut conn)
            .expect("apply test migrations");
        let compat = r#"id = "C-1"
statement = "source statement""#;
        conn.execute(
            "INSERT INTO claims (
                 id, statement, status, where_stated, last_verified,
                 formal_proof, status_note, compat_toml_text
             ) VALUES ('C-1', 'source statement', 'Verified', 'source',
                       '2026-08-03', NULL, NULL, ?1)",
            rusqlite::params![compat],
        )
        .expect("insert source claim");
        conn.execute(
            "INSERT INTO artifacts (
                 id, key, title, citation, status, minimum_requirement_met,
                 canonical_functional_url, canonical_download_path
             ) VALUES ('A-1', 'test-artifact', 'Test artifact', 'Test citation',
                       'downloaded', 1, NULL, 'test-artifact.toml')",
            [],
        )
        .expect("insert evidence artifact");
        ProvenanceStore { conn }
    }

    fn source_state_hash() -> String {
        let compat = r#"id = "C-1"
statement = "source statement""#;
        sha256_claim_state(&ClaimSourceState {
            id: "C-1".to_string(),
            statement: "source statement".to_string(),
            status: "Verified".to_string(),
            where_stated: "source".to_string(),
            last_verified: "2026-08-03".to_string(),
            formal_proof: None,
            status_note: None,
            compat_toml_text: compat.to_string(),
            evidence_spec_json: None,
        })
    }

    fn transition_spec(key: &str) -> ClaimTransitionSpec {
        ClaimTransitionSpec {
            transition: ClaimTransitionRequest {
                transition_key: key.to_string(),
                source_claim_id: "C-1".to_string(),
                expected_prior_status: "Verified".to_string(),
                experiment_verdict: "Inconclusive".to_string(),
                evidence_artifact_ids: vec!["A-1".to_string()],
                experiment_ids: Vec::new(),
                exercised_falsifier: "independent fixture falsifier".to_string(),
                proposed_claim_status: "Provisional".to_string(),
                rationale: "separate source and implementation".to_string(),
                unresolved_assumptions: vec!["boundary term".to_string()],
                actor: "test".to_string(),
                reason: "transition test".to_string(),
                transition_timestamp: "2026-08-03T00:00:00Z".to_string(),
                expected_source_state_sha256: source_state_hash(),
                expected_claim_id_max: 1,
                reserved_claim_ids: Vec::new(),
            },
            successors: vec![SuccessorClaimSpec {
                proposal_key: "successor-a".to_string(),
                statement: "narrower implementation statement".to_string(),
                initial_status: "Theoretical".to_string(),
                source_or_implementation_boundary: "implementation boundary".to_string(),
                required_falsifier: "independent implementation defect".to_string(),
                where_stated: vec!["source equation".to_string()],
                evidence_artifact_ids: vec!["A-1".to_string()],
                predecessor_relation_kind: "narrows".to_string(),
            }],
        }
    }

    #[test]
    fn evidence_contract_changes_invalidate_pending_transition_hashes() -> Result<()> {
        let mut store = transition_test_store();
        let mut spec = transition_spec("contract-prestate");
        assert_eq!(
            store.claim_transition_source_state_sha256("C-1")?,
            source_state_hash()
        );
        store.conn.execute("INSERT INTO claim_evidence VALUES ('C-1', '{\"evidence_layer\":\"implementation_conformance\"}')", [])?;
        assert!(
            store
                .apply_claim_transition(&spec, "contract-prestate-v1")
                .is_err()
        );
        spec.transition.expected_source_state_sha256 =
            store.claim_transition_source_state_sha256("C-1")?;
        store.conn.execute("UPDATE claim_evidence SET spec_json='{\"evidence_layer\":\"phenomenological_mapping\"}' WHERE claim_id='C-1'", [])?;
        assert!(
            store
                .apply_claim_transition(&spec, "contract-prestate-v1")
                .is_err()
        );
        assert_eq!(
            store.claim_by_id("C-1")?.expect("source").status,
            "Verified"
        );
        spec.transition.expected_source_state_sha256 =
            store.claim_transition_source_state_sha256("C-1")?;
        store.apply_claim_transition(&spec, "contract-prestate-v1")?;
        Ok(())
    }
    #[test]
    fn apply_is_atomic_and_exact_replay_is_idempotent() -> Result<()> {
        let mut store = transition_test_store();
        let spec = transition_spec("transition-test-1");
        let first = store.apply_claim_transition(&spec, "transition-test-1-v1")?;
        assert!(!first.exact_replay);
        assert_eq!(first.allocated_successors[0].canonical_claim_id, "C-2");
        assert_eq!(
            store.claim_by_id("C-1")?.expect("source").status,
            "Provisional"
        );
        assert_eq!(
            store.claim_by_id("C-2")?.expect("successor").status,
            "Theoretical"
        );
        assert_eq!(store.list_claim_transition_events()?.len(), 1);
        assert_eq!(store.list_claim_relations()?.len(), 1);
        store.verify_claim_transition_invariants()?;
        let direct = store
            .conn
            .execute("UPDATE claims SET status = 'Verified' WHERE id = 'C-1'", []);
        assert!(direct.is_err());
        let (events_export, relations_export) = store.claim_transition_compat_texts()?;
        assert!(events_export.contains("expected_prior_status = \"Verified\""));
        assert!(events_export.contains("experiment_verdict = \"Inconclusive\""));
        assert!(events_export.contains("A-1"));
        assert!(events_export.contains("C-2"));
        assert!(relations_export.contains("predecessor_claim_id = \"C-1\""));
        assert!(relations_export.contains("successor_claim_id = \"C-2\""));

        let replay = store.apply_claim_transition(&spec, "transition-test-1-v1")?;
        assert!(replay.exact_replay);
        assert_eq!(replay.event_id, first.event_id);
        assert_eq!(store.list_claim_transition_events()?.len(), 1);
        assert_eq!(store.list_claim_relations()?.len(), 1);

        let changed = store.apply_claim_transition(&spec, "transition-test-1-v2");
        assert!(changed.is_err());
        assert_eq!(store.list_claim_transition_events()?.len(), 1);
        Ok(())
    }

    #[test]
    fn missing_evidence_rolls_back_without_status_change() -> Result<()> {
        let mut store = transition_test_store();
        let mut spec = transition_spec("transition-test-rollback");
        spec.transition.evidence_artifact_ids = vec!["missing".to_string()];
        let result = store.apply_claim_transition(&spec, "transition-test-rollback");
        assert!(result.is_err());
        assert_eq!(
            store.claim_by_id("C-1")?.expect("source").status,
            "Verified"
        );
        assert_eq!(store.list_claim_transition_events()?.len(), 0);
        assert_eq!(store.list_claims()?.len(), 1);
        Ok(())
    }

    #[test]
    fn stale_status_is_rejected_before_writes() -> Result<()> {
        let mut store = transition_test_store();
        let mut spec = transition_spec("transition-test-stale-status");
        spec.transition.expected_prior_status = "Theoretical".to_string();
        let result = store.apply_claim_transition(&spec, "transition-test-stale-status");
        assert!(result.is_err());
        assert_eq!(store.list_claim_transition_events()?.len(), 0);
        assert_eq!(store.list_claims()?.len(), 1);
        Ok(())
    }

    #[test]
    fn nonexistent_experiment_rolls_back_before_event_insert() -> Result<()> {
        let mut store = transition_test_store();
        let mut spec = transition_spec("transition-test-missing-experiment");
        spec.transition.experiment_ids = vec!["E-missing".to_string()];
        let result = store.apply_claim_transition(&spec, "transition-test-missing-experiment");
        assert!(result.is_err());
        assert_eq!(store.list_claim_transition_events()?.len(), 0);
        assert_eq!(
            store.claim_by_id("C-1")?.expect("source").status,
            "Verified"
        );
        Ok(())
    }

    #[test]
    fn invalid_status_and_verdict_are_rejected() -> Result<()> {
        let mut invalid_status = transition_test_store();
        let mut status_spec = transition_spec("transition-test-invalid-status");
        status_spec.transition.proposed_claim_status = "Unknown".to_string();
        assert!(
            invalid_status
                .apply_claim_transition(&status_spec, "transition-test-invalid-status")
                .is_err()
        );

        let mut invalid_verdict = transition_test_store();
        let mut verdict_spec = transition_spec("transition-test-invalid-verdict");
        verdict_spec.transition.experiment_verdict = "Unknown".to_string();
        assert!(
            invalid_verdict
                .apply_claim_transition(&verdict_spec, "transition-test-invalid-verdict")
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn dry_run_allocates_the_same_ids_as_apply() -> Result<()> {
        let mut store = transition_test_store();
        let spec = transition_spec("transition-test-plan-apply");
        let plan = store.plan_claim_transition(&spec, "transition-test-plan-apply")?;
        let applied = store.apply_claim_transition(&spec, "transition-test-plan-apply")?;
        assert_eq!(plan.allocated_successors, applied.allocated_successors);
        assert!(!plan.exact_replay);
        assert!(!applied.exact_replay);
        Ok(())
    }

    #[test]
    fn successor_collision_is_rejected() -> Result<()> {
        let store = transition_test_store();
        store.conn.execute(
            "INSERT INTO claims (
                 id, statement, status, where_stated, last_verified,
                 formal_proof, status_note, compat_toml_text
             ) VALUES ('C-2', 'existing', 'Verified', 'source',
                       '2026-08-03', NULL, NULL, 'existing')",
            [],
        )?;
        let spec = transition_spec("transition-test-collision");
        let allocated = vec![AllocatedSuccessor {
            proposal_key: "successor-a".to_string(),
            canonical_claim_id: "C-2".to_string(),
            relation_kind: "narrows".to_string(),
        }];
        assert!(validate_successor_references(&store, &spec, &allocated).is_err());
        Ok(())
    }

    #[test]
    fn relation_cycle_is_rejected() -> Result<()> {
        let store = transition_test_store();
        for id in ["C-2", "C-3"] {
            store.conn.execute(
                "INSERT INTO claims (
                     id, statement, status, where_stated, last_verified,
                     formal_proof, status_note, compat_toml_text
                 ) VALUES (?1, ?2, 'Verified', 'source',
                           '2026-08-03', NULL, NULL, ?2)",
                params![id, id],
            )?;
        }
        store.conn.execute(
            "INSERT INTO claim_transition_events (
                 transition_key, source_claim_id, expected_prior_status,
                 experiment_verdict, proposed_claim_status, exercised_falsifier,
                 rationale, actor, reason, transition_ts_utc,
                 transition_spec_sha256, expected_source_state_sha256,
                 expected_claim_id_max
             ) VALUES ('cycle-event', 'C-1', 'Verified', 'Inconclusive',
                       'Provisional', 'falsifier', 'rationale', 'test', 'reason',
                       '2026-08-03T00:00:00Z', 'hash', 'hash', 3)",
            [],
        )?;
        let event_id = store.conn.last_insert_rowid();
        store.conn.execute(
            "INSERT INTO claim_relations (
                 predecessor_claim_id, successor_claim_id, relation_kind,
                 transition_event_id
             ) VALUES ('C-2', 'C-3', 'narrows', ?1)",
            params![event_id],
        )?;
        let allocated = vec![AllocatedSuccessor {
            proposal_key: "cycle-successor".to_string(),
            canonical_claim_id: "C-2".to_string(),
            relation_kind: "narrows".to_string(),
        }];
        assert!(validate_relation_graph(&store.conn, "C-3", &allocated).is_err());
        Ok(())
    }

    #[test]
    fn direct_status_mutation_requires_transition_event() -> Result<()> {
        let store = transition_test_store();
        let result = store.conn.execute(
            "UPDATE claims SET status = 'Provisional' WHERE id = 'C-1'",
            [],
        );
        assert!(result.is_err());
        Ok(())
    }
}
