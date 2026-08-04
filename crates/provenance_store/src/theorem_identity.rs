//! Validation for the typed theorem namespace and explicit claim relations.

use anyhow::{Context, Result, bail};
use chrono::DateTime;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path};

use crate::{
    CLI_APPLICATION_ID,
    migrations::{CANONICAL_CLAIM_STATUSES, JUSTIFIED_UNLINKED_THEOREM_IDS},
    sha256_hex,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TheoremIdentitySpec {
    pub binding_key: String,
    pub actor: String,
    pub reason: String,
    pub applied_at: String,
    #[serde(rename = "binding", default)]
    pub bindings: Vec<TheoremBindingSpec>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TheoremBindingSpec {
    pub proposal_key: String,
    pub stable_id: String,
    pub legacy_name: String,
    pub proof_path: String,
    pub title: String,
    pub statement: String,
    pub initial_status: String,
    #[serde(default)]
    pub where_stated: Vec<String>,
    pub assumptions: String,
    pub kernel_result: String,
    pub replay_command: String,
    pub falsifier: String,
    #[serde(default)]
    pub evidence_artifact_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TheoremIdentityBindResult {
    pub binding_key: String,
    pub spec_sha256: String,
    pub claim_mappings: Vec<TheoremClaimMapping>,
    pub exact_replay: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TheoremClaimMapping {
    pub proposal_key: String,
    pub stable_theorem_id: String,
    pub claim_id: String,
}

pub fn parse_theorem_identity_spec(raw: &str) -> Result<TheoremIdentitySpec> {
    if raw.trim().is_empty() {
        bail!("theorem identity specification is empty");
    }
    validate_ascii("theorem identity specification", raw)?;
    let spec: TheoremIdentitySpec = toml::from_str(raw).context("parse theorem identity TOML")?;
    validate_identity_spec(&spec)?;
    Ok(spec)
}

pub(crate) fn validate_identity_spec(spec: &TheoremIdentitySpec) -> Result<()> {
    for (field, value) in [
        ("binding_key", spec.binding_key.as_str()),
        ("actor", spec.actor.as_str()),
        ("reason", spec.reason.as_str()),
        ("applied_at", spec.applied_at.as_str()),
    ] {
        validate_nonempty_ascii(field, value)?;
    }
    DateTime::parse_from_rfc3339(&spec.applied_at).context("applied_at must be RFC3339")?;
    if spec.bindings.is_empty() {
        bail!("theorem identity specification requires at least one binding");
    }
    let mut proposal_keys = BTreeSet::new();
    let mut stable_ids = BTreeSet::new();
    let mut legacy_names = BTreeSet::new();
    for binding in &spec.bindings {
        for (field, value) in [
            ("proposal_key", binding.proposal_key.as_str()),
            ("stable_id", binding.stable_id.as_str()),
            ("legacy_name", binding.legacy_name.as_str()),
            ("proof_path", binding.proof_path.as_str()),
            ("title", binding.title.as_str()),
            ("statement", binding.statement.as_str()),
            ("initial_status", binding.initial_status.as_str()),
            ("assumptions", binding.assumptions.as_str()),
            ("kernel_result", binding.kernel_result.as_str()),
            ("replay_command", binding.replay_command.as_str()),
            ("falsifier", binding.falsifier.as_str()),
        ] {
            validate_nonempty_ascii(field, value)?;
        }
        if !binding.stable_id.starts_with("THM-") {
            bail!("stable_id must start with THM-: {}", binding.stable_id);
        }
        if !CANONICAL_CLAIM_STATUSES.contains(&binding.initial_status.as_str()) {
            bail!("invalid initial claim status {}", binding.initial_status);
        }
        if !proposal_keys.insert(&binding.proposal_key) {
            bail!(
                "duplicate theorem binding proposal key {}",
                binding.proposal_key
            );
        }
        if !stable_ids.insert(&binding.stable_id) {
            bail!("duplicate stable theorem identity {}", binding.stable_id);
        }
        if !legacy_names.insert(&binding.legacy_name) {
            bail!("duplicate theorem legacy name {}", binding.legacy_name);
        }
        if binding.where_stated.is_empty() {
            bail!(
                "theorem binding {} requires where_stated",
                binding.proposal_key
            );
        }
        for reference in &binding.where_stated {
            validate_nonempty_ascii("where_stated", reference)?;
        }
        validate_distinct_nonempty_ascii(
            "evidence_artifact_ids",
            &binding.evidence_artifact_ids,
            false,
        )?;
    }
    Ok(())
}

pub fn bind_theorem_identities(
    conn: &mut Connection,
    repo_root: &Path,
    spec: &TheoremIdentitySpec,
    raw_spec: &str,
) -> Result<TheoremIdentityBindResult> {
    let spec_sha256 = sha256_hex(raw_spec);
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    if let Some((stored_hash, theorem_ids_json, claim_ids_json)) = tx
        .query_row(
            "SELECT spec_sha256, theorem_ids_json, claim_ids_json
             FROM theorem_identity_events WHERE binding_key = ?1",
            params![spec.binding_key],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?
    {
        if stored_hash != spec_sha256 {
            bail!(
                "theorem binding key {} already exists with different content hash",
                spec.binding_key
            );
        }
        let theorem_ids: Vec<String> = serde_json::from_str(&theorem_ids_json)?;
        let claim_ids: Vec<String> = serde_json::from_str(&claim_ids_json)?;
        let mut claim_mappings = Vec::with_capacity(spec.bindings.len());
        for binding in &spec.bindings {
            let index = theorem_ids
                .iter()
                .position(|stable_id| stable_id == &binding.stable_id)
                .with_context(|| {
                    format!(
                        "replayed theorem identity is missing: {}",
                        binding.stable_id
                    )
                })?;
            claim_mappings.push(TheoremClaimMapping {
                proposal_key: binding.proposal_key.clone(),
                stable_theorem_id: theorem_ids[index].clone(),
                claim_id: claim_ids[index].clone(),
            });
        }
        tx.commit()?;
        return Ok(TheoremIdentityBindResult {
            binding_key: spec.binding_key.clone(),
            spec_sha256,
            claim_mappings,
            exact_replay: true,
        });
    }

    let mut ordered = spec.bindings.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.proposal_key.cmp(&right.proposal_key));
    let mut reserved = reserved_numeric_ids(&tx)?;
    let mut next_id = max_numeric_claim_id(&tx)?;
    let mut mappings = Vec::with_capacity(ordered.len());
    let mut theorem_ids = Vec::with_capacity(ordered.len());
    let mut claim_ids = Vec::with_capacity(ordered.len());

    for binding in ordered {
        let (existing_stable_id, existing_proof_path): (String, String) = tx
            .query_row(
                "SELECT stable_id, proof_path FROM theorem_identities
                 WHERE legacy_name = ?1",
                params![binding.legacy_name],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .with_context(|| {
                format!(
                    "theorem legacy name is not registered: {}",
                    binding.legacy_name
                )
            })?;
        if existing_proof_path != binding.proof_path {
            bail!(
                "proof path does not match legacy theorem {}",
                binding.legacy_name
            );
        }
        if !repo_root.join(&binding.proof_path).is_file() {
            bail!("proof path is missing: {}", binding.proof_path);
        }
        let stable_conflict: Option<String> = tx
            .query_row(
                "SELECT legacy_name FROM theorem_identities
                 WHERE stable_id = ?1 AND legacy_name <> ?2",
                params![binding.stable_id, binding.legacy_name],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(legacy_name) = stable_conflict {
            bail!(
                "stable theorem identity {} already belongs to {}",
                binding.stable_id,
                legacy_name
            );
        }
        if existing_stable_id != binding.stable_id {
            tx.execute(
                "UPDATE theorem_identities SET stable_id = ?1
                 WHERE legacy_name = ?2",
                params![binding.stable_id, binding.legacy_name],
            )?;
        }
        loop {
            next_id = next_id
                .checked_add(1)
                .context("formal theorem claim id allocation overflow")?;
            if !reserved.contains(&next_id) {
                break;
            }
        }
        let claim_id = format!("C-{next_id}");
        reserved.insert(next_id);
        let where_stated = binding.where_stated.join("; ");
        let status_note = format!(
            "Kernel-checked formal proposition for {}. Scope is limited to the stated assumptions and does not establish a physical or implementation claim.",
            binding.stable_id
        );
        let compat_toml_text = format!(
            "id = {claim_id:?}\nstatement = {statement:?}\nstatus = {status:?}\nwhere_stated = {where_stated:?}\nlast_verified = {date:?}\nformal_proof = {proof:?}\nstatus_note = {note:?}",
            statement = binding.statement,
            status = binding.initial_status,
            date = &spec.applied_at[..10],
            proof = binding.proof_path,
            note = status_note,
        );
        tx.execute(
            "INSERT INTO claims (
                id, statement, status, where_stated, last_verified,
                formal_proof, status_note, compat_toml_text
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                claim_id,
                binding.statement,
                binding.initial_status,
                where_stated,
                &spec.applied_at[..10],
                binding.proof_path,
                status_note,
                compat_toml_text,
            ],
        )?;
        tx.execute(
            "INSERT INTO claim_revisions (
                 claim_id, field_name, prev_value_sha256, new_value_sha256,
                 actor, reason, operation, application_id
             ) VALUES (?1, 'record', NULL, ?2, ?3, ?4, 'create', ?5)",
            params![
                claim_id,
                sha256_hex(&compat_toml_text),
                spec.actor,
                spec.reason,
                CLI_APPLICATION_ID,
            ],
        )?;
        tx.execute(
            "UPDATE theorem_identities
             SET identity_kind = 'explicit_link', assumptions = ?2,
                 kernel_result = ?3, replay_command = ?4, falsifier = ?5
             WHERE stable_id = ?1",
            params![
                binding.stable_id,
                binding.assumptions,
                binding.kernel_result,
                binding.replay_command,
                binding.falsifier,
            ],
        )?;
        tx.execute(
            "INSERT INTO theorem_claim_links (
                 theorem_stable_id, claim_id, relation_kind
             ) VALUES (?1, ?2, 'formal_proposition')",
            params![binding.stable_id, claim_id],
        )?;
        for artifact_id in &binding.evidence_artifact_ids {
            tx.execute(
                "INSERT INTO theorem_identity_evidence (theorem_stable_id, artifact_id)
                 VALUES (?1, ?2)",
                params![binding.stable_id, artifact_id],
            )?;
        }
        let linked_claims = linked_claim_ids(&tx, &binding.stable_id)?;
        tx.execute(
            "UPDATE theorems SET title = ?2, linked_claim_ids_json = ?3
             WHERE id = ?1",
            params![
                binding.legacy_name,
                binding.title,
                serde_json::to_string(&linked_claims)?,
            ],
        )?;
        theorem_ids.push(binding.stable_id.clone());
        claim_ids.push(claim_id.clone());
        mappings.push(TheoremClaimMapping {
            proposal_key: binding.proposal_key.clone(),
            stable_theorem_id: binding.stable_id.clone(),
            claim_id,
        });
    }

    tx.execute(
        "INSERT INTO theorem_identity_events (
             binding_key, spec_sha256, actor, reason, applied_at,
             theorem_ids_json, claim_ids_json
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            spec.binding_key,
            spec_sha256,
            spec.actor,
            spec.reason,
            spec.applied_at,
            serde_json::to_string(&theorem_ids)?,
            serde_json::to_string(&claim_ids)?,
        ],
    )?;
    tx.commit()?;
    Ok(TheoremIdentityBindResult {
        binding_key: spec.binding_key.clone(),
        spec_sha256,
        claim_mappings: mappings,
        exact_replay: false,
    })
}

fn max_numeric_claim_id(conn: &Connection) -> Result<i64> {
    let mut statement = conn.prepare("SELECT id FROM claims")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    let mut max_id = 0_i64;
    for row in rows {
        if let Some(value) = row?.strip_prefix("C-")
            && let Ok(value) = value.parse::<i64>()
        {
            max_id = max_id.max(value);
        }
    }
    Ok(max_id)
}

fn reserved_numeric_ids(conn: &Connection) -> Result<BTreeSet<i64>> {
    let mut reserved = BTreeSet::new();
    let mut statement = conn.prepare("SELECT legacy_name FROM theorem_identities")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    for row in rows {
        let value = row?;
        let Some(suffix) = value.strip_prefix('C') else {
            continue;
        };
        let digits = suffix
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect::<String>();
        if let Ok(number) = digits.parse::<i64>() {
            reserved.insert(number);
        }
    }
    Ok(reserved)
}

fn linked_claim_ids(conn: &rusqlite::Connection, stable_id: &str) -> Result<Vec<String>> {
    let mut statement = conn.prepare(
        "SELECT claim_id FROM theorem_claim_links
         WHERE theorem_stable_id = ?1 ORDER BY claim_id",
    )?;
    let rows = statement.query_map(params![stable_id], |row| row.get(0))?;
    Ok(rows.collect::<std::result::Result<Vec<String>, _>>()?)
}

fn validate_ascii(field: &str, value: &str) -> Result<()> {
    if !value.is_ascii() {
        bail!("{field} must contain ASCII only");
    }
    Ok(())
}

fn validate_nonempty_ascii(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} must not be empty");
    }
    validate_ascii(field, value)
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
            bail!("{field} contains duplicate value {value}");
        }
    }
    Ok(())
}

pub(crate) fn default_stable_theorem_id(legacy_name: &str) -> String {
    format!("THM-LEGACY-{legacy_name}")
}

pub(crate) fn validate_theorem_identities(conn: &Connection, repo_root: &Path) -> Result<()> {
    let mut statement = conn.prepare(
        "SELECT stable_id, legacy_name, proof_path, identity_kind
         FROM theorem_identities ORDER BY stable_id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    for row in rows {
        let (stable_id, legacy_name, proof_path, identity_kind) = row?;
        if !stable_id.is_ascii() || !stable_id.starts_with("THM-") {
            bail!("invalid stable theorem identity {stable_id}");
        }
        if !legacy_name.is_ascii() || legacy_name.trim().is_empty() {
            bail!("theorem {stable_id} has an invalid legacy name");
        }
        if !repo_root.join(&proof_path).is_file() {
            bail!("theorem {stable_id} proof path is missing: {proof_path}");
        }
        if !matches!(
            identity_kind.as_str(),
            "explicit_link" | "legacy_alias" | "unresolved"
        ) {
            bail!("theorem {stable_id} has invalid identity kind {identity_kind}");
        }

        let linked_claims = explicit_claim_links(conn, &stable_id, &proof_path)?;
        let is_claim_like = numeric_claim_like_prefix(&legacy_name);
        match identity_kind.as_str() {
            "explicit_link" if linked_claims.is_empty() => {
                bail!(
                    "theorem {legacy_name} declares explicit_link without an explicit claim relation"
                )
            }
            "legacy_alias" if !is_claim_like && linked_claims.is_empty() => {
                bail!("non-claim theorem {legacy_name} cannot use legacy_alias")
            }
            "unresolved" if is_claim_like => {
                bail!("claim-like theorem {legacy_name} has unresolved identity")
            }
            _ => {}
        }
    }

    let mut projection = conn.prepare(
        "SELECT theorems.id, theorem_identities.stable_id,
                theorems.linked_claim_ids_json
         FROM theorems
         LEFT JOIN theorem_identities
           ON theorem_identities.legacy_name = theorems.id
         ORDER BY theorems.id",
    )?;
    let projection_rows = projection.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    for row in projection_rows {
        let (legacy_name, stable_id, linked_claim_ids_json) = row?;
        let stable_id = stable_id
            .with_context(|| format!("theorem projection {legacy_name} has no identity"))?;
        let projected: Vec<String> = serde_json::from_str(&linked_claim_ids_json)
            .with_context(|| format!("parse theorem links for {legacy_name}"))?;
        let mut explicit = linked_claims_for_identity(conn, &stable_id)?;
        explicit.sort();
        let mut projected = projected;
        projected.sort();
        if projected != explicit {
            bail!("theorem projection {legacy_name} disagrees with explicit theorem_claim_links");
        }
    }

    Ok(())
}

fn explicit_claim_links(
    conn: &Connection,
    stable_id: &str,
    proof_path: &str,
) -> Result<Vec<String>> {
    let mut statement = conn.prepare(
        "SELECT claims.id
         FROM theorem_claim_links
         JOIN claims ON claims.id = theorem_claim_links.claim_id
         WHERE theorem_claim_links.theorem_stable_id = ?1
           AND theorem_claim_links.relation_kind = 'formal_proposition'
           AND (
               claims.formal_proof = ?2
               OR instr(claims.where_stated, ?2) > 0
               OR instr(COALESCE(claims.status_note, ''), ?2) > 0
           )
         ORDER BY claims.id",
    )?;
    let rows = statement.query_map(params![stable_id, proof_path], |row| row.get(0))?;
    Ok(rows.collect::<std::result::Result<Vec<String>, _>>()?)
}

fn linked_claims_for_identity(conn: &Connection, stable_id: &str) -> Result<Vec<String>> {
    let mut statement = conn.prepare(
        "SELECT claim_id FROM theorem_claim_links
         WHERE theorem_stable_id = ?1 ORDER BY claim_id",
    )?;
    let rows = statement.query_map(params![stable_id], |row| row.get(0))?;
    Ok(rows.collect::<std::result::Result<Vec<String>, _>>()?)
}

fn numeric_claim_like_prefix(value: &str) -> bool {
    let Some(suffix) = value.strip_prefix('C') else {
        return false;
    };
    if suffix.starts_with('_') {
        return true;
    }
    let digit_count = suffix
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .count();
    if digit_count == 0 {
        return false;
    }
    suffix
        .chars()
        .nth(digit_count)
        .map(|character| character == '_' || character.is_ascii_alphabetic())
        .unwrap_or(false)
}

pub(crate) fn is_declared_legacy_alias(legacy_name: &str) -> bool {
    JUSTIFIED_UNLINKED_THEOREM_IDS.contains(&legacy_name)
}

#[cfg(test)]
mod tests {
    use super::numeric_claim_like_prefix;

    #[test]
    fn numeric_prefix_is_claim_like_without_inferring_a_claim() {
        assert!(numeric_claim_like_prefix("C1635_SedenionDriverSemantics"));
        assert!(numeric_claim_like_prefix("C1140b_PathionGap_6_10"));
        assert!(numeric_claim_like_prefix("C_ConjugateInvolution"));
        assert!(!numeric_claim_like_prefix("ConjugateInvolution"));
        assert!(!numeric_claim_like_prefix("THM-SASS-DRIVER-001"));
    }
}
