//! Registry TOML loaders for claim, insight, and experiment records.
//!
//! - `load_claims_from_registry`: parses `[[claim]]` rows and runs each
//!   record through `normalize_claim_record` (canonicalizes status,
//!   captures legacy tokens as status_note, re-renders compat_toml_text).
//! - `load_insights_from_registry`: parses `[[insight]]` rows,
//!   normalizes status, merges `claims` + `related_claims` into a
//!   sorted-deduped claim_refs vector, and renders the normalized
//!   compat_toml_text via `render_normalized_insight_compat_toml`.
//! - `load_experiments_from_registry`: parses `[[experiment]]` rows,
//!   merges `claims` + `claim_refs`, and preserves the source TOML
//!   table verbatim as the compat_toml_text via `render_toml_table`.

use anyhow::{Context, Result};
use provenance_core::ClaimRecord;
use toml::Value;

use super::{
    claim_proofs::render_normalized_insight_compat_toml,
    status_normalize::{normalize_claim_record, normalize_insight_status},
    toml_helpers::{optional_string_field, render_toml_table, string_array_field, string_field},
    types::{ExperimentCompatRecord, InsightCompatRecord},
};

pub(crate) fn load_claims_from_registry(raw: &str) -> Result<Vec<ClaimRecord>> {
    let value: Value = toml::from_str(raw).context("parse claims registry")?;
    let claims = value
        .get("claim")
        .and_then(Value::as_array)
        .context("claim array missing")?;
    let mut out = Vec::new();
    for claim in claims {
        let table = claim.as_table().context("claim row must be table")?;
        let mut record = ClaimRecord {
            id: string_field(table, "id"),
            statement: string_field(table, "statement"),
            status: string_field(table, "status"),
            where_stated: string_field(table, "where_stated"),
            last_verified: string_field(table, "last_verified"),
            formal_proof: optional_string_field(table, "formal_proof"),
            status_note: optional_string_field(table, "status_note"),
            compat_toml_text: String::new(),
        };
        normalize_claim_record(&mut record)?;
        out.push(record);
    }
    Ok(out)
}

pub(crate) fn load_insights_from_registry(raw: &str) -> Result<Vec<InsightCompatRecord>> {
    let value: Value = toml::from_str(raw).context("parse insights registry")?;
    let insights = value
        .get("insight")
        .and_then(Value::as_array)
        .context("insight array missing")?;
    let mut out = Vec::new();
    for insight in insights {
        let table = insight.as_table().context("insight row must be table")?;
        let title = optional_string_field(table, "title")
            .or_else(|| optional_string_field(table, "insight"))
            .unwrap_or_else(|| string_field(table, "id"));
        let raw_status = optional_string_field(table, "status");
        let status = raw_status
            .as_deref()
            .map(normalize_insight_status)
            .unwrap_or("unknown")
            .to_string();
        let mut claim_refs = string_array_field(table, "claims");
        claim_refs.extend(string_array_field(table, "related_claims"));
        claim_refs.sort();
        claim_refs.dedup();
        out.push(InsightCompatRecord {
            id: string_field(table, "id"),
            title,
            status,
            claim_refs,
            status_note: optional_string_field(table, "status_note"),
            compat_toml_text: render_normalized_insight_compat_toml(table, raw_status.as_deref())?,
        });
    }
    Ok(out)
}

pub(crate) fn load_experiments_from_registry(raw: &str) -> Result<Vec<ExperimentCompatRecord>> {
    let value: Value = toml::from_str(raw).context("parse experiments registry")?;
    let experiments = value
        .get("experiment")
        .and_then(Value::as_array)
        .context("experiment array missing")?;
    let mut out = Vec::new();
    for experiment in experiments {
        let table = experiment
            .as_table()
            .context("experiment row must be table")?;
        let title =
            optional_string_field(table, "title").unwrap_or_else(|| string_field(table, "id"));
        let status = optional_string_field(table, "status")
            .or_else(|| optional_string_field(table, "status_token"))
            .unwrap_or_else(|| "unknown".to_string());
        let mut claim_refs = string_array_field(table, "claims");
        claim_refs.extend(string_array_field(table, "claim_refs"));
        claim_refs.sort();
        claim_refs.dedup();
        out.push(ExperimentCompatRecord {
            id: string_field(table, "id"),
            title,
            status,
            binary: optional_string_field(table, "binary"),
            claim_refs,
            status_note: optional_string_field(table, "status_note"),
            compat_toml_text: render_toml_table(table)?,
        });
    }
    Ok(out)
}
