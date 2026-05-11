//! Claim <-> Rocq proof correlation and compat-export rendering for
//! individual claim/insight rows.
//!
//! - `canonical_formal_proof_for_claim`: resolves the canonical
//!   `formal_proof` path for a claim by checking the claim's own
//!   formal_proof field, scanning where_stated/status_note/formal_proof
//!   for `proofs/.../*.v` references, and falling back to a verified
//!   proof from the inventory keyed by claim id.
//! - `preferred_primary_verified_proof_for_claim`: picks the
//!   shortest-suffix verified proof entry from the inventory, preferring
//!   primary stems over suffixed variants.
//! - `proof_entry_priority`: priority key for the sort above.
//! - `render_normalized_claim_compat_toml`: re-emits a ClaimRecord into
//!   compat_toml_text, splicing live SQLite columns.
//! - `render_normalized_insight_compat_toml`: re-emits an insight TOML
//!   table with the normalized status token applied.
//! - `extract_proof_paths`: scrapes `proofs/...*.v` paths from a text
//!   blob (where_stated / status_note / formal_proof).
//! - `link_claims_for_proof`: enumerates ClaimRecord ids that reference
//!   a given proof path or theorem stem.
//! - `normalized_claim_id_from_theorem_stem`: maps a theorem stem
//!   (e.g. C123_Foo) to the canonical claim id (C-123) when applicable.

use std::path::Path;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use provenance_core::ClaimRecord;
use toml::Value;

use super::status_normalize::normalize_insight_status;
use super::toml_helpers::render_toml_table;
use super::{ProofInventory, ProofInventoryEntry};

pub(crate) fn canonical_formal_proof_for_claim(
    repo_root: &Path,
    claim: &ClaimRecord,
    proof_inventory: &ProofInventory,
) -> Option<String> {
    if let Some(formal_proof) = claim.formal_proof.as_deref()
        && !formal_proof.trim().is_empty()
        && repo_root.join(formal_proof).exists()
    {
        return Some(formal_proof.trim().to_string());
    }

    let mut referenced_paths = extract_proof_paths(&claim.where_stated);
    if let Some(status_note) = &claim.status_note {
        referenced_paths.extend(extract_proof_paths(status_note));
    }
    if let Some(formal_proof) = claim.formal_proof.as_deref() {
        referenced_paths.extend(extract_proof_paths(formal_proof));
    }
    referenced_paths.retain(|path| repo_root.join(path).exists());
    referenced_paths.sort();
    referenced_paths.dedup();

    if let Some(primary_verified) =
        preferred_primary_verified_proof_for_claim(claim, proof_inventory)
    {
        return Some(primary_verified);
    }

    if referenced_paths.len() == 1 {
        return referenced_paths.into_iter().next();
    }

    None
}

pub(crate) fn preferred_primary_verified_proof_for_claim(
    claim: &ClaimRecord,
    proof_inventory: &ProofInventory,
) -> Option<String> {
    let claim_prefix = claim.id.strip_prefix("C-").unwrap_or(&claim.id);
    let mut candidates = proof_inventory
        .verified_by_claim_id
        .get(&claim.id)
        .cloned()
        .unwrap_or_default();
    candidates.sort_by(|lhs, rhs| {
        proof_entry_priority(claim_prefix, lhs)
            .cmp(&proof_entry_priority(claim_prefix, rhs))
            .reverse()
            .then_with(|| lhs.path.as_str().cmp(rhs.path.as_str()))
    });
    candidates
        .first()
        .map(|entry| entry.path.as_str().to_string())
}

fn proof_entry_priority(claim_prefix: &str, entry: &ProofInventoryEntry) -> (u8, usize) {
    let suffix = entry.stem.strip_prefix('C').unwrap_or(&entry.stem);
    let suffix = suffix.strip_prefix(claim_prefix).unwrap_or(suffix);
    let primary_rank = if suffix.starts_with('_') || suffix.is_empty() {
        2
    } else {
        1
    };
    (primary_rank, usize::MAX - entry.stem.len())
}

pub(crate) fn render_normalized_claim_compat_toml(row: &ClaimRecord) -> Result<String> {
    let mut table = if row.compat_toml_text.trim().is_empty() {
        let mut table = toml::map::Map::new();
        table.insert("id".to_string(), Value::String(row.id.clone()));
        table.insert(
            "statement".to_string(),
            Value::String(row.statement.clone()),
        );
        table.insert("status".to_string(), Value::String(row.status.clone()));
        table.insert(
            "where_stated".to_string(),
            Value::String(row.where_stated.clone()),
        );
        table.insert(
            "last_verified".to_string(),
            Value::String(row.last_verified.clone()),
        );
        if let Some(status_note) = &row.status_note {
            table.insert(
                "status_note".to_string(),
                Value::String(status_note.clone()),
            );
        }
        table
    } else {
        toml::from_str::<toml::map::Map<String, Value>>(&row.compat_toml_text)
            .context("parse normalized claim compat row")?
    };
    table.insert("status".to_string(), Value::String(row.status.clone()));
    match &row.status_note {
        Some(status_note) if !status_note.trim().is_empty() => {
            table.insert(
                "status_note".to_string(),
                Value::String(status_note.clone()),
            );
        }
        _ => {
            table.remove("status_note");
        }
    }
    match &row.formal_proof {
        Some(formal_proof) if !formal_proof.trim().is_empty() => {
            table.insert(
                "formal_proof".to_string(),
                Value::String(formal_proof.clone()),
            );
        }
        _ => {
            table.remove("formal_proof");
        }
    }
    render_toml_table(&table)
}

pub(crate) fn render_normalized_insight_compat_toml(
    table: &toml::map::Map<String, Value>,
    raw_status: Option<&str>,
) -> Result<String> {
    let mut table = table.clone();
    match raw_status.map(normalize_insight_status) {
        Some(status) if !status.trim().is_empty() => {
            table.insert("status".to_string(), Value::String(status.to_string()));
        }
        _ => {
            table.remove("status");
        }
    }
    render_toml_table(&table)
}

pub(crate) fn extract_proof_paths(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = text;
    while let Some(start) = cursor.find("proofs/") {
        let candidate = &cursor[start..];
        let mut end = 0usize;
        for ch in candidate.chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '/' | '_' | '-' | '.') {
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        let path = candidate[..end].trim_matches('`').trim_matches('"').trim();
        if path.ends_with(".v") {
            out.push(path.to_string());
        }
        cursor = &candidate[end..];
    }
    out
}

pub(crate) fn link_claims_for_proof(
    proof_path: &Utf8PathBuf,
    stem: &str,
    claims: &[ClaimRecord],
) -> Vec<String> {
    let proof_path_str = proof_path.as_str();
    let normalized_claim_id = normalized_claim_id_from_theorem_stem(stem);
    let mut out = Vec::new();
    for claim in claims {
        let matches = claim.id == stem
            || normalized_claim_id.as_deref() == Some(&claim.id)
            || claim
                .formal_proof
                .as_deref()
                .map(|path| path.trim() == proof_path_str)
                .unwrap_or(false)
            || claim.where_stated.contains(proof_path_str)
            || claim
                .status_note
                .as_deref()
                .map(|note| note.contains(proof_path_str))
                .unwrap_or(false);
        if matches {
            out.push(claim.id.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

pub(crate) fn normalized_claim_id_from_theorem_stem(stem: &str) -> Option<String> {
    let suffix = stem.strip_prefix('C')?;
    let digits = suffix
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    Some(format!("C-{digits}"))
}
