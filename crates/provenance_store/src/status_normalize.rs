//! Status-token normalization and status-note merging helpers.
//!
//! - `normalize_claim_record`: canonicalizes `ClaimRecord.status` against
//!   `CANONICAL_CLAIM_STATUSES`, captures the pre-normalization token as
//!   a status_note (`merge_status_note`), and re-renders the cached
//!   compat_toml_text via `render_normalized_claim_compat_toml`.
//! - `normalize_claim_status`: maps free-form status tokens onto the
//!   canonical set with structured legacy-token capture.
//! - `normalize_insight_status`: maps insight tokens onto the canonical
//!   set (verified, open, superseded, cross-validation-complete, partial).
//! - `match_case_insensitive`: case-insensitive constant-set lookup.
//! - `merge_status_note`: appends a 'Legacy status token: <raw>' tag to
//!   the existing note when canonicalization stripped a legacy token.

use anyhow::Result;
use provenance_core::ClaimRecord;

use super::{
    migrations::{CANONICAL_CLAIM_STATUSES, CANONICAL_INSIGHT_STATUSES},
    render_normalized_claim_compat_toml,
};

pub(crate) fn normalize_claim_record(claim: &mut ClaimRecord) -> Result<()> {
    let (canonical_status, legacy_status_note) = normalize_claim_status(&claim.status);
    claim.status = canonical_status;
    claim.status_note = merge_status_note(claim.status_note.take(), legacy_status_note);
    claim.compat_toml_text = render_normalized_claim_compat_toml(claim)?;
    Ok(())
}

pub(crate) fn normalize_claim_status(raw: &str) -> (String, Option<String>) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return (String::new(), None);
    }
    if let Some(canonical) = match_case_insensitive(trimmed, CANONICAL_CLAIM_STATUSES) {
        return (canonical.to_string(), None);
    }
    if let Some(paren_idx) = trimmed.find('(') {
        let base = trimmed[..paren_idx].trim();
        if let Some(canonical) = match_case_insensitive(base, CANONICAL_CLAIM_STATUSES) {
            return (canonical.to_string(), Some(trimmed.to_string()));
        }
    }
    match trimmed {
        "Open" | "Pending" | "Active" | "Proposed" | "Deferred" | "Speculative" => {
            ("Provisional".to_string(), Some(trimmed.to_string()))
        }
        "Conjecture" => ("Theoretical".to_string(), Some(trimmed.to_string())),
        "Falsified" => ("Refuted".to_string(), Some(trimmed.to_string())),
        "Closed/Verified" => ("Verified".to_string(), Some(trimmed.to_string())),
        "Closed/Falsified" => ("Closed/Refuted".to_string(), Some(trimmed.to_string())),
        "Closed/Methodology-Mismatch" => (
            "Closed/Methodology-Insufficient".to_string(),
            Some(trimmed.to_string()),
        ),
        _ => (trimmed.to_string(), None),
    }
}

pub(crate) fn normalize_insight_status(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(canonical) = match_case_insensitive(trimmed, CANONICAL_INSIGHT_STATUSES) {
        return canonical;
    }
    match trimmed {
        "Active" | "Proposed" | "Speculative" => "open",
        "Verified" => "verified",
        "Superseded" => "superseded",
        "Partial" => "partial",
        _ => trimmed,
    }
}

pub(crate) fn match_case_insensitive<'a>(raw: &str, allowed: &'a [&'a str]) -> Option<&'a str> {
    allowed
        .iter()
        .copied()
        .find(|candidate| candidate.eq_ignore_ascii_case(raw))
}

pub(crate) fn merge_status_note(
    existing: Option<String>,
    legacy_status: Option<String>,
) -> Option<String> {
    match (existing, legacy_status) {
        (existing, None) => existing,
        (None, Some(raw)) => Some(format!("Legacy status token: {raw}")),
        (Some(existing), Some(raw)) => {
            let legacy_note = format!("Legacy status token: {raw}");
            if existing.contains(&legacy_note) || existing.contains(&raw) {
                Some(existing)
            } else {
                Some(format!("{existing} | {legacy_note}"))
            }
        }
    }
}
