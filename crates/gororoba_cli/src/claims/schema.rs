//! Canonical metadata schema for claim-audit trackers.
//!
//! Single source of truth for status tokens, task tokens, and domain categories.
//! Replaces: src/verification/claims_metadata_schema.py
//!
//! Two status vocabularies coexist:
//! - `CANONICAL_CLAIMS_STATUS_TOKENS`: legacy markdown-era tokens (11)
//! - `TOML_CLAIM_STATUSES`: TOML-era canonical tokens (16)
//!
//! The consolidation pipeline normalizes all statuses to `TOML_CLAIM_STATUSES`.

/// Canonical status tokens for CLAIMS_EVIDENCE_MATRIX.md rows (legacy).
/// These appear as `**Token**` in the Status column.
pub const CANONICAL_CLAIMS_STATUS_TOKENS: &[&str] = &[
    "Verified",
    "Partially verified",
    "Unverified",
    "Speculative",
    "Modeled",
    "Literature",
    "Theoretical",
    "Not supported",
    "Refuted",
    "Clarified",
    "Established",
];

/// Canonical TOML claim status tokens (authoritative for registry/claims.toml).
///
/// Derived from the actual status distribution in the registry:
/// - 8 terminal states for resolved claims
/// - 3 open/partial states for in-progress claims
/// - 5 Closed/* sub-statuses for negative/obstructed outcomes
pub const TOML_CLAIM_STATUSES: &[&str] = &[
    // Terminal positive
    "Verified",
    "Established",
    // Terminal negative
    "Refuted",
    // Partial/open
    "Partial",
    "Provisional",
    "Theoretical",
    "Inconclusive",
    // Lifecycle
    "Superseded",
    // Closed sub-statuses
    "Closed/Negative-Result",
    "Closed/Obstructed",
    "Closed/Research-Program",
    "Closed/Toy",
    "Closed/Analogy",
    "Closed/Source-Insufficient",
    "Closed/Methodology-Insufficient",
    "Closed/Refuted",
];

/// Status tokens that indicate a claim is still "open" (needs work).
pub const OPEN_STATUS_TOKENS: &[&str] = &[
    "Unverified",
    "Partially verified",
    "Speculative",
    "Modeled",
    "Literature",
    "Theoretical",
    "Clarified",
];

/// Canonical task status tokens for CLAIMS_TASKS.md rows.
pub const CANONICAL_TASK_STATUS_TOKENS: &[&str] = &[
    "TODO",
    "IN PROGRESS",
    "PARTIAL",
    "DONE",
    "REFUTED",
    "DEFERRED",
    "BLOCKED",
];

/// Canonical domain categories for CLAIMS_DOMAIN_MAP.csv.
pub const CANONICAL_DOMAINS: &[&str] = &[
    "meta",
    "algebra",
    "spectral",
    "holography",
    "open-systems",
    "tensor-networks",
    "cosmology",
    "gravitational-waves",
    "stellar-cartography",
    "materials",
    "engineering",
    "datasets",
    "visualization",
    "cpp",
    "coq",
    "legacy",
];

/// Normalize a raw status string to the canonical TOML status token.
///
/// Rules:
/// 1. Case normalization: "verified" -> "Verified"
/// 2. Variant collapse: "Verified (algebraic)" -> "Verified"
///    (the parenthetical detail is returned as the second element)
/// 3. Merge redundant: "Closed/Refuted" stays as-is (distinct from plain "Refuted")
///
/// Returns (canonical_status, optional_status_note).
pub fn normalize_status(raw: &str) -> (String, Option<String>) {
    let trimmed = raw.trim();

    // Check for parenthetical variants: "Verified (algebraic)" -> base + note
    if let Some(paren_start) = trimmed.find('(') {
        let base = trimmed[..paren_start].trim();
        let note = trimmed[paren_start..]
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim()
            .to_string();
        let (canonical, _) = normalize_status(base);
        let merged_note = if note.is_empty() { None } else { Some(note) };
        return (canonical, merged_note);
    }

    // Case normalization: title-case the first character
    let title_cased = if trimmed.starts_with(|c: char| c.is_ascii_lowercase()) {
        let mut chars = trimmed.chars();
        match chars.next() {
            Some(first) => {
                let upper: String = first.to_uppercase().collect();
                format!("{upper}{}", chars.as_str())
            }
            None => trimmed.to_string(),
        }
    } else {
        trimmed.to_string()
    };

    // Direct match against canonical set
    if TOML_CLAIM_STATUSES.contains(&title_cased.as_str()) {
        return (title_cased, None);
    }

    // Legacy mappings
    match title_cased.as_str() {
        "Open" | "Pending" => ("Provisional".to_string(), Some(title_cased)),
        "Partially verified" | "Partial" => ("Partial".to_string(), None),
        "Not supported" => ("Refuted".to_string(), Some("Not supported".to_string())),
        "Unverified" | "Speculative" | "Modeled" | "Literature" | "Clarified" => {
            ("Provisional".to_string(), Some(title_cased))
        }
        _ => {
            // Unknown status: keep as-is but flag
            (title_cased, Some("unrecognized status".to_string()))
        }
    }
}

/// Check if a status token is in the TOML canonical set.
pub fn is_toml_canonical_status(token: &str) -> bool {
    TOML_CLAIM_STATUSES.contains(&token)
}

/// Check if a status token is canonical (legacy markdown set).
pub fn is_canonical_status(token: &str) -> bool {
    CANONICAL_CLAIMS_STATUS_TOKENS.contains(&token)
}

/// Check if a status token indicates an open claim.
pub fn is_open_status(token: &str) -> bool {
    OPEN_STATUS_TOKENS.contains(&token)
}

/// Check if a task status token is canonical.
pub fn is_canonical_task_status(token: &str) -> bool {
    CANONICAL_TASK_STATUS_TOKENS.contains(&token)
}

/// Check if a domain is canonical.
pub fn is_canonical_domain(domain: &str) -> bool {
    CANONICAL_DOMAINS.contains(&domain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_status_tokens_count() {
        assert_eq!(CANONICAL_CLAIMS_STATUS_TOKENS.len(), 11);
    }

    #[test]
    fn test_toml_status_tokens_count() {
        assert_eq!(TOML_CLAIM_STATUSES.len(), 16);
    }

    #[test]
    fn test_canonical_task_tokens_count() {
        assert_eq!(CANONICAL_TASK_STATUS_TOKENS.len(), 7);
    }

    #[test]
    fn test_canonical_domains_count() {
        assert_eq!(CANONICAL_DOMAINS.len(), 16);
    }

    #[test]
    fn test_is_canonical_status() {
        assert!(is_canonical_status("Verified"));
        assert!(is_canonical_status("Refuted"));
        assert!(!is_canonical_status("Invalid"));
        assert!(!is_canonical_status(""));
    }

    #[test]
    fn test_open_status_subset_of_canonical() {
        for &token in OPEN_STATUS_TOKENS {
            assert!(
                is_canonical_status(token),
                "Open token {token:?} must be canonical"
            );
        }
    }

    #[test]
    fn test_is_open_status() {
        assert!(is_open_status("Unverified"));
        assert!(is_open_status("Speculative"));
        assert!(!is_open_status("Verified"));
        assert!(!is_open_status("Refuted"));
    }

    #[test]
    fn test_normalize_status_case() {
        let (s, n) = normalize_status("verified");
        assert_eq!(s, "Verified");
        assert!(n.is_none());
    }

    #[test]
    fn test_normalize_status_parenthetical() {
        let (s, n) = normalize_status("Verified (algebraic)");
        assert_eq!(s, "Verified");
        assert_eq!(n.unwrap(), "algebraic");
    }

    #[test]
    fn test_normalize_status_closed_variants() {
        let (s, n) = normalize_status("Closed/Negative-Result");
        assert_eq!(s, "Closed/Negative-Result");
        assert!(n.is_none());
    }

    #[test]
    fn test_normalize_status_legacy_open() {
        let (s, n) = normalize_status("Open");
        assert_eq!(s, "Provisional");
        assert_eq!(n.unwrap(), "Open");
    }

    #[test]
    fn test_normalize_status_identity() {
        for &canonical in TOML_CLAIM_STATUSES {
            let (s, n) = normalize_status(canonical);
            assert_eq!(s, canonical, "Canonical token should be idempotent");
            assert!(n.is_none(), "Canonical token should produce no note");
        }
    }
}
