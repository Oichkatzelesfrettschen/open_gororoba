//! Canonical metadata schema for claim-audit trackers.
//!
//! Single source of truth for status tokens, task tokens, and domain categories.
//! Replaces: src/verification/claims_metadata_schema.py

/// Canonical status tokens for CLAIMS_EVIDENCE_MATRIX.md rows.
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

/// Check if a status token is canonical.
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
}
