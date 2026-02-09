//! Claims audit reports.
//!
//! Replaces 7 Python analysis scripts:
//! - claims_id_inventory.py
//! - claims_status_inventory.py
//! - claims_staleness_report.py
//! - claims_status_contradictions.py
//! - claims_bold_tokens_inventory.py
//! - claims_priority_ranking.py
//! - claims_batch_backlog.py

use regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::LazyLock;

use super::parser::{parse_claim_rows, shorten, ClaimRow};
use super::schema::{is_canonical_status, is_open_status, CANONICAL_CLAIMS_STATUS_TOKENS};

// --- ID Inventory Report ---

/// Result of claim ID inventory analysis.
#[derive(Debug)]
pub struct IdInventory {
    pub total: usize,
    pub min_id: u32,
    pub max_id: u32,
    pub duplicates: Vec<(String, usize)>,
    pub gaps: Vec<u32>,
}

/// Analyze claim IDs for duplicates and gaps.
pub fn id_inventory(claims: &[ClaimRow]) -> IdInventory {
    if claims.is_empty() {
        return IdInventory {
            total: 0,
            min_id: 0,
            max_id: 0,
            duplicates: Vec::new(),
            gaps: Vec::new(),
        };
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut nums: Vec<u32> = Vec::new();
    for c in claims {
        *counts.entry(c.claim_id.clone()).or_default() += 1;
        nums.push(c.claim_num);
    }

    let min_id = *nums.iter().min().unwrap_or(&0);
    let max_id = *nums.iter().max().unwrap_or(&0);

    let mut duplicates: Vec<(String, usize)> = counts.into_iter().filter(|(_, n)| *n > 1).collect();
    duplicates.sort();

    let present: BTreeSet<u32> = nums.iter().copied().collect();
    let gaps: Vec<u32> = (min_id..=max_id).filter(|n| !present.contains(n)).collect();

    IdInventory {
        total: claims.len(),
        min_id,
        max_id,
        duplicates,
        gaps,
    }
}

/// Render ID inventory report as markdown.
pub fn render_id_inventory(inv: &IdInventory, matrix_path: &str) -> String {
    let mut out = String::new();
    out.push_str("# Claims ID Inventory\n\n");
    out.push_str(&format!("Matrix: `{matrix_path}`\n"));
    out.push_str(&format!("Total claim rows parsed: {}\n\n", inv.total));

    if inv.total == 0 {
        out.push_str("No claim IDs parsed.\n");
        return out;
    }

    out.push_str(&format!("- Min ID: C-{:03}\n", inv.min_id));
    out.push_str(&format!("- Max ID: C-{:03}\n\n", inv.max_id));

    out.push_str("## Duplicate claim IDs\n\n");
    if inv.duplicates.is_empty() {
        out.push_str("- None\n");
    } else {
        for (id, count) in &inv.duplicates {
            out.push_str(&format!("- {id} (count={count})\n"));
        }
    }
    out.push('\n');

    out.push_str("## Gaps within min..max\n\n");
    if inv.gaps.is_empty() {
        out.push_str("- None\n");
    } else {
        for chunk in inv.gaps.chunks(24) {
            let ids: Vec<String> = chunk.iter().map(|n| format!("C-{n:03}")).collect();
            out.push_str(&format!("- {}\n", ids.join(", ")));
        }
    }
    out.push('\n');
    out
}

// --- Status Inventory Report ---

/// Result of status distribution analysis.
#[derive(Debug)]
pub struct StatusInventory {
    pub total: usize,
    pub counts: BTreeMap<String, usize>,
    pub open_claims: Vec<OpenClaim>,
    pub missing_where_paths: Vec<String>,
    pub legacy_inline: Vec<String>,
}

/// An open claim entry for the status report.
#[derive(Debug)]
pub struct OpenClaim {
    pub claim_id: String,
    pub status_token: String,
    pub last_verified: String,
    pub claim_short: String,
}

/// Analyze claims by status distribution and identify open claims.
pub fn status_inventory(claims: &[ClaimRow]) -> StatusInventory {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut open_claims = Vec::new();
    let mut missing_where_paths = Vec::new();
    let mut legacy_inline = Vec::new();

    for c in claims {
        let token = if is_canonical_status(&c.status_token) {
            c.status_token.clone()
        } else {
            "Other".to_string()
        };
        *counts.entry(token.clone()).or_default() += 1;

        if is_open_status(&c.status_token) {
            open_claims.push(OpenClaim {
                claim_id: c.claim_id.clone(),
                status_token: c.status_token.clone(),
                last_verified: c.last_verified_date.clone().unwrap_or_default(),
                claim_short: shorten(&c.claim_text, 110),
            });
        }

        // Hygiene: check "Where stated" has a path pointer.
        let ws = &c.where_stated;
        let has_path = ws.contains('`')
            || ws.contains("src/")
            || ws.contains("docs/")
            || ws.contains("data/")
            || ws.contains("crates/");
        if !has_path && !ws.contains("multiple docs") {
            missing_where_paths.push(c.claim_id.clone());
        }

        // Legacy: embedded experiment tags in claim text.
        if c.claim_text.contains("\\|v") {
            legacy_inline.push(c.claim_id.clone());
        }
    }

    // Sort open claims by date (oldest first), then by ID.
    open_claims.sort_by(|a, b| {
        a.last_verified
            .cmp(&b.last_verified)
            .then(a.claim_id.cmp(&b.claim_id))
    });

    StatusInventory {
        total: claims.len(),
        counts,
        open_claims,
        missing_where_paths,
        legacy_inline,
    }
}

/// Render status inventory report as markdown.
pub fn render_status_inventory(inv: &StatusInventory, matrix_path: &str) -> String {
    let mut out = String::new();
    out.push_str("# Claims Status Inventory\n\n");
    out.push_str(&format!("Matrix: `{matrix_path}`\n"));
    out.push_str(&format!("Total claims parsed: {}\n\n", inv.total));

    out.push_str("## Counts by canonical status token\n\n");
    for &token in CANONICAL_CLAIMS_STATUS_TOKENS {
        if let Some(&count) = inv.counts.get(token) {
            out.push_str(&format!("- {token}: {count}\n"));
        }
    }
    if let Some(&other) = inv.counts.get("Other") {
        out.push_str(&format!("- Other: {other}\n"));
    }
    out.push('\n');

    out.push_str("## Open claims (oldest first)\n\n");
    for oc in &inv.open_claims {
        let lv = if oc.last_verified.is_empty() {
            "UNKNOWN_DATE"
        } else {
            &oc.last_verified
        };
        out.push_str(&format!(
            "- {} ({}, last_verified={}): {}\n",
            oc.claim_id, oc.status_token, lv, oc.claim_short
        ));
    }
    out.push('\n');

    out.push_str("## Metadata hygiene notes\n\n");
    out.push_str(&format!(
        "- Rows with missing/weak `Where stated` pointers: {}\n",
        inv.missing_where_paths.len()
    ));
    out.push_str(&format!(
        "- Rows with legacy inline experiment tags: {}\n",
        inv.legacy_inline.len()
    ));
    out.push('\n');
    out
}

// --- Staleness Report ---

/// Result of staleness analysis.
#[derive(Debug)]
pub struct StalenessReport {
    pub total: usize,
    pub missing_date: Vec<String>,
    pub invalid_date: Vec<String>,
    pub stale_claims: Vec<(String, String)>, // (claim_id, date)
}

/// Analyze claims for missing or stale verification dates.
pub fn staleness_report(claims: &[ClaimRow], stale_before: &str) -> StalenessReport {
    let mut missing_date = Vec::new();
    let mut invalid_date = Vec::new();
    let mut stale_claims = Vec::new();

    for c in claims {
        if c.last_verified.is_empty() {
            missing_date.push(c.claim_id.clone());
            continue;
        }
        match &c.last_verified_date {
            None => {
                invalid_date.push(c.claim_id.clone());
            }
            Some(date) => {
                if date.as_str() < stale_before {
                    stale_claims.push((c.claim_id.clone(), date.clone()));
                }
            }
        }
    }

    StalenessReport {
        total: claims.len(),
        missing_date,
        invalid_date,
        stale_claims,
    }
}

/// Render staleness report as markdown.
pub fn render_staleness_report(
    report: &StalenessReport,
    stale_before: &str,
    matrix_path: &str,
) -> String {
    let mut out = String::new();
    out.push_str("# Claims Staleness Report\n\n");
    out.push_str(&format!("Matrix: `{matrix_path}`\n"));
    out.push_str(&format!("Total claims: {}\n", report.total));
    out.push_str(&format!("Stale threshold: before {stale_before}\n\n"));

    out.push_str(&format!(
        "## Missing date: {} claims\n\n",
        report.missing_date.len()
    ));
    for chunk in report.missing_date.chunks(16) {
        let ids = chunk.join(", ");
        out.push_str(&format!("- {ids}\n"));
    }
    out.push('\n');

    out.push_str(&format!(
        "## Invalid date format: {} claims\n\n",
        report.invalid_date.len()
    ));
    for chunk in report.invalid_date.chunks(16) {
        let ids = chunk.join(", ");
        out.push_str(&format!("- {ids}\n"));
    }
    out.push('\n');

    out.push_str(&format!(
        "## Stale (before {}): {} claims\n\n",
        stale_before,
        report.stale_claims.len()
    ));
    for (id, date) in &report.stale_claims {
        out.push_str(&format!("- {id} (last verified: {date})\n"));
    }
    out.push('\n');
    out
}

// --- Status Contradictions ---

static BOLD_TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\*\*([^*]+)\*\*").expect("valid regex"));

/// A claim with multiple status tokens in its status cell.
#[derive(Debug)]
pub struct StatusContradiction {
    pub claim_id: String,
    pub tokens: Vec<String>,
    pub lineno: usize,
}

/// Find claims where the status cell contains multiple bold tokens.
pub fn status_contradictions(claims: &[ClaimRow]) -> Vec<StatusContradiction> {
    let mut contradictions = Vec::new();
    for c in claims {
        let tokens: Vec<String> = BOLD_TOKEN_RE
            .captures_iter(&c.status_cell)
            .map(|cap| cap[1].trim().to_string())
            .collect();
        if tokens.len() > 1 {
            contradictions.push(StatusContradiction {
                claim_id: c.claim_id.clone(),
                tokens,
                lineno: c.lineno,
            });
        }
    }
    contradictions
}

/// Render contradictions report as markdown.
pub fn render_contradictions(contras: &[StatusContradiction], matrix_path: &str) -> String {
    let mut out = String::new();
    out.push_str("# Claims Status Contradictions\n\n");
    out.push_str(&format!("Matrix: `{matrix_path}`\n\n"));

    if contras.is_empty() {
        out.push_str("No contradictions found.\n");
    } else {
        out.push_str(&format!(
            "Found {} claims with multiple status tokens:\n\n",
            contras.len()
        ));
        for c in contras {
            out.push_str(&format!(
                "- {} (line {}): {}\n",
                c.claim_id,
                c.lineno,
                c.tokens.join(", ")
            ));
        }
    }
    out.push('\n');
    out
}

// --- Bold Tokens Inventory ---

/// Non-canonical bold token found in a claim row.
#[derive(Debug)]
pub struct NonCanonicalToken {
    pub claim_id: String,
    pub token: String,
    pub column: usize,
}

/// Find non-canonical bold tokens across all claim cells.
pub fn bold_tokens_inventory(claims: &[ClaimRow]) -> Vec<NonCanonicalToken> {
    let mut found = Vec::new();
    for c in claims {
        let cells = [
            (1, &c.claim_text),
            (2, &c.where_stated),
            (3, &c.status_cell),
            (4, &c.last_verified),
            (5, &c.evidence_notes),
        ];
        for (col, cell) in cells {
            for cap in BOLD_TOKEN_RE.captures_iter(cell) {
                let token = cap[1].trim().to_string();
                if !is_canonical_status(&token) {
                    found.push(NonCanonicalToken {
                        claim_id: c.claim_id.clone(),
                        token,
                        column: col,
                    });
                }
            }
        }
    }
    found
}

/// Render bold tokens inventory as markdown.
pub fn render_bold_tokens(tokens: &[NonCanonicalToken], matrix_path: &str) -> String {
    let mut out = String::new();
    out.push_str("# Non-Canonical Bold Tokens Inventory\n\n");
    out.push_str(&format!("Matrix: `{matrix_path}`\n\n"));

    if tokens.is_empty() {
        out.push_str("No non-canonical bold tokens found.\n");
    } else {
        // Group by token.
        let mut by_token: BTreeMap<&str, Vec<&NonCanonicalToken>> = BTreeMap::new();
        for t in tokens {
            by_token.entry(&t.token).or_default().push(t);
        }
        out.push_str(&format!(
            "Found {} non-canonical bold tokens ({} unique):\n\n",
            tokens.len(),
            by_token.len()
        ));
        for (token, entries) in &by_token {
            let ids: Vec<&str> = entries.iter().map(|e| e.claim_id.as_str()).collect();
            out.push_str(&format!("- **{token}**: {}\n", ids.join(", ")));
        }
    }
    out.push('\n');
    out
}

// --- Priority Ranking ---

/// Claim with frequency-based priority ranking.
#[derive(Debug)]
pub struct PriorityEntry {
    pub claim_id: String,
    pub status_token: String,
    pub mention_count: usize,
}

/// Rank open claims by how frequently they are referenced across documentation.
///
/// `doc_corpus` is the concatenated text of all docs/*.md files.
pub fn priority_ranking(claims: &[ClaimRow], doc_corpus: &str) -> Vec<PriorityEntry> {
    static CLAIM_REF_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\bC-\d{3}\b").expect("valid regex"));

    // Count mentions of each claim ID across the corpus.
    let mut mention_counts: HashMap<String, usize> = HashMap::new();
    for mat in CLAIM_REF_RE.find_iter(doc_corpus) {
        *mention_counts.entry(mat.as_str().to_string()).or_default() += 1;
    }

    let mut entries: Vec<PriorityEntry> = claims
        .iter()
        .filter(|c| is_open_status(&c.status_token))
        .map(|c| PriorityEntry {
            claim_id: c.claim_id.clone(),
            status_token: c.status_token.clone(),
            mention_count: mention_counts.get(&c.claim_id).copied().unwrap_or(0),
        })
        .collect();

    // Sort by mention count descending, then by claim ID.
    entries.sort_by(|a, b| {
        b.mention_count
            .cmp(&a.mention_count)
            .then(a.claim_id.cmp(&b.claim_id))
    });

    entries
}

/// Render priority ranking as markdown.
pub fn render_priority_ranking(entries: &[PriorityEntry], matrix_path: &str) -> String {
    let mut out = String::new();
    out.push_str("# Claims Priority Ranking\n\n");
    out.push_str(&format!("Matrix: `{matrix_path}`\n"));
    out.push_str(&format!("Open claims ranked: {}\n\n", entries.len()));

    out.push_str("| Rank | Claim | Status | Mentions |\n");
    out.push_str("| --- | --- | --- | --- |\n");
    for (i, e) in entries.iter().enumerate() {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            i + 1,
            e.claim_id,
            e.status_token,
            e.mention_count
        ));
    }
    out.push('\n');
    out
}

// --- Run All Audits ---

/// Run all audit reports and return combined output.
pub fn run_all_audits(
    matrix_text: &str,
    matrix_path: &str,
    doc_corpus: &str,
    stale_before: &str,
) -> String {
    let claims = parse_claim_rows(matrix_text);
    let mut output = String::new();

    // ID inventory.
    let inv = id_inventory(&claims);
    output.push_str(&render_id_inventory(&inv, matrix_path));
    output.push_str("---\n\n");

    // Status inventory.
    let status = status_inventory(&claims);
    output.push_str(&render_status_inventory(&status, matrix_path));
    output.push_str("---\n\n");

    // Staleness.
    let stale = staleness_report(&claims, stale_before);
    output.push_str(&render_staleness_report(&stale, stale_before, matrix_path));
    output.push_str("---\n\n");

    // Contradictions.
    let contras = status_contradictions(&claims);
    output.push_str(&render_contradictions(&contras, matrix_path));
    output.push_str("---\n\n");

    // Bold tokens.
    let tokens = bold_tokens_inventory(&claims);
    output.push_str(&render_bold_tokens(&tokens, matrix_path));
    output.push_str("---\n\n");

    // Priority ranking.
    let prio = priority_ranking(&claims, doc_corpus);
    output.push_str(&render_priority_ranking(&prio, matrix_path));

    output
}

/// Collect text from all .md files under a directory for priority ranking.
pub fn collect_doc_corpus(docs_dir: &Path) -> String {
    let mut corpus = String::new();
    if let Ok(entries) = std::fs::read_dir(docs_dir) {
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
            .map(|e| e.path())
            .collect();
        paths.sort();
        for path in paths {
            if let Ok(text) = std::fs::read_to_string(&path) {
                corpus.push_str(&text);
                corpus.push('\n');
            }
        }
    }
    corpus
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::parser::parse_claim_rows;

    fn sample_matrix() -> &'static str {
        "\
| Claim | Description | Where | Status | Verified | Notes |
| --- | --- | --- | --- | --- | --- |
| C-001 | First claim | `src/lib.rs` | **Verified** | 2026-01-15 | tested |
| C-002 | Second claim | `docs/foo.md` | **Speculative** | 2025-06-01 | pending |
| C-003 | Third claim | `src/bar.rs` | **Unverified** | 2026-02-01 | todo |
| C-005 | Fifth claim | no path here | **Verified** | 2026-01-20 | done |
"
    }

    #[test]
    fn test_id_inventory_basic() {
        let claims = parse_claim_rows(sample_matrix());
        let inv = id_inventory(&claims);
        assert_eq!(inv.total, 4);
        assert_eq!(inv.min_id, 1);
        assert_eq!(inv.max_id, 5);
        assert!(inv.duplicates.is_empty());
        assert_eq!(inv.gaps, vec![4]); // C-004 missing
    }

    #[test]
    fn test_status_inventory_basic() {
        let claims = parse_claim_rows(sample_matrix());
        let inv = status_inventory(&claims);
        assert_eq!(inv.total, 4);
        assert_eq!(inv.counts.get("Verified"), Some(&2));
        assert_eq!(inv.counts.get("Speculative"), Some(&1));
        assert_eq!(inv.counts.get("Unverified"), Some(&1));
        assert_eq!(inv.open_claims.len(), 2); // Speculative + Unverified
    }

    #[test]
    fn test_staleness_report_basic() {
        let claims = parse_claim_rows(sample_matrix());
        let report = staleness_report(&claims, "2026-01-01");
        assert_eq!(report.missing_date.len(), 0);
        assert_eq!(report.stale_claims.len(), 1); // C-002 is before 2026-01-01
        assert_eq!(report.stale_claims[0].0, "C-002");
    }

    #[test]
    fn test_contradictions_none() {
        let claims = parse_claim_rows(sample_matrix());
        let contras = status_contradictions(&claims);
        assert!(contras.is_empty());
    }

    #[test]
    fn test_contradictions_detected() {
        let text = "\
| Claim | Description | Where | Status | Verified | Notes |
| --- | --- | --- | --- | --- | --- |
| C-001 | Claim | `src/x.rs` | **Verified** **Refuted** | 2026-01-01 | conflict |
";
        let claims = parse_claim_rows(text);
        let contras = status_contradictions(&claims);
        assert_eq!(contras.len(), 1);
        assert_eq!(contras[0].tokens, vec!["Verified", "Refuted"]);
    }

    #[test]
    fn test_priority_ranking() {
        let claims = parse_claim_rows(sample_matrix());
        let corpus = "See C-002 and C-002 and C-003 for details. C-001 is verified.";
        let prio = priority_ranking(&claims, corpus);
        // Only open claims: C-002 (Speculative) and C-003 (Unverified).
        assert_eq!(prio.len(), 2);
        assert_eq!(prio[0].claim_id, "C-002"); // 2 mentions
        assert_eq!(prio[0].mention_count, 2);
        assert_eq!(prio[1].claim_id, "C-003"); // 1 mention
    }

    #[test]
    fn test_missing_where_paths() {
        let claims = parse_claim_rows(sample_matrix());
        let inv = status_inventory(&claims);
        assert_eq!(inv.missing_where_paths, vec!["C-005"]); // "no path here"
    }
}
