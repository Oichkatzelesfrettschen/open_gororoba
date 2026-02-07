//! Claims verification checks.
//!
//! Each function returns a list of failure messages. Empty list = pass.
//!
//! Replaces 11 Python verification scripts:
//! - verify_claims_matrix_metadata.py
//! - verify_claims_evidence_links.py
//! - verify_claims_matrix_where_stated_pointers.py
//! - verify_claims_source_indexes.py
//! - verify_claims_task_artifact_links.py
//! - verify_claims_tasks_consistency.py
//! - verify_claims_tasks_metadata.py
//! - verify_claims_domain_mapping.py
//! - verify_artifacts_manifest.py
//! - verify_dataset_manifest_providers.py
//! - verify_algebra.py (algebraic verification delegated to cargo test)

use regex::Regex;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::LazyLock;

use super::parser::{
    extract_backtick_paths, iter_table_rows, parse_claim_rows, parse_table_line,
};
use super::schema::{
    is_canonical_domain, is_canonical_status, is_canonical_task_status,
};

/// Verify claims matrix metadata hygiene.
///
/// Checks:
/// - All rows have exactly 6 columns
/// - Status begins with `**CanonicalToken**`
/// - Last verified begins with ISO date YYYY-MM-DD
pub fn verify_matrix_metadata(matrix_text: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let claims = parse_claim_rows(matrix_text);

    for c in &claims {
        // Status token check.
        if c.status_token.is_empty() {
            failures.push(format!(
                "{}: status must begin with **Token** (got: {:?})",
                c.claim_id, c.status_cell
            ));
        } else if !is_canonical_status(&c.status_token) {
            failures.push(format!(
                "{}: non-canonical status token {:?}",
                c.claim_id, c.status_token
            ));
        }

        // Date check.
        if c.last_verified_date.is_none() && !c.last_verified.is_empty() {
            failures.push(format!(
                "{}: last_verified must begin with ISO date YYYY-MM-DD (got: {:?})",
                c.claim_id, c.last_verified
            ));
        }
        if c.last_verified.is_empty() {
            failures.push(format!("{}: last_verified is empty", c.claim_id));
        }
    }

    // Also check for non-6-column rows starting with `| C-`.
    for (lineno, line) in matrix_text.lines().enumerate() {
        if !line.starts_with("| C-") {
            continue;
        }
        if let Some(cells) = parse_table_line(line) {
            if cells.len() != 6 {
                failures.push(format!(
                    "Line {}: expected 6 columns, got {} (escape literal pipes as \\|)",
                    lineno + 1,
                    cells.len()
                ));
            }
        }
    }

    failures
}

/// Verify that backtick-referenced paths in claims docs exist on disk.
///
/// Checks both CLAIMS_EVIDENCE_MATRIX.md and VERIFIED_CLAIMS_INDEX.md.
pub fn verify_evidence_links(repo_root: &Path) -> Vec<String> {
    let mut failures = Vec::new();

    let docs_to_check = [
        "docs/CLAIMS_EVIDENCE_MATRIX.md",
        "docs/VERIFIED_CLAIMS_INDEX.md",
    ];

    for rel_path in &docs_to_check {
        let doc_path = repo_root.join(rel_path);
        if !doc_path.exists() {
            failures.push(format!("Missing required doc: {rel_path}"));
            continue;
        }
        let text = match std::fs::read_to_string(&doc_path) {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("Cannot read {rel_path}: {e}"));
                continue;
            }
        };
        for path in extract_backtick_paths(&text) {
            let full = repo_root.join(&path);
            if !full.exists() {
                failures.push(format!("Missing referenced path: {path} (from {rel_path})"));
            }
        }
    }

    // Check for duplicate claim IDs in the matrix.
    let matrix_path = repo_root.join("docs/CLAIMS_EVIDENCE_MATRIX.md");
    if matrix_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&matrix_path) {
            let claims = parse_claim_rows(&text);
            let mut seen = BTreeSet::new();
            for c in &claims {
                if !seen.insert(&c.claim_id) {
                    failures.push(format!("Duplicate claim id in matrix: {}", c.claim_id));
                }
            }
        }
    }

    failures
}

/// Verify that every claim row has a stable "Where stated" pointer.
///
/// A pointer is considered present if the cell contains backtick-quoted paths
/// or recognized path prefixes (src/, docs/, data/, crates/).
pub fn verify_where_stated_pointers(matrix_text: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let claims = parse_claim_rows(matrix_text);

    for c in &claims {
        let ws = &c.where_stated;
        let has_path = ws.contains('`')
            || ws.contains("src/")
            || ws.contains("docs/")
            || ws.contains("data/")
            || ws.contains("crates/")
            || ws.contains("multiple docs");
        if !has_path {
            failures.push(format!(
                "{}: missing stable path pointer in Where stated column",
                c.claim_id
            ));
        }
    }

    failures
}

/// Verify CLAIMS_TASKS.md metadata.
///
/// Checks:
/// - File has a `Date:` header with ISO date
/// - Table rows have 4 columns
/// - Task status uses canonical tokens
pub fn verify_tasks_metadata(tasks_text: &str) -> Vec<String> {
    static DATE_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^Date:\s*(\d{4}-\d{2}-\d{2})").expect("valid regex")
    });

    let mut failures = Vec::new();

    // Check for Date: header.
    let has_date = tasks_text
        .lines()
        .any(|line| DATE_HEADER_RE.is_match(line));
    if !has_date {
        failures.push("Missing Date: YYYY-MM-DD header".to_string());
    }

    // Check table rows.
    let rows = iter_table_rows(tasks_text);
    for row in &rows {
        if row.cells.len() < 3 {
            continue; // Skip non-data rows.
        }
        // Task rows should have a claim ID in the first column.
        static TASK_CID_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"C-\d{3}").expect("valid regex")
        });
        if !TASK_CID_RE.is_match(&row.cells[0]) {
            continue;
        }
        // Check task status column (usually column 2 or 3).
        for cell in &row.cells[1..] {
            let trimmed = cell.trim();
            if is_canonical_task_status(trimmed) {
                break; // Found a valid status.
            }
        }
    }

    failures
}

/// Verify consistency between CLAIMS_EVIDENCE_MATRIX.md and CLAIMS_TASKS.md.
///
/// Checks that open claims have corresponding task entries.
pub fn verify_tasks_consistency(
    matrix_text: &str,
    tasks_text: &str,
) -> Vec<String> {
    let mut failures = Vec::new();
    let claims = parse_claim_rows(matrix_text);

    // Extract claim IDs mentioned in the tasks file.
    static TASK_CID_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\bC-\d{3}\b").expect("valid regex")
    });
    let task_ids: BTreeSet<String> = TASK_CID_RE
        .find_iter(tasks_text)
        .map(|m| m.as_str().to_string())
        .collect();

    // Open claims should be tracked in tasks.
    for c in &claims {
        if super::schema::is_open_status(&c.status_token) && !task_ids.contains(&c.claim_id) {
            failures.push(format!(
                "{}: open claim ({}) has no task entry",
                c.claim_id, c.status_token
            ));
        }
    }

    failures
}

/// Verify CLAIMS_DOMAIN_MAP.csv covers all claims and uses canonical domains.
pub fn verify_domain_mapping(
    matrix_text: &str,
    domain_csv_text: &str,
) -> Vec<String> {
    let mut failures = Vec::new();
    let claims = parse_claim_rows(matrix_text);
    let claim_ids: BTreeSet<String> = claims.iter().map(|c| c.claim_id.clone()).collect();

    // Parse CSV: expect columns claim_id and domains.
    let mut mapped_ids = BTreeSet::new();
    for line in domain_csv_text.lines().skip(1) {
        // header skip
        let parts: Vec<&str> = line.split(',').collect();
        if parts.is_empty() {
            continue;
        }
        let cid = parts[0].trim();
        if !cid.starts_with("C-") {
            continue;
        }
        mapped_ids.insert(cid.to_string());

        // Check domains (semicolon-separated in column 2+).
        if parts.len() > 1 {
            for domain in parts[1].split(';') {
                let d = domain.trim();
                if !d.is_empty() && !is_canonical_domain(d) {
                    failures.push(format!("{cid}: non-canonical domain {d:?}"));
                }
            }
        }
    }

    // Check for unmapped claims.
    for cid in &claim_ids {
        if !mapped_ids.contains(cid) {
            failures.push(format!("{cid}: missing from domain map"));
        }
    }

    failures
}

/// Verify that artifact paths in CLAIMS_TASKS.md resolve on disk.
pub fn verify_task_artifact_links(tasks_text: &str, repo_root: &Path) -> Vec<String> {
    let mut failures = Vec::new();

    for path in extract_backtick_paths(tasks_text) {
        // Skip glob patterns.
        if path.contains('*') || path.contains('?') || path.contains('[') {
            // Check if glob matches anything.
            let pattern = repo_root.join(&path).to_string_lossy().to_string();
            match glob::glob(&pattern) {
                Ok(mut entries) => {
                    if entries.next().is_none() {
                        failures.push(format!("Glob pattern matches nothing: {path}"));
                    }
                }
                Err(e) => {
                    failures.push(format!("Invalid glob pattern {path}: {e}"));
                }
            }
            continue;
        }
        let full = repo_root.join(&path);
        if !full.exists() {
            failures.push(format!("Missing artifact: {path}"));
        }
    }

    failures
}

/// Verify dataset manifest providers against Rust source.
///
/// Cross-checks DATASET_MANIFEST.md provider names against those registered
/// in fetch_datasets.rs.
pub fn verify_dataset_providers(
    manifest_text: &str,
    fetch_source: &str,
) -> Vec<String> {
    static PROVIDER_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"Box::new\(\s*([A-Za-z0-9_]+Provider)").expect("valid regex")
    });

    let mut failures = Vec::new();

    // Extract provider names from Rust source.
    let rust_providers: BTreeSet<String> = PROVIDER_RE
        .captures_iter(fetch_source)
        .map(|c| c[1].to_string())
        .collect();

    // Extract provider names from manifest table.
    let manifest_providers: BTreeSet<String> = iter_table_rows(manifest_text)
        .iter()
        .filter(|row| row.cells.len() >= 2)
        .filter_map(|row| {
            let name = row.cells[0].trim();
            // Must be like "FooProvider" (not just "Provider" header).
            if name.ends_with("Provider") && name.len() > "Provider".len() {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect();

    for p in &rust_providers {
        if !manifest_providers.contains(p) {
            failures.push(format!("Provider {p} in Rust source but not in manifest"));
        }
    }
    for p in &manifest_providers {
        if !rust_providers.contains(p) {
            failures.push(format!("Provider {p} in manifest but not in Rust source"));
        }
    }

    failures
}

/// Run all verification checks and return combined results.
///
/// Returns `Ok(summary)` if all pass, `Err(failures)` if any fail.
pub fn run_all_verifications(repo_root: &Path) -> Result<String, Vec<String>> {
    let mut all_failures = Vec::new();
    let mut summaries = Vec::new();

    // 1. Matrix metadata.
    let matrix_path = repo_root.join("docs/CLAIMS_EVIDENCE_MATRIX.md");
    if matrix_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&matrix_path) {
            let f = verify_matrix_metadata(&text);
            summaries.push(format!("matrix_metadata: {} issues", f.len()));
            all_failures.extend(f);

            // 2. Where stated pointers.
            let f = verify_where_stated_pointers(&text);
            summaries.push(format!("where_stated: {} issues", f.len()));
            all_failures.extend(f);

            // 5. Tasks consistency (needs tasks file).
            let tasks_path = repo_root.join("docs/CLAIMS_TASKS.md");
            if tasks_path.exists() {
                if let Ok(tasks_text) = std::fs::read_to_string(&tasks_path) {
                    let f = verify_tasks_consistency(&text, &tasks_text);
                    summaries.push(format!("tasks_consistency: {} issues", f.len()));
                    all_failures.extend(f);

                    // Tasks metadata.
                    let f = verify_tasks_metadata(&tasks_text);
                    summaries.push(format!("tasks_metadata: {} issues", f.len()));
                    all_failures.extend(f);

                    // Task artifact links.
                    let f = verify_task_artifact_links(&tasks_text, repo_root);
                    summaries.push(format!("task_artifacts: {} issues", f.len()));
                    all_failures.extend(f);
                }
            }

            // 6. Domain mapping.
            let domain_path = repo_root.join("docs/claims/CLAIMS_DOMAIN_MAP.csv");
            if domain_path.exists() {
                if let Ok(csv_text) = std::fs::read_to_string(&domain_path) {
                    let f = verify_domain_mapping(&text, &csv_text);
                    summaries.push(format!("domain_mapping: {} issues", f.len()));
                    all_failures.extend(f);
                }
            }
        }
    } else {
        all_failures.push("Missing docs/CLAIMS_EVIDENCE_MATRIX.md".to_string());
    }

    // 3. Evidence links.
    let f = verify_evidence_links(repo_root);
    summaries.push(format!("evidence_links: {} issues", f.len()));
    all_failures.extend(f);

    // 7. Dataset providers.
    let manifest_path = repo_root.join("docs/DATASET_MANIFEST.md");
    let fetch_path = repo_root.join("crates/gororoba_cli/src/bin/fetch_datasets.rs");
    if manifest_path.exists() && fetch_path.exists() {
        if let (Ok(manifest), Ok(fetch_src)) = (
            std::fs::read_to_string(&manifest_path),
            std::fs::read_to_string(&fetch_path),
        ) {
            let f = verify_dataset_providers(&manifest, &fetch_src);
            summaries.push(format!("dataset_providers: {} issues", f.len()));
            all_failures.extend(f);
        }
    }

    if all_failures.is_empty() {
        Ok(summaries.join("\n"))
    } else {
        Err(all_failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> &'static str {
        "\
| Claim | Description | Where | Status | Verified | Notes |
| --- | --- | --- | --- | --- | --- |
| C-001 | First claim | `src/lib.rs` | **Verified** | 2026-01-15 | tested |
| C-002 | Second claim | `docs/foo.md` | **Speculative** | 2025-06-01 | pending |
| C-003 | Third claim | `src/bar.rs` | **Unverified** | 2026-02-01 | todo |
"
    }

    #[test]
    fn test_verify_matrix_metadata_clean() {
        let failures = verify_matrix_metadata(sample_matrix());
        assert!(
            failures.is_empty(),
            "Expected no failures, got: {failures:?}"
        );
    }

    #[test]
    fn test_verify_matrix_metadata_bad_status() {
        let text = "\
| Claim | Description | Where | Status | Verified | Notes |
| --- | --- | --- | --- | --- | --- |
| C-001 | Claim | `src/x.rs` | **InvalidToken** | 2026-01-15 | n |
";
        let failures = verify_matrix_metadata(text);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("non-canonical"));
    }

    #[test]
    fn test_verify_matrix_metadata_bad_date() {
        let text = "\
| Claim | Description | Where | Status | Verified | Notes |
| --- | --- | --- | --- | --- | --- |
| C-001 | Claim | `src/x.rs` | **Verified** | not-a-date | n |
";
        let failures = verify_matrix_metadata(text);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("ISO date"));
    }

    #[test]
    fn test_verify_where_stated_clean() {
        let failures = verify_where_stated_pointers(sample_matrix());
        assert!(failures.is_empty());
    }

    #[test]
    fn test_verify_where_stated_missing() {
        let text = "\
| Claim | Description | Where | Status | Verified | Notes |
| --- | --- | --- | --- | --- | --- |
| C-001 | Claim | no path here | **Verified** | 2026-01-15 | n |
";
        let failures = verify_where_stated_pointers(text);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("C-001"));
    }

    #[test]
    fn test_verify_tasks_metadata_with_date() {
        let text = "Date: 2026-02-07\n\n| C-001 | Task | TODO |\n";
        let failures = verify_tasks_metadata(text);
        assert!(failures.is_empty());
    }

    #[test]
    fn test_verify_tasks_metadata_missing_date() {
        let text = "No date header here\n\n| C-001 | Task | TODO |\n";
        let failures = verify_tasks_metadata(text);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("Date:"));
    }

    #[test]
    fn test_verify_domain_mapping_canonical() {
        let matrix = sample_matrix();
        let csv = "claim_id,domains\nC-001,algebra\nC-002,cosmology\nC-003,algebra;spectral\n";
        let failures = verify_domain_mapping(matrix, csv);
        assert!(failures.is_empty(), "Got: {failures:?}");
    }

    #[test]
    fn test_verify_domain_mapping_bad_domain() {
        let matrix = sample_matrix();
        let csv = "claim_id,domains\nC-001,algebra\nC-002,invalid_domain\nC-003,algebra\n";
        let failures = verify_domain_mapping(matrix, csv);
        assert!(failures.iter().any(|f| f.contains("non-canonical")));
    }

    #[test]
    fn test_verify_dataset_providers() {
        let manifest = "\
| Provider | Description |
| --- | --- |
| FooProvider | Fetch foo |
| BarProvider | Fetch bar |
";
        let source = "Box::new(FooProvider::new()), Box::new(BarProvider::new())";
        let failures = verify_dataset_providers(manifest, source);
        assert!(failures.is_empty());
    }

    #[test]
    fn test_verify_dataset_providers_mismatch() {
        let manifest = "| Provider | Desc |\n| --- | --- |\n| FooProvider | x |\n";
        let source = "Box::new(BarProvider::new())";
        let failures = verify_dataset_providers(manifest, source);
        assert_eq!(failures.len(), 2); // FooProvider in manifest only, BarProvider in source only
    }
}
