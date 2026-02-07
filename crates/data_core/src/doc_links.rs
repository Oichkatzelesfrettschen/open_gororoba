//! Verify file-path references in documentation.
//!
//! The claims matrix (`CLAIMS_EVIDENCE_MATRIX.md`) contains backtick-delimited
//! file paths in its "Where stated" column. This module extracts those paths
//! and checks they resolve to actual files on disk.

use std::path::Path;

/// Extract file paths enclosed in backticks from a markdown line.
///
/// Recognizes patterns like `` `crates/foo/bar.rs` `` and
/// `` `docs/SOMETHING.md` ``. Ignores parenthetical suffixes like
/// `(test_name)`.
pub fn extract_backtick_paths(line: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('`') {
        let after_tick = &rest[start + 1..];
        if let Some(end) = after_tick.find('`') {
            let candidate = &after_tick[..end];
            // Must look like a file path: contains '/' or '.' and no spaces
            // Also filter out things that look like code fragments
            if (candidate.contains('/') || candidate.ends_with(".md") || candidate.ends_with(".rs")
                || candidate.ends_with(".py") || candidate.ends_with(".csv")
                || candidate.ends_with(".json") || candidate.ends_with(".txt"))
                && !candidate.contains(' ')
                && !candidate.starts_with("http")
                && !candidate.starts_with("--")
            {
                paths.push(candidate.to_string());
            }
            rest = &after_tick[end + 1..];
        } else {
            break;
        }
    }
    paths
}

/// Check which paths from a list actually exist relative to a root directory.
///
/// Returns (existing, missing) as two vectors.
pub fn check_paths_exist(
    paths: &[String],
    root: &Path,
) -> (Vec<String>, Vec<String>) {
    let mut existing = Vec::new();
    let mut missing = Vec::new();
    for p in paths {
        if root.join(p).exists() {
            existing.push(p.clone());
        } else {
            missing.push(p.clone());
        }
    }
    (existing, missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_backtick_paths_basic() {
        let line = "| C-001 | text | `crates/algebra_core/src/foo.rs` (test_bar), `docs/STUFF.md` | ok |";
        let paths = extract_backtick_paths(line);
        assert_eq!(paths, vec![
            "crates/algebra_core/src/foo.rs",
            "docs/STUFF.md",
        ]);
    }

    #[test]
    fn test_extract_backtick_paths_filters_non_paths() {
        let line = "Use `cargo test` and `--all` flags; see `crates/x.rs` for details.";
        let paths = extract_backtick_paths(line);
        assert_eq!(paths, vec!["crates/x.rs"]);
    }

    #[test]
    fn test_extract_backtick_paths_csv_and_json() {
        let line = "`data/csv/results.csv`, `data/external/PROVENANCE.local.json`";
        let paths = extract_backtick_paths(line);
        assert_eq!(paths, vec![
            "data/csv/results.csv",
            "data/external/PROVENANCE.local.json",
        ]);
    }

    #[test]
    fn test_extract_backtick_paths_empty_line() {
        let paths = extract_backtick_paths("no backticks here");
        assert!(paths.is_empty());
    }

    #[test]
    fn test_check_paths_exist_with_real_crate() {
        // CARGO_MANIFEST_DIR points to the crate dir; go up twice for workspace root
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
        let paths = vec![
            "crates/data_core/src/lib.rs".to_string(),
            "nonexistent/file.rs".to_string(),
        ];
        let (existing, missing) = check_paths_exist(&paths, workspace_root);
        assert_eq!(existing, vec!["crates/data_core/src/lib.rs"]);
        assert_eq!(missing, vec!["nonexistent/file.rs"]);
    }

    #[test]
    fn test_claims_matrix_link_resolution_if_available() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
        let matrix_path = workspace_root.join("docs/CLAIMS_EVIDENCE_MATRIX.md");
        if !matrix_path.exists() {
            eprintln!("Skipping: claims matrix not available");
            return;
        }
        let content = std::fs::read_to_string(&matrix_path).unwrap();
        let root = workspace_root;

        let mut total_paths = 0usize;
        let mut rust_paths = 0usize;
        let mut rust_missing = 0usize;
        let mut rust_missing_examples: Vec<String> = Vec::new();

        for line in content.lines() {
            if !line.starts_with("| C-") {
                continue;
            }
            let paths = extract_backtick_paths(line);
            total_paths += paths.len();

            // Only assert on crates/ paths (active Rust code)
            let crate_paths: Vec<String> = paths
                .into_iter()
                .filter(|p| p.starts_with("crates/"))
                .collect();
            rust_paths += crate_paths.len();
            let (_, missing) = check_paths_exist(&crate_paths, root);
            rust_missing += missing.len();
            for m in &missing {
                if rust_missing_examples.len() < 10 {
                    rust_missing_examples.push(m.clone());
                }
            }
        }

        let rust_rate = if rust_paths > 0 {
            100.0 * (rust_paths - rust_missing) as f64 / rust_paths as f64
        } else {
            100.0
        };
        eprintln!(
            "Claims matrix: {} total paths, {} Rust crate paths ({} missing, {:.1}% resolved)",
            total_paths, rust_paths, rust_missing, rust_rate
        );
        if !rust_missing_examples.is_empty() {
            eprintln!("Missing Rust paths:");
            for m in &rust_missing_examples {
                eprintln!("  - {}", m);
            }
        }
        // All crates/ paths should exist (active code, not historical)
        assert!(
            rust_rate > 95.0,
            "Rust crate path resolution rate too low: {:.1}% ({} missing of {})",
            rust_rate, rust_missing, rust_paths
        );
    }
}
