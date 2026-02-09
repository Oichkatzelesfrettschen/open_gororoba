//! Provenance hash verification for cached datasets.
//!
//! The Python script `bin/record_external_hashes.py` writes
//! `data/external/PROVENANCE.local.json` containing SHA-256 hashes
//! for every file under `data/external/`. This module reads that
//! JSON and verifies the hashes still match the cached files.

use crate::fetcher::{compute_sha256, FetchError};
use serde::Deserialize;
use std::path::Path;

/// A single file entry from PROVENANCE.local.json.
#[derive(Debug, Deserialize)]
pub struct ProvenanceEntry {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Top-level structure of PROVENANCE.local.json.
#[derive(Debug, Deserialize)]
pub struct ProvenanceManifest {
    pub generated_at_utc: String,
    pub hashes: Vec<ProvenanceEntry>,
}

/// Result of verifying a single file.
#[derive(Debug)]
pub struct VerifyResult {
    pub path: String,
    pub expected_sha256: String,
    pub actual_sha256: Option<String>,
    pub ok: bool,
    pub error: Option<String>,
}

/// Load a provenance manifest from a JSON file.
pub fn load_provenance(path: &Path) -> Result<ProvenanceManifest, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read provenance: {}", e)))?;
    let manifest: ProvenanceManifest = serde_json::from_str(&content)
        .map_err(|e| FetchError::Validation(format!("Parse provenance: {}", e)))?;
    Ok(manifest)
}

/// Verify all entries in a provenance manifest against files under `root_dir`.
///
/// Returns a Vec of results. Files that are missing or have hash mismatches
/// are marked with `ok = false`.
pub fn verify_provenance(manifest: &ProvenanceManifest, root_dir: &Path) -> Vec<VerifyResult> {
    manifest
        .hashes
        .iter()
        .map(|entry| {
            let file_path = root_dir.join(&entry.path);
            if !file_path.exists() {
                return VerifyResult {
                    path: entry.path.clone(),
                    expected_sha256: entry.sha256.clone(),
                    actual_sha256: None,
                    ok: false,
                    error: Some("file not found".to_string()),
                };
            }
            match compute_sha256(&file_path) {
                Ok(actual) => {
                    let ok = actual == entry.sha256;
                    VerifyResult {
                        path: entry.path.clone(),
                        expected_sha256: entry.sha256.clone(),
                        actual_sha256: Some(actual),
                        ok,
                        error: if ok {
                            None
                        } else {
                            Some("hash mismatch".to_string())
                        },
                    }
                }
                Err(e) => VerifyResult {
                    path: entry.path.clone(),
                    expected_sha256: entry.sha256.clone(),
                    actual_sha256: None,
                    ok: false,
                    error: Some(format!("hash error: {}", e)),
                },
            }
        })
        .collect()
}

/// Count how many entries passed and failed verification.
pub fn verification_summary(results: &[VerifyResult]) -> (usize, usize) {
    let passed = results.iter().filter(|r| r.ok).count();
    let failed = results.len() - passed;
    (passed, failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_load_provenance_synthetic() {
        let json = r#"{
            "generated_at_utc": "2026-02-07T00:00:00+00:00",
            "root": "data/external",
            "hashes": [
                {"path": "test.csv", "sha256": "abc123", "size_bytes": 42, "mtime_utc": "2026-01-01T00:00:00+00:00"}
            ]
        }"#;
        let f = write_temp(json);
        let manifest = load_provenance(f.path()).unwrap();
        assert_eq!(manifest.hashes.len(), 1);
        assert_eq!(manifest.hashes[0].path, "test.csv");
        assert_eq!(manifest.hashes[0].sha256, "abc123");
        assert_eq!(manifest.hashes[0].size_bytes, 42);
    }

    #[test]
    fn test_load_provenance_rejects_bad_json() {
        let f = write_temp("not json");
        assert!(load_provenance(f.path()).is_err());
    }

    #[test]
    fn test_verify_provenance_matching_file() {
        // Create a temp dir with a known file
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("hello.txt");
        std::fs::write(&file_path, b"hello world\n").unwrap();
        let actual_hash = compute_sha256(&file_path).unwrap();

        let manifest = ProvenanceManifest {
            generated_at_utc: "2026-02-07T00:00:00+00:00".to_string(),
            hashes: vec![ProvenanceEntry {
                path: "hello.txt".to_string(),
                sha256: actual_hash.clone(),
                size_bytes: 12,
            }],
        };

        let results = verify_provenance(&manifest, dir.path());
        assert_eq!(results.len(), 1);
        assert!(results[0].ok, "Hash should match");
        assert_eq!(
            results[0].actual_sha256.as_deref(),
            Some(actual_hash.as_str())
        );
    }

    #[test]
    fn test_verify_provenance_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = ProvenanceManifest {
            generated_at_utc: "2026-02-07T00:00:00+00:00".to_string(),
            hashes: vec![ProvenanceEntry {
                path: "nonexistent.csv".to_string(),
                sha256: "abc123".to_string(),
                size_bytes: 0,
            }],
        };

        let results = verify_provenance(&manifest, dir.path());
        assert_eq!(results.len(), 1);
        assert!(!results[0].ok);
        assert_eq!(results[0].error.as_deref(), Some("file not found"));
    }

    #[test]
    fn test_verify_provenance_hash_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("data.bin");
        std::fs::write(&file_path, b"actual content").unwrap();

        let manifest = ProvenanceManifest {
            generated_at_utc: "2026-02-07T00:00:00+00:00".to_string(),
            hashes: vec![ProvenanceEntry {
                path: "data.bin".to_string(),
                sha256: "0000000000000000000000000000000000000000000000000000000000000000"
                    .to_string(),
                size_bytes: 14,
            }],
        };

        let results = verify_provenance(&manifest, dir.path());
        assert_eq!(results.len(), 1);
        assert!(!results[0].ok);
        assert_eq!(results[0].error.as_deref(), Some("hash mismatch"));
    }

    #[test]
    fn test_verification_summary() {
        let results = vec![
            VerifyResult {
                path: "a".into(),
                expected_sha256: "x".into(),
                actual_sha256: Some("x".into()),
                ok: true,
                error: None,
            },
            VerifyResult {
                path: "b".into(),
                expected_sha256: "y".into(),
                actual_sha256: None,
                ok: false,
                error: Some("missing".into()),
            },
            VerifyResult {
                path: "c".into(),
                expected_sha256: "z".into(),
                actual_sha256: Some("z".into()),
                ok: true,
                error: None,
            },
        ];
        let (passed, failed) = verification_summary(&results);
        assert_eq!(passed, 2);
        assert_eq!(failed, 1);
    }

    #[test]
    fn test_verify_provenance_if_available() {
        let prov_path = Path::new("data/external/PROVENANCE.local.json");
        if !prov_path.exists() {
            eprintln!("Skipping: PROVENANCE.local.json not available");
            return;
        }
        let manifest = load_provenance(prov_path).expect("failed to load provenance");
        assert!(
            !manifest.hashes.is_empty(),
            "Provenance should have entries"
        );
        // Spot-check: verify first 5 entries
        let root = Path::new("data/external");
        let results = verify_provenance(&manifest, root);
        let (passed, failed) = verification_summary(&results);
        eprintln!(
            "Provenance verification: {}/{} passed, {} failed",
            passed,
            results.len(),
            failed
        );
        // Allow some missing files (not all datasets may be cached)
        // but hashes that DO exist should match
        for r in &results {
            if r.actual_sha256.is_some() {
                assert!(
                    r.ok,
                    "Hash mismatch for {}: expected {}, got {:?}",
                    r.path, r.expected_sha256, r.actual_sha256
                );
            }
        }
    }
}
