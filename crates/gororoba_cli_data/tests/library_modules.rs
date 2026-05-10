//! External integration tests for the gororoba_cli_data library modules (PH-3 B3, Layer 2).
//!
//! WHY: The inline #[cfg(test)] blocks in project_api_contract.rs and acquisition_dossier.rs
//! exercise the internal logic thoroughly. These tests serve a complementary role: they verify
//! that the public module paths as seen by a DOWNSTREAM CONSUMER are correct. If a type is
//! renamed, moved to pub(crate), or the module tree is restructured, these tests catch it at
//! the facade boundary rather than inside the crate's own tests.
//!
//! They are intentionally non-redundant with inline tests: the focus is public-path accessibility
//! and the handful of cross-module behaviors that can only be seen from outside.

// ---- project_api_contract -------------------------------------------------------

use gororoba_cli_data::project_api_contract::{
    AcquisitionJournalRow, ProjectApiContract, ProjectApiCrosswalkBinding, ProjectApiCrosswalkFile,
    ProjectApiEntrypoints, journal_multi_value, repo_path, resolve_crosswalk_binding,
    split_journal_multi_value,
};

// ---- acquisition_dossier --------------------------------------------------------

use gororoba_cli_data::acquisition_dossier::{
    DossierBatchEntry, DossierBatchManifest, DossierHit, ResearchDossier, StageSuggestion,
};

use std::path::{Path, PathBuf};

// =============================================================================
// ProjectApiContract
// =============================================================================

/// Public module path is reachable; contract round-trips through TOML.
#[test]
fn project_api_contract_toml_round_trip() {
    let contract = ProjectApiContract {
        schema_version: Some(1),
        project_id: Some("external-consumer-test".to_string()),
        entrypoints: ProjectApiEntrypoints {
            inventory: "inv.toml".to_string(),
            eras: "eras.toml".to_string(),
            gaps: "gaps.toml".to_string(),
            nomenclature: "nomenclature.toml".to_string(),
            candidate_expansion: "candidates.toml".to_string(),
            crosswalk: "crosswalk.toml".to_string(),
            acquisition_journal: "journal.tsv".to_string(),
            availability_summary: "availability.toml".to_string(),
            row_availability_ledger: "row_ledger.toml".to_string(),
            century_bucket_summary: "centuries.toml".to_string(),
            search_queue: "search_queue.toml".to_string(),
            acquisition_stack: "stack.toml".to_string(),
        },
    };
    let serialized = toml::to_string_pretty(&contract).expect("serialize");
    let parsed: ProjectApiContract = toml::from_str(&serialized).expect("deserialize");
    assert_eq!(parsed.schema_version, Some(1));
    assert_eq!(parsed.project_id.as_deref(), Some("external-consumer-test"));
    assert_eq!(parsed.entrypoints.acquisition_stack, "stack.toml");
}

// =============================================================================
// ProjectApiCrosswalkFile + resolve_crosswalk_binding
// =============================================================================

/// resolve_crosswalk_binding finds by search_target_id from an external call site.
#[test]
fn crosswalk_lookup_by_search_target_id_external() {
    let crosswalk = ProjectApiCrosswalkFile {
        binding: vec![ProjectApiCrosswalkBinding {
            id: "b-ext-1".to_string(),
            search_target_id: "target-ext-1".to_string(),
            ..ProjectApiCrosswalkBinding::default()
        }],
        ..ProjectApiCrosswalkFile::default()
    };
    let hit = resolve_crosswalk_binding(&crosswalk, Some("target-ext-1"), None);
    assert!(hit.is_some());
    assert_eq!(hit.unwrap().id, "b-ext-1");
}

/// resolve_crosswalk_binding returns None for unknown keys.
#[test]
fn crosswalk_lookup_miss_returns_none_external() {
    let crosswalk = ProjectApiCrosswalkFile::default();
    assert!(resolve_crosswalk_binding(&crosswalk, Some("no-such"), None).is_none());
}

/// resolve_crosswalk_binding matches by candidate_id.
#[test]
fn crosswalk_lookup_by_candidate_id_external() {
    let crosswalk = ProjectApiCrosswalkFile {
        binding: vec![ProjectApiCrosswalkBinding {
            id: "b-ext-2".to_string(),
            search_target_id: "tgt-ext-2".to_string(),
            candidate_id: Some("cand-ext".to_string()),
            ..ProjectApiCrosswalkBinding::default()
        }],
        ..ProjectApiCrosswalkFile::default()
    };
    let hit = resolve_crosswalk_binding(&crosswalk, None, Some("cand-ext"));
    assert!(hit.is_some());
    assert_eq!(hit.unwrap().id, "b-ext-2");
}

// =============================================================================
// AcquisitionJournalRow
// =============================================================================

/// header() has exactly 31 tab-separated fields (compile-time stable contract).
#[test]
fn acquisition_journal_header_column_count_external() {
    let count = AcquisitionJournalRow::header().split('\t').count();
    assert_eq!(count, 31);
}

/// TSV round-trip is lossless for a minimally populated row.
#[test]
fn acquisition_journal_tsv_round_trip_external() {
    let row = AcquisitionJournalRow {
        at_utc: "2026-04-13T00:00:00Z".to_string(),
        action: "record".to_string(),
        session_id: "sess-external".to_string(),
        source_id: "src-ext".to_string(),
        search_target_id: "tgt-ext".to_string(),
        status: "downloaded".to_string(),
        outcome: "recorded_downloaded".to_string(),
        url: "https://example.test/paper.pdf".to_string(),
        bytes: "42".to_string(),
        sha256: "deadbeef".to_string(),
        http_code: "200".to_string(),
        ..AcquisitionJournalRow::default()
    };
    let line = row.to_tsv_line();
    let parsed = AcquisitionJournalRow::from_tsv_line(&line).expect("parse tsv");
    assert_eq!(parsed.session_id, "sess-external");
    assert_eq!(parsed.sha256, "deadbeef");
    assert_eq!(parsed.http_code, "200");
}

/// from_tsv_line returns an Err for wrong field count.
#[test]
fn acquisition_journal_wrong_field_count_external() {
    let result = AcquisitionJournalRow::from_tsv_line("only\ttwo\tfields");
    assert!(result.is_err());
}

// =============================================================================
// journal_multi_value / split_journal_multi_value
// =============================================================================

/// Semi-colon encoding and decoding round-trips correctly.
#[test]
fn journal_multi_value_encoding_external() {
    let values = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
    let encoded = journal_multi_value(&values);
    assert_eq!(encoded, "alpha;beta;gamma");
    let decoded = split_journal_multi_value(&encoded);
    assert_eq!(decoded, values);
}

/// split_journal_multi_value filters empty and whitespace-only segments.
#[test]
fn split_journal_multi_value_filters_empty_external() {
    let decoded = split_journal_multi_value(";;x; ;y;;");
    assert_eq!(decoded, vec!["x", "y"]);
}

// =============================================================================
// repo_path
// =============================================================================

/// repo_path passes absolute paths through unchanged.
#[test]
fn repo_path_absolute_passthrough_external() {
    let result = repo_path(Path::new("/workspace"), Path::new("/other/file.toml"));
    assert_eq!(result, PathBuf::from("/other/file.toml"));
}

/// repo_path joins relative paths to the repo root.
#[test]
fn repo_path_relative_join_external() {
    let result = repo_path(Path::new("/workspace"), Path::new("data/claims.toml"));
    assert_eq!(result, PathBuf::from("/workspace/data/claims.toml"));
}

// =============================================================================
// ResearchDossier
// =============================================================================

/// ResearchDossier module path is reachable; JSON round-trip works.
#[test]
fn research_dossier_json_round_trip_external() {
    let dossier = ResearchDossier {
        schema_version: 4,
        project_id: "external-consumer".to_string(),
        title: "External Dossier Test".to_string(),
        limit_per_query: 20,
        year_min: 2018,
        min_relevance: 40,
        top_hits: vec![DossierHit {
            rank: 1,
            canonical_id: "ext-hit-1".to_string(),
            relevance_score: 88,
            title: "Sample Paper".to_string(),
            year: 2023,
            ..DossierHit::default()
        }],
        stage_suggestions: vec![StageSuggestion {
            rank: 1,
            suggestion_id: "sug-1".to_string(),
            default_selected: true,
            ..StageSuggestion::default()
        }],
        ..ResearchDossier::default()
    };
    let json = serde_json::to_string_pretty(&dossier).expect("serialize");
    let parsed: ResearchDossier = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.schema_version, 4);
    assert_eq!(parsed.project_id, "external-consumer");
    assert_eq!(parsed.top_hits.len(), 1);
    assert_eq!(parsed.top_hits[0].relevance_score, 88);
    assert!(parsed.stage_suggestions[0].default_selected);
}

// =============================================================================
// DossierBatchManifest
// =============================================================================

/// DossierBatchManifest round-trips through TOML via the public path.
#[test]
fn dossier_batch_manifest_toml_round_trip_external() {
    let manifest = DossierBatchManifest {
        schema_version: 2,
        project_id: "batch-external".to_string(),
        year_min: 2010,
        limit_per_query: 8,
        min_relevance: 20,
        entries: vec![DossierBatchEntry {
            search_target_id: "tgt-batch-ext".to_string(),
            title: "Batch Entry".to_string(),
            suggestion_count: 7,
            ..DossierBatchEntry::default()
        }],
        ..DossierBatchManifest::default()
    };
    let serialized = toml::to_string_pretty(&manifest).expect("serialize");
    let parsed: DossierBatchManifest = toml::from_str(&serialized).expect("deserialize");
    assert_eq!(parsed.schema_version, 2);
    assert_eq!(parsed.project_id, "batch-external");
    assert_eq!(parsed.entries[0].suggestion_count, 7);
}
