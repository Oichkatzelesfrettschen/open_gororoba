//! Integration tests for the claims consolidation pipeline.
//!
//! Tests cover: status normalization, similarity detection, metadata enrichment,
//! cross-reference graph, merge operations, conflict resolution, and full
//! pipeline idempotence.

use gororoba_cli::claims::consolidate::*;
use gororoba_cli::claims::schema;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_claim(id: &str, statement: &str, status: &str) -> FullClaimEntry {
    FullClaimEntry {
        id: id.to_string(),
        statement: Some(statement.to_string()),
        description: None,
        where_stated: None,
        status: status.to_string(),
        last_verified: None,
        what_would_verify_refute: None,
        supporting_evidence: None,
        verification_method: None,
        confidence: None,
        phase: None,
        sprint: None,
        dependencies: None,
        claims: None,
        insights: None,
        status_note: None,
    }
}

fn make_insight(id: &str, claims: &[&str]) -> InsightEntry {
    InsightEntry {
        id: id.to_string(),
        title: Some(format!("Insight {id}")),
        date: None,
        status: Some("verified".to_string()),
        claims: claims.iter().map(|s| s.to_string()).collect(),
        sprint: None,
        summary: None,
    }
}

fn make_experiment(id: &str, claims: &[&str]) -> ExperimentEntry {
    ExperimentEntry {
        id: id.to_string(),
        title: Some(format!("Experiment {id}")),
        binary: None,
        claims: claims.iter().map(|s| s.to_string()).collect(),
    }
}

fn make_marker(id: &str, claim_refs: &[&str], severity: &str) -> ConflictMarker {
    ConflictMarker {
        id: id.to_string(),
        marker_kind: Some("claim_status_statement_tension".to_string()),
        severity: Some(severity.to_string()),
        status: Some("open".to_string()),
        claim_refs: claim_refs.iter().map(|s| s.to_string()).collect(),
        source_registry: Some("registry/claims.toml".to_string()),
        source_document: claim_refs.first().map(|s| s.to_string()),
        section_label: Some("statement".to_string()),
        line_start: Some(0),
        line_end: Some(0),
        positive_evidence: vec!["Positive claim language.".to_string()],
        negative_evidence: vec!["REFUTED".to_string()],
        jaccard_overlap: Some(0.0),
        notes: None,
    }
}

// ---------------------------------------------------------------------------
// Status normalization tests
// ---------------------------------------------------------------------------

#[test]
fn test_normalize_case_insensitive() {
    let mut claims = vec![
        make_claim("C-001", "Test A", "verified"),
        make_claim("C-002", "Test B", "Verified"),
        make_claim("C-003", "Test C", "VERIFIED"),
    ];
    let modified = normalize_all_statuses(&mut claims);
    // "verified" and "VERIFIED" should be normalized
    assert!(modified >= 1);
    assert_eq!(claims[0].status, "Verified");
    assert_eq!(claims[1].status, "Verified");
}

#[test]
fn test_normalize_parenthetical_variant() {
    let (canonical, note) = schema::normalize_status("Verified (algebraic)");
    assert_eq!(canonical, "Verified");
    assert_eq!(note.unwrap(), "algebraic");
}

#[test]
fn test_normalize_closed_variants_preserved() {
    let closed_statuses = [
        "Closed/Negative-Result",
        "Closed/Obstructed",
        "Closed/Research-Program",
        "Closed/Toy",
        "Closed/Analogy",
        "Closed/Source-Insufficient",
        "Closed/Methodology-Insufficient",
        "Closed/Refuted",
    ];
    for &status in &closed_statuses {
        let (canonical, note) = schema::normalize_status(status);
        assert_eq!(
            canonical, status,
            "Closed status should be preserved: {status}"
        );
        assert!(
            note.is_none(),
            "Closed status should produce no note: {status}"
        );
    }
}

#[test]
fn test_normalize_idempotent() {
    let mut claims = vec![
        make_claim("C-001", "Test", "Verified"),
        make_claim("C-002", "Test", "Refuted"),
        make_claim("C-003", "Test", "Established"),
    ];
    normalize_all_statuses(&mut claims);
    // Run again -- should be idempotent
    let modified = normalize_all_statuses(&mut claims);
    assert_eq!(modified, 0, "Second normalization should change nothing");
}

// ---------------------------------------------------------------------------
// Similarity detection tests
// ---------------------------------------------------------------------------

#[test]
fn test_similarity_finds_near_duplicates() {
    let claims = vec![
        make_claim(
            "C-001",
            "Cayley-Dickson algebras become non-associative at dimension 8 and beyond.",
            "Verified",
        ),
        make_claim(
            "C-002",
            "Cayley-Dickson algebras become non-associative at dimension eight and beyond.",
            "Verified",
        ),
        make_claim(
            "C-003",
            "Lattice Boltzmann method converges for Poiseuille flow at all Reynolds numbers tested.",
            "Verified",
        ),
    ];
    let pairs = find_similar_pairs(&claims, 0.85);
    assert!(!pairs.is_empty(), "Should find C-001/C-002 as similar");
    assert_eq!(pairs[0].claim_a, "C-001");
    assert_eq!(pairs[0].claim_b, "C-002");
    assert!(pairs[0].score >= 0.85);
    // C-003 should not be similar to either
    assert!(
        pairs
            .iter()
            .all(|p| p.claim_a != "C-003" && p.claim_b != "C-003"),
        "C-003 should not match"
    );
}

#[test]
fn test_similarity_skips_linked_pairs() {
    let mut c1 = make_claim(
        "C-001",
        "Cayley-Dickson algebras become non-associative at dimension 8 and beyond.",
        "Verified",
    );
    c1.claims = Some(vec!["C-002".to_string()]);
    let c2 = make_claim(
        "C-002",
        "Cayley-Dickson algebras become non-associative at dimension eight and beyond.",
        "Verified",
    );
    let pairs = find_similar_pairs(&[c1, c2], 0.85);
    assert!(pairs.is_empty(), "Already-linked pairs should be skipped");
}

#[test]
fn test_similarity_ignores_short_statements() {
    let claims = vec![
        make_claim("C-001", "Short.", "Verified"),
        make_claim("C-002", "Short.", "Verified"),
    ];
    let pairs = find_similar_pairs(&claims, 0.85);
    assert!(pairs.is_empty(), "Short statements should be skipped");
}

// ---------------------------------------------------------------------------
// Metadata enrichment tests
// ---------------------------------------------------------------------------

#[test]
fn test_enrich_phase_from_where_stated() {
    let mut claims = vec![make_claim("C-001", "Test claim", "Verified")];
    claims[0].where_stated = Some("crates/vacuum_frustration/src/frustration.rs".to_string());
    enrich_metadata(&mut claims, &[], &[]);
    assert_eq!(claims[0].phase.as_deref(), Some("Phase 1"));
}

#[test]
fn test_enrich_confidence_verified_with_test() {
    let mut claims = vec![make_claim("C-001", "Test claim", "Verified")];
    claims[0].where_stated = Some("crates/algebra_core/src/test_boxkites.rs".to_string());
    enrich_metadata(&mut claims, &[], &[]);
    assert_eq!(claims[0].confidence.as_deref(), Some("high"));
}

#[test]
fn test_enrich_confidence_verified_without_test() {
    let mut claims = vec![make_claim("C-001", "Test claim", "Verified")];
    claims[0].where_stated = Some("docs/theory/THEORY.md".to_string());
    enrich_metadata(&mut claims, &[], &[]);
    assert_eq!(claims[0].confidence.as_deref(), Some("medium"));
}

#[test]
fn test_enrich_confidence_refuted() {
    let mut claims = vec![make_claim("C-001", "Test claim", "Refuted")];
    enrich_metadata(&mut claims, &[], &[]);
    assert_eq!(claims[0].confidence.as_deref(), Some("n/a"));
}

#[test]
fn test_enrich_insight_reverse_lookup() {
    let mut claims = vec![
        make_claim("C-100", "First claim", "Verified"),
        make_claim("C-200", "Second claim", "Verified"),
    ];
    let insights = vec![
        make_insight("I-001", &["C-100", "C-200"]),
        make_insight("I-002", &["C-100"]),
    ];
    enrich_metadata(&mut claims, &insights, &[]);
    let c100_insights = claims[0].insights.as_ref().unwrap();
    assert!(c100_insights.contains(&"I-001".to_string()));
    assert!(c100_insights.contains(&"I-002".to_string()));
    let c200_insights = claims[1].insights.as_ref().unwrap();
    assert!(c200_insights.contains(&"I-001".to_string()));
    assert!(!c200_insights.contains(&"I-002".to_string()));
}

#[test]
fn test_enrich_description_from_long_statement() {
    let long_stmt = "This is a very long claim statement that exceeds 200 characters. \
                     It contains multiple sentences. The first sentence should be extracted \
                     as the description. Additional context follows here with more detail \
                     about the experimental setup and results obtained from the analysis.";
    let mut claims = vec![make_claim("C-001", long_stmt, "Verified")];
    enrich_metadata(&mut claims, &[], &[]);
    let desc = claims[0].description.as_ref().unwrap();
    assert!(
        desc.ends_with('.'),
        "Description should end with period: {desc}"
    );
    assert!(
        desc.len() < long_stmt.len(),
        "Description should be shorter than statement"
    );
}

// ---------------------------------------------------------------------------
// Cross-reference graph tests
// ---------------------------------------------------------------------------

#[test]
fn test_crossref_bidirectional() {
    let mut claims = vec![
        make_claim(
            "C-001",
            "This claim relates to findings from C-002.",
            "Verified",
        ),
        make_claim("C-002", "Standalone claim with no references.", "Verified"),
    ];
    let added = build_crossref_graph(&mut claims, &[], &[]);
    assert!(added > 0);
    assert!(
        claims[0]
            .claims
            .as_ref()
            .unwrap()
            .contains(&"C-002".to_string()),
        "C-001 should reference C-002"
    );
    assert!(
        claims[1]
            .claims
            .as_ref()
            .unwrap()
            .contains(&"C-001".to_string()),
        "C-002 should back-reference C-001 (bidirectional)"
    );
}

#[test]
fn test_crossref_no_self_refs() {
    let mut claims = vec![
        make_claim(
            "C-001",
            "Self-referencing: see C-001 for details about C-002.",
            "Verified",
        ),
        make_claim("C-002", "Another claim.", "Verified"),
    ];
    build_crossref_graph(&mut claims, &[], &[]);
    let refs = claims[0].claims.as_ref().unwrap();
    assert!(
        !refs.contains(&"C-001".to_string()),
        "Should not self-reference"
    );
    assert!(
        refs.contains(&"C-002".to_string()),
        "Should reference C-002"
    );
}

#[test]
fn test_crossref_via_shared_insight() {
    let mut claims = vec![
        make_claim("C-100", "Claim A", "Verified"),
        make_claim("C-200", "Claim B", "Verified"),
    ];
    let insights = vec![make_insight("I-001", &["C-100", "C-200"])];
    build_crossref_graph(&mut claims, &insights, &[]);
    assert!(
        claims[0]
            .claims
            .as_ref()
            .unwrap()
            .contains(&"C-200".to_string()),
        "Claims sharing an insight should be cross-linked"
    );
}

#[test]
fn test_crossref_via_shared_experiment() {
    let mut claims = vec![
        make_claim("C-100", "Claim A", "Verified"),
        make_claim("C-200", "Claim B", "Verified"),
    ];
    let experiments = vec![make_experiment("E-001", &["C-100", "C-200"])];
    build_crossref_graph(&mut claims, &[], &experiments);
    assert!(
        claims[0]
            .claims
            .as_ref()
            .unwrap()
            .contains(&"C-200".to_string()),
        "Claims sharing an experiment should be cross-linked"
    );
}

// ---------------------------------------------------------------------------
// Merge operation tests
// ---------------------------------------------------------------------------

#[test]
fn test_merge_preserves_primary() {
    let mut claims = vec![
        make_claim("C-020", "Legacy CSV rejection primary claim.", "Refuted"),
        make_claim("C-044", "Legacy CSV secondary detail.", "Refuted"),
        make_claim("C-536", "Another CSV rejection note.", "Refuted"),
    ];
    let merged = merge_claims(&mut claims);
    assert!(merged >= 2, "Should merge at least C-044 and C-536");
    // Primary should still be Refuted
    assert_eq!(claims[0].status, "Refuted");
    assert_eq!(claims[0].id, "C-020");
    // Secondaries should be Superseded
    assert_eq!(claims[1].status, "Superseded");
    assert_eq!(claims[2].status, "Superseded");
    // Primary should have secondary statements in supporting_evidence
    let evidence = claims[0].supporting_evidence.as_ref().unwrap();
    assert!(evidence.iter().any(|e| e.contains("C-044")));
    assert!(evidence.iter().any(|e| e.contains("C-536")));
}

#[test]
fn test_merge_secondary_status_note() {
    let mut claims = vec![
        make_claim("C-020", "Primary.", "Refuted"),
        make_claim("C-044", "Secondary.", "Refuted"),
    ];
    merge_claims(&mut claims);
    let note = claims[1].status_note.as_ref().unwrap();
    assert!(
        note.contains("C-020"),
        "Status note should reference primary: {note}"
    );
}

#[test]
fn test_merge_idempotent() {
    let mut claims = vec![
        make_claim("C-020", "Primary.", "Refuted"),
        make_claim("C-044", "Secondary.", "Refuted"),
    ];
    merge_claims(&mut claims);
    // Run again
    let merged = merge_claims(&mut claims);
    assert_eq!(merged, 0, "Second merge should change nothing (idempotent)");
}

// ---------------------------------------------------------------------------
// Conflict marker resolution tests
// ---------------------------------------------------------------------------

#[test]
fn test_conflict_resolution_refuted_claim() {
    let claims = vec![make_claim(
        "C-020",
        "Legacy data is valid and consistent.",
        "Refuted",
    )];
    let mut markers = vec![make_marker("CM-0001", &["C-020"], "high")];
    let resolved = resolve_conflict_markers(&claims, &mut markers);
    assert!(
        resolved > 0,
        "Refuted claim with positive statement should resolve"
    );
    assert_ne!(
        markers[0].status.as_deref(),
        Some("open"),
        "Marker should no longer be open"
    );
}

#[test]
fn test_conflict_resolution_populates_jaccard() {
    let claims = vec![make_claim("C-001", "Test claim.", "Verified")];
    let mut markers = vec![make_marker("CM-0001", &["C-001"], "medium")];
    markers[0].positive_evidence = vec!["alpha beta gamma".to_string()];
    markers[0].negative_evidence = vec!["gamma delta epsilon".to_string()];
    resolve_conflict_markers(&claims, &mut markers);
    let j = markers[0].jaccard_overlap.unwrap();
    assert!(j > 0.0 && j < 1.0, "Jaccard should be between 0 and 1: {j}");
}

#[test]
fn test_conflict_resolution_with_status_note() {
    let mut claim = make_claim("C-001", "Ambiguous claim.", "Verified");
    claim.status_note = Some("Verified with caveats: see sprint 25 discussion.".to_string());
    let claims = vec![claim];
    let mut markers = vec![make_marker("CM-0001", &["C-001"], "medium")];
    let resolved = resolve_conflict_markers(&claims, &mut markers);
    assert!(
        resolved > 0,
        "Claim with status_note should resolve its conflict marker"
    );
}

// ---------------------------------------------------------------------------
// Full pipeline tests
// ---------------------------------------------------------------------------

#[test]
fn test_full_pipeline_runs() {
    let mut claims = vec![
        make_claim("C-001", "First verified claim.", "verified"),
        make_claim("C-002", "Second claim referencing C-001.", "Verified"),
    ];
    let insights = vec![make_insight("I-001", &["C-001", "C-002"])];
    let experiments = vec![make_experiment("E-001", &["C-001"])];
    let mut markers = Vec::new();

    let result = run_full(&mut claims, &insights, &experiments, &mut markers);
    // At least status normalization should have run
    assert!(
        result.statuses_normalized > 0 || result.fields_enriched > 0,
        "Pipeline should do something"
    );
    // C-001 should be normalized to "Verified"
    assert_eq!(claims[0].status, "Verified");
    // Both should have confidence set
    assert!(claims[0].confidence.is_some());
    assert!(claims[1].confidence.is_some());
}

#[test]
fn test_full_pipeline_idempotent() {
    let mut claims = vec![
        make_claim("C-001", "First verified claim.", "Verified"),
        make_claim("C-002", "Second claim referencing C-001.", "Verified"),
    ];
    let insights = vec![make_insight("I-001", &["C-001", "C-002"])];
    let experiments: Vec<ExperimentEntry> = Vec::new();
    let mut markers = Vec::new();

    // First run
    run_full(&mut claims, &insights, &experiments, &mut markers);
    // Snapshot state
    let snapshot: Vec<String> = claims.iter().map(|c| format!("{:?}", c)).collect();

    // Second run
    let result2 = run_full(&mut claims, &insights, &experiments, &mut markers);
    let snapshot2: Vec<String> = claims.iter().map(|c| format!("{:?}", c)).collect();

    assert_eq!(
        snapshot, snapshot2,
        "Second run should produce identical output"
    );
    assert_eq!(result2.statuses_normalized, 0, "No statuses to normalize");
}

// ---------------------------------------------------------------------------
// Claim ref extraction tests
// ---------------------------------------------------------------------------

#[test]
fn test_extract_claim_refs_from_text() {
    let refs = extract_claim_refs("Relates to C-100 and extends C-200 findings.");
    assert_eq!(refs, vec!["C-100", "C-200"]);
}

#[test]
fn test_extract_claim_refs_four_digit() {
    let refs = extract_claim_refs("See C-1234 for details.");
    assert_eq!(refs, vec!["C-1234"]);
}

#[test]
fn test_extract_claim_refs_deduplication() {
    let refs = extract_claim_refs("C-100 confirmed by C-100 in separate analysis.");
    assert_eq!(refs, vec!["C-100"]);
}

// ---------------------------------------------------------------------------
// I/O round-trip test
// ---------------------------------------------------------------------------

#[test]
fn test_claims_toml_round_trip() {
    let claims = vec![
        make_claim("C-001", "Test claim one.", "Verified"),
        make_claim("C-002", "Test claim two.", "Refuted"),
    ];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("claims.toml");
    write_claims(&path, &claims).unwrap();
    let loaded = load_claims(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].id, "C-001");
    assert_eq!(loaded[1].status, "Refuted");
}

#[test]
fn test_analysis_report_display() {
    let claims = vec![
        make_claim("C-001", "Test", "Verified"),
        make_claim("C-002", "Test", "verified"),
    ];
    let report = analyze(&claims, &[], &[], &[]);
    let output = format!("{report}");
    assert!(output.contains("Total claims: 2"));
    assert!(output.contains("Non-Canonical Statuses"));
}
