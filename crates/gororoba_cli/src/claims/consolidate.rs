//! Claims registry consolidation pipeline.
//!
//! Deduplicates, normalizes, enriches, cross-links, and synthesizes the
//! claims registry into a higher-quality knowledge graph.
//!
//! Subsystems:
//! - Status normalization (case, variant collapse)
//! - Similarity detection (Jaro-Winkler on statement text)
//! - Metadata auto-enrichment (phase, confidence, dependencies)
//! - Cross-reference graph builder (claims <-> insights <-> experiments)
//! - Deduplication engine (merge redundant claims)
//! - Conflict marker resolution

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};

use super::schema::normalize_status;

// ---------------------------------------------------------------------------
// Serde structs
// ---------------------------------------------------------------------------

/// Full claim entry with all 16 union keys as optional (except id and status).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FullClaimEntry {
    pub id: String,
    #[serde(default)]
    pub statement: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub where_stated: Option<String>,
    pub status: String,
    #[serde(default)]
    pub last_verified: Option<String>,
    #[serde(default)]
    pub what_would_verify_refute: Option<String>,
    #[serde(default)]
    pub supporting_evidence: Option<Vec<String>>,
    #[serde(default)]
    pub verification_method: Option<String>,
    #[serde(default)]
    pub confidence: Option<String>,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub sprint: Option<u32>,
    #[serde(default)]
    pub dependencies: Option<Vec<String>>,
    #[serde(default)]
    pub claims: Option<Vec<String>>,
    #[serde(default)]
    pub insights: Option<Vec<String>>,
    #[serde(default)]
    pub status_note: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ClaimsFile {
    pub claim: Vec<FullClaimEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InsightEntry {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub claims: Vec<String>,
    #[serde(default)]
    pub sprint: Option<u32>,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct InsightsFile {
    pub insight: Vec<InsightEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExperimentEntry {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub binary: Option<String>,
    #[serde(default)]
    pub claims: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ExperimentsFile {
    pub experiment: Vec<ExperimentEntry>,
}

/// A conflict marker from conflict_markers.toml.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConflictMarker {
    pub id: String,
    #[serde(default)]
    pub marker_kind: Option<String>,
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub claim_refs: Vec<String>,
    #[serde(default)]
    pub source_registry: Option<String>,
    #[serde(default)]
    pub source_document: Option<String>,
    #[serde(default)]
    pub section_label: Option<String>,
    #[serde(default)]
    pub line_start: Option<u32>,
    #[serde(default)]
    pub line_end: Option<u32>,
    #[serde(default)]
    pub positive_evidence: Vec<String>,
    #[serde(default)]
    pub negative_evidence: Vec<String>,
    #[serde(default)]
    pub jaccard_overlap: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConflictMarkersHeader {
    #[serde(default)]
    pub updated: Option<String>,
    #[serde(default)]
    pub authoritative: Option<bool>,
    #[serde(default)]
    pub marker_count: Option<u32>,
    #[serde(default)]
    pub high_severity_count: Option<u32>,
    #[serde(default)]
    pub medium_severity_count: Option<u32>,
    #[serde(default)]
    pub low_severity_count: Option<u32>,
    #[serde(default)]
    pub kind_count: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ConflictMarkersFile {
    #[serde(default)]
    pub conflict_markers: Option<ConflictMarkersHeader>,
    #[serde(default)]
    pub marker: Vec<ConflictMarker>,
}

// ---------------------------------------------------------------------------
// Analysis report
// ---------------------------------------------------------------------------

/// Read-only analysis report of the claims registry state.
#[derive(Debug, Default)]
pub struct AnalysisReport {
    pub total_claims: usize,
    pub status_distribution: BTreeMap<String, usize>,
    pub non_canonical_statuses: Vec<(String, String)>,
    pub missing_description: usize,
    pub missing_confidence: usize,
    pub missing_phase: usize,
    pub missing_sprint: usize,
    pub claims_with_crossrefs: usize,
    pub claims_with_insight_links: usize,
    pub similarity_pairs: Vec<SimilarityPair>,
    pub enrichment_candidates: usize,
    pub conflict_markers_resolvable: usize,
    pub conflict_markers_total: usize,
}

impl std::fmt::Display for AnalysisReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Claims Registry Analysis ===")?;
        writeln!(f, "Total claims: {}", self.total_claims)?;
        writeln!(f)?;
        writeln!(f, "--- Status Distribution ---")?;
        for (status, count) in &self.status_distribution {
            writeln!(f, "  {status}: {count}")?;
        }
        if !self.non_canonical_statuses.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "--- Non-Canonical Statuses ({}) ---",
                self.non_canonical_statuses.len()
            )?;
            for (id, status) in &self.non_canonical_statuses {
                writeln!(f, "  {id}: \"{status}\"")?;
            }
        }
        writeln!(f)?;
        writeln!(f, "--- Metadata Gaps ---")?;
        writeln!(f, "  Missing description: {}", self.missing_description)?;
        writeln!(f, "  Missing confidence:  {}", self.missing_confidence)?;
        writeln!(f, "  Missing phase:       {}", self.missing_phase)?;
        writeln!(f, "  Missing sprint:      {}", self.missing_sprint)?;
        writeln!(f)?;
        writeln!(f, "--- Cross-References ---")?;
        writeln!(
            f,
            "  Claims with claim cross-refs:  {}/{}",
            self.claims_with_crossrefs, self.total_claims
        )?;
        writeln!(
            f,
            "  Claims with insight links:     {}/{}",
            self.claims_with_insight_links, self.total_claims
        )?;
        if !self.similarity_pairs.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "--- Similarity Pairs (>0.85) ({}) ---",
                self.similarity_pairs.len()
            )?;
            for pair in &self.similarity_pairs {
                writeln!(
                    f,
                    "  {} <-> {} (score={:.4})",
                    pair.claim_a, pair.claim_b, pair.score
                )?;
            }
        }
        writeln!(f)?;
        writeln!(f, "--- Enrichment ---")?;
        writeln!(
            f,
            "  Auto-enrichable claims: {}",
            self.enrichment_candidates
        )?;
        writeln!(f)?;
        writeln!(f, "--- Conflict Markers ---")?;
        writeln!(
            f,
            "  Resolvable: {}/{}",
            self.conflict_markers_resolvable, self.conflict_markers_total
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Similarity detection
// ---------------------------------------------------------------------------

/// A pair of claims with similar statement text.
#[derive(Debug, Clone)]
pub struct SimilarityPair {
    pub claim_a: String,
    pub claim_b: String,
    pub score: f64,
}

/// Find pairs of claims with similar statement text using Jaro-Winkler.
pub fn find_similar_pairs(claims: &[FullClaimEntry], threshold: f64) -> Vec<SimilarityPair> {
    let mut pairs = Vec::new();

    for i in 0..claims.len() {
        let stmt_a = match &claims[i].statement {
            Some(s) => s.as_str(),
            None => continue,
        };
        // Skip very short statements (likely placeholders)
        if stmt_a.len() < 20 {
            continue;
        }

        // Skip claims already cross-referenced to each other
        let a_refs: HashSet<&str> = claims[i]
            .claims
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        for j in (i + 1)..claims.len() {
            // Skip if already linked
            if a_refs.contains(claims[j].id.as_str()) {
                continue;
            }

            let stmt_b = match &claims[j].statement {
                Some(s) => s.as_str(),
                None => continue,
            };
            if stmt_b.len() < 20 {
                continue;
            }

            let score = strsim::jaro_winkler(stmt_a, stmt_b);
            if score >= threshold {
                pairs.push(SimilarityPair {
                    claim_a: claims[i].id.clone(),
                    claim_b: claims[j].id.clone(),
                    score,
                });
            }
        }
    }

    pairs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

// ---------------------------------------------------------------------------
// Status normalization (batch)
// ---------------------------------------------------------------------------

/// Normalize all claim statuses in-place. Returns count of claims modified.
pub fn normalize_all_statuses(claims: &mut [FullClaimEntry]) -> usize {
    let mut modified = 0;
    for claim in claims.iter_mut() {
        let (canonical, note) = normalize_status(&claim.status);
        if canonical != claim.status {
            claim.status = canonical;
            modified += 1;
        }
        if let Some(n) = note {
            if claim.status_note.is_none() {
                claim.status_note = Some(n);
                // The status was already counted as modified above if it changed;
                // if only the note was added, count that too.
                if modified == 0 || claim.status == normalize_status(&claim.status).0 {
                    // Already counted or no status change
                }
            }
        }
    }
    modified
}

// ---------------------------------------------------------------------------
// Metadata auto-enrichment
// ---------------------------------------------------------------------------

/// Phase mapping from crate/module paths in where_stated.
const PHASE_CRATE_MAP: &[(&str, &str)] = &[
    ("vacuum_frustration", "Phase 1"),
    ("lbm_3d", "Phase 1"),
    ("lattice_filtration", "Phase 2"),
    ("neural_homotopy", "Phase 3"),
    ("gororoba_engine", "Phase 4"),
    ("cosmic_scheduler", "Phase 0"),
    ("algebra_core", "algebra"),
    ("cosmology_core", "cosmology"),
    ("gr_core", "general-relativity"),
    ("materials_core", "materials"),
    ("optics_core", "optics"),
    ("quantum_core", "quantum"),
    ("stats_core", "statistics"),
    ("spectral_core", "spectral"),
    ("lbm_core", "fluids"),
    ("data_core", "data"),
];

/// Infer the phase from the where_stated field.
fn infer_phase(where_stated: &str) -> Option<String> {
    for &(pattern, phase) in PHASE_CRATE_MAP {
        if where_stated.contains(pattern) {
            return Some(phase.to_string());
        }
    }
    None
}

/// Derive confidence from status and evidence availability.
fn derive_confidence(claim: &FullClaimEntry) -> String {
    match claim.status.as_str() {
        "Verified" | "Established" => {
            // Check for test references in where_stated or supporting_evidence
            let has_test_ref = claim
                .where_stated
                .as_ref()
                .is_some_and(|s| s.contains("test_") || s.contains("tests/"));
            let has_evidence = claim
                .supporting_evidence
                .as_ref()
                .is_some_and(|v| !v.is_empty());
            if has_test_ref || has_evidence {
                "high".to_string()
            } else {
                "medium".to_string()
            }
        }
        "Partial" | "Provisional" | "Theoretical" | "Inconclusive" => "low".to_string(),
        "Refuted" | "Superseded" => "n/a".to_string(),
        s if s.starts_with("Closed/") => "n/a".to_string(),
        _ => "low".to_string(),
    }
}

/// Extract C-NNN references from text.
pub fn extract_claim_refs(text: &str) -> Vec<String> {
    let re = Regex::new(r"C-(\d{3,4})").expect("valid regex");
    let mut refs: BTreeSet<String> = BTreeSet::new();
    for cap in re.captures_iter(text) {
        refs.insert(format!("C-{}", &cap[1]));
    }
    refs.into_iter().collect()
}

/// Enrich metadata fields that can be auto-derived.
/// Returns count of fields populated.
pub fn enrich_metadata(
    claims: &mut [FullClaimEntry],
    insights: &[InsightEntry],
    _experiments: &[ExperimentEntry],
) -> usize {
    // Build reverse lookup: claim_id -> list of insight IDs that reference it
    let mut insight_reverse: HashMap<String, Vec<String>> = HashMap::new();
    for insight in insights {
        for claim_ref in &insight.claims {
            insight_reverse
                .entry(claim_ref.clone())
                .or_default()
                .push(insight.id.clone());
        }
    }

    let mut enriched = 0;

    for claim in claims.iter_mut() {
        // Phase inference
        if claim.phase.is_none() {
            if let Some(ref ws) = claim.where_stated {
                if let Some(phase) = infer_phase(ws) {
                    claim.phase = Some(phase);
                    enriched += 1;
                }
            }
        }

        // Confidence derivation
        if claim.confidence.is_none() {
            claim.confidence = Some(derive_confidence(claim));
            enriched += 1;
        }

        // Dependencies: scan statement + what_would_verify_refute for C-NNN refs
        if claim.dependencies.is_none() {
            let mut dep_refs = Vec::new();
            if let Some(ref stmt) = claim.statement {
                dep_refs.extend(extract_claim_refs(stmt));
            }
            if let Some(ref wvr) = claim.what_would_verify_refute {
                dep_refs.extend(extract_claim_refs(wvr));
            }
            // Remove self-references
            dep_refs.retain(|r| r != &claim.id);
            // Deduplicate
            let unique: BTreeSet<String> = dep_refs.into_iter().collect();
            let dep_vec: Vec<String> = unique.into_iter().collect();
            if !dep_vec.is_empty() {
                claim.dependencies = Some(dep_vec);
                enriched += 1;
            }
        }

        // Insight reverse lookup
        if claim.insights.is_none() || claim.insights.as_ref().is_some_and(|v| v.is_empty()) {
            if let Some(insight_ids) = insight_reverse.get(&claim.id) {
                claim.insights = Some(insight_ids.clone());
                enriched += 1;
            }
        }

        // Description: extract first sentence from long statements
        if claim.description.is_none() {
            if let Some(ref stmt) = claim.statement {
                if stmt.len() > 200 {
                    // Take first sentence (up to first period followed by space or end)
                    let desc = stmt
                        .find(". ")
                        .map(|i| &stmt[..=i])
                        .unwrap_or_else(|| {
                            if stmt.len() > 200 {
                                &stmt[..200]
                            } else {
                                stmt
                            }
                        })
                        .to_string();
                    claim.description = Some(desc);
                    enriched += 1;
                }
            }
        }
    }

    enriched
}

// ---------------------------------------------------------------------------
// Cross-reference graph builder
// ---------------------------------------------------------------------------

/// Build bidirectional cross-references between claims.
/// Returns count of new links added.
pub fn build_crossref_graph(
    claims: &mut [FullClaimEntry],
    insights: &[InsightEntry],
    experiments: &[ExperimentEntry],
) -> usize {
    let valid_ids: HashSet<String> = claims.iter().map(|c| c.id.clone()).collect();
    let mut links_added = 0;

    // For each insight, record all claims it references
    let mut insight_claims: HashMap<String, Vec<String>> = HashMap::new();
    for insight in insights {
        for claim_ref in &insight.claims {
            if valid_ids.contains(claim_ref) {
                insight_claims
                    .entry(insight.id.clone())
                    .or_default()
                    .push(claim_ref.clone());
            }
        }
    }

    // For each experiment, record all claims it references
    let mut experiment_claims: HashMap<String, Vec<String>> = HashMap::new();
    for exp in experiments {
        for claim_ref in &exp.claims {
            if valid_ids.contains(claim_ref) {
                experiment_claims
                    .entry(exp.id.clone())
                    .or_default()
                    .push(claim_ref.clone());
            }
        }
    }

    // Build adjacency: for each claim, find mentions of other C-NNN IDs
    // in statement + what_would_verify_refute + supporting_evidence
    let mut adjacency: HashMap<String, BTreeSet<String>> = HashMap::new();
    for claim in claims.iter() {
        let mut refs = BTreeSet::new();
        if let Some(ref stmt) = claim.statement {
            for r in extract_claim_refs(stmt) {
                if r != claim.id && valid_ids.contains(&r) {
                    refs.insert(r);
                }
            }
        }
        if let Some(ref wvr) = claim.what_would_verify_refute {
            for r in extract_claim_refs(wvr) {
                if r != claim.id && valid_ids.contains(&r) {
                    refs.insert(r);
                }
            }
        }
        if let Some(ref ev) = claim.supporting_evidence {
            for item in ev {
                for r in extract_claim_refs(item) {
                    if r != claim.id && valid_ids.contains(&r) {
                        refs.insert(r);
                    }
                }
            }
        }
        if !refs.is_empty() {
            adjacency.insert(claim.id.clone(), refs);
        }
    }

    // Make adjacency bidirectional
    let keys: Vec<String> = adjacency.keys().cloned().collect();
    for id in &keys {
        let targets: Vec<String> = adjacency.get(id).cloned().unwrap_or_default().into_iter().collect();
        for target in targets {
            adjacency
                .entry(target)
                .or_default()
                .insert(id.clone());
        }
    }

    // Also add claims that share an insight
    for claim_list in insight_claims.values() {
        for i in 0..claim_list.len() {
            for j in (i + 1)..claim_list.len() {
                adjacency
                    .entry(claim_list[i].clone())
                    .or_default()
                    .insert(claim_list[j].clone());
                adjacency
                    .entry(claim_list[j].clone())
                    .or_default()
                    .insert(claim_list[i].clone());
            }
        }
    }

    // Also add claims that share an experiment
    for claim_list in experiment_claims.values() {
        for i in 0..claim_list.len() {
            for j in (i + 1)..claim_list.len() {
                adjacency
                    .entry(claim_list[i].clone())
                    .or_default()
                    .insert(claim_list[j].clone());
                adjacency
                    .entry(claim_list[j].clone())
                    .or_default()
                    .insert(claim_list[i].clone());
            }
        }
    }

    // Apply adjacency to claims
    for claim in claims.iter_mut() {
        if let Some(neighbors) = adjacency.get(&claim.id) {
            let existing: BTreeSet<String> = claim
                .claims
                .as_ref()
                .map(|v| v.iter().cloned().collect())
                .unwrap_or_default();
            let new_refs: Vec<String> = neighbors
                .iter()
                .filter(|n| !existing.contains(*n))
                .cloned()
                .collect();
            if !new_refs.is_empty() {
                let mut merged: Vec<String> = existing.into_iter().collect();
                merged.extend(new_refs.iter().cloned());
                merged.sort();
                merged.dedup();
                links_added += new_refs.len();
                claim.claims = Some(merged);
            }
        }
    }

    links_added
}

// ---------------------------------------------------------------------------
// Deduplication engine
// ---------------------------------------------------------------------------

/// Pre-identified merge targets. Each tuple is (primary, secondaries).
pub const MERGE_TARGETS: &[(&str, &[&str])] = &[
    // Legacy CSV rejection claims
    ("C-020", &["C-044", "C-536"]),
    // Mass-spectrum-refuted cluster
    (
        "C-068",
        &["C-072", "C-073", "C-078", "C-079", "C-083", "C-085", "C-091"],
    ),
    // Superseded by C-071
    ("C-071", &["C-080"]),
];

/// Merge secondary claims into the primary.
/// Secondaries get status "Superseded" with a status_note pointing to primary.
/// Returns count of claims merged (i.e., secondaries processed).
pub fn merge_claims(claims: &mut [FullClaimEntry]) -> usize {
    let mut merged_count = 0;

    for &(primary_id, secondaries) in MERGE_TARGETS {
        // Find primary index
        let primary_idx = match claims.iter().position(|c| c.id == primary_id) {
            Some(idx) => idx,
            None => continue,
        };

        // Collect data from secondaries before mutating
        let mut secondary_statements = Vec::new();
        let mut secondary_refs = BTreeSet::new();
        let mut secondary_insight_refs = BTreeSet::new();

        for &sec_id in secondaries {
            if let Some(sec) = claims.iter().find(|c| c.id == sec_id) {
                if let Some(ref stmt) = sec.statement {
                    secondary_statements.push(format!("[{sec_id}] {stmt}"));
                }
                if let Some(ref refs) = sec.claims {
                    for r in refs {
                        secondary_refs.insert(r.clone());
                    }
                }
                if let Some(ref ins) = sec.insights {
                    for i in ins {
                        secondary_insight_refs.insert(i.clone());
                    }
                }
            }
        }

        // Enrich primary with secondary data
        {
            let primary = &mut claims[primary_idx];

            // Append secondary statements to supporting_evidence
            let evidence = primary.supporting_evidence.get_or_insert_with(Vec::new);
            evidence.extend(secondary_statements);

            // Merge cross-references
            if !secondary_refs.is_empty() {
                let claim_refs = primary.claims.get_or_insert_with(Vec::new);
                for r in &secondary_refs {
                    if !claim_refs.contains(r) && r != primary_id {
                        claim_refs.push(r.clone());
                    }
                }
                claim_refs.sort();
            }

            // Merge insight references
            if !secondary_insight_refs.is_empty() {
                let insight_refs = primary.insights.get_or_insert_with(Vec::new);
                for i in &secondary_insight_refs {
                    if !insight_refs.contains(i) {
                        insight_refs.push(i.clone());
                    }
                }
                insight_refs.sort();
            }
        }

        // Mark secondaries as Superseded
        for &sec_id in secondaries {
            if let Some(sec) = claims.iter_mut().find(|c| c.id == sec_id) {
                // Only merge if not already superseded
                if sec.status != "Superseded" {
                    sec.status = "Superseded".to_string();
                    sec.status_note = Some(format!("Merged into {primary_id}"));
                    merged_count += 1;
                }
            }
        }
    }

    merged_count
}

// ---------------------------------------------------------------------------
// Conflict marker resolution
// ---------------------------------------------------------------------------

/// Compute Jaccard overlap between two sets of whitespace-tokenized words.
fn jaccard_overlap(a: &str, b: &str) -> f64 {
    let set_a: HashSet<&str> = a.split_whitespace().collect();
    let set_b: HashSet<&str> = b.split_whitespace().collect();
    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Attempt to resolve conflict markers by checking if status normalization
/// resolves the tension. Returns count of markers resolved.
pub fn resolve_conflict_markers(
    claims: &[FullClaimEntry],
    markers: &mut [ConflictMarker],
) -> usize {
    let claim_map: HashMap<&str, &FullClaimEntry> =
        claims.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut resolved = 0;

    for marker in markers.iter_mut() {
        // Compute jaccard_overlap from positive/negative evidence
        let pos_text = marker.positive_evidence.join(" ");
        let neg_text = marker.negative_evidence.join(" ");
        if !pos_text.is_empty() || !neg_text.is_empty() {
            marker.jaccard_overlap = Some(jaccard_overlap(&pos_text, &neg_text));
        }

        // Check if the referenced claim's status explains the tension
        for claim_ref in &marker.claim_refs {
            if let Some(claim) = claim_map.get(claim_ref.as_str()) {
                // If claim has a status_note explaining the tension, mark resolved
                if claim.status_note.is_some() && marker.status.as_deref() == Some("open") {
                    marker.status = Some("resolved-by-normalization".to_string());
                    resolved += 1;
                    break;
                }

                // If claim is Refuted/Closed/* and marker is about status-statement
                // tension, the tension is expected (the statement describes what was
                // originally claimed, not what was concluded)
                if (claim.status == "Refuted" || claim.status.starts_with("Closed/"))
                    && marker.marker_kind.as_deref()
                        == Some("claim_status_statement_tension")
                    && marker.status.as_deref() == Some("open")
                {
                    marker.status = Some("resolved-expected-tension".to_string());
                    marker.notes = Some(format!(
                        "Refuted/Closed claims naturally have positive statement language \
                         describing the original hypothesis. Status '{}' is the conclusion.",
                        claim.status
                    ));
                    resolved += 1;
                    break;
                }

                // If claim is Verified and marker says negative language but the
                // what_would_verify_refute explains it, mark resolved
                if claim.status == "Verified"
                    && marker.marker_kind.as_deref()
                        == Some("claim_status_statement_tension")
                    && claim
                        .what_would_verify_refute
                        .as_ref()
                        .is_some_and(|w| w.len() > 100)
                    && marker.status.as_deref() == Some("open")
                {
                    marker.status = Some("resolved-detailed-verification".to_string());
                    resolved += 1;
                    break;
                }
            }
        }
    }

    resolved
}

// ---------------------------------------------------------------------------
// Full analysis (read-only)
// ---------------------------------------------------------------------------

/// Perform read-only analysis of the claims registry.
pub fn analyze(
    claims: &[FullClaimEntry],
    insights: &[InsightEntry],
    _experiments: &[ExperimentEntry],
    markers: &[ConflictMarker],
) -> AnalysisReport {
    let mut report = AnalysisReport {
        total_claims: claims.len(),
        ..Default::default()
    };

    // Status distribution and non-canonical detection
    for claim in claims {
        *report
            .status_distribution
            .entry(claim.status.clone())
            .or_insert(0) += 1;

        let (canonical, _) = normalize_status(&claim.status);
        if canonical != claim.status {
            report
                .non_canonical_statuses
                .push((claim.id.clone(), claim.status.clone()));
        }
    }

    // Metadata gaps
    for claim in claims {
        if claim.description.is_none() {
            report.missing_description += 1;
        }
        if claim.confidence.is_none() {
            report.missing_confidence += 1;
        }
        if claim.phase.is_none() {
            report.missing_phase += 1;
        }
        if claim.sprint.is_none() {
            report.missing_sprint += 1;
        }
        if claim.claims.as_ref().is_some_and(|v| !v.is_empty()) {
            report.claims_with_crossrefs += 1;
        }
        if claim.insights.as_ref().is_some_and(|v| !v.is_empty()) {
            report.claims_with_insight_links += 1;
        }
    }

    // Similarity pairs
    report.similarity_pairs = find_similar_pairs(claims, 0.85);

    // Enrichment candidates: count claims that would gain at least one field
    let insight_reverse: HashSet<String> = {
        let mut set = HashSet::new();
        for insight in insights {
            for cr in &insight.claims {
                set.insert(cr.clone());
            }
        }
        set
    };

    for claim in claims {
        let would_gain_phase =
            claim.phase.is_none() && claim.where_stated.as_ref().is_some_and(|ws| infer_phase(ws).is_some());
        let would_gain_confidence = claim.confidence.is_none();
        let would_gain_insight = (claim.insights.is_none()
            || claim.insights.as_ref().is_some_and(|v| v.is_empty()))
            && insight_reverse.contains(&claim.id);
        if would_gain_phase || would_gain_confidence || would_gain_insight {
            report.enrichment_candidates += 1;
        }
    }

    // Conflict markers
    report.conflict_markers_total = markers.len();
    // Estimate resolvable: those where the claim has a status_note or is
    // Refuted/Closed with status-statement tension
    let claim_map: HashMap<&str, &FullClaimEntry> =
        claims.iter().map(|c| (c.id.as_str(), c)).collect();
    for marker in markers {
        if marker.status.as_deref() != Some("open") {
            continue;
        }
        for claim_ref in &marker.claim_refs {
            if let Some(claim) = claim_map.get(claim_ref.as_str()) {
                if claim.status_note.is_some()
                    || claim.status == "Refuted"
                    || claim.status.starts_with("Closed/")
                    || (claim.status == "Verified"
                        && claim
                            .what_would_verify_refute
                            .as_ref()
                            .is_some_and(|w| w.len() > 100))
                {
                    report.conflict_markers_resolvable += 1;
                    break;
                }
            }
        }
    }

    report
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

/// Load claims registry from a TOML file.
pub fn load_claims(path: &Path) -> Result<Vec<FullClaimEntry>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let file: ClaimsFile = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;
    Ok(file.claim)
}

/// Load insights registry from a TOML file.
pub fn load_insights(path: &Path) -> Result<Vec<InsightEntry>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let file: InsightsFile = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;
    Ok(file.insight)
}

/// Load experiments registry from a TOML file.
pub fn load_experiments(path: &Path) -> Result<Vec<ExperimentEntry>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let file: ExperimentsFile = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;
    Ok(file.experiment)
}

/// Load conflict markers from a TOML file.
pub fn load_conflict_markers(path: &Path) -> Result<(Option<ConflictMarkersHeader>, Vec<ConflictMarker>), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let file: ConflictMarkersFile = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;
    Ok((file.conflict_markers, file.marker))
}

/// Write claims registry to a TOML file.
pub fn write_claims(path: &Path, claims: &[FullClaimEntry]) -> Result<(), String> {
    let file = ClaimsFile {
        claim: claims.to_vec(),
    };
    let content = toml::to_string_pretty(&file)
        .map_err(|e| format!("Failed to serialize claims: {e}"))?;
    std::fs::write(path, content)
        .map_err(|e| format!("Failed to write {}: {e}", path.display()))
}

/// Write conflict markers to a TOML file.
pub fn write_conflict_markers(
    path: &Path,
    header: &Option<ConflictMarkersHeader>,
    markers: &[ConflictMarker],
) -> Result<(), String> {
    let file = ConflictMarkersFile {
        conflict_markers: header.clone(),
        marker: markers.to_vec(),
    };
    let content = toml::to_string_pretty(&file)
        .map_err(|e| format!("Failed to serialize conflict markers: {e}"))?;

    // Prepend the original comment header
    let output = format!(
        "# Conflict marker registry (Wave 5 strict schema).\n\
         # Updated by claims-consolidate pipeline.\n\n{content}"
    );
    std::fs::write(path, output)
        .map_err(|e| format!("Failed to write {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

/// Result of running the full consolidation pipeline.
#[derive(Debug, Default)]
pub struct ConsolidationResult {
    pub statuses_normalized: usize,
    pub fields_enriched: usize,
    pub crosslinks_added: usize,
    pub claims_merged: usize,
    pub conflicts_resolved: usize,
}

impl std::fmt::Display for ConsolidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Consolidation Result ===")?;
        writeln!(f, "  Statuses normalized:  {}", self.statuses_normalized)?;
        writeln!(f, "  Fields enriched:      {}", self.fields_enriched)?;
        writeln!(f, "  Cross-links added:    {}", self.crosslinks_added)?;
        writeln!(f, "  Claims merged:        {}", self.claims_merged)?;
        writeln!(f, "  Conflicts resolved:   {}", self.conflicts_resolved)?;
        Ok(())
    }
}

/// Run the full consolidation pipeline in-memory.
pub fn run_full(
    claims: &mut [FullClaimEntry],
    insights: &[InsightEntry],
    experiments: &[ExperimentEntry],
    markers: &mut [ConflictMarker],
) -> ConsolidationResult {
    let statuses_normalized = normalize_all_statuses(claims);
    let fields_enriched = enrich_metadata(claims, insights, experiments);
    let crosslinks_added = build_crossref_graph(claims, insights, experiments);
    let claims_merged = merge_claims(claims);
    let conflicts_resolved = resolve_conflict_markers(claims, markers);

    ConsolidationResult {
        statuses_normalized,
        fields_enriched,
        crosslinks_added,
        claims_merged,
        conflicts_resolved,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_extract_claim_refs_basic() {
        let refs = extract_claim_refs("See C-100 and C-200 for details.");
        assert_eq!(refs, vec!["C-100", "C-200"]);
    }

    #[test]
    fn test_extract_claim_refs_dedup() {
        let refs = extract_claim_refs("C-100 confirmed by C-100 again");
        assert_eq!(refs, vec!["C-100"]);
    }

    #[test]
    fn test_extract_claim_refs_empty() {
        let refs = extract_claim_refs("No claim references here.");
        assert!(refs.is_empty());
    }

    #[test]
    fn test_similarity_detection() {
        let claims = vec![
            make_claim("C-001", "The Cayley-Dickson construction produces non-associative algebras at dimension 8 and beyond.", "Verified"),
            make_claim("C-002", "The Cayley-Dickson construction produces non-associative algebras at dimension eight and beyond.", "Verified"),
            make_claim("C-003", "Lattice Boltzmann method converges for Poiseuille flow.", "Verified"),
        ];
        let pairs = find_similar_pairs(&claims, 0.85);
        assert!(!pairs.is_empty(), "Should find C-001/C-002 as similar");
        assert_eq!(pairs[0].claim_a, "C-001");
        assert_eq!(pairs[0].claim_b, "C-002");
    }

    #[test]
    fn test_similarity_skips_already_linked() {
        let mut c1 = make_claim("C-001", "The Cayley-Dickson construction produces non-associative algebras at dimension 8 and beyond.", "Verified");
        c1.claims = Some(vec!["C-002".to_string()]);
        let c2 = make_claim("C-002", "The Cayley-Dickson construction produces non-associative algebras at dimension eight and beyond.", "Verified");
        let claims = vec![c1, c2];
        let pairs = find_similar_pairs(&claims, 0.85);
        assert!(pairs.is_empty(), "Should skip already-linked pair");
    }

    #[test]
    fn test_normalize_all_statuses() {
        let mut claims = vec![
            make_claim("C-001", "Test", "verified"),
            make_claim("C-002", "Test", "Verified"),
        ];
        let modified = normalize_all_statuses(&mut claims);
        assert_eq!(modified, 1);
        assert_eq!(claims[0].status, "Verified");
        assert_eq!(claims[1].status, "Verified");
    }

    #[test]
    fn test_derive_confidence_verified_with_test() {
        let mut claim = make_claim("C-001", "Test", "Verified");
        claim.where_stated = Some("crates/algebra_core/src/test_foo.rs".to_string());
        assert_eq!(derive_confidence(&claim), "high");
    }

    #[test]
    fn test_derive_confidence_refuted() {
        let claim = make_claim("C-001", "Test", "Refuted");
        assert_eq!(derive_confidence(&claim), "n/a");
    }

    #[test]
    fn test_infer_phase() {
        assert_eq!(
            infer_phase("crates/vacuum_frustration/src/lib.rs"),
            Some("Phase 1".to_string())
        );
        assert_eq!(
            infer_phase("crates/lattice_filtration/src/lib.rs"),
            Some("Phase 2".to_string())
        );
        assert_eq!(infer_phase("docs/theory/foo.md"), None);
    }

    #[test]
    fn test_jaccard_overlap_identical() {
        let j = jaccard_overlap("hello world", "hello world");
        assert!((j - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_overlap_disjoint() {
        let j = jaccard_overlap("hello world", "foo bar");
        assert!((j - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_overlap_partial() {
        let j = jaccard_overlap("hello world", "hello bar");
        // intersection = {hello}, union = {hello, world, bar}
        assert!((j - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_enrich_metadata_confidence() {
        let mut claims = vec![make_claim("C-001", "Test claim", "Verified")];
        let enriched = enrich_metadata(&mut claims, &[], &[]);
        assert!(enriched > 0);
        assert_eq!(claims[0].confidence.as_deref(), Some("medium"));
    }

    #[test]
    fn test_enrich_metadata_insight_reverse_lookup() {
        let mut claims = vec![make_claim("C-100", "Test claim", "Verified")];
        let insights = vec![InsightEntry {
            id: "I-001".to_string(),
            title: Some("Test insight".to_string()),
            date: None,
            status: None,
            claims: vec!["C-100".to_string()],
            sprint: None,
            summary: None,
        }];
        enrich_metadata(&mut claims, &insights, &[]);
        assert_eq!(
            claims[0].insights.as_ref().unwrap(),
            &vec!["I-001".to_string()]
        );
    }

    #[test]
    fn test_build_crossref_bidirectional() {
        let mut claims = vec![
            make_claim(
                "C-001",
                "This claim relates to C-002 findings.",
                "Verified",
            ),
            make_claim("C-002", "Standalone claim.", "Verified"),
        ];
        let added = build_crossref_graph(&mut claims, &[], &[]);
        assert!(added > 0);
        // C-001 should reference C-002
        assert!(claims[0]
            .claims
            .as_ref()
            .unwrap()
            .contains(&"C-002".to_string()));
        // C-002 should back-reference C-001 (bidirectional)
        assert!(claims[1]
            .claims
            .as_ref()
            .unwrap()
            .contains(&"C-001".to_string()));
    }

    #[test]
    fn test_consolidation_result_display() {
        let result = ConsolidationResult {
            statuses_normalized: 2,
            fields_enriched: 100,
            crosslinks_added: 50,
            claims_merged: 3,
            conflicts_resolved: 10,
        };
        let display = format!("{result}");
        assert!(display.contains("Statuses normalized:  2"));
        assert!(display.contains("Claims merged:        3"));
    }
}
