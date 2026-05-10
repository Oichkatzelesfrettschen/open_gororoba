//! formal-proof-tighten: a second pass at the claim formal_proof field.
//!
//! # Purpose and call sites
//!
//! `formal-proof-backfill` is the seven-rule first-pass classifier
//! (run once during Stage B). After that pass, ~1117 claims remain in
//! the catch-all `pending:reviewed_pending` bucket because none of
//! the seven rules matched. This binary applies *additional* rules
//! that look at evidence-path patterns the first pass intentionally
//! ignored, narrowing the residual `pending:*` set to the claims that
//! genuinely need human review.
//!
//! Called on demand from the WS-CLAIMS-001 follow-up workstream
//! (TaskList #116). Default is dry-run; pass `--apply` to mutate.
//!
//! # Why a second binary instead of extending the first?
//!
//! The first-pass rule ladder is documented in
//! `docs/engineering/formal_proof_field_schema_2026_05_09.md` and
//! every entry in `claim_revisions` references that schema. Changing
//! the ladder would invalidate the audit narrative ("which rule
//! produced which classification"). A separate binary gives the
//! second pass its own rule ladder with a distinct rule-id namespace
//! (rules 8..14), keeping the audit trail unambiguous.
//!
//! # The five new rules (8 through 12)
//!
//! Each rule operates on a `ClassifyInput` (claim id + status +
//! statement + where_stated + arXiv/doi regexes + verified-proof and
//! theories-proof indices) and returns a (proposed_proof, rule_id)
//! pair if it matches.
//!
//! ```text
//! Rule 8  src/scripts/...py present in where_stated
//!         -> "na_methodology:script_evidence"
//!         Rationale: the claim is supported by a Python analysis
//!         script + companion CSV/JSON; it's a methodology claim, not
//!         a theorem candidate.
//!
//! Rule 9  Only crates/... paths in where_stated, no docs/* and no
//!         scripts -- pure-Rust impl with tests
//!         -> "na_methodology:rust_test"
//!         Rationale: the claim is exercised by a Rust integration
//!         test; the test is the evidence.
//!
//! Rule 10 docs/external_sources/*.md present
//!         -> "external:source-doc:<filename>"
//!         Rationale: the claim cites an external bibliography entry;
//!         the bibliography file is the evidence pointer.
//!
//! Rule 11 docs/theory/*.md only (no code or scripts)
//!         -> "pending:doc_only_review"
//!         Rationale: a more specific bucket than the generic
//!         pending:reviewed_pending -- flags claims that are
//!         documentation-only and need either implementation or
//!         escalation to one of the na_* classes.
//!
//! Rule 12 data/csv/*.csv present (data artifact lane)
//!         -> "na_observational:data_artifact"
//!         Rationale: the claim is supported by an observational
//!         dataset captured into a CSV under data/csv/; the dataset
//!         is the evidence.
//! ```
//!
//! Rules 8-12 are evaluated in order; the first match wins. Claims
//! that match no rule retain `pending:reviewed_pending` (the catch-all
//! from rule 7). The audit trail records the rule_id (8..12) so a
//! reviewer can later filter by classification source.
//!
//! # Scope: only mutates `pending:reviewed_pending`
//!
//! Existing `na_*`, `external:*`, and `proofs/*` bindings are LEFT
//! ALONE. This binary only narrows the `pending:reviewed_pending`
//! population; it does not re-classify anything that was already
//! placed by the first pass.
//!
//! # Concrete example
//!
//! For a claim where_stated like:
//!
//!   `docs/external_sources/C043_COMPACT_OBJECT_CATALOG_SOURCES.md, src/scripts/data/fetch_chime_frb.py, data/csv/compact_objects_catalog.csv`
//!
//! Rule 10 fires first (external_sources match), so the proposed
//! proof becomes
//! `external:source-doc:C043_COMPACT_OBJECT_CATALOG_SOURCES.md` with
//! rule_id = 10. Rules 8 and 12 would also have matched if rule 10
//! had not; the order encodes the priority "documented citation > data
//! artifact > script evidence".

use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "formal-proof-tighten")]
struct Args {
    /// Path to the canonical SQLite control plane.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,
    /// Apply mutations. Default is dry-run (writes manifest only).
    #[arg(long, default_value_t = false)]
    apply: bool,
    /// Output manifest path.
    #[arg(long, default_value = "data/output/audit/formal_proof_tighten.toml")]
    out: PathBuf,
    /// Reviewer name. Defaults to $USER. Recorded in claim_revisions.
    #[arg(long)]
    actor: Option<String>,
}

#[derive(Serialize, Debug)]
struct Proposal {
    /// Claim id (e.g. C-441).
    id: String,
    /// New formal_proof value.
    proposed_formal_proof: String,
    /// Rule index (8..12) that produced the proposal. See module rustdoc.
    rule: u8,
}

#[derive(Serialize, Debug)]
struct Manifest {
    generated_at: String,
    db_path: String,
    rule_counts: std::collections::BTreeMap<u8, u64>,
    /// All proposals, sorted by claim id.
    proposals: Vec<Proposal>,
    /// Total claims still in `pending:reviewed_pending` after this pass.
    residual_pending: u64,
}

/// Try to classify a `pending:reviewed_pending` row using rules 8..12.
///
/// Returns `Some((proposed_proof, rule_id))` if a rule matches, or
/// `None` if the row should stay pending.
///
/// # Why these rules in this order
///
/// External-source citations (rule 10) trump data artifacts (rule 12)
/// because a citation tells the reviewer *where* to look; the data
/// artifact alone tells them only *what* the claim observed. Rule 11
/// (docs-only) is last among the documentation rules because it
/// represents the weakest evidence -- a theory note without
/// supporting code or data.
///
/// # Mutation safety
///
/// The function is pure: takes only `&str` inputs, returns owned
/// strings. The DB write happens at the call site, inside a
/// BEGIN IMMEDIATE transaction provided by `provenance_store::
/// claim_update_formal_proof`.
fn tighten(where_stated: &str) -> Option<(String, u8)> {
    // Rule 10: docs/external_sources/*.md citation.
    let ext_re = Regex::new(r"docs/external_sources/([A-Za-z0-9_]+\.md)").ok()?;
    if let Some(m) = ext_re.captures(where_stated) {
        return Some((format!("external:source-doc:{}", &m[1]), 10));
    }

    // Rule 13: data/external/SOURCES.toml provenance/source-manifest claim.
    // These are dataset-provenance assertions ("X is reachable at URL Y";
    // "endpoint Z returns 404"). The SOURCES.toml entry IS the evidence.
    if where_stated.contains("data/external/SOURCES.toml") {
        return Some(("external:source-manifest".to_string(), 13));
    }

    // Rule 14: external URL evidence (ftp:// / https:// / http:// in
    // where_stated). These are "this dataset lives at this URL" claims;
    // the URL itself is the pointer. We extract just the scheme +
    // first path segment to keep the value short and stable.
    let url_re = Regex::new(r"(?:ftp|https?)://[^\s,)]+").ok()?;
    if let Some(m) = url_re.find(where_stated) {
        let url = m.as_str();
        // Truncate to first 80 chars to keep the field reasonable.
        let short = if url.len() > 80 { &url[..80] } else { url };
        return Some((format!("external:url:{}", short), 14));
    }

    // Rule 8: Python script evidence under src/scripts/, tests/, or examples/.
    let py_re = Regex::new(r"(?:src/scripts|tests|examples)/[^,\s]*\.py").ok()?;
    if py_re.is_match(where_stated) {
        return Some(("na_methodology:script_evidence".to_string(), 8));
    }

    // Rule 12: data/csv/*.csv evidence (data artifact lane).
    let csv_re = Regex::new(r"data/csv/[^,\s]*\.csv").ok()?;
    if csv_re.is_match(where_stated) {
        return Some(("na_observational:data_artifact".to_string(), 12));
    }

    // Rule 15: registry/experiments.toml (E-NNN) cross-reference. Claims
    // that point at a numbered experiment are "the experiment is the
    // evidence" claims; the experiment record carries the actual
    // status / output_path / reproducibility metadata.
    let exp_re = Regex::new(r"registry/experiments\.toml.*\(E-\d+\)").ok()?;
    if exp_re.is_match(where_stated) {
        return Some(("na_methodology:experiment_ref".to_string(), 15));
    }

    // Rule 9: only `crates/*` references, no docs and no scripts.
    let has_crates = where_stated.contains("crates/");
    let has_docs = where_stated.contains("docs/");
    let has_scripts = where_stated.contains("src/scripts/")
        || where_stated.contains("tests/")
        || where_stated.contains("examples/");
    if has_crates && !has_docs && !has_scripts {
        return Some(("na_methodology:rust_test".to_string(), 9));
    }

    // Rule 11: docs/theory/*.md only (no code or scripts).
    let theory_re = Regex::new(r"docs/theory/[^,\s]*\.md").ok()?;
    if theory_re.is_match(where_stated) && !has_crates && !has_scripts {
        return Some(("pending:doc_only_review".to_string(), 11));
    }

    // Rule 11b: docs/research/*.md only (research narratives without
    // code -- a sub-bucket of docs-only review for narrative work).
    let research_re = Regex::new(r"docs/research/[^,\s]*\.md").ok()?;
    if research_re.is_match(where_stated) && !has_crates && !has_scripts {
        return Some(("pending:research_narrative_review".to_string(), 11));
    }

    // Rule 11c: any docs/* path with no code reference is doc-only.
    if has_docs && !has_crates && !has_scripts {
        return Some(("pending:doc_only_review".to_string(), 11));
    }

    None
}

fn main() -> Result<()> {
    let args = Args::parse();
    let actor = args
        .actor
        .clone()
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "unknown".to_string());

    let conn = rusqlite::Connection::open_with_flags(
        &args.db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .with_context(|| format!("open canonical db {}", args.db.display()))?;

    // Walk only pending:reviewed_pending rows; the first-pass
    // classifications (na_*, external:*, proofs/*) are out of scope.
    let mut stmt = conn.prepare(
        "SELECT id, where_stated FROM claims WHERE formal_proof = 'pending:reviewed_pending' ORDER BY id"
    )?;
    let rows: Vec<(String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);
    drop(conn);

    let mut proposals = Vec::new();
    let mut rule_counts: std::collections::BTreeMap<u8, u64> = Default::default();
    let mut residual = 0u64;
    for (id, where_stated) in &rows {
        match tighten(where_stated) {
            Some((proof, rule)) => {
                proposals.push(Proposal {
                    id: id.clone(),
                    proposed_formal_proof: proof,
                    rule,
                });
                *rule_counts.entry(rule).or_insert(0) += 1;
            }
            None => {
                residual += 1;
            }
        }
    }

    let manifest = Manifest {
        generated_at: chrono::Utc::now().to_rfc3339(),
        db_path: args.db.display().to_string(),
        rule_counts,
        proposals,
        residual_pending: residual,
    };

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let toml_text = toml::to_string_pretty(&manifest)?;
    std::fs::write(&args.out, &toml_text)?;
    eprintln!(
        "wrote {} ({} proposals, {} residual pending)",
        args.out.display(),
        manifest.proposals.len(),
        manifest.residual_pending
    );

    if args.apply {
        eprintln!("applying {} mutations ...", manifest.proposals.len());
        let mut store = provenance_store::ProvenanceStore::open(&args.db)?;
        let reason = "WS-CLAIMS-001 follow-up: formal_proof_tighten rule ladder 8..12; \
                      see docs/engineering/formal_proof_field_schema_2026_05_09.md and \
                      crates/gororoba_cli_data/src/bin/formal_proof_tighten.rs";
        let mut applied = 0u64;
        for p in &manifest.proposals {
            store.claim_update_formal_proof(
                &p.id,
                &p.proposed_formal_proof,
                &actor,
                Some(reason),
            )?;
            applied += 1;
            if applied.is_multiple_of(100) {
                eprintln!("applied {} / {} ...", applied, manifest.proposals.len());
            }
        }
        eprintln!("done: applied {} updates", applied);
    }

    Ok(())
}
