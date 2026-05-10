//! formal-proof-backfill: apply the formal_proof field heuristic from
//! docs/engineering/formal_proof_field_schema_2026_05_09.md to every
//! claim row with an empty/NULL formal_proof.
//!
//! Default mode is dry-run: walk the canonical SQLite, classify each
//! row, and write a TOML manifest of proposed updates. Pass `--apply`
//! to actually run the mutations (each via claim_update_formal_proof,
//! which records a row in claim_revisions for provenance).
//!
//! The 7-rule ladder:
//!   1. status in (Refuted, Falsified, Closed_negative_result, Closed_refuted)
//!      -> na_empirical
//!   2. arXiv: or doi.org/ in where_stated  -> external:<extracted>
//!   3. proofs/verified/<id>_*.v exists      -> proofs/verified/<file>
//!   4. proofs/theories/<id>_*.v exists      -> proofs/theories/<file>
//!   5. LBM / GPU / simulation in statement  -> na_methodology:simulation
//!   6. observed / measured / detected in statement -> na_observational
//!   7. otherwise                            -> pending:reviewed_pending

use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "formal-proof-backfill",
    about = "Backfill claims.formal_proof per the 7-rule heuristic"
)]
struct Args {
    /// Path to the canonical SQLite control plane.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    /// Path to write the proposal TOML manifest.
    #[arg(long, default_value = "data/output/audit/formal_proof_backfill.toml")]
    output: PathBuf,
    /// Repository root (used to scan proofs/verified and proofs/theories).
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    /// If true, run claim_update_formal_proof for each proposed update.
    /// Default false (dry-run only).
    #[arg(long, default_value_t = false)]
    apply: bool,
    /// Actor recorded on each claim_revisions row when --apply is set.
    /// Defaults to $USER.
    #[arg(long)]
    actor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Proposal {
    id: String,
    proposed_formal_proof: String,
    rule: u8,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct Manifest {
    meta: Meta,
    counts_by_rule: BTreeMap<String, u64>,
    counts_by_proposal_prefix: BTreeMap<String, u64>,
    proposals: Vec<Proposal>,
}

#[derive(Debug, Clone, Serialize)]
struct Meta {
    generated_at: String,
    binary: String,
    canonical_db: String,
    apply: bool,
    total_empty_rows: u64,
    proposals_count: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Build the proof-file index once.
    let verified_index = index_proof_files(&args.repo_root.join("proofs/verified"))?;
    let theories_index = index_proof_files(&args.repo_root.join("proofs/theories"))?;
    let arxiv_re = Regex::new(r"arXiv:(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})")?;
    let doi_re = Regex::new(r"doi\.org/(\S+)")?;

    let mut store = provenance_store::ProvenanceStore::open(&args.canonical_db)
        .with_context(|| format!("open {}", args.canonical_db.display()))?;
    let claims = store.list_claims()?;
    let mut proposals: Vec<Proposal> = Vec::new();
    for claim in &claims {
        let is_empty = claim
            .formal_proof
            .as_deref()
            .map(str::is_empty)
            .unwrap_or(true);
        if !is_empty {
            continue;
        }
        let (proposed, rule) = classify(ClassifyInput {
            id: &claim.id,
            status: &claim.status,
            where_stated: &claim.where_stated,
            statement: &claim.statement,
            verified_index: &verified_index,
            theories_index: &theories_index,
            arxiv_re: &arxiv_re,
            doi_re: &doi_re,
        });
        proposals.push(Proposal {
            id: claim.id.clone(),
            proposed_formal_proof: proposed,
            rule,
            status: claim.status.clone(),
        });
    }

    let mut counts_by_rule: BTreeMap<String, u64> = BTreeMap::new();
    let mut counts_by_proposal_prefix: BTreeMap<String, u64> = BTreeMap::new();
    for p in &proposals {
        *counts_by_rule
            .entry(format!("rule_{}", p.rule))
            .or_default() += 1;
        let prefix = p
            .proposed_formal_proof
            .split([':', '/'])
            .next()
            .unwrap_or(&p.proposed_formal_proof)
            .to_string();
        *counts_by_proposal_prefix.entry(prefix).or_default() += 1;
    }

    let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let total = proposals.len() as u64;
    let manifest = Manifest {
        meta: Meta {
            generated_at: now,
            binary: env!("CARGO_BIN_NAME").to_string(),
            canonical_db: args.canonical_db.display().to_string(),
            apply: args.apply,
            total_empty_rows: total,
            proposals_count: total,
        },
        counts_by_rule,
        counts_by_proposal_prefix,
        proposals: proposals.clone(),
    };
    let toml_text = toml::to_string_pretty(&manifest)?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, &toml_text)?;
    eprintln!("wrote {} ({} proposals)", args.output.display(), total);

    if !args.apply {
        eprintln!("dry-run: no SQLite writes; pass --apply to commit changes");
        return Ok(());
    }

    let actor = args
        .actor
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "formal-proof-backfill".to_string());
    let reason = "DEBT-CLAIMS-PROOF backfill: apply 7-rule heuristic from \
                  docs/engineering/formal_proof_field_schema_2026_05_09.md";
    let mut applied = 0u64;
    for p in proposals {
        store.claim_update_formal_proof(&p.id, &p.proposed_formal_proof, &actor, Some(reason))?;
        applied += 1;
        if applied.is_multiple_of(100) {
            eprintln!("applied {} / {} ...", applied, total);
        }
    }
    eprintln!("done: applied {} updates", applied);
    Ok(())
}

/// Bundle of `classify` arguments. Keeps the function under
/// clippy::too_many_arguments without resorting to #[allow]; each field is
/// borrowed so the input itself is zero-copy at call sites.
struct ClassifyInput<'a> {
    id: &'a str,
    status: &'a str,
    where_stated: &'a str,
    statement: &'a str,
    verified_index: &'a BTreeMap<String, String>,
    theories_index: &'a BTreeMap<String, String>,
    arxiv_re: &'a Regex,
    doi_re: &'a Regex,
}

fn classify(input: ClassifyInput<'_>) -> (String, u8) {
    let ClassifyInput {
        id,
        status,
        where_stated,
        statement,
        verified_index,
        theories_index,
        arxiv_re,
        doi_re,
    } = input;
    // Rule 1: refuted/falsified/closed-negative -> na_empirical
    let s = status.to_ascii_lowercase();
    if s == "refuted" || s == "falsified" || s == "closed_negative_result" || s == "closed_refuted"
    {
        return ("na_empirical".to_string(), 1);
    }
    // Rule 2: arXiv: or doi in where_stated -> external:<key>
    if let Some(m) = arxiv_re.captures(where_stated) {
        return (format!("external:arXiv:{}", &m[1]), 2);
    }
    if let Some(m) = doi_re.captures(where_stated) {
        return (format!("external:doi:{}", &m[1]), 2);
    }
    // Rule 3: proofs/verified/<id>_*.v
    if let Some(path) = verified_index.get(id) {
        return (path.clone(), 3);
    }
    // Rule 4: proofs/theories/<id>_*.v
    if let Some(path) = theories_index.get(id) {
        return (path.clone(), 4);
    }
    // Rule 5: LBM/GPU/simulation in statement -> na_methodology:simulation
    let stmt_lower = statement.to_ascii_lowercase();
    if stmt_lower.contains("lbm")
        || stmt_lower.contains("gpu")
        || stmt_lower.contains("simulation")
        || stmt_lower.contains("benchmark")
    {
        return ("na_methodology:simulation".to_string(), 5);
    }
    // Rule 6: observed/measured/detected -> na_observational
    if stmt_lower.contains("observed")
        || stmt_lower.contains("measured")
        || stmt_lower.contains("detected")
        || stmt_lower.contains("observation")
    {
        return ("na_observational".to_string(), 6);
    }
    // Rule 7: otherwise -> pending:reviewed_pending
    ("pending:reviewed_pending".to_string(), 7)
}

/// Build a map from claim_id (e.g., "C-441") to "proofs/verified/C0441_Foo.v"
/// by walking the directory and matching the leading numeric prefix on each
/// .v file. The claim_id "C-441" matches files starting with "C0441_" or
/// "C441_" depending on the project convention.
fn index_proof_files(dir: &Path) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    if !dir.exists() {
        return Ok(out);
    }
    let id_re = Regex::new(r"^C(\d+)_")?;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(n) => n,
            None => continue,
        };
        if !name.ends_with(".v") {
            continue;
        }
        if let Some(m) = id_re.captures(name) {
            let n: u64 = m[1].parse().unwrap_or(0);
            // Both "C-441" and "C0441" / "C441" should match.
            let canonical_id = format!("C-{}", n);
            // Emit a repo-relative path: strip a leading "./" so the
            // resulting field reads "proofs/verified/Cxxx_Foo.v" rather
            // than "./proofs/verified/Cxxx_Foo.v".
            let dir_str = dir.display().to_string();
            let dir_clean = dir_str.strip_prefix("./").unwrap_or(&dir_str);
            let path = format!("{}/{}", dir_clean, name);
            out.insert(canonical_id, path);
        }
    }
    Ok(out)
}
