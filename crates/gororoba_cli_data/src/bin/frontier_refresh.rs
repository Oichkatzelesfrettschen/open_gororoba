use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use provenance_core::ClaimRecord;
use provenance_store::ProvenanceStore;
use rusqlite::{Connection, OpenFlags, params};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(name = "frontier-refresh")]
struct Cli {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,
    #[arg(
        long,
        default_value = "data/output/audit/2026-08-04/post-transition-falsification-frontier.toml"
    )]
    output: PathBuf,
}

#[derive(Serialize)]
struct FrontierSnapshot {
    campaign: String,
    generated_at: String,
    database_sha256: String,
    claim_count: usize,
    theorem_count: usize,
    transition_event_count: i64,
    compound_claim_count: usize,
    false_confidence_count: usize,
    open_adjudication_count: usize,
    status_counts: BTreeMap<String, usize>,
    epistemic_layer_counts: BTreeMap<String, usize>,
    next_bounded_tranches: Vec<Tranche>,
    false_confidence: Vec<QueueEntry>,
    open_adjudication: Vec<QueueEntry>,
    false_confidence_spotlight: Vec<QueueSpotlight>,
    open_adjudication_spotlight: Vec<QueueSpotlight>,
}

#[derive(Serialize)]
struct Tranche {
    order: u8,
    name: String,
    claim_ids: Vec<String>,
    discriminator: String,
    holdout_rule: String,
    excluded_claims: Vec<String>,
    blocked_claims: Vec<String>,
}

#[derive(Clone, Serialize)]
struct QueueEntry {
    claim_id: String,
    #[serde(skip)]
    statement: String,
    status: String,
    epistemic_layers: Vec<String>,
    compound: bool,
    risk_flags: Vec<String>,
    dependent_count: i64,
    ready_discriminator: bool,
    discriminator: String,
    #[serde(skip)]
    falsifier: String,
    shared_mechanism: String,
    estimated_cost: String,
    ordering_key: String,
}

#[derive(Serialize)]
struct QueueSpotlight {
    claim_id: String,
    statement: String,
    status: String,
    epistemic_layers: Vec<String>,
    compound: bool,
    risk_flags: Vec<String>,
    dependent_count: i64,
    ready_discriminator: bool,
    discriminator: String,
    falsifier: String,
    shared_mechanism: String,
    estimated_cost: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let db_path = cli.db;
    let store = ProvenanceStore::open_read_only(&db_path)?;
    let claims = store.list_claims()?;
    let conn = Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut status_counts = BTreeMap::new();
    let mut layer_counts = BTreeMap::new();
    let mut all_entries = Vec::with_capacity(claims.len());
    let mut compound_claim_count = 0usize;

    for claim in &claims {
        let entry = classify_claim(&conn, claim)?;
        if entry.compound {
            compound_claim_count += 1;
        }
        for layer in &entry.epistemic_layers {
            *layer_counts.entry(layer.clone()).or_insert(0) += 1;
        }
        *status_counts.entry(claim.status.clone()).or_insert(0) += 1;
        all_entries.push(entry);
    }

    let mut false_confidence = all_entries
        .iter()
        .filter(|entry| is_false_confidence_candidate(entry))
        .cloned()
        .collect::<Vec<_>>();
    let mut open_adjudication = all_entries
        .iter()
        .filter(|entry| is_open_adjudication_candidate(entry))
        .cloned()
        .collect::<Vec<_>>();
    false_confidence.sort_by(|left, right| left.ordering_key.cmp(&right.ordering_key));
    open_adjudication.sort_by(|left, right| left.ordering_key.cmp(&right.ordering_key));
    let false_confidence_spotlight = spotlight(&false_confidence);
    let open_adjudication_spotlight = spotlight(&open_adjudication);

    let snapshot = FrontierSnapshot {
        campaign: "post-transition-falsification-frontier".to_string(),
        generated_at: Utc::now().to_rfc3339(),
        database_sha256: sha256_file(&db_path)?,
        claim_count: claims.len(),
        theorem_count: store.list_theorems()?.len(),
        transition_event_count: scalar_count(
            &conn,
            "SELECT COUNT(*) FROM claim_transition_events",
        )?,
        compound_claim_count,
        false_confidence_count: false_confidence.len(),
        open_adjudication_count: open_adjudication.len(),
        status_counts,
        epistemic_layer_counts: layer_counts,
        next_bounded_tranches: next_tranches(&claims),
        false_confidence,
        open_adjudication,
        false_confidence_spotlight,
        open_adjudication_spotlight,
    };
    let output = toml::to_string_pretty(&snapshot)?;
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.output, output)
        .with_context(|| format!("write frontier snapshot {}", cli.output.display()))?;
    println!(
        "wrote {} claims={} false_confidence={} open_adjudication={} compound={}",
        cli.output.display(),
        snapshot.claim_count,
        snapshot.false_confidence_count,
        snapshot.open_adjudication_count,
        snapshot.compound_claim_count
    );
    Ok(())
}

fn classify_claim(conn: &Connection, claim: &ClaimRecord) -> Result<QueueEntry> {
    let mut layers = Vec::new();
    let corpus = format!(
        "{} {} {}",
        claim.statement,
        claim.where_stated,
        claim.status_note.as_deref().unwrap_or("")
    )
    .to_ascii_lowercase();
    if claim.formal_proof.is_some() || corpus.contains("proofs/") {
        layers.push("formal_theorem_binding".to_string());
    }
    if corpus.contains("equation")
        || corpus.contains("theorem")
        || corpus.contains("identity")
        || corpus.contains("source")
        || corpus.contains("arxiv")
    {
        layers.push("source_proposition".to_string());
    }
    if corpus.contains("implementation")
        || corpus.contains("repository")
        || corpus.contains("reproduce")
        || corpus.contains("code")
        || corpus.contains("validator")
    {
        layers.push("implementation_conformance".to_string());
    }
    if corpus.contains("parameter")
        || corpus.contains("mapping")
        || corpus.contains("observable")
        || corpus.contains("physical")
        || corpus.contains("phenomen")
    {
        layers.push("phenomenological_mapping".to_string());
    }
    if corpus.contains("statistic")
        || corpus.contains("p-value")
        || corpus.contains("confidence")
        || corpus.contains("resampling")
    {
        layers.push("statistical_inference".to_string());
    }
    if corpus.contains("gpu")
        || corpus.contains("backend")
        || corpus.contains("throughput")
        || corpus.contains("latency")
        || corpus.contains("vulkan")
        || corpus.contains("cuda")
        || corpus.contains("r300")
    {
        layers.push("performance_backend".to_string());
    }
    if corpus.contains("data/")
        || corpus.contains("dataset")
        || corpus.contains("parser")
        || corpus.contains("hash")
        || corpus.contains("units")
        || corpus.contains("provenance")
    {
        layers.push("data_provenance".to_string());
    }
    layers.sort();
    layers.dedup();
    if layers.is_empty() {
        layers.push("source_proposition".to_string());
    }
    let compound = layers.len() > 1;
    let mut risk_flags = Vec::new();
    let positive_status = matches!(
        claim.status.as_str(),
        "Verified" | "Established" | "Theoretical"
    );
    if positive_status
        && (corpus.contains("inconclusive")
            || corpus.contains("methodology")
            || corpus.contains("unresolved")
            || corpus.contains("blocked"))
    {
        risk_flags.push("status_evidence_contradiction".to_string());
    }
    if corpus.contains("tautolog")
        || corpus.contains("bookkeeping")
        || corpus.contains("same implementation")
        || corpus.contains("surrogate")
        || corpus.contains("legacy")
    {
        risk_flags.push("non_independent_oracle".to_string());
    }
    if compound {
        risk_flags.push("compound_claim".to_string());
    }
    let dependent_count = dependent_count(conn, &claim.id)?;
    let (discriminator, ready_discriminator, falsifier) = discriminator_for(&layers, claim);
    let shared_mechanism = mechanism_for(&corpus);
    let estimated_cost = cost_for(&layers);
    let ordering_key = format!(
        "{}-{:08}-{}-{}-{}-{}-{}",
        if risk_flags
            .iter()
            .any(|flag| flag == "status_evidence_contradiction")
        {
            "0"
        } else {
            "1"
        },
        9_999_999_i64.saturating_sub(dependent_count),
        if risk_flags
            .iter()
            .any(|flag| flag == "non_independent_oracle")
        {
            "0"
        } else {
            "1"
        },
        if compound { "0" } else { "1" },
        if ready_discriminator { "0" } else { "1" },
        shared_mechanism,
        claim.id
    );
    Ok(QueueEntry {
        claim_id: claim.id.clone(),
        statement: claim.statement.clone(),
        status: claim.status.clone(),
        epistemic_layers: layers,
        compound,
        risk_flags,
        dependent_count,
        ready_discriminator,
        discriminator,
        falsifier,
        shared_mechanism,
        estimated_cost,
        ordering_key,
    })
}

fn dependent_count(conn: &Connection, claim_id: &str) -> Result<i64> {
    conn.query_row(
        "SELECT
            (SELECT COUNT(*) FROM claim_relations
             WHERE predecessor_claim_id = ?1 OR successor_claim_id = ?1)
          + (SELECT COUNT(*) FROM insights
             JOIN json_each(insights.claim_refs_json)
               ON json_each.value = ?1)
          + (SELECT COUNT(*) FROM experiments_cp
             JOIN json_each(experiments_cp.claim_refs_json)
               ON json_each.value = ?1)",
        params![claim_id],
        |row| row.get(0),
    )
    .map_err(Into::into)
}

fn discriminator_for(layers: &[String], claim: &ClaimRecord) -> (String, bool, String) {
    if layers.iter().any(|layer| layer == "formal_theorem_binding") {
        return (
            "kernel result plus explicit assumption inventory".to_string(),
            claim.formal_proof.is_some(),
            "Change one assumption or defining coefficient and require the kernel or independent proof check to detect it.".to_string(),
        );
    }
    if layers
        .iter()
        .any(|layer| layer == "implementation_conformance")
    {
        return (
            "independent implementation with component comparison and mutations".to_string(),
            true,
            "Use an independently assembled implementation, compare components, and exercise a declared sign or omission mutation.".to_string(),
        );
    }
    if layers
        .iter()
        .any(|layer| layer == "phenomenological_mapping")
    {
        return (
            "held-out observable with competing parameter baselines".to_string(),
            true,
            "Freeze the fit subset, compare competing baselines on held-out observables, and test parameter identifiability.".to_string(),
        );
    }
    if layers.iter().any(|layer| layer == "statistical_inference") {
        return (
            "frozen split, declared null family, and multiplicity policy".to_string(),
            true,
            "Freeze the split and resampling unit before testing the declared null family."
                .to_string(),
        );
    }
    if layers.iter().any(|layer| layer == "performance_backend") {
        return (
            "identical inputs, trace parity, and hardware metadata".to_string(),
            true,
            "Compare identical inputs with trace parity and record hardware and physical-validity constraints.".to_string(),
        );
    }
    (
        "source snapshot, hashes, units, and independent transcription".to_string(),
        true,
        "Reconstruct the source proposition from a hashed snapshot and compare an independent transcription.".to_string(),
    )
}

fn mechanism_for(corpus: &str) -> String {
    for (needle, mechanism) in [
        ("ward", "photon-graviton-ward"),
        ("mie", "mie-channel-oracle"),
        ("tcmt", "tcmt-channel-oracle"),
        ("sfwm", "sfwm-source-reproduction"),
        ("heliosphere", "heliosphere-protocol"),
        ("lbm", "lbm-parity"),
        ("ultrametric", "ultrametric-parity"),
        ("sedenion", "cayley-dickson-formal"),
        ("zero divisor", "cayley-dickson-formal"),
    ] {
        if corpus.contains(needle) {
            return mechanism.to_string();
        }
    }
    "unclassified-shared-mechanism".to_string()
}

fn cost_for(layers: &[String]) -> String {
    if layers
        .iter()
        .any(|layer| layer == "performance_backend" || layer == "phenomenological_mapping")
    {
        "high".to_string()
    } else if layers.len() > 1 {
        "medium".to_string()
    } else {
        "low".to_string()
    }
}

fn is_false_confidence_candidate(entry: &QueueEntry) -> bool {
    matches!(
        entry.status.as_str(),
        "Verified" | "Established" | "Theoretical"
    ) || !entry.risk_flags.is_empty()
}

fn is_open_adjudication_candidate(entry: &QueueEntry) -> bool {
    !matches!(
        entry.status.as_str(),
        "Verified" | "Established" | "Closed/Negative-Result"
    )
}

fn spotlight(entries: &[QueueEntry]) -> Vec<QueueSpotlight> {
    entries
        .iter()
        .take(32)
        .map(|entry| QueueSpotlight {
            claim_id: entry.claim_id.clone(),
            statement: entry.statement.clone(),
            status: entry.status.clone(),
            epistemic_layers: entry.epistemic_layers.clone(),
            compound: entry.compound,
            risk_flags: entry.risk_flags.clone(),
            dependent_count: entry.dependent_count,
            ready_discriminator: entry.ready_discriminator,
            discriminator: entry.discriminator.clone(),
            falsifier: entry.falsifier.clone(),
            shared_mechanism: entry.shared_mechanism.clone(),
            estimated_cost: entry.estimated_cost.clone(),
        })
        .collect()
}

fn next_tranches(claims: &[ClaimRecord]) -> Vec<Tranche> {
    let existing_ids = claims
        .iter()
        .map(|claim| claim.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let declared_ids = |ids: &[&str]| {
        ids.iter()
            .filter(|id| existing_ids.contains(**id))
            .map(|id| (*id).to_string())
            .collect::<Vec<_>>()
    };
    vec![
        Tranche {
            order: 1,
            name: "P2A source-faithful channel semantics".to_string(),
            claim_ids: declared_ids(&[
                "C-848", "C-849", "C-850", "C-1640", "C-1641", "C-1642", "C-1643",
                "C-1644",
            ]),
            discriminator: "Complex channel amplitudes with separate scattering, extinction, absorption, unitarity, reciprocity, time reversal, and passive-loss predicates.".to_string(),
            holdout_rule: "Do not promote C-849 or C-850 during oracle repair; freeze the held-out sweep before comparison.".to_string(),
            excluded_claims: vec!["C-851 antiresonance control".to_string()],
            blocked_claims: vec![
                "C-864 source parameter constraint".to_string(),
                "C-867 paper-comparison observable".to_string(),
                "C-1638 energy constraint successor".to_string(),
                "C-1639 dependent energy successor".to_string(),
            ],
        },
        Tranche {
            order: 2,
            name: "P2B held-out Mie and TCMT reproduction".to_string(),
            claim_ids: declared_ids(&["C-849", "C-850"]),
            discriminator: "Frozen training parameters and held-out complex channel amplitudes with phase and magnitude errors reported separately.".to_string(),
            holdout_rule: "Fit only on the declared training subset and never select a winner on the held-out subset.".to_string(),
            excluded_claims: vec![],
            blocked_claims: vec![],
        },
        Tranche {
            order: 3,
            name: "SFWM source reproduction".to_string(),
            claim_ids: declared_ids(&["C-832", "C-834", "C-839"]),
            discriminator: "Separate paper-calibrated, Sellmeier-derived, direct SFWM, cascaded SHG plus SPDC, and total detected infrared quantities.".to_string(),
            holdout_rule: "Preserve C-833 as the Sellmeier-ordering control and do not infer a numerical rate bound from a qualitative dominance statement.".to_string(),
            excluded_claims: vec!["C-833 Sellmeier-ordering control".to_string()],
            blocked_claims: vec![],
        },
    ]
}

fn scalar_count(conn: &Connection, sql: &str) -> Result<i64> {
    conn.query_row(sql, [], |row| row.get(0))
        .map_err(Into::into)
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
