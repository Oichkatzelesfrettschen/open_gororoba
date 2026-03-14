use anyhow::{Context, Result, bail};
use clap::Parser;
use provenance_store::{ControlPlaneCompatKind, ProvenanceStore};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use toml::Value;

#[derive(Parser, Debug)]
#[command(
    name = "integrity-resolution",
    about = "Build or verify canonical integrity-resolution lane registries"
)]
struct Args {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,
    #[arg(long, default_value_t = false)]
    verify: bool,
    #[arg(long, default_value = "registry/conflict_markers.toml")]
    conflict_out: PathBuf,
    #[arg(long, default_value = "registry/lacunae.toml")]
    lacunae_out: PathBuf,
    #[arg(long, default_value = "registry/schema_signatures.toml")]
    schema_out: PathBuf,
}

#[derive(Debug, Clone)]
struct ConflictMarker {
    id: String,
    marker_kind: String,
    severity: String,
    status: String,
    claim_refs: Vec<String>,
    source_registry: String,
    source_document: String,
    section_label: String,
    line_start: i64,
    line_end: i64,
    positive_evidence: Vec<String>,
    negative_evidence: Vec<String>,
    jaccard_overlap: f64,
    notes: String,
}

#[derive(Debug, Clone)]
struct Lacuna {
    id: String,
    origin: String,
    area: String,
    title: String,
    description: String,
    priority: String,
    status: String,
    claim_refs: Vec<String>,
    source_refs: Vec<String>,
    related_marker_ids: Vec<String>,
}

#[derive(Debug, Clone)]
struct SchemaSignature {
    id: String,
    path: String,
    top_level_keys: Vec<String>,
    schema_version: String,
    schema_sha256: String,
    content_sha256: String,
    array_table_count: usize,
    table_count: usize,
    shape_json: String,
}

#[derive(Debug, Default, Clone)]
struct ConflictMeta {
    marker_count: usize,
    high_severity_count: usize,
    medium_severity_count: usize,
    low_severity_count: usize,
    kind_count: usize,
}

#[derive(Debug, Default, Clone)]
struct LacunaMeta {
    lacuna_count: usize,
    open_count: usize,
    high_priority_count: usize,
    medium_priority_count: usize,
    low_priority_count: usize,
    origin_kind_count: usize,
}

#[derive(Debug, Default, Clone)]
struct SchemaMeta {
    signature_count: usize,
    version: usize,
}

const POSITIVE_TERMS: &[&str] = &[
    "verified",
    "established",
    "confirmed",
    "holds",
    "consistent",
    "valid",
    "supported",
    "reproduced",
];
const NEGATIVE_TERMS: &[&str] = &[
    "refuted",
    "invalid",
    "fails",
    "contradiction",
    "inconsistent",
    "obstruction",
    "inconclusive",
    "insufficient",
];
const UNRESOLVED_STATUS_TOKENS: &[&str] = &[
    "PARTIAL",
    "INCONCLUSIVE",
    "THEORETICAL",
    "CLOSED_SOURCE_INSUFFICIENT",
    "CLOSED_METHODOLOGY_INSUFFICIENT",
];
const VERIFIED_STATUS_TOKENS: &[&str] = &["VERIFIED", "ESTABLISHED"];
const REFUTED_STATUS_TOKENS: &[&str] = &["REFUTED", "CLOSED_REFUTED", "CLOSED_NEGATIVE_RESULT"];
const STOPWORDS: &[&str] = &[
    "about", "across", "after", "against", "all", "also", "analysis", "being", "between", "both",
    "claim", "data", "does", "each", "from", "have", "into", "its", "model", "more", "must",
    "needs", "only", "other", "over", "results", "should", "some", "than", "their", "there",
    "these", "this", "through", "under", "using", "when", "where", "which", "while", "with",
    "without",
];

const DB_BACKED_COMPAT_SIGNATURE_PATHS: &[&str] = &[
    "registry/claims.toml",
    "registry/insights.toml",
    "registry/experiments.toml",
    "registry/binaries.toml",
];

fn main() -> Result<()> {
    let args = Args::parse();
    let repo_root = args.repo_root.canonicalize().context("resolve repo root")?;
    if args.verify {
        return verify_integrity_resolution(&repo_root, &args);
    }
    build_integrity_resolution(&repo_root, &args)
}

fn build_integrity_resolution(repo_root: &Path, args: &Args) -> Result<()> {
    let claims_rows = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Claims,
            "registry/claims.toml",
        )?,
        "claim",
    )
    .to_vec();
    let insights_rows = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Insights,
            "registry/insights.toml",
        )?,
        "insight",
    )
    .to_vec();
    let experiments_rows = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Experiments,
            "registry/experiments.toml",
        )?,
        "experiment",
    )
    .to_vec();
    let docs_root_rows = table_array(
        &load_toml(&repo_root.join("registry/docs_root_narratives.toml"))?,
        "document",
    )
    .to_vec();
    let research_rows = table_array(
        &load_toml(&repo_root.join("registry/research_narratives.toml"))?,
        "document",
    )
    .to_vec();
    let external_rows = table_array(
        &load_toml(&repo_root.join("registry/external_sources.toml"))?,
        "document",
    )
    .to_vec();
    let artifact_rows = table_array(
        &load_toml(&repo_root.join("registry/data_artifact_narratives.toml"))?,
        "document",
    )
    .to_vec();
    let legacy_lacunae = if repo_root.join(&args.lacunae_out).exists() {
        table_array(&load_toml(&repo_root.join(&args.lacunae_out))?, "lacuna").to_vec()
    } else {
        Vec::new()
    };

    let (conflict_rows, conflict_meta) = build_conflict_markers(
        &claims_rows,
        &docs_root_rows,
        &research_rows,
        &external_rows,
        &artifact_rows,
    )?;
    let (lacunae_rows, lacunae_meta) = build_lacunae(
        &claims_rows,
        &conflict_rows,
        &insights_rows,
        &experiments_rows,
        &legacy_lacunae,
    )?;

    write_ascii(
        &repo_root.join(&args.conflict_out),
        &render_conflict_markers(&conflict_rows, &conflict_meta),
    )?;
    write_ascii(
        &repo_root.join(&args.lacunae_out),
        &render_lacunae(&lacunae_rows, &lacunae_meta),
    )?;

    let registry_paths = vec![
        "registry/artifact_experiment_links.toml",
        "registry/external_sources.toml",
        "registry/project_csv_canonical_datasets.toml",
        "registry/dataset_label_aliases.toml",
        "registry/project_csv_canonical.toml",
        "registry/claims_atoms.toml",
        "registry/claims_evidence_edges.toml",
        "registry/knowledge/equation_atoms_v2.toml",
        "registry/knowledge/equation_symbol_table.toml",
        "registry/knowledge/proof_skeletons.toml",
        "registry/knowledge/derivation_steps.toml",
        "registry/bibliography.toml",
        "registry/bibliography_normalized.toml",
        "registry/provenance_sources.toml",
        "registry/narrative_paragraph_atoms.toml",
        "registry/conflict_markers.toml",
        "registry/lacunae.toml",
        "registry/registry_events.toml",
        "registry/third_party_markdown_cache.toml",
        "registry/third_party_source_verification.toml",
    ]
    .into_iter()
    .filter(|path| !DB_BACKED_COMPAT_SIGNATURE_PATHS.contains(path))
    .map(ToOwned::to_owned)
    .collect::<Vec<_>>();
    let (signature_rows, signature_meta) = build_schema_signatures(repo_root, &registry_paths)?;
    write_ascii(
        &repo_root.join(&args.schema_out),
        &render_schema_signatures(&signature_rows, &signature_meta, &registry_paths),
    )?;

    println!(
        "Wrote integrity-resolution registry lane artifacts (canonical registry-integrity-resolution script): conflict_markers={} lacunae={} schema_signatures={}",
        conflict_rows.len(),
        lacunae_rows.len(),
        signature_rows.len()
    );
    Ok(())
}

fn verify_integrity_resolution(repo_root: &Path, args: &Args) -> Result<()> {
    let conflict_path = repo_root.join(&args.conflict_out);
    let lacunae_path = repo_root.join(&args.lacunae_out);
    for path in [&conflict_path, &lacunae_path] {
        if !path.exists() {
            println!("ERROR: missing required file: {}", path.display());
            std::process::exit(1);
        }
    }
    assert_ascii_file(&conflict_path)?;
    assert_ascii_file(&lacunae_path)?;

    let claims = table_array(
        &load_control_plane_registry(
            repo_root,
            &args.db,
            ControlPlaneCompatKind::Claims,
            "registry/claims.toml",
        )?,
        "claim",
    )
    .to_vec();
    let conflict_raw = load_toml(&conflict_path)?;
    let lacunae_raw = load_toml(&lacunae_path)?;
    let claim_ids = claims
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .collect::<BTreeSet<_>>();
    let markers = table_array(&conflict_raw, "marker");
    let lacunae = table_array(&lacunae_raw, "lacuna");

    let mut failures = Vec::new();

    if table_int_in(&conflict_raw, "conflict_markers", "marker_count") != markers.len() as i64 {
        failures.push("conflict_markers marker_count metadata mismatch".to_string());
    }
    let mut seen_marker_ids = BTreeSet::new();
    let mut severity_counts = BTreeMap::from([
        ("high".to_string(), 0usize),
        ("medium".to_string(), 0usize),
        ("low".to_string(), 0usize),
    ]);
    for row in markers {
        let marker_id = table_str(row, "id").to_string();
        if !seen_marker_ids.insert(marker_id.clone()) {
            failures.push(format!("duplicate conflict marker id: {}", marker_id));
            break;
        }
        let severity = table_str(row, "severity").to_string();
        if let Some(count) = severity_counts.get_mut(&severity) {
            *count += 1;
        } else {
            failures.push(format!(
                "invalid marker severity: {} -> {}",
                marker_id, severity
            ));
        }
        let line_start = table_int(row, "line_start");
        let line_end = table_int(row, "line_end");
        if line_start < 0 || line_end < 0 || (line_start > 0 && line_end < line_start) {
            failures.push(format!(
                "invalid marker line span: {} ({}, {})",
                marker_id, line_start, line_end
            ));
        }
        for claim_ref in string_list(row, "claim_refs") {
            if !claim_ids.contains(&claim_ref) {
                failures.push(format!(
                    "marker references unknown claim id: {} -> {}",
                    marker_id, claim_ref
                ));
            }
        }
        if table_str(row, "marker_kind").trim().is_empty() {
            failures.push(format!("marker missing marker_kind: {}", marker_id));
        }
        if table_str(row, "source_registry").trim().is_empty() {
            failures.push(format!("marker missing source_registry: {}", marker_id));
        }
        if table_str(row, "status").trim().is_empty() {
            failures.push(format!("marker missing status: {}", marker_id));
        }
        if !value_is_array(row, "positive_evidence") {
            failures.push(format!("marker positive_evidence not list: {}", marker_id));
        }
        if !value_is_array(row, "negative_evidence") {
            failures.push(format!("marker negative_evidence not list: {}", marker_id));
        }
    }
    if table_int_in(&conflict_raw, "conflict_markers", "high_severity_count")
        != severity_counts.get("high").copied().unwrap_or(0) as i64
    {
        failures.push("conflict_markers high_severity_count metadata mismatch".to_string());
    }
    if table_int_in(&conflict_raw, "conflict_markers", "medium_severity_count")
        != severity_counts.get("medium").copied().unwrap_or(0) as i64
    {
        failures.push("conflict_markers medium_severity_count metadata mismatch".to_string());
    }
    if table_int_in(&conflict_raw, "conflict_markers", "low_severity_count")
        != severity_counts.get("low").copied().unwrap_or(0) as i64
    {
        failures.push("conflict_markers low_severity_count metadata mismatch".to_string());
    }

    if table_int_in(&lacunae_raw, "lacunae", "lacuna_count") != lacunae.len() as i64 {
        failures.push("lacunae lacuna_count metadata mismatch".to_string());
    }
    let mut seen_lacuna_ids = BTreeSet::new();
    let mut open_count = 0usize;
    let mut priority_counts = BTreeMap::from([
        ("high".to_string(), 0usize),
        ("medium".to_string(), 0usize),
        ("low".to_string(), 0usize),
    ]);
    for row in lacunae {
        let lacuna_id = table_str(row, "id").to_string();
        if !seen_lacuna_ids.insert(lacuna_id.clone()) {
            failures.push(format!("duplicate lacuna id: {}", lacuna_id));
            break;
        }
        if table_str(row, "status") == "open" {
            open_count += 1;
        }
        let priority = table_str(row, "priority").to_string();
        if let Some(count) = priority_counts.get_mut(&priority) {
            *count += 1;
        } else {
            failures.push(format!(
                "invalid lacuna priority: {} -> {}",
                lacuna_id, priority
            ));
        }
        for claim_ref in string_list(row, "claim_refs") {
            if !claim_ids.contains(&claim_ref) {
                failures.push(format!(
                    "lacuna references unknown claim id: {} -> {}",
                    lacuna_id, claim_ref
                ));
            }
        }
        for marker_id in string_list(row, "related_marker_ids") {
            if !seen_marker_ids.contains(&marker_id) {
                failures.push(format!(
                    "lacuna references unknown conflict marker id: {} -> {}",
                    lacuna_id, marker_id
                ));
            }
        }
        if table_str(row, "title").trim().is_empty() {
            failures.push(format!("lacuna missing title: {}", lacuna_id));
        }
        if table_str(row, "description").trim().is_empty() {
            failures.push(format!("lacuna missing description: {}", lacuna_id));
        }
        if table_str(row, "origin").trim().is_empty() {
            failures.push(format!("lacuna missing origin: {}", lacuna_id));
        }
    }
    if table_int_in(&lacunae_raw, "lacunae", "open_count") != open_count as i64 {
        failures.push("lacunae open_count metadata mismatch".to_string());
    }
    if table_int_in(&lacunae_raw, "lacunae", "high_priority_count")
        != priority_counts.get("high").copied().unwrap_or(0) as i64
    {
        failures.push("lacunae high_priority_count metadata mismatch".to_string());
    }
    if table_int_in(&lacunae_raw, "lacunae", "medium_priority_count")
        != priority_counts.get("medium").copied().unwrap_or(0) as i64
    {
        failures.push("lacunae medium_priority_count metadata mismatch".to_string());
    }
    if table_int_in(&lacunae_raw, "lacunae", "low_priority_count")
        != priority_counts.get("low").copied().unwrap_or(0) as i64
    {
        failures.push("lacunae low_priority_count metadata mismatch".to_string());
    }

    if !failures.is_empty() {
        println!(
            "ERROR: integrity-resolution registry lane verification failed (canonical registry-integrity-resolution script)."
        );
        for item in failures.iter().take(250) {
            println!("- {}", item);
        }
        if failures.len() > 250 {
            println!("- ... and {} more failures", failures.len() - 250);
        }
        std::process::exit(1);
    }

    println!(
        "OK: integrity-resolution registry lane verified (canonical registry-integrity-resolution script). markers={} lacunae={}",
        markers.len(),
        lacunae.len()
    );
    Ok(())
}

fn build_conflict_markers(
    claims_rows: &[Value],
    docs_root_rows: &[Value],
    research_rows: &[Value],
    external_rows: &[Value],
    artifact_rows: &[Value],
) -> Result<(Vec<ConflictMarker>, ConflictMeta)> {
    let mut markers = Vec::new();
    let mut seq = 0usize;

    let mut claim_by_id = BTreeMap::<String, (String, String)>::new();
    let mut statement_tokens = BTreeMap::<String, BTreeSet<String>>::new();
    for row in claims_rows {
        let claim_id = collapse(table_str(row, "id"));
        if claim_id.is_empty() {
            continue;
        }
        let status_token = status_token(table_str(row, "status"));
        let statement = collapse(table_str(row, "statement"));
        claim_by_id.insert(claim_id.clone(), (statement.clone(), status_token.clone()));
        statement_tokens.insert(claim_id.clone(), token_set(&statement)?);
        let lower = statement.to_lowercase();
        let has_positive = POSITIVE_TERMS.iter().any(|term| lower.contains(term));
        let has_negative = NEGATIVE_TERMS.iter().any(|term| lower.contains(term));
        if REFUTED_STATUS_TOKENS.contains(&status_token.as_str()) && has_positive {
            seq += 1;
            markers.push(ConflictMarker {
                id: format!("CM-{:04}", seq),
                marker_kind: "claim_status_statement_tension".to_string(),
                severity: "high".to_string(),
                status: "open".to_string(),
                claim_refs: vec![claim_id.clone()],
                source_registry: "registry/claims.toml".to_string(),
                source_document: claim_id.clone(),
                section_label: "statement".to_string(),
                line_start: 0,
                line_end: 0,
                positive_evidence: vec![statement.clone()],
                negative_evidence: vec![status_token.clone()],
                jaccard_overlap: 0.0,
                notes: "Refuted/negative status conflicts with positive language in statement."
                    .to_string(),
            });
        }
        if VERIFIED_STATUS_TOKENS.contains(&status_token.as_str()) && has_negative {
            seq += 1;
            markers.push(ConflictMarker {
                id: format!("CM-{:04}", seq),
                marker_kind: "claim_status_statement_tension".to_string(),
                severity: "high".to_string(),
                status: "open".to_string(),
                claim_refs: vec![claim_id.clone()],
                source_registry: "registry/claims.toml".to_string(),
                source_document: claim_id.clone(),
                section_label: "statement".to_string(),
                line_start: 0,
                line_end: 0,
                positive_evidence: vec![status_token.clone()],
                negative_evidence: vec![statement.clone()],
                jaccard_overlap: 0.0,
                notes: "Verified status conflicts with negative language in statement.".to_string(),
            });
        }
    }

    let claim_ids = claim_by_id.keys().cloned().collect::<Vec<_>>();
    for (idx, a_id) in claim_ids.iter().enumerate() {
        let Some((a_statement, a_status)) = claim_by_id.get(a_id).cloned() else {
            continue;
        };
        if !VERIFIED_STATUS_TOKENS.contains(&a_status.as_str())
            && !REFUTED_STATUS_TOKENS.contains(&a_status.as_str())
        {
            continue;
        }
        for b_id in claim_ids.iter().skip(idx + 1) {
            let Some((b_statement, b_status)) = claim_by_id.get(b_id).cloned() else {
                continue;
            };
            if !VERIFIED_STATUS_TOKENS.contains(&b_status.as_str())
                && !REFUTED_STATUS_TOKENS.contains(&b_status.as_str())
            {
                continue;
            }
            let opposite = (VERIFIED_STATUS_TOKENS.contains(&a_status.as_str())
                && REFUTED_STATUS_TOKENS.contains(&b_status.as_str()))
                || (REFUTED_STATUS_TOKENS.contains(&a_status.as_str())
                    && VERIFIED_STATUS_TOKENS.contains(&b_status.as_str()));
            if !opposite {
                continue;
            }
            let score = jaccard(
                statement_tokens.get(a_id).cloned().unwrap_or_default(),
                statement_tokens.get(b_id).cloned().unwrap_or_default(),
            );
            if score < 0.70 {
                continue;
            }
            seq += 1;
            markers.push(ConflictMarker {
                id: format!("CM-{:04}", seq),
                marker_kind: "claim_semantic_status_conflict".to_string(),
                severity: "medium".to_string(),
                status: "open".to_string(),
                claim_refs: vec![a_id.clone(), b_id.clone()],
                source_registry: "registry/claims.toml".to_string(),
                source_document: format!("{}|{}", a_id, b_id),
                section_label: "cross_claim_statement".to_string(),
                line_start: 0,
                line_end: 0,
                positive_evidence: vec![
                    format!("{}: {}", a_id, a_statement),
                    format!("{}: {}", b_id, b_statement),
                ],
                negative_evidence: vec![
                    format!("{}: {}", a_id, a_status),
                    format!("{}: {}", b_id, b_status),
                ],
                jaccard_overlap: (score * 1_000_000.0).round() / 1_000_000.0,
                notes: "Semantically similar claims have opposite truth-status categories."
                    .to_string(),
            });
        }
    }

    let corpora = [
        ("registry/docs_root_narratives.toml", docs_root_rows),
        ("registry/research_narratives.toml", research_rows),
        ("registry/external_sources.toml", external_rows),
        ("registry/data_artifact_narratives.toml", artifact_rows),
    ];
    for (source_registry, rows) in corpora {
        for row in rows {
            let body = table_str(row, "body_markdown");
            if body.trim().is_empty() {
                continue;
            }
            let source_markdown = collapse(table_str(row, "source_markdown"));
            let source_uid = collapse(table_str(row, "id"));
            for section in split_sections(body)? {
                let text = collapse(&section.body);
                if text.is_empty() {
                    continue;
                }
                let lower = text.to_lowercase();
                let pos_hits = POSITIVE_TERMS
                    .iter()
                    .filter(|term| lower.contains(**term))
                    .map(|s| (*s).to_string())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let neg_hits = NEGATIVE_TERMS
                    .iter()
                    .filter(|term| lower.contains(**term))
                    .map(|s| (*s).to_string())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                if pos_hits.is_empty() || neg_hits.is_empty() {
                    continue;
                }
                seq += 1;
                markers.push(ConflictMarker {
                    id: format!("CM-{:04}", seq),
                    marker_kind: "section_polarity_conflict".to_string(),
                    severity: "medium".to_string(),
                    status: "open".to_string(),
                    claim_refs: extract_claim_refs(&text)?,
                    source_registry: source_registry.to_string(),
                    source_document: if source_markdown.is_empty() {
                        source_uid.clone()
                    } else {
                        source_markdown.clone()
                    },
                    section_label: section.title.clone(),
                    line_start: section.line_start,
                    line_end: section.line_end,
                    positive_evidence: pos_hits,
                    negative_evidence: neg_hits,
                    jaccard_overlap: 0.0,
                    notes: "Section contains both positive and negative validation language."
                        .to_string(),
                });
            }
        }
    }

    markers.sort_by(|a, b| a.id.cmp(&b.id));
    let mut kind_counts = BTreeSet::new();
    let mut meta = ConflictMeta::default();
    for marker in &markers {
        kind_counts.insert(marker.marker_kind.clone());
        match marker.severity.as_str() {
            "high" => meta.high_severity_count += 1,
            "medium" => meta.medium_severity_count += 1,
            "low" => meta.low_severity_count += 1,
            _ => {}
        }
    }
    meta.marker_count = markers.len();
    meta.kind_count = kind_counts.len();
    Ok((markers, meta))
}

fn build_lacunae(
    claims_rows: &[Value],
    conflict_markers: &[ConflictMarker],
    insights_rows: &[Value],
    experiments_rows: &[Value],
    legacy_lacunae: &[Value],
) -> Result<(Vec<Lacuna>, LacunaMeta)> {
    let mut rows = Vec::new();
    let mut seen_ids = BTreeSet::new();

    for row in legacy_lacunae {
        let lacuna_id = collapse(table_str(row, "id"));
        if lacuna_id.is_empty() || !seen_ids.insert(lacuna_id.clone()) {
            continue;
        }
        rows.push(Lacuna {
            id: lacuna_id,
            origin: "legacy_manual".to_string(),
            area: {
                let area = collapse(table_str(row, "area"));
                if area.is_empty() {
                    "general".to_string()
                } else {
                    area
                }
            },
            title: collapse(table_str(row, "title")),
            description: collapse(table_str(row, "description")),
            priority: {
                let priority = collapse(table_str(row, "priority"));
                if priority.is_empty() {
                    "medium".to_string()
                } else {
                    priority
                }
            },
            status: {
                let status = collapse(table_str(row, "status"));
                if status.is_empty() {
                    "open".to_string()
                } else {
                    status
                }
            },
            claim_refs: extract_claim_refs(table_str(row, "description"))?,
            source_refs: Vec::new(),
            related_marker_ids: Vec::new(),
        });
    }

    for row in claims_rows {
        let claim_id = collapse(table_str(row, "id"));
        let status = status_token(table_str(row, "status"));
        if !UNRESOLVED_STATUS_TOKENS.contains(&status.as_str()) {
            continue;
        }
        let lacuna_id = if let Some(stripped) = claim_id.strip_prefix("C-") {
            format!("L-{}", stripped)
        } else {
            format!("L-AUTO-{:04}", rows.len() + 1)
        };
        if !seen_ids.insert(lacuna_id.clone()) {
            continue;
        }
        let statement = collapse(table_str(row, "statement"));
        rows.push(Lacuna {
            id: lacuna_id,
            origin: "claims_status_scan".to_string(),
            area: "claims".to_string(),
            title: format!("Unresolved claim status for {}", claim_id),
            description: format!(
                "{} remains unresolved with status token {}: {}",
                claim_id, status, statement
            ),
            priority: if ["INCONCLUSIVE", "PARTIAL"].contains(&status.as_str()) {
                "high".to_string()
            } else {
                "medium".to_string()
            },
            status: "open".to_string(),
            claim_refs: vec![claim_id],
            source_refs: vec!["registry/claims.toml".to_string()],
            related_marker_ids: Vec::new(),
        });
    }

    for marker in conflict_markers {
        let marker_id = marker.id.clone();
        let lacuna_id = format!(
            "L-CM-{}",
            marker_id.split('-').next_back().unwrap_or("0000")
        );
        if !seen_ids.insert(lacuna_id.clone()) {
            continue;
        }
        rows.push(Lacuna {
            id: lacuna_id,
            origin: "conflict_marker_scan".to_string(),
            area: "consistency".to_string(),
            title: format!("Resolve conflict marker {}", marker_id),
            description: if marker.notes.is_empty() {
                "Unresolved contradiction marker.".to_string()
            } else {
                marker.notes.clone()
            },
            priority: if marker.severity == "high" {
                "high".to_string()
            } else {
                "medium".to_string()
            },
            status: "open".to_string(),
            claim_refs: marker.claim_refs.clone(),
            source_refs: vec![
                marker.source_registry.clone(),
                marker.source_document.clone(),
            ],
            related_marker_ids: vec![marker_id],
        });
    }

    let claim_ids = claims_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .collect::<BTreeSet<_>>();
    let mut dangling = 0usize;
    for (source_name, source_rows, claims_key) in [
        ("insights", insights_rows, "claims"),
        ("experiments", experiments_rows, "claims"),
    ] {
        for row in source_rows {
            let source_id = collapse(table_str(row, "id"));
            let refs = string_list(row, claims_key);
            let missing = refs
                .into_iter()
                .filter(|claim_ref| !claim_ids.contains(claim_ref))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            if missing.is_empty() {
                continue;
            }
            dangling += 1;
            let lacuna_id = format!(
                "L-DANGLING-{}-{:03}",
                &source_name[..1].to_uppercase(),
                dangling
            );
            if !seen_ids.insert(lacuna_id.clone()) {
                continue;
            }
            rows.push(Lacuna {
                id: lacuna_id,
                origin: "crossref_scan".to_string(),
                area: source_name.to_string(),
                title: format!(
                    "Dangling claim references in {} entry {}",
                    source_name, source_id
                ),
                description: format!(
                    "Entry {} references unknown claims: {}",
                    source_id,
                    missing.join(", ")
                ),
                priority: "high".to_string(),
                status: "open".to_string(),
                claim_refs: Vec::new(),
                source_refs: vec![format!("registry/{}.toml", source_name)],
                related_marker_ids: Vec::new(),
            });
        }
    }

    rows.sort_by(|a, b| a.id.cmp(&b.id));
    let mut origin_kinds = BTreeSet::new();
    let mut meta = LacunaMeta::default();
    for row in &rows {
        origin_kinds.insert(row.origin.clone());
        if row.status == "open" {
            meta.open_count += 1;
        }
        match row.priority.as_str() {
            "high" => meta.high_priority_count += 1,
            "medium" => meta.medium_priority_count += 1,
            "low" => meta.low_priority_count += 1,
            _ => {}
        }
    }
    meta.lacuna_count = rows.len();
    meta.origin_kind_count = origin_kinds.len();
    Ok((rows, meta))
}

fn build_schema_signatures(
    repo_root: &Path,
    registry_paths: &[String],
) -> Result<(Vec<SchemaSignature>, SchemaMeta)> {
    let mut rows = Vec::new();
    for path in registry_paths {
        let full_path = repo_root.join(path);
        if !full_path.exists() {
            continue;
        }
        let text = fs::read_to_string(&full_path)
            .with_context(|| format!("read {}", full_path.display()))?;
        let data: Value =
            toml::from_str(&text).with_context(|| format!("parse TOML {}", full_path.display()))?;
        let mut top_level_keys = data
            .as_table()
            .map(|table| table.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        top_level_keys.sort();
        let shapes = top_level_keys
            .iter()
            .filter_map(|key| {
                data.get(key)
                    .map(|value| (key.clone(), shape_summary(value)))
            })
            .collect::<BTreeMap<_, _>>();
        let schema_payload = serde_json::json!({
            "path": path,
            "top_level_keys": top_level_keys,
            "shapes": shapes,
        });
        let normalized =
            serde_json::to_string(&schema_payload).context("serialize schema payload")?;
        let schema_sha256 = sha256_hex(normalized.as_bytes());
        let content_sha256 = sha256_hex(text.as_bytes());
        rows.push(SchemaSignature {
            id: String::new(),
            path: path.clone(),
            top_level_keys: top_level_keys.clone(),
            schema_version: "v1".to_string(),
            schema_sha256,
            content_sha256,
            array_table_count: top_level_keys
                .iter()
                .filter(|key| {
                    data.get(key.as_str())
                        .and_then(Value::as_array)
                        .map(|rows| rows.iter().all(Value::is_table))
                        .unwrap_or(false)
                })
                .count(),
            table_count: top_level_keys
                .iter()
                .filter(|key| data.get(key.as_str()).map(Value::is_table).unwrap_or(false))
                .count(),
            shape_json: normalized,
        });
    }
    rows.sort_by(|a, b| a.path.cmp(&b.path));
    for (idx, row) in rows.iter_mut().enumerate() {
        row.id = format!("SIG-{:04}", idx + 1);
    }
    let signature_count = rows.len();
    Ok((
        rows,
        SchemaMeta {
            signature_count,
            version: 1,
        },
    ))
}

fn render_conflict_markers(rows: &[ConflictMarker], meta: &ConflictMeta) -> String {
    let mut lines = vec![
        "# Conflict marker registry (integrity-resolution lane strict schema; legacy batch3 compatibility).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/integrity_resolution.rs.".to_string(),
        "".to_string(),
        "[conflict_markers]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("marker_count = {}", meta.marker_count),
        format!("high_severity_count = {}", meta.high_severity_count),
        format!("medium_severity_count = {}", meta.medium_severity_count),
        format!("low_severity_count = {}", meta.low_severity_count),
        format!("kind_count = {}", meta.kind_count),
        "".to_string(),
    ];
    for row in rows {
        lines.extend([
            "[[marker]]".to_string(),
            format!("id = {}", esc(&row.id)),
            format!("marker_kind = {}", esc(&row.marker_kind)),
            format!("severity = {}", esc(&row.severity)),
            format!("status = {}", esc(&row.status)),
            format!("claim_refs = {}", render_list(&row.claim_refs)),
            format!("source_registry = {}", esc(&row.source_registry)),
            format!("source_document = {}", esc(&row.source_document)),
            format!("section_label = {}", esc(&row.section_label)),
            format!("line_start = {}", row.line_start),
            format!("line_end = {}", row.line_end),
            format!(
                "positive_evidence = {}",
                render_list(&row.positive_evidence)
            ),
            format!(
                "negative_evidence = {}",
                render_list(&row.negative_evidence)
            ),
            format!("jaccard_overlap = {:.6}", row.jaccard_overlap),
            format!("notes = {}", esc(&row.notes)),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn render_lacunae(rows: &[Lacuna], meta: &LacunaMeta) -> String {
    let mut lines = vec![
        "# Lacunae registry (integrity-resolution lane strict schema; legacy batch3 compatibility).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/integrity_resolution.rs.".to_string(),
        "".to_string(),
        "[lacunae]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "status = \"active\"".to_string(),
        format!("lacuna_count = {}", meta.lacuna_count),
        format!("open_count = {}", meta.open_count),
        format!("high_priority_count = {}", meta.high_priority_count),
        format!("medium_priority_count = {}", meta.medium_priority_count),
        format!("low_priority_count = {}", meta.low_priority_count),
        format!("origin_kind_count = {}", meta.origin_kind_count),
        "".to_string(),
    ];
    for row in rows {
        lines.extend([
            "[[lacuna]]".to_string(),
            format!("id = {}", esc(&row.id)),
            format!("origin = {}", esc(&row.origin)),
            format!("area = {}", esc(&row.area)),
            format!("title = {}", esc(&row.title)),
            format!("description = {}", esc(&row.description)),
            format!("priority = {}", esc(&row.priority)),
            format!("status = {}", esc(&row.status)),
            format!("claim_refs = {}", render_list(&row.claim_refs)),
            format!("source_refs = {}", render_list(&row.source_refs)),
            format!(
                "related_marker_ids = {}",
                render_list(&row.related_marker_ids)
            ),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn render_schema_signatures(
    rows: &[SchemaSignature],
    meta: &SchemaMeta,
    registry_paths: &[String],
) -> String {
    let mut sorted_paths = registry_paths.to_vec();
    sorted_paths.sort();
    let mut lines = vec![
        "# Registry schema signatures (integrity-resolution lane strict schema; legacy batch3 compatibility).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/integrity_resolution.rs.".to_string(),
        "".to_string(),
        "[schema_signatures]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("version = {}", meta.version),
        format!("signature_count = {}", meta.signature_count),
        format!("registry_paths = {}", render_list(&sorted_paths)),
        "".to_string(),
    ];
    for row in rows {
        lines.extend([
            "[[signature]]".to_string(),
            format!("id = {}", esc(&row.id)),
            format!("path = {}", esc(&row.path)),
            format!("top_level_keys = {}", render_list(&row.top_level_keys)),
            format!("schema_version = {}", esc(&row.schema_version)),
            format!("schema_sha256 = {}", esc(&row.schema_sha256)),
            format!("content_sha256 = {}", esc(&row.content_sha256)),
            format!("array_table_count = {}", row.array_table_count),
            format!("table_count = {}", row.table_count),
            format!("shape_json = {}", esc(&row.shape_json)),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn status_token(status: &str) -> String {
    let mut token = collapse(status).to_uppercase();
    token = token.replace(['/', '-', ' '], "_");
    let cleaned = token
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .collect::<String>();
    let squashed = cleaned
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    if squashed.is_empty() {
        "UNSPECIFIED".to_string()
    } else {
        squashed
    }
}

fn extract_claim_refs(text: &str) -> Result<Vec<String>> {
    let claim_re = Regex::new(r"\bC-\d{3}\b")?;
    Ok(sorted_unique_regex(text, &claim_re))
}

fn token_set(text: &str) -> Result<BTreeSet<String>> {
    let word_re = Regex::new(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b")?;
    Ok(word_re
        .find_iter(&ascii_clean(text))
        .map(|m| m.as_str().to_lowercase())
        .filter(|token| token.len() >= 4 && !STOPWORDS.contains(&token.as_str()))
        .collect())
}

fn jaccard(a: BTreeSet<String>, b: BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(&b).count();
    let union = a.union(&b).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

#[derive(Debug, Clone)]
struct Section {
    title: String,
    line_start: i64,
    line_end: i64,
    body: String,
}

fn split_sections(text: &str) -> Result<Vec<Section>> {
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+?)\s*$")?;
    let lines = ascii_clean(text)
        .lines()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let mut sections = Vec::new();
    let mut current_title = "(root)".to_string();
    let mut current_start = 1i64;
    let mut current_lines = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        let line_no = idx as i64 + 1;
        if let Some(caps) = heading_re.captures(line) {
            sections.push(Section {
                title: current_title,
                line_start: current_start,
                line_end: current_start.max(line_no - 1),
                body: current_lines.join("\n"),
            });
            current_title = collapse(caps.get(2).map(|m| m.as_str()).unwrap_or_default());
            current_start = line_no;
            current_lines.clear();
            continue;
        }
        current_lines.push(line.clone());
    }
    sections.push(Section {
        title: current_title,
        line_start: current_start,
        line_end: current_start.max(lines.len() as i64),
        body: current_lines.join("\n"),
    });
    Ok(sections)
}

fn shape_summary(value: &Value) -> serde_json::Value {
    if let Some(table) = value.as_table() {
        let mut keys = table.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        serde_json::json!({"type": "table", "keys": keys})
    } else if let Some(array) = value.as_array() {
        if array.is_empty() {
            serde_json::json!({"type": "array", "row_count": 0, "entry_kind": "empty"})
        } else if array.iter().all(Value::is_table) {
            let key_sets = array
                .iter()
                .filter_map(Value::as_table)
                .map(|table| table.keys().cloned().collect::<BTreeSet<_>>())
                .collect::<Vec<_>>();
            let intersection = key_sets
                .iter()
                .cloned()
                .reduce(|a, b| a.intersection(&b).cloned().collect())
                .unwrap_or_default()
                .into_iter()
                .collect::<Vec<_>>();
            let union = key_sets
                .iter()
                .cloned()
                .reduce(|a, b| a.union(&b).cloned().collect())
                .unwrap_or_default()
                .into_iter()
                .collect::<Vec<_>>();
            serde_json::json!({
                "type": "array",
                "row_count": array.len(),
                "entry_kind": "table",
                "required_keys": intersection,
                "union_keys": union,
            })
        } else {
            let entry_types = array
                .iter()
                .map(|item| match item {
                    Value::String(_) => "string",
                    Value::Integer(_) => "integer",
                    Value::Float(_) => "float",
                    Value::Boolean(_) => "boolean",
                    Value::Array(_) => "array",
                    Value::Table(_) => "table",
                    Value::Datetime(_) => "datetime",
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            serde_json::json!({
                "type": "array",
                "row_count": array.len(),
                "entry_kind": "scalar_or_mixed",
                "entry_types": entry_types,
            })
        }
    } else if value.is_str() {
        serde_json::json!({"type": "str"})
    } else if value.is_integer() {
        serde_json::json!({"type": "int"})
    } else if value.is_float() {
        serde_json::json!({"type": "float"})
    } else if value.is_bool() {
        serde_json::json!({"type": "bool"})
    } else if value.is_datetime() {
        serde_json::json!({"type": "datetime"})
    } else {
        serde_json::json!({"type": "unknown"})
    }
}

fn load_toml(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse TOML {}", path.display()))
}

fn load_control_plane_registry(
    repo_root: &Path,
    db_rel_path: &Path,
    kind: ControlPlaneCompatKind,
    fallback_rel_path: &str,
) -> Result<Value> {
    let db_path = repo_root.join(db_rel_path);
    if db_path.exists() {
        let mut store = ProvenanceStore::open(&db_path)
            .with_context(|| format!("open canonical control-plane DB {}", db_path.display()))?;
        let text = store.control_plane_compat_text(kind).with_context(|| {
            format!(
                "render {:?} compatibility text from {}",
                kind,
                db_path.display()
            )
        })?;
        return toml::from_str(&text)
            .with_context(|| format!("parse {:?} compatibility TOML", kind));
    }
    load_toml(&repo_root.join(fallback_rel_path))
}

fn table_array<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn table_str<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn table_int(value: &Value, key: &str) -> i64 {
    value.get(key).and_then(Value::as_integer).unwrap_or(0)
}

fn table_int_in(value: &Value, table: &str, key: &str) -> i64 {
    value
        .get(table)
        .and_then(|child| child.get(key))
        .and_then(Value::as_integer)
        .unwrap_or(-1)
}

fn string_list(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn value_is_array(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_array).is_some()
}

fn write_ascii(path: &Path, text: &str) -> Result<()> {
    assert_ascii(text, &path.display().to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

fn assert_ascii_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    assert_ascii(&text, &path.display().to_string())
}

fn assert_ascii(text: &str, context: &str) -> Result<()> {
    let bad: BTreeSet<char> = text.chars().filter(|ch| (*ch as u32) > 127).collect();
    if !bad.is_empty() {
        let sample: String = bad.iter().take(20).copied().collect();
        bail!("ERROR: Non-ASCII output in {context}: {:?}", sample);
    }
    Ok(())
}

fn ascii_clean(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        if ch == '\n' || ch == '\r' || ch == '\t' {
            out.push(ch);
        } else if (ch as u32) < 32 {
            out.push(' ');
        } else if (ch as u32) <= 127 {
            out.push(ch);
        } else {
            out.push_str(&format!("\\u{:04X}", ch as u32));
        }
    }
    out
}

fn collapse(text: &str) -> String {
    ascii_clean(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sorted_unique_regex(text: &str, regex: &Regex) -> Vec<String> {
    regex
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn esc(value: &str) -> String {
    serde_json::to_string(&ascii_clean(value)).unwrap_or_else(|_| "\"\"".to_string())
}

fn render_list(values: &[String]) -> String {
    if values.is_empty() {
        "[]".to_string()
    } else {
        format!(
            "[{}]",
            values
                .iter()
                .map(|item| esc(item))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
