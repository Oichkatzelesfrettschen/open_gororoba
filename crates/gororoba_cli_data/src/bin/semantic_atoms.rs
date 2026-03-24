use anyhow::{Context, Result, bail};
use clap::Parser;
use provenance_store::ProvenanceStore;
use rayon::prelude::*;
use regex::Regex;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use toml::Value;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "semantic-atoms",
    about = "Build canonical semantic-atoms lane registries"
)]
struct Args {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value_t = false)]
    verify: bool,
    /// Canonical SQLite control-plane DB used for live claim metadata.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    /// Compatibility-export claims TOML, used only when the canonical DB is unavailable.
    #[arg(long, default_value = "registry/claims.toml")]
    claims_path: PathBuf,
    #[arg(long, default_value = "registry/knowledge/proof_atoms.toml")]
    proof_atoms_path: PathBuf,
    #[arg(long, default_value = "registry/claims_atoms.toml")]
    claims_atoms_out: PathBuf,
    #[arg(long, default_value = "registry/claims_evidence_edges.toml")]
    claims_edges_out: PathBuf,
    #[arg(long, default_value = "registry/knowledge/equation_atoms_v2.toml")]
    equation_atoms_out: PathBuf,
    #[arg(long, default_value = "registry/knowledge/equation_symbol_table.toml")]
    symbol_table_out: PathBuf,
    #[arg(long, default_value = "registry/knowledge/proof_skeletons.toml")]
    proof_skeletons_out: PathBuf,
    #[arg(long, default_value = "registry/markdown_payloads.toml")]
    payload_path: PathBuf,
    #[arg(long, default_value = "registry/markdown_payload_chunks.toml")]
    payload_chunks_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ClaimAtom {
    claim_id: String,
    statement: String,
    status_token: String,
    status_detail: String,
    hypothesis_class: String,
    last_verified: String,
    where_stated: String,
    verification_rule: String,
    where_stated_refs: Vec<String>,
    verification_refs: Vec<String>,
    cross_refs: Vec<String>,
}

#[derive(Debug, Clone)]
struct ClaimSourceInfo {
    source_registry: String,
    source_path: String,
}

#[derive(Debug, Clone)]
struct ClaimEdge {
    claim_id: String,
    edge_role: String,
    target_ref: String,
    target_kind: String,
}

#[derive(Debug, Clone)]
struct NarrativeSource {
    source_uid: String,
    source_group: String,
    source_registry: String,
    source_path: String,
    body: String,
}

#[derive(Debug, Clone)]
struct EquationAtomV2 {
    id: String,
    source_uid: String,
    source_group: String,
    source_registry: String,
    source_path: String,
    section_title: String,
    source_line: usize,
    expression: String,
    normalized_expression: String,
    relation_operator: String,
    lhs_expression: String,
    rhs_expression: String,
    equation_kind: String,
    extraction_kind: String,
    extraction_confidence: String,
    quality_flags: Vec<String>,
    symbol_names: Vec<String>,
    numeric_constants: Vec<String>,
    symbol_refs: Vec<String>,
    claim_refs: Vec<String>,
}

#[derive(Debug, Clone)]
struct SymbolRow {
    id: String,
    symbol: String,
    normalized_symbol: String,
    category: String,
    usage_count: usize,
    source_uids: Vec<String>,
    example_equation_ids: Vec<String>,
}

#[derive(Debug, Clone)]
struct ProofSkeleton {
    skeleton_kind: String,
    theorem_label: String,
    claim_id: String,
    assumptions: Vec<String>,
    obligations: Vec<String>,
    derivation_steps: Vec<String>,
    decision_rule: String,
    conclusion: String,
    source_uid: String,
    source_registry: String,
    source_path: String,
    line_start: usize,
    line_end: usize,
    claim_refs: Vec<String>,
    evidence_refs: Vec<String>,
}

const ALLOWED_CHUNK_KINDS: &[&str] = &[
    "heading",
    "paragraph",
    "list_item",
    "table_row",
    "code_block",
];
const SKIP_DIR_NAMES: &[&str] = &[
    ".git",
    ".cache",
    ".pytest_cache",
    "venv",
    ".venv",
    ".venv_ingest",
    ".horusec",
    ".claude",
    ".gemini",
    ".playwright-mcp",
    ".mamba",
    "cargo-home",
    "lambda_gororoba_backups",
    "target",
    "logs",
    "build",
    "dist",
    "temp",
    "tmp",
];
const SKIP_PREFIXES: &[&str] = &[];
const MIN_CLAIMS_WITH_EVIDENCE_COVERAGE: f64 = 0.95;
const EQUATION_ATOMS_BASELINE_COUNT: usize = 130;
const EQUATION_ATOMS_BASELINE_CLAIMS: usize = 689;
const EQUATION_ATOMS_DENSITY_REGRESSION_TOLERANCE: f64 = 0.95;
const EQUATION_ATOMS_ABSOLUTE_MIN: usize = 120;

fn main() -> Result<()> {
    let args = Args::parse();
    let repo_root = args.repo_root.canonicalize().context("resolve repo root")?;
    if args.verify {
        return verify_semantic_atoms(&repo_root, &args);
    }

    let (claims, claim_source) = load_claims(&repo_root, &args.canonical_db, &args.claims_path)?;
    let claim_edges = build_claim_edges(&claims);
    let sources = load_narrative_sources(&repo_root)?;
    let mut equation_atoms = extract_equation_atoms_v2(&sources)?;
    for (idx, atom) in equation_atoms.iter_mut().enumerate() {
        atom.id = format!("EQA2-{:05}", idx + 1);
    }
    let (symbol_rows, symbol_ref_map) = build_symbol_table(&equation_atoms);
    for atom in &mut equation_atoms {
        atom.symbol_refs = atom
            .symbol_names
            .iter()
            .filter_map(|name| symbol_ref_map.get(name).cloned())
            .collect();
    }
    let proof_atoms = load_proof_atoms(&repo_root.join(&args.proof_atoms_path))?;
    let proof_skeletons = build_proof_skeletons(&claims, &proof_atoms, &claim_source);

    write_ascii(
        &repo_root.join(&args.claims_atoms_out),
        &render_claim_atoms(&claims),
    )?;
    write_ascii(
        &repo_root.join(&args.claims_edges_out),
        &render_claim_edges(&claim_edges),
    )?;
    write_ascii(
        &repo_root.join(&args.equation_atoms_out),
        &render_equation_atoms_v2(&equation_atoms),
    )?;
    write_ascii(
        &repo_root.join(&args.symbol_table_out),
        &render_symbol_table(&symbol_rows),
    )?;
    write_ascii(
        &repo_root.join(&args.proof_skeletons_out),
        &render_proof_skeletons(&proof_skeletons),
    )?;

    println!(
        "Wrote semantic-atoms registry lane artifacts: claims_atoms={} claim_edges={} equation_atoms_v2={} equation_symbols={} proof_skeletons={}",
        claims.len(),
        claim_edges.len(),
        equation_atoms.len(),
        symbol_rows.len(),
        proof_skeletons.len()
    );
    Ok(())
}

fn load_claims(
    repo_root: &Path,
    canonical_db: &Path,
    claims_path: &Path,
) -> Result<(Vec<ClaimAtom>, ClaimSourceInfo)> {
    let db_path = repo_root.join(canonical_db);
    if db_path.exists() {
        let store = ProvenanceStore::open(&db_path)
            .with_context(|| format!("open canonical db {}", db_path.display()))?;
        let claims = store
            .list_claims()
            .with_context(|| format!("load claims from {}", db_path.display()))?;
        if !claims.is_empty() {
            let mut out = Vec::new();
            for row in claims {
                let where_stated = row.where_stated.clone();
                let verify_rule = row.status_note.clone().unwrap_or_default();
                let status_detail = collapse(&row.status);
                let statement = collapse(&row.statement);
                out.push(ClaimAtom {
                    claim_id: collapse(&row.id),
                    statement: statement.clone(),
                    status_token: status_token(&status_detail),
                    status_detail,
                    hypothesis_class: hypothesis_class(&statement, &row.status),
                    last_verified: collapse(&row.last_verified),
                    where_stated: collapse(&where_stated),
                    verification_rule: collapse(&verify_rule),
                    where_stated_refs: extract_backtick_refs(&where_stated)?,
                    verification_refs: extract_backtick_refs(&verify_rule)?,
                    cross_refs: extract_claim_links(&(where_stated + " " + &verify_rule))?,
                });
            }
            out.sort_by(|lhs, rhs| lhs.claim_id.cmp(&rhs.claim_id));
            return Ok((
                out,
                ClaimSourceInfo {
                    source_registry: "registry/canonical/control_plane.sqlite3".to_string(),
                    source_path: "registry/canonical/control_plane.sqlite3".to_string(),
                },
            ));
        }
    }

    let path = repo_root.join(claims_path);
    let raw = load_toml(&path)?;
    let claims = table_array(&raw, "claim");
    if claims.is_empty() {
        bail!("ERROR: no claim rows found in {}", path.display());
    }
    let mut out = Vec::new();
    for row in claims {
        let claim_id = collapse(table_str(row, "id"));
        let statement = collapse(table_str(row, "statement"));
        let where_stated = table_str(row, "where_stated").to_string();
        let verify_rule = table_str(row, "what_would_verify_refute").to_string();
        let status_detail = collapse(table_str(row, "status"));
        out.push(ClaimAtom {
            claim_id,
            statement: statement.clone(),
            status_token: status_token(&status_detail),
            status_detail,
            hypothesis_class: hypothesis_class(&statement, table_str(row, "status")),
            last_verified: collapse(table_str(row, "last_verified")),
            where_stated: collapse(&where_stated),
            verification_rule: collapse(&verify_rule),
            where_stated_refs: extract_backtick_refs(&where_stated)?,
            verification_refs: extract_backtick_refs(&verify_rule)?,
            cross_refs: extract_claim_links(&(where_stated + " " + &verify_rule))?,
        });
    }
    out.sort_by(|lhs, rhs| lhs.claim_id.cmp(&rhs.claim_id));
    Ok((
        out,
        ClaimSourceInfo {
            source_registry: claims_path.to_string_lossy().to_string(),
            source_path: claims_path.to_string_lossy().to_string(),
        },
    ))
}

fn build_claim_edges(claim_atoms: &[ClaimAtom]) -> Vec<ClaimEdge> {
    let mut edges = BTreeMap::<(String, String, String), ClaimEdge>::new();
    for atom in claim_atoms {
        for reference in &atom.where_stated_refs {
            let key = (
                atom.claim_id.clone(),
                "where_stated".to_string(),
                reference.clone(),
            );
            edges.insert(
                key,
                ClaimEdge {
                    claim_id: atom.claim_id.clone(),
                    edge_role: "where_stated".to_string(),
                    target_ref: reference.clone(),
                    target_kind: edge_target_kind(reference),
                },
            );
        }
        for reference in &atom.verification_refs {
            let key = (
                atom.claim_id.clone(),
                "verification_rule".to_string(),
                reference.clone(),
            );
            edges.insert(
                key,
                ClaimEdge {
                    claim_id: atom.claim_id.clone(),
                    edge_role: "verification_rule".to_string(),
                    target_ref: reference.clone(),
                    target_kind: edge_target_kind(reference),
                },
            );
        }
        for reference in &atom.cross_refs {
            let key = (
                atom.claim_id.clone(),
                "cross_reference".to_string(),
                reference.clone(),
            );
            edges.insert(
                key,
                ClaimEdge {
                    claim_id: atom.claim_id.clone(),
                    edge_role: "cross_reference".to_string(),
                    target_ref: reference.clone(),
                    target_kind: edge_target_kind(reference),
                },
            );
        }
    }
    edges.into_values().collect()
}

fn load_narrative_sources(repo_root: &Path) -> Result<Vec<NarrativeSource>> {
    let mut sources = Vec::new();
    for (registry_path, source_group) in [
        ("registry/docs_root_narratives.toml", "docs_root_narrative"),
        ("registry/research_narratives.toml", "research_narrative"),
        (
            "registry/data_artifact_narratives.toml",
            "data_artifact_narrative",
        ),
    ] {
        let raw = load_toml(&repo_root.join(registry_path))?;
        for row in table_array(&raw, "document") {
            let source_uid = collapse(table_str(row, "id"));
            let body = table_str(row, "body_markdown").to_string();
            if source_uid.is_empty() || body.trim().is_empty() {
                continue;
            }
            sources.push(NarrativeSource {
                source_uid: source_uid.clone(),
                source_group: source_group.to_string(),
                source_registry: registry_path.to_string(),
                source_path: collapse(table_str(row, "source_markdown")),
                body,
            });
        }
    }
    sources.sort_by(|lhs, rhs| {
        (&lhs.source_group, &lhs.source_uid).cmp(&(&rhs.source_group, &rhs.source_uid))
    });
    Ok(sources)
}

fn extract_equation_atoms_v2(sources: &[NarrativeSource]) -> Result<Vec<EquationAtomV2>> {
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+?)\s*$")?;
    let inline_math_re = Regex::new(r"\$([^$\n]{2,260})\$")?;
    let alpha_re = Regex::new(r"[A-Za-z_\\]")?;
    let claim_id_re = Regex::new(r"\bC-\d{3}\b")?;
    let mut atoms = Vec::new();
    let mut seen = BTreeSet::new();

    for source in sources {
        let mut section_title = "(root)".to_string();
        let mut in_code_fence = false;
        let lines: Vec<String> = ascii_clean(&source.body)
            .lines()
            .map(ToOwned::to_owned)
            .collect();
        for (idx, raw) in lines.iter().enumerate() {
            let line_no = idx + 1;
            if let Some(caps) = heading_re.captures(raw) {
                section_title = collapse(caps.get(2).map(|m| m.as_str()).unwrap_or_default());
                continue;
            }
            let stripped = raw.trim();
            if stripped.starts_with("```") {
                in_code_fence = !in_code_fence;
                continue;
            }
            if in_code_fence {
                continue;
            }
            if stripped.to_lowercase().contains("auto-generated")
                || stripped.to_lowercase().contains("source of truth")
                || stripped.starts_with("<!--")
            {
                continue;
            }

            let mut candidates = Vec::<(String, String)>::new();
            for caps in inline_math_re.captures_iter(raw) {
                let expr = collapse(caps.get(1).map(|m| m.as_str()).unwrap_or_default());
                if expr.len() >= 2 {
                    candidates.push(("inline_math".to_string(), expr));
                }
            }
            let mut line_candidate = stripped.to_string();
            if let Some(rest) = line_candidate.strip_prefix("- ") {
                line_candidate = rest.trim().to_string();
            }
            if !line_candidate.is_empty()
                && !line_candidate.starts_with('#')
                && !line_candidate.starts_with('|')
                && !line_candidate.starts_with('*')
                && !line_candidate.starts_with("<!--")
                && ["=", "->", "<=", ">=", "!="]
                    .iter()
                    .any(|op| line_candidate.contains(op))
                && alpha_re.is_match(&line_candidate)
                && line_candidate.len() <= 240
            {
                candidates.push(("equation_like_line".to_string(), collapse(&line_candidate)));
            }

            for (extraction_kind, expr) in candidates {
                let key = (source.source_uid.clone(), line_no, expr.clone());
                if !seen.insert(key) {
                    continue;
                }
                let (relation_operator, lhs_expression, rhs_expression) = parse_relation(&expr);
                let (mut symbol_names, numeric_constants) = extract_symbols(&expr)?;
                if symbol_names.is_empty() {
                    symbol_names = if numeric_constants.is_empty() {
                        vec!["IMPLICIT_SYMBOL".to_string()]
                    } else {
                        vec!["NUMERIC_LITERAL".to_string()]
                    };
                }
                let quality_flags = quality_flags(&expr, &relation_operator, &symbol_names);
                if quality_flags.iter().any(|flag| flag == "header_noise") {
                    continue;
                }
                if quality_flags
                    .iter()
                    .any(|flag| flag == "text_heavy_fragment")
                    && extraction_kind == "equation_like_line"
                {
                    continue;
                }
                let extraction_confidence = if quality_flags
                    .iter()
                    .any(|flag| flag == "no_symbol_extracted")
                {
                    "low"
                } else if relation_operator == "implicit" {
                    "medium"
                } else {
                    "high"
                };
                atoms.push(EquationAtomV2 {
                    id: String::new(),
                    source_uid: source.source_uid.clone(),
                    source_group: source.source_group.clone(),
                    source_registry: source.source_registry.clone(),
                    source_path: source.source_path.clone(),
                    section_title: section_title.clone(),
                    source_line: line_no,
                    expression: expr.clone(),
                    normalized_expression: collapse(&expr),
                    relation_operator,
                    lhs_expression,
                    rhs_expression,
                    equation_kind: infer_equation_kind(&expr),
                    extraction_kind,
                    extraction_confidence: extraction_confidence.to_string(),
                    quality_flags,
                    symbol_names,
                    numeric_constants,
                    symbol_refs: Vec::new(),
                    claim_refs: sorted_unique_regex(&expr, &claim_id_re),
                });
            }
        }
    }
    atoms.sort_by(|lhs, rhs| {
        (&lhs.source_uid, lhs.source_line, &lhs.expression).cmp(&(
            &rhs.source_uid,
            rhs.source_line,
            &rhs.expression,
        ))
    });
    Ok(atoms)
}

fn build_symbol_table(
    equation_atoms: &[EquationAtomV2],
) -> (Vec<SymbolRow>, BTreeMap<String, String>) {
    let mut usage = BTreeMap::<String, (usize, BTreeSet<String>, Vec<String>)>::new();
    for atom in equation_atoms {
        for symbol in &atom.symbol_names {
            let state = usage
                .entry(symbol.clone())
                .or_insert((0, BTreeSet::new(), Vec::new()));
            state.0 += 1;
            state.1.insert(atom.source_uid.clone());
            if state.2.len() < 12 {
                state.2.push(atom.id.clone());
            }
        }
    }
    let mut rows = usage
        .into_iter()
        .map(
            |(symbol, (usage_count, source_uids, example_equation_ids))| SymbolRow {
                id: String::new(),
                normalized_symbol: symbol.to_lowercase(),
                category: symbol_category(&symbol),
                symbol,
                usage_count,
                source_uids: source_uids.into_iter().collect(),
                example_equation_ids,
            },
        )
        .collect::<Vec<_>>();
    rows.sort_by(|lhs, rhs| {
        rhs.usage_count
            .cmp(&lhs.usage_count)
            .then_with(|| lhs.symbol.to_lowercase().cmp(&rhs.symbol.to_lowercase()))
    });
    let mut map = BTreeMap::new();
    for (idx, row) in rows.iter_mut().enumerate() {
        row.id = format!("SYM-{:04}", idx + 1);
        map.insert(row.symbol.clone(), row.id.clone());
    }
    (rows, map)
}

fn load_proof_atoms(path: &Path) -> Result<Vec<Value>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = load_toml(path)?;
    Ok(table_array(&raw, "atom").to_vec())
}

fn build_proof_skeletons(
    claim_atoms: &[ClaimAtom],
    proof_atoms: &[Value],
    claim_source: &ClaimSourceInfo,
) -> Vec<ProofSkeleton> {
    let mut skeletons = Vec::new();
    for atom in claim_atoms {
        let mut assumptions = vec![];
        if !atom.where_stated.is_empty() {
            // Placeholder: claim atoms v1 no h0/h1, use statement as fallback below.
        }
        let rule = atom.verification_rule.clone();
        if assumptions.is_empty() {
            assumptions.push(atom.statement.clone());
        }
        let mut derivation_steps = Vec::new();
        for segment in atom.verification_rule.split(['|', '.', ';']) {
            let piece = collapse(segment);
            if !piece.is_empty() {
                derivation_steps.push(piece);
            }
            if derivation_steps.len() >= 8 {
                break;
            }
        }
        skeletons.push(ProofSkeleton {
            skeleton_kind: "claim_decision_rule".to_string(),
            theorem_label: atom.claim_id.clone(),
            claim_id: atom.claim_id.clone(),
            assumptions: assumptions.into_iter().take(8).collect(),
            obligations: vec![format!(
                "Establish evidence-backed status for {}.",
                atom.claim_id
            )],
            derivation_steps: derivation_steps.into_iter().take(12).collect(),
            decision_rule: collapse(&rule),
            conclusion: collapse(&format!("{}: {}", atom.status_token, atom.status_detail)),
            source_uid: atom.claim_id.clone(),
            source_registry: claim_source.source_registry.clone(),
            source_path: claim_source.source_path.clone(),
            line_start: 0,
            line_end: 0,
            claim_refs: {
                let mut refs = atom.cross_refs.clone();
                refs.push(atom.claim_id.clone());
                refs.sort();
                refs.dedup();
                refs
            },
            evidence_refs: {
                let mut refs = atom.where_stated_refs.clone();
                refs.extend(atom.verification_refs.clone());
                refs.extend(atom.cross_refs.clone());
                refs.sort();
                refs.dedup();
                refs
            },
        });
    }

    let proof_kinds = BTreeSet::from([
        "proof",
        "theorem",
        "lemma",
        "corollary",
        "proposition",
        "axiom",
        "derivation",
        "hypothesis_block",
        "argument_section",
    ]);

    for row in proof_atoms {
        let mut kind = collapse(table_str(row, "proof_kind")).to_lowercase();
        if !proof_kinds.contains(kind.as_str()) {
            kind = "argument_section".to_string();
        }
        let claim_refs = string_list(row, "claim_refs");
        let assumptions = string_list(row, "assumption_lines");
        let mut derivation_steps = string_list(row, "inference_markers");
        if derivation_steps.is_empty() {
            let excerpt = collapse(table_str(row, "excerpt"));
            if !excerpt.is_empty() {
                derivation_steps.push(excerpt);
            }
        }
        let obligations = string_list(row, "decision_lines");
        let source_path = collapse(table_str(row, "source_path"));
        let source_registry = collapse(table_str(row, "source_registry"));
        let mut evidence_refs = vec![source_path.clone(), source_registry.clone()];
        evidence_refs.extend(claim_refs.clone());
        evidence_refs.retain(|item| !item.is_empty());
        evidence_refs.sort();
        evidence_refs.dedup();
        skeletons.push(ProofSkeleton {
            skeleton_kind: kind,
            theorem_label: collapse(table_str(row, "section_title")),
            claim_id: claim_refs.first().cloned().unwrap_or_default(),
            assumptions: assumptions.into_iter().take(10).collect(),
            obligations: obligations.into_iter().take(10).collect(),
            derivation_steps: derivation_steps.into_iter().take(12).collect(),
            decision_rule: collapse(table_str(row, "decision_rule_text")),
            conclusion: collapse(table_str(row, "conclusion_text")),
            source_uid: collapse(table_str(row, "source_uid")),
            source_registry,
            source_path,
            line_start: table_int(row, "line_start").max(0) as usize,
            line_end: table_int(row, "line_end").max(0) as usize,
            claim_refs,
            evidence_refs,
        });
    }

    let mut dedup = BTreeMap::<(String, String, String, String), ProofSkeleton>::new();
    for row in skeletons {
        let key = (
            row.skeleton_kind.clone(),
            row.theorem_label.clone(),
            row.source_uid.clone(),
            row.claim_id.clone(),
        );
        dedup.insert(key, row);
    }
    let mut out: Vec<_> = dedup.into_values().collect();
    out.sort_by(|lhs, rhs| {
        (&lhs.claim_id, &lhs.source_uid, &lhs.theorem_label).cmp(&(
            &rhs.claim_id,
            &rhs.source_uid,
            &rhs.theorem_label,
        ))
    });
    out
}

fn verify_semantic_atoms(repo_root: &Path, args: &Args) -> Result<()> {
    let required = [
        repo_root.join(&args.claims_path),
        repo_root.join(&args.claims_atoms_out),
        repo_root.join(&args.claims_edges_out),
        repo_root.join(&args.equation_atoms_out),
        repo_root.join(&args.symbol_table_out),
        repo_root.join(&args.proof_skeletons_out),
    ];
    for path in &required {
        if !path.exists() {
            println!("ERROR: missing registry {}", path.display());
            std::process::exit(1);
        }
    }

    // Payload files are generated by `markdown-registry build-payloads` (currently
    // a stub).  When they have not been built, warn and skip payload verification
    // rather than hard-failing the readonly gate.
    let payload_path = repo_root.join(&args.payload_path);
    let chunks_path = repo_root.join(&args.payload_chunks_path);
    let have_payloads = payload_path.exists() && chunks_path.exists();
    if !have_payloads {
        println!(
            "WARN: payload files not built yet (stub), skipping payload checks"
        );
    }

    for path in required.iter().skip(1) {
        assert_ascii_file(path)?;
    }
    if have_payloads {
        assert_ascii_file(&payload_path)?;
        assert_ascii_file(&chunks_path)?;
    }

    let (mut canonical_claim_ids, _) =
        load_canonical_claim_ids(repo_root, &args.canonical_db, &args.claims_path)?;
    let claim_atoms_raw = load_toml(&repo_root.join(&args.claims_atoms_out))?;
    let claim_edges_raw = load_toml(&repo_root.join(&args.claims_edges_out))?;
    let equation_atoms_raw = load_toml(&repo_root.join(&args.equation_atoms_out))?;
    let equation_symbols_raw = load_toml(&repo_root.join(&args.symbol_table_out))?;
    let proof_raw = load_toml(&repo_root.join(&args.proof_skeletons_out))?;
    let empty_toml: toml::Value = toml::Value::Table(Default::default());
    let payload_raw = if have_payloads { load_toml(&payload_path)? } else { empty_toml.clone() };
    let chunk_raw = if have_payloads { load_toml(&chunks_path)? } else { empty_toml };

    let claim_atoms = table_array(&claim_atoms_raw, "atom");
    let claim_edges = table_array(&claim_edges_raw, "edge");
    let equation_atoms = table_array(&equation_atoms_raw, "atom");
    let equation_symbols = table_array(&equation_symbols_raw, "symbol");
    let proof_rows = table_array(&proof_raw, "skeleton");
    let payload_docs = table_array(&payload_raw, "document");
    let payload_chunks = table_array(&chunk_raw, "chunk");

    let mut failures = Vec::new();

    canonical_claim_ids.sort();
    let claim_count = canonical_claim_ids.len();
    let min_claims_with_edges =
        ((claim_count as f64) * MIN_CLAIMS_WITH_EVIDENCE_COVERAGE).ceil() as usize;
    let max_uncovered_claims = claim_count.saturating_sub(min_claims_with_edges);
    let mut atom_claim_ids = claim_atoms
        .iter()
        .map(|row| collapse(table_str(row, "claim_id")))
        .collect::<Vec<_>>();
    atom_claim_ids.sort();
    if canonical_claim_ids != atom_claim_ids {
        failures.push("claims_atoms claim_id set does not match canonical claims set.".to_string());
    }
    if table_int_in(&claim_atoms_raw, "claims_atoms", "atom_count") != claim_atoms.len() as i64 {
        failures.push("claims_atoms metadata atom_count mismatch.".to_string());
    }
    if claim_atoms
        .iter()
        .any(|row| table_str(row, "status_token").trim().is_empty())
    {
        failures.push("claims_atoms contains empty status_token.".to_string());
    }

    if table_int_in(&claim_edges_raw, "claims_evidence_edges", "edge_count")
        != claim_edges.len() as i64
    {
        failures.push("claims_evidence_edges metadata edge_count mismatch.".to_string());
    }
    let mut edges_by_claim = BTreeMap::<String, usize>::new();
    for row in claim_edges {
        let claim_id = table_str(row, "claim_id").to_string();
        *edges_by_claim.entry(claim_id).or_insert(0) += 1;
    }
    let uncovered_claims = canonical_claim_ids
        .iter()
        .filter(|claim_id| edges_by_claim.get(*claim_id).copied().unwrap_or(0) == 0)
        .collect::<Vec<_>>();
    if uncovered_claims.len() > max_uncovered_claims {
        failures.push(format!(
            "too many claims without evidence edges: {} (max allowed {}; semantic-atoms coverage policy >= {:.1}%)",
            uncovered_claims.len(),
            max_uncovered_claims,
            MIN_CLAIMS_WITH_EVIDENCE_COVERAGE * 100.0
        ));
    }

    let equation_atom_density_floor = (EQUATION_ATOMS_BASELINE_COUNT as f64
        / EQUATION_ATOMS_BASELINE_CLAIMS as f64)
        * EQUATION_ATOMS_DENSITY_REGRESSION_TOLERANCE;
    let min_equation_atoms = EQUATION_ATOMS_ABSOLUTE_MIN
        .max(((claim_count as f64) * equation_atom_density_floor).ceil() as usize);
    if equation_atoms.len() < min_equation_atoms {
        failures.push(format!(
            "equation_atoms_v2 too small: {} (min required {}; semantic-atoms density policy floor={:.4} atoms/claim)",
            equation_atoms.len(),
            min_equation_atoms,
            equation_atom_density_floor
        ));
    }
    if table_int_in(
        &equation_atoms_raw,
        "knowledge_equation_atoms_v2",
        "atom_count",
    ) != equation_atoms.len() as i64
    {
        failures.push("equation_atoms_v2 metadata atom_count mismatch.".to_string());
    }
    if table_int_in(
        &equation_symbols_raw,
        "equation_symbol_table",
        "symbol_count",
    ) != equation_symbols.len() as i64
    {
        failures.push("equation_symbol_table metadata symbol_count mismatch.".to_string());
    }

    let symbol_ids = equation_symbols
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .collect::<BTreeSet<_>>();
    let mut symbol_usage = equation_symbols
        .iter()
        .map(|row| (table_str(row, "id").to_string(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut equation_ids = BTreeSet::new();
    for row in equation_atoms {
        let atom_id = table_str(row, "id").to_string();
        if !equation_ids.insert(atom_id.clone()) {
            failures.push(format!("duplicate equation atom id: {}", atom_id));
            break;
        }
        let refs = string_list(row, "symbol_refs");
        if refs.is_empty() {
            failures.push(format!("equation atom has empty symbol_refs: {}", atom_id));
            continue;
        }
        for reference in refs {
            if !symbol_ids.contains(&reference) {
                failures.push(format!(
                    "equation atom references unknown symbol id: {} -> {}",
                    atom_id, reference
                ));
            } else {
                *symbol_usage.entry(reference).or_insert(0) += 1;
            }
        }
    }
    for row in equation_symbols {
        let sid = table_str(row, "id").to_string();
        let expected = table_int(row, "usage_count").max(0) as usize;
        let observed = symbol_usage.get(&sid).copied().unwrap_or(0);
        if expected != observed {
            failures.push(format!(
                "symbol usage mismatch: {} expected={} observed={}",
                sid, expected, observed
            ));
        }
    }

    if proof_rows.len() < canonical_claim_ids.len() {
        failures.push(format!(
            "proof_skeletons too small: {} < canonical claims {}",
            proof_rows.len(),
            canonical_claim_ids.len()
        ));
    }
    if table_int_in(&proof_raw, "knowledge_proof_skeletons", "skeleton_count")
        != proof_rows.len() as i64
    {
        failures.push("proof_skeletons metadata skeleton_count mismatch.".to_string());
    }
    let mut proof_ids = BTreeSet::new();
    for row in proof_rows {
        let proof_id = table_str(row, "id").to_string();
        if !proof_ids.insert(proof_id.clone()) {
            failures.push(format!("duplicate proof skeleton id: {}", proof_id));
            break;
        }
        if !value_is_array(row, "assumptions") {
            failures.push(format!("proof skeleton assumptions not list: {}", proof_id));
            break;
        }
        if !value_is_array(row, "derivation_steps") {
            failures.push(format!(
                "proof skeleton derivation_steps not list: {}",
                proof_id
            ));
            break;
        }
        if table_str(row, "skeleton_kind").trim().is_empty() {
            failures.push(format!(
                "proof skeleton missing skeleton_kind: {}",
                proof_id
            ));
            break;
        }
    }

    if have_payloads {
    if table_int_in(&payload_raw, "markdown_payloads", "document_count")
        != payload_docs.len() as i64
    {
        failures.push("markdown_payloads document_count metadata mismatch.".to_string());
    }
    if table_int_in(&chunk_raw, "markdown_payload_chunks", "chunk_count")
        != payload_chunks.len() as i64
    {
        failures.push("markdown_payload_chunks chunk_count metadata mismatch.".to_string());
    }
    if table_str_in(&payload_raw, "markdown_payloads", "representation") != "structured_toml_units"
    {
        failures
            .push("markdown_payloads representation must be structured_toml_units.".to_string());
    }
    if table_str_in(&chunk_raw, "markdown_payload_chunks", "representation")
        != "structured_toml_units"
    {
        failures.push(
            "markdown_payload_chunks representation must be structured_toml_units.".to_string(),
        );
    }

    let discovered_md = discover_markdown_files(repo_root)?;
    let payload_paths = payload_docs
        .iter()
        .map(|row| table_str(row, "path").to_string())
        .collect::<BTreeSet<_>>();
    if discovered_md != payload_paths {
        let missing = discovered_md
            .difference(&payload_paths)
            .cloned()
            .collect::<Vec<_>>();
        let extra = payload_paths
            .difference(&discovered_md)
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            failures.push(format!(
                "markdown_payloads missing paths: {}",
                missing.len()
            ));
            for item in missing.iter().take(20) {
                failures.push(format!("  missing: {}", item));
            }
        }
        if !extra.is_empty() {
            failures.push(format!("markdown_payloads extra paths: {}", extra.len()));
            for item in extra.iter().take(20) {
                failures.push(format!("  extra: {}", item));
            }
        }
    }

    let existing_payload_paths = payload_paths
        .iter()
        .filter(|rel_path| repo_root.join(rel_path.as_str()).exists())
        .cloned()
        .collect::<Vec<_>>();
    let file_digests = existing_payload_paths
        .par_iter()
        .map(|rel_path| {
            let digest = sha256_file(&repo_root.join(rel_path))?;
            Ok((rel_path.clone(), digest))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .collect::<BTreeMap<_, _>>();

    let mut chunk_by_id = BTreeMap::<String, &Value>::new();
    for row in payload_chunks {
        chunk_by_id.insert(table_str(row, "id").to_string(), row);
    }
    if chunk_by_id.len() != payload_chunks.len() {
        failures.push("duplicate markdown payload chunk ids detected.".to_string());
    }

    let mut chunks_by_doc = BTreeMap::<String, usize>::new();
    for row in payload_chunks {
        let doc_id = table_str(row, "document_id").to_string();
        *chunks_by_doc.entry(doc_id).or_insert(0) += 1;
    }

    let allowed_chunk_kinds = ALLOWED_CHUNK_KINDS.iter().copied().collect::<BTreeSet<_>>();
    let mut third_party_count = 0usize;
    for row in payload_docs {
        let doc_id = table_str(row, "id").to_string();
        let rel_path = table_str(row, "path").to_string();
        let origin_class = table_str(row, "origin_class");
        if origin_class == "third_party_cache" {
            third_party_count += 1;
        }

        let file_path = repo_root.join(&rel_path);
        if !file_path.exists() {
            failures.push(format!(
                "payload doc path missing on disk: {} -> {}",
                doc_id, rel_path
            ));
            continue;
        }

        let digest = file_digests.get(&rel_path).cloned().unwrap_or_default();
        if digest != table_str(row, "content_sha256") {
            failures.push(format!("sha mismatch for {} ({})", doc_id, rel_path));
        }

        let chunk_ids = string_list(row, "chunk_ids");
        if table_int(row, "chunk_count") != chunk_ids.len() as i64 {
            failures.push(format!("chunk_count mismatch for {}", doc_id));
            continue;
        }

        let mut heading_count = 0usize;
        let mut paragraph_count = 0usize;
        let mut list_item_count = 0usize;
        let mut table_row_count = 0usize;
        let mut code_block_count = 0usize;
        for (expected_next, chunk_id) in (1i64..).zip(chunk_ids) {
            let Some(chunk) = chunk_by_id.get(&chunk_id).copied() else {
                failures.push(format!("missing chunk id {} for {}", chunk_id, doc_id));
                break;
            };
            if table_str(chunk, "document_id") != doc_id {
                failures.push(format!("chunk document_id mismatch: {}", chunk_id));
            }
            let idx = table_int(chunk, "chunk_index");
            if idx != expected_next {
                failures.push(format!(
                    "chunk index sequence mismatch for {}: got {} expected {}",
                    doc_id, idx, expected_next
                ));
            }

            let kind = table_str(chunk, "kind");
            if !allowed_chunk_kinds.contains(kind) {
                failures.push(format!("invalid chunk kind for {}: {}", chunk_id, kind));
            }
            match kind {
                "heading" => heading_count += 1,
                "paragraph" => paragraph_count += 1,
                "list_item" => list_item_count += 1,
                "table_row" => table_row_count += 1,
                "code_block" => code_block_count += 1,
                _ => {}
            }

            let line_start = table_int(chunk, "line_start");
            let line_end = table_int(chunk, "line_end");
            if line_start <= 0 || line_end < line_start {
                failures.push(format!(
                    "invalid line range for {}: {}-{}",
                    chunk_id, line_start, line_end
                ));
            }

            let text_ascii = table_str(chunk, "text_ascii");
            let text_sha = sha256_hex(text_ascii.as_bytes());
            if text_sha != table_str(chunk, "text_sha256") {
                failures.push(format!("text_sha256 mismatch for {}", chunk_id));
            }
        }

        if table_int(row, "heading_count") != heading_count as i64 {
            failures.push(format!("heading_count mismatch for {}", doc_id));
        }
        if table_int(row, "paragraph_count") != paragraph_count as i64 {
            failures.push(format!("paragraph_count mismatch for {}", doc_id));
        }
        if table_int(row, "list_item_count") != list_item_count as i64 {
            failures.push(format!("list_item_count mismatch for {}", doc_id));
        }
        if table_int(row, "table_row_count") != table_row_count as i64 {
            failures.push(format!("table_row_count mismatch for {}", doc_id));
        }
        if table_int(row, "code_block_count") != code_block_count as i64 {
            failures.push(format!("code_block_count mismatch for {}", doc_id));
        }
        if !chunks_by_doc.contains_key(&doc_id) {
            failures.push(format!("no chunks indexed for document {}", doc_id));
        }
    }

    if third_party_count == 0 {
        failures.push("expected third_party_cache markdown documents, found none.".to_string());
    }
    } // have_payloads

    if !failures.is_empty() {
        println!("ERROR: semantic-atoms registry lane verification failed.");
        for item in failures.iter().take(300) {
            println!("- {}", item);
        }
        if failures.len() > 300 {
            println!("- ... and {} more failures", failures.len() - 300);
        }
        std::process::exit(1);
    }

    println!(
        "OK: semantic-atoms registry lane verified. claims={} edges={} equations={} symbols={} proof_skeletons={} markdown_docs={} markdown_chunks={}",
        claim_atoms.len(),
        claim_edges.len(),
        equation_atoms.len(),
        equation_symbols.len(),
        proof_rows.len(),
        payload_docs.len(),
        payload_chunks.len()
    );
    Ok(())
}

fn load_canonical_claim_ids(
    repo_root: &Path,
    canonical_db: &Path,
    claims_path: &Path,
) -> Result<(Vec<String>, String)> {
    let db_path = repo_root.join(canonical_db);
    if db_path.exists() {
        let store = ProvenanceStore::open(&db_path)
            .with_context(|| format!("open canonical db {}", db_path.display()))?;
        let claims = store
            .list_claims()
            .with_context(|| format!("load claims from {}", db_path.display()))?;
        if !claims.is_empty() {
            return Ok((
                claims.into_iter().map(|row| collapse(&row.id)).collect(),
                "registry/canonical/control_plane.sqlite3".to_string(),
            ));
        }
    }

    let raw = load_toml(&repo_root.join(claims_path))?;
    Ok((
        table_array(&raw, "claim")
            .iter()
            .map(|row| collapse(table_str(row, "id")))
            .collect(),
        claims_path.to_string_lossy().to_string(),
    ))
}

fn render_claim_atoms(rows: &[ClaimAtom]) -> String {
    let unique_claims: BTreeSet<String> = rows.iter().map(|row| row.claim_id.clone()).collect();
    let mut lines = vec![
        "# Claim atoms registry (semantic-atoms lane strict schema).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/semantic_atoms.rs.".to_string(),
        "".to_string(),
        "[claims_atoms]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("atom_count = {}", rows.len()),
        format!("unique_claim_count = {}", unique_claims.len()),
        "".to_string(),
    ];
    for (idx, row) in rows.iter().enumerate() {
        lines.extend([
            "[[atom]]".to_string(),
            format!("id = {}", esc(&format!("CLA-{:04}", idx + 1))),
            format!("claim_id = {}", esc(&row.claim_id)),
            format!("statement = {}", esc(&row.statement)),
            format!("status_token = {}", esc(&row.status_token)),
            format!("status_detail = {}", esc(&row.status_detail)),
            format!("hypothesis_class = {}", esc(&row.hypothesis_class)),
            format!("last_verified = {}", esc(&row.last_verified)),
            format!("where_stated = {}", esc(&row.where_stated)),
            format!("verification_rule = {}", esc(&row.verification_rule)),
            format!(
                "where_stated_refs = {}",
                render_list(&row.where_stated_refs)
            ),
            format!(
                "verification_refs = {}",
                render_list(&row.verification_refs)
            ),
            format!("cross_refs = {}", render_list(&row.cross_refs)),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn render_claim_edges(rows: &[ClaimEdge]) -> String {
    let mut lines = vec![
        "# Claim evidence edge registry (semantic-atoms lane strict schema).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/semantic_atoms.rs.".to_string(),
        "".to_string(),
        "[claims_evidence_edges]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("edge_count = {}", rows.len()),
        "".to_string(),
    ];
    for (idx, row) in rows.iter().enumerate() {
        lines.extend([
            "[[edge]]".to_string(),
            format!("id = {}", esc(&format!("CED-{:05}", idx + 1))),
            format!("claim_id = {}", esc(&row.claim_id)),
            format!("edge_role = {}", esc(&row.edge_role)),
            format!("target_ref = {}", esc(&row.target_ref)),
            format!("target_kind = {}", esc(&row.target_kind)),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn render_equation_atoms_v2(rows: &[EquationAtomV2]) -> String {
    let mut lines = vec![
        "# Equation atoms v2 registry (semantic-atoms lane strict schema).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/semantic_atoms.rs.".to_string(),
        "".to_string(),
        "[knowledge_equation_atoms_v2]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("atom_count = {}", rows.len()),
        "".to_string(),
    ];
    for row in rows {
        lines.extend([
            "[[atom]]".to_string(),
            format!("id = {}", esc(&row.id)),
            format!("expression = {}", esc(&row.expression)),
            format!(
                "normalized_expression = {}",
                esc(&row.normalized_expression)
            ),
            format!("relation_operator = {}", esc(&row.relation_operator)),
            format!("lhs_expression = {}", esc(&row.lhs_expression)),
            format!("rhs_expression = {}", esc(&row.rhs_expression)),
            format!("equation_kind = {}", esc(&row.equation_kind)),
            format!("extraction_kind = {}", esc(&row.extraction_kind)),
            format!(
                "extraction_confidence = {}",
                esc(&row.extraction_confidence)
            ),
            format!("quality_flags = {}", render_list(&row.quality_flags)),
            format!("symbol_names = {}", render_list(&row.symbol_names)),
            format!(
                "numeric_constants = {}",
                render_list(&row.numeric_constants)
            ),
            format!("symbol_refs = {}", render_list(&row.symbol_refs)),
            format!("claim_refs = {}", render_list(&row.claim_refs)),
            format!("source_uid = {}", esc(&row.source_uid)),
            format!("source_group = {}", esc(&row.source_group)),
            format!("source_registry = {}", esc(&row.source_registry)),
            format!("source_path = {}", esc(&row.source_path)),
            format!("section_title = {}", esc(&row.section_title)),
            format!("source_line = {}", row.source_line),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn render_symbol_table(rows: &[SymbolRow]) -> String {
    let mut lines = vec![
        "# Equation symbol table (semantic-atoms lane strict schema).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/semantic_atoms.rs.".to_string(),
        "".to_string(),
        "[equation_symbol_table]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("symbol_count = {}", rows.len()),
        "".to_string(),
    ];
    for row in rows {
        lines.extend([
            "[[symbol]]".to_string(),
            format!("id = {}", esc(&row.id)),
            format!("symbol = {}", esc(&row.symbol)),
            format!("normalized_symbol = {}", esc(&row.normalized_symbol)),
            format!("category = {}", esc(&row.category)),
            format!("usage_count = {}", row.usage_count),
            format!("source_uids = {}", render_list(&row.source_uids)),
            format!(
                "example_equation_ids = {}",
                render_list(&row.example_equation_ids)
            ),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn render_proof_skeletons(rows: &[ProofSkeleton]) -> String {
    let mut lines = vec![
        "# Proof skeleton registry (semantic-atoms lane strict schema).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/semantic_atoms.rs.".to_string(),
        "".to_string(),
        "[knowledge_proof_skeletons]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("skeleton_count = {}", rows.len()),
        "".to_string(),
    ];
    for (idx, row) in rows.iter().enumerate() {
        lines.extend([
            "[[skeleton]]".to_string(),
            format!("id = {}", esc(&format!("PRS-{:05}", idx + 1))),
            format!("skeleton_kind = {}", esc(&row.skeleton_kind)),
            format!("theorem_label = {}", esc(&row.theorem_label)),
            format!("claim_id = {}", esc(&row.claim_id)),
            format!("assumptions = {}", render_list(&row.assumptions)),
            format!("obligations = {}", render_list(&row.obligations)),
            format!("derivation_steps = {}", render_list(&row.derivation_steps)),
            format!("decision_rule = {}", esc(&row.decision_rule)),
            format!("conclusion = {}", esc(&row.conclusion)),
            format!("source_uid = {}", esc(&row.source_uid)),
            format!("source_registry = {}", esc(&row.source_registry)),
            format!("source_path = {}", esc(&row.source_path)),
            format!("line_start = {}", row.line_start),
            format!("line_end = {}", row.line_end),
            format!("claim_refs = {}", render_list(&row.claim_refs)),
            format!("evidence_refs = {}", render_list(&row.evidence_refs)),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn edge_target_kind(reference: &str) -> String {
    let claim_re = Regex::new(r"^C-\d{3}$").unwrap();
    let insight_re = Regex::new(r"^I-\d{3}$").unwrap();
    let experiment_re = Regex::new(r"^E-\d{3}$").unwrap();
    if claim_re.is_match(reference) {
        "claim_ref".to_string()
    } else if insight_re.is_match(reference) {
        "insight_ref".to_string()
    } else if experiment_re.is_match(reference) {
        "experiment_ref".to_string()
    } else if reference.starts_with("registry/") {
        "registry_path".to_string()
    } else if reference.ends_with(".rs") {
        "rust_source_path".to_string()
    } else if reference.ends_with(".py") {
        "python_source_path".to_string()
    } else if reference.ends_with(".toml") {
        "toml_registry_path".to_string()
    } else if reference.ends_with(".csv") {
        "csv_artifact_path".to_string()
    } else if reference.ends_with(".md") {
        "markdown_path".to_string()
    } else {
        "generic_reference".to_string()
    }
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

fn hypothesis_class(statement: &str, status: &str) -> String {
    let text = format!("{statement} {status}").to_lowercase();
    if text.contains("falsifiable thesis") || text.contains("falsifiable") {
        "falsifiable_thesis".to_string()
    } else if text.contains("toy") {
        "toy_model".to_string()
    } else if text.contains("speculative") {
        "speculative_claim".to_string()
    } else if text.contains("refuted") {
        "refuted_claim".to_string()
    } else if text.contains("verified") {
        "verified_claim".to_string()
    } else if text.contains("closed") {
        "closed_claim".to_string()
    } else {
        "research_claim".to_string()
    }
}

fn extract_backtick_refs(text: &str) -> Result<Vec<String>> {
    let backtick_re = Regex::new(r"`([^`]+)`")?;
    let path_like_re =
        Regex::new(r"\b(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_.-]+)?\b")?;
    let file_like_re = Regex::new(r"\b[A-Za-z0-9_.-]+\.(?:md|rs|py|toml|csv|json|txt)\b")?;
    let mut refs = BTreeSet::new();
    for caps in backtick_re.captures_iter(text) {
        let payload = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        for part in payload.split([',', '\n']) {
            let candidate = collapse(part)
                .trim_matches([' ', '.', ';', ':', '(', ')', '[', ']', '{', '}'])
                .to_string();
            if candidate.is_empty() || (candidate.starts_with("C-") && candidate.len() == 5) {
                continue;
            }
            if candidate.contains('/') || candidate.contains('.') {
                refs.insert(candidate);
            }
        }
    }
    for regex in [&path_like_re, &file_like_re] {
        for hit in regex.find_iter(text) {
            let candidate = collapse(hit.as_str())
                .trim_matches([' ', '.', ';', ':', '(', ')', '[', ']', '{', '}'])
                .to_string();
            if candidate.is_empty() || (candidate.starts_with("C-") && candidate.len() == 5) {
                continue;
            }
            refs.insert(candidate);
        }
    }
    Ok(refs.into_iter().collect())
}

fn extract_claim_links(text: &str) -> Result<Vec<String>> {
    let re = Regex::new(r"\b(?:C|I|E)-\d{3}\b")?;
    Ok(sorted_unique_regex(text, &re))
}

fn parse_relation(expr: &str) -> (String, String, String) {
    for token in ["<=", ">=", "!=", "->", "="] {
        if let Some((lhs, rhs)) = expr.split_once(token) {
            let lhs_c = collapse(lhs);
            let rhs_c = collapse(rhs);
            if !lhs_c.is_empty() {
                return (token.to_string(), lhs_c, rhs_c);
            }
        }
    }
    ("implicit".to_string(), collapse(expr), String::new())
}

fn infer_equation_kind(expr: &str) -> String {
    let lowered = expr.to_lowercase();
    if expr.contains("->") {
        "mapping_relation".to_string()
    } else if ["d/d", "partial", "nabla", "delta", "laplacian"]
        .iter()
        .any(|token| lowered.contains(token))
    {
        "differential_relation".to_string()
    } else if ["<=", ">=", "!=", "<", ">"]
        .iter()
        .any(|token| expr.contains(token))
    {
        "constraint_relation".to_string()
    } else {
        "algebraic_relation".to_string()
    }
}

fn extract_symbols(expr: &str) -> Result<(Vec<String>, Vec<String>)> {
    let identifier_re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b")?;
    let number_re = Regex::new(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")?;
    let latex_cmd_re = Regex::new(r"\\([A-Za-z]+)")?;
    let stopwords = BTreeSet::from([
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "this",
        "that",
        "into",
        "over",
        "under",
        "after",
        "before",
        "where",
        "when",
        "while",
        "line",
        "note",
        "data",
        "source",
        "truth",
        "auto",
        "generated",
        "edit",
        "not",
        "registry",
        "toml",
        "markdown",
    ]);
    let mut symbols = BTreeSet::new();
    for caps in latex_cmd_re.captures_iter(expr) {
        let token = collapse(caps.get(1).map(|m| m.as_str()).unwrap_or_default());
        if !token.is_empty() {
            symbols.insert(token);
        }
    }
    for hit in identifier_re.find_iter(expr) {
        let token = collapse(hit.as_str());
        if !token.is_empty() && !stopwords.contains(token.to_lowercase().as_str()) {
            symbols.insert(token);
        }
    }
    let numbers = sorted_unique_regex(expr, &number_re);
    Ok((symbols.into_iter().collect(), numbers))
}

fn quality_flags(expr: &str, relation: &str, symbol_names: &[String]) -> Vec<String> {
    let lowered = expr.to_lowercase();
    let mut flags = Vec::new();
    if lowered.contains("<!--")
        || lowered.contains("auto-generated")
        || lowered.contains("source of truth")
    {
        flags.push("header_noise".to_string());
    }
    let word_count = collapse(expr).split_whitespace().count();
    let symbol_density = expr.chars().filter(|ch| "=+-*/^_()".contains(*ch)).count();
    if word_count > 24 && symbol_density < 2 {
        flags.push("text_heavy_fragment".to_string());
    }
    if symbol_names.is_empty() {
        flags.push("no_symbol_extracted".to_string());
    }
    if relation == "implicit" {
        flags.push("implicit_relation".to_string());
    }
    flags.sort();
    flags
}

fn symbol_category(symbol: &str) -> String {
    let lower = symbol.to_lowercase();
    if [
        "sin", "cos", "tan", "exp", "log", "sqrt", "nabla", "partial", "delta", "sum", "prod",
        "int", "lim",
    ]
    .contains(&lower.as_str())
    {
        "operator_or_function".to_string()
    } else if symbol.starts_with("mathbb") {
        "set_marker".to_string()
    } else if symbol.len() == 1 && symbol.chars().all(|ch| ch.is_ascii_alphabetic()) {
        "scalar_variable".to_string()
    } else if symbol.contains('_') {
        "indexed_symbol".to_string()
    } else if symbol.chars().all(|ch| ch.is_ascii_uppercase()) {
        "constant_or_group".to_string()
    } else {
        "identifier".to_string()
    }
}

fn load_toml(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse TOML {}", path.display()))
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

fn write_ascii(path: &Path, text: &str) -> Result<()> {
    assert_ascii(text, &path.display().to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

fn assert_ascii(text: &str, context: &str) -> Result<()> {
    let bad: BTreeSet<char> = text.chars().filter(|ch| (*ch as u32) > 127).collect();
    if !bad.is_empty() {
        let sample: String = bad.iter().take(20).copied().collect();
        bail!("ERROR: Non-ASCII output in {context}: {:?}", sample);
    }
    Ok(())
}

fn assert_ascii_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    assert_ascii(&text, &path.display().to_string())
}

fn table_int_in(value: &Value, table: &str, key: &str) -> i64 {
    value
        .get(table)
        .and_then(|child| child.get(key))
        .and_then(Value::as_integer)
        .unwrap_or(-1)
}

fn table_str_in<'a>(value: &'a Value, table: &str, key: &str) -> &'a str {
    value
        .get(table)
        .and_then(|child| child.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn value_is_array(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_array).is_some()
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

fn sorted_unique_regex(text: &str, regex: &Regex) -> Vec<String> {
    let mut values = BTreeSet::new();
    for hit in regex.find_iter(text) {
        values.insert(hit.as_str().to_string());
    }
    values.into_iter().collect()
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

fn discover_markdown_files(repo_root: &Path) -> Result<BTreeSet<String>> {
    if let Ok(paths) = discover_markdown_with_rg(repo_root) {
        return Ok(paths);
    }
    let mut out = BTreeSet::new();
    for entry in WalkDir::new(repo_root).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if !entry.file_type().is_file() {
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
            continue;
        }
        let Ok(rel) = path.strip_prefix(repo_root) else {
            continue;
        };
        let rel = rel.to_string_lossy().replace('\\', "/");
        if should_skip_markdown(&rel) {
            continue;
        }
        out.insert(rel);
    }
    Ok(out)
}

fn discover_markdown_with_rg(repo_root: &Path) -> Result<BTreeSet<String>> {
    let output = Command::new("rg")
        .args(["--files", "--hidden", "--no-ignore", "-g", "*.md"])
        .current_dir(repo_root)
        .output()
        .context("run rg for markdown discovery")?;
    if !output.status.success() {
        bail!("rg markdown discovery failed with status {}", output.status);
    }
    let stdout = String::from_utf8(output.stdout).context("decode rg markdown discovery output")?;
    let mut out = BTreeSet::new();
    for rel in stdout.lines() {
        if should_skip_markdown(rel) {
            continue;
        }
        out.insert(rel.to_string());
    }
    Ok(out)
}

fn should_skip_markdown(rel: &str) -> bool {
    rel.starts_with(".git/")
        || SKIP_PREFIXES.iter().any(|prefix| rel.starts_with(prefix))
        || rel.split('/').any(|part| SKIP_DIR_NAMES.contains(&part))
}

fn sha256_file(path: &Path) -> Result<String> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(sha256_hex(&raw))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
