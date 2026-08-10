use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use provenance_core::ClaimRecord;
use provenance_store::ProvenanceStore;
use regex::Regex;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use toml::Value;

const MIN_PROOF_ATOM_COUNT: usize = 5;

#[derive(Parser, Debug)]
#[command(
    name = "knowledge-atoms",
    about = "Build and verify structured knowledge-atom registries"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Build(BuildArgs),
    Verify(VerifyArgs),
}

#[derive(Args, Debug, Clone)]
struct BuildArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(long, default_value = "registry/knowledge/claim_atoms.toml")]
    claim_out: PathBuf,
    #[arg(long, default_value = "registry/knowledge/equation_atoms.toml")]
    equation_out: PathBuf,
    #[arg(long, default_value = "registry/knowledge/proof_atoms.toml")]
    proof_out: PathBuf,
    #[arg(long, default_value = "registry/knowledge/structured_corpora.toml")]
    summary_out: PathBuf,
}

#[derive(Args, Debug, Clone)]
struct VerifyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/knowledge/claim_atoms.toml")]
    claim_registry: PathBuf,
    #[arg(long, default_value = "registry/knowledge/equation_atoms.toml")]
    equation_registry: PathBuf,
    #[arg(long, default_value = "registry/knowledge/proof_atoms.toml")]
    proof_registry: PathBuf,
    #[arg(long, default_value = "registry/knowledge/structured_corpora.toml")]
    summary_registry: PathBuf,
}

#[derive(Debug, Clone)]
struct SourceDoc {
    source_uid: String,
    source_group: String,
    source_registry: String,
    source_path: String,
    derivation_input: String,
    origin_path_state: String,
    canonical_body_sha256: String,
    title: String,
    line_count: usize,
    body: String,
}

#[derive(Debug, Clone)]
struct Section {
    title: String,
    level: usize,
    line_start: usize,
    line_end: usize,
    body_lines: Vec<String>,
}

#[derive(Debug, Clone)]
struct ClaimAtom {
    claim_id: String,
    statement: String,
    status_token: String,
    status_detail: String,
    last_verified: String,
    where_stated: String,
    verification_rule: String,
    h0: String,
    h1: String,
    decision_rule: String,
    hypothesis_block_present: bool,
    evidence_refs: Vec<String>,
    source_uid: String,
    source_group: String,
    source_registry: String,
    source_path: String,
    source_line: usize,
}

#[derive(Debug, Clone)]
struct EquationAtom {
    source_uid: String,
    source_group: String,
    source_registry: String,
    source_path: String,
    section_title: String,
    source_line: usize,
    expression: String,
    relation_operator: String,
    lhs_expression: String,
    rhs_expression: String,
    extraction_kind: String,
    equation_kind: String,
    symbol_names: Vec<String>,
    numeric_constants: Vec<String>,
    domain_hint: String,
    claim_refs: Vec<String>,
}

#[derive(Debug, Clone)]
struct ProofAtom {
    source_uid: String,
    source_group: String,
    source_registry: String,
    source_path: String,
    section_title: String,
    section_level: usize,
    line_start: usize,
    line_end: usize,
    proof_kind: String,
    step_count: usize,
    supports_claim: bool,
    assumption_lines: Vec<String>,
    decision_lines: Vec<String>,
    conclusion_lines: Vec<String>,
    inference_markers: Vec<String>,
    assumption_text: String,
    decision_rule_text: String,
    conclusion_text: String,
    excerpt: String,
    claim_refs: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Build(args) => build(&args),
        Command::Verify(args) => verify(&args),
    }
}

fn build(args: &BuildArgs) -> Result<()> {
    let repo_root = args.repo_root.canonicalize().context("resolve repo root")?;
    let claim_source = load_primary_claim_source(&repo_root, &args.canonical_db)?;
    let sources = load_sources(&repo_root, claim_source.clone())?;

    let claim_atoms = parse_claim_rows(&claim_source)?;
    let mut equation_atoms = Vec::new();
    let mut proof_atoms = Vec::new();
    for source in &sources {
        equation_atoms.extend(extract_equations_from_source(source)?);
        proof_atoms.extend(extract_proofs_from_source(source)?);
    }

    if !proof_atoms
        .iter()
        .any(|atom| atom.source_group == "research_narrative")
    {
        for source in &sources {
            if source.source_group != "research_narrative" {
                continue;
            }
            if let Some(fallback) = build_fallback_proof_atom(source)? {
                proof_atoms.push(fallback);
                break;
            }
        }
    }

    proof_atoms.sort_by(|lhs, rhs| {
        (&lhs.source_uid, lhs.line_start, &lhs.section_title).cmp(&(
            &rhs.source_uid,
            rhs.line_start,
            &rhs.section_title,
        ))
    });

    write_ascii(
        &repo_root.join(&args.claim_out),
        &render_claim_atoms(&claim_atoms),
    )?;
    write_ascii(
        &repo_root.join(&args.equation_out),
        &render_equation_atoms(&equation_atoms),
    )?;
    write_ascii(
        &repo_root.join(&args.proof_out),
        &render_proof_atoms(&proof_atoms),
    )?;
    write_ascii(
        &repo_root.join(&args.summary_out),
        &render_structured_corpora(&sources, &claim_atoms, &equation_atoms, &proof_atoms),
    )?;

    println!(
        "Wrote structured knowledge atoms: claims={} equations={} proofs={} sources={}",
        claim_atoms.len(),
        equation_atoms.len(),
        proof_atoms.len(),
        sources.len()
    );
    Ok(())
}

fn verify(args: &VerifyArgs) -> Result<()> {
    let root = args.repo_root.canonicalize().context("resolve repo root")?;
    let claim_path = root.join(&args.claim_registry);
    let equation_path = root.join(&args.equation_registry);
    let proof_path = root.join(&args.proof_registry);
    let summary_path = root.join(&args.summary_registry);
    for path in [&claim_path, &equation_path, &proof_path, &summary_path] {
        if !path.exists() {
            bail!("ERROR: missing structured registry: {}", path.display());
        }
        assert_ascii_file(path)?;
    }

    let claim = load_toml(&claim_path)?;
    let equation = load_toml(&equation_path)?;
    let proof = load_toml(&proof_path)?;
    let summary = load_toml(&summary_path)?;

    let claim_atoms = table_array(&claim, "atom");
    let equation_atoms = table_array(&equation, "atom");
    let proof_atoms = table_array(&proof, "atom");
    let source_rows = table_array(&summary, "source");

    let mut failures = Vec::new();
    if claim
        .get("knowledge_claim_atoms")
        .and_then(Value::as_table)
        .and_then(|row| row.get("atom_count"))
        .and_then(Value::as_integer)
        .unwrap_or(-1)
        != claim_atoms.len() as i64
    {
        failures.push("claim_atoms atom_count metadata mismatch.".to_string());
    }
    if equation
        .get("knowledge_equation_atoms")
        .and_then(Value::as_table)
        .and_then(|row| row.get("atom_count"))
        .and_then(Value::as_integer)
        .unwrap_or(-1)
        != equation_atoms.len() as i64
    {
        failures.push("equation_atoms atom_count metadata mismatch.".to_string());
    }
    if proof
        .get("knowledge_proof_atoms")
        .and_then(Value::as_table)
        .and_then(|row| row.get("atom_count"))
        .and_then(Value::as_integer)
        .unwrap_or(-1)
        != proof_atoms.len() as i64
    {
        failures.push("proof_atoms atom_count metadata mismatch.".to_string());
    }
    if summary
        .get("structured_corpora")
        .and_then(Value::as_table)
        .and_then(|row| row.get("source_count"))
        .and_then(Value::as_integer)
        .unwrap_or(-1)
        != source_rows.len() as i64
    {
        failures.push("structured_corpora source_count metadata mismatch.".to_string());
    }

    unique_ids(claim_atoms, "id", &mut failures, "claim_atoms");
    unique_ids(equation_atoms, "id", &mut failures, "equation_atoms");
    unique_ids(proof_atoms, "id", &mut failures, "proof_atoms");
    unique_ids(source_rows, "id", &mut failures, "structured_corpora");

    if claim_atoms.len() < 300 {
        failures.push(format!(
            "claim_atoms too small: {} < 300",
            claim_atoms.len()
        ));
    }
    if equation_atoms.len() < 150 {
        failures.push(format!(
            "equation_atoms too small: {} < 150",
            equation_atoms.len()
        ));
    }
    if proof_atoms.len() < MIN_PROOF_ATOM_COUNT {
        failures.push(format!(
            "proof_atoms too small: {} < {}",
            proof_atoms.len(),
            MIN_PROOF_ATOM_COUNT
        ));
    }

    let claim_groups: BTreeSet<String> = claim_atoms
        .iter()
        .map(|row| table_str(row, "source_group").to_string())
        .collect();
    if !claim_groups.contains("doc_claim_matrix") {
        failures.push("claim_atoms missing doc_claim_matrix coverage.".to_string());
    }

    let eq_groups: BTreeSet<String> = equation_atoms
        .iter()
        .map(|row| table_str(row, "source_group").to_string())
        .collect();
    for required in [
        "doc_claim_matrix",
        "research_narrative",
        "data_artifact_narrative",
    ] {
        if !eq_groups.contains(required) {
            failures.push(format!("equation_atoms missing source_group={required}"));
        }
    }

    let proof_groups: BTreeSet<String> = proof_atoms
        .iter()
        .map(|row| table_str(row, "source_group").to_string())
        .collect();
    if !proof_groups.contains("research_narrative") {
        failures.push("proof_atoms missing research_narrative coverage.".to_string());
    }

    for atom in claim_atoms {
        let claim_id = table_str(atom, "claim_id");
        if !claim_id.starts_with("C-") {
            failures.push(format!("invalid claim_id format: {claim_id}"));
            break;
        }
        if atom
            .get("hypothesis_block_present")
            .and_then(Value::as_bool)
            .is_none()
        {
            failures.push(format!(
                "claim atom missing bool hypothesis flag: {}",
                table_str(atom, "id")
            ));
            break;
        }
    }

    for atom in equation_atoms {
        if table_str(atom, "expression").trim().is_empty() {
            failures.push("equation atom has empty expression.".to_string());
            break;
        }
        if table_str(atom, "relation_operator").trim().is_empty() {
            failures.push(format!(
                "equation atom missing relation_operator: {}",
                table_str(atom, "id")
            ));
            break;
        }
        if table_str(atom, "lhs_expression").trim().is_empty() {
            failures.push(format!(
                "equation atom missing lhs_expression: {}",
                table_str(atom, "id")
            ));
            break;
        }
    }

    for atom in proof_atoms {
        if table_int(atom, "step_count") <= 0 {
            failures.push(format!(
                "proof atom has invalid step_count: {}",
                table_str(atom, "id")
            ));
            break;
        }
        for key in [
            "assumption_lines",
            "decision_lines",
            "conclusion_lines",
            "inference_markers",
        ] {
            if atom.get(key).and_then(Value::as_array).is_none() {
                failures.push(format!(
                    "proof atom missing list field {key}: {}",
                    table_str(atom, "id")
                ));
                break;
            }
        }
    }

    for row in source_rows {
        if let Err(error) = verify_source_provenance(&root, row) {
            failures.push(error.to_string());
        }
        if row
            .get("narrative_compaction_recommended")
            .and_then(Value::as_bool)
            != Some(true)
        {
            failures.push(format!(
                "source row missing compaction recommendation: {}",
                table_str(row, "source_uid")
            ));
        }
        if table_int(row, "target_summary_max_lines") <= 0 {
            failures.push(format!(
                "source row has invalid target_summary_max_lines: {}",
                table_str(row, "source_uid")
            ));
        }
    }

    if !failures.is_empty() {
        eprintln!("ERROR: structured knowledge atom verification failed.");
        for item in failures.iter().take(200) {
            eprintln!("- {item}");
        }
        if failures.len() > 200 {
            eprintln!("- ... and {} more failures", failures.len() - 200);
        }
        bail!("structured knowledge atom verification failed");
    }

    println!(
        "OK: structured knowledge atoms verified. claims={} equations={} proofs={} sources={}",
        claim_atoms.len(),
        equation_atoms.len(),
        proof_atoms.len(),
        source_rows.len()
    );
    Ok(())
}

fn load_sources(repo_root: &Path, claim_source: SourceDoc) -> Result<Vec<SourceDoc>> {
    let mut sources = vec![claim_source];
    for (registry_path, source_group) in [
        ("registry/research_narratives.toml", "research_narrative"),
        (
            "registry/data_artifact_narratives.toml",
            "data_artifact_narrative",
        ),
    ] {
        let raw = load_toml(&repo_root.join(registry_path))?;
        for row in table_array(&raw, "document") {
            let source_uid = table_str(row, "id").trim().to_string();
            let source_path = table_str(row, "source_markdown").trim().to_string();
            if source_uid.is_empty() || source_path.is_empty() {
                continue;
            }
            let registered_body = table_str(row, "body_markdown").to_string();
            let origin_path_state = registry_origin_path_state(repo_root, &source_path);
            let (body, derivation_input) = if registered_body.trim().is_empty() {
                if origin_path_state == "origin_path_absent" {
                    bail!(
                        "registry source {source_uid} retains neither body_markdown nor origin path {source_path}"
                    );
                }
                let path = repo_root.join(&source_path);
                (
                    fs::read_to_string(&path)
                        .with_context(|| format!("read working-tree source {}", path.display()))?,
                    "working_tree_markdown".to_string(),
                )
            } else {
                (registered_body, "registry_body_markdown".to_string())
            };
            sources.push(SourceDoc {
                source_uid: source_uid.clone(),
                source_group: source_group.to_string(),
                source_registry: registry_path.to_string(),
                source_path,
                derivation_input,
                origin_path_state,
                canonical_body_sha256: sha256_hex(&body),
                title: {
                    let title = table_str(row, "title");
                    if title.is_empty() {
                        source_uid.clone()
                    } else {
                        collapse_ws(title)
                    }
                },
                line_count: line_count(&body),
                body,
            });
        }
    }
    sources.sort_by(|lhs, rhs| {
        (&lhs.source_group, &lhs.source_uid).cmp(&(&rhs.source_group, &rhs.source_uid))
    });
    Ok(sources)
}

fn load_primary_claim_source(repo_root: &Path, canonical_db: &Path) -> Result<SourceDoc> {
    let db_path = repo_root.join(canonical_db);
    if db_path.exists() {
        let store = ProvenanceStore::open(&db_path)
            .with_context(|| format!("open canonical db {}", db_path.display()))?;
        let claims = store
            .list_claims()
            .with_context(|| format!("load claims from {}", db_path.display()))?;
        if !claims.is_empty() {
            let body = render_claim_matrix_from_control_plane(&claims);
            return Ok(SourceDoc {
                source_uid: "CONTROL_PLANE_CLAIMS".to_string(),
                source_group: "doc_claim_matrix".to_string(),
                source_registry: "registry/canonical/control_plane.sqlite3".to_string(),
                source_path: "registry/canonical/control_plane.sqlite3".to_string(),
                derivation_input: "control_plane_claim_rows".to_string(),
                origin_path_state: "not_applicable".to_string(),
                canonical_body_sha256: sha256_hex(&body),
                title: "Claims / Evidence Matrix (SQLite compatibility render)".to_string(),
                line_count: line_count(&body),
                body,
            });
        }
    }

    let live_matrix_path = repo_root.join("docs/CLAIMS_EVIDENCE_MATRIX.md");
    if live_matrix_path.exists() {
        let body = fs::read_to_string(&live_matrix_path)
            .with_context(|| format!("read {}", live_matrix_path.display()))?;
        return Ok(SourceDoc {
            source_uid: "CLAIMS_EVIDENCE_MATRIX".to_string(),
            source_group: "doc_claim_matrix".to_string(),
            source_registry: "docs/CLAIMS_EVIDENCE_MATRIX.md".to_string(),
            source_path: "docs/CLAIMS_EVIDENCE_MATRIX.md".to_string(),
            derivation_input: "working_tree_markdown".to_string(),
            origin_path_state: "working_tree_path_present".to_string(),
            canonical_body_sha256: sha256_hex(&body),
            title: title_from_markdown("docs/CLAIMS_EVIDENCE_MATRIX.md", &body),
            line_count: line_count(&body),
            body,
        });
    }

    let fallback_path = repo_root.join("registry/knowledge/docs/DOC-0023.toml");
    if fallback_path.exists() {
        let raw = load_toml(&fallback_path)?;
        let payload = raw
            .get("document")
            .context("DOC-0023 missing [document] table")?;
        let body = table_str(payload, "content_markdown").to_string();
        return Ok(SourceDoc {
            source_uid: "DOC-0023".to_string(),
            source_group: "doc_claim_matrix".to_string(),
            source_registry: "registry/knowledge/docs/DOC-0023.toml".to_string(),
            source_path: table_str(payload, "source_path").to_string(),
            derivation_input: "registry_document_payload".to_string(),
            origin_path_state: registry_origin_path_state(
                repo_root,
                table_str(payload, "source_path"),
            ),
            canonical_body_sha256: sha256_hex(&body),
            title: collapse_ws(table_str(payload, "title")),
            line_count: table_int(payload, "source_line_count").max(0) as usize,
            body,
        });
    }

    bail!(
        "ERROR: missing both docs/CLAIMS_EVIDENCE_MATRIX.md and registry/knowledge/docs/DOC-0023.toml"
    )
}

fn render_claim_matrix_from_control_plane(claims: &[ClaimRecord]) -> String {
    let mut lines = vec![
        "# Claims / Evidence Matrix (Markdown Mirror)".to_string(),
        String::new(),
        "<!-- AUTO-GENERATED: DO NOT EDIT -->".to_string(),
        "<!-- Source of truth: registry/canonical/control_plane.sqlite3 -->".to_string(),
        String::new(),
        "| ID | Claim | Where stated | Status | Last verified | What would verify/refute it |"
            .to_string(),
        "|---:|---|---|---|---|---|".to_string(),
    ];
    for claim in claims {
        let verify_rule = claim.status_note.clone().unwrap_or_default();
        lines.push(format!(
            "| {} | {} | {} | **{}** | {} | {} |",
            claim.id,
            pipe_escape(&claim.statement),
            pipe_escape(&claim.where_stated),
            pipe_escape(&claim.status),
            pipe_escape(&claim.last_verified),
            pipe_escape(&verify_rule)
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn pipe_escape(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn parse_claim_rows(doc: &SourceDoc) -> Result<Vec<ClaimAtom>> {
    let claim_id_re = Regex::new(r"\bC-\d{3}\b")?;
    let evidence_ref_re = Regex::new(r"\b(?:C|I|E)-\d{3}\b")?;
    let direct_claim_heading_re = Regex::new(r"^(C-\d{3})\s*:\s*(.+)$")?;
    let mut atoms = Vec::new();
    let mut claim_row_by_id: BTreeMap<String, usize> = BTreeMap::new();
    let lines: Vec<String> = ascii_sanitize(&doc.body)
        .lines()
        .map(ToOwned::to_owned)
        .collect();

    for (idx, raw) in lines.iter().enumerate() {
        let line = raw.trim();
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = line
            .trim_matches('|')
            .split('|')
            .map(|cell| cell.trim().to_string())
            .collect();
        if cells.len() < 6 {
            continue;
        }
        let claim_id = cells[0].clone();
        if !claim_id_re
            .find(&claim_id)
            .map(|m| m.as_str() == claim_id)
            .unwrap_or(false)
        {
            continue;
        }
        let status_raw = cells[3].clone();
        let status_token = extract_bold_token(&status_raw)
            .map(|token| collapse_ws(&token).to_uppercase().replace(' ', "_"))
            .unwrap_or_else(|| "UNSPECIFIED".to_string());
        let evidence_blob = format!(
            "{} {} {} {}",
            cells[2],
            status_raw,
            cells[4],
            cells[5..].join(" | ")
        );
        let atom = ClaimAtom {
            claim_id: claim_id.clone(),
            statement: collapse_ws(&cells[1]),
            status_token,
            status_detail: collapse_ws(&cells[3]),
            last_verified: collapse_ws(&cells[4]),
            where_stated: collapse_ws(&cells[2]),
            verification_rule: collapse_ws(&cells[5..].join(" | ")),
            h0: String::new(),
            h1: String::new(),
            decision_rule: String::new(),
            hypothesis_block_present: false,
            evidence_refs: sorted_unique_regex(&evidence_blob, &evidence_ref_re),
            source_uid: doc.source_uid.clone(),
            source_group: doc.source_group.clone(),
            source_registry: doc.source_registry.clone(),
            source_path: doc.source_path.clone(),
            source_line: idx + 1,
        };
        claim_row_by_id.insert(claim_id, atoms.len());
        atoms.push(atom);
    }

    for section in split_sections(&doc.body)? {
        let heading = section.title.trim().to_string();
        let Some(caps) = direct_claim_heading_re.captures(&heading) else {
            continue;
        };
        let claim_id = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        let Some(atom_index) = claim_row_by_id.get(claim_id).copied() else {
            continue;
        };
        let mut h0 = String::new();
        let mut h1 = String::new();
        let mut decision = String::new();
        for raw in &section.body_lines {
            let line = collapse_ws(raw);
            if line.is_empty() {
                continue;
            }
            let lowered = line.to_lowercase();
            if lowered.contains("**h0**") || lowered.starts_with("h0:") {
                h0 = line;
            } else if lowered.contains("**h1**") || lowered.starts_with("h1:") {
                h1 = line;
            } else if lowered.contains("decision rule") {
                decision = line;
            }
        }
        if let Some(atom) = atoms.get_mut(atom_index) {
            atom.h0 = h0;
            atom.h1 = h1;
            atom.decision_rule = decision;
            atom.hypothesis_block_present =
                !(atom.h0.is_empty() && atom.h1.is_empty() && atom.decision_rule.is_empty());
        }
    }

    atoms.sort_by(|lhs, rhs| {
        (&lhs.claim_id, lhs.source_line).cmp(&(&rhs.claim_id, rhs.source_line))
    });
    Ok(atoms)
}

fn extract_equations_from_source(doc: &SourceDoc) -> Result<Vec<EquationAtom>> {
    let claim_id_re = Regex::new(r"\bC-\d{3}\b")?;
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+?)\s*$")?;
    let inline_math_re = Regex::new(r"\$([^$\n]{3,240})\$")?;
    let mut atoms = Vec::new();
    let mut seen = BTreeSet::new();
    let mut section_title = "(root)".to_string();
    let mut in_code_fence = false;
    let lines: Vec<String> = ascii_sanitize(&doc.body)
        .lines()
        .map(ToOwned::to_owned)
        .collect();
    for (idx, raw) in lines.iter().enumerate() {
        let line_no = idx + 1;
        let stripped = raw.trim();
        if stripped.starts_with("<!--") && stripped.ends_with("-->") {
            continue;
        }
        if let Some(caps) = heading_re.captures(raw) {
            section_title = collapse_ws(caps.get(2).map(|m| m.as_str()).unwrap_or_default());
            continue;
        }
        if stripped.starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        for caps in inline_math_re.captures_iter(raw) {
            let expr = collapse_ws(caps.get(1).map(|m| m.as_str()).unwrap_or_default());
            if expr.len() < 3 {
                continue;
            }
            let key = format!("{}::{line_no}::{expr}", doc.source_uid);
            if !seen.insert(key) {
                continue;
            }
            let (relation_operator, lhs_expression, rhs_expression) = parse_relation(&expr);
            let (symbol_names, numeric_constants) = extract_symbol_roles(&expr)?;
            atoms.push(EquationAtom {
                source_uid: doc.source_uid.clone(),
                source_group: doc.source_group.clone(),
                source_registry: doc.source_registry.clone(),
                source_path: doc.source_path.clone(),
                section_title: section_title.clone(),
                source_line: line_no,
                expression: expr.clone(),
                relation_operator,
                lhs_expression,
                rhs_expression,
                extraction_kind: "inline_math".to_string(),
                equation_kind: classify_equation(&expr),
                symbol_names,
                numeric_constants,
                domain_hint: infer_domain_hint(doc, &section_title, &expr),
                claim_refs: sorted_unique_regex(&expr, &claim_id_re),
            });
        }

        let mut candidate = stripped.to_string();
        if candidate.starts_with("- ") {
            candidate = candidate[2..].trim().to_string();
        }
        if candidate.is_empty()
            || candidate.starts_with('#')
            || candidate.starts_with('|')
            || candidate.starts_with('*')
            || in_code_fence
        {
            continue;
        }
        if !["=", "->", "<=", ">=", "!="]
            .iter()
            .any(|op| candidate.contains(op))
        {
            continue;
        }
        if !candidate
            .chars()
            .any(|ch| ch.is_ascii_alphabetic() || ch == '_')
        {
            continue;
        }
        let symbol_count = candidate
            .chars()
            .filter(|ch| "=+-*/^_()<>".contains(*ch))
            .count();
        if symbol_count < 2 || candidate.len() > 220 {
            continue;
        }
        let expr = collapse_ws(&candidate);
        let key = format!("{}::{line_no}::{expr}", doc.source_uid);
        if !seen.insert(key) {
            continue;
        }
        let (relation_operator, lhs_expression, rhs_expression) = parse_relation(&expr);
        let (symbol_names, numeric_constants) = extract_symbol_roles(&expr)?;
        atoms.push(EquationAtom {
            source_uid: doc.source_uid.clone(),
            source_group: doc.source_group.clone(),
            source_registry: doc.source_registry.clone(),
            source_path: doc.source_path.clone(),
            section_title: section_title.clone(),
            source_line: line_no,
            expression: expr.clone(),
            relation_operator,
            lhs_expression,
            rhs_expression,
            extraction_kind: "equation_like_line".to_string(),
            equation_kind: classify_equation(&expr),
            symbol_names,
            numeric_constants,
            domain_hint: infer_domain_hint(doc, &section_title, &expr),
            claim_refs: sorted_unique_regex(&expr, &claim_id_re),
        });
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

fn extract_proofs_from_source(doc: &SourceDoc) -> Result<Vec<ProofAtom>> {
    let claim_id_re = Regex::new(r"\bC-\d{3}\b")?;
    let proof_keywords = [
        "proof",
        "theorem",
        "lemma",
        "corollary",
        "proposition",
        "axiom",
        "derivation",
        "hypothesis",
    ];
    let mut atoms = Vec::new();
    for section in split_sections(&doc.body)? {
        let section_text = section.body_lines.join("\n");
        let section_text_ascii = ascii_sanitize(&section_text);
        let lowered_title = section.title.to_lowercase();
        let lowered_body = section_text_ascii.to_lowercase();
        let is_candidate = proof_keywords
            .iter()
            .any(|token| lowered_title.contains(token))
            || ["**h0**", "**h1**", "decision rule", "therefore", "hence"]
                .iter()
                .any(|marker| lowered_body.contains(marker));
        if !is_candidate {
            continue;
        }

        let non_empty_lines: Vec<String> = section
            .body_lines
            .iter()
            .map(|line| collapse_ws(line))
            .filter(|line| !line.is_empty())
            .collect();
        let assumption_lines = filter_lines(
            &non_empty_lines,
            &["**h0**", "**h1**", "assume", "given", "hypothesis"],
            12,
        );
        let decision_lines = filter_lines(&non_empty_lines, &["decision rule"], 12);
        let conclusion_lines = filter_lines(
            &non_empty_lines,
            &[
                "status",
                "therefore",
                "hence",
                "rejected",
                "verified",
                "refuted",
            ],
            12,
        );
        let inference_markers = filter_lines(
            &non_empty_lines,
            &["therefore", "hence", "implies", "follows"],
            12,
        );
        let excerpt = collapse_ws(
            &non_empty_lines
                .iter()
                .take(6)
                .map(|line| truncate(line, 120))
                .collect::<Vec<_>>()
                .join(" || "),
        );
        atoms.push(ProofAtom {
            source_uid: doc.source_uid.clone(),
            source_group: doc.source_group.clone(),
            source_registry: doc.source_registry.clone(),
            source_path: doc.source_path.clone(),
            section_title: section.title.clone(),
            section_level: section.level,
            line_start: section.line_start,
            line_end: section.line_end,
            proof_kind: classify_proof_section(&section.title, &section_text_ascii),
            step_count: non_empty_lines.len(),
            supports_claim: claim_id_re.is_match(&section_text_ascii),
            assumption_text: collapse_ws(&assumption_lines.join(" | ")),
            decision_rule_text: collapse_ws(&decision_lines.join(" | ")),
            conclusion_text: collapse_ws(&conclusion_lines.join(" | ")),
            excerpt,
            claim_refs: sorted_unique_regex(&section_text_ascii, &claim_id_re),
            assumption_lines,
            decision_lines,
            conclusion_lines,
            inference_markers,
        });
    }
    atoms.sort_by(|lhs, rhs| {
        (&lhs.source_uid, lhs.line_start, &lhs.section_title).cmp(&(
            &rhs.source_uid,
            rhs.line_start,
            &rhs.section_title,
        ))
    });
    Ok(atoms)
}

fn build_fallback_proof_atom(doc: &SourceDoc) -> Result<Option<ProofAtom>> {
    let claim_id_re = Regex::new(r"\bC-\d{3}\b")?;
    for section in split_sections(&doc.body)? {
        let section_text = section.body_lines.join("\n");
        let section_text_ascii = ascii_sanitize(&section_text);
        let non_empty_lines: Vec<String> = section
            .body_lines
            .iter()
            .map(|line| collapse_ws(line))
            .filter(|line| !line.is_empty())
            .collect();
        if non_empty_lines.len() < 3 {
            continue;
        }
        if non_empty_lines
            .iter()
            .take(3)
            .all(|line| line.starts_with('|'))
        {
            continue;
        }
        let assumption_lines = non_empty_lines.iter().take(2).cloned().collect::<Vec<_>>();
        let mut decision_lines = filter_lines(
            &non_empty_lines,
            &["decision rule", "implies", "follows"],
            2,
        );
        let mut conclusion_lines = filter_lines(
            &non_empty_lines,
            &[
                "status",
                "therefore",
                "hence",
                "result",
                "conclusion",
                "verified",
                "refuted",
            ],
            2,
        );
        if decision_lines.is_empty() && non_empty_lines.len() >= 2 {
            decision_lines = non_empty_lines.iter().skip(1).take(1).cloned().collect();
        }
        if conclusion_lines.is_empty() {
            conclusion_lines = non_empty_lines.iter().rev().take(1).cloned().collect();
            conclusion_lines.reverse();
        }
        let inference_markers = filter_lines(
            &non_empty_lines,
            &["therefore", "hence", "implies", "follows"],
            4,
        );
        let claim_refs = sorted_unique_regex(&section_text_ascii, &claim_id_re);
        let excerpt = collapse_ws(
            &non_empty_lines
                .iter()
                .take(6)
                .map(|line| truncate(line, 120))
                .collect::<Vec<_>>()
                .join(" || "),
        );
        return Ok(Some(ProofAtom {
            source_uid: doc.source_uid.clone(),
            source_group: doc.source_group.clone(),
            source_registry: doc.source_registry.clone(),
            source_path: doc.source_path.clone(),
            section_title: section.title.clone(),
            section_level: section.level,
            line_start: section.line_start,
            line_end: section.line_end,
            proof_kind: "argument_section".to_string(),
            step_count: non_empty_lines.len(),
            supports_claim: !claim_refs.is_empty(),
            assumption_text: collapse_ws(&assumption_lines.join(" | ")),
            decision_rule_text: collapse_ws(&decision_lines.join(" | ")),
            conclusion_text: collapse_ws(&conclusion_lines.join(" | ")),
            excerpt,
            claim_refs,
            assumption_lines,
            decision_lines,
            conclusion_lines,
            inference_markers,
        }));
    }
    Ok(None)
}

fn split_sections(body: &str) -> Result<Vec<Section>> {
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+?)\s*$")?;
    let lines: Vec<String> = ascii_sanitize(body)
        .lines()
        .map(ToOwned::to_owned)
        .collect();
    let mut sections = Vec::new();
    let mut current_title = "(root)".to_string();
    let mut current_level = 0usize;
    let mut current_start = 1usize;
    let mut current_body = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        let line_no = idx + 1;
        if let Some(caps) = heading_re.captures(line) {
            sections.push(Section {
                title: current_title,
                level: current_level,
                line_start: current_start,
                line_end: current_start.max(line_no.saturating_sub(1)),
                body_lines: current_body,
            });
            current_title = collapse_ws(caps.get(2).map(|m| m.as_str()).unwrap_or_default());
            current_level = caps.get(1).map(|m| m.as_str().len()).unwrap_or(0);
            current_start = line_no;
            current_body = Vec::new();
        } else {
            current_body.push(line.clone());
        }
    }
    sections.push(Section {
        title: current_title,
        level: current_level,
        line_start: current_start,
        line_end: current_start.max(lines.len()),
        body_lines: current_body,
    });
    Ok(sections)
}

fn render_claim_atoms(claim_atoms: &[ClaimAtom]) -> String {
    let unique_claim_ids: BTreeSet<String> = claim_atoms
        .iter()
        .map(|atom| atom.claim_id.clone())
        .collect();
    let mut lines = vec![
        "# Structured claim atoms extracted from selected high-information corpora.".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/knowledge_atoms.rs".to_string(),
        "".to_string(),
        "[knowledge_claim_atoms]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("atom_count = {}", claim_atoms.len()),
        format!("unique_claim_id_count = {}", unique_claim_ids.len()),
        "".to_string(),
    ];
    for (idx, atom) in claim_atoms.iter().enumerate() {
        lines.push("[[atom]]".to_string());
        lines.push(format!("id = {}", esc(&format!("CLA-{:04}", idx + 1))));
        lines.push(format!("claim_id = {}", esc(&atom.claim_id)));
        lines.push(format!("statement = {}", esc(&atom.statement)));
        lines.push(format!("status_token = {}", esc(&atom.status_token)));
        lines.push(format!("status_detail = {}", esc(&atom.status_detail)));
        lines.push(format!("last_verified = {}", esc(&atom.last_verified)));
        lines.push(format!("where_stated = {}", esc(&atom.where_stated)));
        lines.push(format!(
            "verification_rule = {}",
            esc(&atom.verification_rule)
        ));
        lines.push(format!("h0 = {}", esc(&atom.h0)));
        lines.push(format!("h1 = {}", esc(&atom.h1)));
        lines.push(format!("decision_rule = {}", esc(&atom.decision_rule)));
        lines.push(format!(
            "hypothesis_block_present = {}",
            if atom.hypothesis_block_present {
                "true"
            } else {
                "false"
            }
        ));
        lines.push(format!(
            "evidence_refs = {}",
            render_list(&atom.evidence_refs)
        ));
        lines.push(format!("source_uid = {}", esc(&atom.source_uid)));
        lines.push(format!("source_group = {}", esc(&atom.source_group)));
        lines.push(format!("source_registry = {}", esc(&atom.source_registry)));
        lines.push(format!("source_path = {}", esc(&atom.source_path)));
        lines.push(format!("source_line = {}", atom.source_line));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_equation_atoms(equation_atoms: &[EquationAtom]) -> String {
    let mut lines = vec![
        "# Structured equation atoms extracted from selected high-information corpora.".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/knowledge_atoms.rs".to_string(),
        "".to_string(),
        "[knowledge_equation_atoms]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("atom_count = {}", equation_atoms.len()),
        "".to_string(),
    ];
    for (idx, atom) in equation_atoms.iter().enumerate() {
        lines.push("[[atom]]".to_string());
        lines.push(format!("id = {}", esc(&format!("EQA-{:04}", idx + 1))));
        lines.push(format!("expression = {}", esc(&atom.expression)));
        lines.push(format!(
            "relation_operator = {}",
            esc(&atom.relation_operator)
        ));
        lines.push(format!("lhs_expression = {}", esc(&atom.lhs_expression)));
        lines.push(format!("rhs_expression = {}", esc(&atom.rhs_expression)));
        lines.push(format!("equation_kind = {}", esc(&atom.equation_kind)));
        lines.push(format!("extraction_kind = {}", esc(&atom.extraction_kind)));
        lines.push(format!(
            "symbol_names = {}",
            render_list(&atom.symbol_names)
        ));
        lines.push(format!(
            "numeric_constants = {}",
            render_list(&atom.numeric_constants)
        ));
        lines.push(format!("domain_hint = {}", esc(&atom.domain_hint)));
        lines.push(format!("section_title = {}", esc(&atom.section_title)));
        lines.push(format!("source_uid = {}", esc(&atom.source_uid)));
        lines.push(format!("source_group = {}", esc(&atom.source_group)));
        lines.push(format!("source_registry = {}", esc(&atom.source_registry)));
        lines.push(format!("source_path = {}", esc(&atom.source_path)));
        lines.push(format!("source_line = {}", atom.source_line));
        lines.push(format!("claim_refs = {}", render_list(&atom.claim_refs)));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_proof_atoms(proof_atoms: &[ProofAtom]) -> String {
    let mut lines = vec![
        "# Structured proof/derivation atoms extracted from selected high-information corpora."
            .to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/knowledge_atoms.rs".to_string(),
        "".to_string(),
        "[knowledge_proof_atoms]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("atom_count = {}", proof_atoms.len()),
        "".to_string(),
    ];
    for (idx, atom) in proof_atoms.iter().enumerate() {
        lines.push("[[atom]]".to_string());
        lines.push(format!("id = {}", esc(&format!("PRF-{:04}", idx + 1))));
        lines.push(format!("proof_kind = {}", esc(&atom.proof_kind)));
        lines.push(format!("section_title = {}", esc(&atom.section_title)));
        lines.push(format!("section_level = {}", atom.section_level));
        lines.push(format!("line_start = {}", atom.line_start));
        lines.push(format!("line_end = {}", atom.line_end));
        lines.push(format!("step_count = {}", atom.step_count));
        lines.push(format!(
            "supports_claim = {}",
            if atom.supports_claim { "true" } else { "false" }
        ));
        lines.push(format!(
            "assumption_lines = {}",
            render_list(&atom.assumption_lines)
        ));
        lines.push(format!(
            "decision_lines = {}",
            render_list(&atom.decision_lines)
        ));
        lines.push(format!(
            "conclusion_lines = {}",
            render_list(&atom.conclusion_lines)
        ));
        lines.push(format!(
            "inference_markers = {}",
            render_list(&atom.inference_markers)
        ));
        lines.push(format!("assumption_text = {}", esc(&atom.assumption_text)));
        lines.push(format!(
            "decision_rule_text = {}",
            esc(&atom.decision_rule_text)
        ));
        lines.push(format!("conclusion_text = {}", esc(&atom.conclusion_text)));
        lines.push(format!("excerpt = {}", esc(&atom.excerpt)));
        lines.push(format!("claim_refs = {}", render_list(&atom.claim_refs)));
        lines.push(format!("source_uid = {}", esc(&atom.source_uid)));
        lines.push(format!("source_group = {}", esc(&atom.source_group)));
        lines.push(format!("source_registry = {}", esc(&atom.source_registry)));
        lines.push(format!("source_path = {}", esc(&atom.source_path)));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_structured_corpora(
    sources: &[SourceDoc],
    claim_atoms: &[ClaimAtom],
    equation_atoms: &[EquationAtom],
    proof_atoms: &[ProofAtom],
) -> String {
    let mut claim_counts = BTreeMap::<String, usize>::new();
    let mut eq_counts = BTreeMap::<String, usize>::new();
    let mut proof_counts = BTreeMap::<String, usize>::new();
    for atom in claim_atoms {
        *claim_counts.entry(atom.source_uid.clone()).or_default() += 1;
    }
    for atom in equation_atoms {
        *eq_counts.entry(atom.source_uid.clone()).or_default() += 1;
    }
    for atom in proof_atoms {
        *proof_counts.entry(atom.source_uid.clone()).or_default() += 1;
    }
    let mut lines = vec![
        "# Structured corpus coverage and narrative compaction plan.".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/knowledge_atoms.rs".to_string(),
        "".to_string(),
        "[structured_corpora]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        format!("source_count = {}", sources.len()),
        format!("claim_atom_count = {}", claim_atoms.len()),
        format!("equation_atom_count = {}", equation_atoms.len()),
        format!("proof_atom_count = {}", proof_atoms.len()),
        "".to_string(),
    ];
    for (idx, source) in sources.iter().enumerate() {
        let target_summary_max_lines = (source.line_count / 6).clamp(8, 48);
        lines.push("[[source]]".to_string());
        lines.push(format!("id = {}", esc(&format!("SCP-{:04}", idx + 1))));
        lines.push(format!("source_uid = {}", esc(&source.source_uid)));
        lines.push(format!("source_group = {}", esc(&source.source_group)));
        lines.push(format!(
            "source_registry = {}",
            esc(&source.source_registry)
        ));
        lines.push(format!("source_path = {}", esc(&source.source_path)));
        lines.push(format!(
            "derivation_input = {}",
            esc(&source.derivation_input)
        ));
        lines.push(format!(
            "origin_path_state = {}",
            esc(&source.origin_path_state)
        ));
        lines.push(format!(
            "canonical_body_sha256 = {}",
            esc(&source.canonical_body_sha256)
        ));
        lines.push(format!("title = {}", esc(&source.title)));
        lines.push(format!("line_count = {}", source.line_count));
        lines.push(format!(
            "claim_atom_count = {}",
            claim_counts.get(&source.source_uid).copied().unwrap_or(0)
        ));
        lines.push(format!(
            "equation_atom_count = {}",
            eq_counts.get(&source.source_uid).copied().unwrap_or(0)
        ));
        lines.push(format!(
            "proof_atom_count = {}",
            proof_counts.get(&source.source_uid).copied().unwrap_or(0)
        ));
        lines.push("narrative_compaction_recommended = true".to_string());
        lines.push("reduction_stage = \"structured_atoms_extracted\"".to_string());
        lines.push(format!(
            "target_summary_max_lines = {}",
            target_summary_max_lines
        ));
        lines.push("next_step = \"replace long body_markdown with structured summary overlays driven by claim/equation/proof atoms\"".to_string());
        lines.push(String::new());
    }
    lines.join("\n")
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

fn verify_source_provenance(repo_root: &Path, row: &Value) -> Result<()> {
    let source_uid = table_str(row, "source_uid");
    let source_path = table_str(row, "source_path");
    let source_registry = table_str(row, "source_registry");
    let derivation_input = table_str(row, "derivation_input");
    let origin_path_state = table_str(row, "origin_path_state");
    let expected_hash = table_str(row, "canonical_body_sha256");
    if !is_sha256_hex(expected_hash) {
        bail!("source {source_uid} has invalid canonical_body_sha256");
    }

    match derivation_input {
        "working_tree_markdown" => {
            if origin_path_state != "working_tree_path_present" {
                bail!(
                    "source {source_uid} uses working_tree_markdown with origin_path_state={origin_path_state}"
                );
            }
            let path = repo_root.join(source_path);
            let body = fs::read_to_string(&path)
                .with_context(|| format!("read working-tree source {}", path.display()))?;
            if sha256_hex(&body) != expected_hash {
                bail!(
                    "source {source_uid} working-tree body hash differs from canonical_body_sha256"
                );
            }
        }
        "control_plane_claim_rows" => {
            if origin_path_state != "not_applicable" {
                bail!(
                    "source {source_uid} uses control_plane_claim_rows with origin_path_state={origin_path_state}"
                );
            }
            let db_path = repo_root.join(source_registry);
            let store = ProvenanceStore::open(&db_path)
                .with_context(|| format!("open control-plane source {}", db_path.display()))?;
            let claims = store
                .list_claims()
                .with_context(|| format!("load control-plane claims from {}", db_path.display()))?;
            if sha256_hex(&render_claim_matrix_from_control_plane(&claims)) != expected_hash {
                bail!(
                    "source {source_uid} control-plane render differs from canonical_body_sha256"
                );
            }
        }
        "registry_body_markdown" => {
            verify_registry_origin_path(repo_root, source_uid, source_path, origin_path_state)?;
            let body = registry_document_body(repo_root, source_registry, source_uid)?;
            if sha256_hex(&body) != expected_hash {
                bail!("source {source_uid} registry body differs from canonical_body_sha256");
            }
        }
        "registry_document_payload" => {
            verify_registry_origin_path(repo_root, source_uid, source_path, origin_path_state)?;
            let body = registry_document_payload_body(repo_root, source_registry, source_uid)?;
            if sha256_hex(&body) != expected_hash {
                bail!(
                    "source {source_uid} registry document payload differs from canonical_body_sha256"
                );
            }
        }
        _ => bail!("source {source_uid} has unsupported derivation_input={derivation_input}"),
    }
    Ok(())
}

fn verify_registry_origin_path(
    repo_root: &Path,
    source_uid: &str,
    source_path: &str,
    origin_path_state: &str,
) -> Result<()> {
    let path_exists = repo_root.join(source_path).is_file();
    if origin_path_state_is_consistent(origin_path_state, path_exists) {
        return Ok(());
    }
    match origin_path_state {
        "working_tree_path_present" => {
            bail!(
                "source {source_uid} records a present working-tree origin path that is absent: {source_path}"
            )
        }
        "origin_path_absent" => {
            bail!(
                "source {source_uid} records an absent origin path that is present: {source_path}"
            )
        }
        _ => bail!("source {source_uid} has invalid origin_path_state={origin_path_state}"),
    }
}

fn origin_path_state_is_consistent(origin_path_state: &str, path_exists: bool) -> bool {
    matches!(
        (origin_path_state, path_exists),
        ("working_tree_path_present", true) | ("origin_path_absent", false)
    )
}

fn registry_document_body(
    repo_root: &Path,
    source_registry: &str,
    source_uid: &str,
) -> Result<String> {
    let registry = load_toml(&repo_root.join(source_registry))?;
    let row = table_array(&registry, "document")
        .iter()
        .find(|row| table_str(row, "id") == source_uid)
        .with_context(|| format!("find source {source_uid} in {source_registry}"))?;
    Ok(table_str(row, "body_markdown").to_string())
}

fn registry_document_payload_body(
    repo_root: &Path,
    source_registry: &str,
    source_uid: &str,
) -> Result<String> {
    let registry = load_toml(&repo_root.join(source_registry))?;
    let document = registry
        .get("document")
        .context("registry document payload missing [document] table")?;
    if table_str(document, "id") != source_uid {
        bail!("registry document payload source id differs from {source_uid}");
    }
    Ok(table_str(document, "content_markdown").to_string())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn registry_origin_path_state(repo_root: &Path, source_path: &str) -> String {
    if repo_root.join(source_path).is_file() {
        "working_tree_path_present".to_string()
    } else {
        "origin_path_absent".to_string()
    }
}

fn sha256_hex(text: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(text.as_bytes());
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn unique_ids(rows: &[Value], key: &str, failures: &mut Vec<String>, label: &str) {
    let values: Vec<String> = rows
        .iter()
        .map(|row| table_str(row, key).to_string())
        .collect();
    let unique: BTreeSet<String> = values.iter().cloned().collect();
    if values.len() != unique.len() {
        failures.push(format!("{label}: duplicate {key} values detected."));
    }
}

fn write_ascii(path: &Path, text: &str) -> Result<()> {
    assert_ascii_text(text, &path.display().to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

fn assert_ascii_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    assert_ascii_text(&text, &path.display().to_string())
}

fn assert_ascii_text(text: &str, context: &str) -> Result<()> {
    let bad: BTreeSet<char> = text.chars().filter(|ch| (*ch as u32) > 127).collect();
    if !bad.is_empty() {
        let sample: String = bad.iter().take(20).copied().collect();
        bail!("ERROR: Non-ASCII output in {context}: {sample:?}");
    }
    Ok(())
}

fn ascii_sanitize(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        match ch {
            '\u{2018}' | '\u{2019}' => out.push('\''),
            '\u{201C}' | '\u{201D}' => out.push('"'),
            '\u{2013}' | '\u{2014}' => out.push('-'),
            '\u{2026}' => out.push_str("..."),
            '\u{00A0}' => out.push(' '),
            '\n' | '\r' | '\t' => out.push(ch),
            _ if (ch as u32) < 32 => out.push(' '),
            _ if (ch as u32) <= 127 => out.push(ch),
            _ => out.push_str(&format!("<U+{:04X}>", ch as u32)),
        }
    }
    out
}

fn collapse_ws(text: &str) -> String {
    ascii_sanitize(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn title_from_markdown(path: &str, body: &str) -> String {
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("# ") {
            return collapse_ws(rest);
        }
    }
    Path::new(path)
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or(path)
        .to_string()
}

fn line_count(body: &str) -> usize {
    body.matches('\n').count() + usize::from(!body.is_empty())
}

fn sorted_unique_regex(text: &str, regex: &Regex) -> Vec<String> {
    let mut values = BTreeSet::new();
    for hit in regex.find_iter(text) {
        values.insert(hit.as_str().to_string());
    }
    values.into_iter().collect()
}

fn extract_bold_token(text: &str) -> Option<String> {
    let start = text.find("**")?;
    let end = text[start + 2..].find("**")?;
    Some(text[start + 2..start + 2 + end].to_string())
}

fn classify_equation(expr: &str) -> String {
    if expr.contains("->") {
        return "mapping_relation".to_string();
    }
    if ["d/d", "partial", "nabla", "Delta"]
        .iter()
        .any(|token| expr.contains(token))
    {
        return "differential_relation".to_string();
    }
    if ["<=", ">=", "!=", "<", ">"]
        .iter()
        .any(|token| expr.contains(token))
    {
        return "inequality_or_constraint".to_string();
    }
    "algebraic_relation".to_string()
}

fn parse_relation(expr: &str) -> (String, String, String) {
    for token in ["<=", ">=", "!=", "->", "="] {
        if let Some((lhs, rhs)) = expr.split_once(token) {
            let lhs_clean = collapse_ws(lhs);
            let rhs_clean = collapse_ws(rhs);
            if !lhs_clean.is_empty() {
                return (token.to_string(), lhs_clean, rhs_clean);
            }
        }
    }
    ("implicit".to_string(), collapse_ws(expr), String::new())
}

fn extract_symbol_roles(expr: &str) -> Result<(Vec<String>, Vec<String>)> {
    let identifier_re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b")?;
    let number_re = Regex::new(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")?;
    Ok((
        sorted_unique_regex(expr, &identifier_re),
        sorted_unique_regex(expr, &number_re),
    ))
}

fn infer_domain_hint(doc: &SourceDoc, section_title: &str, expr: &str) -> String {
    let lowered = format!("{} {section_title} {expr}", doc.source_path).to_lowercase();
    if ["quantum", "schrodinger", "tensor", "mera", "chern"]
        .iter()
        .any(|token| lowered.contains(token))
    {
        return "quantum".to_string();
    }
    if ["gr", "kerr", "schwarzschild", "gravastar", "geodesic"]
        .iter()
        .any(|token| lowered.contains(token))
    {
        return "general_relativity".to_string();
    }
    if ["cosmo", "hubble", "bao", "pantheon", "flrw"]
        .iter()
        .any(|token| lowered.contains(token))
    {
        return "cosmology".to_string();
    }
    if ["material", "optic", "grin", "metamaterial", "ema"]
        .iter()
        .any(|token| lowered.contains(token))
    {
        return "materials_optics".to_string();
    }
    if ["ultrametric", "bootstrap", "frechet", "stat"]
        .iter()
        .any(|token| lowered.contains(token))
    {
        return "statistics".to_string();
    }
    if ["cayley", "clifford", "jordan", "algebra", "boxkite"]
        .iter()
        .any(|token| lowered.contains(token))
    {
        return "algebra".to_string();
    }
    "cross_domain".to_string()
}

fn classify_proof_section(title: &str, body: &str) -> String {
    let lowered_title = title.to_lowercase();
    if lowered_title.contains("theorem") {
        return "theorem".to_string();
    }
    if lowered_title.contains("lemma") {
        return "lemma".to_string();
    }
    if lowered_title.contains("corollary") {
        return "corollary".to_string();
    }
    if lowered_title.contains("axiom") {
        return "axiom".to_string();
    }
    if lowered_title.contains("proof") {
        return "proof".to_string();
    }
    if lowered_title.contains("derivation") {
        return "derivation".to_string();
    }
    let lowered_body = body.to_lowercase();
    if lowered_title.contains("hypothesis")
        || lowered_body.contains("**h0**")
        || lowered_body.contains("**h1**")
    {
        return "hypothesis_block".to_string();
    }
    "argument_section".to_string()
}

fn truncate(text: &str, max_chars: usize) -> String {
    let out = collapse_ws(text);
    if out.len() <= max_chars {
        out
    } else {
        format!("{}...", &out[..max_chars - 3])
    }
}

fn filter_lines(lines: &[String], markers: &[&str], limit: usize) -> Vec<String> {
    lines
        .iter()
        .filter(|line| {
            let lowered = line.to_lowercase();
            markers.iter().any(|marker| lowered.contains(marker))
        })
        .take(limit)
        .cloned()
        .collect()
}

fn esc(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn render_list(items: &[String]) -> String {
    if items.is_empty() {
        return "[]".to_string();
    }
    format!(
        "[{}]",
        items
            .iter()
            .map(|item| esc(item))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_source(body: &str) -> SourceDoc {
        SourceDoc {
            source_uid: "TEST".to_string(),
            source_group: "test".to_string(),
            source_registry: "registry/test.toml".to_string(),
            source_path: "docs/test.md".to_string(),
            derivation_input: "registry_body_markdown".to_string(),
            origin_path_state: "origin_path_absent".to_string(),
            canonical_body_sha256: sha256_hex(body),
            title: "Test".to_string(),
            line_count: line_count(body),
            body: body.to_string(),
        }
    }

    #[test]
    fn retained_body_hash_is_valid_sha256() {
        let hash = sha256_hex("canonical registry body");
        assert!(is_sha256_hex(&hash));
        assert!(!is_sha256_hex("not-a-sha256"));
    }

    #[test]
    fn origin_path_state_records_presence_without_hiding_absence() {
        assert!(origin_path_state_is_consistent(
            "working_tree_path_present",
            true
        ));
        assert!(origin_path_state_is_consistent("origin_path_absent", false));
        assert!(!origin_path_state_is_consistent(
            "working_tree_path_present",
            false
        ));
        assert!(!origin_path_state_is_consistent("origin_path_absent", true));
    }

    #[test]
    fn generated_html_metadata_does_not_become_an_equation_atom() {
        let source = test_source("<!-- Source of truth: a -> b -->\n$x = y$");
        let atoms = extract_equations_from_source(&source).expect("extract equations");
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].expression, "x = y");
    }
}
