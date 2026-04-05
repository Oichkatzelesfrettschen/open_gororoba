use anyhow::{Context, Result, bail};
use clap::Parser;
use provenance_store::ProvenanceStore;
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
    name = "evidence-provenance",
    about = "Build or verify canonical evidence-provenance lane registries"
)]
struct Args {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value_t = false)]
    verify: bool,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(long, default_value = "registry/knowledge/derivation_steps.toml")]
    derivation_out: PathBuf,
    #[arg(long, default_value = "registry/bibliography_normalized.toml")]
    bibliography_out: PathBuf,
    #[arg(long, default_value = "registry/provenance_sources.toml")]
    provenance_out: PathBuf,
    #[arg(long, default_value = "registry/narrative_paragraph_atoms.toml")]
    paragraph_out: PathBuf,
}

#[derive(Debug, Clone)]
struct DerivationStep {
    id: String,
    skeleton_id: String,
    skeleton_kind: String,
    source_path: String,
    source_uid: String,
    claim_id: String,
    claim_refs: Vec<String>,
    step_index: usize,
    step_kind: String,
    text: String,
    text_sha256: String,
    equation_refs: Vec<String>,
    symbol_refs: Vec<String>,
    numeric_constants: Vec<String>,
    key_tokens: Vec<String>,
    depends_on_step_ids: Vec<String>,
    line_start: i64,
    line_end: i64,
}

#[derive(Debug, Clone)]
struct BibliographyNormalized {
    id: String,
    source_entry_id: String,
    order_index: i64,
    group: String,
    section: String,
    authors: Vec<String>,
    author_count: usize,
    publication_year: i64,
    title: String,
    venue: String,
    arxiv_id: String,
    doi_list: Vec<String>,
    url_list: Vec<String>,
    document_type: String,
    evidence_relevance_score: usize,
    claim_refs: Vec<String>,
    parse_warnings: Vec<String>,
    raw_citation: String,
    raw_notes: Vec<String>,
    source_line: i64,
}

#[derive(Debug, Clone)]
struct ProvenanceRecord {
    id: String,
    document_id: String,
    source_markdown: String,
    source_kind: String,
    source_ref: String,
    sha256: String,
    retrieved_date: String,
    claim_refs: Vec<String>,
    authority_level: String,
    verification_level: String,
    content_kind: String,
    notes: String,
}

#[derive(Debug, Clone)]
struct NarrativeParagraph {
    id: String,
    source_registry: String,
    document_id: String,
    source_markdown: String,
    paragraph_index: usize,
    line_start: usize,
    line_end: usize,
    paragraph_kind: String,
    text: String,
    text_sha256: String,
    claim_refs: Vec<String>,
    equation_refs: Vec<String>,
    symbol_refs: Vec<String>,
    numeric_constants: Vec<String>,
    key_tokens: Vec<String>,
}

#[derive(Debug, Default, Clone)]
struct DerivationMeta {
    step_count: usize,
    skeleton_count: usize,
    skeleton_with_steps_count: usize,
    claim_linked_step_count: usize,
    max_steps_per_skeleton: usize,
}

#[derive(Debug, Default, Clone)]
struct BibliographyMeta {
    entry_count: usize,
    parse_warning_count: usize,
}

#[derive(Debug, Default, Clone)]
struct ProvenanceMeta {
    document_count: usize,
    record_count: usize,
    url_record_count: usize,
    path_record_count: usize,
    hash_record_count: usize,
}

#[derive(Debug, Default, Clone)]
struct NarrativeParagraphMeta {
    paragraph_count: usize,
    document_count: usize,
    docs_root_document_count: usize,
    research_document_count: usize,
    external_sources_document_count: usize,
    artifact_document_count: usize,
    docs_root_paragraph_count: usize,
    research_paragraph_count: usize,
    external_sources_paragraph_count: usize,
    artifact_paragraph_count: usize,
}

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "with",
];

fn main() -> Result<()> {
    let args = Args::parse();
    let repo_root = args.repo_root.canonicalize().context("resolve repo root")?;
    if args.verify {
        return verify_evidence_provenance(&repo_root, &args);
    }
    build_evidence_provenance(&repo_root, &args)
}

fn build_evidence_provenance(repo_root: &Path, args: &Args) -> Result<()> {
    let proof_rows = table_array(
        &load_toml(&repo_root.join("registry/knowledge/proof_skeletons.toml"))?,
        "skeleton",
    )
    .to_vec();
    let bibliography_entries = table_array(
        &load_toml(&repo_root.join("registry/bibliography.toml"))?,
        "entry",
    )
    .to_vec();
    let external_docs = table_array(
        &load_toml(&repo_root.join("registry/external_sources.toml"))?,
        "document",
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
    let artifact_rows = table_array(
        &load_toml(&repo_root.join("registry/data_artifact_narratives.toml"))?,
        "document",
    )
    .to_vec();

    let (derivation_rows, derivation_meta) = build_derivation_steps(&proof_rows)?;
    let (bibliography_rows, bibliography_meta) =
        build_bibliography_normalized(&bibliography_entries)?;
    let (provenance_rows, provenance_meta) = build_provenance_sources(&external_docs)?;
    let (paragraph_rows, paragraph_meta) = build_narrative_paragraph_atoms(
        &docs_root_rows,
        &research_rows,
        &external_docs,
        &artifact_rows,
    )?;

    write_ascii(
        &repo_root.join(&args.derivation_out),
        &render_derivation_steps(&derivation_rows, &derivation_meta),
    )?;
    write_ascii(
        &repo_root.join(&args.bibliography_out),
        &render_bibliography_normalized(&bibliography_rows, &bibliography_meta),
    )?;
    write_ascii(
        &repo_root.join(&args.provenance_out),
        &render_provenance_sources(&provenance_rows, &provenance_meta),
    )?;
    write_ascii(
        &repo_root.join(&args.paragraph_out),
        &render_narrative_paragraph_atoms(&paragraph_rows, &paragraph_meta),
    )?;

    println!(
        "Wrote canonical evidence-provenance registries: derivations={} bibliography={} provenance={} paragraphs={}",
        derivation_rows.len(),
        bibliography_rows.len(),
        provenance_rows.len(),
        paragraph_rows.len()
    );
    Ok(())
}

fn verify_evidence_provenance(repo_root: &Path, args: &Args) -> Result<()> {
    let required = [
        repo_root.join("registry/knowledge/proof_skeletons.toml"),
        repo_root.join("registry/bibliography.toml"),
        repo_root.join("registry/external_sources.toml"),
        repo_root.join("registry/docs_root_narratives.toml"),
        repo_root.join("registry/research_narratives.toml"),
        repo_root.join("registry/data_artifact_narratives.toml"),
        repo_root.join(&args.derivation_out),
        repo_root.join(&args.bibliography_out),
        repo_root.join(&args.provenance_out),
        repo_root.join(&args.paragraph_out),
    ];
    for path in &required {
        if !path.exists() {
            println!("ERROR: missing required registry: {}", path.display());
            std::process::exit(1);
        }
    }
    for path in [
        repo_root.join(&args.derivation_out),
        repo_root.join(&args.bibliography_out),
        repo_root.join(&args.provenance_out),
        repo_root.join(&args.paragraph_out),
    ] {
        assert_ascii_file(&path)?;
    }

    let claim_ids = load_claim_ids(repo_root, &args.canonical_db)?;
    let proof_raw = load_toml(&repo_root.join("registry/knowledge/proof_skeletons.toml"))?;
    let bib_raw = load_toml(&repo_root.join("registry/bibliography.toml"))?;
    let external_raw = load_toml(&repo_root.join("registry/external_sources.toml"))?;
    let docs_root_raw = load_toml(&repo_root.join("registry/docs_root_narratives.toml"))?;
    let research_raw = load_toml(&repo_root.join("registry/research_narratives.toml"))?;
    let artifact_raw = load_toml(&repo_root.join("registry/data_artifact_narratives.toml"))?;
    let derivation_raw = load_toml(&repo_root.join(&args.derivation_out))?;
    let bib_norm_raw = load_toml(&repo_root.join(&args.bibliography_out))?;
    let provenance_raw = load_toml(&repo_root.join(&args.provenance_out))?;
    let paragraph_raw = load_toml(&repo_root.join(&args.paragraph_out))?;

    let mut failures = Vec::new();

    let proof_rows = table_array(&proof_raw, "skeleton");
    let proof_ids = proof_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .collect::<BTreeSet<_>>();

    let derivation_rows = table_array(&derivation_raw, "step");
    if table_int_in(&derivation_raw, "knowledge_derivation_steps", "step_count")
        != derivation_rows.len() as i64
    {
        failures.push("derivation step_count metadata mismatch".to_string());
    }
    if table_int_in(
        &derivation_raw,
        "knowledge_derivation_steps",
        "skeleton_count",
    ) != proof_rows.len() as i64
    {
        failures.push("derivation skeleton_count metadata mismatch".to_string());
    }
    let mut seen_step_ids = BTreeSet::new();
    let mut by_skeleton = BTreeMap::<String, Vec<&Value>>::new();
    for row in derivation_rows {
        let step_id = table_str(row, "id").to_string();
        if !seen_step_ids.insert(step_id.clone()) {
            failures.push(format!("duplicate derivation step id: {}", step_id));
            break;
        }
        let skeleton_id = table_str(row, "skeleton_id").to_string();
        if !proof_ids.contains(&skeleton_id) {
            failures.push(format!(
                "derivation step references unknown skeleton_id: {}",
                skeleton_id
            ));
        }
        let claim_id = table_str(row, "claim_id").to_string();
        if !claim_id.is_empty() && !claim_ids.contains(&claim_id) {
            failures.push(format!(
                "derivation step references unknown claim_id: {} -> {}",
                step_id, claim_id
            ));
        }
        for claim_ref in string_list(row, "claim_refs") {
            if !claim_ids.contains(&claim_ref) {
                failures.push(format!(
                    "derivation step contains unknown claim ref: {} -> {}",
                    step_id, claim_ref
                ));
            }
        }
        by_skeleton.entry(skeleton_id).or_default().push(row);
    }
    for (skeleton_id, rows) in by_skeleton {
        let mut rows = rows;
        rows.sort_by_key(|row| table_int(row, "step_index"));
        let mut prev_id = String::new();
        for (expected, row) in (1i64..).zip(rows) {
            let idx = table_int(row, "step_index");
            if idx != expected {
                failures.push(format!(
                    "derivation step_index gap in {}: got {} expected {}",
                    skeleton_id, idx, expected
                ));
                break;
            }
            let deps = string_list(row, "depends_on_step_ids");
            if expected == 1 && !deps.is_empty() {
                failures.push(format!(
                    "first derivation step should have no depends_on: {}",
                    table_str(row, "id")
                ));
            }
            if expected > 1 && deps != vec![prev_id.clone()] {
                failures.push(format!(
                    "derivation dependency mismatch: {} expected [{}] got {:?}",
                    table_str(row, "id"),
                    prev_id,
                    deps
                ));
            }
            prev_id = table_str(row, "id").to_string();
        }
    }

    let bib_entries = table_array(&bib_raw, "entry");
    let bib_norm_entries = table_array(&bib_norm_raw, "entry");
    if table_int_in(&bib_norm_raw, "bibliography_normalized", "entry_count")
        != bib_norm_entries.len() as i64
    {
        failures.push("bibliography_normalized entry_count metadata mismatch".to_string());
    }
    if bib_norm_entries.len() != bib_entries.len() {
        failures.push(format!(
            "bibliography_normalized count mismatch: {} vs source {}",
            bib_norm_entries.len(),
            bib_entries.len()
        ));
    }
    let source_bib_ids = bib_entries
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .collect::<BTreeSet<_>>();
    let mut seen_norm_source = BTreeSet::new();
    for row in bib_norm_entries {
        let source_id = table_str(row, "source_entry_id").to_string();
        if !source_bib_ids.contains(&source_id) {
            failures.push(format!(
                "bibliography_normalized unknown source_entry_id: {}",
                source_id
            ));
        }
        if !seen_norm_source.insert(source_id.clone()) {
            failures.push(format!(
                "bibliography_normalized duplicate source_entry_id: {}",
                source_id
            ));
        }
        let year = table_int(row, "publication_year");
        if year != 0 && !(1500..=2100).contains(&year) {
            failures.push(format!(
                "bibliography_normalized invalid publication_year: {} -> {}",
                source_id, year
            ));
        }
        if table_str(row, "document_type") == "arxiv_preprint"
            && table_str(row, "arxiv_id").is_empty()
        {
            failures.push(format!(
                "bibliography_normalized arxiv_preprint missing arxiv_id: {}",
                source_id
            ));
        }
        for claim_ref in string_list(row, "claim_refs") {
            if !claim_ids.contains(&claim_ref) {
                failures.push(format!(
                    "bibliography_normalized unknown claim_ref: {} -> {}",
                    source_id, claim_ref
                ));
            }
        }
    }

    let external_docs = table_array(&external_raw, "document");
    let external_doc_ids = external_docs
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .collect::<BTreeSet<_>>();
    let provenance_rows = table_array(&provenance_raw, "record");
    if table_int_in(&provenance_raw, "provenance_sources", "record_count")
        != provenance_rows.len() as i64
    {
        failures.push("provenance_sources record_count metadata mismatch".to_string());
    }
    if table_int_in(&provenance_raw, "provenance_sources", "document_count")
        != external_doc_ids.len() as i64
    {
        failures.push("provenance_sources document_count metadata mismatch".to_string());
    }
    let sha_re = Regex::new(r"^[a-f0-9]{64}$")?;
    let mut kind_counts = BTreeMap::<String, usize>::new();
    let mut by_doc = BTreeMap::<String, usize>::new();
    for row in provenance_rows {
        let kind = table_str(row, "source_kind").to_string();
        *kind_counts.entry(kind.clone()).or_insert(0) += 1;
        let doc_id = table_str(row, "document_id").to_string();
        *by_doc.entry(doc_id.clone()).or_insert(0) += 1;
        if !external_doc_ids.contains(&doc_id) {
            failures.push(format!(
                "provenance_sources unknown document_id: {}",
                doc_id
            ));
        }
        if !["url", "path", "sha256"].contains(&kind.as_str()) {
            failures.push(format!("provenance_sources invalid source_kind: {}", kind));
        }
        if kind == "sha256" {
            let digest = table_str(row, "sha256");
            if !sha_re.is_match(digest) {
                failures.push(format!(
                    "provenance_sources invalid sha256 digest: {}",
                    table_str(row, "id")
                ));
            }
            if table_str(row, "source_ref") != digest {
                failures.push(format!(
                    "provenance_sources sha256 source_ref mismatch: {}",
                    table_str(row, "id")
                ));
            }
        }
        for claim_ref in string_list(row, "claim_refs") {
            if !claim_ids.contains(&claim_ref) {
                failures.push(format!(
                    "provenance_sources unknown claim_ref: {} -> {}",
                    table_str(row, "id"),
                    claim_ref
                ));
            }
        }
    }
    if table_int_in(&provenance_raw, "provenance_sources", "url_record_count")
        != kind_counts.get("url").copied().unwrap_or(0) as i64
    {
        failures.push("provenance_sources url_record_count metadata mismatch".to_string());
    }
    if table_int_in(&provenance_raw, "provenance_sources", "path_record_count")
        != kind_counts.get("path").copied().unwrap_or(0) as i64
    {
        failures.push("provenance_sources path_record_count metadata mismatch".to_string());
    }
    if table_int_in(&provenance_raw, "provenance_sources", "hash_record_count")
        != kind_counts.get("sha256").copied().unwrap_or(0) as i64
    {
        failures.push("provenance_sources hash_record_count metadata mismatch".to_string());
    }
    for doc_id in external_doc_ids {
        if by_doc.get(&doc_id).copied().unwrap_or(0) == 0 {
            failures.push(format!(
                "provenance_sources missing records for external source doc: {}",
                doc_id
            ));
        }
    }

    let paragraph_rows = table_array(&paragraph_raw, "paragraph");
    if table_int_in(
        &paragraph_raw,
        "narrative_paragraph_atoms",
        "paragraph_count",
    ) != paragraph_rows.len() as i64
    {
        failures.push("narrative_paragraph_atoms paragraph_count metadata mismatch".to_string());
    }
    let allowed_source_registries = BTreeSet::from([
        "registry/docs_root_narratives.toml".to_string(),
        "registry/research_narratives.toml".to_string(),
        "registry/external_sources.toml".to_string(),
        "registry/data_artifact_narratives.toml".to_string(),
    ]);
    let mut seen_paragraph_ids = BTreeSet::new();
    let mut paragraph_by_doc = BTreeMap::<(String, String), Vec<&Value>>::new();
    let mut paragraph_doc_coverage = BTreeMap::<String, BTreeSet<(String, String)>>::new();
    for row in paragraph_rows {
        let pid = table_str(row, "id").to_string();
        if !seen_paragraph_ids.insert(pid.clone()) {
            failures.push(format!("duplicate narrative paragraph id: {}", pid));
            break;
        }
        let source_registry = table_str(row, "source_registry").to_string();
        if !allowed_source_registries.contains(&source_registry) {
            failures.push(format!(
                "narrative paragraph invalid source_registry: {}",
                source_registry
            ));
        }
        let doc_id = table_str(row, "document_id").to_string();
        let source_markdown = table_str(row, "source_markdown").to_string();
        paragraph_by_doc
            .entry((source_registry.clone(), doc_id.clone()))
            .or_default()
            .push(row);
        paragraph_doc_coverage
            .entry(source_registry)
            .or_default()
            .insert((doc_id, source_markdown));
        let line_start = table_int(row, "line_start");
        let line_end = table_int(row, "line_end");
        if line_start <= 0 || line_end <= 0 || line_end < line_start {
            failures.push(format!("narrative paragraph invalid line span: {}", pid));
        }
        for claim_ref in string_list(row, "claim_refs") {
            if !claim_ids.contains(&claim_ref) {
                failures.push(format!(
                    "narrative paragraph unknown claim_ref: {} -> {}",
                    pid, claim_ref
                ));
            }
        }
    }
    for ((source_registry, doc_id), rows) in paragraph_by_doc {
        let mut rows = rows;
        rows.sort_by_key(|row| table_int(row, "paragraph_index"));
        for (expected, row) in (1i64..).zip(rows) {
            let idx = table_int(row, "paragraph_index");
            if idx != expected {
                failures.push(format!(
                    "narrative paragraph_index gap for {}::{} got {} expected {}",
                    source_registry, doc_id, idx, expected
                ));
                break;
            }
        }
    }
    let docs_root_expected = nonempty_body_doc_ids(table_array(&docs_root_raw, "document"));
    let research_expected = nonempty_body_doc_ids(table_array(&research_raw, "document"));
    let external_expected = nonempty_body_doc_ids(table_array(&external_raw, "document"));
    let artifact_expected = nonempty_body_doc_ids(table_array(&artifact_raw, "document"));
    let expected_map = BTreeMap::from([
        (
            "registry/docs_root_narratives.toml".to_string(),
            docs_root_expected,
        ),
        (
            "registry/research_narratives.toml".to_string(),
            research_expected,
        ),
        (
            "registry/external_sources.toml".to_string(),
            external_expected,
        ),
        (
            "registry/data_artifact_narratives.toml".to_string(),
            artifact_expected,
        ),
    ]);
    for (source_registry, expected_docs) in &expected_map {
        let observed_docs = paragraph_doc_coverage
            .get(source_registry)
            .cloned()
            .unwrap_or_default();
        let missing = expected_docs
            .difference(&observed_docs)
            .cloned()
            .collect::<Vec<_>>();
        let extra = observed_docs
            .difference(expected_docs)
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            failures.push(format!(
                "narrative paragraph coverage missing docs in {}: {}",
                source_registry,
                missing.len()
            ));
        }
        if !extra.is_empty() {
            failures.push(format!(
                "narrative paragraph coverage extra docs in {}: {}",
                source_registry,
                extra.len()
            ));
        }
    }
    let expected_document_count = expected_map.values().map(BTreeSet::len).sum::<usize>() as i64;
    if table_int_in(
        &paragraph_raw,
        "narrative_paragraph_atoms",
        "document_count",
    ) != expected_document_count
    {
        failures.push("narrative_paragraph_atoms document_count metadata mismatch".to_string());
    }

    if !failures.is_empty() {
        println!("ERROR: canonical evidence-provenance registry verification failed.");
        for item in failures.iter().take(300) {
            println!("- {}", item);
        }
        if failures.len() > 300 {
            println!("- ... and {} more failures", failures.len() - 300);
        }
        std::process::exit(1);
    }

    println!(
        "OK: canonical evidence-provenance registries verified. derivations={} bibliography_normalized={} provenance_records={} narrative_paragraphs={}",
        table_array(&derivation_raw, "step").len(),
        table_array(&bib_norm_raw, "entry").len(),
        table_array(&provenance_raw, "record").len(),
        table_array(&paragraph_raw, "paragraph").len()
    );
    Ok(())
}

fn load_claim_ids(repo_root: &Path, canonical_db: &Path) -> Result<BTreeSet<String>> {
    let db_path = repo_root.join(canonical_db);
    if db_path.exists() {
        let store = ProvenanceStore::open(&db_path)
            .with_context(|| format!("open canonical db {}", db_path.display()))?;
        let claims = store
            .list_claims()
            .with_context(|| format!("load claims from {}", db_path.display()))?;
        if !claims.is_empty() {
            return Ok(claims.into_iter().map(|row| row.id).collect());
        }
    }

    let claims_raw = load_toml(&repo_root.join("registry/claims.toml"))?;
    Ok(claim_set(table_array(&claims_raw, "claim")))
}

fn build_derivation_steps(proof_rows: &[Value]) -> Result<(Vec<DerivationStep>, DerivationMeta)> {
    let equation_re = Regex::new(r"\bEQA2?-\d{4,5}\b")?;
    let symbol_re = Regex::new(r"\bSYM-\d{4}\b")?;
    let number_re = Regex::new(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")?;
    let identifier_re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b")?;

    let mut steps = Vec::new();
    let mut seq = 0usize;
    let mut max_per_skeleton = 0usize;
    let mut linked_claim_steps = 0usize;
    let mut skeleton_with_steps = 0usize;
    for row in proof_rows
        .iter()
        .sorted_by_key(|row| table_str(row, "id").to_string())
    {
        let skeleton_id = collapse(table_str(&row, "id"));
        if skeleton_id.is_empty() {
            continue;
        }
        let claim_id = collapse(table_str(&row, "claim_id"));
        let claim_refs = normalize_claim_refs(string_list(&row, "claim_refs"), &claim_id)?;
        let mut parts = split_derivation_chunks(string_list(&row, "derivation_steps"));
        if parts.is_empty() {
            let mut fallback = Vec::new();
            fallback.extend(
                string_list(&row, "assumptions")
                    .into_iter()
                    .map(|item| collapse(&item))
                    .filter(|item| !item.is_empty()),
            );
            fallback.extend(
                string_list(&row, "obligations")
                    .into_iter()
                    .map(|item| collapse(&item))
                    .filter(|item| !item.is_empty()),
            );
            let decision = collapse(table_str(&row, "decision_rule"));
            if !decision.is_empty() {
                fallback.push(decision);
            }
            let conclusion = collapse(table_str(&row, "conclusion"));
            if !conclusion.is_empty() {
                fallback.push(conclusion);
            }
            parts = fallback;
        }
        if parts.is_empty() {
            continue;
        }
        skeleton_with_steps += 1;
        max_per_skeleton = max_per_skeleton.max(parts.len());
        let mut prev_step_id = String::new();
        for (idx, text) in parts.into_iter().enumerate() {
            seq += 1;
            let step_id = format!("DS-{:06}", seq);
            let equation_refs = sorted_unique_regex(&text, &equation_re);
            let symbol_refs = sorted_unique_regex(&text, &symbol_re);
            let number_refs = sorted_unique_regex(&text, &number_re);
            let key_tokens = identifier_re
                .find_iter(&text)
                .map(|m| m.as_str().to_string())
                .filter(|token| {
                    token.len() >= 2 && !STOPWORDS.contains(&token.to_lowercase().as_str())
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .take(20)
                .collect::<Vec<_>>();
            if !claim_refs.is_empty() {
                linked_claim_steps += 1;
            }
            steps.push(DerivationStep {
                id: step_id.clone(),
                skeleton_id: skeleton_id.clone(),
                skeleton_kind: collapse(table_str(&row, "skeleton_kind")),
                source_path: collapse(table_str(&row, "source_path")),
                source_uid: collapse(table_str(&row, "source_uid")),
                claim_id: claim_id.clone(),
                claim_refs: claim_refs.clone(),
                step_index: idx + 1,
                step_kind: classify_step(&text),
                text_sha256: sha256_hex(text.as_bytes()),
                text,
                equation_refs,
                symbol_refs,
                numeric_constants: number_refs,
                key_tokens,
                depends_on_step_ids: if prev_step_id.is_empty() {
                    Vec::new()
                } else {
                    vec![prev_step_id.clone()]
                },
                line_start: table_int(&row, "line_start"),
                line_end: table_int(&row, "line_end"),
            });
            prev_step_id = step_id;
        }
    }
    Ok((
        steps,
        DerivationMeta {
            step_count: seq,
            skeleton_count: proof_rows.len(),
            skeleton_with_steps_count: skeleton_with_steps,
            claim_linked_step_count: linked_claim_steps,
            max_steps_per_skeleton: max_per_skeleton,
        },
    ))
}

fn build_bibliography_normalized(
    entries: &[Value],
) -> Result<(Vec<BibliographyNormalized>, BibliographyMeta)> {
    let arxiv_re = Regex::new(r"\barXiv[:\s]*([A-Za-z\-]+/\d{7}|\d{4}\.\d{4,5}(?:v\d+)?)\b")?;
    let year_re = Regex::new(r"\((\d{4})\)")?;
    let claim_re = Regex::new(r"\bC-\d{3}\b")?;
    let mut out = Vec::new();
    let mut total_warnings = 0usize;
    let mut rows = entries.to_vec();
    rows.sort_by_key(|row| table_int(row, "order_index"));
    for (idx, row) in rows.iter().enumerate() {
        let citation = collapse(table_str(row, "citation_markdown"));
        let notes = string_list(row, "notes")
            .into_iter()
            .map(|item| collapse(&item))
            .filter(|item| !item.is_empty())
            .collect::<Vec<_>>();
        let urls = string_list(row, "urls")
            .into_iter()
            .filter(|item| !item.trim().is_empty())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let doi_list = string_list(row, "dois")
            .into_iter()
            .filter(|item| !item.trim().is_empty())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let authors = extract_authors(&citation)?;
        let publication_year = year_re
            .captures(&citation)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse::<i64>().ok())
            .unwrap_or(0);
        let (title, tail) = extract_title(&citation)?;
        let venue = extract_venue(&tail)?;
        let arxiv_id = arxiv_re
            .captures(&citation)
            .and_then(|caps| caps.get(1))
            .map(|m| collapse(m.as_str()))
            .unwrap_or_default();
        let mut warnings = Vec::new();
        if authors.is_empty() {
            warnings.push("missing_authors".to_string());
        }
        if publication_year == 0 {
            warnings.push("missing_year".to_string());
        }
        if title.is_empty() {
            warnings.push("missing_title".to_string());
        }
        if venue.is_empty() {
            warnings.push("missing_venue".to_string());
        }
        total_warnings += warnings.len();
        let corpus = format!("{} {}", citation, notes.join(" "));
        out.push(BibliographyNormalized {
            id: format!("BIBN-{:04}", idx + 1),
            source_entry_id: collapse(table_str(row, "id")),
            order_index: {
                let v = table_int(row, "order_index");
                if v == 0 { (idx + 1) as i64 } else { v }
            },
            group: collapse(table_str(row, "group")),
            section: collapse(table_str(row, "section")),
            author_count: authors.len(),
            authors,
            publication_year,
            title,
            venue,
            arxiv_id: arxiv_id.clone(),
            doi_list: doi_list.clone(),
            url_list: urls.clone(),
            document_type: document_type(&citation, &arxiv_id, &doi_list, &urls),
            evidence_relevance_score: relevance_score(&citation, &notes),
            claim_refs: sorted_unique_regex(&corpus, &claim_re),
            parse_warnings: warnings,
            raw_citation: citation,
            raw_notes: notes,
            source_line: table_int(row, "source_line"),
        });
    }
    Ok((
        out,
        BibliographyMeta {
            entry_count: entries.len(),
            parse_warning_count: total_warnings,
        },
    ))
}

fn build_provenance_sources(
    documents: &[Value],
) -> Result<(Vec<ProvenanceRecord>, ProvenanceMeta)> {
    let url_re = Regex::new(r"https?://[^\s)]+")?;
    let sha_re = Regex::new(r"\b([a-fA-F0-9]{64})\b")?;
    let date_re = Regex::new(r"(?:Date retrieved|retrieved)\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})")?;
    let mut rows = Vec::new();
    let mut seq = 0usize;
    let mut url_count = 0usize;
    let mut path_count = 0usize;
    let mut hash_count = 0usize;
    let mut doc_ids = BTreeSet::new();
    let mut sorted_docs = documents.to_vec();
    sorted_docs.sort_by_key(|row| table_str(row, "id").to_string());
    for doc in sorted_docs {
        let doc_id = collapse(table_str(&doc, "id"));
        if doc_id.is_empty() {
            continue;
        }
        doc_ids.insert(doc_id.clone());
        let rows_before = rows.len();
        let source_markdown = collapse(table_str(&doc, "source_markdown"));
        let body = ascii_clean(table_str(&doc, "body_markdown"));
        let claim_refs = normalize_claim_refs(string_list(&doc, "claim_refs"), "")?;
        let mut urls = string_list(&doc, "url_refs")
            .into_iter()
            .filter(|item| !item.trim().is_empty())
            .collect::<BTreeSet<_>>();
        for hit in url_re.find_iter(&body) {
            urls.insert(hit.as_str().to_string());
        }
        let mut paths = string_list(&doc, "path_refs")
            .into_iter()
            .filter(|item| !item.trim().is_empty())
            .collect::<BTreeSet<_>>();
        for path in extract_local_paths(&body)? {
            paths.insert(path);
        }
        let hashes = sha_re
            .captures_iter(&body)
            .filter_map(|caps| caps.get(1))
            .map(|m| m.as_str().to_lowercase())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let retrieved_date = date_re
            .captures_iter(&body)
            .filter_map(|caps| caps.get(1))
            .map(|m| m.as_str().to_string())
            .next()
            .unwrap_or_default();
        let authority_level = collapse(table_str(&doc, "authority_level"));
        let verification_level = collapse(table_str(&doc, "verification_level"));
        let content_kind = collapse(table_str(&doc, "content_kind"));
        let notes = collapse(table_str(&doc, "notes"));
        let sorted_urls = urls.into_iter().collect::<Vec<_>>();
        let sorted_paths = paths.into_iter().collect::<Vec<_>>();
        let path_hash_map = sorted_paths
            .iter()
            .enumerate()
            .filter_map(|(idx, path)| hashes.get(idx).map(|hash| (path.clone(), hash.clone())))
            .collect::<BTreeMap<_, _>>();
        for url in sorted_urls {
            seq += 1;
            url_count += 1;
            rows.push(ProvenanceRecord {
                id: format!("PSR-{:05}", seq),
                document_id: doc_id.clone(),
                source_markdown: source_markdown.clone(),
                source_kind: "url".to_string(),
                source_ref: collapse(&url),
                sha256: String::new(),
                retrieved_date: retrieved_date.clone(),
                claim_refs: claim_refs.clone(),
                authority_level: authority_level.clone(),
                verification_level: verification_level.clone(),
                content_kind: content_kind.clone(),
                notes: notes.clone(),
            });
        }
        for path in sorted_paths {
            seq += 1;
            path_count += 1;
            rows.push(ProvenanceRecord {
                id: format!("PSR-{:05}", seq),
                document_id: doc_id.clone(),
                source_markdown: source_markdown.clone(),
                source_kind: "path".to_string(),
                source_ref: collapse(&path),
                sha256: path_hash_map.get(&path).cloned().unwrap_or_default(),
                retrieved_date: retrieved_date.clone(),
                claim_refs: claim_refs.clone(),
                authority_level: authority_level.clone(),
                verification_level: verification_level.clone(),
                content_kind: content_kind.clone(),
                notes: notes.clone(),
            });
        }
        for digest in hashes {
            seq += 1;
            hash_count += 1;
            rows.push(ProvenanceRecord {
                id: format!("PSR-{:05}", seq),
                document_id: doc_id.clone(),
                source_markdown: source_markdown.clone(),
                source_kind: "sha256".to_string(),
                source_ref: digest.clone(),
                sha256: digest,
                retrieved_date: retrieved_date.clone(),
                claim_refs: claim_refs.clone(),
                authority_level: authority_level.clone(),
                verification_level: verification_level.clone(),
                content_kind: content_kind.clone(),
                notes: notes.clone(),
            });
        }
        if rows.len() == rows_before {
            seq += 1;
            path_count += 1;
            rows.push(ProvenanceRecord {
                id: format!("PSR-{:05}", seq),
                document_id: doc_id,
                source_markdown: source_markdown.clone(),
                source_kind: "path".to_string(),
                source_ref: source_markdown,
                sha256: String::new(),
                retrieved_date,
                claim_refs,
                authority_level,
                verification_level,
                content_kind,
                notes: if notes.is_empty() {
                    "Fallback provenance anchor for document with no extracted urls/hashes."
                        .to_string()
                } else {
                    notes
                },
            });
        }
    }
    Ok((
        rows,
        ProvenanceMeta {
            document_count: doc_ids.len(),
            record_count: seq,
            url_record_count: url_count,
            path_record_count: path_count,
            hash_record_count: hash_count,
        },
    ))
}

fn build_narrative_paragraph_atoms(
    docs_root_rows: &[Value],
    research_rows: &[Value],
    external_rows: &[Value],
    artifact_rows: &[Value],
) -> Result<(Vec<NarrativeParagraph>, NarrativeParagraphMeta)> {
    let claim_re = Regex::new(r"\bC-\d{3}\b")?;
    let equation_re = Regex::new(r"\bEQA2?-\d{4,5}\b")?;
    let symbol_re = Regex::new(r"\bSYM-\d{4}\b")?;
    let number_re = Regex::new(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")?;
    let identifier_re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b")?;
    let sources = [
        ("registry/docs_root_narratives.toml", docs_root_rows),
        ("registry/research_narratives.toml", research_rows),
        ("registry/external_sources.toml", external_rows),
        ("registry/data_artifact_narratives.toml", artifact_rows),
    ];
    let mut rows = Vec::new();
    let mut seq = 0usize;
    let mut doc_counts = BTreeMap::<String, usize>::new();
    let mut para_counts = BTreeMap::<String, usize>::new();
    for (source_registry, docs) in sources {
        let mut docs = docs.to_vec();
        docs.sort_by_key(|row| table_str(row, "id").to_string());
        for doc in docs {
            let body = table_str(&doc, "body_markdown").to_string();
            if body.trim().is_empty() {
                continue;
            }
            let doc_id = collapse(table_str(&doc, "id"));
            let source_markdown = collapse(table_str(&doc, "source_markdown"));
            let claim_refs_doc = normalize_claim_refs(string_list(&doc, "claim_refs"), "")?;
            let blocks = split_blocks(&body);
            if blocks.is_empty() {
                continue;
            }
            *doc_counts.entry(source_registry.to_string()).or_insert(0) += 1;
            for (paragraph_index, (line_start, line_end, block)) in blocks.into_iter().enumerate() {
                let text = collapse(&block);
                if text.is_empty() {
                    continue;
                }
                seq += 1;
                let mut claim_refs = claim_refs_doc.clone();
                claim_refs.extend(sorted_unique_regex(&text, &claim_re));
                claim_refs.sort();
                claim_refs.dedup();
                let equation_refs = sorted_unique_regex(&text, &equation_re);
                let symbol_refs = sorted_unique_regex(&text, &symbol_re);
                let numeric_constants = sorted_unique_regex(&text, &number_re);
                let key_tokens = identifier_re
                    .find_iter(&text)
                    .map(|m| m.as_str().to_string())
                    .filter(|token| {
                        token.len() >= 2 && !STOPWORDS.contains(&token.to_lowercase().as_str())
                    })
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .take(24)
                    .collect::<Vec<_>>();
                *para_counts.entry(source_registry.to_string()).or_insert(0) += 1;
                rows.push(NarrativeParagraph {
                    id: format!("NPA-{:06}", seq),
                    source_registry: source_registry.to_string(),
                    document_id: doc_id.clone(),
                    source_markdown: source_markdown.clone(),
                    paragraph_index: paragraph_index + 1,
                    line_start,
                    line_end,
                    paragraph_kind: paragraph_kind(&block)?,
                    text_sha256: sha256_hex(text.as_bytes()),
                    text,
                    claim_refs,
                    equation_refs,
                    symbol_refs,
                    numeric_constants,
                    key_tokens,
                });
            }
        }
    }
    Ok((
        rows,
        NarrativeParagraphMeta {
            paragraph_count: seq,
            document_count: doc_counts.values().sum(),
            docs_root_document_count: doc_counts
                .get("registry/docs_root_narratives.toml")
                .copied()
                .unwrap_or(0),
            research_document_count: doc_counts
                .get("registry/research_narratives.toml")
                .copied()
                .unwrap_or(0),
            external_sources_document_count: doc_counts
                .get("registry/external_sources.toml")
                .copied()
                .unwrap_or(0),
            artifact_document_count: doc_counts
                .get("registry/data_artifact_narratives.toml")
                .copied()
                .unwrap_or(0),
            docs_root_paragraph_count: para_counts
                .get("registry/docs_root_narratives.toml")
                .copied()
                .unwrap_or(0),
            research_paragraph_count: para_counts
                .get("registry/research_narratives.toml")
                .copied()
                .unwrap_or(0),
            external_sources_paragraph_count: para_counts
                .get("registry/external_sources.toml")
                .copied()
                .unwrap_or(0),
            artifact_paragraph_count: para_counts
                .get("registry/data_artifact_narratives.toml")
                .copied()
                .unwrap_or(0),
        },
    ))
}

fn render_derivation_steps(rows: &[DerivationStep], meta: &DerivationMeta) -> String {
    let mut lines = vec![
        "# Derivation step registry (evidence-provenance lane strict schema; legacy batch2 compatibility).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/evidence_provenance.rs.".to_string(),
        "".to_string(),
        "[knowledge_derivation_steps]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "source_registry = \"registry/knowledge/proof_skeletons.toml\"".to_string(),
        format!("step_count = {}", meta.step_count),
        format!("skeleton_count = {}", meta.skeleton_count),
        format!("skeleton_with_steps_count = {}", meta.skeleton_with_steps_count),
        format!("claim_linked_step_count = {}", meta.claim_linked_step_count),
        format!("max_steps_per_skeleton = {}", meta.max_steps_per_skeleton),
        "".to_string(),
    ];
    for row in rows {
        lines.extend([
            "[[step]]".to_string(),
            format!("id = {}", esc(&row.id)),
            format!("skeleton_id = {}", esc(&row.skeleton_id)),
            format!("skeleton_kind = {}", esc(&row.skeleton_kind)),
            format!("source_path = {}", esc(&row.source_path)),
            format!("source_uid = {}", esc(&row.source_uid)),
            format!("claim_id = {}", esc(&row.claim_id)),
            format!("claim_refs = {}", render_list(&row.claim_refs)),
            format!("step_index = {}", row.step_index),
            format!("step_kind = {}", esc(&row.step_kind)),
            format!("text = {}", esc(&row.text)),
            format!("text_sha256 = {}", esc(&row.text_sha256)),
            format!("equation_refs = {}", render_list(&row.equation_refs)),
            format!("symbol_refs = {}", render_list(&row.symbol_refs)),
            format!(
                "numeric_constants = {}",
                render_list(&row.numeric_constants)
            ),
            format!("key_tokens = {}", render_list(&row.key_tokens)),
            format!(
                "depends_on_step_ids = {}",
                render_list(&row.depends_on_step_ids)
            ),
            format!("line_start = {}", row.line_start),
            format!("line_end = {}", row.line_end),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn render_bibliography_normalized(
    rows: &[BibliographyNormalized],
    meta: &BibliographyMeta,
) -> String {
    let mut lines = vec![
        "# Normalized bibliography registry (evidence-provenance lane strict schema; legacy batch2 compatibility).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/evidence_provenance.rs.".to_string(),
        "".to_string(),
        "[bibliography_normalized]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "source_registry = \"registry/bibliography.toml\"".to_string(),
        format!("entry_count = {}", meta.entry_count),
        format!("parse_warning_count = {}", meta.parse_warning_count),
        "".to_string(),
    ];
    for row in rows {
        lines.extend([
            "[[entry]]".to_string(),
            format!("id = {}", esc(&row.id)),
            format!("source_entry_id = {}", esc(&row.source_entry_id)),
            format!("order_index = {}", row.order_index),
            format!("group = {}", esc(&row.group)),
            format!("section = {}", esc(&row.section)),
            format!("authors = {}", render_list(&row.authors)),
            format!("author_count = {}", row.author_count),
            format!("publication_year = {}", row.publication_year),
            format!("title = {}", esc(&row.title)),
            format!("venue = {}", esc(&row.venue)),
            format!("arxiv_id = {}", esc(&row.arxiv_id)),
            format!("doi_list = {}", render_list(&row.doi_list)),
            format!("url_list = {}", render_list(&row.url_list)),
            format!("document_type = {}", esc(&row.document_type)),
            format!(
                "evidence_relevance_score = {}",
                row.evidence_relevance_score
            ),
            format!("claim_refs = {}", render_list(&row.claim_refs)),
            format!("parse_warnings = {}", render_list(&row.parse_warnings)),
            format!("raw_citation = {}", esc(&row.raw_citation)),
            format!("raw_notes = {}", render_list(&row.raw_notes)),
            format!("source_line = {}", row.source_line),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn render_provenance_sources(rows: &[ProvenanceRecord], meta: &ProvenanceMeta) -> String {
    let mut lines = vec![
        "# Provenance source registry (evidence-provenance lane strict schema; legacy batch2 compatibility).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/evidence_provenance.rs.".to_string(),
        "".to_string(),
        "[provenance_sources]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "source_registry = \"registry/external_sources.toml\"".to_string(),
        format!("document_count = {}", meta.document_count),
        format!("record_count = {}", meta.record_count),
        format!("url_record_count = {}", meta.url_record_count),
        format!("path_record_count = {}", meta.path_record_count),
        format!("hash_record_count = {}", meta.hash_record_count),
        "".to_string(),
    ];
    for row in rows {
        lines.extend([
            "[[record]]".to_string(),
            format!("id = {}", esc(&row.id)),
            format!("document_id = {}", esc(&row.document_id)),
            format!("source_markdown = {}", esc(&row.source_markdown)),
            format!("source_kind = {}", esc(&row.source_kind)),
            format!("source_ref = {}", esc(&row.source_ref)),
            format!("sha256 = {}", esc(&row.sha256)),
            format!("retrieved_date = {}", esc(&row.retrieved_date)),
            format!("claim_refs = {}", render_list(&row.claim_refs)),
            format!("authority_level = {}", esc(&row.authority_level)),
            format!("verification_level = {}", esc(&row.verification_level)),
            format!("content_kind = {}", esc(&row.content_kind)),
            format!("notes = {}", esc(&row.notes)),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn render_narrative_paragraph_atoms(
    rows: &[NarrativeParagraph],
    meta: &NarrativeParagraphMeta,
) -> String {
    let mut lines = vec![
        "# Narrative paragraph atom registry (evidence-provenance lane strict schema; legacy batch2 compatibility).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/evidence_provenance.rs.".to_string(),
        "".to_string(),
        "[narrative_paragraph_atoms]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "source_registries = [\"registry/docs_root_narratives.toml\", \"registry/research_narratives.toml\", \"registry/external_sources.toml\", \"registry/data_artifact_narratives.toml\"]".to_string(),
        format!("document_count = {}", meta.document_count),
        format!("paragraph_count = {}", meta.paragraph_count),
        format!("docs_root_document_count = {}", meta.docs_root_document_count),
        format!("research_document_count = {}", meta.research_document_count),
        format!(
            "external_sources_document_count = {}",
            meta.external_sources_document_count
        ),
        format!("artifact_document_count = {}", meta.artifact_document_count),
        format!("docs_root_paragraph_count = {}", meta.docs_root_paragraph_count),
        format!("research_paragraph_count = {}", meta.research_paragraph_count),
        format!(
            "external_sources_paragraph_count = {}",
            meta.external_sources_paragraph_count
        ),
        format!("artifact_paragraph_count = {}", meta.artifact_paragraph_count),
        "".to_string(),
    ];
    for row in rows {
        lines.extend([
            "[[paragraph]]".to_string(),
            format!("id = {}", esc(&row.id)),
            format!("source_registry = {}", esc(&row.source_registry)),
            format!("document_id = {}", esc(&row.document_id)),
            format!("source_markdown = {}", esc(&row.source_markdown)),
            format!("paragraph_index = {}", row.paragraph_index),
            format!("line_start = {}", row.line_start),
            format!("line_end = {}", row.line_end),
            format!("paragraph_kind = {}", esc(&row.paragraph_kind)),
            format!("text = {}", esc(&row.text)),
            format!("text_sha256 = {}", esc(&row.text_sha256)),
            format!("claim_refs = {}", render_list(&row.claim_refs)),
            format!("equation_refs = {}", render_list(&row.equation_refs)),
            format!("symbol_refs = {}", render_list(&row.symbol_refs)),
            format!(
                "numeric_constants = {}",
                render_list(&row.numeric_constants)
            ),
            format!("key_tokens = {}", render_list(&row.key_tokens)),
            "".to_string(),
        ]);
    }
    lines.join("\n")
}

fn normalize_claim_refs(values: Vec<String>, claim_id: &str) -> Result<Vec<String>> {
    let claim_re = Regex::new(r"\bC-\d{3}\b")?;
    let mut refs = values
        .into_iter()
        .filter(|item| claim_re.is_match(item))
        .collect::<BTreeSet<_>>();
    if claim_re.is_match(claim_id) {
        refs.insert(claim_id.to_string());
    }
    Ok(refs.into_iter().collect())
}

fn classify_step(text: &str) -> String {
    let lower = text.to_lowercase();
    if lower.contains("h0") || lower.contains("h1") || lower.contains("assumption") {
        "assumption".to_string()
    } else if lower.contains("decision rule") || lower.contains("decision:") {
        "decision_rule".to_string()
    } else if lower.contains("required evidence") || lower.contains("evidence") {
        "evidence_requirement".to_string()
    } else if lower.contains("counterexample") || lower.contains("exhibit") {
        "witness_construction".to_string()
    } else if ["status", "verified", "refuted", "rejected"]
        .iter()
        .any(|token| lower.contains(token))
    {
        "status_update".to_string()
    } else if ["therefore", "thus", "hence"]
        .iter()
        .any(|token| lower.contains(token))
    {
        "conclusion".to_string()
    } else {
        "derivation_step".to_string()
    }
}

fn split_derivation_chunks(values: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for raw in values {
        let text = ascii_clean(&raw).replace('\r', "");
        for chunk in text.split("||") {
            let line = collapse(chunk).trim_start_matches('-').trim().to_string();
            if !line.is_empty() {
                out.push(line);
            }
        }
    }
    let mut deduped = Vec::new();
    let mut prev = String::new();
    for item in out {
        if item == prev {
            continue;
        }
        prev = item.clone();
        deduped.push(item);
    }
    deduped
}

fn extract_authors(citation: &str) -> Result<Vec<String>> {
    let bold_re = Regex::new(r"^\*\*(.+?)\.\*\*")?;
    let author_blob = if let Some(caps) = bold_re.captures(citation) {
        caps.get(1)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default()
    } else {
        citation
            .split_once('(')
            .map(|(prefix, _)| prefix.to_string())
            .unwrap_or_default()
    };
    Ok(author_blob
        .replace('&', " and ")
        .split(|ch| [',', ';'].contains(&ch))
        .flat_map(|part| part.split("and"))
        .map(|part| part.trim_matches([' ', '.']))
        .map(collapse)
        .filter(|item| !item.is_empty())
        .collect())
}

fn extract_title(citation: &str) -> Result<(String, String)> {
    let italic_re = Regex::new(r"\)\.\s*\*(.+?)\*\.")?;
    let quoted_re = Regex::new(r#"\)\.\s*"([^"]+)"\."#).ok();
    if let Some(caps) = italic_re.captures(citation) {
        let title = collapse(caps.get(1).map(|m| m.as_str()).unwrap_or_default());
        let tail = citation
            .get(caps.get(0).map(|m| m.end()).unwrap_or(0)..)
            .unwrap_or_default()
            .to_string();
        return Ok((title, tail));
    }
    if let Some(quoted_re) = quoted_re
        && let Some(caps) = quoted_re.captures(citation)
    {
        let title = collapse(caps.get(1).map(|m| m.as_str()).unwrap_or_default());
        let tail = citation
            .get(caps.get(0).map(|m| m.end()).unwrap_or(0)..)
            .unwrap_or_default()
            .to_string();
        return Ok((title, tail));
    }
    let after_year = Regex::new(r"\)\.\s*(.+)")?;
    let mut fallback = if let Some(caps) = after_year.captures(citation) {
        caps.get(1)
            .map(|m| m.as_str())
            .unwrap_or_default()
            .to_string()
    } else {
        citation.to_string()
    };
    if let Some(idx) = fallback.find('[') {
        fallback.truncate(idx);
    }
    Ok((collapse(fallback.trim_matches([' ', '.'])), String::new()))
}

fn extract_venue(tail: &str) -> Result<String> {
    let link_re = Regex::new(r"\[[^\]]+\]\([^)]+\)")?;
    let url_re = Regex::new(r"https?://[^\s)]+")?;
    let text = link_re.replace_all(tail, "");
    let text = url_re.replace_all(&text, "");
    Ok(collapse(text.trim_matches([' ', '.', ';', ':'])))
}

fn document_type(citation: &str, arxiv_id: &str, doi_list: &[String], urls: &[String]) -> String {
    let lower = citation.to_lowercase();
    if !arxiv_id.is_empty() {
        "arxiv_preprint".to_string()
    } else if !doi_list.is_empty() {
        "doi_reference".to_string()
    } else if urls
        .iter()
        .any(|url| url.ends_with(".csv") || url.contains("zenodo"))
    {
        "dataset_reference".to_string()
    } else if lower.contains("collaboration") {
        "collaboration_report".to_string()
    } else if lower.contains("journal") || lower.contains("rev.") || lower.contains("phys.") {
        "journal_article".to_string()
    } else if lower.contains("proceedings") || lower.contains("conference") {
        "conference_paper".to_string()
    } else {
        "general_reference".to_string()
    }
}

fn relevance_score(citation: &str, notes: &[String]) -> usize {
    let corpus = format!("{} {}", citation, notes.join(" ")).to_lowercase();
    let mut score = 0usize;
    for token in [
        "source",
        "dataset",
        "evidence",
        "verify",
        "verification",
        "reproducible",
        "hash",
    ] {
        if corpus.contains(token) {
            score += 1;
        }
    }
    score.min(7)
}

fn extract_local_paths(body: &str) -> Result<Vec<String>> {
    let backtick_re = Regex::new(r"`([^`]+)`")?;
    let mut out = BTreeSet::new();
    for payload in backtick_re
        .captures_iter(body)
        .filter_map(|caps| caps.get(1))
        .map(|m| collapse(m.as_str()))
    {
        let candidate = payload
            .trim_matches([' ', '.', ';', ':', ',', '(', ')', '[', ']', '{', '}'])
            .to_string();
        if candidate.is_empty() {
            continue;
        }
        if (!candidate.contains('/') && !candidate.contains('.'))
            || candidate.starts_with("http://")
            || candidate.starts_with("https://")
        {
            continue;
        }
        out.insert(candidate);
    }
    Ok(out.into_iter().collect())
}

fn paragraph_kind(text: &str) -> Result<String> {
    let numbered_re = Regex::new(r"^\d+\.\s+")?;
    let stripped = text.trim();
    let lines = stripped
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return Ok("empty".to_string());
    }
    if lines[0].starts_with("```") {
        return Ok("code_block".to_string());
    }
    if lines[0].starts_with('#') {
        return Ok("heading".to_string());
    }
    if lines[0].starts_with("<!--") {
        return Ok("comment_block".to_string());
    }
    if lines.len() >= 2 && lines[..2].iter().all(|line| line.contains('|')) {
        return Ok("table_block".to_string());
    }
    if lines
        .iter()
        .all(|line| line.starts_with("- ") || line.starts_with("* ") || numbered_re.is_match(line))
    {
        return Ok("list_block".to_string());
    }
    Ok("prose".to_string())
}

fn split_blocks(text: &str) -> Vec<(usize, usize, String)> {
    let lines = ascii_clean(text)
        .lines()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let filtered = lines
        .into_iter()
        .filter(|line| !line.starts_with("<!-- AUTO-GENERATED: DO NOT EDIT -->"))
        .filter(|line| !line.starts_with("<!-- Source of truth:"))
        .collect::<Vec<_>>();
    let mut blocks = Vec::new();
    let mut buf = Vec::new();
    let mut start_line = 1usize;
    let mut in_code = false;
    for (idx, line) in filtered.iter().enumerate() {
        let line_no = idx + 1;
        let stripped = line.trim();
        let fence = stripped.starts_with("```");
        if fence {
            if buf.is_empty() {
                start_line = line_no;
            }
            buf.push(line.clone());
            in_code = !in_code;
            continue;
        }
        if in_code {
            if buf.is_empty() {
                start_line = line_no;
            }
            buf.push(line.clone());
            continue;
        }
        if !stripped.is_empty() {
            if buf.is_empty() {
                start_line = line_no;
            }
            buf.push(line.clone());
            continue;
        }
        if !buf.is_empty() {
            let block_text = buf.join("\n").trim_end().to_string();
            blocks.push((start_line, line_no - 1, block_text));
            buf.clear();
        }
    }
    if !buf.is_empty() {
        blocks.push((
            start_line,
            filtered.len(),
            buf.join("\n").trim_end().to_string(),
        ));
    }
    blocks
}

fn claim_set(rows: &[Value]) -> BTreeSet<String> {
    rows.iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|id| id.starts_with("C-") || id.starts_with("T-"))
        .collect()
}

fn nonempty_body_doc_ids(rows: &[Value]) -> BTreeSet<(String, String)> {
    rows.iter()
        .filter_map(|row| {
            let doc_id = table_str(row, "id");
            let body = table_str(row, "body_markdown");
            if doc_id.is_empty() || !has_paragraph_payload(body) {
                None
            } else {
                Some((
                    doc_id.to_string(),
                    table_str(row, "source_markdown").to_string(),
                ))
            }
        })
        .collect()
}

fn has_paragraph_payload(body: &str) -> bool {
    !split_blocks(body).is_empty()
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

fn table_int_in(value: &Value, table: &str, key: &str) -> i64 {
    value
        .get(table)
        .and_then(|child| child.get(key))
        .and_then(Value::as_integer)
        .unwrap_or(-1)
}

fn assert_ascii_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    assert_ascii(&text, &path.display().to_string())
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
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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

trait SortableVec<T> {
    fn sorted_by_key<K: Ord, F: FnMut(&T) -> K>(self, f: F) -> Vec<T>;
}

impl<T> SortableVec<T> for std::slice::Iter<'_, T>
where
    T: Clone,
{
    fn sorted_by_key<K: Ord, F: FnMut(&T) -> K>(self, mut f: F) -> Vec<T> {
        let mut values = self.cloned().collect::<Vec<_>>();
        values.sort_by_key(|item| f(item));
        values
    }
}
