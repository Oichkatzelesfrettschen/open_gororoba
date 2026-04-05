use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
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
    name = "artifact-scrolls",
    about = "Build and verify canonical artifact scroll registries"
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
    /// Canonical SQLite control-plane DB used for claim metadata; falls back to claims_registry if absent.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(long, default_value = "registry/data_artifact_narratives.toml")]
    source_registry: PathBuf,
    #[arg(long, default_value = "registry/knowledge/equation_atoms.toml")]
    equation_registry: PathBuf,
    #[arg(long, default_value = "registry/knowledge/proof_atoms.toml")]
    proof_registry: PathBuf,
    /// Compatibility-export claims TOML, used only when the canonical DB is unavailable.
    #[arg(long, default_value = "registry/claims.toml")]
    claims_registry: PathBuf,
    #[arg(long, default_value = "registry/artifact_scrolls.toml")]
    index_out: PathBuf,
    #[arg(long, default_value = "registry/knowledge/artifacts")]
    scroll_dir: PathBuf,
}

#[derive(Args, Debug, Clone)]
struct VerifyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/artifact_scrolls.toml")]
    index_path: PathBuf,
    #[arg(long, default_value = "registry/data_artifact_narratives.toml")]
    source_registry: PathBuf,
    #[arg(long, default_value = "registry/knowledge/equation_atoms.toml")]
    equation_registry: PathBuf,
    #[arg(long, default_value = "registry/knowledge/proof_atoms.toml")]
    proof_registry: PathBuf,
}

#[derive(Debug, Clone)]
struct Section {
    title: String,
    level: usize,
    line_start: usize,
    line_end: usize,
    lines: Vec<String>,
}

#[derive(Debug, Clone)]
struct SourceRef {
    kind: String,
    value: String,
    line: usize,
    excerpt: String,
}

#[derive(Debug, Clone)]
struct SectionRow {
    id: String,
    title: String,
    level: usize,
    line_start: usize,
    line_end: usize,
    paragraph_count: usize,
    char_count: usize,
    fingerprint: String,
    summary: String,
    claim_refs: Vec<String>,
    equation_ref_ids: Vec<String>,
    proof_ref_ids: Vec<String>,
    body_text: String,
}

#[derive(Debug, Clone)]
struct ScrollCounts {
    section_count: usize,
    claim_ref_count: usize,
    equation_ref_count: usize,
    proof_ref_count: usize,
    source_ref_count: usize,
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
    let source_data = load_toml(&repo_root.join(&args.source_registry))?;
    let equation_data = load_toml(&repo_root.join(&args.equation_registry))?;
    let proof_data = load_toml(&repo_root.join(&args.proof_registry))?;
    let (claim_status, claim_source_ref) =
        load_claim_status_map(&repo_root, &args.canonical_db, &args.claims_registry)?;

    let mut equations_by_source: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for row in table_array(&equation_data, "atom") {
        let source_path = table_str(row, "source_path").trim();
        if !source_path.is_empty() {
            equations_by_source
                .entry(source_path.to_string())
                .or_default()
                .push(row.clone());
        }
    }
    for rows in equations_by_source.values_mut() {
        rows.sort_by_key(|row| table_int(row, "source_line"));
    }

    let mut proofs_by_source: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for row in table_array(&proof_data, "atom") {
        let source_path = table_str(row, "source_path").trim();
        if !source_path.is_empty() {
            proofs_by_source
                .entry(source_path.to_string())
                .or_default()
                .push(row.clone());
        }
    }
    for rows in proofs_by_source.values_mut() {
        rows.sort_by_key(|row| table_int(row, "line_start"));
    }

    let mut docs = table_array(&source_data, "document").to_vec();
    docs.sort_by(|lhs, rhs| {
        table_str(lhs, "source_markdown").cmp(table_str(rhs, "source_markdown"))
    });

    let scroll_dir = repo_root.join(&args.scroll_dir);
    fs::create_dir_all(&scroll_dir).with_context(|| format!("mkdir {}", scroll_dir.display()))?;

    let mut index_rows = Vec::new();
    for doc in docs {
        let source_markdown = table_str(&doc, "source_markdown").trim().to_string();
        let scroll_id = table_str(&doc, "id").trim().to_string();
        if source_markdown.is_empty() || scroll_id.is_empty() {
            continue;
        }
        let scroll_rel = format!(
            "{}/{}.toml",
            args.scroll_dir.to_string_lossy().trim_end_matches('/'),
            scroll_id
        );
        let scroll_path = repo_root.join(&scroll_rel);
        let (scroll_text, counts, fingerprint) = render_scroll_toml(
            &doc,
            &claim_status,
            &claim_source_ref,
            equations_by_source
                .get(&source_markdown)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            proofs_by_source
                .get(&source_markdown)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        )?;
        write_ascii(&scroll_path, &scroll_text)?;
        index_rows.push(IndexRow {
            id: scroll_id.clone(),
            source_markdown,
            title: table_str(&doc, "title").to_string(),
            content_kind: table_str(&doc, "content_kind").to_string(),
            scroll_path: scroll_rel,
            canonical: true,
            section_count: counts.section_count,
            claim_ref_count: counts.claim_ref_count,
            equation_ref_count: counts.equation_ref_count,
            proof_ref_count: counts.proof_ref_count,
            source_ref_count: counts.source_ref_count,
            dedup_fingerprint: fingerprint,
        });
    }
    write_ascii(&repo_root.join(&args.index_out), &render_index(&index_rows))?;
    println!(
        "Wrote artifact scroll registries: scrolls={} index={} dir={}",
        index_rows.len(),
        args.index_out.display(),
        args.scroll_dir.display()
    );
    Ok(())
}

fn verify(args: &VerifyArgs) -> Result<()> {
    let root = args.repo_root.canonicalize().context("resolve repo root")?;
    let index_path = root.join(&args.index_path);
    let source_path = root.join(&args.source_registry);
    let equation_path = root.join(&args.equation_registry);
    let proof_path = root.join(&args.proof_registry);

    let mut failures = Vec::new();
    for path in [&index_path, &source_path, &equation_path, &proof_path] {
        if !path.exists() {
            failures.push(format!("missing registry: {}", path.display()));
            continue;
        }
        if let Err(err) = assert_ascii_file(path) {
            failures.push(err.to_string());
        }
    }
    if !failures.is_empty() {
        eprintln!("ERROR: artifact scroll verification failed.");
        for item in failures {
            eprintln!("- {item}");
        }
        bail!("artifact scroll verification failed");
    }

    let index = load_toml(&index_path)?;
    let source = load_toml(&source_path)?;
    let equation = load_toml(&equation_path)?;
    let proof = load_toml(&proof_path)?;

    let index_meta = table_value_table(&index, "artifact_scrolls");
    let index_rows = table_array(&index, "scroll");
    let source_docs = table_array(&source, "document");
    let equation_atoms = table_array(&equation, "atom");
    let proof_atoms = table_array(&proof, "atom");

    let eq_ids: BTreeSet<String> = equation_atoms
        .iter()
        .map(|row| table_str(row, "id").trim().to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let proof_ids: BTreeSet<String> = proof_atoms
        .iter()
        .map(|row| table_str(row, "id").trim().to_string())
        .filter(|value| !value.is_empty())
        .collect();

    let mut eq_counts_by_source = BTreeMap::<String, usize>::new();
    for row in equation_atoms {
        let source_markdown = table_str(row, "source_path").trim();
        if !source_markdown.is_empty() {
            *eq_counts_by_source
                .entry(source_markdown.to_string())
                .or_default() += 1;
        }
    }
    let mut proof_counts_by_source = BTreeMap::<String, usize>::new();
    for row in proof_atoms {
        let source_markdown = table_str(row, "source_path").trim();
        if !source_markdown.is_empty() {
            *proof_counts_by_source
                .entry(source_markdown.to_string())
                .or_default() += 1;
        }
    }

    if table_int_from_table(index_meta, "scroll_count") != index_rows.len() as i64 {
        failures.push("artifact_scrolls.scroll_count metadata mismatch".to_string());
    }

    let mut index_by_source = BTreeMap::<String, &Value>::new();
    for row in index_rows {
        let source_markdown = table_str(row, "source_markdown").trim().to_string();
        if source_markdown.is_empty() {
            failures.push("index row missing source_markdown".to_string());
            continue;
        }
        if index_by_source.contains_key(&source_markdown) {
            failures.push(format!(
                "duplicate source_markdown in index: {source_markdown}"
            ));
        }
        index_by_source.insert(source_markdown, row);
    }

    for doc in source_docs {
        let source_markdown = table_str(doc, "source_markdown").trim();
        if !source_markdown.is_empty() && !index_by_source.contains_key(source_markdown) {
            failures.push(format!(
                "missing scroll index for source document: {source_markdown}"
            ));
        }
    }

    for row in index_rows {
        let source_markdown = table_str(row, "source_markdown").trim();
        let scroll_rel = table_str(row, "scroll_path").trim();
        let scroll_file = root.join(scroll_rel);
        if scroll_rel.is_empty() {
            failures.push(format!("{source_markdown}: empty scroll_path in index"));
            continue;
        }
        if !scroll_file.exists() {
            failures.push(format!(
                "{source_markdown}: missing scroll file {scroll_rel}"
            ));
            continue;
        }
        if let Err(err) = assert_ascii_file(&scroll_file) {
            failures.push(err.to_string());
            continue;
        }

        let scroll = load_toml(&scroll_file)?;
        let scroll_meta = table_value_table(&scroll, "scroll");
        let section_rows = table_array(&scroll, "section");
        let claim_rows = table_array(&scroll, "claim_ref");
        let equation_rows = table_array(&scroll, "equation_ref");
        let proof_rows = table_array(&scroll, "proof_ref");
        let source_rows = table_array(&scroll, "source_ref");

        if table_str_from_table(scroll_meta, "source_markdown").trim() != source_markdown {
            failures.push(format!(
                "{source_markdown}: scroll metadata source_markdown mismatch"
            ));
        }
        if table_str_from_table(scroll_meta, "canonical_registry").trim()
            != "registry/artifact_scrolls.toml"
        {
            failures.push(format!(
                "{source_markdown}: canonical_registry must be registry/artifact_scrolls.toml"
            ));
        }
        if table_bool_from_table(scroll_meta, "authoritative") != Some(true) {
            failures.push(format!(
                "{source_markdown}: authoritative flag must be true"
            ));
        }

        compare_count(
            &mut failures,
            source_markdown,
            "section_count",
            table_int_from_table(scroll_meta, "section_count"),
            section_rows.len(),
        );
        compare_count(
            &mut failures,
            source_markdown,
            "claim_ref_count",
            table_int_from_table(scroll_meta, "claim_ref_count"),
            claim_rows.len(),
        );
        compare_count(
            &mut failures,
            source_markdown,
            "equation_ref_count",
            table_int_from_table(scroll_meta, "equation_ref_count"),
            equation_rows.len(),
        );
        compare_count(
            &mut failures,
            source_markdown,
            "proof_ref_count",
            table_int_from_table(scroll_meta, "proof_ref_count"),
            proof_rows.len(),
        );
        compare_count(
            &mut failures,
            source_markdown,
            "source_ref_count",
            table_int_from_table(scroll_meta, "source_ref_count"),
            source_rows.len(),
        );

        if section_rows.is_empty() {
            failures.push(format!("{source_markdown}: scroll has zero sections"));
        }
        for section in section_rows {
            if table_str(section, "body_text").trim().is_empty() {
                failures.push(format!(
                    "{source_markdown}: empty section body_text in {}",
                    table_str(section, "id")
                ));
                break;
            }
        }

        compare_count(
            &mut failures,
            source_markdown,
            "index section_count",
            table_int(row, "section_count"),
            section_rows.len(),
        );
        compare_count(
            &mut failures,
            source_markdown,
            "index claim_ref_count",
            table_int(row, "claim_ref_count"),
            claim_rows.len(),
        );
        compare_count(
            &mut failures,
            source_markdown,
            "index equation_ref_count",
            table_int(row, "equation_ref_count"),
            equation_rows.len(),
        );
        compare_count(
            &mut failures,
            source_markdown,
            "index proof_ref_count",
            table_int(row, "proof_ref_count"),
            proof_rows.len(),
        );
        compare_count(
            &mut failures,
            source_markdown,
            "index source_ref_count",
            table_int(row, "source_ref_count"),
            source_rows.len(),
        );

        let expected_eq = eq_counts_by_source
            .get(source_markdown)
            .copied()
            .unwrap_or(0);
        if table_int(row, "equation_ref_count") != expected_eq as i64 {
            failures.push(format!(
                "{source_markdown}: equation_ref_count {} != knowledge atom count {}",
                table_int(row, "equation_ref_count"),
                expected_eq
            ));
        }
        let expected_proofs = proof_counts_by_source
            .get(source_markdown)
            .copied()
            .unwrap_or(0);
        if table_int(row, "proof_ref_count") != expected_proofs as i64 {
            failures.push(format!(
                "{source_markdown}: proof_ref_count {} != knowledge atom count {}",
                table_int(row, "proof_ref_count"),
                expected_proofs
            ));
        }

        for eq in equation_rows {
            let eq_id = table_str(eq, "id").trim();
            if !eq_id.is_empty() && !eq_ids.contains(eq_id) {
                failures.push(format!(
                    "{source_markdown}: unknown equation_ref id {eq_id}"
                ));
                break;
            }
        }
        for prf in proof_rows {
            let prf_id = table_str(prf, "id").trim();
            if !prf_id.is_empty() && !proof_ids.contains(prf_id) {
                failures.push(format!("{source_markdown}: unknown proof_ref id {prf_id}"));
                break;
            }
        }
    }

    let total_sections: usize = index_rows
        .iter()
        .map(|row| table_int(row, "section_count").max(0) as usize)
        .sum();
    let total_claims: usize = index_rows
        .iter()
        .map(|row| table_int(row, "claim_ref_count").max(0) as usize)
        .sum();
    let total_equations: usize = index_rows
        .iter()
        .map(|row| table_int(row, "equation_ref_count").max(0) as usize)
        .sum();
    let total_proofs: usize = index_rows
        .iter()
        .map(|row| table_int(row, "proof_ref_count").max(0) as usize)
        .sum();
    let total_sources: usize = index_rows
        .iter()
        .map(|row| table_int(row, "source_ref_count").max(0) as usize)
        .sum();

    compare_total(
        &mut failures,
        index_meta,
        "total_section_count",
        total_sections,
    );
    compare_total(
        &mut failures,
        index_meta,
        "total_claim_ref_count",
        total_claims,
    );
    compare_total(
        &mut failures,
        index_meta,
        "total_equation_ref_count",
        total_equations,
    );
    compare_total(
        &mut failures,
        index_meta,
        "total_proof_ref_count",
        total_proofs,
    );
    compare_total(
        &mut failures,
        index_meta,
        "total_source_ref_count",
        total_sources,
    );

    if !failures.is_empty() {
        eprintln!("ERROR: artifact scroll verification failed.");
        for item in failures.iter().take(200) {
            eprintln!("- {item}");
        }
        if failures.len() > 200 {
            eprintln!("- ... and {} more failures", failures.len() - 200);
        }
        bail!("artifact scroll verification failed");
    }

    println!(
        "OK: artifact scroll registry verified. scrolls={} sections={} claims={} equations={} proofs={} source_refs={}",
        index_rows.len(),
        total_sections,
        total_claims,
        total_equations,
        total_proofs,
        total_sources
    );
    Ok(())
}

#[derive(Debug, Clone)]
struct IndexRow {
    id: String,
    source_markdown: String,
    title: String,
    content_kind: String,
    scroll_path: String,
    canonical: bool,
    section_count: usize,
    claim_ref_count: usize,
    equation_ref_count: usize,
    proof_ref_count: usize,
    source_ref_count: usize,
    dedup_fingerprint: String,
}

fn render_scroll_toml(
    doc: &Value,
    claim_status: &BTreeMap<String, String>,
    claim_source_ref: &str,
    equation_rows: &[Value],
    proof_rows: &[Value],
) -> Result<(String, ScrollCounts, String)> {
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+?)\s*$")?;
    let claim_re = Regex::new(r"\bC-\d{3}\b")?;
    let source_path = table_str(doc, "source_markdown").trim().to_string();
    let body = ascii_sanitize(table_str(doc, "body_markdown").trim_end_matches('\n'));
    let sections = split_sections(&body, &heading_re)?;
    let key_terms = extract_key_terms(&body)?;
    let source_refs = extract_source_refs(&body)?;
    let mut claim_refs: BTreeSet<String> = string_list(doc, "claim_refs").into_iter().collect();
    for hit in claim_re.find_iter(&body) {
        claim_refs.insert(hit.as_str().to_string());
    }
    let claim_refs: Vec<String> = claim_refs.into_iter().collect();
    let unknown_claim_refs: Vec<String> = claim_refs
        .iter()
        .filter(|item| !claim_status.contains_key(item.as_str()))
        .cloned()
        .collect();
    let document_fingerprint = fingerprint(&body);

    let mut section_rows = Vec::new();
    for (idx, section) in sections.iter().enumerate() {
        let section_text = ascii_sanitize(section.lines.join("\n").trim_end_matches('\n'));
        let section_claim_refs = sorted_unique_regex(&section_text, &claim_re);
        let section_equations: Vec<&Value> = equation_rows
            .iter()
            .filter(|row| {
                let line = table_int(row, "source_line");
                section.line_start as i64 <= line && line <= section.line_end as i64
            })
            .collect();
        let section_proofs: Vec<&Value> = proof_rows
            .iter()
            .filter(|row| {
                let line_end = table_int(row, "line_end");
                let line_start = table_int(row, "line_start");
                !(line_end < section.line_start as i64 || line_start > section.line_end as i64)
            })
            .collect();
        if section_text.trim().is_empty()
            && section_claim_refs.is_empty()
            && section_equations.is_empty()
            && section_proofs.is_empty()
        {
            continue;
        }
        let paragraph_count = section_text
            .split("\n\n")
            .filter(|part| !part.trim().is_empty())
            .count()
            .max(1);
        section_rows.push(SectionRow {
            id: format!("{}-SEC-{:03}", table_str(doc, "id"), idx + 1),
            title: collapse_ws(&section.title),
            level: section.level,
            line_start: section.line_start,
            line_end: section.line_end,
            paragraph_count,
            char_count: section_text.len(),
            fingerprint: fingerprint(&section_text),
            summary: collapse_ws(&section_text).chars().take(220).collect(),
            claim_refs: section_claim_refs,
            equation_ref_ids: section_equations
                .iter()
                .filter_map(|row| {
                    let id = table_str(row, "id");
                    if id.is_empty() {
                        None
                    } else {
                        Some(id.to_string())
                    }
                })
                .collect(),
            proof_ref_ids: section_proofs
                .iter()
                .filter_map(|row| {
                    let id = table_str(row, "id");
                    if id.is_empty() {
                        None
                    } else {
                        Some(id.to_string())
                    }
                })
                .collect(),
            body_text: section_text,
        });
    }

    let mut lines = vec![
        "# Structured artifact scroll (TOML lane pending SQLite promotion).".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/artifact_scrolls.rs".to_string(),
        "".to_string(),
        "[scroll]".to_string(),
        format!("id = {}", esc(table_str(doc, "id"))),
        format!("source_uid = {}", esc(table_str(doc, "id"))),
        "source_registry = \"registry/data_artifact_narratives.toml\"".to_string(),
        format!("source_markdown = {}", esc(&source_path)),
        format!("title = {}", esc(table_str(doc, "title"))),
        format!("content_kind = {}", esc(table_str(doc, "content_kind"))),
        "canonical_registry = \"registry/artifact_scrolls.toml\"".to_string(),
        "authoritative = true".to_string(),
        "updated = \"deterministic\"".to_string(),
        format!("line_count = {}", table_int(doc, "line_count").max(0)),
        format!("section_count = {}", section_rows.len()),
        format!("claim_ref_count = {}", claim_refs.len()),
        format!("equation_ref_count = {}", equation_rows.len()),
        format!("proof_ref_count = {}", proof_rows.len()),
        format!("source_ref_count = {}", source_refs.len()),
        format!("unknown_claim_ref_count = {}", unknown_claim_refs.len()),
        format!("dedup_fingerprint = {}", esc(&document_fingerprint)),
        format!("key_terms = {}", render_list(&key_terms)),
        format!("claim_refs = {}", render_list(&claim_refs)),
        format!("unknown_claim_refs = {}", render_list(&unknown_claim_refs)),
        "".to_string(),
    ];

    for claim_id in &claim_refs {
        lines.push("[[claim_ref]]".to_string());
        lines.push(format!("id = {}", esc(claim_id)));
        lines.push(format!(
            "status_token = {}",
            esc(claim_status
                .get(claim_id)
                .map(String::as_str)
                .unwrap_or("UNKNOWN"))
        ));
        lines.push(format!("source = {}", esc(claim_source_ref)));
        lines.push(String::new());
    }
    for row in equation_rows {
        lines.push("[[equation_ref]]".to_string());
        lines.push(format!("id = {}", esc(table_str(row, "id"))));
        lines.push(format!(
            "source_line = {}",
            table_int(row, "source_line").max(0)
        ));
        lines.push(format!(
            "equation_kind = {}",
            esc(table_str(row, "equation_kind"))
        ));
        lines.push(format!(
            "relation_operator = {}",
            esc(table_str(row, "relation_operator"))
        ));
        lines.push(format!(
            "domain_hint = {}",
            esc(table_str(row, "domain_hint"))
        ));
        lines.push(format!(
            "expression = {}",
            esc(table_str(row, "expression"))
        ));
        lines.push(format!(
            "symbol_names = {}",
            render_list(&string_list(row, "symbol_names"))
        ));
        lines.push(format!(
            "numeric_constants = {}",
            render_list(&string_list(row, "numeric_constants"))
        ));
        lines.push(format!(
            "claim_refs = {}",
            render_list(&string_list(row, "claim_refs"))
        ));
        lines.push(String::new());
    }
    for row in proof_rows {
        lines.push("[[proof_ref]]".to_string());
        lines.push(format!("id = {}", esc(table_str(row, "id"))));
        lines.push(format!(
            "proof_kind = {}",
            esc(table_str(row, "proof_kind"))
        ));
        lines.push(format!(
            "section_title = {}",
            esc(table_str(row, "section_title"))
        ));
        lines.push(format!(
            "line_start = {}",
            table_int(row, "line_start").max(0)
        ));
        lines.push(format!("line_end = {}", table_int(row, "line_end").max(0)));
        lines.push(format!(
            "step_count = {}",
            table_int(row, "step_count").max(0)
        ));
        lines.push(format!(
            "supports_claim = {}",
            if table_bool(row, "supports_claim") {
                "true"
            } else {
                "false"
            }
        ));
        lines.push(format!(
            "claim_refs = {}",
            render_list(&string_list(row, "claim_refs"))
        ));
        lines.push(format!("excerpt = {}", esc(table_str(row, "excerpt"))));
        lines.push(String::new());
    }
    for (idx, row) in source_refs.iter().enumerate() {
        lines.push("[[source_ref]]".to_string());
        lines.push(format!("id = {}", esc(&format!("SRC-{:03}", idx + 1))));
        lines.push(format!("kind = {}", esc(&row.kind)));
        lines.push(format!("value = {}", esc(&row.value)));
        lines.push(format!("line = {}", row.line));
        lines.push(format!("excerpt = {}", esc(&row.excerpt)));
        lines.push(String::new());
    }
    for row in &section_rows {
        lines.push("[[section]]".to_string());
        lines.push(format!("id = {}", esc(&row.id)));
        lines.push(format!("title = {}", esc(&row.title)));
        lines.push(format!("level = {}", row.level));
        lines.push(format!("line_start = {}", row.line_start));
        lines.push(format!("line_end = {}", row.line_end));
        lines.push(format!("paragraph_count = {}", row.paragraph_count));
        lines.push(format!("char_count = {}", row.char_count));
        lines.push(format!("fingerprint = {}", esc(&row.fingerprint)));
        lines.push(format!("summary = {}", esc(&row.summary)));
        lines.push(format!("claim_refs = {}", render_list(&row.claim_refs)));
        lines.push(format!(
            "equation_ref_ids = {}",
            render_list(&row.equation_ref_ids)
        ));
        lines.push(format!(
            "proof_ref_ids = {}",
            render_list(&row.proof_ref_ids)
        ));
        lines.push(format!("body_text = {}", esc(&row.body_text)));
        lines.push(String::new());
    }

    Ok((
        lines.join("\n"),
        ScrollCounts {
            section_count: section_rows.len(),
            claim_ref_count: claim_refs.len(),
            equation_ref_count: equation_rows.len(),
            proof_ref_count: proof_rows.len(),
            source_ref_count: source_refs.len(),
        },
        document_fingerprint,
    ))
}

fn load_claim_status_map(
    repo_root: &Path,
    canonical_db: &Path,
    claims_registry: &Path,
) -> Result<(BTreeMap<String, String>, String)> {
    let db_path = repo_root.join(canonical_db);
    if db_path.exists() {
        let store = ProvenanceStore::open(&db_path)
            .with_context(|| format!("open canonical db {}", db_path.display()))?;
        let claims = store
            .list_claims()
            .with_context(|| format!("load claims from {}", db_path.display()))?;
        if !claims.is_empty() {
            let map = claims
                .into_iter()
                .map(|row| {
                    (
                        row.id,
                        collapse_ws(&row.status).to_uppercase().replace(' ', "_"),
                    )
                })
                .collect();
            return Ok((map, "registry/canonical/control_plane.sqlite3".to_string()));
        }
    }

    let claims_data = load_toml(&repo_root.join(claims_registry))?;
    let map = table_array(&claims_data, "claim")
        .iter()
        .filter_map(|row| {
            let id = table_str(row, "id").trim().to_string();
            if id.is_empty() {
                None
            } else {
                Some((
                    id,
                    collapse_ws(table_str(row, "status"))
                        .to_uppercase()
                        .replace(' ', "_"),
                ))
            }
        })
        .collect();
    Ok((map, claims_registry.to_string_lossy().to_string()))
}

fn render_index(rows: &[IndexRow]) -> String {
    let total_sections: usize = rows.iter().map(|row| row.section_count).sum();
    let total_claims: usize = rows.iter().map(|row| row.claim_ref_count).sum();
    let total_equations: usize = rows.iter().map(|row| row.equation_ref_count).sum();
    let total_proofs: usize = rows.iter().map(|row| row.proof_ref_count).sum();
    let total_sources: usize = rows.iter().map(|row| row.source_ref_count).sum();
    let mut lines = vec![
        "# Canonical index for structured artifact scrolls.".to_string(),
        "# Generated by crates/gororoba_cli_data/src/bin/artifact_scrolls.rs".to_string(),
        "".to_string(),
        "[artifact_scrolls]".to_string(),
        "updated = \"deterministic\"".to_string(),
        "authoritative = true".to_string(),
        "source_registry = \"registry/data_artifact_narratives.toml\"".to_string(),
        format!("scroll_count = {}", rows.len()),
        format!("total_section_count = {}", total_sections),
        format!("total_claim_ref_count = {}", total_claims),
        format!("total_equation_ref_count = {}", total_equations),
        format!("total_proof_ref_count = {}", total_proofs),
        format!("total_source_ref_count = {}", total_sources),
        "".to_string(),
    ];
    for row in rows {
        lines.push("[[scroll]]".to_string());
        lines.push(format!("id = {}", esc(&row.id)));
        lines.push(format!("source_uid = {}", esc(&row.id)));
        lines.push(format!("source_markdown = {}", esc(&row.source_markdown)));
        lines.push(format!("title = {}", esc(&row.title)));
        lines.push(format!("content_kind = {}", esc(&row.content_kind)));
        lines.push(format!("scroll_path = {}", esc(&row.scroll_path)));
        lines.push(format!(
            "canonical = {}",
            if row.canonical { "true" } else { "false" }
        ));
        lines.push(format!("section_count = {}", row.section_count));
        lines.push(format!("claim_ref_count = {}", row.claim_ref_count));
        lines.push(format!("equation_ref_count = {}", row.equation_ref_count));
        lines.push(format!("proof_ref_count = {}", row.proof_ref_count));
        lines.push(format!("source_ref_count = {}", row.source_ref_count));
        lines.push(format!(
            "dedup_fingerprint = {}",
            esc(&row.dedup_fingerprint)
        ));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn split_sections(text: &str, heading_re: &Regex) -> Result<Vec<Section>> {
    let lines: Vec<String> = ascii_sanitize(text)
        .lines()
        .map(ToOwned::to_owned)
        .collect();
    let mut sections = Vec::new();
    let mut current_title = "(root)".to_string();
    let mut current_level = 0usize;
    let mut current_start = 1usize;
    let mut current_lines = Vec::new();
    for (idx, raw) in lines.iter().enumerate() {
        let line_no = idx + 1;
        if let Some(caps) = heading_re.captures(raw) {
            sections.push(Section {
                title: current_title,
                level: current_level,
                line_start: current_start,
                line_end: current_start.max(line_no.saturating_sub(1)),
                lines: current_lines,
            });
            current_title = collapse_ws(caps.get(2).map(|m| m.as_str()).unwrap_or_default());
            current_level = caps.get(1).map(|m| m.as_str().len()).unwrap_or(0);
            current_start = line_no;
            current_lines = Vec::new();
        } else {
            current_lines.push(raw.clone());
        }
    }
    sections.push(Section {
        title: current_title,
        level: current_level,
        line_start: current_start,
        line_end: current_start.max(lines.len()),
        lines: current_lines,
    });

    let mut trimmed = Vec::new();
    for section in sections {
        let body = section.lines.join("\n").trim().to_string();
        if section.title == "(root)" && body.is_empty() {
            continue;
        }
        trimmed.push(section);
    }
    if trimmed.is_empty() {
        trimmed.push(Section {
            title: "(root)".to_string(),
            level: 0,
            line_start: 1,
            line_end: lines.len().max(1),
            lines,
        });
    }
    Ok(trimmed)
}

fn extract_key_terms(text: &str) -> Result<Vec<String>> {
    let identifier_re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")?;
    let stopwords = BTreeSet::from([
        "this", "that", "with", "from", "into", "their", "there", "where", "when", "then", "than",
        "have", "has", "been", "being", "were", "will", "would", "should", "could", "about",
        "between", "because", "while", "using", "used", "also", "only", "over", "under", "after",
        "before", "these", "those", "which", "such", "very", "more", "most", "some", "many",
        "much", "across", "without", "within", "through", "report", "final", "results", "result",
        "section",
    ]);
    let mut counts = BTreeMap::<String, usize>::new();
    for token in identifier_re.find_iter(&ascii_sanitize(text)) {
        let lowered = token.as_str().to_lowercase();
        if stopwords.contains(lowered.as_str()) || lowered.starts_with("http") || lowered.len() < 4
        {
            continue;
        }
        *counts.entry(lowered).or_default() += 1;
    }
    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
    Ok(ranked.into_iter().take(24).map(|(item, _)| item).collect())
}

fn extract_source_refs(text: &str) -> Result<Vec<SourceRef>> {
    let url_re = Regex::new(r#"https?://[^\s)>"']+"#)?;
    let doi_re = Regex::new(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")?;
    let arxiv_re = Regex::new(r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\b")?;
    let citation_year_re = Regex::new(r"(19|20)\d{2}")?;
    let mut refs = Vec::new();
    let mut seen = BTreeSet::new();
    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = ascii_sanitize(raw).trim().to_string();
        if line.is_empty() {
            continue;
        }
        for hit in url_re.find_iter(&line) {
            let value = hit.as_str().to_string();
            let key = format!("url::{value}::{line_no}");
            if seen.insert(key) {
                refs.push(SourceRef {
                    kind: "url".to_string(),
                    value,
                    line: line_no,
                    excerpt: collapse_ws(&line).chars().take(220).collect(),
                });
            }
        }
        for hit in doi_re.find_iter(&line) {
            let value = hit.as_str().to_string();
            let key = format!("doi::{value}::{line_no}");
            if seen.insert(key) {
                refs.push(SourceRef {
                    kind: "doi".to_string(),
                    value,
                    line: line_no,
                    excerpt: collapse_ws(&line).chars().take(220).collect(),
                });
            }
        }
        for hit in arxiv_re.find_iter(&line) {
            let value = hit
                .as_str()
                .replace("ARXIV:", "arXiv:")
                .replace("ArXiv:", "arXiv:");
            let key = format!("arxiv::{value}::{line_no}");
            if seen.insert(key) {
                refs.push(SourceRef {
                    kind: "arxiv".to_string(),
                    value,
                    line: line_no,
                    excerpt: collapse_ws(&line).chars().take(220).collect(),
                });
            }
        }
        let maybe_citation =
            (line.starts_with('-') || line.starts_with('*') || line.starts_with('['))
                && citation_year_re.is_match(&line)
                && line.len() >= 24;
        if maybe_citation {
            let citation_value: String = collapse_ws(&line).chars().take(240).collect();
            let key = format!("citation_line::{citation_value}::{line_no}");
            if seen.insert(key) {
                refs.push(SourceRef {
                    kind: "citation_line".to_string(),
                    value: citation_value.clone(),
                    line: line_no,
                    excerpt: citation_value,
                });
            }
        }
    }
    refs.sort_by(|lhs, rhs| {
        (&lhs.line, &lhs.kind, &lhs.value).cmp(&(&rhs.line, &rhs.kind, &rhs.value))
    });
    Ok(refs)
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

fn table_bool(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(false)
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

fn table_value_table<'a>(value: &'a Value, key: &str) -> &'a toml::map::Map<String, Value> {
    static EMPTY: std::sync::OnceLock<toml::map::Map<String, Value>> = std::sync::OnceLock::new();
    value
        .get(key)
        .and_then(Value::as_table)
        .unwrap_or_else(|| EMPTY.get_or_init(Default::default))
}

fn table_str_from_table<'a>(value: &'a toml::map::Map<String, Value>, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn table_int_from_table(value: &toml::map::Map<String, Value>, key: &str) -> i64 {
    value.get(key).and_then(Value::as_integer).unwrap_or(0)
}

fn table_bool_from_table(value: &toml::map::Map<String, Value>, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
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

fn fingerprint(text: &str) -> String {
    let normalized = collapse_ws(text);
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sorted_unique_regex(text: &str, regex: &Regex) -> Vec<String> {
    let mut out = BTreeSet::new();
    for hit in regex.find_iter(text) {
        out.insert(hit.as_str().to_string());
    }
    out.into_iter().collect()
}

fn esc(value: &str) -> String {
    serde_json::to_string(&ascii_sanitize(value)).unwrap_or_else(|_| "\"\"".to_string())
}

fn render_list(items: &[String]) -> String {
    if items.is_empty() {
        "[]".to_string()
    } else {
        format!(
            "[{}]",
            items
                .iter()
                .map(|item| esc(item))
                .collect::<Vec<_>>()
                .join(", ")
        )
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
        bail!("non-ASCII content in {context}: {:?}", sample);
    }
    Ok(())
}

fn compare_count(
    failures: &mut Vec<String>,
    source_markdown: &str,
    label: &str,
    expected: i64,
    actual: usize,
) {
    if expected != actual as i64 {
        failures.push(format!("{source_markdown}: {label} mismatch"));
    }
}

fn compare_total(
    failures: &mut Vec<String>,
    meta: &toml::map::Map<String, Value>,
    key: &str,
    actual: usize,
) {
    if table_int_from_table(meta, key) != actual as i64 {
        failures.push(format!("index {key} mismatch"));
    }
}
