//! Document extraction-quality inventory and routing contract.
//!
//! The original document bytes are the citation authority and stay external to
//! Git and Git LFS. This binary reads those bytes read-only, identifies each
//! source by SHA-256, reconciles the tracked ingest manifest and the existing
//! `docpipe` derivatives, and assigns every unique source identity a single
//! routing decision plus deterministic reason codes. It never invokes an OCR
//! engine, never rewrites a derivative, and treats extracted text as evidence,
//! never as authority.
//!
//! The inventory subcommand emits one compact TOML record per unique source
//! SHA-256. Page count and PDF validity come from the pure-Rust `lopdf` parser
//! so the run is deterministic and needs no native `libpdfium.so`. Signals that
//! require per-page or coordinate geometry the stored derivative does not carry
//! (reading-order geometry, per-page density median) are implemented as pure,
//! unit-tested functions and recorded with an explicit not-evaluated state so a
//! data limitation stays visible instead of silently narrowing coverage.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

/// Schema tag stamped into every emitted inventory so downstream tranches can
/// bind to a stable contract version.
const SCHEMA_VERSION: &str = "document-mineru-inventory/v1";

// ---- Classification thresholds ------------------------------------------
//
// Each constant names the behaviour it separates. A native-text research
// article carries roughly 1500-3500 non-whitespace characters of body text per
// page; an image-only scan yields near-zero native text because the glyphs live
// in a raster, not a font. The cutoff sits far below any real text page so only
// genuinely text-starved documents trip it.
const SCAN_MIN_NONWS_PER_PAGE: f64 = 200.0;

// Math-dense papers spend a large fraction of their characters on operators,
// relations, Greek letters, and sub/superscript scaffolding. Above this share
// of non-whitespace characters the body reads as formula-heavy; if the stored
// extraction then holds zero equations, the equation surface is missing rather
// than merely sparse.
const FORMULA_MATH_CHAR_RATIO_MIN: f64 = 0.020;

// Tabular layouts leave multi-space column gutters that survive text
// extraction. A handful of aligned rows is ordinary prose wrapping; a document
// with many gutter-aligned lines and no captured tables has table structure the
// extractor dropped.
const TABLE_ALIGNED_LINES_MIN: usize = 8;

// A clean extraction carries essentially no U+FFFD replacement characters;
// their presence means bytes were decoded past a broken encoding boundary.
const CORRUPTION_REPLACEMENT_RATIO_MAX: f64 = 0.001;

// Stray C0 control characters (excluding tab, newline, carriage return) signal
// glyph-to-Unicode mapping failures such as ligatures collapsing to STX.
const CORRUPTION_CONTROL_RATIO_MAX: f64 = 0.002;

// When most non-empty lines are exact duplicates the extractor looped or
// stuttered rather than reading distinct body text.
const CORRUPTION_REPEATED_LINE_RATIO_MAX: f64 = 0.50;

// A multi-column page shows two horizontal bands of text whose vertical spans
// overlap: left-column and right-column lines share the same y range. Fewer
// overlapping pairs than this are incidental; more indicate a genuine column
// split that linear extraction reads out of order.
const READING_ORDER_MIN_OVERLAP_PAIRS: usize = 4;

#[derive(Parser)]
#[command(
    name = "document-mineru-replay",
    about = "Inventory and route document extraction quality (no OCR execution)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Reconcile manifests, original bytes, and existing derivatives into a
    /// deterministic routing inventory.
    Inventory(InventoryArgs),
}

#[derive(Args)]
struct InventoryArgs {
    /// Tracked ingest manifest mapping stable ids to original PDF paths.
    #[arg(long, default_value = "data/papers/documents_ingest/MANIFEST.toml")]
    ingest_manifest: PathBuf,

    /// Provenance header for the licensed papers corpus lane.
    #[arg(long, default_value = "data/papers/corpus/MANIFEST.toml")]
    papers_manifest: PathBuf,

    /// Original-source roots holding the authoritative document bytes.
    #[arg(long = "source-root", default_values_t = [String::from("data/papers/documents_ingest")])]
    source_roots: Vec<String>,

    /// Existing-extraction roots holding docpipe derivatives.
    #[arg(long = "extraction-root", default_values_t = [String::from("data/papers/documents_extracted")])]
    extraction_roots: Vec<String>,

    /// External-source contract registry.
    #[arg(long, default_value = "data/external/SOURCES.toml")]
    sources: PathBuf,

    /// Output inventory path.
    #[arg(long, default_value = "data/output/document_mineru/inventory.toml")]
    out: PathBuf,

    /// Diagnostic filter by canonical id or SHA-256 prefix; does not change the
    /// record schema or ordering.
    #[arg(long)]
    only: Option<String>,

    /// Opt in to the licensed papers-corpus lane. Off by default because the
    /// corpus manifest is a header without per-file entries and carries no
    /// docpipe extractions, so including it would flood the inventory with
    /// unextracted sources.
    #[arg(long, default_value_t = false)]
    include_corpus: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inventory(args) => run_inventory(&args),
    }
}

// ---- Manifest and contract parsing --------------------------------------

#[derive(Deserialize)]
struct IngestManifest {
    #[serde(default)]
    paper: Vec<IngestPaper>,
}

#[derive(Deserialize)]
struct IngestPaper {
    id: String,
    local_pdf: String,
}

#[derive(Deserialize)]
struct SourcesFile {
    #[serde(default)]
    source: Vec<SourceContract>,
}

#[derive(Deserialize)]
struct SourceContract {
    id: String,
    #[serde(default)]
    path_glob: String,
    #[serde(default)]
    access_class: String,
    #[serde(default)]
    manual_manifest_refs: Vec<String>,
}

/// Subset of the docpipe `paper.toml` needed for reconciliation. Arrays default
/// to empty so a partial derivative still parses; a hard TOML error is the
/// malformed-derivative signal.
#[derive(Deserialize, Default)]
struct StoredExtraction {
    #[serde(default)]
    sections: Vec<toml::Value>,
    #[serde(default)]
    equations: Vec<toml::Value>,
    #[serde(default)]
    tables: Vec<toml::Value>,
    #[serde(default)]
    figures: Vec<toml::Value>,
    #[serde(default)]
    full_text: String,
}

// ---- Output schema -------------------------------------------------------

#[derive(Serialize)]
struct Inventory {
    schema_version: String,
    summary: Summary,
    #[serde(rename = "document")]
    document: Vec<DocumentRecord>,
}

#[derive(Serialize, Default)]
struct Summary {
    unique_documents: usize,
    alias_paths: usize,
    valid_pdf: usize,
    blocked_source: usize,
    existing_extractions: usize,
    retain_docpipe: usize,
    mineru_candidate: usize,
    mineru_required: usize,
    manual_review: usize,
    duplicate_identities: usize,
    source_contract_gaps: usize,
    license_unknown: usize,
    reading_order_evaluated: usize,
    reading_order_positioned_text_unavailable: usize,
    per_page_density_evaluated: usize,
    per_page_density_unavailable: usize,
}

#[derive(Serialize)]
struct DocumentRecord {
    canonical_id: String,
    source_sha256: String,
    source_size_bytes: u64,
    media_type: String,
    parse_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    page_count: Option<u64>,
    alias_paths: Vec<String>,
    manifest_ids: Vec<String>,
    corpus_lanes: Vec<String>,
    source_contract: String,
    retrieval_manifest_ref: String,
    license_state: String,
    derivative_tracking: String,
    docpipe_output: String,
    docpipe_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    paper_toml_sha256: Option<String>,
    // Extraction metrics; zero when no derivative reconciles.
    full_text_chars: u64,
    non_whitespace_chars: u64,
    section_count: u64,
    equation_count: u64,
    table_count: u64,
    figure_count: u64,
    image_asset_count: u64,
    mean_nonws_chars_per_page: f64,
    replacement_char_ratio: f64,
    control_char_ratio: f64,
    repeated_line_ratio: f64,
    math_char_ratio: f64,
    table_aligned_line_count: u64,
    // Per-signal evaluation state: evaluated, or a named not-evaluated reason.
    signal_state_scan: String,
    signal_state_formula: String,
    signal_state_table: String,
    signal_state_reading_order: String,
    signal_state_text_corruption: String,
    routing: String,
    reason_codes: Vec<String>,
    manual_review_required: bool,
}

// ---- Core inventory ------------------------------------------------------

/// A present source file grouped under one SHA-256 identity.
struct SourceIdentity {
    sha256: String,
    size_bytes: u64,
    media_type: MediaType,
    page_count: Option<u64>,
    parse_status: ParseStatus,
    aliases: BTreeSet<String>,
    manifest_ids: BTreeSet<String>,
    corpus_lanes: BTreeSet<String>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum MediaType {
    Pdf,
    Html,
    Empty,
    Other,
    Missing,
}

impl MediaType {
    fn as_str(self) -> &'static str {
        match self {
            MediaType::Pdf => "pdf",
            MediaType::Html => "html",
            MediaType::Empty => "empty",
            MediaType::Other => "other",
            MediaType::Missing => "missing",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ParseStatus {
    Parsed,
    Unparseable,
    NotPdf,
    Missing,
}

impl ParseStatus {
    fn as_str(self) -> &'static str {
        match self {
            ParseStatus::Parsed => "parsed",
            ParseStatus::Unparseable => "unparseable",
            ParseStatus::NotPdf => "not_pdf",
            ParseStatus::Missing => "missing",
        }
    }
}

fn run_inventory(args: &InventoryArgs) -> Result<()> {
    if !args.ingest_manifest.exists() {
        bail!(
            "ingest manifest absent: {}",
            args.ingest_manifest.display()
        );
    }
    for root in &args.source_roots {
        if !Path::new(root).exists() {
            bail!("source root absent: {root}");
        }
    }

    let contracts = load_contracts(&args.sources)?;

    // Collect present source files keyed by SHA-256, plus the manifest-id ->
    // SHA map used to enforce the inverse identity constraint.
    let mut identities: BTreeMap<String, SourceIdentity> = BTreeMap::new();
    let mut manifest_id_to_sha: BTreeMap<String, String> = BTreeMap::new();
    let mut missing_sources: Vec<(String, String)> = Vec::new();

    // Ingest lane: manifest maps stable id -> original PDF path.
    let manifest_text = std::fs::read_to_string(&args.ingest_manifest)
        .with_context(|| format!("read ingest manifest {}", args.ingest_manifest.display()))?;
    let manifest: IngestManifest = toml::from_str(&manifest_text)
        .with_context(|| format!("parse ingest manifest {}", args.ingest_manifest.display()))?;
    for paper in &manifest.paper {
        let rel = normalize_rel(&paper.local_pdf);
        let path = Path::new(&rel);
        if !path.exists() {
            missing_sources.push((paper.id.clone(), rel));
            continue;
        }
        ingest_identity(
            path,
            &rel,
            &paper.id,
            "documents_ingest",
            &mut identities,
            &mut manifest_id_to_sha,
        )?;
    }

    // Optional corpus lane: the manifest is a header, so enumerate by walking.
    if args.include_corpus {
        if !args.papers_manifest.exists() {
            bail!(
                "papers corpus manifest absent while --include-corpus set: {}",
                args.papers_manifest.display()
            );
        }
        let corpus_root = args
            .papers_manifest
            .parent()
            .unwrap_or_else(|| Path::new("."));
        for entry in WalkDir::new(corpus_root)
            .sort_by_file_name()
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let rel = normalize_rel(&entry.path().to_string_lossy());
            if rel.ends_with("MANIFEST.toml") {
                continue;
            }
            let stem = entry
                .path()
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| rel.clone());
            ingest_identity(
                entry.path(),
                &rel,
                &format!("corpus:{stem}"),
                "papers_corpus",
                &mut identities,
                &mut manifest_id_to_sha,
            )?;
        }
    }

    // Index existing extractions by directory name so a source id resolves to
    // its docpipe output, and orphaned outputs surface separately.
    let extractions = index_extractions(&args.extraction_roots);
    let mut consumed_extractions: BTreeSet<String> = BTreeSet::new();

    let duplicate_identities = identities
        .values()
        .filter(|identity| identity.aliases.len() > 1)
        .count();

    // Build one output record per SHA identity.
    let mut records: Vec<DocumentRecord> = Vec::new();
    for identity in identities.values() {
        let canonical_id = identity
            .manifest_ids
            .iter()
            .next()
            .cloned()
            .unwrap_or_else(|| identity.sha256.clone());
        // A byte-identical collapse gives one identity several manifest ids,
        // and docpipe extracted each id's slug separately. Consume every one of
        // those extraction dirs so a twin derivative of a known source is not
        // later mislabeled an orphan; reconcile against the first by id order.
        let matched: Vec<(String, PathBuf)> = identity
            .manifest_ids
            .iter()
            .filter_map(|id| extractions.get(id).map(|dir| (id.clone(), dir.clone())))
            .collect();
        for (id, _) in &matched {
            consumed_extractions.insert(id.clone());
        }
        let extraction = matched.into_iter().next();
        let (contract_id, retrieval_ref, access_class) =
            match_contract(&contracts, &identity.aliases);
        records.push(build_record(
            identity,
            &canonical_id,
            extraction.as_ref(),
            &contract_id,
            &retrieval_ref,
            &access_class,
        )?);
    }

    // Missing sources: no bytes to hash, so route as blocked by manifest id.
    for (id, rel) in &missing_sources {
        records.push(missing_record(id, rel));
    }

    // Orphaned extractions: a derivative exists with no matching source.
    for (id, dir) in &extractions {
        if consumed_extractions.contains(id) {
            continue;
        }
        records.push(orphan_record(id, dir));
    }

    // Deterministic order: source SHA then canonical id.
    records.sort_by(|a, b| {
        a.source_sha256
            .cmp(&b.source_sha256)
            .then_with(|| a.canonical_id.cmp(&b.canonical_id))
    });

    // Fail loudly on duplicate canonical ids after ordering.
    let mut seen_ids: BTreeSet<&str> = BTreeSet::new();
    for record in &records {
        if !seen_ids.insert(&record.canonical_id) {
            bail!("duplicate canonical id: {}", record.canonical_id);
        }
    }

    // Optional diagnostic filter; applied after ordering so schema is stable.
    if let Some(filter) = &args.only {
        records.retain(|r| {
            r.canonical_id == *filter || r.source_sha256.starts_with(filter.as_str())
        });
    }

    let summary = summarize(&records, duplicate_identities);
    let inventory = Inventory {
        schema_version: SCHEMA_VERSION.to_string(),
        summary,
        document: records,
    };

    let toml_text = toml::to_string_pretty(&inventory).context("serialize inventory TOML")?;
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }
    std::fs::write(&args.out, &toml_text)
        .with_context(|| format!("write inventory {}", args.out.display()))?;

    let s = &inventory.summary;
    println!(
        "inventory: {} unique documents, {} aliases, {} valid PDFs, {} blocked; \
routing retain={} candidate={} required={} manual={}; \
contract gaps={} license unknown={}; reading-order unavailable={}",
        s.unique_documents,
        s.alias_paths,
        s.valid_pdf,
        s.blocked_source,
        s.retain_docpipe,
        s.mineru_candidate,
        s.mineru_required,
        s.manual_review,
        s.source_contract_gaps,
        s.license_unknown,
        s.reading_order_positioned_text_unavailable,
    );
    println!("wrote {}", args.out.display());
    Ok(())
}

fn ingest_identity(
    path: &Path,
    rel: &str,
    manifest_id: &str,
    lane: &str,
    identities: &mut BTreeMap<String, SourceIdentity>,
    manifest_id_to_sha: &mut BTreeMap<String, String>,
) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read source {rel}"))?;
    let sha = sha256_hex(&bytes);

    // Inverse identity constraint: a manifest id must resolve to exactly one
    // byte stream. Two distinct SHAs for one id is a reconciliation failure.
    if let Some(prev) = manifest_id_to_sha.get(manifest_id) {
        if prev != &sha {
            bail!(
                "manifest id {manifest_id} resolves to two byte streams: {prev} and {sha}"
            );
        }
    } else {
        manifest_id_to_sha.insert(manifest_id.to_string(), sha.clone());
    }

    let media = detect_media_type(&bytes);
    let (page_count, parse_status) = pdf_facts(path, media);

    let identity = identities
        .entry(sha.clone())
        .or_insert_with(|| SourceIdentity {
            sha256: sha.clone(),
            size_bytes: bytes.len() as u64,
            media_type: media,
            page_count,
            parse_status,
            aliases: BTreeSet::new(),
            manifest_ids: BTreeSet::new(),
            corpus_lanes: BTreeSet::new(),
        });
    identity.aliases.insert(rel.to_string());
    identity.manifest_ids.insert(manifest_id.to_string());
    identity.corpus_lanes.insert(lane.to_string());
    Ok(())
}

fn build_record(
    identity: &SourceIdentity,
    canonical_id: &str,
    extraction: Option<&(String, PathBuf)>,
    contract_id: &str,
    retrieval_ref: &str,
    access_class: &str,
) -> Result<DocumentRecord> {
    let (license_state, derivative_tracking) = license_policy(access_class);

    // Reconcile the docpipe derivative when present.
    let mut metrics = ExtractionMetrics::default();
    let mut docpipe_output = String::new();
    let mut docpipe_status = if extraction.is_some() {
        "reconciled"
    } else {
        "absent"
    }
    .to_string();
    let mut paper_toml_sha = None;

    if let Some((_, dir)) = extraction {
        docpipe_output = emit_path(&normalize_rel(&dir.to_string_lossy()));
        let paper_toml = dir.join("paper.toml");
        match std::fs::read_to_string(&paper_toml) {
            Ok(text) => match toml::from_str::<StoredExtraction>(&text) {
                Ok(stored) => {
                    paper_toml_sha = Some(sha256_hex(text.as_bytes()));
                    metrics = extraction_metrics(&stored, dir);
                }
                Err(_) => docpipe_status = "malformed".to_string(),
            },
            Err(_) => docpipe_status = "orphaned".to_string(),
        }
    }

    // Mean density is the live proxy for the per-page median the joined
    // derivative cannot supply; it needs a reconciled extraction and a page
    // count from the source PDF.
    if docpipe_status == "reconciled"
        && let Some(pages) = identity.page_count.filter(|p| *p > 0)
    {
        metrics.mean_nonws_chars_per_page =
            round6(metrics.non_whitespace_chars as f64 / pages as f64);
    }

    let signals = evaluate_signals(identity, &docpipe_status);
    let (routing, reason_codes, manual_review_required) =
        classify(identity, &metrics, &docpipe_status, extraction.is_some(), &signals);

    Ok(DocumentRecord {
        canonical_id: canonical_id.to_string(),
        source_sha256: identity.sha256.clone(),
        source_size_bytes: identity.size_bytes,
        media_type: identity.media_type.as_str().to_string(),
        parse_status: identity.parse_status.as_str().to_string(),
        page_count: identity.page_count,
        alias_paths: identity.aliases.iter().map(|a| emit_path(a)).collect(),
        manifest_ids: identity.manifest_ids.iter().cloned().collect(),
        corpus_lanes: identity.corpus_lanes.iter().cloned().collect(),
        source_contract: contract_id.to_string(),
        retrieval_manifest_ref: retrieval_ref.to_string(),
        license_state,
        derivative_tracking,
        docpipe_output,
        docpipe_status,
        paper_toml_sha256: paper_toml_sha,
        full_text_chars: metrics.full_text_chars,
        non_whitespace_chars: metrics.non_whitespace_chars,
        section_count: metrics.section_count,
        equation_count: metrics.equation_count,
        table_count: metrics.table_count,
        figure_count: metrics.figure_count,
        image_asset_count: metrics.image_asset_count,
        mean_nonws_chars_per_page: metrics.mean_nonws_chars_per_page,
        replacement_char_ratio: metrics.replacement_char_ratio,
        control_char_ratio: metrics.control_char_ratio,
        repeated_line_ratio: metrics.repeated_line_ratio,
        math_char_ratio: metrics.math_char_ratio,
        table_aligned_line_count: metrics.table_aligned_line_count,
        signal_state_scan: signals.scan.clone(),
        signal_state_formula: signals.formula.clone(),
        signal_state_table: signals.table.clone(),
        signal_state_reading_order: signals.reading_order.clone(),
        signal_state_text_corruption: signals.text_corruption.clone(),
        routing: routing.to_string(),
        reason_codes,
        manual_review_required,
    })
}

#[derive(Default)]
struct ExtractionMetrics {
    full_text_chars: u64,
    non_whitespace_chars: u64,
    section_count: u64,
    equation_count: u64,
    table_count: u64,
    figure_count: u64,
    image_asset_count: u64,
    mean_nonws_chars_per_page: f64,
    replacement_char_ratio: f64,
    control_char_ratio: f64,
    repeated_line_ratio: f64,
    math_char_ratio: f64,
    table_aligned_line_count: u64,
}

fn extraction_metrics(stored: &StoredExtraction, dir: &Path) -> ExtractionMetrics {
    let text = &stored.full_text;
    let non_ws = non_whitespace_count(text);
    let image_dir = dir.join("images");
    let image_asset_count = if image_dir.is_dir() {
        WalkDir::new(&image_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .count() as u64
    } else {
        0
    };
    ExtractionMetrics {
        full_text_chars: text.chars().count() as u64,
        non_whitespace_chars: non_ws as u64,
        section_count: stored.sections.len() as u64,
        equation_count: stored.equations.len() as u64,
        table_count: stored.tables.len() as u64,
        figure_count: stored.figures.len() as u64,
        image_asset_count,
        mean_nonws_chars_per_page: 0.0, // set by evaluate_signals using page count
        replacement_char_ratio: round6(replacement_char_ratio(text)),
        control_char_ratio: round6(control_char_ratio(text)),
        repeated_line_ratio: round6(repeated_line_ratio(text)),
        math_char_ratio: round6(math_char_ratio(text)),
        table_aligned_line_count: table_aligned_line_count(text) as u64,
    }
}

struct SignalStates {
    scan: String,
    formula: String,
    table: String,
    reading_order: String,
    text_corruption: String,
}

fn evaluate_signals(identity: &SourceIdentity, docpipe_status: &str) -> SignalStates {
    let reconciled = docpipe_status == "reconciled";
    let per_page = identity.page_count.is_some() && reconciled;
    SignalStates {
        scan: state(per_page, "page_count_unavailable"),
        formula: state(reconciled, "no_reconciled_extraction"),
        table: state(reconciled, "no_reconciled_extraction"),
        // Reading order needs positioned spans; the stored derivative joins
        // pages into one string and carries no coordinate geometry.
        reading_order: "positioned_text_unavailable".to_string(),
        text_corruption: state(reconciled, "no_reconciled_extraction"),
    }
}

fn state(evaluated: bool, unavailable_reason: &str) -> String {
    if evaluated {
        "evaluated".to_string()
    } else {
        unavailable_reason.to_string()
    }
}

#[allow(clippy::too_many_lines)]
fn classify(
    identity: &SourceIdentity,
    metrics: &ExtractionMetrics,
    docpipe_status: &str,
    has_extraction: bool,
    signals: &SignalStates,
) -> (&'static str, Vec<String>, bool) {
    let mut reasons: BTreeSet<String> = BTreeSet::new();

    // Blocked: the original cannot enter either pipeline.
    match identity.media_type {
        MediaType::Missing => {
            reasons.insert("source_missing".to_string());
            return ("blocked_source", sorted(reasons), false);
        }
        MediaType::Empty => {
            reasons.insert("empty_source".to_string());
            return ("blocked_source", sorted(reasons), false);
        }
        MediaType::Html | MediaType::Other => {
            reasons.insert("extension_signature_mismatch".to_string());
            return ("blocked_source", sorted(reasons), false);
        }
        MediaType::Pdf => {}
    }
    // A PDF the pure-Rust parser rejects is not proven incapable: pdfium and the
    // OCR pipeline are more permissive, and the stored extraction may already be
    // clean. Record the rejection through parse_status and route on the
    // extraction rather than blocking the document here.

    // Contradiction: a reconciled extraction with substantial text over a PDF
    // the parser reports as zero pages cannot be interpreted deterministically.
    if docpipe_status == "reconciled"
        && metrics.non_whitespace_chars > 0
        && identity.page_count == Some(0)
    {
        reasons.insert("page_count_text_contradiction".to_string());
        return ("manual_review", sorted(reasons), true);
    }

    // Required: valid PDF whose extraction is absent, malformed, or textless.
    if !has_extraction {
        reasons.insert("extraction_absent".to_string());
        return ("mineru_required", sorted(reasons), false);
    }
    if docpipe_status == "malformed" {
        reasons.insert("extraction_malformed".to_string());
        return ("mineru_required", sorted(reasons), false);
    }
    if docpipe_status == "orphaned" {
        reasons.insert("extraction_orphaned".to_string());
        return ("mineru_required", sorted(reasons), false);
    }
    // Zero native text is a scan with no citable text layer; figure placeholders
    // do not make it usable, so it needs OCR regardless of extracted images.
    if metrics.non_whitespace_chars == 0 {
        reasons.insert("no_usable_text".to_string());
        return ("mineru_required", sorted(reasons), false);
    }

    // Candidate signals over a usable extraction.
    if signals.scan == "evaluated" && metrics.mean_nonws_chars_per_page < SCAN_MIN_NONWS_PER_PAGE {
        reasons.insert("scan_risk".to_string());
    }
    if signals.formula == "evaluated"
        && metrics.math_char_ratio >= FORMULA_MATH_CHAR_RATIO_MIN
        && metrics.equation_count == 0
    {
        reasons.insert("formula_risk".to_string());
    }
    if signals.table == "evaluated"
        && metrics.table_aligned_line_count as usize >= TABLE_ALIGNED_LINES_MIN
        && metrics.table_count == 0
    {
        reasons.insert("table_risk".to_string());
    }
    if signals.text_corruption == "evaluated"
        && (metrics.replacement_char_ratio > CORRUPTION_REPLACEMENT_RATIO_MAX
            || metrics.control_char_ratio > CORRUPTION_CONTROL_RATIO_MAX
            || metrics.repeated_line_ratio > CORRUPTION_REPEATED_LINE_RATIO_MAX)
    {
        reasons.insert("text_corruption_risk".to_string());
    }

    if reasons.is_empty() {
        ("retain_docpipe", sorted(reasons), false)
    } else {
        ("mineru_candidate", sorted(reasons), false)
    }
}

// ---- Records for missing sources and orphaned extractions ----------------

fn missing_record(id: &str, rel: &str) -> DocumentRecord {
    DocumentRecord {
        canonical_id: id.to_string(),
        source_sha256: String::new(),
        source_size_bytes: 0,
        media_type: MediaType::Missing.as_str().to_string(),
        parse_status: ParseStatus::Missing.as_str().to_string(),
        page_count: None,
        alias_paths: vec![emit_path(rel)],
        manifest_ids: vec![id.to_string()],
        corpus_lanes: vec!["documents_ingest".to_string()],
        source_contract: String::new(),
        retrieval_manifest_ref: String::new(),
        license_state: "unknown".to_string(),
        derivative_tracking: "no_new_tracked_content".to_string(),
        docpipe_output: String::new(),
        docpipe_status: "absent".to_string(),
        paper_toml_sha256: None,
        full_text_chars: 0,
        non_whitespace_chars: 0,
        section_count: 0,
        equation_count: 0,
        table_count: 0,
        figure_count: 0,
        image_asset_count: 0,
        mean_nonws_chars_per_page: 0.0,
        replacement_char_ratio: 0.0,
        control_char_ratio: 0.0,
        repeated_line_ratio: 0.0,
        math_char_ratio: 0.0,
        table_aligned_line_count: 0,
        signal_state_scan: "source_missing".to_string(),
        signal_state_formula: "source_missing".to_string(),
        signal_state_table: "source_missing".to_string(),
        signal_state_reading_order: "source_missing".to_string(),
        signal_state_text_corruption: "source_missing".to_string(),
        routing: "blocked_source".to_string(),
        reason_codes: vec!["source_missing".to_string()],
        manual_review_required: false,
    }
}

fn orphan_record(id: &str, dir: &Path) -> DocumentRecord {
    DocumentRecord {
        canonical_id: format!("orphan:{id}"),
        source_sha256: String::new(),
        source_size_bytes: 0,
        media_type: MediaType::Missing.as_str().to_string(),
        parse_status: ParseStatus::Missing.as_str().to_string(),
        page_count: None,
        alias_paths: Vec::new(),
        manifest_ids: vec![id.to_string()],
        corpus_lanes: Vec::new(),
        source_contract: String::new(),
        retrieval_manifest_ref: String::new(),
        license_state: "unknown".to_string(),
        derivative_tracking: "no_new_tracked_content".to_string(),
        docpipe_output: emit_path(&normalize_rel(&dir.to_string_lossy())),
        docpipe_status: "orphaned".to_string(),
        paper_toml_sha256: None,
        full_text_chars: 0,
        non_whitespace_chars: 0,
        section_count: 0,
        equation_count: 0,
        table_count: 0,
        figure_count: 0,
        image_asset_count: 0,
        mean_nonws_chars_per_page: 0.0,
        replacement_char_ratio: 0.0,
        control_char_ratio: 0.0,
        repeated_line_ratio: 0.0,
        math_char_ratio: 0.0,
        table_aligned_line_count: 0,
        signal_state_scan: "no_source".to_string(),
        signal_state_formula: "no_source".to_string(),
        signal_state_table: "no_source".to_string(),
        signal_state_reading_order: "no_source".to_string(),
        signal_state_text_corruption: "no_source".to_string(),
        routing: "manual_review".to_string(),
        reason_codes: vec!["derivative_without_source".to_string()],
        manual_review_required: true,
    }
}

// ---- Contract matching and license policy --------------------------------

fn load_contracts(path: &Path) -> Result<Vec<SourceContract>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read sources {}", path.display()))?;
    let parsed: SourcesFile =
        toml::from_str(&text).with_context(|| format!("parse sources {}", path.display()))?;
    Ok(parsed.source)
}

/// Return (contract id, retrieval manifest ref, access class) for the first
/// contract whose glob covers any alias. Empty contract id marks an uncovered
/// source.
fn match_contract(
    contracts: &[SourceContract],
    aliases: &BTreeSet<String>,
) -> (String, String, String) {
    for contract in contracts {
        if contract.path_glob.is_empty() {
            continue;
        }
        let Ok(pattern) = glob::Pattern::new(&contract.path_glob) else {
            continue;
        };
        if aliases.iter().any(|a| pattern.matches(a)) {
            let retrieval = contract.manual_manifest_refs.first().cloned().unwrap_or_default();
            return (contract.id.clone(), retrieval, contract.access_class.clone());
        }
    }
    (String::new(), String::new(), String::new())
}

fn license_policy(access_class: &str) -> (String, String) {
    let license_state = match access_class {
        "public" | "public_api" | "open" => "known_permissive",
        "manual_or_licensed" => "known_restricted",
        _ => "unknown",
    };
    // Only a known-permissive source may gain new tracked compact derivatives in
    // a later tranche; everything else defaults to tracking nothing new.
    let tracking = if license_state == "known_permissive" {
        "track_compact"
    } else {
        "no_new_tracked_content"
    };
    (license_state.to_string(), tracking.to_string())
}

// ---- Extraction indexing -------------------------------------------------

/// Map extraction directory name -> directory path across all roots.
fn index_extractions(roots: &[String]) -> BTreeMap<String, PathBuf> {
    let mut map = BTreeMap::new();
    for root in roots {
        let root_path = Path::new(root);
        if !root_path.is_dir() {
            continue;
        }
        let mut entries: Vec<PathBuf> = std::fs::read_dir(root_path)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect()
            })
            .unwrap_or_default();
        entries.sort();
        for dir in entries {
            if let Some(name) = dir.file_name().map(|n| n.to_string_lossy().to_string()) {
                map.entry(name).or_insert(dir);
            }
        }
    }
    map
}

fn summarize(records: &[DocumentRecord], duplicate_identities: usize) -> Summary {
    let mut s = Summary {
        duplicate_identities,
        ..Summary::default()
    };
    for r in records {
        // Orphan/missing records carry no source SHA; count real identities only.
        if !r.source_sha256.is_empty() {
            s.unique_documents += 1;
        }
        s.alias_paths += r.alias_paths.len();
        if r.media_type == "pdf" && r.parse_status == "parsed" {
            s.valid_pdf += 1;
        }
        if r.docpipe_status == "reconciled" {
            s.existing_extractions += 1;
        }
        if r.source_contract.is_empty() && !r.source_sha256.is_empty() {
            s.source_contract_gaps += 1;
        }
        if r.license_state == "unknown" && !r.source_sha256.is_empty() {
            s.license_unknown += 1;
        }
        if r.signal_state_reading_order == "evaluated" {
            s.reading_order_evaluated += 1;
        } else if r.signal_state_reading_order == "positioned_text_unavailable" {
            s.reading_order_positioned_text_unavailable += 1;
        }
        if r.signal_state_scan == "evaluated" {
            s.per_page_density_evaluated += 1;
        } else {
            s.per_page_density_unavailable += 1;
        }
        match r.routing.as_str() {
            "retain_docpipe" => s.retain_docpipe += 1,
            "mineru_candidate" => s.mineru_candidate += 1,
            "mineru_required" => s.mineru_required += 1,
            "blocked_source" => s.blocked_source += 1,
            "manual_review" => s.manual_review += 1,
            _ => {}
        }
    }
    s
}

// ---- Pure helpers --------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Normalize a path to `/` separators, preserving the exact bytes so it still
/// resolves on disk. This is the value used for existence checks, reads, and
/// directory walks.
fn normalize_rel(path: &str) -> String {
    path.replace('\\', "/")
}

/// Encode a normalized path for emission into the tracked inventory: non-ASCII
/// bytes and any literal `%` become `%XX` so the artifact stays ASCII while the
/// alias remains a reversible pointer. The SHA-256 carries the exact identity;
/// this is applied only when a path string enters an output record, never to a
/// path used to touch the filesystem.
fn emit_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for &byte in path.as_bytes() {
        if byte < 0x80 && byte != b'%' {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}

fn sorted(set: BTreeSet<String>) -> Vec<String> {
    set.into_iter().collect()
}

/// Detect media type from bytes, never from the filename suffix.
fn detect_media_type(bytes: &[u8]) -> MediaType {
    if bytes.is_empty() {
        return MediaType::Empty;
    }
    if bytes.starts_with(b"%PDF-") {
        return MediaType::Pdf;
    }
    // Leading whitespace then an angle bracket, or a DOCTYPE/html token, marks an
    // HTML body mislabeled with a document suffix.
    let head_len = bytes.len().min(512);
    let head = String::from_utf8_lossy(&bytes[..head_len]);
    let trimmed = head.trim_start();
    let lower = trimmed.to_ascii_lowercase();
    if trimmed.starts_with('<')
        || lower.contains("<!doctype html")
        || lower.contains("<html")
    {
        return MediaType::Html;
    }
    MediaType::Other
}

/// Page count and parse status from the pure-Rust lopdf parser. A non-PDF media
/// type never reaches lopdf.
fn pdf_facts(path: &Path, media: MediaType) -> (Option<u64>, ParseStatus) {
    if media != MediaType::Pdf {
        return (None, ParseStatus::NotPdf);
    }
    match lopdf::Document::load(path) {
        Ok(doc) => (Some(doc.get_pages().len() as u64), ParseStatus::Parsed),
        Err(_) => (None, ParseStatus::Unparseable),
    }
}

fn non_whitespace_count(text: &str) -> usize {
    text.chars().filter(|c| !c.is_whitespace()).count()
}

fn replacement_char_ratio(text: &str) -> f64 {
    let total = text.chars().count();
    if total == 0 {
        return 0.0;
    }
    let count = text.chars().filter(|&c| c == '\u{FFFD}').count();
    count as f64 / total as f64
}

fn control_char_ratio(text: &str) -> f64 {
    let total = text.chars().count();
    if total == 0 {
        return 0.0;
    }
    let count = text
        .chars()
        .filter(|&c| c.is_control() && c != '\t' && c != '\n' && c != '\r')
        .count();
    count as f64 / total as f64
}

fn repeated_line_ratio(text: &str) -> f64 {
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return 0.0;
    }
    let unique: BTreeSet<&str> = lines.iter().copied().collect();
    let duplicates = lines.len() - unique.len();
    duplicates as f64 / lines.len() as f64
}

fn is_math_char(c: char) -> bool {
    matches!(
        c,
        '=' | '+'
            | '\\'
            | '^'
            | '_'
            | '<'
            | '>'
            | '\u{2211}' // n-ary summation
            | '\u{222B}' // integral
            | '\u{2202}' // partial differential
            | '\u{221A}' // square root
            | '\u{00B1}' // plus-minus
            | '\u{00D7}' // multiplication
            | '\u{00F7}' // division
            | '\u{2264}' // less-than or equal
            | '\u{2265}' // greater-than or equal
            | '\u{2260}' // not equal
            | '\u{2248}' // almost equal
            | '\u{221E}' // infinity
    ) || ('\u{0391}'..='\u{03C9}').contains(&c) // Greek block
}

fn math_char_ratio(text: &str) -> f64 {
    let non_ws = non_whitespace_count(text);
    if non_ws == 0 {
        return 0.0;
    }
    let count = text.chars().filter(|&c| is_math_char(c)).count();
    count as f64 / non_ws as f64
}

/// Count lines that carry two or more multi-space gutters, the residue of
/// column-aligned tabular layout after text extraction.
fn table_aligned_line_count(text: &str) -> usize {
    text.lines().filter(|line| count_gutters(line) >= 2).count()
}

fn count_gutters(line: &str) -> usize {
    let mut gutters = 0;
    let mut run = 0;
    for c in line.chars() {
        if c == ' ' {
            run += 1;
        } else {
            if run >= 2 {
                gutters += 1;
            }
            run = 0;
        }
    }
    gutters
}

/// Median non-whitespace characters per page. Pure over per-page text so a later
/// tranche with per-page extraction can feed it; the stored joined derivative
/// cannot, which is why the live path records the density as unavailable. Held
/// as a tested capability until per-page extraction is wired.
#[allow(dead_code)]
fn median_nonws_per_page(pages: &[&str]) -> f64 {
    if pages.is_empty() {
        return 0.0;
    }
    let mut counts: Vec<usize> = pages.iter().map(|p| non_whitespace_count(p)).collect();
    counts.sort_unstable();
    let mid = counts.len() / 2;
    if counts.len() % 2 == 1 {
        counts[mid] as f64
    } else {
        (counts[mid - 1] + counts[mid]) as f64 / 2.0
    }
}

/// Detect a multi-column page from positioned spans: two horizontal bands whose
/// vertical ranges overlap. Pure over spans so the reading-order rule is tested
/// even though the stored derivative carries no coordinates. Held as a tested
/// capability until positioned-text extraction is wired.
#[allow(dead_code)]
fn detect_multicolumn(spans: &[PositionedSpan]) -> bool {
    if spans.len() < 4 {
        return false;
    }
    let xs: Vec<f64> = spans.iter().map(|s| s.x).collect();
    let min_x = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let split = (min_x + max_x) / 2.0;
    let left: Vec<&PositionedSpan> = spans.iter().filter(|s| s.x < split).collect();
    let right: Vec<&PositionedSpan> = spans.iter().filter(|s| s.x >= split).collect();
    if left.is_empty() || right.is_empty() {
        return false;
    }
    let mut overlaps = 0;
    for l in &left {
        for r in &right {
            let l_top = l.y;
            let l_bot = l.y + l.height;
            let r_top = r.y;
            let r_bot = r.y + r.height;
            if l_top < r_bot && r_top < l_bot {
                overlaps += 1;
            }
        }
    }
    overlaps >= READING_ORDER_MIN_OVERLAP_PAIRS
}

/// Minimal positioned span for the reading-order rule; mirrors the coordinate
/// fields of docpipe's positioned text.
#[allow(dead_code)]
struct PositionedSpan {
    x: f64,
    y: f64,
    height: f64,
}

#[cfg(test)]
#[path = "document_mineru_replay/tests.rs"]
mod tests;
