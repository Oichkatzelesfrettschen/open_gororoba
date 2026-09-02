use anyhow::{Context, Result, bail};
use chrono::Utc;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};
use toml::Value;
use walkdir::WalkDir;

const ARTIFACT_LOCAL_PREFIXES: &[&str] = &[
    "archive/",
    "data/",
    "papers/",
    "registry/knowledge/artifacts/",
];

const NON_AUTHORITATIVE_REGISTRY_PREFIXES: &[&str] =
    &["registry/knowledge/", "registry/source_lanes/"];

const NON_AUTHORITATIVE_REGISTRY_EXACT: &[&str] = &[
    "registry/embedded_markdown_chunks.toml",
    "registry/embedded_markdown_payloads.toml",
    "registry/bibliography_normalized.toml",
    "registry/markdown_corpus_registry.toml",
    "registry/markdown_governance.toml",
    "registry/markdown_inventory.toml",
    "registry/markdown_origin_audit.toml",
    "registry/markdown_owner_map.toml",
    "registry/markdown_payload_chunks.toml",
    "registry/markdown_payloads.toml",
    "registry/source_infrastructure.toml",
    "registry/toml_inventory.toml",
];

const NON_AUTHORITATIVE_REPORT_PREFIXES: &[&str] = &[
    "reports/blocked_artifact_retry_plan_",
    "reports/blocked_artifact_recovery_attempts_",
    "reports/data_origin_audit_",
    "reports/external_redownload_audit_",
    "reports/data_semantic_validate_external_",
    "reports/literature_inventory_",
];

const TITLE_KEYS: &[&str] = &[
    "title",
    "paper_title",
    "name",
    "citation",
    "citation_markdown",
    "reference",
];

const CITATION_KEYS: &[&str] = &["citation", "citation_markdown", "reference", "summary"];

const ID_KEYS: &[&str] = &["id", "key", "slug", "paper_id", "artifact_id"];

const REFERENCE_HOST_HINTS: &[&str] = &[
    "arxiv.org",
    "export.arxiv.org",
    "academia.edu",
    "scispace.com",
    "doi.org",
    "core.ac.uk",
    "sciencedirect.com",
    "linkinghub.elsevier.com",
    "msp.org",
    "projecteuclid.org",
    "mathnet.ru",
    "researchgate.net",
    "tandfonline.com",
    "mdpi.com",
    "link.springer.com",
    "springer.com",
    "degruyter.com",
    "cambridge.org",
    "iopscience.iop.org",
    "numdam.org",
    "aimspress.com",
    "dergipark.org.tr",
    "dr.lib.iastate.edu",
    "repository.essex.ac.uk",
    "openreview.net",
    "archive.org",
    "web.archive.org",
    "osf.io",
    "gutenberg.org",
    "jvoight.github.io",
    "kconrad.math.uconn.edu",
    "wstein.org",
    "journals.aps.org",
    "harvest.aps.org",
    "royalsocietypublishing.org",
    "zenodo.org",
    "isidore.co",
    "cms.math.ca",
    "bibliotekanauki.pl",
    "pldml.icm.edu.pl",
    "sciendo.com",
    "jwbales.us",
    "journals.sagepub.com",
    "pubmed.ncbi.nlm.nih.gov",
    "raw.githubusercontent.com",
    "inspirehep.net",
];

const DATASET_EXTENSIONS: &[&str] = &[
    ".csv", ".tsv", ".json", ".jsonl", ".parquet", ".h5", ".hdf5", ".nc", ".npy", ".npz",
    ".feather", ".xlsx", ".xls",
];

const SLIDE_ARTIFACT_EXTENSIONS: &[&str] = &[
    ".ppt", ".pptx", ".odp", ".key", ".zip", ".tar", ".gz", ".7z", ".ipynb", ".doc", ".docx",
];

const PDF_EXTENSIONS: &[&str] = &[".pdf"];

const LANE_ORDER: &[&str] = &[
    "datasets",
    "slides_artifacts",
    "papers_pdf",
    "web_references",
];

const BEST_PRACTICE_SOURCES: &[&str] = &[
    "https://www.w3.org/TR/prov-overview/",
    "https://www.nature.com/articles/sdata201618",
    "https://doi.org/10.25490/a97f-egyk",
    "https://schema.datacite.org/meta/kernel-4.5/",
    "https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files",
    "https://openlineage.io/docs/",
];

/// `downloaded` asserts the repository itself carries the bytes, so it holds
/// only when git tracks the path (a Git LFS pointer counts). A file present in
/// one checkout and gitignored everywhere else is
/// `remotely_materializable`: the row records canonical_url, sha256,
/// byte_length, retrieval_command and license_disposition, and per-host
/// presence moves to the gitignored materialization manifest.
const VALID_STATUSES: &[&str] = &[
    "downloaded",
    "remotely_materializable",
    "downloadable",
    "blocked",
    "citation_only_no_link",
    "unverified",
];

#[derive(Clone, Debug, Default)]
struct LinkObservation {
    status: String,
}

#[derive(Clone, Debug, Default)]
struct CandidateRecord {
    source_kind: String,
    source_ref: String,
    identity_override: Option<String>,
    title: String,
    citation: String,
    dois: Vec<String>,
    links: Vec<String>,
    local_paths: Vec<String>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct UnifiedArtifact {
    key: String,
    title: String,
    citation: String,
    source_kinds: Vec<String>,
    source_refs: Vec<String>,
    doi_list: Vec<String>,
    links: Vec<String>,
    local_paths: Vec<String>,
    notes: Vec<String>,
    working_mirrors: Vec<String>,
    working_pdf_mirrors: Vec<String>,
    nonworking_mirrors: Vec<String>,
    unverified_mirrors: Vec<String>,
    downloaded_paths: Vec<String>,
    host_only_paths: Vec<String>,
    lane: String,
    sha256: String,
    byte_length: u64,
    retrieval_command: String,
    license_disposition: String,
    canonical_functional_url: String,
    canonical_download_path: String,
    status: String,
    minimum_requirement_met: bool,
    manual_intervention_required: bool,
    manual_intervention_reason: String,
}

#[derive(Clone, Debug, Default)]
pub struct BuildSummary {
    pub artifact_count: usize,
    pub downloaded_count: usize,
    pub remotely_materializable_count: usize,
    pub materializable_without_url_count: usize,
    /// Rows the checked-in catalog carried whose keys this host's scan did not
    /// observe; they are re-seeded from their durable fields.
    pub retained_prior_row_count: usize,
    pub row_counts: Vec<RowCountReport>,
    /// Per-host presence for every observed path. Host state, so it never
    /// reaches a committed registry; the caller writes the gitignored manifest.
    pub host_materialization: Vec<HostMaterializationRow>,
}

#[derive(Clone, Debug, Default)]
pub struct VerifySummary {
    pub artifact_count: usize,
    pub downloaded_count: usize,
    pub remotely_materializable_count: usize,
    pub downloadable_count: usize,
    pub blocked_count: usize,
    pub citation_only_count: usize,
    pub unverified_count: usize,
    pub missing_minimum_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SourceInfrastructureSummary {
    pub total_artifact_count: usize,
    pub lane_counts: BTreeMap<String, usize>,
}

pub fn default_repo_root() -> PathBuf {
    repo_root::resolve!()
}

// Cached Regex factories (url_re, url_inline_re, doi_re, bib_entry_re)
// and pure text helpers (ascii_sanitize, escape_toml, render_list,
// assert_ascii, slug) live in the `text_helpers` submodule. The #[path]
// attribute is required because source_provenance.rs is itself loaded
// via #[path] from the provenance_ops crate (see
// crates/provenance_ops/src/lib.rs); the relative submodule lookup
// otherwise resolves against provenance_ops/src/source_provenance/,
// not the canonical gororoba_cli_data/src/source_provenance/ location.
#[path = "source_provenance/text_helpers.rs"]
mod text_helpers;
use text_helpers::{assert_ascii, escape_toml, render_list, slug, url_re};

// DOI parsing/identity helpers (normalize_doi, extract_dois,
// doi_to_url, doi_from_url, extract_dois_from_urls) live in the
// `doi_helpers` submodule. Uses the same #[path] indirection as
// text_helpers because source_provenance.rs is loaded via #[path]
// from the provenance_ops crate.
#[path = "source_provenance/doi_helpers.rs"]
mod doi_helpers;
use doi_helpers::{doi_to_url, extract_dois_from_urls, normalize_doi};

// URL canonicalization helpers (strip_url_wrappers,
// rewrite_arxiv_typo_prefix, apply_host_specific_rewrites,
// filter_tracking_query_params, normalize_url,
// is_non_reference_service_url, find_urls) live in the `url_helpers`
// submodule.
#[path = "source_provenance/url_helpers.rs"]
mod url_helpers;
use url_helpers::{find_urls, normalize_url};

// Reference identity + alias-expansion helpers
// (arxiv_equivalent_urls, strip_arxiv_version, core_id_from_url,
// mdpi_path_looks_article, cambridge_content_id,
// canonical_identity_url, expand_reference_aliases) live in the
// `identity_aliases` submodule.
#[path = "source_provenance/identity_aliases.rs"]
mod identity_aliases;
use identity_aliases::{cambridge_content_id, canonical_identity_url, expand_reference_aliases};

// File-loading helpers (load_toml_value, read_text_lossy,
// read_tsv_rows, derive_status) live in the `file_io` submodule.
#[path = "source_provenance/file_io.rs"]
mod file_io;
use file_io::load_toml_value;

// Download-attempt observation pipeline (LinkMap type alias plus
// collect_link_observations, merge_sqlite_download_observations,
// derive_attempt_status, collect_download_map,
// provenance_intake_roots, extend_download_map_from_local_artifacts,
// register_download_aliases) lives in the `download_pipeline`
// submodule.
#[path = "source_provenance/download_pipeline.rs"]
mod download_pipeline;
use download_pipeline::{
    collect_download_map, collect_link_observations, extend_download_map_from_local_artifacts,
};

// URL/path classification predicates (looks_like_reference_url,
// is_citation_locator_url, key_is_citation_locator,
// is_artifact_local_path) live in the `reference_predicates`
// submodule.
#[path = "source_provenance/reference_predicates.rs"]
mod reference_predicates;
use reference_predicates::{
    is_artifact_local_path, is_citation_locator_url, key_is_citation_locator,
};

// CandidateRecord extraction from TOML / BibTeX / text source files
// (extract_candidates_from_source_file dispatcher + the 3 format
// extractors + classify_toml_field_key + pick_first_str) lives in
// the `candidate_extract` submodule.
#[path = "source_provenance/candidate_extract.rs"]
mod candidate_extract;
use candidate_extract::extract_candidates_from_source_file;

// Two-phase output writer and the retention predicate that separates
// repository truth from per-host materialization. Same #[path] indirection.
#[path = "source_provenance/staged_write.rs"]
pub mod staged_write;
pub use staged_write::{DEFAULT_SHRINK_THRESHOLD, RowCountReport, ShrinkPolicy, StagedWriteSet};

#[path = "source_provenance/artifact_retention.rs"]
pub mod artifact_retention;
pub use artifact_retention::{
    HostMaterializationRow, RetentionSet, observe_host_materialization,
    render_host_materialization, retrieval_command, write_host_materialization,
};

fn normalize_identity_hint(hint: &str) -> String {
    let trimmed = hint.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(url) = trimmed.strip_prefix("url:") {
        let normalized = normalize_url(url);
        if !normalized.is_empty() {
            if let Some(content_id) = cambridge_content_id(&normalized) {
                return format!("cambridge:{}", content_id.to_ascii_lowercase());
            }
            return format!("url:{}", normalized.to_ascii_lowercase());
        }
    }
    if let Some(doi) = trimmed.strip_prefix("doi:") {
        let normalized = normalize_doi(doi);
        if !normalized.is_empty() {
            return format!("doi:{}", normalized.to_ascii_lowercase());
        }
    }
    if url_re().is_match(trimmed) {
        let normalized = normalize_url(trimmed);
        if !normalized.is_empty() {
            if let Some(content_id) = cambridge_content_id(&normalized) {
                return format!("cambridge:{}", content_id.to_ascii_lowercase());
            }
            return format!("url:{}", normalized.to_ascii_lowercase());
        }
    }
    let normalized_doi = normalize_doi(trimmed);
    if !normalized_doi.is_empty() {
        return format!("doi:{}", normalized_doi.to_ascii_lowercase());
    }
    trimmed.to_string()
}

fn extract_strings(value: &Value) -> Vec<String> {
    match value {
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            }
        }
        Value::Array(items) => items
            .iter()
            .filter_map(|item| item.as_str().map(|s| s.trim().to_string()))
            .filter(|item| !item.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_urls(value: &Value) -> Vec<String> {
    let mut urls = Vec::new();
    for text in extract_strings(value) {
        for part in text.split('|') {
            let normalized = normalize_url(part);
            if url_re().is_match(&normalized) {
                urls.push(normalized);
            } else {
                urls.extend(find_urls(part));
            }
        }
    }
    dedupe(urls)
}

fn dedupe(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() || !seen.insert(trimmed.clone()) {
            continue;
        }
        out.push(trimmed);
    }
    out
}

fn extract_local_paths(value: &Value, repo_root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for text in extract_strings(value) {
        for part in text.split('|') {
            let candidate = part.trim();
            if candidate.is_empty() || !candidate.is_ascii() || url_re().is_match(candidate) {
                continue;
            }
            let path = Path::new(candidate);
            if path.is_absolute() {
                if path.exists()
                    && let Ok(relative) = path.strip_prefix(repo_root)
                {
                    let rel = relative.to_string_lossy().replace('\\', "/");
                    if is_artifact_local_path(&rel) {
                        out.push(rel);
                    }
                }
                continue;
            }
            let full = repo_root.join(path);
            if full.exists() {
                let rel = path.to_string_lossy().replace('\\', "/");
                if is_artifact_local_path(&rel) {
                    out.push(rel);
                }
            }
        }
    }
    dedupe(out)
}

fn discover_candidate_source_files(repo_root: &Path) -> Vec<PathBuf> {
    let suffixes = [".toml", ".bib", ".bibtex", ".md", ".txt", ".rst"];
    let text_suffixes = [".md", ".txt", ".rst"];
    let text_keywords = [
        "source",
        "bibli",
        "reconcil",
        "artifact",
        "intake",
        "cayley",
        "sedenion",
        "octonion",
        "quaternion",
        "mirror",
        "provenance",
    ];
    let allowed_prefixes = ["registry/", "reports/", "docs/", "papers/", "data/papers/"];
    let excluded_prefixes = [
        ".git/",
        "target/",
        "data/external/intake/",
        "data/external/raw/",
        "data/external/cache/",
        "registry/source_lanes/",
    ];
    let excluded_exact = [
        "registry/artifact_source_of_truth.toml",
        "reports/artifact_source_of_truth_reconciliation_2026_02_15.toml",
        "reports/artifact_blocked_links_2026_02_15.tsv",
        "reports/artifact_missing_minimum_2026_02_15.tsv",
    ];
    let mut paths = BTreeSet::new();
    let walker = WalkDir::new(repo_root).into_iter().filter_entry(|e| {
        let rel = e
            .path()
            .strip_prefix(repo_root)
            .unwrap_or(e.path())
            .to_string_lossy()
            .replace('\\', "/");
        if rel.is_empty() || rel == "." {
            return true;
        }
        let rel_with_slash = format!("{rel}/");
        !excluded_prefixes
            .iter()
            .any(|prefix| rel.starts_with(prefix) || rel_with_slash.starts_with(prefix))
            && !NON_AUTHORITATIVE_REPORT_PREFIXES
                .iter()
                .any(|prefix| rel.starts_with(prefix) || rel_with_slash.starts_with(prefix))
            && !NON_AUTHORITATIVE_REGISTRY_PREFIXES
                .iter()
                .any(|prefix| rel.starts_with(prefix) || rel_with_slash.starts_with(prefix))
            && !rel.contains("/lambda_gororoba_backups/")
    });
    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(repo_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let suffix = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        let suffix = format!(".{suffix}");
        if !suffixes.contains(&suffix.as_str()) {
            continue;
        }
        if rel != "refs.bib"
            && !allowed_prefixes
                .iter()
                .any(|prefix| rel.starts_with(prefix))
        {
            continue;
        }
        if excluded_exact.contains(&rel.as_str())
            || NON_AUTHORITATIVE_REGISTRY_EXACT.contains(&rel.as_str())
            || rel.contains("/files/open-pdf/")
            || rel.ends_with("_link_search.md")
            || rel.ends_with(".proxy.txt")
        {
            continue;
        }
        if text_suffixes.contains(&suffix.as_str()) {
            let lowered = rel.to_ascii_lowercase();
            if !text_keywords.iter().any(|token| lowered.contains(token)) {
                continue;
            }
        }
        paths.insert(path.to_path_buf());
    }
    paths.into_iter().collect()
}

fn candidates_from_bibliography(repo_root: &Path) -> Result<Vec<CandidateRecord>> {
    let mut candidates = Vec::new();
    let bibliography_path = repo_root.join("registry/bibliography.toml");
    if !bibliography_path.exists() {
        return Ok(candidates);
    }
    let value = load_toml_value(&bibliography_path)?;
    let Some(entries) = value.get("entry").and_then(Value::as_array) else {
        return Ok(candidates);
    };
    for entry in entries {
        let Some(table) = entry.as_table() else {
            continue;
        };
        let entry_id = table.get("id").and_then(Value::as_str).unwrap_or("").trim();
        let citation = table
            .get("citation_markdown")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let title = citation.clone();
        let mut links = Vec::new();
        if let Some(urls) = table.get("urls").and_then(Value::as_array) {
            for value in urls.iter().filter_map(Value::as_str) {
                let normalized = normalize_url(value);
                if !normalized.is_empty() {
                    links.push(normalized);
                }
            }
        }
        let mut dois = Vec::new();
        if let Some(values) = table.get("dois").and_then(Value::as_array) {
            for value in values.iter().filter_map(Value::as_str) {
                let normalized = normalize_doi(value);
                if !normalized.is_empty() {
                    dois.push(normalized);
                }
            }
        }
        links.extend(dois.iter().map(|doi| doi_to_url(doi)));
        let notes = table
            .get("notes")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|note| note.trim().to_string())
                    .filter(|note| !note.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        candidates.push(CandidateRecord {
            source_kind: "bibliography_entry".to_string(),
            source_ref: if entry_id.is_empty() {
                "BIB-UNKNOWN".to_string()
            } else {
                entry_id.to_string()
            },
            identity_override: None,
            title: title.clone(),
            citation,
            dois: dedupe(dois),
            links: dedupe(links),
            local_paths: Vec::new(),
            notes,
        });
    }
    Ok(candidates)
}

fn candidates_from_external_sources(repo_root: &Path) -> Result<Vec<CandidateRecord>> {
    let mut candidates = Vec::new();
    let external_sources_path = repo_root.join("registry/external_sources.toml");
    if !external_sources_path.exists() {
        return Ok(candidates);
    }
    let value = load_toml_value(&external_sources_path)?;
    let Some(documents) = value.get("document").and_then(Value::as_array) else {
        return Ok(candidates);
    };
    for document in documents {
        let Some(table) = document.as_table() else {
            continue;
        };
        let doc_id = table.get("id").and_then(Value::as_str).unwrap_or("").trim();
        let title = table
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let mut links = Vec::new();
        if let Some(url_refs) = table.get("url_refs").and_then(Value::as_array) {
            for value in url_refs.iter().filter_map(Value::as_str) {
                let normalized = normalize_url(value);
                if !normalized.is_empty() {
                    links.push(normalized);
                }
            }
        }
        let mut existing_paths = Vec::new();
        if let Some(path_refs) = table.get("path_refs").and_then(Value::as_array) {
            for path_ref in path_refs.iter().filter_map(Value::as_str) {
                let trimmed = path_ref.trim();
                if !trimmed.is_empty()
                    && is_artifact_local_path(trimmed)
                    && repo_root.join(trimmed).exists()
                {
                    existing_paths.push(trimmed.to_string());
                }
            }
        }
        let notes = table
            .get("notes")
            .and_then(Value::as_str)
            .map(|note| note.trim().to_string())
            .filter(|note| !note.is_empty())
            .into_iter()
            .collect::<Vec<_>>();
        candidates.push(CandidateRecord {
            source_kind: "external_source_document".to_string(),
            source_ref: if doc_id.is_empty() {
                "XS-UNKNOWN".to_string()
            } else {
                doc_id.to_string()
            },
            identity_override: None,
            title: title.clone(),
            citation: title,
            dois: Vec::new(),
            links: dedupe(links),
            local_paths: dedupe(existing_paths),
            notes,
        });
    }
    Ok(candidates)
}

fn candidates_from_cdcs(repo_root: &Path) -> Result<Vec<CandidateRecord>> {
    let mut candidates = Vec::new();
    let cdcs_path = repo_root.join("registry/cayley_dickson_canonical_sources.toml");
    if !cdcs_path.exists() {
        return Ok(candidates);
    }
    let value = load_toml_value(&cdcs_path)?;
    let Some(papers) = value.get("paper").and_then(Value::as_array) else {
        return Ok(candidates);
    };
    for paper in papers {
        let Some(table) = paper.as_table() else {
            continue;
        };
        let key = table
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or("CDCS-UNKNOWN")
            .trim();
        let title = table
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let doi = normalize_doi(table.get("doi").and_then(Value::as_str).unwrap_or(""));
        let mut links = Vec::new();
        for field in [
            "working_mirrors",
            "working_pdf_mirrors",
            "nonworking_mirrors",
            "manual_intervention_urls",
        ] {
            if let Some(values) = table.get(field).and_then(Value::as_array) {
                for value in values.iter().filter_map(Value::as_str) {
                    let normalized = normalize_url(value);
                    if !normalized.is_empty() {
                        links.push(normalized);
                    }
                }
            }
        }
        let canonical_url = normalize_url(
            table
                .get("canonical_functional_url")
                .and_then(Value::as_str)
                .unwrap_or(""),
        );
        if !canonical_url.is_empty() {
            links.push(canonical_url);
        }
        if !doi.is_empty() {
            links.push(doi_to_url(&doi));
        }
        let mut local_paths = Vec::new();
        let canonical_pdf_path = table
            .get("canonical_pdf_path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if !canonical_pdf_path.is_empty()
            && is_artifact_local_path(canonical_pdf_path)
            && repo_root.join(canonical_pdf_path).exists()
        {
            local_paths.push(canonical_pdf_path.to_string());
        }
        let mut notes = Vec::new();
        let status = table
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if !status.is_empty() {
            notes.push(format!("status={status}"));
        }
        let reason = table
            .get("manual_intervention_reason")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if !reason.is_empty() {
            notes.push(reason.to_string());
        }
        candidates.push(CandidateRecord {
            source_kind: "canonical_cayley_dickson".to_string(),
            source_ref: key.to_string(),
            identity_override: None,
            title: title.clone(),
            citation: title,
            dois: if doi.is_empty() {
                Vec::new()
            } else {
                vec![doi]
            },
            links: dedupe(links),
            local_paths: dedupe(local_paths),
            notes: dedupe(notes),
        });
    }
    Ok(candidates)
}

fn build_candidates(repo_root: &Path) -> Result<(Vec<CandidateRecord>, Vec<String>)> {
    let mut candidates = Vec::new();
    candidates.extend(candidates_from_bibliography(repo_root)?);
    candidates.extend(candidates_from_external_sources(repo_root)?);
    candidates.extend(candidates_from_cdcs(repo_root)?);

    let discovered_files = discover_candidate_source_files(repo_root);
    for file in &discovered_files {
        candidates.extend(extract_candidates_from_source_file(repo_root, file)?);
    }
    for candidate in &mut candidates {
        let link_dois = extract_dois_from_urls(&candidate.links);
        if !link_dois.is_empty() {
            let mut all_dois = candidate.dois.clone();
            all_dois.extend(link_dois.iter().cloned());
            candidate.dois = dedupe(all_dois);
        }
        let mut all_links = candidate.links.clone();
        all_links.extend(link_dois.iter().map(|doi| doi_to_url(doi)));
        let alias_expansions = all_links
            .iter()
            .flat_map(|url| expand_reference_aliases(url))
            .collect::<Vec<_>>();
        all_links.extend(alias_expansions);
        candidate.links = dedupe(all_links);
    }
    let source_files = discovered_files
        .iter()
        .map(|path| {
            path.strip_prefix(repo_root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    Ok((candidates, source_files))
}

fn identity_key(candidate: &CandidateRecord) -> String {
    if let Some(identity_override) = &candidate.identity_override {
        return identity_override.clone();
    }
    if let Some(doi) = candidate.dois.first() {
        return format!("doi:{}", doi.to_ascii_lowercase());
    }
    if let Some(url) = canonical_identity_url(&candidate.links) {
        if url.starts_with("cambridge:") {
            return url;
        }
        return format!("url:{}", url.to_ascii_lowercase());
    }
    if !candidate.title.is_empty() {
        return format!("title:{}", slug(&candidate.title));
    }
    format!(
        "source:{}:{}",
        candidate.source_kind,
        slug(&candidate.source_ref)
    )
}

fn unify_candidates(candidates: Vec<CandidateRecord>) -> Vec<UnifiedArtifact> {
    let mut merged: BTreeMap<String, UnifiedArtifact> = BTreeMap::new();
    for candidate in candidates {
        let key = identity_key(&candidate);
        let entry = merged
            .entry(key.clone())
            .or_insert_with(|| UnifiedArtifact {
                key,
                ..UnifiedArtifact::default()
            });
        if entry.title.is_empty() && !candidate.title.is_empty() {
            entry.title = candidate.title.clone();
        }
        if entry.citation.is_empty() && !candidate.citation.is_empty() {
            entry.citation = candidate.citation.clone();
        }
        entry.source_kinds.push(candidate.source_kind);
        entry.source_refs.push(candidate.source_ref);
        entry.doi_list.extend(candidate.dois);
        entry.links.extend(candidate.links);
        entry.local_paths.extend(candidate.local_paths);
        entry.notes.extend(candidate.notes);
    }
    let mut out = merged.into_values().collect::<Vec<_>>();
    for artifact in &mut out {
        artifact.source_kinds = dedupe(std::mem::take(&mut artifact.source_kinds));
        artifact.source_refs = dedupe(std::mem::take(&mut artifact.source_refs));
        artifact.doi_list = dedupe(std::mem::take(&mut artifact.doi_list));
        artifact.links = dedupe(std::mem::take(&mut artifact.links));
        artifact.local_paths = dedupe(std::mem::take(&mut artifact.local_paths));
        artifact.notes = dedupe(std::mem::take(&mut artifact.notes));
    }
    out
}

fn classify_artifacts(
    artifacts: &mut [UnifiedArtifact],
    observations: &HashMap<String, Vec<LinkObservation>>,
    download_map: &HashMap<String, Vec<String>>,
    repo_root: &Path,
    retention: &RetentionSet,
    carry_forward: &HashMap<String, DurableFacts>,
) {
    for artifact in artifacts {
        let mut working = Vec::new();
        let mut working_pdf = Vec::new();
        let mut nonworking = Vec::new();
        let mut unverified = Vec::new();
        let mut downloaded = artifact.local_paths.clone();

        for url in &artifact.links {
            let obs_list = observations.get(url).cloned().unwrap_or_default();
            let statuses = obs_list
                .iter()
                .map(|obs| obs.status.as_str())
                .collect::<Vec<_>>();
            let has_pdf_ok = obs_list.iter().any(|obs| obs.status == "pdf_ok");
            let has_ok = obs_list.iter().any(|obs| obs.status == "ok_nonpdf");
            let has_nonworking = statuses.iter().any(|status| {
                (status.starts_with("http_")
                    && !matches!(
                        *status,
                        "http_200" | "http_201" | "http_202" | "http_203" | "http_204"
                    ))
                    || *status == "failed"
            });
            if has_pdf_ok {
                working.push(url.clone());
                working_pdf.push(url.clone());
            } else if has_ok {
                working.push(url.clone());
            } else if has_nonworking {
                nonworking.push(url.clone());
            } else {
                unverified.push(url.clone());
            }
            if let Some(paths) = download_map.get(url) {
                downloaded.extend(paths.clone());
            }
        }

        artifact.working_mirrors = dedupe(working);
        artifact.working_pdf_mirrors = dedupe(working_pdf);
        artifact.nonworking_mirrors = dedupe(nonworking);
        artifact.unverified_mirrors = dedupe(unverified);

        // Split the observed paths by the retention predicate. A path git
        // tracks is repository truth and keeps the `downloaded` status; a path
        // that exists only in this checkout is host state, so it leaves the
        // registry row and moves to the materialization manifest.
        let observed_paths = dedupe(downloaded);
        let (retained, host_only): (Vec<String>, Vec<String>) = observed_paths
            .into_iter()
            .partition(|path| retention.contains(path));
        artifact.downloaded_paths = retained;
        artifact.host_only_paths = host_only;

        let citation_locator_identity = key_is_citation_locator(&artifact.key);
        let citation_locator_only_links = !artifact.links.is_empty()
            && artifact
                .links
                .iter()
                .all(|url| is_citation_locator_url(url));

        artifact.canonical_functional_url = if let Some(url) = artifact.working_pdf_mirrors.first()
        {
            url.clone()
        } else if let Some(url) = artifact.working_mirrors.first() {
            url.clone()
        } else if let Some(url) = artifact.links.first() {
            url.clone()
        } else {
            String::new()
        };
        artifact.canonical_download_path = artifact
            .downloaded_paths
            .first()
            .cloned()
            .unwrap_or_default();

        artifact.status = if !artifact.downloaded_paths.is_empty() {
            "downloaded".to_string()
        } else if !artifact.host_only_paths.is_empty() {
            "remotely_materializable".to_string()
        } else if !artifact.working_mirrors.is_empty() {
            "downloadable".to_string()
        } else if citation_locator_identity
            || citation_locator_only_links
            || artifact.links.is_empty()
        {
            "citation_only_no_link".to_string()
        } else if !artifact.nonworking_mirrors.is_empty() && artifact.working_mirrors.is_empty() {
            "blocked".to_string()
        } else {
            "unverified".to_string()
        };

        // The lane is a property of the artifact, so it is decided while the
        // host-only paths are still in memory and then written into the row.
        // Deciding it later from the exported row would move every artifact
        // known only by a local .pdf into web_references.
        // A lane the catalog already recorded is repository truth; the scan
        // replaces it only when this host's evidence names a more specific
        // medium, so a missing local copy never moves a dataset or a paper
        // into web_references.
        let scanned_lane = classify_media_lane(
            artifact
                .links
                .iter()
                .chain(artifact.downloaded_paths.iter())
                .chain(artifact.host_only_paths.iter())
                .chain(std::iter::once(&artifact.canonical_functional_url)),
        );
        if artifact.lane.is_empty() || scanned_lane != "web_references" {
            artifact.lane = scanned_lane;
        }

        // Content identity comes from whichever copy this host holds. When no
        // copy is present the previously exported values carry forward, so a
        // checkout missing the PDFs never erases a hash another host measured.
        let carried = carry_forward.get(&artifact.key);
        let measured = artifact
            .downloaded_paths
            .iter()
            .chain(artifact.host_only_paths.iter())
            .find_map(|path| artifact_retention::file_identity(&repo_root.join(path)));
        match measured {
            Some(identity) => {
                artifact.sha256 = identity.sha256;
                artifact.byte_length = identity.byte_length;
            }
            None => {
                artifact.sha256 = carried.map(|facts| facts.sha256.clone()).unwrap_or_default();
                artifact.byte_length = carried.map(|facts| facts.byte_length).unwrap_or_default();
            }
        }
        // A carried URL is adopted only when it is still one of the artifact's
        // links, because the verifier requires canonical_functional_url to
        // appear in all_links.
        if artifact.canonical_functional_url.is_empty()
            && let Some(url) = carried.map(|facts| facts.canonical_url.clone())
            && artifact.links.contains(&url)
        {
            artifact.canonical_functional_url = url;
        }
        artifact.license_disposition = carried
            .map(|facts| facts.license_disposition.clone())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "unreviewed".to_string());
        artifact.retrieval_command = if artifact.status == "remotely_materializable" {
            // The retrieval target is derived from the artifact key, never from
            // the session-dated intake path this host happens to hold, so the
            // committed command reads the same on every checkout.
            let target = materialization_target(&artifact.key, &artifact.lane);
            let command = retrieval_command(&artifact.canonical_functional_url, &target);
            if command.is_empty() {
                carried
                    .map(|facts| facts.retrieval_command.clone())
                    .unwrap_or_default()
            } else {
                command
            }
        } else {
            String::new()
        };

        // A row with a URL and a hash is satisfiable from any host, so it meets
        // the minimum even though no local copy backs it.
        artifact.minimum_requirement_met = !artifact.working_mirrors.is_empty()
            || !artifact.downloaded_paths.is_empty()
            || (artifact.status == "remotely_materializable"
                && !artifact.canonical_functional_url.is_empty()
                && !artifact.sha256.is_empty());
        artifact.manual_intervention_required = !artifact.links.is_empty()
            && !artifact.minimum_requirement_met
            && !citation_locator_identity
            && !citation_locator_only_links;
        artifact.manual_intervention_reason = if artifact.manual_intervention_required {
            "No working mirror observed from current fetch/retry ledgers; manual link intervention required.".to_string()
        } else {
            String::new()
        };
    }
}

/// Assigns the lane from every extension the artifact is known by.
fn classify_media_lane<'a, I: Iterator<Item = &'a String>>(values: I) -> String {
    let values = values
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if values
        .iter()
        .any(|value| value_endswith_any(value, DATASET_EXTENSIONS))
    {
        "datasets".to_string()
    } else if values
        .iter()
        .any(|value| value_endswith_any(value, SLIDE_ARTIFACT_EXTENSIONS))
    {
        "slides_artifacts".to_string()
    } else if values
        .iter()
        .any(|value| value_endswith_any(value, PDF_EXTENSIONS))
    {
        "papers_pdf".to_string()
    } else {
        "web_references".to_string()
    }
}

/// Host-invariant destination for a re-fetched artifact.
fn materialization_target(key: &str, lane: &str) -> String {
    let stem = slug(key);
    match lane {
        "papers_pdf" => format!("papers/pdf/{stem}.pdf"),
        "datasets" => format!("data/external/datasets/{stem}"),
        "slides_artifacts" => format!("data/external/artifacts/{stem}"),
        _ => String::new(),
    }
}

/// Facts a prior export measured that a host without the file cannot recompute.
#[derive(Clone, Debug, Default)]
struct DurableFacts {
    sha256: String,
    byte_length: u64,
    retrieval_command: String,
    license_disposition: String,
    canonical_url: String,
}

fn load_durable_facts(path: &Path) -> HashMap<String, DurableFacts> {
    let mut facts = HashMap::new();
    let Ok(value) = load_toml_value(path) else {
        return facts;
    };
    let Some(rows) = value.get("artifact").and_then(Value::as_array) else {
        return facts;
    };
    for row in rows {
        let Some(table) = row.as_table() else { continue };
        let key = table
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if key.is_empty() {
            continue;
        }
        let text = |field: &str| {
            table
                .get(field)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        facts.insert(
            key,
            DurableFacts {
                sha256: text("sha256"),
                byte_length: table
                    .get("byte_length")
                    .and_then(Value::as_integer)
                    .unwrap_or_default()
                    .max(0) as u64,
                retrieval_command: text("retrieval_command"),
                license_disposition: text("license_disposition"),
                canonical_url: text("canonical_functional_url"),
            },
        );
    }
    facts
}

/// The durable identity of a row in the checked-in catalog: its id and the
/// fields that describe the artifact independently of any host. A row whose
/// key the current scan does not observe is re-seeded from these, so the
/// catalog never shrinks because a checkout lacks an untracked directory.
#[derive(Clone, Debug, Default)]
struct PriorRow {
    id: String,
    title: String,
    citation: String,
    source_kinds: Vec<String>,
    source_refs: Vec<String>,
    doi_list: Vec<String>,
    links: Vec<String>,
    lane: String,
}

fn load_prior_rows(path: &Path) -> BTreeMap<String, PriorRow> {
    let mut rows_by_key = BTreeMap::new();
    let Ok(value) = load_toml_value(path) else {
        return rows_by_key;
    };
    let Some(rows) = value.get("artifact").and_then(Value::as_array) else {
        return rows_by_key;
    };
    for row in rows {
        let Some(table) = row.as_table() else { continue };
        let text = |field: &str| {
            table
                .get(field)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let list = |field: &str| {
            table
                .get(field)
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let key = text("key");
        if key.is_empty() || text("id").is_empty() {
            continue;
        }
        rows_by_key.insert(
            key,
            PriorRow {
                id: text("id"),
                title: text("title"),
                citation: text("citation"),
                source_kinds: list("source_kinds"),
                source_refs: list("source_refs"),
                doi_list: list("doi_list"),
                links: list("all_links"),
                // A row exported before the lane field existed still names
                // its medium through the paths and links it recorded, which
                // is what classify_lane reads.
                lane: classify_lane(table).0,
            },
        );
    }
    rows_by_key
}

/// Carry the catalog's lane onto every observed row, re-seed every prior row
/// whose key the scan did not observe, then restore key order. Returns the
/// number of rows re-seeded.
fn seed_missing_prior_rows(
    artifacts: &mut Vec<UnifiedArtifact>,
    prior: &BTreeMap<String, PriorRow>,
) -> usize {
    let observed = artifacts
        .iter()
        .map(|artifact| artifact.key.clone())
        .collect::<HashSet<_>>();
    for artifact in artifacts.iter_mut() {
        if artifact.lane.is_empty()
            && let Some(row) = prior.get(&artifact.key)
        {
            artifact.lane = row.lane.clone();
        }
    }
    let mut seeded = 0;
    for (key, row) in prior {
        if observed.contains(key) {
            continue;
        }
        artifacts.push(UnifiedArtifact {
            key: key.clone(),
            title: row.title.clone(),
            citation: row.citation.clone(),
            source_kinds: row.source_kinds.clone(),
            source_refs: row.source_refs.clone(),
            doi_list: row.doi_list.clone(),
            links: row.links.clone(),
            lane: row.lane.clone(),
            ..UnifiedArtifact::default()
        });
        seeded += 1;
    }
    artifacts.sort_by(|a, b| a.key.cmp(&b.key));
    seeded
}

/// An artifact keeps the id the catalog already gave its key; a new key takes
/// the next number after the largest prior id, in key order. Ids therefore
/// never move when a row is added or when a scan observes fewer candidates.
fn assign_stable_ids(
    artifacts: &[UnifiedArtifact],
    prior: &BTreeMap<String, PriorRow>,
) -> Vec<String> {
    let mut next = prior
        .values()
        .filter_map(|row| row.id.strip_prefix("ASOT-")?.parse::<u64>().ok())
        .max()
        .unwrap_or(0);
    artifacts
        .iter()
        .map(|artifact| {
            if let Some(row) = prior.get(&artifact.key) {
                row.id.clone()
            } else {
                next += 1;
                format!("ASOT-{next:04}")
            }
        })
        .collect()
}

fn render_artifact_registry(
    artifacts: &[UnifiedArtifact],
    ids: &[String],
    source_tables: &[String],
    source_files: &[String],
    now: &str,
) -> String {
    let total = artifacts.len();
    let remotely_materializable = artifacts
        .iter()
        .filter(|artifact| artifact.status == "remotely_materializable")
        .count();
    let downloaded = artifacts
        .iter()
        .filter(|artifact| artifact.status == "downloaded")
        .count();
    let downloadable = artifacts
        .iter()
        .filter(|artifact| artifact.status == "downloadable")
        .count();
    let blocked = artifacts
        .iter()
        .filter(|artifact| artifact.status == "blocked")
        .count();
    let citation_only = artifacts
        .iter()
        .filter(|artifact| artifact.status == "citation_only_no_link")
        .count();
    let unverified = artifacts
        .iter()
        .filter(|artifact| artifact.status == "unverified")
        .count();
    let missing_minimum = artifacts
        .iter()
        .filter(|artifact| !artifact.minimum_requirement_met)
        .count();
    let manual = artifacts
        .iter()
        .filter(|artifact| artifact.manual_intervention_required)
        .count();

    let mut lines = vec![
        "# Single source-of-truth registry for cited artifacts and mirror status.".to_string(),
        String::new(),
        "[artifact_source_of_truth]".to_string(),
        "id = \"ASOT-2026-02-15\"".to_string(),
        format!("updated = {}", escape_toml(now)),
        "authoritative = true".to_string(),
        "policy = \"Keep one working mirror minimum per artifact; retain working mirrors and non-working mirrors for manual intervention.\"".to_string(),
        "minimum_requirement = \"1 paper/document/artifact => >= 1 working mirror or downloaded local artifact.\"".to_string(),
        format!("source_table_count = {}", source_tables.len()),
        format!("source_tables = {}", render_list(source_tables)),
        format!("source_file_count = {}", source_files.len()),
        format!("source_files = {}", render_list(source_files)),
        format!("artifact_count = {total}"),
        format!("downloaded_count = {downloaded}"),
        format!("remotely_materializable_count = {remotely_materializable}"),
        format!("downloadable_count = {downloadable}"),
        format!("blocked_count = {blocked}"),
        format!("citation_only_no_link_count = {citation_only}"),
        format!("unverified_count = {unverified}"),
        format!("missing_minimum_requirement_count = {missing_minimum}"),
        format!("manual_intervention_required_count = {manual}"),
        String::new(),
    ];

    let missing_keys = artifacts
        .iter()
        .filter(|artifact| !artifact.minimum_requirement_met)
        .map(|artifact| artifact.key.clone())
        .collect::<Vec<_>>();
    lines.push("[coverage]".to_string());
    lines.push(format!(
        "artifacts_without_working_mirror = {}",
        render_list(&missing_keys)
    ));
    lines.push(format!(
        "artifacts_without_working_mirror_count = {}",
        missing_keys.len()
    ));
    lines.push(String::new());

    for (index, artifact) in artifacts.iter().enumerate() {
        lines.push("[[artifact]]".to_string());
        lines.push(format!("id = {}", escape_toml(&ids[index])));
        lines.push(format!("key = {}", escape_toml(&artifact.key)));
        lines.push(format!("title = {}", escape_toml(&artifact.title)));
        lines.push(format!("citation = {}", escape_toml(&artifact.citation)));
        lines.push(format!(
            "source_kinds = {}",
            render_list(&artifact.source_kinds)
        ));
        lines.push(format!(
            "source_refs = {}",
            render_list(&artifact.source_refs)
        ));
        lines.push(format!("doi_list = {}", render_list(&artifact.doi_list)));
        lines.push(format!(
            "canonical_functional_url = {}",
            escape_toml(&artifact.canonical_functional_url)
        ));
        lines.push(format!(
            "canonical_download_path = {}",
            escape_toml(&artifact.canonical_download_path)
        ));
        lines.push(format!("status = {}", escape_toml(&artifact.status)));
        lines.push(format!("lane = {}", escape_toml(&artifact.lane)));
        lines.push(format!("sha256 = {}", escape_toml(&artifact.sha256)));
        lines.push(format!("byte_length = {}", artifact.byte_length));
        lines.push(format!(
            "retrieval_command = {}",
            escape_toml(&artifact.retrieval_command)
        ));
        lines.push(format!(
            "license_disposition = {}",
            escape_toml(&artifact.license_disposition)
        ));
        lines.push(format!(
            "host_only_path_count = {}",
            artifact.host_only_paths.len()
        ));
        lines.push(format!(
            "minimum_requirement_met = {}",
            artifact.minimum_requirement_met
        ));
        lines.push(format!(
            "manual_intervention_required = {}",
            artifact.manual_intervention_required
        ));
        lines.push(format!(
            "manual_intervention_reason = {}",
            escape_toml(&artifact.manual_intervention_reason)
        ));
        lines.push(format!(
            "working_mirror_count = {}",
            artifact.working_mirrors.len()
        ));
        lines.push(format!(
            "working_pdf_mirror_count = {}",
            artifact.working_pdf_mirrors.len()
        ));
        lines.push(format!(
            "nonworking_mirror_count = {}",
            artifact.nonworking_mirrors.len()
        ));
        lines.push(format!(
            "unverified_mirror_count = {}",
            artifact.unverified_mirrors.len()
        ));
        lines.push(format!(
            "downloaded_path_count = {}",
            artifact.downloaded_paths.len()
        ));
        lines.push(format!("all_links = {}", render_list(&artifact.links)));
        lines.push(format!(
            "working_mirrors = {}",
            render_list(&artifact.working_mirrors)
        ));
        lines.push(format!(
            "working_pdf_mirrors = {}",
            render_list(&artifact.working_pdf_mirrors)
        ));
        lines.push(format!(
            "nonworking_mirrors = {}",
            render_list(&artifact.nonworking_mirrors)
        ));
        lines.push(format!(
            "unverified_mirrors = {}",
            render_list(&artifact.unverified_mirrors)
        ));
        lines.push(format!(
            "downloaded_paths = {}",
            render_list(&artifact.downloaded_paths)
        ));
        lines.push(format!("notes = {}", render_list(&artifact.notes)));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_reconciliation_report(artifacts: &[UnifiedArtifact], now: &str) -> String {
    let total = artifacts.len();
    let remotely_materializable = artifacts
        .iter()
        .filter(|artifact| artifact.status == "remotely_materializable")
        .count();
    let downloaded = artifacts
        .iter()
        .filter(|artifact| artifact.status == "downloaded")
        .count();
    let downloadable = artifacts
        .iter()
        .filter(|artifact| artifact.status == "downloadable")
        .count();
    let blocked = artifacts
        .iter()
        .filter(|artifact| artifact.status == "blocked")
        .count();
    let citation_only = artifacts
        .iter()
        .filter(|artifact| artifact.status == "citation_only_no_link")
        .count();
    let unverified = artifacts
        .iter()
        .filter(|artifact| artifact.status == "unverified")
        .count();
    let missing = artifacts
        .iter()
        .filter(|artifact| !artifact.minimum_requirement_met)
        .collect::<Vec<_>>();

    let mut lines = vec![
        "# Reconciliation summary for artifact_source_of_truth.toml".to_string(),
        String::new(),
        "[report]".to_string(),
        "id = \"ASOT-RECON-2026-02-15\"".to_string(),
        format!("updated = {}", escape_toml(now)),
        "authoritative = true".to_string(),
        format!("artifact_count = {total}"),
        format!("downloaded_count = {downloaded}"),
        format!("remotely_materializable_count = {remotely_materializable}"),
        format!("downloadable_count = {downloadable}"),
        format!("blocked_count = {blocked}"),
        format!("citation_only_no_link_count = {citation_only}"),
        format!("unverified_count = {unverified}"),
        format!("missing_minimum_requirement_count = {}", missing.len()),
        String::new(),
    ];
    for artifact in missing {
        lines.push("[[missing_minimum_requirement]]".to_string());
        lines.push(format!("key = {}", escape_toml(&artifact.key)));
        lines.push(format!("title = {}", escape_toml(&artifact.title)));
        lines.push(format!("status = {}", escape_toml(&artifact.status)));
        lines.push(format!(
            "source_refs = {}",
            render_list(&artifact.source_refs)
        ));
        lines.push(format!("all_links = {}", render_list(&artifact.links)));
        lines.push(format!(
            "nonworking_mirrors = {}",
            render_list(&artifact.nonworking_mirrors)
        ));
        lines.push(format!(
            "unverified_mirrors = {}",
            render_list(&artifact.unverified_mirrors)
        ));
        lines.push(String::new());
    }
    lines.join("\n")
}

/// Renders the master registry and its reconciliation report into `set`
/// without touching either file. The caller commits the whole export in one
/// rename pass, so a refused shrink leaves every previous output intact.
pub fn stage_artifact_source_of_truth(
    repo_root: &Path,
    out_registry: &Path,
    out_report: &Path,
    retention: &RetentionSet,
    set: &mut StagedWriteSet,
) -> Result<(BuildSummary, String)> {
    let now = Utc::now().format("%Y-%m-%d").to_string();
    let (observations, source_tables) = collect_link_observations(repo_root)?;
    let mut download_map = collect_download_map(repo_root)?;
    let (candidates, source_files) = build_candidates(repo_root)?;
    let mut artifacts = unify_candidates(candidates);
    let prior_rows = load_prior_rows(out_registry);
    let retained_prior_row_count = seed_missing_prior_rows(&mut artifacts, &prior_rows);
    let ids = assign_stable_ids(&artifacts, &prior_rows);
    extend_download_map_from_local_artifacts(&mut download_map, &artifacts);
    let carry_forward = load_durable_facts(out_registry);
    classify_artifacts(
        &mut artifacts,
        &observations,
        &download_map,
        repo_root,
        retention,
        &carry_forward,
    );
    let registry_text =
        render_artifact_registry(&artifacts, &ids, &source_tables, &source_files, &now);
    let report_text = render_reconciliation_report(&artifacts, &now);
    assert_ascii(&registry_text, &out_registry.display().to_string())?;
    assert_ascii(&report_text, &out_report.display().to_string())?;
    let downloaded_count = artifacts
        .iter()
        .filter(|artifact| artifact.status == "downloaded")
        .count();
    let materializable = artifacts
        .iter()
        .filter(|artifact| artifact.status == "remotely_materializable")
        .collect::<Vec<_>>();
    let summary = BuildSummary {
        artifact_count: artifacts.len(),
        downloaded_count,
        remotely_materializable_count: materializable.len(),
        // A materializable row with no URL names bytes no other host can
        // obtain. It is reported, not dressed up with a retrieval command.
        materializable_without_url_count: materializable
            .iter()
            .filter(|artifact| artifact.canonical_functional_url.is_empty())
            .count(),
        row_counts: Vec::new(),
        retained_prior_row_count,
        host_materialization: artifacts
            .iter()
            .enumerate()
            .filter(|(_, artifact)| {
                artifact.status == "downloaded" || artifact.status == "remotely_materializable"
            })
            .flat_map(|(index, artifact)| {
                artifact
                    .downloaded_paths
                    .iter()
                    .chain(artifact.host_only_paths.iter())
                    .map(|path| {
                        observe_host_materialization(
                            repo_root,
                            retention,
                            &ids[index],
                            &artifact.key,
                            &artifact.status,
                            path,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    };
    set.stage(out_registry, registry_text.clone(), "[[artifact]]");
    set.stage(out_report, report_text, "[[artifact]]");
    Ok((summary, registry_text))
}

pub fn build_artifact_source_of_truth(
    repo_root: &Path,
    out_registry: &Path,
    out_report: &Path,
) -> Result<BuildSummary> {
    let retention = RetentionSet::from_git_index(repo_root);
    let mut set = StagedWriteSet::new();
    let (mut summary, _) =
        stage_artifact_source_of_truth(repo_root, out_registry, out_report, &retention, &mut set)?;
    summary.row_counts = set.commit(&ShrinkPolicy::permissive())?;
    Ok(summary)
}

fn lane_description(name: &str) -> &'static str {
    match name {
        "datasets" => "Numerical or tabular research datasets and machine-readable data artifacts.",
        "slides_artifacts" => {
            "Slides, decks, archives, notebooks, and non-dataset non-paper binary artifacts."
        }
        "papers_pdf" => "Paper-oriented references with PDF documents or PDF mirrors.",
        _ => "Reference URLs without locally identified PDF/data/artifact files.",
    }
}

fn value_endswith_any(value: &str, extensions: &[&str]) -> bool {
    let lowered = value.to_ascii_lowercase();
    extensions
        .iter()
        .any(|ext| lowered.ends_with(ext) || lowered.contains(&format!("{ext}?")))
}

fn classify_lane(artifact: &toml::map::Map<String, Value>) -> (String, Vec<String>) {
    // A row carrying the durable lane field decides its own lane; the
    // extension scan below stays for rows exported before the field existed.
    if let Some(lane) = artifact
        .get("lane")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|lane| LANE_ORDER.contains(lane))
    {
        return (lane.to_string(), vec![lane.to_string()]);
    }
    let mut values = Vec::new();
    if let Some(items) = artifact.get("all_links").and_then(Value::as_array) {
        for value in items.iter().filter_map(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                values.push(trimmed.to_string());
            }
        }
    }
    if let Some(items) = artifact.get("downloaded_paths").and_then(Value::as_array) {
        for value in items.iter().filter_map(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                values.push(trimmed.to_string());
            }
        }
    }
    let canonical_url = artifact
        .get("canonical_functional_url")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !canonical_url.is_empty() {
        values.push(canonical_url.to_string());
    }
    let canonical_path = artifact
        .get("canonical_download_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !canonical_path.is_empty() {
        values.push(canonical_path.to_string());
    }

    let has_dataset = values
        .iter()
        .any(|value| value_endswith_any(value, DATASET_EXTENSIONS));
    let has_slide_artifact = values
        .iter()
        .any(|value| value_endswith_any(value, SLIDE_ARTIFACT_EXTENSIONS));
    let has_pdf = values
        .iter()
        .any(|value| value_endswith_any(value, PDF_EXTENSIONS));
    let mut tags = Vec::new();
    if has_dataset {
        tags.push("datasets".to_string());
    }
    if has_slide_artifact {
        tags.push("slides_artifacts".to_string());
    }
    if has_pdf {
        tags.push("papers_pdf".to_string());
    }
    if tags.is_empty() {
        tags.push("web_references".to_string());
    }
    let primary = if has_dataset {
        "datasets"
    } else if has_slide_artifact {
        "slides_artifacts"
    } else if has_pdf {
        "papers_pdf"
    } else {
        "web_references"
    };
    (primary.to_string(), tags)
}

fn render_lane(
    name: &str,
    artifacts: &[toml::map::Map<String, Value>],
    generated_at: &str,
) -> String {
    let counts = artifacts.iter().fold(HashMap::new(), |mut acc, artifact| {
        let status = artifact
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        *acc.entry(status).or_insert(0usize) += 1;
        acc
    });
    let missing_minimum = artifacts
        .iter()
        .filter(|artifact| {
            !artifact
                .get("minimum_requirement_met")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .count();
    let mut lines = vec![
        format!("# Lane: {name}"),
        String::new(),
        "[lane]".to_string(),
        format!(
            "id = {}",
            escape_toml(&format!("SLANE-{}-2026-02-15", name.to_ascii_uppercase()))
        ),
        format!("name = {}", escape_toml(name)),
        format!("description = {}", escape_toml(lane_description(name))),
        format!("generated_at = {}", escape_toml(generated_at)),
        format!("artifact_count = {}", artifacts.len()),
        format!(
            "downloaded_count = {}",
            counts.get("downloaded").copied().unwrap_or_default()
        ),
        format!(
            "remotely_materializable_count = {}",
            counts
                .get("remotely_materializable")
                .copied()
                .unwrap_or_default()
        ),
        format!(
            "downloadable_count = {}",
            counts.get("downloadable").copied().unwrap_or_default()
        ),
        format!(
            "blocked_count = {}",
            counts.get("blocked").copied().unwrap_or_default()
        ),
        format!(
            "citation_only_no_link_count = {}",
            counts
                .get("citation_only_no_link")
                .copied()
                .unwrap_or_default()
        ),
        format!(
            "unverified_count = {}",
            counts.get("unverified").copied().unwrap_or_default()
        ),
        format!("missing_minimum_requirement_count = {missing_minimum}"),
        String::new(),
    ];
    for artifact in artifacts {
        let source_refs = artifact
            .get("source_refs")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        lines.push("[[artifact_ref]]".to_string());
        lines.push(format!(
            "id = {}",
            escape_toml(
                artifact
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
            )
        ));
        lines.push(format!(
            "key = {}",
            escape_toml(
                artifact
                    .get("key")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
            )
        ));
        lines.push(format!(
            "title = {}",
            escape_toml(
                artifact
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
            )
        ));
        lines.push(format!(
            "status = {}",
            escape_toml(
                artifact
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
            )
        ));
        for field in ["sha256", "retrieval_command", "license_disposition"] {
            lines.push(format!(
                "{field} = {}",
                escape_toml(artifact.get(field).and_then(Value::as_str).unwrap_or("").trim())
            ));
        }
        lines.push(format!(
            "byte_length = {}",
            artifact
                .get("byte_length")
                .and_then(Value::as_integer)
                .unwrap_or_default()
        ));
        lines.push(format!(
            "minimum_requirement_met = {}",
            artifact
                .get("minimum_requirement_met")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        ));
        lines.push(format!(
            "canonical_functional_url = {}",
            escape_toml(
                artifact
                    .get("canonical_functional_url")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim(),
            )
        ));
        lines.push(format!(
            "canonical_download_path = {}",
            escape_toml(
                artifact
                    .get("canonical_download_path")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim(),
            )
        ));
        lines.push(format!("source_refs = {}", render_list(&source_refs)));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_infrastructure(
    source_path: &str,
    lane_files: &BTreeMap<String, String>,
    lane_counts: &BTreeMap<String, usize>,
    total_artifacts: usize,
    generated_at: &str,
) -> String {
    let mut lines = vec![
        "# Canonical source infrastructure manifest.".to_string(),
        String::new(),
        "[source_infrastructure]".to_string(),
        "id = \"SINFRA-2026-02-15\"".to_string(),
        format!("generated_at = {}", escape_toml(generated_at)),
        "authoritative = true".to_string(),
        "policy_version = 1".to_string(),
        "policy = \"artifact_source_of_truth.toml is authoritative; lane files are deterministic projections and must never diverge from master.\"".to_string(),
        "best_practice = \"single authoritative master, deterministic generated lanes, explicit blocked/manual intervention tracking, provenance-preserving mirrors, reproducible verification gates.\"".to_string(),
        format!(
            "best_practice_sources = {}",
            render_list(&BEST_PRACTICE_SOURCES.iter().map(|value| value.to_string()).collect::<Vec<_>>())
        ),
        format!("master_registry = {}", escape_toml(source_path)),
        format!("lane_count = {}", LANE_ORDER.len()),
        format!("total_artifact_count = {total_artifacts}"),
        String::new(),
    ];
    for lane in LANE_ORDER {
        lines.push("[[lane]]".to_string());
        lines.push(format!("name = {}", escape_toml(lane)));
        lines.push(format!(
            "description = {}",
            escape_toml(lane_description(lane))
        ));
        lines.push(format!(
            "path = {}",
            escape_toml(lane_files.get(*lane).map(String::as_str).unwrap_or(""))
        ));
        lines.push(format!(
            "artifact_count = {}",
            lane_counts.get(*lane).copied().unwrap_or_default()
        ));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_source_infrastructure_report(
    lane_counts: &BTreeMap<String, usize>,
    total_artifacts: usize,
    generated_at: &str,
) -> String {
    let mut lines = vec![
        "# Reconciliation report for source infrastructure lanes.".to_string(),
        String::new(),
        "[report]".to_string(),
        "id = \"SINFRA-RECON-2026-02-15\"".to_string(),
        format!("generated_at = {}", escape_toml(generated_at)),
        "authoritative = true".to_string(),
        format!("total_artifact_count = {total_artifacts}"),
        format!(
            "lane_total_count = {}",
            lane_counts.values().copied().sum::<usize>()
        ),
        String::new(),
    ];
    for lane in LANE_ORDER {
        lines.push("[[lane_summary]]".to_string());
        lines.push(format!("name = {}", escape_toml(lane)));
        lines.push(format!(
            "artifact_count = {}",
            lane_counts.get(*lane).copied().unwrap_or_default()
        ));
        lines.push(String::new());
    }
    lines.join("\n")
}

/// Projects the staged master text into the four lane files, the
/// infrastructure manifest and its report. Taking the master as text rather
/// than a path lets the whole export stage before anything is renamed.
pub fn stage_source_truth_infrastructure(
    repo_root: &Path,
    master_text: &str,
    master_path: &Path,
    out_infrastructure: &Path,
    lane_dir: &Path,
    out_report: &Path,
    set: &mut StagedWriteSet,
) -> Result<SourceInfrastructureSummary> {
    let value: Value =
        toml::from_str(master_text).context("parse staged artifact source of truth")?;
    let artifacts = value
        .get("artifact")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let generated_at = Utc::now().format("%Y-%m-%d").to_string();
    let mut lane_map: BTreeMap<String, Vec<toml::map::Map<String, Value>>> = LANE_ORDER
        .iter()
        .map(|lane| (lane.to_string(), Vec::new()))
        .collect();

    for artifact in artifacts {
        let Some(table) = artifact.as_table().cloned() else {
            continue;
        };
        let (primary, _) = classify_lane(&table);
        lane_map.entry(primary).or_default().push(table);
    }

    let mut lane_files = BTreeMap::new();
    let mut lane_counts = BTreeMap::new();
    for lane in LANE_ORDER {
        let mut lane_artifacts = lane_map.remove(*lane).unwrap_or_default();
        lane_artifacts.sort_by(|left, right| {
            left.get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .cmp(right.get("id").and_then(Value::as_str).unwrap_or(""))
        });
        let lane_text = render_lane(lane, &lane_artifacts, &generated_at);
        let lane_path = lane_dir.join(format!("{lane}.toml"));
        assert_ascii(&lane_text, &lane_path.display().to_string())?;
        set.stage(&lane_path, lane_text, "[[artifact_ref]]");
        let rel = lane_path
            .strip_prefix(repo_root)
            .unwrap_or(lane_path.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        lane_files.insert((*lane).to_string(), rel);
        lane_counts.insert((*lane).to_string(), lane_artifacts.len());
    }

    let infrastructure_text = render_infrastructure(
        &master_path
            .strip_prefix(repo_root)
            .unwrap_or(master_path)
            .to_string_lossy()
            .replace('\\', "/"),
        &lane_files,
        &lane_counts,
        lane_counts.values().copied().sum(),
        &generated_at,
    );
    let report_text = render_source_infrastructure_report(
        &lane_counts,
        lane_counts.values().copied().sum(),
        &generated_at,
    );
    assert_ascii(
        &infrastructure_text,
        &out_infrastructure.display().to_string(),
    )?;
    assert_ascii(&report_text, &out_report.display().to_string())?;
    set.stage(out_infrastructure, infrastructure_text, "[[lane]]");
    set.stage(out_report, report_text, "[[lane]]");
    Ok(SourceInfrastructureSummary {
        total_artifact_count: lane_counts.values().copied().sum(),
        lane_counts,
    })
}

pub fn build_source_truth_infrastructure(
    repo_root: &Path,
    source_path: &Path,
    out_infrastructure: &Path,
    lane_dir: &Path,
    out_report: &Path,
) -> Result<SourceInfrastructureSummary> {
    let master_text = fs::read_to_string(source_path)
        .with_context(|| format!("read {}", source_path.display()))?;
    let mut set = StagedWriteSet::new();
    let summary = stage_source_truth_infrastructure(
        repo_root,
        &master_text,
        source_path,
        out_infrastructure,
        lane_dir,
        out_report,
        &mut set,
    )?;
    set.commit(&ShrinkPolicy::permissive())?;
    Ok(summary)
}

#[derive(Default)]
struct ArtifactCounts {
    downloaded: usize,
    remotely_materializable: usize,
    downloadable: usize,
    blocked: usize,
    citation_only: usize,
    unverified: usize,
    missing_minimum: usize,
    manual: usize,
}

#[derive(Default)]
struct ValidationState {
    ids: HashSet<String>,
    keys: HashSet<String>,
    counts: ArtifactCounts,
    failures: Vec<String>,
}

fn validate_artifact_entry(
    index: usize,
    artifact: &Value,
    repo_root: &Path,
    coverage_missing_keys: &[String],
    state: &mut ValidationState,
) {
    let Some(table) = artifact.as_table() else {
        state
            .failures
            .push(format!("artifact[{index}] is not a table"));
        return;
    };
    let art_id = table
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let key = table
        .get("key")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let status = table
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let minimum_met = table
        .get("minimum_requirement_met")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let manual = table
        .get("manual_intervention_required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let all_links = table
        .get("all_links")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let working = table
        .get("working_mirrors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let working_pdf = table
        .get("working_pdf_mirrors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let nonworking = table
        .get("nonworking_mirrors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let unverified_mirrors = table
        .get("unverified_mirrors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let downloaded_paths = table
        .get("downloaded_paths")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let canonical_url = table
        .get("canonical_functional_url")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let canonical_path = table
        .get("canonical_download_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();

    if art_id.is_empty() {
        state.failures.push(format!("artifact[{index}] missing id"));
    } else if !state.ids.insert(art_id.clone()) {
        state
            .failures
            .push(format!("duplicate artifact id: {art_id}"));
    }
    if key.is_empty() {
        state.failures.push(format!(
            "{} missing key",
            if art_id.is_empty() {
                format!("index {index}")
            } else {
                art_id.clone()
            }
        ));
    } else if !state.keys.insert(key.clone()) {
        state
            .failures
            .push(format!("duplicate artifact key: {key}"));
    }
    if !VALID_STATUSES.contains(&status.as_str()) {
        state
            .failures
            .push(format!("{art_id}: invalid status {status:?}"));
    }
    if !canonical_url.is_empty() && !all_links.contains(&canonical_url) {
        state.failures.push(format!(
            "{art_id}: canonical_functional_url not in all_links"
        ));
    }
    match status.as_str() {
        "downloaded" => {
            state.counts.downloaded += 1;
            if downloaded_paths.is_empty() {
                state.failures.push(format!(
                    "{art_id}: downloaded status requires downloaded_paths"
                ));
            }
        }
        "remotely_materializable" => {
            state.counts.remotely_materializable += 1;
            if !downloaded_paths.is_empty() {
                state.failures.push(format!(
                    "{art_id}: remotely_materializable rows carry no downloaded_paths; \
                     per-host presence belongs in the materialization manifest"
                ));
            }
        }
        "downloadable" => state.counts.downloadable += 1,
        "blocked" => {
            state.counts.blocked += 1;
            if !working.is_empty() {
                state
                    .failures
                    .push(format!("{art_id}: blocked status but has working_mirrors"));
            }
        }
        "citation_only_no_link" => {
            state.counts.citation_only += 1;
            if !all_links.is_empty()
                && !key_is_citation_locator(&key)
                && !all_links.iter().all(|url| is_citation_locator_url(url))
            {
                state.failures.push(format!(
                    "{art_id}: citation_only_no_link but all_links is not empty"
                ));
            }
        }
        "unverified" => state.counts.unverified += 1,
        _ => {}
    }
    let sha256 = table
        .get("sha256")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let materializable_minimum =
        status == "remotely_materializable" && !canonical_url.is_empty() && !sha256.is_empty();
    if minimum_met != (!working.is_empty() || !downloaded_paths.is_empty() || materializable_minimum)
    {
        state.failures.push(format!(
            "{art_id}: minimum_requirement_met mismatch with working/downloaded mirrors"
        ));
    }
    if !minimum_met {
        state.counts.missing_minimum += 1;
        if !coverage_missing_keys.contains(&key) {
            state.failures.push(format!(
                "{art_id}: missing minimum requirement but key absent from coverage.artifacts_without_working_mirror"
            ));
        }
    }
    if manual {
        state.counts.manual += 1;
    }
    if working_pdf.len() > working.len() {
        state.failures.push(format!(
            "{art_id}: working_pdf_mirrors cannot exceed working_mirrors"
        ));
    }
    if !canonical_path.is_empty() && !repo_root.join(&canonical_path).exists() {
        state.failures.push(format!(
            "{art_id}: canonical_download_path does not exist: {canonical_path}"
        ));
    }
    for path in &downloaded_paths {
        if !repo_root.join(path).exists() {
            state
                .failures
                .push(format!("{art_id}: downloaded path missing on disk: {path}"));
        }
    }
    if !minimum_met
        && status != "citation_only_no_link"
        && nonworking.is_empty()
        && unverified_mirrors.is_empty()
    {
        state.failures.push(format!(
            "{art_id}: neither nonworking nor unverified mirrors recorded despite missing minimum"
        ));
    }
}

fn verify_header_counts(
    head: &toml::map::Map<String, Value>,
    coverage: &toml::map::Map<String, Value>,
    counts: &ArtifactCounts,
    artifact_count: usize,
    coverage_missing_keys: &[String],
    failures: &mut Vec<String>,
) {
    let expected_counts = [
        ("artifact_count", artifact_count),
        ("downloaded_count", counts.downloaded),
        ("remotely_materializable_count", counts.remotely_materializable),
        ("downloadable_count", counts.downloadable),
        ("blocked_count", counts.blocked),
        ("citation_only_no_link_count", counts.citation_only),
        ("unverified_count", counts.unverified),
        ("missing_minimum_requirement_count", counts.missing_minimum),
        ("manual_intervention_required_count", counts.manual),
    ];
    for (key, expected) in expected_counts {
        let observed = head.get(key).and_then(Value::as_integer).unwrap_or(-1);
        if observed != expected as i64 {
            failures.push(format!(
                "header {key} mismatch: header={observed} computed={expected}"
            ));
        }
    }
    let source_files = head
        .get("source_files")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    let source_file_count = head
        .get("source_file_count")
        .and_then(Value::as_integer)
        .unwrap_or(-1);
    if source_file_count != source_files.len() as i64 {
        failures
            .push("header source_file_count mismatch with source_files list length".to_string());
    }
    let source_tables = head
        .get("source_tables")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    let source_table_count = head
        .get("source_table_count")
        .and_then(Value::as_integer)
        .unwrap_or(-1);
    if source_table_count != source_tables.len() as i64 {
        failures
            .push("header source_table_count mismatch with source_tables list length".to_string());
    }
    let coverage_count = coverage
        .get("artifacts_without_working_mirror_count")
        .and_then(Value::as_integer)
        .unwrap_or(-1);
    if coverage_count != coverage_missing_keys.len() as i64 {
        failures.push(
            "coverage artifacts_without_working_mirror_count mismatch with list length".to_string(),
        );
    }
    if coverage_count != counts.missing_minimum as i64 {
        failures.push(
            "coverage artifacts_without_working_mirror_count mismatch with computed missing minimum count".to_string(),
        );
    }
}

pub fn verify_artifact_source_of_truth(
    repo_root: &Path,
    registry_path: &Path,
) -> Result<VerifySummary> {
    let value = load_toml_value(registry_path)?;
    let Some(head) = value
        .get("artifact_source_of_truth")
        .and_then(Value::as_table)
    else {
        bail!("artifact_source_of_truth header missing");
    };
    let Some(coverage) = value.get("coverage").and_then(Value::as_table) else {
        bail!("coverage table missing");
    };
    let artifacts = value
        .get("artifact")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let coverage_missing_keys = coverage
        .get("artifacts_without_working_mirror")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut state = ValidationState::default();
    for (index, artifact) in artifacts.iter().enumerate() {
        validate_artifact_entry(
            index,
            artifact,
            repo_root,
            &coverage_missing_keys,
            &mut state,
        );
    }
    verify_header_counts(
        head,
        coverage,
        &state.counts,
        artifacts.len(),
        &coverage_missing_keys,
        &mut state.failures,
    );

    if !state.failures.is_empty() {
        bail!(
            "artifact source-of-truth verification failed:\n- {}",
            state.failures.join("\n- ")
        );
    }
    Ok(VerifySummary {
        artifact_count: artifacts.len(),
        downloaded_count: state.counts.downloaded,
        remotely_materializable_count: state.counts.remotely_materializable,
        downloadable_count: state.counts.downloadable,
        blocked_count: state.counts.blocked,
        citation_only_count: state.counts.citation_only,
        unverified_count: state.counts.unverified,
        missing_minimum_count: state.counts.missing_minimum,
    })
}

pub fn verify_source_infrastructure(
    repo_root: &Path,
    infrastructure_path: &Path,
) -> Result<SourceInfrastructureSummary> {
    let infra = load_toml_value(infrastructure_path)?;
    let infra_head = infra
        .get("source_infrastructure")
        .and_then(Value::as_table)
        .context("source_infrastructure header missing")?;
    let lane_defs = infra
        .get("lane")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let master_rel = infra_head
        .get("master_registry")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if master_rel.is_empty() {
        bail!("infrastructure missing master_registry");
    }
    let master_path = repo_root.join(&master_rel);
    if !master_path.exists() {
        bail!("missing master registry: {}", master_path.display());
    }
    let master = load_toml_value(&master_path)?;
    let artifacts = master
        .get("artifact")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut failures = Vec::new();
    let master_ids = artifacts
        .iter()
        .filter_map(|artifact| artifact.get("id").and_then(Value::as_str).map(str::trim))
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    let master_id_set = master_ids.iter().cloned().collect::<HashSet<_>>();
    if master_ids.len() != master_id_set.len() {
        failures.push("master has duplicate or empty artifact ids".to_string());
    }
    let mut lane_membership = HashMap::new();
    let mut lane_total = 0usize;
    let mut lane_counts = BTreeMap::new();
    for lane_def in lane_defs {
        let Some(table) = lane_def.as_table() else {
            failures.push("lane definition missing name/path".to_string());
            continue;
        };
        let lane_name = table
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let lane_rel = table
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let expected_count = table
            .get("artifact_count")
            .and_then(Value::as_integer)
            .unwrap_or(-1);
        if lane_name.is_empty() || lane_rel.is_empty() {
            failures.push("lane definition missing name/path".to_string());
            continue;
        }
        let lane_path = repo_root.join(&lane_rel);
        if !lane_path.exists() {
            failures.push(format!("lane file missing: {lane_rel}"));
            continue;
        }
        let lane_data = load_toml_value(&lane_path)?;
        let lane_head = lane_data
            .get("lane")
            .and_then(Value::as_table)
            .cloned()
            .unwrap_or_default();
        let lane_artifacts = lane_data
            .get("artifact_ref")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let lane_count = lane_artifacts.len();
        lane_total += lane_count;
        lane_counts.insert(lane_name.clone(), lane_count);
        if lane_head
            .get("artifact_count")
            .and_then(Value::as_integer)
            .unwrap_or(-1)
            != lane_count as i64
        {
            failures.push(format!("lane header artifact_count mismatch: {lane_rel}"));
        }
        if expected_count != lane_count as i64 {
            failures.push(format!(
                "infrastructure artifact_count mismatch for lane {lane_name}: infra={expected_count} lane={lane_count}"
            ));
        }
        for artifact in lane_artifacts {
            let Some(artifact_table) = artifact.as_table() else {
                failures.push(format!("{lane_rel}: artifact_ref missing id"));
                continue;
            };
            let aid = artifact_table
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            if aid.is_empty() {
                failures.push(format!("{lane_rel}: artifact_ref missing id"));
                continue;
            }
            if !master_id_set.contains(&aid) {
                failures.push(format!("{lane_rel}: unknown artifact id {aid}"));
                continue;
            }
            if let Some(existing) = lane_membership.insert(aid.clone(), lane_name.clone())
                && existing != lane_name
            {
                failures.push(format!(
                    "artifact {aid} appears in multiple lanes: {existing}, {lane_name}"
                ));
            }
        }
    }
    let missing_from_lanes = master_id_set
        .difference(&lane_membership.keys().cloned().collect())
        .count();
    if missing_from_lanes > 0 {
        failures.push(format!(
            "{missing_from_lanes} master artifacts missing lane assignment"
        ));
    }
    let infra_total = infra_head
        .get("total_artifact_count")
        .and_then(Value::as_integer)
        .unwrap_or(-1);
    if infra_total != artifacts.len() as i64 {
        failures.push(format!(
            "infrastructure total_artifact_count mismatch: infra={infra_total} master={}",
            artifacts.len()
        ));
    }
    if lane_total != artifacts.len() {
        failures.push(format!(
            "lane total mismatch: lane_total={lane_total} master={}",
            artifacts.len()
        ));
    }
    if !failures.is_empty() {
        bail!(
            "source infrastructure verification failed:\n- {}",
            failures.join("\n- ")
        );
    }
    Ok(SourceInfrastructureSummary {
        total_artifact_count: artifacts.len(),
        lane_counts,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn stable_ids_keep_prior_numbers_and_allocate_after_the_maximum() {
        let mut prior = super::BTreeMap::new();
        prior.insert(
            "url:b".to_string(),
            super::PriorRow {
                id: "ASOT-0007".to_string(),
                ..Default::default()
            },
        );
        prior.insert(
            "url:d".to_string(),
            super::PriorRow {
                id: "ASOT-0003".to_string(),
                ..Default::default()
            },
        );
        let artifacts = ["url:a", "url:b", "url:c", "url:d"]
            .iter()
            .map(|key| super::UnifiedArtifact {
                key: key.to_string(),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let ids = super::assign_stable_ids(&artifacts, &prior);
        assert_eq!(ids, ["ASOT-0008", "ASOT-0007", "ASOT-0009", "ASOT-0003"]);
    }

    #[test]
    fn prior_rows_the_scan_did_not_observe_are_reseeded_in_key_order() {
        let mut prior = super::BTreeMap::new();
        prior.insert(
            "url:z".to_string(),
            super::PriorRow {
                id: "ASOT-0002".to_string(),
                title: "kept".to_string(),
                links: vec!["https://example.org/z".to_string()],
                ..Default::default()
            },
        );
        prior.insert(
            "url:m".to_string(),
            super::PriorRow {
                id: "ASOT-0001".to_string(),
                ..Default::default()
            },
        );
        let mut artifacts = vec![super::UnifiedArtifact {
            key: "url:m".to_string(),
            title: "observed".to_string(),
            ..Default::default()
        }];
        let seeded = super::seed_missing_prior_rows(&mut artifacts, &prior);
        assert_eq!(seeded, 1);
        assert_eq!(artifacts.len(), 2);
        assert_eq!(artifacts[0].key, "url:m");
        assert_eq!(artifacts[0].title, "observed");
        assert_eq!(artifacts[1].key, "url:z");
        assert_eq!(artifacts[1].title, "kept");
        assert!(artifacts[1].local_paths.is_empty());
    }

    use super::*;

    #[test]
    fn artifact_local_path_filter_rejects_agents() {
        assert!(!is_artifact_local_path("AGENTS.md"));
        assert!(is_artifact_local_path("data/example/file.pdf"));
    }

    #[test]
    fn lane_classification_prefers_dataset() {
        let mut artifact = toml::map::Map::new();
        artifact.insert(
            "canonical_download_path".to_string(),
            Value::String("data/example/table.csv".to_string()),
        );
        let (primary, tags) = classify_lane(&artifact);
        assert_eq!(primary, "datasets");
        assert!(tags.contains(&"datasets".to_string()));
    }

    #[test]
    fn normalize_url_upgrades_mdpi_http_and_drops_query_noise() {
        let normalized =
            normalize_url("http://www.mdpi.com/2073-8994/16/5/626/pdf?version=1715949362");
        assert_eq!(normalized, "https://www.mdpi.com/2073-8994/16/5/626/pdf");
    }

    #[test]
    fn normalize_url_upgrades_dx_doi_and_ou_http_fallbacks() {
        assert_eq!(
            normalize_url("http://dx.doi.org/10.1007/BF00653317"),
            "https://doi.org/10.1007/BF00653317"
        );
        assert_eq!(
            normalize_url("http://arxiv.org/abs/1009.1166"),
            "https://arxiv.org/abs/1009.1166"
        );
        assert_eq!(
            normalize_url("https://www2.math.ou.edu/~kmartin/quaint/ch3.pdf"),
            "http://www2.math.ou.edu/~kmartin/quaint/ch3.pdf"
        );
    }

    #[test]
    fn normalize_url_rewrites_archive_onion_aliases_to_public_archive() {
        assert_eq!(
            normalize_url(
                "https://archivep75mbjunhxc6x4j5mwjmomyxb573v42baldlqu56ruil2oiad.onion/download/arxiv-1602.02317/1602.02317.pdf"
            ),
            "https://archive.org/download/arxiv-1602.02317/1602.02317.pdf"
        );
    }

    #[test]
    fn canonical_identity_prefers_mdpi_pdf_variant() {
        let urls = vec!["https://www.mdpi.com/2073-8994/16/5/626".to_string()];
        assert_eq!(
            canonical_identity_url(&urls).as_deref(),
            Some("https://www.mdpi.com/2073-8994/16/5/626/pdf")
        );
    }

    #[test]
    fn cambridge_content_id_unifies_article_and_pdf_variants() {
        let article = "https://www.cambridge.org/core/journals/canadian-mathematical-bulletin/article/conjugacy-classes-of-subalgebras-of-the-real-sedenions/E3602D99D8C6C96F78EADAD2EDC1BC27";
        let pdf = "https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E3602D99D8C6C96F78EADAD2EDC1BC27/S0008439500006020a.pdf/conjugacy-classes-of-subalgebras-of-the-real-sedenions.pdf";
        assert_eq!(
            cambridge_content_id(article).as_deref(),
            Some("e3602d99d8c6c96f78eadad2edc1bc27")
        );
        assert_eq!(
            cambridge_content_id(pdf).as_deref(),
            Some("e3602d99d8c6c96f78eadad2edc1bc27")
        );
        let urls = vec![article.to_string(), pdf.to_string()];
        assert_eq!(
            canonical_identity_url(&urls).as_deref(),
            Some("cambridge:e3602d99d8c6c96f78eadad2edc1bc27")
        );
    }

    #[test]
    fn identity_key_keeps_cambridge_identity_prefix() {
        let candidate = CandidateRecord {
            links: vec![
                "https://www.cambridge.org/core/journals/canadian-mathematical-bulletin/article/conjugacy-classes-of-subalgebras-of-the-real-sedenions/E3602D99D8C6C96F78EADAD2EDC1BC27".to_string(),
            ],
            ..CandidateRecord::default()
        };
        assert_eq!(
            identity_key(&candidate),
            "cambridge:e3602d99d8c6c96f78eadad2edc1bc27"
        );
    }

    #[test]
    fn aps_abstract_pages_are_citation_locators() {
        assert!(is_citation_locator_url(
            "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.43.744"
        ));
        assert!(key_is_citation_locator(
            "url:https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.43.744"
        ));
    }

    #[test]
    fn cambridge_service_pages_are_citation_locators() {
        assert!(is_citation_locator_url(
            "https://www.cambridge.org/core/product/identifier/S0960129503004110/type/journal_article"
        ));
        assert!(is_citation_locator_url(
            "https://www.cambridge.org/core/tdm/tdm-policy.json"
        ));
    }

    #[test]
    fn numdam_item_pages_are_citation_locators_but_pdfs_are_not() {
        assert!(is_citation_locator_url(
            "https://www.numdam.org/item/AIHPA_1965__2_4_283_0/"
        ));
        assert!(!is_citation_locator_url(
            "https://www.numdam.org/item/JMPA_1923_9_2__281_0.pdf"
        ));
    }

    #[test]
    fn soho_archive_portals_are_citation_locators() {
        assert!(is_citation_locator_url(
            "https://soho.nascom.nasa.gov/data/archive/"
        ));
        assert!(is_citation_locator_url(
            "https://ssa.esac.esa.int/ssa-sl-tap/tap/capabilities"
        ));
        assert!(is_citation_locator_url(
            "https://www.cosmos.esa.int/web/soho/mission-long-files"
        ));
    }

    #[test]
    fn actaphys_r_pdf_surface_expands_to_fulltext_alias() {
        let aliases = expand_reference_aliases("https://www.actaphys.uj.edu.pl/R/27/8/1849/pdf");
        assert!(aliases.contains(
            &"https://www.actaphys.uj.edu.pl/fulltext?series=Reg&vol=27&page=1849".to_string()
        ));
    }

    #[test]
    fn iastate_download_surface_expands_to_api_content_alias() {
        let aliases = expand_reference_aliases(
            "https://dr.lib.iastate.edu/bitstreams/79b32677-687d-4469-979a-df3da9fcf6db/download",
        );
        assert!(aliases.contains(
            &"https://dr.lib.iastate.edu/server/api/core/bitstreams/79b32677-687d-4469-979a-df3da9fcf6db/content".to_string()
        ));
    }
}
