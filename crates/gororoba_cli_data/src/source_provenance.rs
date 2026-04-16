use anyhow::{Context, Result, bail};
use chrono::Utc;
use csv::ReaderBuilder;
use regex::Regex;
use rusqlite::Connection;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};
use toml::Value;
use url::Url;
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

const VALID_STATUSES: &[&str] = &[
    "downloaded",
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
}

#[derive(Clone, Debug, Default)]
pub struct VerifySummary {
    pub artifact_count: usize,
    pub downloaded_count: usize,
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
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate must be nested under repo/crates")
        .to_path_buf()
}

fn url_re() -> &'static Regex {
    static URL_RE: OnceLock<Regex> = OnceLock::new();
    URL_RE.get_or_init(|| Regex::new(r"(?i)^https?://").expect("valid URL regex"))
}

fn url_inline_re() -> &'static Regex {
    static URL_INLINE_RE: OnceLock<Regex> = OnceLock::new();
    URL_INLINE_RE
        .get_or_init(|| Regex::new(r#"(?i)https?://[^\s<>()"']+"#).expect("valid inline URL regex"))
}

fn doi_re() -> &'static Regex {
    static DOI_RE: OnceLock<Regex> = OnceLock::new();
    DOI_RE.get_or_init(|| {
        Regex::new(r"(?i)10\.\d{4,9}/[-._;()/:A-Za-z0-9]+").expect("valid DOI regex")
    })
}

fn bib_entry_re() -> &'static Regex {
    static BIB_ENTRY_RE: OnceLock<Regex> = OnceLock::new();
    BIB_ENTRY_RE.get_or_init(|| {
        Regex::new(r"(?s)@(?P<etype>[A-Za-z]+)\s*\{\s*(?P<key>[^,]+)\s*,(?P<body>.*?)\n\}\s*")
            .expect("valid BibTeX regex")
    })
}

fn ascii_sanitize(text: &str) -> String {
    text.chars()
        .map(|ch| {
            let code = ch as u32;
            if code >= 128 || (code < 32 && !matches!(ch, '\n' | '\r' | '\t')) || code == 127 {
                ' '
            } else {
                ch
            }
        })
        .collect()
}

fn escape_toml(text: &str) -> String {
    let sanitized = ascii_sanitize(text);
    let escaped = sanitized
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

fn render_list(values: &[String]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    let body = values
        .iter()
        .map(|value| escape_toml(value))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

fn assert_ascii(text: &str, context: &str) -> Result<()> {
    if !text.is_ascii() {
        bail!("non-ASCII output in {context}");
    }
    Ok(())
}

fn slug(text: &str) -> String {
    let mut out = String::new();
    let mut last_was_sep = false;
    for ch in text.to_ascii_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed
    }
}

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

fn strip_url_wrappers(value: &mut String) {
    while let Some(ch) = value.chars().next() {
        if "(<[{\"'".contains(ch) {
            value.remove(0);
        } else {
            break;
        }
    }
    while let Some(ch) = value.chars().last() {
        if ">)]}\"'`.,;:".contains(ch) {
            value.pop();
        } else {
            break;
        }
    }
}

fn rewrite_arxiv_typo_prefix(trimmed: &str) -> Option<String> {
    if let Some(suffix) = trimmed
        .strip_prefix("http://arxiv.org.abs/")
        .or_else(|| trimmed.strip_prefix("https://arxiv.org.abs/"))
    {
        return Some(normalize_url(&format!("https://arxiv.org/abs/{suffix}")));
    }
    if let Some(suffix) = trimmed
        .strip_prefix("http://arxiv.org/abs.")
        .or_else(|| trimmed.strip_prefix("https://arxiv.org/abs."))
    {
        return Some(normalize_url(&format!("https://arxiv.org/abs/{suffix}")));
    }
    None
}

fn apply_host_specific_rewrites(parsed: &mut Url, host: &str) {
    if parsed.scheme() == "http"
        && matches!(
            host,
            "arxiv.org"
                | "export.arxiv.org"
                | "www.mdpi.com"
                | "mdpi.com"
                | "doi.org"
                | "dx.doi.org"
                | "www.doi.org"
                | "www.cambridge.org"
                | "cambridge.org"
                | "www.academia.edu"
                | "academia.edu"
                | "www.researchgate.net"
                | "researchgate.net"
        )
    {
        let _ = parsed.set_scheme("https");
    }
    if host == "dx.doi.org" || host == "www.doi.org" {
        let _ = parsed.set_host(Some("doi.org"));
    }
    if host == "www2.math.ou.edu" {
        let _ = parsed.set_scheme("http");
    }
    parsed.set_fragment(None);
    if host == "arxiv.org" {
        if parsed.path().starts_with("/pdf/") && !parsed.path().ends_with(".pdf") {
            parsed.set_path(&format!("{}.pdf", parsed.path()));
        } else if let Some(vc_suffix) = parsed.path().strip_prefix("/vc/") {
            let parts: Vec<_> = vc_suffix.split('/').collect();
            if parts.len() >= 4 && parts[1] == "papers" && parts[3].ends_with(".pdf") {
                parsed.set_path(&format!("/pdf/{}/{}", parts[0], parts[3]));
            }
        }
    }
    if (host == "core.ac.uk" || host == "files01.core.ac.uk")
        && parsed.path().starts_with("/download/")
    {
        let path = parsed.path().to_string();
        if let Some(suffix) = path.strip_prefix("/download/")
            && suffix.ends_with(".pdf")
            && !suffix.starts_with("pdf/")
        {
            parsed.set_path(&format!("/download/pdf/{suffix}"));
        }
    }
    if host == "www.sciencedirect.com" && parsed.path().contains("/pdfft") {
        parsed.set_path(&parsed.path().replace("/pdfft", "/pdf"));
    }
}

fn filter_tracking_query_params(parsed: &mut Url) {
    let filtered = parsed
        .query_pairs()
        .filter(|(key, _)| {
            let lower = key.to_ascii_lowercase();
            !lower.starts_with("utm_")
                && !matches!(
                    lower.as_str(),
                    "download" | "isdtmredir" | "md5" | "pid" | "version" | "code"
                )
        })
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        parsed.set_query(None);
    } else {
        let mut qp = parsed.query_pairs_mut();
        qp.clear();
        for (k, v) in filtered {
            qp.append_pair(&k, &v);
        }
        drop(qp);
    }
}

fn normalize_url(url: &str) -> String {
    let mut value = url.trim().trim_matches('`').to_string();
    if value.contains('|') {
        for part in value.split('|') {
            let normalized = normalize_url(part);
            if !normalized.is_empty() {
                return normalized;
            }
        }
        return String::new();
    }
    strip_url_wrappers(&mut value);
    let trimmed = value.trim();
    if let Some(rewritten) = rewrite_arxiv_typo_prefix(trimmed) {
        return rewritten;
    }
    let Ok(parsed) = Url::parse(trimmed) else {
        return trimmed.to_string();
    };
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    if host == "archivep75mbjunhxc6x4j5mwjmomyxb573v42baldlqu56ruil2oiad.onion"
        && parsed.path().starts_with("/download/")
    {
        return normalize_url(&format!("https://archive.org{}", parsed.path()));
    }
    if host == "idp.springer.com" {
        for (key, val) in parsed.query_pairs() {
            if key == "redirect_uri" {
                let redirected = val.trim();
                if !redirected.is_empty() {
                    return normalize_url(redirected);
                }
            }
        }
        return String::new();
    }
    let mut parsed = parsed;
    apply_host_specific_rewrites(&mut parsed, &host);
    if is_non_reference_service_url(&parsed) {
        return String::new();
    }
    filter_tracking_query_params(&mut parsed);
    parsed.to_string()
}

fn is_non_reference_service_url(parsed: &Url) -> bool {
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    let path = parsed.path().to_ascii_lowercase();
    if matches!(
        host.as_str(),
        "apollo.archive.org"
            | "av.archive.org"
            | "av.dev.archive.org"
            | "emularity-bios.ux-b.archive.org"
            | "emularity-config.ux-b.archive.org"
            | "emularity-engine.ux-b.archive.org"
            | "esm.archive.org"
            | "esm.ext.archive.org"
            | "offshoot.prod.archive.org"
            | "polyfill.archive.org"
    ) {
        return true;
    }
    if host == "archive.org" {
        return path.starts_with("/services/")
            || path.starts_with("/components/")
            || path.starts_with("/includes/")
            || path.starts_with("/offshoot_assets/")
            || path.starts_with("/upload/app/")
            || path == "/"
            || path == "/v/"
            || path.starts_with("/v/");
    }
    false
}

fn find_urls(text: &str) -> Vec<String> {
    url_inline_re()
        .find_iter(text)
        .map(|m| normalize_url(m.as_str()))
        .filter(|value| url_re().is_match(value))
        .collect()
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

fn normalize_doi(doi: &str) -> String {
    let mut value = doi.trim().to_string();
    let lower = value.to_ascii_lowercase();
    if lower.starts_with("https://doi.org/") {
        value = value["https://doi.org/".len()..].to_string();
    } else if lower.starts_with("http://doi.org/") {
        value = value["http://doi.org/".len()..].to_string();
    }
    if value.to_ascii_lowercase().starts_with("doi:") {
        value = value[4..].trim().to_string();
    }
    let mut normalized = value
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(['.', ',', ';', ')'])
        .to_string();
    loop {
        let lower = normalized.to_ascii_lowercase();
        if lower.ends_with("/fulltext") {
            normalized.truncate(normalized.len() - "/fulltext".len());
            continue;
        }
        if lower.ends_with("/pdf") {
            normalized.truncate(normalized.len() - "/pdf".len());
            continue;
        }
        if lower.ends_with(".pdf") {
            normalized.truncate(normalized.len() - ".pdf".len());
            continue;
        }
        break;
    }
    if doi_re().is_match(&normalized)
        && doi_re().find(&normalized).map(|m| m.as_str()) == Some(normalized.as_str())
    {
        normalized
    } else {
        String::new()
    }
}

fn extract_dois(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for text in extract_strings(value) {
        let cleaned = normalize_doi(&text);
        if doi_re().is_match(&cleaned)
            && doi_re().find(&cleaned).map(|m| m.as_str()) == Some(cleaned.as_str())
        {
            out.push(cleaned);
            continue;
        }
        for capture in doi_re().find_iter(&text) {
            out.push(normalize_doi(capture.as_str()));
        }
    }
    dedupe(out)
}

fn doi_to_url(doi: &str) -> String {
    format!("https://doi.org/{doi}")
}

fn doi_from_url(url: &str) -> String {
    if let Ok(parsed) = Url::parse(url) {
        let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
        if matches!(host.as_str(), "doi.org" | "dx.doi.org") {
            let doi = normalize_doi(parsed.path().trim_start_matches('/'));
            if !doi.is_empty() {
                return doi;
            }
        }
    }
    String::new()
}

fn extract_dois_from_urls(urls: &[String]) -> Vec<String> {
    dedupe(
        urls.iter()
            .map(|url| doi_from_url(url))
            .filter(|doi| !doi.is_empty())
            .collect(),
    )
}

fn looks_like_reference_url(url: &str) -> bool {
    if !url_re().is_match(url) {
        return false;
    }
    let Ok(parsed) = Url::parse(url) else {
        return false;
    };
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    if host == "idp.springer.com" || is_non_reference_service_url(&parsed) {
        return false;
    }
    if REFERENCE_HOST_HINTS
        .iter()
        .any(|hint| host == *hint || host.ends_with(&format!(".{hint}")))
    {
        return true;
    }
    if host.ends_with(".arxiv.org") || host.ends_with(".scispace.com") {
        return true;
    }
    let path = parsed.path().to_ascii_lowercase();
    path.ends_with(".pdf") || path.contains("/pdf")
}

fn is_citation_locator_url(url: &str) -> bool {
    let Ok(parsed) = Url::parse(url) else {
        return false;
    };
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    let path = parsed.path().to_ascii_lowercase();
    if matches!(host.as_str(), "doi.org" | "dx.doi.org") {
        return true;
    }
    if host == "arxiv.org" || host.ends_with(".arxiv.org") {
        return path.starts_with("/abs") || path.starts_with("/abs.") || path.starts_with("/abs/");
    }
    if host == "scispace.com" {
        return path.starts_with("/papers/");
    }
    if host == "linkinghub.elsevier.com" {
        return path.starts_with("/retrieve/pii/");
    }
    if host == "link.springer.com" {
        return path.starts_with("/article/")
            || path.starts_with("/chapter/")
            || path.starts_with("/referenceworkentry/");
    }
    if host == "www.cambridge.org" || host == "cambridge.org" {
        return path.starts_with("/core/product/identifier/") || path.starts_with("/core/tdm/");
    }
    if host == "zenodo.org" {
        return path.starts_with("/record/") || path.starts_with("/records/");
    }
    if host == "ncatlab.org" {
        return path.starts_with("/nlab/show/");
    }
    if host == "osf.io" {
        if path == "/" {
            return true;
        }
        let trimmed = path.trim_matches('/');
        if trimmed.is_empty() || trimmed.contains('.') {
            return false;
        }
        if trimmed.ends_with("/wiki") || trimmed == "wiki" {
            return true;
        }
        return trimmed
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-');
    }
    if host == "core.ac.uk" {
        return path.starts_with("/reader/");
    }
    if host == "www.numdam.org" || host == "numdam.org" {
        return path.starts_with("/item/") && !path.ends_with(".pdf");
    }
    if host == "www.sciencedirect.com" {
        return path.starts_with("/science/article/pii/") || path.starts_with("/journal/");
    }
    if host == "soho.nascom.nasa.gov" {
        return path.starts_with("/data/archive") || path.starts_with("/data/.dash/");
    }
    if host == "soho.esac.esa.int" {
        return path.starts_with("/data/archive/");
    }
    if host == "ssa.esac.esa.int" {
        return path.starts_with("/ssa/") || path.starts_with("/ssa-sl-tap/");
    }
    if host == "www.cosmos.esa.int" {
        return path.starts_with("/web/soho/");
    }
    if host == "journals.aps.org" {
        return path.contains("/abstract/");
    }
    if host == "archive.org" {
        return path.starts_with("/details/") || path.starts_with("/stream/");
    }
    if host == "web.archive.org" {
        return path.starts_with("/web/");
    }
    false
}

fn key_is_citation_locator(key: &str) -> bool {
    if key.to_ascii_lowercase().starts_with("doi:") {
        return true;
    }
    if let Some(url) = key.strip_prefix("url:") {
        return is_citation_locator_url(url);
    }
    false
}

fn is_artifact_local_path(path: &str) -> bool {
    ARTIFACT_LOCAL_PREFIXES
        .iter()
        .any(|prefix| path.trim().starts_with(prefix))
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

fn load_toml_value(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn read_text_lossy(path: &Path) -> Result<String> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

fn derive_status(row: &HashMap<String, String>) -> String {
    let status = row.get("status").cloned().unwrap_or_default();
    if !status.is_empty() {
        return status;
    }
    let result = row.get("result").cloned().unwrap_or_default();
    if !result.is_empty() {
        return result;
    }
    let http_code = row.get("http_code").cloned().unwrap_or_default();
    let is_pdf_raw = row
        .get("is_pdf")
        .cloned()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let is_pdf = matches!(is_pdf_raw.as_str(), "yes" | "true" | "1");
    if http_code.starts_with('2') && is_pdf {
        return "pdf_ok".to_string();
    }
    if http_code.starts_with('2') {
        return "ok_nonpdf".to_string();
    }
    if !http_code.is_empty() {
        return format!("http_{http_code}");
    }
    "unknown".to_string()
}

fn read_tsv_rows(path: &Path) -> Result<Vec<HashMap<String, String>>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .with_context(|| format!("open TSV {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("read TSV headers {}", path.display()))?
        .clone();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.with_context(|| format!("read TSV record {}", path.display()))?;
        let mut row = HashMap::new();
        for (header, field) in headers.iter().zip(record.iter()) {
            row.insert(header.to_string(), field.trim().to_string());
        }
        rows.push(row);
    }
    Ok(rows)
}

type LinkMap = (HashMap<String, Vec<LinkObservation>>, Vec<String>);
fn collect_link_observations(repo_root: &Path) -> Result<LinkMap> {
    let mut table_paths = Vec::new();
    for intake_root in provenance_intake_roots(repo_root) {
        if intake_root.exists() {
            for entry in WalkDir::new(&intake_root)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                if (name.starts_with("fetch_results") && name.ends_with("_normalized.tsv"))
                    || name.starts_with("mirror_retry_results")
                    || name.starts_with("link_audit_results")
                {
                    table_paths.push(path.to_path_buf());
                }
            }
        }
    }
    table_paths.sort();
    table_paths.dedup();

    let mut observations: HashMap<String, Vec<LinkObservation>> = HashMap::new();
    let mut source_tables = Vec::new();
    for path in table_paths {
        let rel = path
            .strip_prefix(repo_root)
            .unwrap_or(path.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        source_tables.push(rel.clone());
        for row in read_tsv_rows(&path)? {
            let url = normalize_url(row.get("url").map(String::as_str).unwrap_or_default());
            if !url_re().is_match(&url) {
                continue;
            }
            let status = derive_status(&row);
            observations
                .entry(url.clone())
                .or_default()
                .push(LinkObservation {
                    status: status.clone(),
                });

            let effective = normalize_url(
                row.get("url_effective")
                    .map(String::as_str)
                    .unwrap_or_default(),
            );
            if url_re().is_match(&effective) && effective != url {
                observations
                    .entry(effective)
                    .or_default()
                    .push(LinkObservation {
                        status: status.clone(),
                    });
            }
        }
    }
    let sqlite_path = repo_root.join("registry/canonical/control_plane.sqlite3");
    if sqlite_path.exists() {
        source_tables
            .push("registry/canonical/control_plane.sqlite3::download_attempts".to_string());
        merge_sqlite_download_observations(&sqlite_path, &mut observations)?;
    }
    Ok((observations, source_tables))
}

fn merge_sqlite_download_observations(
    sqlite_path: &Path,
    observations: &mut HashMap<String, Vec<LinkObservation>>,
) -> Result<()> {
    let conn = Connection::open(sqlite_path)
        .with_context(|| format!("open sqlite {}", sqlite_path.display()))?;
    let mut stmt = conn.prepare(
        "SELECT j.requested_url, j.final_url, a.http_code, a.is_pdf, a.succeeded, a.failure_class
         FROM download_attempts a
         JOIN download_jobs j ON j.id = a.job_id",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<i64>>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, Option<String>>(5)?,
        ))
    })?;
    for row in rows {
        let (requested_url, final_url, http_code, is_pdf, succeeded, failure_class) = row?;
        let status = derive_attempt_status(http_code, is_pdf != 0, succeeded != 0, failure_class);
        let requested_url = normalize_url(&requested_url);
        if url_re().is_match(&requested_url) {
            observations
                .entry(requested_url)
                .or_default()
                .push(LinkObservation {
                    status: status.clone(),
                });
        }
        if let Some(final_url) = final_url {
            let final_url = normalize_url(&final_url);
            if url_re().is_match(&final_url) {
                observations
                    .entry(final_url)
                    .or_default()
                    .push(LinkObservation {
                        status: status.clone(),
                    });
            }
        }
    }
    Ok(())
}

fn derive_attempt_status(
    http_code: Option<i64>,
    is_pdf: bool,
    succeeded: bool,
    failure_class: Option<String>,
) -> String {
    if let Some(code) = http_code {
        let code_str = code.to_string();
        if code_str.starts_with('2') && is_pdf {
            return "pdf_ok".to_string();
        }
        if code_str.starts_with('2') {
            return "ok_nonpdf".to_string();
        }
        return format!("http_{code}");
    }
    if succeeded {
        return if is_pdf {
            "pdf_ok".to_string()
        } else {
            "ok_nonpdf".to_string()
        };
    }
    if let Some(failure_class) = failure_class
        && !failure_class.trim().is_empty()
    {
        return "failed".to_string();
    }
    "unknown".to_string()
}

fn collect_download_map(repo_root: &Path) -> Result<HashMap<String, Vec<String>>> {
    let mut url_to_paths: HashMap<String, Vec<String>> = HashMap::new();
    for intake_root in provenance_intake_roots(repo_root) {
        if intake_root.exists() {
            for entry in WalkDir::new(&intake_root)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if !path.is_file()
                    || path.file_name().and_then(|n| n.to_str()) != Some("pdf_success_added.tsv")
                {
                    continue;
                }
                let pdf_dir = path.parent().unwrap_or(path).join("pdf_success");
                for row in read_tsv_rows(path)? {
                    let source_url = normalize_url(
                        row.get("source_url")
                            .map(String::as_str)
                            .unwrap_or_default(),
                    );
                    let name = row.get("canonical_pdf_name").cloned().unwrap_or_default();
                    if !url_re().is_match(&source_url) || name.is_empty() {
                        continue;
                    }
                    let candidate = pdf_dir.join(&name);
                    if candidate.exists() {
                        let rel = candidate
                            .strip_prefix(repo_root)
                            .unwrap_or(candidate.as_path())
                            .to_string_lossy()
                            .replace('\\', "/");
                        register_download_aliases(&mut url_to_paths, &source_url, &rel);
                    }
                }
            }
        }
    }

    let cdcs_path = repo_root.join("registry/cayley_dickson_canonical_sources.toml");
    if cdcs_path.exists() {
        let data = load_toml_value(&cdcs_path)?;
        if let Some(papers) = data.get("paper").and_then(Value::as_array) {
            for paper in papers {
                let Some(table) = paper.as_table() else {
                    continue;
                };
                let path = table
                    .get("canonical_pdf_path")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                let url = normalize_url(
                    table
                        .get("canonical_functional_url")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
                if path.is_empty() {
                    continue;
                }
                let candidate = repo_root.join(path);
                if !candidate.exists() {
                    continue;
                }
                let rel = candidate
                    .strip_prefix(repo_root)
                    .unwrap_or(candidate.as_path())
                    .to_string_lossy()
                    .replace('\\', "/");
                if url_re().is_match(&url) {
                    register_download_aliases(&mut url_to_paths, &url, &rel);
                }
                if let Some(mirrors) = table.get("working_pdf_mirrors").and_then(Value::as_array) {
                    for mirror in mirrors.iter().filter_map(Value::as_str) {
                        let mirror_url = normalize_url(mirror);
                        if url_re().is_match(&mirror_url) {
                            register_download_aliases(&mut url_to_paths, &mirror_url, &rel);
                        }
                    }
                }
            }
        }
    }

    let brown_report = repo_root.join("reports/cayley_dickson_source_recovery_2026_02_15.toml");
    if brown_report.exists() {
        let data = load_toml_value(&brown_report)?;
        if let Some(table) = data.get("brown_1972").and_then(Value::as_table) {
            let path = table
                .get("canonical_pdf_path")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let url = normalize_url(
                table
                    .get("core_download_url")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
            if !path.is_empty() && url_re().is_match(&url) {
                let candidate = repo_root.join(path);
                if candidate.exists() {
                    let rel = candidate
                        .strip_prefix(repo_root)
                        .unwrap_or(candidate.as_path())
                        .to_string_lossy()
                        .replace('\\', "/");
                    url_to_paths.entry(url).or_default().push(rel);
                }
            }
        }
    }

    for paths in url_to_paths.values_mut() {
        *paths = dedupe(std::mem::take(paths));
    }
    Ok(url_to_paths)
}

fn provenance_intake_roots(repo_root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![
        repo_root.join("data/external/intake"),
        repo_root.join("data/papers/intake"),
    ];
    roots.sort();
    roots.dedup();
    roots
}

fn extend_download_map_from_local_artifacts(
    download_map: &mut HashMap<String, Vec<String>>,
    artifacts: &[UnifiedArtifact],
) {
    for artifact in artifacts {
        if artifact.local_paths.is_empty() {
            continue;
        }
        for rel in &artifact.local_paths {
            for link in &artifact.links {
                register_download_aliases(download_map, link, rel);
            }
        }
    }
}

fn register_download_aliases(
    url_to_paths: &mut HashMap<String, Vec<String>>,
    source_url: &str,
    rel: &str,
) {
    let aliases = arxiv_equivalent_urls(source_url);
    if aliases.is_empty() {
        url_to_paths
            .entry(source_url.to_string())
            .or_default()
            .push(rel.to_string());
        return;
    }
    for alias in aliases {
        url_to_paths.entry(alias).or_default().push(rel.to_string());
    }
}

fn arxiv_equivalent_urls(url: &str) -> Vec<String> {
    let Ok(parsed) = Url::parse(url) else {
        return Vec::new();
    };
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    if host != "arxiv.org" {
        return Vec::new();
    }
    let path = parsed.path();
    let ident = if let Some(rest) = path.strip_prefix("/pdf/") {
        rest.strip_suffix(".pdf").unwrap_or(rest)
    } else if let Some(rest) = path.strip_prefix("/abs/") {
        rest
    } else {
        return Vec::new();
    };
    let base_ident = strip_arxiv_version(ident);
    let mut aliases = vec![
        normalize_url(&format!("https://arxiv.org/abs/{ident}")),
        normalize_url(&format!("https://arxiv.org/pdf/{ident}.pdf")),
    ];
    if base_ident != ident {
        aliases.push(normalize_url(&format!(
            "https://arxiv.org/abs/{base_ident}"
        )));
        aliases.push(normalize_url(&format!(
            "https://arxiv.org/pdf/{base_ident}.pdf"
        )));
    }
    dedupe(aliases)
}

fn strip_arxiv_version(ident: &str) -> String {
    let Some((prefix, suffix)) = ident.rsplit_once('v') else {
        return ident.to_string();
    };
    if !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()) {
        prefix.to_string()
    } else {
        ident.to_string()
    }
}

fn core_id_from_url(url: &str) -> Option<String> {
    let parsed = Url::parse(url).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    if host != "core.ac.uk" && host != "files01.core.ac.uk" {
        return None;
    }
    let path = parsed.path().trim_matches('/');
    if let Some(id) = path
        .strip_prefix("download/pdf/")
        .and_then(|value| value.strip_suffix(".pdf"))
    {
        return Some(id.to_string());
    }
    for prefix in ["reader/", "works/", "display/"] {
        if let Some(id) = path.strip_prefix(prefix)
            && !id.is_empty()
        {
            return Some(id.to_string());
        }
    }
    None
}

fn mdpi_path_looks_article(parts: &[&str]) -> bool {
    if parts.len() != 4 {
        return false;
    }
    let issn = parts[0];
    let issn_ok = issn.len() == 9
        && issn.chars().enumerate().all(|(idx, ch)| {
            if idx == 4 {
                ch == '-'
            } else {
                ch.is_ascii_digit()
            }
        });
    issn_ok
        && parts[1..]
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_digit()))
}

fn cambridge_content_id(url: &str) -> Option<String> {
    let parsed = Url::parse(url).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    if host != "www.cambridge.org" && host != "cambridge.org" {
        return None;
    }
    let segments = parsed
        .path_segments()?
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    for (index, segment) in segments.iter().enumerate() {
        if segment.eq_ignore_ascii_case("view") {
            return segments
                .get(index + 1)
                .map(|value| value.to_ascii_lowercase());
        }
    }
    segments.last().map(|value| value.to_ascii_lowercase())
}

fn canonical_identity_url(urls: &[String]) -> Option<String> {
    for url in urls {
        if let Some(content_id) = cambridge_content_id(url) {
            return Some(format!("cambridge:{content_id}"));
        }
    }
    for url in urls {
        let Ok(parsed) = Url::parse(url) else {
            continue;
        };
        let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
        if host == "www.mdpi.com" || host == "mdpi.com" {
            let path = parsed.path().trim_end_matches('/');
            let parts = path.trim_matches('/').split('/').collect::<Vec<_>>();
            if mdpi_path_looks_article(&parts) {
                return Some(normalize_url(&format!(
                    "https://{host}/{}/{}/{}/{}/pdf",
                    parts[0], parts[1], parts[2], parts[3]
                )));
            }
            if parts.len() == 5
                && parts[4].eq_ignore_ascii_case("pdf")
                && mdpi_path_looks_article(&parts[..4])
            {
                return Some(normalize_url(url));
            }
        }
    }
    urls.first().cloned()
}

fn expand_reference_aliases(url: &str) -> Vec<String> {
    let mut aliases = Vec::new();
    if let Some(core_id) = core_id_from_url(url) {
        aliases.push(normalize_url(&format!(
            "https://core.ac.uk/download/pdf/{core_id}.pdf"
        )));
        aliases.push(normalize_url(&format!(
            "https://core.ac.uk/reader/{core_id}"
        )));
        aliases.push(normalize_url(&format!(
            "https://core.ac.uk/works/{core_id}"
        )));
        aliases.push(normalize_url(&format!(
            "https://core.ac.uk/display/{core_id}"
        )));
    }
    if let Ok(parsed) = Url::parse(url) {
        let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
        if host == "www.mdpi.com" || host == "mdpi.com" {
            let path = parsed.path().trim_end_matches('/');
            let parts = path.trim_matches('/').split('/').collect::<Vec<_>>();
            if mdpi_path_looks_article(&parts) {
                aliases.push(normalize_url(&format!(
                    "https://{host}/{}/{}/{}/{}/pdf",
                    parts[0], parts[1], parts[2], parts[3]
                )));
            } else if parts.len() == 5
                && parts[4].eq_ignore_ascii_case("pdf")
                && mdpi_path_looks_article(&parts[..4])
            {
                aliases.push(normalize_url(&format!(
                    "https://{host}/{}/{}/{}/{}",
                    parts[0], parts[1], parts[2], parts[3]
                )));
            }
        }
        if host == "www.wolframscience.com" || host == "wolframscience.com" {
            let path = parsed.path();
            if path.contains("/presentations/materials/") {
                aliases.push(normalize_url(&format!(
                    "https://{host}{}",
                    path.replace("/presentations/materials/", "/presentations/material/")
                )));
            }
        }
        if host == "dr.lib.iastate.edu" {
            let path = parsed.path().trim_matches('/');
            let parts = path.split('/').collect::<Vec<_>>();
            if parts.len() == 3 && parts[0] == "bitstreams" && parts[2] == "download" {
                aliases.push(normalize_url(&format!(
                    "https://{host}/server/api/core/bitstreams/{}/content",
                    parts[1]
                )));
            }
        }
        if host == "www.actaphys.uj.edu.pl" || host == "actaphys.uj.edu.pl" {
            let path = parsed.path().trim_matches('/');
            let parts = path.split('/').collect::<Vec<_>>();
            if parts.len() == 5
                && parts[0].eq_ignore_ascii_case("R")
                && parts[4].eq_ignore_ascii_case("pdf")
            {
                aliases.push(normalize_url(&format!(
                    "https://{host}/fulltext?series=Reg&vol={}&page={}",
                    parts[1], parts[3]
                )));
            }
        }
    }
    dedupe(aliases)
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

fn pick_first_str(table: &toml::map::Map<String, Value>, keys: &[&str]) -> String {
    for key in keys {
        if let Some(value) = table.get(*key).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    String::new()
}

fn extract_candidates_from_toml_node(
    repo_root: &Path,
    source_rel: &str,
    node: &Value,
    breadcrumbs: &[String],
    out: &mut Vec<CandidateRecord>,
) {
    match node {
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                let mut next = breadcrumbs.to_vec();
                next.push(index.to_string());
                extract_candidates_from_toml_node(repo_root, source_rel, item, &next, out);
            }
        }
        Value::Table(table) => {
            let mut title = pick_first_str(table, TITLE_KEYS);
            let citation = {
                let picked = pick_first_str(table, CITATION_KEYS);
                if picked.is_empty() {
                    title.clone()
                } else {
                    picked
                }
            };
            let ref_hint = pick_first_str(table, ID_KEYS);
            let identity_override = {
                let hint = pick_first_str(table, &["artifact_key_hint", "identity_hint"]);
                let normalized = normalize_identity_hint(&hint);
                if normalized.is_empty() {
                    None
                } else {
                    Some(normalized)
                }
            };
            let mut source_ref = format!("{source_rel}::{}", breadcrumbs.join("/"));
            if !ref_hint.is_empty() {
                source_ref.push_str("::");
                source_ref.push_str(&ref_hint);
            }

            let mut urls = Vec::new();
            let mut dois = Vec::new();
            let mut local_paths = Vec::new();
            let mut notes = Vec::new();
            for (key, value) in table {
                let lower = key.to_ascii_lowercase();
                if lower.contains("url")
                    || lower.contains("link")
                    || lower.contains("mirror")
                    || lower.contains("href")
                {
                    urls.extend(extract_urls(value));
                } else if lower.contains("doi") {
                    dois.extend(extract_dois(value));
                } else if lower.contains("path")
                    || lower.ends_with("_file")
                    || lower.ends_with("_files")
                    || lower == "files"
                {
                    local_paths.extend(extract_local_paths(value, repo_root));
                } else if matches!(
                    lower.as_str(),
                    "status" | "note" | "notes" | "reason" | "manual_intervention_reason"
                ) {
                    notes.extend(extract_strings(value));
                }
            }
            let filtered_urls = dedupe(
                urls.into_iter()
                    .filter(|url| looks_like_reference_url(url))
                    .collect(),
            );
            let dois = dedupe(dois);
            let local_paths = dedupe(local_paths);
            let notes = dedupe(notes);
            if !filtered_urls.is_empty() || !dois.is_empty() || !local_paths.is_empty() {
                if title.is_empty() {
                    title = if !citation.is_empty() {
                        citation.clone()
                    } else if let Some(url) = filtered_urls.first() {
                        url.clone()
                    } else if let Some(doi) = dois.first() {
                        doi.clone()
                    } else {
                        source_ref.clone()
                    };
                }
                let mut links = filtered_urls.clone();
                links.extend(dois.iter().map(|doi| doi_to_url(doi)));
                out.push(CandidateRecord {
                    source_kind: "toml_source".to_string(),
                    source_ref,
                    identity_override,
                    title: title.clone(),
                    citation: if citation.is_empty() { title } else { citation },
                    dois,
                    links: dedupe(links),
                    local_paths,
                    notes,
                });
            }
            for (key, value) in table {
                if matches!(value, Value::Table(_) | Value::Array(_)) {
                    let mut next = breadcrumbs.to_vec();
                    next.push(key.clone());
                    extract_candidates_from_toml_node(repo_root, source_rel, value, &next, out);
                }
            }
        }
        _ => {}
    }
}

fn extract_bib_field(body: &str, field: &str) -> String {
    let brace = Regex::new(&format!(
        r"(?is){}\s*=\s*\{{(?P<value>.*?)\}}",
        regex::escape(field)
    ))
    .expect("valid brace regex");
    if let Some(captures) = brace.captures(body) {
        return captures
            .name("value")
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default();
    }
    let quote = Regex::new(&format!(
        r#"(?is){}\s*=\s*"(?P<value>.*?)""#,
        regex::escape(field)
    ))
    .expect("valid quote regex");
    quote
        .captures(body)
        .and_then(|captures| {
            captures
                .name("value")
                .map(|m| m.as_str().trim().to_string())
        })
        .unwrap_or_default()
}

fn extract_candidates_from_bib_file(repo_root: &Path, path: &Path) -> Result<Vec<CandidateRecord>> {
    let rel = path
        .strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let text = read_text_lossy(path)?;
    let mut out = Vec::new();
    for captures in bib_entry_re().captures_iter(&text) {
        let etype = captures
            .name("etype")
            .map(|m| m.as_str().trim())
            .unwrap_or("");
        let key = captures
            .name("key")
            .map(|m| m.as_str().trim())
            .unwrap_or("");
        let body = captures.name("body").map(|m| m.as_str()).unwrap_or("");
        let title = extract_bib_field(body, "title");
        let citation = format!("@{etype}{{{key}}}");
        let mut urls = find_urls(body);
        let mut dois = extract_dois(&Value::String(extract_bib_field(body, "doi")));
        if dois.is_empty() {
            dois = extract_dois(&Value::String(body.to_string()));
        }
        if !urls.iter().any(|url| looks_like_reference_url(url)) && dois.is_empty() {
            continue;
        }
        urls = dedupe(
            urls.into_iter()
                .filter(|url| looks_like_reference_url(url))
                .collect(),
        );
        urls.extend(dois.iter().map(|doi| doi_to_url(doi)));
        out.push(CandidateRecord {
            source_kind: "bibtex_entry".to_string(),
            source_ref: format!("{rel}::{key}"),
            identity_override: None,
            title: if title.is_empty() {
                key.to_string()
            } else {
                title
            },
            citation,
            dois: dedupe(dois),
            links: dedupe(urls),
            local_paths: Vec::new(),
            notes: Vec::new(),
        });
    }
    Ok(out)
}

fn clean_line_title(line: &str) -> String {
    let url_free = url_inline_re().replace_all(line.trim(), "");
    let trimmed = url_free
        .trim_matches(|ch: char| " -*|`[]()".contains(ch))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    trimmed.trim().to_string()
}

fn extract_candidates_from_text_file(
    repo_root: &Path,
    path: &Path,
) -> Result<Vec<CandidateRecord>> {
    let rel = path
        .strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let text = read_text_lossy(path)?;
    let mut out = Vec::new();
    for (line_no, raw_line) in text.lines().enumerate() {
        let urls = dedupe(
            find_urls(raw_line)
                .into_iter()
                .filter(|url| looks_like_reference_url(url))
                .collect(),
        );
        let dois = extract_dois(&Value::String(raw_line.to_string()));
        if urls.is_empty() && dois.is_empty() {
            continue;
        }
        let title = {
            let cleaned = clean_line_title(raw_line);
            if !cleaned.is_empty() {
                cleaned
            } else if let Some(url) = urls.first() {
                url.clone()
            } else {
                dois[0].clone()
            }
        };
        if title.eq_ignore_ascii_case("onion-location:") {
            continue;
        }
        let mut links = urls.clone();
        links.extend(dois.iter().map(|doi| doi_to_url(doi)));
        out.push(CandidateRecord {
            source_kind: "text_reference".to_string(),
            source_ref: format!("{rel}:{}", line_no + 1),
            identity_override: None,
            title: title.clone(),
            citation: title,
            dois: dedupe(dois),
            links: dedupe(links),
            local_paths: Vec::new(),
            notes: Vec::new(),
        });
    }
    Ok(out)
}

fn extract_candidates_from_source_file(
    repo_root: &Path,
    path: &Path,
) -> Result<Vec<CandidateRecord>> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
    {
        "bib" | "bibtex" => extract_candidates_from_bib_file(repo_root, path),
        "toml" => {
            let value = match load_toml_value(path) {
                Ok(value) => value,
                Err(_) => return Ok(Vec::new()),
            };
            let rel = path
                .strip_prefix(repo_root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            let mut out = Vec::new();
            extract_candidates_from_toml_node(repo_root, &rel, &value, &[], &mut out);
            Ok(out)
        }
        "md" | "txt" | "rst" => extract_candidates_from_text_file(repo_root, path),
        _ => Ok(Vec::new()),
    }
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
        artifact.downloaded_paths = dedupe(downloaded);
        let citation_locator_identity = key_is_citation_locator(&artifact.key);
        let citation_locator_only_links = !artifact.links.is_empty()
            && artifact
                .links
                .iter()
                .all(|url| is_citation_locator_url(url));
        artifact.minimum_requirement_met =
            !(artifact.working_mirrors.is_empty() && artifact.downloaded_paths.is_empty());
        artifact.manual_intervention_required = !artifact.links.is_empty()
            && !artifact.minimum_requirement_met
            && !citation_locator_identity
            && !citation_locator_only_links;

        artifact.status = if !artifact.downloaded_paths.is_empty() {
            "downloaded".to_string()
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
        artifact.manual_intervention_reason = if artifact.manual_intervention_required {
            "No working mirror observed from current fetch/retry ledgers; manual link intervention required.".to_string()
        } else {
            String::new()
        };
    }
}

fn render_artifact_registry(
    artifacts: &[UnifiedArtifact],
    source_tables: &[String],
    source_files: &[String],
    now: &str,
) -> String {
    let total = artifacts.len();
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
        lines.push(format!(
            "id = {}",
            escape_toml(&format!("ASOT-{:04}", index + 1))
        ));
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

pub fn build_artifact_source_of_truth(
    repo_root: &Path,
    out_registry: &Path,
    out_report: &Path,
) -> Result<BuildSummary> {
    let now = Utc::now().format("%Y-%m-%d").to_string();
    let (observations, source_tables) = collect_link_observations(repo_root)?;
    let mut download_map = collect_download_map(repo_root)?;
    let (candidates, source_files) = build_candidates(repo_root)?;
    let mut artifacts = unify_candidates(candidates);
    extend_download_map_from_local_artifacts(&mut download_map, &artifacts);
    classify_artifacts(&mut artifacts, &observations, &download_map);
    let registry_text = render_artifact_registry(&artifacts, &source_tables, &source_files, &now);
    let report_text = render_reconciliation_report(&artifacts, &now);
    assert_ascii(&registry_text, &out_registry.display().to_string())?;
    assert_ascii(&report_text, &out_report.display().to_string())?;
    if let Some(parent) = out_registry.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = out_report.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_registry, registry_text)
        .with_context(|| format!("write {}", out_registry.display()))?;
    fs::write(out_report, report_text)
        .with_context(|| format!("write {}", out_report.display()))?;
    Ok(BuildSummary {
        artifact_count: artifacts.len(),
    })
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

pub fn build_source_truth_infrastructure(
    repo_root: &Path,
    source_path: &Path,
    out_infrastructure: &Path,
    lane_dir: &Path,
    out_report: &Path,
) -> Result<SourceInfrastructureSummary> {
    let value = load_toml_value(source_path)?;
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

    fs::create_dir_all(lane_dir).with_context(|| format!("create {}", lane_dir.display()))?;
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
        fs::write(&lane_path, lane_text)
            .with_context(|| format!("write {}", lane_path.display()))?;
        let rel = lane_path
            .strip_prefix(repo_root)
            .unwrap_or(lane_path.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        lane_files.insert((*lane).to_string(), rel);
        lane_counts.insert((*lane).to_string(), lane_artifacts.len());
    }

    let infrastructure_text = render_infrastructure(
        &source_path
            .strip_prefix(repo_root)
            .unwrap_or(source_path)
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
    if let Some(parent) = out_infrastructure.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = out_report.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_infrastructure, infrastructure_text)
        .with_context(|| format!("write {}", out_infrastructure.display()))?;
    fs::write(out_report, report_text)
        .with_context(|| format!("write {}", out_report.display()))?;
    Ok(SourceInfrastructureSummary {
        total_artifact_count: lane_counts.values().copied().sum(),
        lane_counts,
    })
}

#[derive(Default)]
struct ArtifactCounts {
    downloaded: usize,
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
    if minimum_met != (!working.is_empty() || !downloaded_paths.is_empty()) {
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
