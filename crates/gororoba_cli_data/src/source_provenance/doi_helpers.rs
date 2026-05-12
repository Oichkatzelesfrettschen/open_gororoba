//! DOI parsing and identity helpers for the source-provenance pipeline.
//!
//! Functions:
//!   * `normalize_doi`           -- canonicalize a DOI string (strip
//!     URL prefix, trailing punctuation, /pdf, /fulltext, .pdf suffix)
//!   * `extract_dois`            -- mine DOIs from a TOML Value
//!   * `doi_to_url` / `doi_from_url` -- canonical URL round-trip
//!   * `extract_dois_from_urls`  -- inverse-map a URL list to DOIs
//!
//! All items are `pub(super)`. Depends on parent's `extract_strings`,
//! `dedupe`, and on `super::text_helpers::doi_re`.

use toml::Value;
use url::Url;

use super::text_helpers::doi_re;
use super::{dedupe, extract_strings};

pub(super) fn normalize_doi(doi: &str) -> String {
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

pub(super) fn extract_dois(value: &Value) -> Vec<String> {
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

pub(super) fn doi_to_url(doi: &str) -> String {
    format!("https://doi.org/{doi}")
}

pub(super) fn doi_from_url(url: &str) -> String {
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

pub(super) fn extract_dois_from_urls(urls: &[String]) -> Vec<String> {
    dedupe(
        urls.iter()
            .map(|url| doi_from_url(url))
            .filter(|doi| !doi.is_empty())
            .collect(),
    )
}
