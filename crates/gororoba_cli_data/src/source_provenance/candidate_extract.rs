//! Extract `CandidateRecord` entries from TOML, BibTeX, and text
//! source files.
//!
//! Entry point: `extract_candidates_from_source_file` dispatches on
//! file extension to one of three format-specific extractors:
//!   * `extract_candidates_from_toml_node` -- recursive TOML walker
//!     that classifies each table key via `classify_toml_field_key`
//!     into Url / Doi / Path / Note / Skip
//!   * `extract_candidates_from_bib_file` -- BibTeX, using
//!     `bib_entry_re` for entry boundaries and `extract_bib_field`
//!     (brace + quoted forms) for individual fields
//!   * `extract_candidates_from_text_file` -- Markdown / plain-text /
//!     reStructuredText line-by-line URL+DOI mining with
//!     `clean_line_title` to strip URL markup
//!
//! `TomlFieldKind` and `classify_toml_field_key` were lifted from
//! `extract_candidates_from_toml_node` to keep its cyclomatic
//! complexity in check.
//!
//! All items `pub(super)`. The submodule accesses parent's private
//! `CandidateRecord`, `TITLE_KEYS`, `CITATION_KEYS`, `ID_KEYS`
//! directly (child modules see private parent items). Wires in
//! `normalize_identity_hint`, `extract_strings`, `extract_urls`,
//! `extract_local_paths`, `dedupe` from the parent, plus
//! `doi_helpers::{doi_to_url, extract_dois}`,
//! `file_io::{load_toml_value, read_text_lossy}`,
//! `reference_predicates::looks_like_reference_url`,
//! `text_helpers::{bib_entry_re, url_inline_re}`,
//! `url_helpers::find_urls`.

use std::path::Path;

use anyhow::Result;
use regex::Regex;
use toml::Value;

use super::doi_helpers::{doi_to_url, extract_dois};
use super::file_io::{load_toml_value, read_text_lossy};
use super::reference_predicates::looks_like_reference_url;
use super::text_helpers::{bib_entry_re, url_inline_re};
use super::url_helpers::find_urls;
use super::{
    CITATION_KEYS, CandidateRecord, ID_KEYS, TITLE_KEYS, dedupe, extract_local_paths,
    extract_strings, extract_urls, normalize_identity_hint,
};

pub(super) fn pick_first_str(table: &toml::map::Map<String, Value>, keys: &[&str]) -> String {
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

/// Classify a lowercase TOML key into the field category it contributes to.
enum TomlFieldKind {
    Url,
    Doi,
    Path,
    Note,
    Skip,
}

fn classify_toml_field_key(lower: &str) -> TomlFieldKind {
    if lower.contains("url")
        || lower.contains("link")
        || lower.contains("mirror")
        || lower.contains("href")
    {
        TomlFieldKind::Url
    } else if lower.contains("doi") {
        TomlFieldKind::Doi
    } else if lower.contains("path")
        || lower.ends_with("_file")
        || lower.ends_with("_files")
        || lower == "files"
    {
        TomlFieldKind::Path
    } else if matches!(
        lower,
        "status" | "note" | "notes" | "reason" | "manual_intervention_reason"
    ) {
        TomlFieldKind::Note
    } else {
        TomlFieldKind::Skip
    }
}

pub(super) fn extract_candidates_from_toml_node(
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
                match classify_toml_field_key(&lower) {
                    TomlFieldKind::Url => urls.extend(extract_urls(value)),
                    TomlFieldKind::Doi => dois.extend(extract_dois(value)),
                    TomlFieldKind::Path => {
                        local_paths.extend(extract_local_paths(value, repo_root))
                    }
                    TomlFieldKind::Note => notes.extend(extract_strings(value)),
                    TomlFieldKind::Skip => {}
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

pub(super) fn extract_candidates_from_bib_file(
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

pub(super) fn extract_candidates_from_text_file(
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

pub(super) fn extract_candidates_from_source_file(
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
