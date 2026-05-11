//! Pure text and regex helpers shared across the source-provenance
//! pipeline.
//!
//! Includes:
//! - Cached `Regex` factories (`url_re`, `url_inline_re`, `doi_re`,
//!   `bib_entry_re`) backed by `OnceLock`.
//! - ASCII-only sanitization (`ascii_sanitize`, `assert_ascii`).
//! - TOML emission helpers (`escape_toml`, `render_list`).
//! - Lower-snake slug generator (`slug`).
//!
//! Every output is ASCII by construction: the policy enforced at the
//! crate boundary (`assert_ascii`) is also the postcondition of
//! `escape_toml` and `slug`.

use std::sync::OnceLock;

use anyhow::{Result, bail};
use regex::Regex;

pub(super) fn url_re() -> &'static Regex {
    static URL_RE: OnceLock<Regex> = OnceLock::new();
    URL_RE.get_or_init(|| Regex::new(r"(?i)^https?://").expect("valid URL regex"))
}

pub(super) fn url_inline_re() -> &'static Regex {
    static URL_INLINE_RE: OnceLock<Regex> = OnceLock::new();
    URL_INLINE_RE
        .get_or_init(|| Regex::new(r#"(?i)https?://[^\s<>()"']+"#).expect("valid inline URL regex"))
}

pub(super) fn doi_re() -> &'static Regex {
    static DOI_RE: OnceLock<Regex> = OnceLock::new();
    DOI_RE.get_or_init(|| {
        Regex::new(r"(?i)10\.\d{4,9}/[-._;()/:A-Za-z0-9]+").expect("valid DOI regex")
    })
}

pub(super) fn bib_entry_re() -> &'static Regex {
    static BIB_ENTRY_RE: OnceLock<Regex> = OnceLock::new();
    BIB_ENTRY_RE.get_or_init(|| {
        Regex::new(r"(?s)@(?P<etype>[A-Za-z]+)\s*\{\s*(?P<key>[^,]+)\s*,(?P<body>.*?)\n\}\s*")
            .expect("valid BibTeX regex")
    })
}

pub(super) fn ascii_sanitize(text: &str) -> String {
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

pub(super) fn escape_toml(text: &str) -> String {
    let sanitized = ascii_sanitize(text);
    let escaped = sanitized
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

pub(super) fn render_list(values: &[String]) -> String {
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

pub(super) fn assert_ascii(text: &str, context: &str) -> Result<()> {
    if !text.is_ascii() {
        bail!("non-ASCII output in {context}");
    }
    Ok(())
}

pub(super) fn slug(text: &str) -> String {
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
