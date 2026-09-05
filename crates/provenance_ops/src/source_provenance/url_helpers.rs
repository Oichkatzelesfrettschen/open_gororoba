//! URL canonicalization and validation helpers.
//!
//! Pipeline:
//!   * `strip_url_wrappers`          -- strip surrounding punctuation
//!   * `rewrite_arxiv_typo_prefix`   -- fix arxiv.org.abs/ typos
//!   * `apply_host_specific_rewrites` -- per-host normalizations
//!     (http -> https for known publishers, dx.doi.org -> doi.org,
//!     arxiv /pdf/ extension, sciencedirect /pdfft -> /pdf, etc.)
//!   * `filter_tracking_query_params` -- drop utm_* and other
//!     well-known tracking params
//!   * `normalize_url`               -- top-level canonicalizer
//!   * `is_non_reference_service_url` -- archive.org service path
//!     blocklist
//!   * `find_urls`                   -- extract canonicalized URLs
//!     from free text
//!
//! All items `pub(super)`. Depends only on `url::Url` and
//! `super::text_helpers::{url_re, url_inline_re}`.

use url::Url;

use super::text_helpers::{url_inline_re, url_re};

pub(super) fn strip_url_wrappers(value: &mut String) {
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

pub(super) fn rewrite_arxiv_typo_prefix(trimmed: &str) -> Option<String> {
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

pub(super) fn apply_host_specific_rewrites(parsed: &mut Url, host: &str) {
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

pub(super) fn filter_tracking_query_params(parsed: &mut Url) {
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

pub(super) fn normalize_url(url: &str) -> String {
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

pub(super) fn is_non_reference_service_url(parsed: &Url) -> bool {
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

pub(super) fn find_urls(text: &str) -> Vec<String> {
    url_inline_re()
        .find_iter(text)
        .map(|m| normalize_url(m.as_str()))
        .filter(|value| url_re().is_match(value))
        .collect()
}
