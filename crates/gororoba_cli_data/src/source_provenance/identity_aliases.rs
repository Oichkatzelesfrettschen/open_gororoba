//! Reference identity + alias-expansion helpers.
//!
//! Functions:
//!   * `arxiv_equivalent_urls`    -- compute the abs/pdf pair (and
//!     version-stripped pair) for an arxiv.org URL
//!   * `strip_arxiv_version`      -- drop a trailing v\d+ from an
//!     arxiv identifier
//!   * `core_id_from_url`         -- extract the core.ac.uk record ID
//!   * `mdpi_path_looks_article`  -- recognize an MDPI ISSN-rooted
//!     article path (4 segments)
//!   * `cambridge_content_id`     -- extract the cambridge.org content
//!     identifier
//!   * `canonical_identity_url`   -- pick the canonical URL from a
//!     candidate list (cambridge content_id first, then MDPI PDF
//!     normalized form, then the first URL)
//!   * `expand_reference_aliases` -- compute every URL alias for a
//!     reference (core variants, MDPI PDF/no-PDF, wolframscience
//!     /presentation typo fix, dr.lib.iastate bitstream variants,
//!     actaphys path-style fix)
//!
//! All items `pub(super)`. Depends on `super::dedupe` and
//! `super::url_helpers::normalize_url`.

use url::Url;

use super::dedupe;
use super::url_helpers::normalize_url;

pub(super) fn arxiv_equivalent_urls(url: &str) -> Vec<String> {
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

pub(super) fn strip_arxiv_version(ident: &str) -> String {
    let Some((prefix, suffix)) = ident.rsplit_once('v') else {
        return ident.to_string();
    };
    if !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()) {
        prefix.to_string()
    } else {
        ident.to_string()
    }
}

pub(super) fn core_id_from_url(url: &str) -> Option<String> {
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

pub(super) fn mdpi_path_looks_article(parts: &[&str]) -> bool {
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

pub(super) fn cambridge_content_id(url: &str) -> Option<String> {
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

pub(super) fn canonical_identity_url(urls: &[String]) -> Option<String> {
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

pub(super) fn expand_reference_aliases(url: &str) -> Vec<String> {
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
