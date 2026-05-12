//! Predicates classifying URLs and paths for the source-provenance
//! pipeline.
//!
//! Functions:
//!   * `looks_like_reference_url`  -- accept URLs from known
//!     reference hosts, .arxiv.org / .scispace.com subdomains, or
//!     anything with a /pdf path component
//!   * `is_citation_locator_url`   -- per-host rules for "this is an
//!     abstract/landing/locator page, not a content URL"
//!   * `key_is_citation_locator`   -- doi: keys always count; url:
//!     keys defer to `is_citation_locator_url`
//!   * `is_artifact_local_path`    -- match against
//!     `ARTIFACT_LOCAL_PREFIXES`
//!
//! All items `pub(super)`. Reads `REFERENCE_HOST_HINTS` and
//! `ARTIFACT_LOCAL_PREFIXES` from the parent (child modules see
//! private parent items) and uses
//! `url_helpers::is_non_reference_service_url` plus
//! `text_helpers::url_re`.

use url::Url;

use super::text_helpers::url_re;
use super::url_helpers::is_non_reference_service_url;
use super::{ARTIFACT_LOCAL_PREFIXES, REFERENCE_HOST_HINTS};

pub(super) fn looks_like_reference_url(url: &str) -> bool {
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

pub(super) fn is_citation_locator_url(url: &str) -> bool {
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

pub(super) fn key_is_citation_locator(key: &str) -> bool {
    if key.to_ascii_lowercase().starts_with("doi:") {
        return true;
    }
    if let Some(url) = key.strip_prefix("url:") {
        return is_citation_locator_url(url);
    }
    false
}

pub(super) fn is_artifact_local_path(path: &str) -> bool {
    ARTIFACT_LOCAL_PREFIXES
        .iter()
        .any(|prefix| path.trim().starts_with(prefix))
}
