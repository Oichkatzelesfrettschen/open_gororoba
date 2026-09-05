//! Syntactic remote metadata admission, independent of retrieval outcomes.

pub(super) fn remote_identity_is_usable(canonical_url: &str, sha256: &str) -> bool {
    url::Url::parse(canonical_url)
        .is_ok_and(|url| matches!(url.scheme(), "https" | "http") && url.host_str().is_some())
        && sha256.len() == 64
        && sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
}
