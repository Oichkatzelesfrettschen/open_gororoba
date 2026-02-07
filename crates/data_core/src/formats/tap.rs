//! TAP (Table Access Protocol) query builder.
//!
//! TAP is the IVOA standard for querying astronomical databases using ADQL
//! (Astronomical Data Query Language). This module provides a minimal sync
//! query interface for TAP endpoints at ESA, CDS, NED, and similar archives.
//!
//! Reference: IVOA TAP 1.1 (2019), https://www.ivoa.net/documents/TAP/

use crate::fetcher::{download_to_string, FetchError};

/// Percent-encode a query string value for use in a URL parameter.
///
/// Encodes all characters except unreserved ones (RFC 3986):
/// ALPHA / DIGIT / "-" / "." / "_" / "~"
/// Space is encoded as `+` (application/x-www-form-urlencoded).
pub fn percent_encode_query(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char);
            }
            b' ' | b'\t' | b'\n' | b'\r' => {
                out.push('+');
            }
            _ => {
                out.push('%');
                out.push(char::from(HEX[(b >> 4) as usize]));
                out.push(char::from(HEX[(b & 0x0f) as usize]));
            }
        }
    }
    out
}

const HEX: &[u8; 16] = b"0123456789ABCDEF";

/// Build a TAP synchronous query URL.
///
/// Encodes the ADQL query and requested format into the TAP sync endpoint URL.
/// Common formats: "csv", "votable", "json", "tsv".
pub fn tap_sync_url(base_url: &str, adql: &str, format: &str) -> String {
    let encoded_query = percent_encode_query(adql);
    format!(
        "{}/sync?REQUEST=doQuery&LANG=ADQL&FORMAT={}&QUERY={}",
        base_url.trim_end_matches('/'),
        format,
        encoded_query
    )
}

/// Execute a TAP synchronous query and return the result as a string.
pub fn tap_query(base_url: &str, adql: &str, format: &str) -> Result<String, FetchError> {
    let url = tap_sync_url(base_url, adql, format);
    download_to_string(&url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tap_sync_url_construction() {
        let url = tap_sync_url(
            "https://gea.esac.esa.int/tap-server/tap",
            "SELECT TOP 10 source_id FROM gaiadr3.gaia_source",
            "csv",
        );
        assert!(url.contains("REQUEST=doQuery"));
        assert!(url.contains("LANG=ADQL"));
        assert!(url.contains("FORMAT=csv"));
        assert!(url.contains("SELECT+TOP+10"));
    }
}
