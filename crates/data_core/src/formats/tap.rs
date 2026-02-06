//! TAP (Table Access Protocol) query builder.
//!
//! TAP is the IVOA standard for querying astronomical databases using ADQL
//! (Astronomical Data Query Language). This module provides a minimal sync
//! query interface for TAP endpoints at ESA, CDS, NED, and similar archives.
//!
//! Reference: IVOA TAP 1.1 (2019), https://www.ivoa.net/documents/TAP/

use crate::fetcher::{download_to_string, FetchError};

/// Build a TAP synchronous query URL.
///
/// Encodes the ADQL query and requested format into the TAP sync endpoint URL.
/// Common formats: "csv", "votable", "json", "tsv".
pub fn tap_sync_url(base_url: &str, adql: &str, format: &str) -> String {
    let encoded_query = adql.replace([' ', '\n', '\t'], "+");
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
