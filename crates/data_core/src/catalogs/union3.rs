//! Union3 legacy supernova likelihood chains.
//!
//! Union3 is a legacy SNe Ia compilation used in modern BAO+SNe joint analyses.
//! This provider caches a published DESI Y3 chain representative for Union3.
//!
//! Source: DESI Y3 BAO cosmology data release
//! https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

/// One row from a whitespace-delimited chain file.
#[derive(Debug, Clone)]
pub struct Union3ChainRow {
    pub weight: f64,
    pub minus_log_posterior: f64,
}

fn parse_f64(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse a DESI/Cobaya chain text file for lightweight integrity checks.
pub fn parse_union3_chain(path: &Path) -> Result<Vec<Union3ChainRow>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    let mut rows = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let cols: Vec<&str> = trimmed.split_whitespace().collect();
        if cols.len() < 2 {
            continue;
        }
        rows.push(Union3ChainRow {
            weight: parse_f64(cols[0]),
            minus_log_posterior: parse_f64(cols[1]),
        });
    }
    Ok(rows)
}

const UNION3_URLS: &[&str] = &[
    "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base/union3/chain.1.txt",
    "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base/union3/chain.updated.yaml",
];

/// Union3 dataset provider.
pub struct Union3Provider;

impl DatasetProvider for Union3Provider {
    fn name(&self) -> &str {
        "Union3 Legacy SN Ia"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("union3_chain_1.txt");
        download_with_fallbacks(self.name(), UNION3_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("union3_chain_1.txt").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_parse_union3_if_available() {
        let path = Path::new("data/external/union3_chain_1.txt");
        if !path.exists() {
            eprintln!("Skipping: Union3 chain not available");
            return;
        }
        let rows = parse_union3_chain(path).expect("failed to parse Union3 chain");
        assert!(!rows.is_empty(), "Union3 chain should not be empty");
    }
}
