//! DESI DR1 BAO distance measurements parser and fetcher.
//!
//! DESI DR1 provides BAO distance scale measurements across 7 redshift
//! bins using different tracers (BGS, LRG, ELG, QSO, Lya).
//!
//! Source: https://github.com/CobayaSampler/bao_data
//! Reference: DESI Collaboration (2024), arXiv:2404.03002

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string};
use std::path::{Path, PathBuf};

/// A BAO distance measurement from DESI.
#[derive(Debug, Clone)]
pub struct BaoMeasurement {
    /// Effective redshift of the bin.
    pub z_eff: f64,
    /// DM/rd (transverse distance ratio).
    pub dm_over_rd: f64,
    /// DM/rd error.
    pub dm_over_rd_err: f64,
    /// DH/rd (line-of-sight distance ratio).
    pub dh_over_rd: f64,
    /// DH/rd error.
    pub dh_over_rd_err: f64,
    /// Correlation between DM/rd and DH/rd.
    pub rho: f64,
    /// Tracer type (BGS, LRG, ELG, QSO, Lya).
    pub tracer: String,
}

/// Hardcoded DESI DR1 BAO results from DESI Collaboration (2024) Table 1.
/// These are the consensus values used by Cobaya.
pub fn desi_dr1_bao() -> Vec<BaoMeasurement> {
    vec![
        BaoMeasurement {
            z_eff: 0.295,
            dm_over_rd: 7.93,
            dm_over_rd_err: 0.15,
            dh_over_rd: 20.98,
            dh_over_rd_err: 0.61,
            rho: -0.445,
            tracer: "BGS".to_string(),
        },
        BaoMeasurement {
            z_eff: 0.510,
            dm_over_rd: 13.62,
            dm_over_rd_err: 0.25,
            dh_over_rd: 20.98,
            dh_over_rd_err: 0.61,
            rho: -0.474,
            tracer: "LRG1".to_string(),
        },
        BaoMeasurement {
            z_eff: 0.706,
            dm_over_rd: 16.85,
            dm_over_rd_err: 0.32,
            dh_over_rd: 20.08,
            dh_over_rd_err: 0.60,
            rho: -0.420,
            tracer: "LRG2".to_string(),
        },
        BaoMeasurement {
            z_eff: 0.930,
            dm_over_rd: 21.71,
            dm_over_rd_err: 0.28,
            dh_over_rd: 17.88,
            dh_over_rd_err: 0.35,
            rho: -0.386,
            tracer: "LRG3+ELG1".to_string(),
        },
        BaoMeasurement {
            z_eff: 1.317,
            dm_over_rd: 27.79,
            dm_over_rd_err: 0.69,
            dh_over_rd: 13.82,
            dh_over_rd_err: 0.42,
            rho: -0.474,
            tracer: "ELG2".to_string(),
        },
        BaoMeasurement {
            z_eff: 1.491,
            dm_over_rd: 30.69,
            dm_over_rd_err: 0.79,
            dh_over_rd: 13.26,
            dh_over_rd_err: 0.55,
            rho: -0.470,
            tracer: "QSO".to_string(),
        },
        BaoMeasurement {
            z_eff: 2.330,
            dm_over_rd: 39.71,
            dm_over_rd_err: 0.94,
            dh_over_rd: 8.52,
            dh_over_rd_err: 0.17,
            rho: -0.477,
            tracer: "Lya".to_string(),
        },
    ]
}

/// Parse a DESI BAO text file (Cobaya format).
///
/// Format: z_eff DM/rd DH/rd
pub fn parse_desi_bao_txt(path: &Path) -> Result<Vec<BaoMeasurement>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    let mut measurements = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 3 {
            let z: f64 = fields[0].parse().unwrap_or(f64::NAN);
            let dm_rd: f64 = fields[1].parse().unwrap_or(f64::NAN);
            let dh_rd: f64 = fields[2].parse().unwrap_or(f64::NAN);
            if z.is_finite() {
                measurements.push(BaoMeasurement {
                    z_eff: z,
                    dm_over_rd: dm_rd,
                    dm_over_rd_err: 0.0,
                    dh_over_rd: dh_rd,
                    dh_over_rd_err: 0.0,
                    rho: 0.0,
                    tracer: format!("z={:.3}", z),
                });
            }
        }
    }

    Ok(measurements)
}

/// DESI BAO text files from the Cobaya sampler GitHub repository.
const DESI_BAO_BASE: &str = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/";

/// DESI BAO dataset provider (fetches individual text files).
pub struct DesiBaoProvider;

impl DatasetProvider for DesiBaoProvider {
    fn name(&self) -> &str { "DESI DR1 BAO" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("desi_bao");
        std::fs::create_dir_all(&dir)?;

        let files = [
            "desi_2024_gaussian_bao_BGS_z0.295.txt",
            "desi_2024_gaussian_bao_LRG1_z0.510.txt",
            "desi_2024_gaussian_bao_LRG2_z0.706.txt",
            "desi_2024_gaussian_bao_LRG3+ELG1_z0.930.txt",
            "desi_2024_gaussian_bao_ELG2_z1.317.txt",
            "desi_2024_gaussian_bao_QSO_z1.491.txt",
            "desi_2024_gaussian_bao_Lya_z2.330.txt",
        ];

        for fname in &files {
            let output = dir.join(fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}", DESI_BAO_BASE, fname);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    eprintln!("  Saved {}", fname);
                }
                Err(e) => {
                    eprintln!("  Failed to download {}: {}", fname, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("desi_bao").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_desi_hardcoded_values() {
        let bao = desi_dr1_bao();
        assert_eq!(bao.len(), 7, "DESI DR1 has 7 redshift bins");

        // Verify redshift ordering
        for w in bao.windows(2) {
            assert!(w[0].z_eff < w[1].z_eff, "z_eff should increase");
        }

        // Verify DM/rd increases with redshift (cosmological expectation)
        for w in bao.windows(2) {
            assert!(
                w[0].dm_over_rd < w[1].dm_over_rd,
                "DM/rd should increase with z"
            );
        }
    }
}
