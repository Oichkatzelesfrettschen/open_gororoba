//! Pantheon+ SH0ES supernova Ia dataset parser and fetcher.
//!
//! Pantheon+ contains 1701 light curves of 1550 spectroscopically confirmed
//! Type Ia supernovae. The distance modulus data is the primary product.
//!
//! Source: https://github.com/PantheonPlusSH0ES/DataRelease
//! Reference: Scolnic et al. (2022), ApJ 938, 113; Brout et al. (2022), ApJ 938, 110

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

/// A Type Ia supernova from Pantheon+.
#[derive(Debug, Clone)]
pub struct Supernova {
    /// CID (candidate ID).
    pub cid: String,
    /// CMB-frame redshift.
    pub z_cmb: f64,
    /// Heliocentric redshift.
    pub z_hel: f64,
    /// Corrected distance modulus (mag).
    pub mu: f64,
    /// Distance modulus error (mag).
    pub mu_err: f64,
    /// Host galaxy mass (log10 M_solar).
    pub host_logmass: f64,
    /// Fitted stretch parameter (x1).
    pub x1: f64,
    /// Fitted color parameter (c).
    pub c: f64,
    /// Survey identifier.
    pub idsurvey: i32,
    /// Whether this SN was used in SH0ES Cepheid calibration.
    pub is_calibrator: bool,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "nan" || s == "NaN" {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse Pantheon+ SH0ES distance file (.dat format).
///
/// The .dat file uses whitespace-delimited columns with a header line
/// starting with '#' or 'CID'.
pub fn parse_pantheon_dat(path: &Path) -> Result<Vec<Supernova>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    let mut sne = Vec::new();
    let mut header_indices: Option<Vec<(String, usize)>> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Header line detection
        if line.starts_with('#') || line.starts_with("CID") || line.starts_with("cid") {
            let clean = line.trim_start_matches('#').trim();
            let cols: Vec<(String, usize)> = clean
                .split_whitespace()
                .enumerate()
                .map(|(i, s)| (s.to_uppercase(), i))
                .collect();
            header_indices = Some(cols);
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 5 {
            continue;
        }

        let get_idx = |name: &str| -> Option<usize> {
            header_indices.as_ref().and_then(|hi| {
                hi.iter()
                    .find(|(n, _)| n == &name.to_uppercase())
                    .map(|(_, i)| *i)
            })
        };

        let get_field = |name: &str| -> f64 {
            get_idx(name)
                .and_then(|i| fields.get(i))
                .map(|s| parse_f64(s))
                .unwrap_or(f64::NAN)
        };

        let get_str = |name: &str| -> String {
            get_idx(name)
                .and_then(|i| fields.get(i))
                .unwrap_or(&"")
                .to_string()
        };

        let cid = get_str("CID");
        if cid.is_empty() && !fields.is_empty() {
            // Fallback: first column is CID
            let cid = fields[0].to_string();
            // Minimal positional parsing if no header found
            if header_indices.is_none() && fields.len() >= 8 {
                sne.push(Supernova {
                    cid,
                    z_cmb: parse_f64(fields[1]),
                    z_hel: parse_f64(fields[2]),
                    mu: parse_f64(fields[3]),
                    mu_err: parse_f64(fields[4]),
                    host_logmass: parse_f64(fields.get(5).unwrap_or(&"")),
                    x1: parse_f64(fields.get(6).unwrap_or(&"")),
                    c: parse_f64(fields.get(7).unwrap_or(&"")),
                    idsurvey: fields.get(8).and_then(|s| s.parse().ok()).unwrap_or(0),
                    is_calibrator: fields.get(9).map(|s| *s == "1").unwrap_or(false),
                });
            }
            continue;
        }

        // Pantheon+ SH0ES uses m_b_corr (corrected apparent magnitude).
        // For distance modulus fitting, use m_b_corr with analytic M_B
        // marginalization. Fall back to MU if m_b_corr is not present.
        let mu_val = {
            let mb = get_field("M_B_CORR");
            if mb.is_nan() {
                get_field("MU")
            } else {
                mb
            }
        };
        let mu_err_val = {
            let mbe = get_field("M_B_CORR_ERR_DIAG");
            if mbe.is_nan() {
                let v1 = get_field("MUERR");
                if v1.is_nan() {
                    get_field("MUERR_VPEC")
                } else {
                    v1
                }
            } else {
                mbe
            }
        };

        sne.push(Supernova {
            cid,
            z_cmb: {
                let v = get_field("ZCMB");
                if v.is_nan() {
                    get_field("ZHD")
                } else {
                    v
                }
            },
            z_hel: get_field("ZHEL"),
            mu: mu_val,
            mu_err: mu_err_val,
            host_logmass: get_field("HOST_LOGMASS"),
            x1: get_field("X1"),
            c: get_field("C"),
            idsurvey: get_field("IDSURVEY") as i32,
            is_calibrator: get_field("IS_CALIBRATOR") > 0.5,
        });
    }

    Ok(sne)
}

const PANTHEON_URLS: &[&str] = &[
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
];

/// Pantheon+ SH0ES dataset provider.
pub struct PantheonProvider;

impl DatasetProvider for PantheonProvider {
    fn name(&self) -> &str {
        "Pantheon+ SH0ES"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("PantheonPlusSH0ES.dat");
        download_with_fallbacks(self.name(), PANTHEON_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("PantheonPlusSH0ES.dat").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;

    fn write_temp(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_pantheon_synthetic_with_header() {
        let dat = "\
# CID ZCMB ZHEL M_B_CORR M_B_CORR_ERR_DIAG HOST_LOGMASS X1 C IDSURVEY IS_CALIBRATOR
SN2005eq 0.028 0.029 34.12 0.15 10.5 0.5 -0.02 4 0
SN2007af 0.005 0.006 31.80 0.12 11.2 -0.3 0.01 15 1
";
        let f = write_temp(dat);
        let sne = parse_pantheon_dat(f.path()).unwrap();
        assert_eq!(sne.len(), 2, "Should parse 2 supernovae");
        assert_eq!(sne[0].cid, "SN2005eq");
        assert!((sne[0].z_cmb - 0.028).abs() < 0.001);
        assert!(
            (sne[0].mu - 34.12).abs() < 0.01,
            "M_B_CORR should map to mu"
        );
        assert!((sne[0].mu_err - 0.15).abs() < 0.01);
        assert_eq!(sne[0].idsurvey, 4);
        assert!(!sne[0].is_calibrator);
        assert!(sne[1].is_calibrator);
    }

    #[test]
    fn test_parse_pantheon_mu_fallback() {
        // When M_B_CORR is absent, parser should fall back to MU column
        let dat = "\
CID ZCMB ZHEL MU MUERR HOST_LOGMASS X1 C IDSURVEY IS_CALIBRATOR
SN_FALLBACK 0.05 0.051 36.5 0.2 10.0 0.1 -0.01 1 0
";
        let f = write_temp(dat);
        let sne = parse_pantheon_dat(f.path()).unwrap();
        assert_eq!(sne.len(), 1);
        assert!(
            (sne[0].mu - 36.5).abs() < 0.01,
            "Should use MU when M_B_CORR absent"
        );
        assert!((sne[0].mu_err - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_parse_pantheon_zhd_fallback() {
        // When ZCMB is absent, parser should fall back to ZHD
        let dat = "\
CID ZHD ZHEL M_B_CORR M_B_CORR_ERR_DIAG HOST_LOGMASS X1 C IDSURVEY IS_CALIBRATOR
SN_ZHD 0.035 0.036 35.0 0.18 10.8 0.2 -0.03 2 0
";
        let f = write_temp(dat);
        let sne = parse_pantheon_dat(f.path()).unwrap();
        assert_eq!(sne.len(), 1);
        assert!(
            (sne[0].z_cmb - 0.035).abs() < 0.001,
            "Should use ZHD when ZCMB absent"
        );
    }

    #[test]
    fn test_parse_pantheon_skips_short_rows() {
        let dat = "\
CID ZCMB ZHEL MU MUERR HOST_LOGMASS X1 C IDSURVEY
SN_GOOD 0.05 0.051 36.5 0.2 10.0 0.1 -0.01 1
TOO_SHORT 0.05 0.051 36.5
";
        let f = write_temp(dat);
        let sne = parse_pantheon_dat(f.path()).unwrap();
        assert_eq!(sne.len(), 1, "Should skip rows with <5 fields");
    }

    #[test]
    fn test_parse_pantheon_empty_file() {
        let dat = "# CID ZCMB ZHEL MU MUERR\n";
        let f = write_temp(dat);
        let sne = parse_pantheon_dat(f.path()).unwrap();
        assert!(sne.is_empty());
    }

    #[test]
    fn test_parse_pantheon_if_available() {
        let path = Path::new("data/external/PantheonPlusSH0ES.dat");
        if !path.exists() {
            eprintln!("Skipping: Pantheon+ data not available");
            return;
        }

        let sne = parse_pantheon_dat(path).expect("Failed to parse Pantheon+ dat");
        assert!(
            sne.len() > 1000,
            "Pantheon+ should have >1000 SNe, got {}",
            sne.len()
        );

        for sn in &sne {
            assert!(
                sn.z_cmb > 0.0 || sn.z_cmb.is_nan(),
                "z_cmb should be positive: {}",
                sn.cid
            );
        }
    }
}
