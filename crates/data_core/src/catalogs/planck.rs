//! Planck 2018 cosmological parameters and MCMC chain fetcher.
//!
//! Provides both hardcoded best-fit values from Planck 2018 VI Table 2
//! and a fetcher for the full MCMC posterior chains from the Planck Legacy
//! Archive.
//!
//! Best-fit: Planck Collaboration VI (2020), A&A 641, A6
//! Chains: https://pla.esac.esa.int/

use crate::fetcher::{
    download_with_fallbacks, extract_tar_gz, DatasetProvider, FetchConfig, FetchError,
};
use std::path::PathBuf;

/// Planck 2018 TT,TE,EE+lowE+lensing best-fit parameters.
/// Source: Planck 2018 VI, Table 2, Column 6 (TT,TE,EE+lowE+lensing).
pub mod bestfit {
    /// Hubble constant (km/s/Mpc).
    pub const H0: f64 = 67.36;
    /// Total matter density parameter.
    pub const OMEGA_M: f64 = 0.3153;
    /// Baryon density parameter.
    pub const OMEGA_B: f64 = 0.0493;
    /// Dark energy density parameter.
    pub const OMEGA_LAMBDA: f64 = 0.6847;
    /// CDM density parameter.
    pub const OMEGA_CDM: f64 = 0.2607;
    /// Baryon density * h^2.
    pub const OMEGA_B_H2: f64 = 0.02237;
    /// CDM density * h^2.
    pub const OMEGA_CDM_H2: f64 = 0.1200;
    /// RMS density fluctuation at 8 Mpc/h.
    pub const SIGMA_8: f64 = 0.8111;
    /// Scalar spectral index.
    pub const N_S: f64 = 0.9649;
    /// Optical depth to reionization.
    pub const TAU: f64 = 0.0544;
    /// Scalar amplitude (10^-9).
    pub const LN_10_10_A_S: f64 = 3.044;
    /// CMB temperature (K).
    pub const T_CMB: f64 = 2.7255;
    /// Sound horizon at drag epoch (Mpc).
    pub const R_DRAG: f64 = 147.09;
    /// Age of universe (Gyr).
    pub const AGE_GYR: f64 = 13.797;
    /// Redshift of matter-radiation equality.
    pub const Z_EQ: f64 = 3387.0;
    /// Redshift at last scattering.
    pub const Z_STAR: f64 = 1089.92;
    /// Redshift at reionization.
    pub const Z_RE: f64 = 7.67;
}

/// WMAP 9-year best-fit parameters for comparison.
/// Source: Hinshaw et al. (2013), ApJS 208, 19 (Table 4, WMAP-only column).
pub mod wmap9 {
    pub const H0: f64 = 69.32;
    pub const OMEGA_M: f64 = 0.2880;
    pub const OMEGA_B: f64 = 0.0472;
    pub const OMEGA_LAMBDA: f64 = 0.7120;
    pub const SIGMA_8: f64 = 0.820;
    pub const N_S: f64 = 0.9608;
    pub const TAU: f64 = 0.0890;
    pub const AGE_GYR: f64 = 13.772;
}

/// Planck 2018 full MCMC chain URLs.
///
/// The IRSA mirror hosts the closest available chain set (TTTEEE+lowl+lowE,
/// without lensing, R3.00). The PLA endpoint that hosted the
/// TTTEEE+lowl+lowE+lensing R3.01 set returns 404 as of 2026-02.
const PLANCK_CHAIN_URLS: &[&str] = &[
    // IRSA mirror: TTTEEE+lowl+lowE (without lensing, R3.00)
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip",
];

/// WMAP 9-year full MCMC chain URLs.
///
/// WMAP 9-year MCMC chains from LAMBDA. ~100 MB.
const WMAP9_CHAIN_URLS: &[&str] =
    &["https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_wmap9_chains_v5.tar.gz"];

/// Planck full MCMC chains dataset provider.
pub struct PlanckChainsProvider;

impl DatasetProvider for PlanckChainsProvider {
    fn name(&self) -> &str {
        "Planck 2018 MCMC Chains"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("planck2018_chains.zip");
        download_with_fallbacks(
            self.name(),
            PLANCK_CHAIN_URLS,
            &output,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("planck2018_chains.zip").exists()
    }
}

/// WMAP 9-year full MCMC chains dataset provider.
pub struct Wmap9ChainsProvider;

impl DatasetProvider for Wmap9ChainsProvider {
    fn name(&self) -> &str {
        "WMAP 9yr MCMC Chains"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("wmap9_chains.tar.gz");
        download_with_fallbacks(self.name(), WMAP9_CHAIN_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("wmap9_chains.tar.gz").exists()
    }
}

impl Wmap9ChainsProvider {
    /// Extract the downloaded tar.gz archive to a directory.
    pub fn extract(&self, config: &FetchConfig) -> Result<Vec<PathBuf>, FetchError> {
        let archive = config.output_dir.join("wmap9_chains.tar.gz");
        let output_dir = config.output_dir.join("wmap9_chains");
        extract_tar_gz(&archive, &output_dir)
    }
}

/// Planck base parameters (best-fit values) from IRSA.
///
/// The PLA ZIP endpoint (which contained getdist .margestats files) returns
/// 404 as of 2026-02. This IRSA TXT file contains the maximum-likelihood
/// parameters from base_plikHM_TTTEEE_lowl_lowE_lensing.
const PLANCK_SUMMARY_URLS: &[&str] = &[
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt",
];

/// Planck 2018 base parameter constraints (best-fit TXT from IRSA).
pub struct PlanckSummaryProvider;

impl DatasetProvider for PlanckSummaryProvider {
    fn name(&self) -> &str {
        "Planck 2018 Summary"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("planck2018_base_params.txt");
        download_with_fallbacks(
            self.name(),
            PLANCK_SUMMARY_URLS,
            &output,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("planck2018_base_params.txt")
            .exists()
    }
}

/// Parse a getdist .margestats file into a map of parameter name -> (mean, std).
///
/// The .margestats file has a header line followed by rows:
/// `param_name  mean  sddev  ...`
pub fn parse_margestats(content: &str) -> Vec<(String, f64, f64)> {
    let mut results = Vec::new();
    let mut header_seen = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Skip header line
        if !header_seen {
            header_seen = true;
            continue;
        }

        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() >= 3 {
            let name = fields[0].to_string();
            let mean = fields[1].parse::<f64>().unwrap_or(f64::NAN);
            let std = fields[2].parse::<f64>().unwrap_or(f64::NAN);
            results.push((name, mean, std));
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::bestfit;

    #[test]
    fn test_planck_flat_universe() {
        // Omega_m + Omega_Lambda should sum to ~1 for flat universe
        let total = bestfit::OMEGA_M + bestfit::OMEGA_LAMBDA;
        assert!(
            (total - 1.0).abs() < 0.01,
            "Planck flatness: Omega_m + Omega_L = {}",
            total
        );
    }

    #[test]
    fn test_planck_baryon_consistency() {
        // Omega_b * h^2 should match OMEGA_B_H2
        let h = bestfit::H0 / 100.0;
        let omega_bh2 = bestfit::OMEGA_B * h * h;
        assert!(
            (omega_bh2 - bestfit::OMEGA_B_H2).abs() < 0.001,
            "Baryon density: {} vs {}",
            omega_bh2,
            bestfit::OMEGA_B_H2
        );
    }
}
