//! Fetch implementation for planck. See planck.rs for best-fit constants and parsers.

use crate::fetcher::{
    DatasetProvider, FetchConfig, FetchError, download_with_fallbacks, extract_tar_gz,
};
use std::path::PathBuf;

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

simple_provider! {
    /// Planck full MCMC chains dataset provider.
    pub struct PlanckChainsProvider;
    name = "Planck 2018 MCMC Chains";
    output = "planck2018_chains.zip";
    urls = PLANCK_CHAIN_URLS;
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

simple_provider! {
    /// Planck 2018 base parameter constraints (best-fit TXT from IRSA).
    pub struct PlanckSummaryProvider;
    name = "Planck 2018 Summary";
    output = "planck2018_base_params.txt";
    urls = PLANCK_SUMMARY_URLS;
}
