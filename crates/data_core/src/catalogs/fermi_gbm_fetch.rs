//! Fetch implementation for fermi_gbm. See fermi_gbm.rs for record types and parsers.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_heasarc_csv};
use std::path::PathBuf;

/// W3Browse batch query URL for the Fermi GBM burst catalog.
///
/// HEASARC TAP only supports VOTable format (not CSV), so we use the W3Browse
/// batch endpoint which returns pipe-delimited text. The `download_heasarc_csv`
/// function converts pipe-delimited output to standard CSV.
///
/// `ResultMax=0` returns all rows; `displaymode=BatchDisplay` returns
/// pipe-delimited; `Fields=` selects specific columns.
///
/// Correct column names verified against HEASARC fermigbrst table (2026-02):
/// - `flux_64` / `flux_1024`: peak photon flux (ph/cm^2/s) on 64ms / 1024ms
/// - `flnc_best_fitting_model`: time-integrated spectral model
/// - `pflx_best_fitting_model`: peak-flux spectral model
const FERMI_GBM_URL: &str = "\
https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?\
tablehead=name%3Dfermigbrst&\
Action=Query&\
Coordinates=Equatorial&\
Equinox=2000&\
Radius=Default&\
NR=&\
GIFsize=0&\
Fields=name%2Ctrigger_time%2Cra%2Cdec%2Ct90%2Ct50%2Cfluence%2Cflux_64%2Cflux_1024%2Cflnc_best_fitting_model%2Cpflx_best_fitting_model&\
ResultMax=0&\
displaymode=BatchDisplay";

/// Fermi GBM burst catalog dataset provider.
pub struct FermiGbmProvider;

impl DatasetProvider for FermiGbmProvider {
    fn name(&self) -> &str {
        "Fermi GBM Burst Catalog"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("fermi_gbm_grbs.csv");
        if config.skip_existing && output.exists() {
            log::info!("{} already cached at {}", self.name(), output.display());
            return Ok(output);
        }
        log::info!("Downloading {} from HEASARC W3Browse...", self.name());
        let bytes = download_heasarc_csv(FERMI_GBM_URL, &output)?;
        log::info!("Saved {} bytes to {}", bytes, output.display());
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("fermi_gbm_grbs.csv").exists()
    }
}
