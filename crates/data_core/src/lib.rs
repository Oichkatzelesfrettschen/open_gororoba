//! data_core: Dataset acquisition, validation, and caching for cosmological catalogs.
//!
//! Provides unified infrastructure for downloading, validating, and parsing
//! astrophysical and geophysical datasets used in ultrametric structure analysis.
//!
//! # Architecture
//!
//! - `fetcher`: Shared HTTP download, SHA-256 checksum, and disk caching.
//! - `catalogs`: Dataset-specific parsers with typed record structs.
//! - `formats`: Parsers for non-CSV data formats (GFC, SHC, Pantheon .dat).
//!
//! # Usage
//!
//! ```no_run
//! use data_core::fetcher::{FetchConfig, DatasetProvider};
//! use data_core::catalogs::gwtc::Gwtc3Provider;
//!
//! let config = FetchConfig::default();
//! let provider = Gwtc3Provider;
//! let path = provider.fetch(&config).unwrap();
//! ```

pub mod fetcher;
pub mod catalogs;
pub mod formats;
pub mod geophysical;

pub use fetcher::{FetchConfig, FetchError, DatasetProvider};
pub use fetcher::{download_to_file, download_to_string, compute_sha256};

pub use catalogs::chime::{FrbEvent, parse_chime_csv, extract_repeaters};
pub use catalogs::gwtc::{GwEvent, parse_gwtc3_csv};
pub use catalogs::atnf::{Pulsar, parse_atnf_csv};
pub use catalogs::mcgill::{Magnetar, parse_mcgill_csv};
pub use catalogs::fermi_gbm::{GrbEvent, parse_fermi_gbm_csv};
pub use catalogs::pantheon::{Supernova, parse_pantheon_dat};
pub use catalogs::desi_bao::{BaoMeasurement, desi_dr1_bao};
pub use catalogs::planck::bestfit as planck2018;
pub use catalogs::nanograv::{FreeSpectrumPoint, parse_nanograv_free_spectrum};
pub use catalogs::sdss::{SdssQuasar, parse_sdss_quasar_csv};
pub use catalogs::gaia::{GaiaSource, parse_gaia_csv};
pub use catalogs::tsi::{TsiMeasurement, parse_tsi_csv};
