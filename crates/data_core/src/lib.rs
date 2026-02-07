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

pub mod catalogs;
pub mod fetcher;
pub mod formats;
pub mod geophysical;

pub use fetcher::{compute_sha256, download_to_file, download_to_string};
pub use fetcher::{DatasetProvider, FetchConfig, FetchError};

pub use catalogs::atnf::{parse_atnf_csv, Pulsar};
pub use catalogs::chime::{extract_repeaters, parse_chime_csv, FrbEvent};
pub use catalogs::desi_bao::{desi_dr1_bao, BaoMeasurement};
pub use catalogs::fermi_gbm::{parse_fermi_gbm_csv, GrbEvent};
pub use catalogs::gaia::{parse_gaia_csv, GaiaSource};
pub use catalogs::gwtc::{parse_gwtc3_csv, GwEvent};
pub use catalogs::hipparcos::hipparcos_row_count;
pub use catalogs::landsat::looks_like_landsat_stac_json;
pub use catalogs::mcgill::{parse_mcgill_csv, Magnetar};
pub use catalogs::nanograv::{parse_nanograv_free_spectrum, FreeSpectrumPoint};
pub use catalogs::pantheon::{parse_pantheon_dat, Supernova};
pub use catalogs::planck::bestfit as planck2018;
pub use catalogs::sdss::{parse_sdss_quasar_csv, SdssQuasar};
pub use catalogs::sorce::{parse_sorce_csv, SorceMeasurement};
pub use catalogs::tsi::{parse_tsi_csv, TsiMeasurement};
pub use catalogs::jarvis::{
    JarvisMaterial, FigshareFile, list_figshare_files, fetch_jarvis_json,
    parse_jarvis_json, sample_materials,
};
pub use catalogs::union3::parse_union3_chain;
