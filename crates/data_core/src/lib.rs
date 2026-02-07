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

/// All dataset provider names that should appear in the manifest.
///
/// This list is the single source of truth for the dataset count.
/// The fetch-datasets binary, DATASET_MANIFEST.md, and this function
/// must all agree on the provider inventory.
pub fn known_provider_names() -> Vec<&'static str> {
    vec![
        "CHIME/FRB Catalog 2",
        "ATNF Pulsar Catalogue",
        "McGill Magnetar Catalog",
        "SDSS DR18 Quasars",
        "Gaia DR3 Nearby Stars",
        "Hipparcos Legacy Catalog",
        "GWTC-3 confident events",
        "GWOSC combined GWTC (O1-O4a)",
        "NANOGrav 15yr Free Spectrum",
        "Fermi GBM Burst Catalog",
        "EHT M87 2018 Data Bundle",
        "EHT SgrA 2022 Data Bundle",
        "TSIS-1 TSI Daily",
        "SORCE TSI Daily",
        "Pantheon+ SH0ES",
        "Union3 Legacy SN Ia",
        "Planck 2018 Summary",
        "WMAP 9yr MCMC Chains",
        "Planck 2018 MCMC Chains",
        "IGRF-13 Coefficients",
        "WMM 2025 Coefficients",
        "GRACE GGM05S Gravity Field",
        "GRACE-FO Gravity Field",
        "GRAIL GRGM1200B Lunar Gravity",
        "EGM2008 Static Geoid",
        "Swarm L1B Magnetic Sample",
        "Landsat C2 L2 STAC Metadata",
        "JPL DE440 Ephemeris Kernel",
        "JPL DE441 Ephemeris Kernel",
        "JPL Horizons Planetary Ephemeris",
    ]
}

/// Number of datasets in the canonical inventory.
pub const DATASET_COUNT: usize = 30;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_provider_count_matches_constant() {
        let names = known_provider_names();
        assert_eq!(
            names.len(),
            DATASET_COUNT,
            "known_provider_names() length must match DATASET_COUNT"
        );
    }

    #[test]
    fn test_no_duplicate_provider_names() {
        let names = known_provider_names();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            names.len(),
            "provider names must be unique"
        );
    }

    #[test]
    fn test_all_providers_instantiable() {
        // Verify that key provider types exist and implement DatasetProvider
        use crate::fetcher::FetchConfig;
        let config = FetchConfig::default();

        let providers: Vec<Box<dyn DatasetProvider>> = vec![
            Box::new(catalogs::chime::ChimeCat2Provider),
            Box::new(catalogs::gwtc::Gwtc3Provider),
            Box::new(catalogs::atnf::AtnfProvider),
            Box::new(catalogs::pantheon::PantheonProvider),
            Box::new(catalogs::tsi::TsisTsiProvider),
            Box::new(catalogs::sorce::SorceTsiProvider),
            Box::new(catalogs::landsat::LandsatStacProvider),
            Box::new(geophysical::swarm::SwarmMagAProvider),
            Box::new(geophysical::de_ephemeris::De440Provider),
        ];

        for p in &providers {
            // Just verify name() doesn't panic and returns non-empty
            assert!(!p.name().is_empty());
            // is_cached should not panic
            let _ = p.is_cached(&config);
        }
    }
}
