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

pub mod benchmarks;
pub mod catalogs;
pub mod doc_links;
pub mod fetcher;
pub mod formats;
pub mod geophysical;
#[cfg(feature = "hdf5-export")]
pub mod hdf5_export;
pub mod provenance;

pub use fetcher::{compute_sha256, download_to_file, download_to_string};
pub use fetcher::{DatasetProvider, FetchConfig, FetchError};

pub use catalogs::aflow::{
    fetch_aflow_dataset, parse_aflow_json, parse_aflow_records, AflowMaterial, AflowProvider,
};
pub use catalogs::atnf::{parse_atnf_csv, Pulsar};
pub use catalogs::chime::{extract_repeaters, parse_chime_csv, FrbEvent};
pub use catalogs::desi_bao::{desi_dr1_bao, BaoMeasurement};
pub use catalogs::fermi_gbm::{parse_fermi_gbm_csv, GrbEvent};
pub use catalogs::gaia::{parse_gaia_csv, GaiaSource};
pub use catalogs::gwtc::{parse_gwtc3_csv, GwEvent};
pub use catalogs::hipparcos::hipparcos_row_count;
pub use catalogs::jarvis::{
    fetch_jarvis_json, list_figshare_files, parse_jarvis_json, sample_materials, FigshareFile,
    JarvisMaterial, JarvisProvider,
};
pub use catalogs::landsat::looks_like_landsat_stac_json;
pub use catalogs::mcgill::{parse_mcgill_csv, Magnetar};
pub use catalogs::nanograv::{parse_nanograv_free_spectrum, FreeSpectrumPoint};
pub use catalogs::pantheon::{parse_pantheon_dat, Supernova};
pub use catalogs::planck::bestfit as planck2018;
pub use catalogs::sdss::{parse_sdss_quasar_csv, SdssQuasar};
pub use catalogs::sorce::{parse_sorce_csv, SorceMeasurement};
pub use catalogs::tsi::{parse_tsi_csv, TsiMeasurement};
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
        "JARVIS-DFT 3D",
        "AFLOW Materials Database",
    ]
}

/// Number of datasets in the canonical inventory.
pub const DATASET_COUNT: usize = 32;

/// The 8 scientific pillars that organize datasets.
pub const PILLARS: &[&str] = &[
    "candle",
    "gravitational",
    "electromagnetic",
    "survey",
    "cmb",
    "solar",
    "geophysical",
    "materials",
];

/// Map each provider name to its scientific pillar.
pub fn provider_pillar(name: &str) -> &'static str {
    match name {
        "Pantheon+ SH0ES" | "Union3 Legacy SN Ia" => "candle",
        "GWTC-3 confident events"
        | "GWOSC combined GWTC (O1-O4a)"
        | "NANOGrav 15yr Free Spectrum" => "gravitational",
        "Fermi GBM Burst Catalog" | "EHT M87 2018 Data Bundle" | "EHT SgrA 2022 Data Bundle" => {
            "electromagnetic"
        }
        "CHIME/FRB Catalog 2"
        | "ATNF Pulsar Catalogue"
        | "McGill Magnetar Catalog"
        | "SDSS DR18 Quasars"
        | "Gaia DR3 Nearby Stars"
        | "Hipparcos Legacy Catalog" => "survey",
        "Planck 2018 Summary" | "WMAP 9yr MCMC Chains" | "Planck 2018 MCMC Chains" => "cmb",
        "TSIS-1 TSI Daily" | "SORCE TSI Daily" => "solar",
        "JARVIS-DFT 3D" | "AFLOW Materials Database" => "materials",
        _ => "geophysical", // IGRF, WMM, GRACE, EGM2008, Swarm, Landsat, DE440/441, Horizons
    }
}

/// Claim IDs backed by each dataset. Returns empty slice for infrastructure datasets.
pub fn claims_for_provider(name: &str) -> &'static [&'static str] {
    match name {
        "CHIME/FRB Catalog 2" => &[
            "C-043", "C-062", "C-071", "C-080", "C-436", "C-437", "C-438", "C-440",
        ],
        "ATNF Pulsar Catalogue" => &["C-043", "C-063", "C-437"],
        "McGill Magnetar Catalog" => &["C-043", "C-063", "C-437"],
        "SDSS DR18 Quasars" => &["C-437"],
        "Gaia DR3 Nearby Stars" => &["C-437"],
        "Hipparcos Legacy Catalog" => &["C-437"],
        "GWTC-3 confident events" => &["C-006", "C-007", "C-025", "C-060"],
        "GWOSC combined GWTC (O1-O4a)" => &["C-061", "C-070", "C-437", "C-439", "C-440", "C-441"],
        "NANOGrav 15yr Free Spectrum" => &["C-059", "C-070"],
        "Fermi GBM Burst Catalog" => &["C-064", "C-437"],
        "Pantheon+ SH0ES" => &["C-038", "C-437", "C-441"],
        "DESI DR1 BAO" => &["C-057", "C-441"],
        "Planck 2018 Summary" => &["C-040", "C-058"],
        _ => &[],
    }
}

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
        assert_eq!(sorted.len(), names.len(), "provider names must be unique");
    }

    #[test]
    fn test_every_provider_has_a_pillar() {
        for name in known_provider_names() {
            let pillar = provider_pillar(name);
            assert!(
                PILLARS.contains(&pillar),
                "Provider {:?} mapped to unknown pillar {:?}",
                name,
                pillar
            );
        }
    }

    #[test]
    fn test_all_pillars_have_providers() {
        for pillar in PILLARS {
            let count = known_provider_names()
                .iter()
                .filter(|n| provider_pillar(n) == *pillar)
                .count();
            assert!(count > 0, "Pillar {:?} has no providers", pillar);
        }
    }

    #[test]
    fn test_claim_backed_provider_count() {
        let backed: Vec<_> = known_provider_names()
            .into_iter()
            .filter(|n| !claims_for_provider(n).is_empty())
            .collect();
        // 12 datasets have claims (DESI is hardcoded, not in provider list)
        assert!(
            backed.len() >= 11,
            "Expected at least 11 claim-backed providers, got {}",
            backed.len()
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
            Box::new(catalogs::jarvis::JarvisProvider),
            Box::new(catalogs::aflow::AflowProvider),
        ];

        for p in &providers {
            // Just verify name() doesn't panic and returns non-empty
            assert!(!p.name().is_empty());
            // is_cached should not panic
            let _ = p.is_cached(&config);
        }
    }
}
