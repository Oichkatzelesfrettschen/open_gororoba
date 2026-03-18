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

#[macro_use]
mod macros;

pub mod benchmarks;
pub mod catalog_feature_cube;
pub mod catalogs;
pub mod cdf_support;
pub mod doc_links;
pub mod download_stack;
pub mod fetcher;
pub mod formats;
pub mod geophysical;
#[cfg(feature = "hdf5-export")]
pub mod hdf5_export;
pub mod heliosphere_event_labels;
pub mod heliosphere_feature_cube;
pub mod parse;
pub mod provenance;
pub mod quality;
pub mod registry;
pub mod seti;
pub mod spatial;
pub mod spice;
#[cfg(feature = "dataframe")]
pub mod tabular;
pub mod time_bounds;

pub use catalog_feature_cube::{
    CatalogFeatureChannel, CatalogFeatureCube, CatalogFeatureCubeManifest, CatalogFeatureRow,
    CatalogNuisanceModel, NuisanceEffectReport, ResidualizedCatalogFeatureCube,
    encode_dictionary_value, parse_catalog_feature_cube_json, pipe_count, stable_dictionary,
};
pub use download_stack::{
    DownloadBackend, DownloadLedgerRow, DownloadRoute, DownloadStack, EndpointCapabilities,
    EndpointSurface, HostPolicyRegistry, HostRoutingPolicy, RetryClass, TransferAttempt,
    TransferKind, TransferRequest, TransferResult, TransferTrace, load_host_policy_registry,
};
pub use fetcher::{
    DatasetProvider, FetchConfig, FetchError, compute_sha256, download_to_file, download_to_string,
};
pub use heliosphere_event_labels::{
    ForecastResidual, HeliosphereEventKind, HeliosphereEventLabel, HeliosphereEventSource,
    HeliosphereEventWindow, fetch_donki_event_labels, fetch_official_forecast_residuals,
    labels_to_prediction_windows,
};
pub use heliosphere_feature_cube::{
    HELIOSPHERE_CHANNEL_NAMES, HELIOSPHERE_DYNAMIC_CHANNEL_NAMES, HELIOSPHERE_DYNAMIC_DIM,
    HELIOSPHERE_FEATURE_DIM, HELIOSPHERE_INVARIANT_CHANNEL_NAMES, HELIOSPHERE_INVARIANT_DIM,
    HELIOSPHERE_SIGNAL_DIM, HELIOSPHERE_SUPPORT_DIM, HeliosphereFeatureCube,
    HeliosphereFeatureCubeManifest, HeliosphereFeatureRow, HeliosphereInvariantSample,
    HeliosphereTransformGroupStats, HeliosphereTransformMode, HeliosphereTransformResult,
    SparseExecutionMode, SparseExecutionPlan, SparseHardwareEnvelope, SparseMemoryPlan,
    compute_invariant_samples, estimate_sparse_execution_plan, estimate_sparse_memory_plan,
    heliosphere_row_datetime, transform_feature_rows, transform_feature_rows_with_stats,
};
pub use quality::{
    RhoQualityError, RhoQualityThresholds, RhoTraceQuality, assess_rho_trace, validate_rho_trace,
};
pub use spatial::{
    CatalogModality, RadiusMatch, SkyGridIndex, SkyPoint, angular_separation_arcsec,
};
#[cfg(feature = "dataframe")]
pub use tabular::{
    TabularError, TabularOverview, csv_records_to_frame, frame_overview, json_records_to_frame,
    provider_inventory_frame,
};

#[cfg(feature = "fits")]
pub use catalogs::lotss::{
    LotssFitsBestMatch, LotssFitsBestMatchSummary, LotssFitsExecutionReport,
    crossmatch_points_against_fits_catalog,
};
pub use catalogs::{
    aflow::{
        AflowMaterial, AflowProvider, fetch_aflow_dataset, parse_aflow_json, parse_aflow_records,
    },
    atnf::{Pulsar, parse_atnf_csv},
    chime::{FrbEvent, extract_repeaters, parse_chime_csv},
    desi_bao::{BaoMeasurement, desi_dr1_bao},
    fermi_gbm::{GrbEvent, parse_fermi_gbm_csv},
    gaia::{GaiaSource, parse_gaia_csv},
    gwtc::{GwEvent, parse_gwtc3_csv},
    hipparcos::hipparcos_row_count,
    jarvis::{
        FigshareFile, JarvisMaterial, JarvisProvider, fetch_jarvis_json, list_figshare_files,
        parse_jarvis_json, sample_materials,
    },
    landsat::looks_like_landsat_stac_json,
    mcgill::{Magnetar, parse_mcgill_csv},
    nanograv::{FreeSpectrumPoint, parse_nanograv_free_spectrum},
    pantheon::{PantheonCovProvider, Supernova, parse_pantheon_cov, parse_pantheon_dat},
    pdg::{PdgMassEntry, parse_pdg_mass_reference_csv},
    planck::bestfit as planck2018,
    psp_fields::{PspFieldsMagRecord, PspFieldsProvider, parse_psp_fields_file},
    sdss::{SdssQuasar, parse_sdss_quasar_csv},
    solar_orbiter_swa::{
        SolarOrbiterSwaProvider, SolarOrbiterSwaRecord, parse_solar_orbiter_swa_file,
    },
    sorce::{SorceMeasurement, parse_sorce_csv},
    tsi::{TsiMeasurement, parse_tsi_csv},
    union3::parse_union3_chain,
    voyager_pws::{VoyagerPwsProvider, VoyagerPwsRecord, parse_voyager_pws_file},
    wow::{
        Bl6equj5Bundle, Bl6equj5ManifestProvider, WowPrintoutProvider, WowPrintoutRow,
        abacad_filter, parse_bl_manifest_csv, parse_wow_printout_csv, wow_char_to_intensity,
    },
};

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
        "JWST Public Observation Metadata",
        "HST Public Observation Metadata",
        "Hipparcos Legacy Catalog",
        "GWTC-3 confident events",
        "GWOSC combined GWTC (O1-O4a)",
        "NANOGrav 15yr Free Spectrum",
        "Fermi GBM Burst Catalog",
        "EHT M87 2018 Data Bundle",
        "EHT SgrA 2022 Data Bundle",
        "DESI DR1 BAO",
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
        "Wow! Signal 1977 Printout Scan",
        "BL 6EQUJ5 GBT Manifest",
    ]
}

/// Number of datasets in the canonical inventory.
pub const DATASET_COUNT: usize = 37;

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
        "Fermi GBM Burst Catalog"
        | "EHT M87 2018 Data Bundle"
        | "EHT SgrA 2022 Data Bundle"
        | "Wow! Signal 1977 Printout Scan"
        | "BL 6EQUJ5 GBT Manifest" => "electromagnetic",
        "CHIME/FRB Catalog 2"
        | "ATNF Pulsar Catalogue"
        | "McGill Magnetar Catalog"
        | "SDSS DR18 Quasars"
        | "Gaia DR3 Nearby Stars"
        | "JWST Public Observation Metadata"
        | "HST Public Observation Metadata"
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
        "JWST Public Observation Metadata" => &[],
        "HST Public Observation Metadata" => &[],
        "Hipparcos Legacy Catalog" => &["C-437"],
        "GWTC-3 confident events" => &["C-006", "C-007", "C-025", "C-060"],
        "GWOSC combined GWTC (O1-O4a)" => &["C-061", "C-070", "C-437", "C-439", "C-440", "C-441"],
        "NANOGrav 15yr Free Spectrum" => &["C-059", "C-070"],
        "Fermi GBM Burst Catalog" => &["C-064", "C-437"],
        "Pantheon+ SH0ES" => &["C-038", "C-437", "C-441"],
        "DESI DR1 BAO" => &["C-057", "C-441"],
        "Planck 2018 Summary" => &["C-040", "C-058"],
        "Wow! Signal 1977 Printout Scan" => &["C-769"],
        "BL 6EQUJ5 GBT Manifest" => &["C-771"],
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
        // 14 datasets have claims (DESI is hardcoded, not in provider list)
        assert!(
            backed.len() >= 13,
            "Expected at least 13 claim-backed providers, got {}",
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
            Box::new(catalogs::wow::WowPrintoutProvider),
            Box::new(catalogs::wow::Bl6equj5ManifestProvider),
        ];

        for p in &providers {
            // Just verify name() doesn't panic and returns non-empty
            assert!(!p.name().is_empty());
            // is_cached should not panic
            let _ = p.is_cached(&config);
        }
    }
}
