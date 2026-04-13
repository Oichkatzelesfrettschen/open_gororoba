//! Re-export integrity tests for the nanograv module extraction (PH-3 B2b).
//!
//! WHY: The nanograv timing engine (5123 lines) was moved from gororoba_cli_data
//! into data_core::nanograv, with pub-use re-exports keeping the old module paths
//! alive. These tests verify that downstream consumers can still resolve every
//! public type and function through the gororoba_cli_data facade.
//!
//! The bulk imports below are intentionally comprehensive -- they exist to prove
//! that the re-export surface is intact, not to be called at runtime. The
//! allow(unused_imports) is therefore correct here.

#[allow(unused_imports)]
use gororoba_cli_data::nanograv_timing::{
    DmxPoint, PairAuditRow, PulsarTimingData, ResidualPoint, TimingModelMetadata, WidebandDmPoint,
    canonical_pulsar_name, load_release,
};

#[allow(unused_imports)]
use gororoba_cli_data::nanograv_timing_model::{
    AstrometryTerms, BinaryFamily, DispersionTerms, DmxWindow, ParameterTerm, ReleaseBand,
    SelectorTerm, TaggedTerm, TimingModel, load_release_timing_models, parse_par_timing_model,
};

#[allow(unused_imports)]
use gororoba_cli_data::nanograv_refit::{
    AggregatedObservation, ChannelResidual, FitParameterEstimate, LinearFitSummary,
    RefitDataset, RefitOutputRow, RefitResult, WidebandToa, build_refit_dataset,
    load_phase1_models_from_report, solve_refit,
};

#[allow(unused_imports)]
use gororoba_cli_data::nanograv_timing_engine::{
    IndependentDataset, IndependentFitSummary, IndependentObservation, IndependentRefitResult,
    IndependentRefitRow, SiteId, StructuredCovarianceCalibration, TimingEphemeris,
    TopocentricToa, build_independent_dataset, build_phase1_independent_datasets,
    solve_independent_refit, tactical_lane_for_pulsar,
};

/// Verify that core timing types are constructible through the re-export path.
#[test]
fn timing_types_accessible() {
    let meta = TimingModelMetadata::default();
    assert_eq!(meta.file_count, 0);

    let data = PulsarTimingData::default();
    assert!(data.pulsar.is_empty());
    assert!(data.wideband_dm.is_empty());
    assert!(data.dmx.is_empty());
    assert!(data.avg_residuals.is_empty());
    assert!(data.full_residuals.is_empty());
}

/// Verify that canonical_pulsar_name works through the re-export.
#[test]
fn canonical_name_reexport() {
    assert_eq!(
        canonical_pulsar_name("J1713+0747_PINT_20230131.wb".to_string()),
        "J1713+0747"
    );
}

/// Verify that timing_model enums and structs are constructible.
#[test]
fn timing_model_types_accessible() {
    assert_eq!(ReleaseBand::Narrowband.as_str(), "narrowband");
    assert_eq!(ReleaseBand::Wideband.as_str(), "wideband");
    assert_eq!(BinaryFamily::Ddk.as_str(), "DDK");

    let _terms = AstrometryTerms::default();
    let _disp = DispersionTerms::default();
    let _window = DmxWindow::default();
}

/// Verify that engine types are accessible (requires nanograv-engine feature).
#[test]
fn engine_types_accessible() {
    assert_eq!(SiteId::GreenBank.as_str(), "gbt");
    assert_eq!(SiteId::Arecibo.as_str(), "arecibo");
    assert_eq!(SiteId::Vla.as_str(), "vla");

    let lane = tactical_lane_for_pulsar("J0030+0451");
    assert!(!lane.is_empty());
}

/// Verify that function signatures are resolvable (type-check, not runtime).
/// We only check that the function pointers have the expected arity by
/// taking references -- no actual data or IO needed.
#[test]
fn function_signatures_resolvable() {
    // These lines verify that the function types match what callers expect.
    // If the re-export broke a signature, this would fail to compile.
    let _: fn(String) -> String = canonical_pulsar_name;
    let _: fn(&std::path::Path) -> anyhow::Result<std::collections::BTreeMap<String, PulsarTimingData>> = load_release;
    let _: fn(&std::path::Path, ReleaseBand) -> anyhow::Result<Vec<TimingModel>> = load_release_timing_models;
    let _: fn(&std::path::Path, ReleaseBand) -> anyhow::Result<TimingModel> = parse_par_timing_model;
}
