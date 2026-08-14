//! Dispatcher over the heliosphere analysis lanes.
//!
//! Each lane owns a module under `gororoba_cli_physics::heliosphere` exposing a
//! `Cli` argument struct and a `run` entry point, so the lanes share one link
//! unit instead of one executable apiece.

use anyhow::Result;
use clap::{Parser, Subcommand};
use gororoba_cli_physics::heliosphere;

#[derive(Parser)]
#[command(name = "heliosphere", about = "Heliosphere analysis lanes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

// One `Command` exists per process: `Cli::parse()` builds it and the match below
// destructures it immediately, so the 600-byte span of the widest lane costs one
// stack frame rather than per-element storage. Boxing the wide variants is the
// lint's suggested fix and does not apply here, because clap's `Subcommand`
// derive requires each payload to implement `Args` and `Box<T>` does not.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Command {
    /// Compute algebraic descriptors from heliosphere feature cubes
    AlgebraDecompose(heliosphere::algebra_decompose::Cli),
    /// Deconfounding audit: separation, nulls, and invariant embeddings
    AssociatorNullAudit(heliosphere::associator_null_audit::Cli),
    /// 5-way baseline comparison against curated labels
    BaselineComparison(heliosphere::baseline_comparison::Cli),
    /// Join associator norms with ESA plasma beta for A(beta) scatter.
    BetaJoin(heliosphere::beta_join::Cli),
    /// Block-bootstrap 95% F1 CI for THEMIS CD associator (plan P6A.S1 task 1.4)
    BootstrapF1(heliosphere::bootstrap_f1::Cli),
    /// Multi-mission boundary survey with 32D CD associator
    BoundarySurvey(heliosphere::boundary_survey::Cli),
    /// High-throughput GPU scan of 16D Takens descriptors for Box-Kite alignment
    #[cfg(feature = "gpu")]
    BoxkiteAlignment(heliosphere::boxkite_alignment::Cli),
    /// Verify box-kite alignment parity across CPU/Vulkan/CUDA backends
    BoxkiteParity(heliosphere::boxkite_parity::Cli),
    /// Commutator ||xy - yx||_2 baseline for CD associator ablation (P6A.S2.T2.7)
    CommutatorBaseline(heliosphere::commutator_baseline::Cli),
    /// Measure cross-mission stability of heliosphere invariant and algebraic descriptors
    CrossMissionInvariance(heliosphere::cross_mission_invariance::Cli),
    /// Phase vs spectral decomposition of CD associator signal
    DecompositionAudit(heliosphere::decomposition_audit::Cli),
    /// Compute SVD effective rank of CD embeddings at various dimensions
    EffectiveRank(heliosphere::effective_rank::Cli),
    /// Project 32D Takens embeddings to top 3 principal components.
    EmbedPca(heliosphere::embed_pca::Cli),
    /// False-alarm attribution: classify CD extra transitions by B-field signature.
    FaAttribution(heliosphere::fa_attribution::Cli),
    /// Cross-mission FA attribution table for methods paper
    FaCrossMissionTable(heliosphere::fa_cross_mission_table::Cli),
    /// Challenge recent heliosphere falsifications with alternate normalization and sparse-policy counter-tests
    FalsificationAudit(heliosphere::falsification_audit::Cli),
    /// Build normalized heliosphere feature cubes from real mission data
    FeatureCube(heliosphere::feature_cube::Cli),
    /// Normalize and/or difference heliosphere feature cubes without dropping rows
    FeatureTransform(heliosphere::feature_transform::Cli),
    /// Classify CD-associator extra detections as rotational vs compressive (JGR reviewer response)
    FteCaseStudy(heliosphere::fte_case_study::Cli),
    /// Full-dataset weekly sweep with curated labels
    FullSweep(heliosphere::full_sweep::Cli),
    /// Hybrid detector: test all union/intersection/weighted combinations.
    HybridDetector(heliosphere::hybrid_detector::Cli),
    /// L2-delay change-point baseline for CD associator ablation (P1 pre-registered)
    L2DelayBaseline(heliosphere::l2_delay_baseline::Cli),
    /// Axis B lag-depth sweep d=4,6,8,12 (R32 algebra fixed) -- CD ablation
    LagDepthSweep(heliosphere::lag_depth_sweep::Cli),
    /// Benchmark dense LBM and sparse memory plans from heliosphere feature cubes
    #[cfg(feature = "gpu")]
    LbmCubeRun(heliosphere::lbm_cube_run::Cli),
    /// Takens time-delay embedding of magnetic field into Cayley-Dickson space
    MagneticTakens(heliosphere::magnetic_takens::Cli),
    /// Generate a surrogate MAVEN bow shock crossing list from FGM regime transitions.
    MavenCrossingGen(heliosphere::maven_crossing_gen::Cli),
    /// Multi-day MMS magnetopause analysis with 32D CD associator
    MmsMultiday(heliosphere::mms_multiday::Cli),
    /// MMS CD associator P/R/F1 against SITL-curated magnetopause ground truth
    MmsSitlLabeled(heliosphere::mms_sitl_labeled::Cli),
    /// Dump per-minute associator norms as CSV for beta scatter analysis.
    NormDump(heliosphere::norm_dump::Cli),
    /// Compute per-radial-bin Omega_mv and Omega_phase order surplus indices
    OrderSurplus(heliosphere::order_surplus::Cli),
    /// PCA leading-PC variance ratio baseline for CD associator ablation (P6A.S2.T2.8)
    PcaVarianceBaseline(heliosphere::pca_variance_baseline::Cli),
    /// Evaluate heliosphere invariant and algebraic predictors against official DONKI event windows
    PredictiveEval(heliosphere::predictive_eval::Cli),
    /// PSP Alfvenic noise control: per-class CD associator fire rates
    PspAlfvenControl(heliosphere::psp_alfven_control::Cli),
    /// Consolidate PSP switchback omega results across multiple perihelion encounters.
    PspEncounterSummary(heliosphere::psp_encounter_summary::Cli),
    /// PSP E3 micro-switchback enrichment: quiet-interval CD fires vs consecutive B-rotation events (E-241)
    PspSwitchbackCorrelation(heliosphere::psp_switchback_correlation::Cli),
    /// Q2: Transverse/compressive mode correlation with CD associator
    Q2Transverse(heliosphere::q2_transverse::Cli),
    /// Map the algebraic quenching point across heliocentric distance using Magnetic Takens embedding
    QuenchScan(heliosphere::quench_scan::Cli),
    /// CD R16 (2-channel Bx,By x 8 lags) ablation -- Axis A dim variant
    R16Ablation(heliosphere::r16_ablation::Cli),
    /// CD R64 (8-channel half-lag x 8 lags) ablation -- Axis A dim variant
    R64Ablation(heliosphere::r64_ablation::Cli),
    /// Dense-random trilinear baseline for CD ablation (P2 pre-registered)
    RandomTrilinear(heliosphere::random_trilinear::Cli),
    /// Sparsity-matched random trilinear baseline (same zero-pattern as CD)
    RandomTrilinearSparse(heliosphere::random_trilinear_sparse::Cli),
    /// Consolidate the four Rosetta normalization ablation JSON files into a single
    RosettaAblationSummary(heliosphere::rosetta_ablation_summary::Cli),
    /// Rosetta 67P draping analysis with 32D CD associator
    RosettaDraping(heliosphere::rosetta_draping::Cli),
    /// Cross-mission log-associator self-similarity survey
    SelfsimLogstats(heliosphere::selfsim_logstats::Cli),
    /// Promote and stress-test the mainline heliosphere sparse-policy candidate across seeds and cubes
    SparsePolicyMainline(heliosphere::sparse_policy_mainline::Cli),
    /// Compare robust and algebra-derived sparse masks against 1024^3 sparse execution budgets
    #[cfg(feature = "gpu")]
    SparsePreservation(heliosphere::sparse_preservation::Cli),
    /// PSP switchback detection + fast-wind Omega_mv analysis
    SwitchbackOmega(heliosphere::switchback_omega::Cli),
    /// Synthetic Plasma Stress Test: 1D MHD wave generator + CD associator.
    SyntheticMhd(heliosphere::synthetic_mhd::Cli),
    /// Sweep Takens tau {1,2,5,10} min on THEMIS E-237 data (E-240)
    TakensTauSweep(heliosphere::takens_tau_sweep::Cli),
    /// Time-aligned OMNI / spacecraft overlay and fleet coverage report
    TemporalOverlay(heliosphere::temporal_overlay::Cli),
    /// General-interval THEMIS CD benchmark (non-SC-selected; plan P6A.S1 task 1.10)
    ThemisGeneralBenchmark(heliosphere::themis_general_benchmark::Cli),
    /// THEMIS CD associator P/R/F1 against Staples et al. (2020) ground truth
    ThemisStaplesLabeled(heliosphere::themis_staples_labeled::Cli),
    /// CD associator analysis of Voyager heliopause crossings (Phase 9A.2-9A.3)
    VoyagerHeliopause(heliosphere::voyager_heliopause::Cli),
    /// Window function sensitivity (boxcar/Hamming/Hann) for CD delay embedding (M-5)
    WindowSensitivity(heliosphere::window_sensitivity::Cli),
    /// Zero-divisor proximity audit for weak-field CD embeddings.
    ZdAudit,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::AlgebraDecompose(cli) => heliosphere::algebra_decompose::run(cli),
        Command::AssociatorNullAudit(cli) => heliosphere::associator_null_audit::run(cli),
        Command::BaselineComparison(cli) => heliosphere::baseline_comparison::run(cli),
        Command::BetaJoin(cli) => {
            heliosphere::beta_join::run(cli);
            Ok(())
        }
        Command::BootstrapF1(cli) => heliosphere::bootstrap_f1::run(cli),
        Command::BoundarySurvey(cli) => heliosphere::boundary_survey::run(cli),
        #[cfg(feature = "gpu")]
        Command::BoxkiteAlignment(cli) => heliosphere::boxkite_alignment::run(cli),
        Command::BoxkiteParity(cli) => heliosphere::boxkite_parity::run(cli),
        Command::CommutatorBaseline(cli) => heliosphere::commutator_baseline::run(cli),
        Command::CrossMissionInvariance(cli) => heliosphere::cross_mission_invariance::run(cli),
        Command::DecompositionAudit(cli) => heliosphere::decomposition_audit::run(cli),
        Command::EffectiveRank(cli) => heliosphere::effective_rank::run(cli),
        Command::EmbedPca(cli) => {
            heliosphere::embed_pca::run(cli);
            Ok(())
        }
        Command::FaAttribution(cli) => heliosphere::fa_attribution::run(cli),
        Command::FaCrossMissionTable(cli) => heliosphere::fa_cross_mission_table::run(cli),
        Command::FalsificationAudit(cli) => heliosphere::falsification_audit::run(cli),
        Command::FeatureCube(cli) => heliosphere::feature_cube::run(cli),
        Command::FeatureTransform(cli) => heliosphere::feature_transform::run(cli),
        Command::FteCaseStudy(cli) => heliosphere::fte_case_study::run(cli),
        Command::FullSweep(cli) => heliosphere::full_sweep::run(cli),
        Command::HybridDetector(cli) => heliosphere::hybrid_detector::run(cli),
        Command::L2DelayBaseline(cli) => heliosphere::l2_delay_baseline::run(cli),
        Command::LagDepthSweep(cli) => heliosphere::lag_depth_sweep::run(cli),
        #[cfg(feature = "gpu")]
        Command::LbmCubeRun(cli) => heliosphere::lbm_cube_run::run(cli),
        Command::MagneticTakens(cli) => heliosphere::magnetic_takens::run(cli),
        Command::MavenCrossingGen(cli) => heliosphere::maven_crossing_gen::run(cli),
        Command::MmsMultiday(cli) => heliosphere::mms_multiday::run(cli),
        Command::MmsSitlLabeled(cli) => heliosphere::mms_sitl_labeled::run(cli),
        Command::NormDump(cli) => {
            heliosphere::norm_dump::run(cli);
            Ok(())
        }
        Command::OrderSurplus(cli) => heliosphere::order_surplus::run(cli),
        Command::PcaVarianceBaseline(cli) => heliosphere::pca_variance_baseline::run(cli),
        Command::PredictiveEval(cli) => heliosphere::predictive_eval::run(cli),
        Command::PspAlfvenControl(cli) => heliosphere::psp_alfven_control::run(cli),
        Command::PspEncounterSummary(cli) => heliosphere::psp_encounter_summary::run(cli),
        Command::PspSwitchbackCorrelation(cli) => heliosphere::psp_switchback_correlation::run(cli),
        Command::Q2Transverse(cli) => heliosphere::q2_transverse::run(cli),
        Command::QuenchScan(cli) => heliosphere::quench_scan::run(cli),
        Command::R16Ablation(cli) => heliosphere::r16_ablation::run(cli),
        Command::R64Ablation(cli) => heliosphere::r64_ablation::run(cli),
        Command::RandomTrilinear(cli) => heliosphere::random_trilinear::run(cli),
        Command::RandomTrilinearSparse(cli) => heliosphere::random_trilinear_sparse::run(cli),
        Command::RosettaAblationSummary(cli) => heliosphere::rosetta_ablation_summary::run(cli),
        Command::RosettaDraping(cli) => heliosphere::rosetta_draping::run(cli),
        Command::SelfsimLogstats(cli) => heliosphere::selfsim_logstats::run(cli),
        Command::SparsePolicyMainline(cli) => heliosphere::sparse_policy_mainline::run(cli),
        #[cfg(feature = "gpu")]
        Command::SparsePreservation(cli) => heliosphere::sparse_preservation::run(cli),
        Command::SwitchbackOmega(cli) => heliosphere::switchback_omega::run(cli),
        Command::SyntheticMhd(cli) => {
            heliosphere::synthetic_mhd::run(cli);
            Ok(())
        }
        Command::TakensTauSweep(cli) => heliosphere::takens_tau_sweep::run(cli),
        Command::TemporalOverlay(cli) => heliosphere::temporal_overlay::run(cli),
        Command::ThemisGeneralBenchmark(cli) => heliosphere::themis_general_benchmark::run(cli),
        Command::ThemisStaplesLabeled(cli) => heliosphere::themis_staples_labeled::run(cli),
        Command::VoyagerHeliopause(cli) => heliosphere::voyager_heliopause::run(cli),
        Command::WindowSensitivity(cli) => heliosphere::window_sensitivity::run(cli),
        Command::ZdAudit => heliosphere::zd_audit::run(),
    }
}
