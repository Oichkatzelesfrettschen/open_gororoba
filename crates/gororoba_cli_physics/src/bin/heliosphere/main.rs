//! Dispatcher over the heliosphere analysis lanes.
//!
//! Each lane owns a sibling module exposing a `Cli` argument struct and a `run`
//! entry point, so the lanes share one link unit instead of one executable
//! apiece. The modules sit under `src/bin/heliosphere/` rather than in the
//! library, which keeps them out of every binary that imports
//! `gororoba_cli_physics` and links them only here.

use anyhow::Result;
use clap::{Parser, Subcommand};
mod algebra_decompose;
mod associator_null_audit;
mod baseline_comparison;
mod beta_join;
mod bootstrap_f1;
mod boundary_survey;
#[cfg(feature = "gpu")]
mod boxkite_alignment;
mod boxkite_parity;
mod commutator_baseline;
mod cross_mission_invariance;
mod decomposition_audit;
mod effective_rank;
mod embed_pca;
mod fa_attribution;
mod fa_cross_mission_table;
mod falsification_audit;
mod feature_cube;
mod feature_transform;
mod fte_case_study;
mod full_sweep;
mod hybrid_detector;
mod l2_delay_baseline;
mod lag_depth_sweep;
#[cfg(feature = "gpu")]
mod lbm_cube_run;
mod magnetic_takens;
mod maven_crossing_gen;
mod mms_multiday;
mod mms_sitl_labeled;
mod norm_dump;
mod order_surplus;
mod pca_variance_baseline;
mod predictive_eval;
mod psp_alfven_control;
mod psp_encounter_summary;
mod psp_switchback_correlation;
mod q2_transverse;
mod quench_scan;
mod r16_ablation;
mod r64_ablation;
mod random_trilinear;
mod random_trilinear_sparse;
mod rosetta_ablation_summary;
mod rosetta_draping;
mod selfsim_logstats;
mod sparse_policy_mainline;
#[cfg(feature = "gpu")]
mod sparse_preservation;
mod switchback_omega;
mod synthetic_mhd;
mod takens_tau_sweep;
mod temporal_overlay;
mod themis_general_benchmark;
mod themis_staples_labeled;
mod voyager_heliopause;
mod window_sensitivity;
mod zd_audit;

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
    AlgebraDecompose(algebra_decompose::Cli),
    /// Deconfounding audit: separation, nulls, and invariant embeddings
    AssociatorNullAudit(associator_null_audit::Cli),
    /// 5-way baseline comparison against curated labels
    BaselineComparison(baseline_comparison::Cli),
    /// Join associator norms with ESA plasma beta for A(beta) scatter.
    BetaJoin(beta_join::Cli),
    /// Block-bootstrap 95% F1 CI for THEMIS CD associator (plan P6A.S1 task 1.4)
    BootstrapF1(bootstrap_f1::Cli),
    /// Multi-mission boundary survey with 32D CD associator
    BoundarySurvey(boundary_survey::Cli),
    /// High-throughput GPU scan of 16D Takens descriptors for Box-Kite alignment
    #[cfg(feature = "gpu")]
    BoxkiteAlignment(boxkite_alignment::Cli),
    /// Verify box-kite alignment parity across CPU/Vulkan/CUDA backends
    BoxkiteParity(boxkite_parity::Cli),
    /// Commutator ||xy - yx||_2 baseline for CD associator ablation (P6A.S2.T2.7)
    CommutatorBaseline(commutator_baseline::Cli),
    /// Measure cross-mission stability of heliosphere invariant and algebraic descriptors
    CrossMissionInvariance(cross_mission_invariance::Cli),
    /// Phase vs spectral decomposition of CD associator signal
    DecompositionAudit(decomposition_audit::Cli),
    /// Compute SVD effective rank of CD embeddings at various dimensions
    EffectiveRank(effective_rank::Cli),
    /// Project 32D Takens embeddings to top 3 principal components.
    EmbedPca(embed_pca::Cli),
    /// False-alarm attribution: classify CD extra transitions by B-field signature.
    FaAttribution(fa_attribution::Cli),
    /// Cross-mission FA attribution table for methods paper
    FaCrossMissionTable(fa_cross_mission_table::Cli),
    /// Challenge recent heliosphere falsifications with alternate normalization and sparse-policy counter-tests
    FalsificationAudit(falsification_audit::Cli),
    /// Build normalized heliosphere feature cubes from real mission data
    FeatureCube(feature_cube::Cli),
    /// Normalize and/or difference heliosphere feature cubes without dropping rows
    FeatureTransform(feature_transform::Cli),
    /// Classify CD-associator extra detections as rotational vs compressive (JGR reviewer response)
    FteCaseStudy(fte_case_study::Cli),
    /// Full-dataset weekly sweep with curated labels
    FullSweep(full_sweep::Cli),
    /// Hybrid detector: test all union/intersection/weighted combinations.
    HybridDetector(hybrid_detector::Cli),
    /// L2-delay change-point baseline for CD associator ablation (P1 pre-registered)
    L2DelayBaseline(l2_delay_baseline::Cli),
    /// Axis B lag-depth sweep d=4,6,8,12 (R32 algebra fixed) -- CD ablation
    LagDepthSweep(lag_depth_sweep::Cli),
    /// Benchmark dense LBM and sparse memory plans from heliosphere feature cubes
    #[cfg(feature = "gpu")]
    LbmCubeRun(lbm_cube_run::Cli),
    /// Takens time-delay embedding of magnetic field into Cayley-Dickson space
    MagneticTakens(magnetic_takens::Cli),
    /// Generate a surrogate MAVEN bow shock crossing list from FGM regime transitions.
    MavenCrossingGen(maven_crossing_gen::Cli),
    /// Multi-day MMS magnetopause analysis with 32D CD associator
    MmsMultiday(mms_multiday::Cli),
    /// MMS CD associator P/R/F1 against SITL-curated magnetopause ground truth
    MmsSitlLabeled(mms_sitl_labeled::Cli),
    /// Dump per-minute associator norms as CSV for beta scatter analysis.
    NormDump(norm_dump::Cli),
    /// Compute per-radial-bin Omega_mv and Omega_phase order surplus indices
    OrderSurplus(order_surplus::Cli),
    /// PCA leading-PC variance ratio baseline for CD associator ablation (P6A.S2.T2.8)
    PcaVarianceBaseline(pca_variance_baseline::Cli),
    /// Evaluate heliosphere invariant and algebraic predictors against official DONKI event windows
    PredictiveEval(predictive_eval::Cli),
    /// PSP Alfvenic noise control: per-class CD associator fire rates
    PspAlfvenControl(psp_alfven_control::Cli),
    /// Consolidate PSP switchback omega results across multiple perihelion encounters.
    PspEncounterSummary(psp_encounter_summary::Cli),
    /// PSP E3 micro-switchback enrichment: quiet-interval CD fires vs consecutive B-rotation events (E-271)
    PspSwitchbackCorrelation(psp_switchback_correlation::Cli),
    /// Q2: Transverse/compressive mode correlation with CD associator
    Q2Transverse(q2_transverse::Cli),
    /// Map the algebraic quenching point across heliocentric distance using Magnetic Takens embedding
    QuenchScan(quench_scan::Cli),
    /// CD R16 (2-channel Bx,By x 8 lags) ablation -- Axis A dim variant
    R16Ablation(r16_ablation::Cli),
    /// CD R64 (8-channel half-lag x 8 lags) ablation -- Axis A dim variant
    R64Ablation(r64_ablation::Cli),
    /// Dense-random trilinear baseline for CD ablation (P2 pre-registered)
    RandomTrilinear(random_trilinear::Cli),
    /// Sparsity-matched random trilinear baseline (same zero-pattern as CD)
    RandomTrilinearSparse(random_trilinear_sparse::Cli),
    /// Consolidate the four Rosetta normalization ablation JSON files into a single
    RosettaAblationSummary(rosetta_ablation_summary::Cli),
    /// Rosetta 67P draping analysis with 32D CD associator
    RosettaDraping(rosetta_draping::Cli),
    /// Cross-mission log-associator self-similarity survey
    SelfsimLogstats(selfsim_logstats::Cli),
    /// Promote and stress-test the mainline heliosphere sparse-policy candidate across seeds and cubes
    SparsePolicyMainline(sparse_policy_mainline::Cli),
    /// Compare robust and algebra-derived sparse masks against 1024^3 sparse execution budgets
    #[cfg(feature = "gpu")]
    SparsePreservation(sparse_preservation::Cli),
    /// PSP switchback detection + fast-wind Omega_mv analysis
    SwitchbackOmega(switchback_omega::Cli),
    /// Synthetic Plasma Stress Test: 1D MHD wave generator + CD associator.
    SyntheticMhd(synthetic_mhd::Cli),
    /// Sweep Takens tau {1,2,5,10} min on THEMIS E-267 data (E-270)
    TakensTauSweep(takens_tau_sweep::Cli),
    /// Time-aligned OMNI / spacecraft overlay and fleet coverage report
    TemporalOverlay(temporal_overlay::Cli),
    /// General-interval THEMIS CD benchmark (non-SC-selected; plan P6A.S1 task 1.10)
    ThemisGeneralBenchmark(themis_general_benchmark::Cli),
    /// THEMIS CD associator P/R/F1 against Staples et al. (2020) ground truth
    ThemisStaplesLabeled(themis_staples_labeled::Cli),
    /// CD associator analysis of Voyager heliopause crossings (Phase 9A.2-9A.3)
    VoyagerHeliopause(voyager_heliopause::Cli),
    /// Window function sensitivity (boxcar/Hamming/Hann) for CD delay embedding (M-5)
    WindowSensitivity(window_sensitivity::Cli),
    /// Zero-divisor proximity audit for weak-field CD embeddings.
    ZdAudit,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::AlgebraDecompose(cli) => algebra_decompose::run(cli),
        Command::AssociatorNullAudit(cli) => associator_null_audit::run(cli),
        Command::BaselineComparison(cli) => baseline_comparison::run(cli),
        Command::BetaJoin(cli) => {
            beta_join::run(cli);
            Ok(())
        }
        Command::BootstrapF1(cli) => bootstrap_f1::run(cli),
        Command::BoundarySurvey(cli) => boundary_survey::run(cli),
        #[cfg(feature = "gpu")]
        Command::BoxkiteAlignment(cli) => boxkite_alignment::run(cli),
        Command::BoxkiteParity(cli) => boxkite_parity::run(cli),
        Command::CommutatorBaseline(cli) => commutator_baseline::run(cli),
        Command::CrossMissionInvariance(cli) => cross_mission_invariance::run(cli),
        Command::DecompositionAudit(cli) => decomposition_audit::run(cli),
        Command::EffectiveRank(cli) => effective_rank::run(cli),
        Command::EmbedPca(cli) => {
            embed_pca::run(cli);
            Ok(())
        }
        Command::FaAttribution(cli) => fa_attribution::run(cli),
        Command::FaCrossMissionTable(cli) => fa_cross_mission_table::run(cli),
        Command::FalsificationAudit(cli) => falsification_audit::run(cli),
        Command::FeatureCube(cli) => feature_cube::run(cli),
        Command::FeatureTransform(cli) => feature_transform::run(cli),
        Command::FteCaseStudy(cli) => fte_case_study::run(cli),
        Command::FullSweep(cli) => full_sweep::run(cli),
        Command::HybridDetector(cli) => hybrid_detector::run(cli),
        Command::L2DelayBaseline(cli) => l2_delay_baseline::run(cli),
        Command::LagDepthSweep(cli) => lag_depth_sweep::run(cli),
        #[cfg(feature = "gpu")]
        Command::LbmCubeRun(cli) => lbm_cube_run::run(cli),
        Command::MagneticTakens(cli) => magnetic_takens::run(cli),
        Command::MavenCrossingGen(cli) => maven_crossing_gen::run(cli),
        Command::MmsMultiday(cli) => mms_multiday::run(cli),
        Command::MmsSitlLabeled(cli) => mms_sitl_labeled::run(cli),
        Command::NormDump(cli) => {
            norm_dump::run(cli);
            Ok(())
        }
        Command::OrderSurplus(cli) => order_surplus::run(cli),
        Command::PcaVarianceBaseline(cli) => pca_variance_baseline::run(cli),
        Command::PredictiveEval(cli) => predictive_eval::run(cli),
        Command::PspAlfvenControl(cli) => psp_alfven_control::run(cli),
        Command::PspEncounterSummary(cli) => psp_encounter_summary::run(cli),
        Command::PspSwitchbackCorrelation(cli) => psp_switchback_correlation::run(cli),
        Command::Q2Transverse(cli) => q2_transverse::run(cli),
        Command::QuenchScan(cli) => quench_scan::run(cli),
        Command::R16Ablation(cli) => r16_ablation::run(cli),
        Command::R64Ablation(cli) => r64_ablation::run(cli),
        Command::RandomTrilinear(cli) => random_trilinear::run(cli),
        Command::RandomTrilinearSparse(cli) => random_trilinear_sparse::run(cli),
        Command::RosettaAblationSummary(cli) => rosetta_ablation_summary::run(cli),
        Command::RosettaDraping(cli) => rosetta_draping::run(cli),
        Command::SelfsimLogstats(cli) => selfsim_logstats::run(cli),
        Command::SparsePolicyMainline(cli) => sparse_policy_mainline::run(cli),
        #[cfg(feature = "gpu")]
        Command::SparsePreservation(cli) => sparse_preservation::run(cli),
        Command::SwitchbackOmega(cli) => switchback_omega::run(cli),
        Command::SyntheticMhd(cli) => {
            synthetic_mhd::run(cli);
            Ok(())
        }
        Command::TakensTauSweep(cli) => takens_tau_sweep::run(cli),
        Command::TemporalOverlay(cli) => temporal_overlay::run(cli),
        Command::ThemisGeneralBenchmark(cli) => themis_general_benchmark::run(cli),
        Command::ThemisStaplesLabeled(cli) => themis_staples_labeled::run(cli),
        Command::VoyagerHeliopause(cli) => voyager_heliopause::run(cli),
        Command::WindowSensitivity(cli) => window_sensitivity::run(cli),
        Command::ZdAudit => zd_audit::run(),
    }
}
