use algebra_experimental::majorana_braiding::{
    MajoranaFrictionSweepConfig, MajoranaFrictionSweepReport, MajoranaFrictionSweepRow,
    majorana_friction_sweep,
};
use anyhow::{Context, Result, anyhow};
use cosmology_core::{
    CdDimensionParams, HarmonicHaloConfig, SweepConfig, detection_threshold, homotopy_lambda,
    nfw_params_from_mass, sweep_obstruction_coupling, v_circ_nfw, v_circ_with_halos,
};
use gororoba_cli_physics::nonlocal_report::{
    NonlocalBenchmarkReport, build_nonlocal_benchmark_report,
};
use materials_core::{
    M3ProjectionConfig, SyntheticCouplingModel, SyntheticCouplingReport, find_calibration_record,
    load_calibration_records,
};
use optics_core::{
    absorber_benchmark::{CouplingTopology, ProjectionGate, run_benchmark},
    tcmt::KerrCavity,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const DEFAULT_CALIBRATION_ID: &str = "nonlocal_cable_chen_2023";
pub const DEFAULT_CALIBRATION_CSV: &str = "data/csv/c010_nonlocal_material_calibrations.csv";
const DEFAULT_OUTPUT_DIR: &str = "data/evidence/thesis_42_support";
const HARMONIC_REFERENCE_POINTS: usize = 64;
const HARMONIC_REFERENCE_M200_SOLAR: f64 = 1.0e12;
const HARMONIC_REFERENCE_Z: f64 = 0.0;
const GRAVASTAR_OBSTRUCTION_NORM: f64 = 8.725;
const GRAVASTAR_COUPLING_MIN: f64 = 0.0;
const GRAVASTAR_COUPLING_MAX: f64 = 0.01;
const GRAVASTAR_SWEEP_STEPS: usize = 9;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceSource {
    pub citation_key: String,
    pub citation_url: String,
    pub source_kind: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceBoundary {
    pub physical_claim_status: String,
    pub model_claim_status: String,
    pub theorem_scope_status: String,
    pub assumption_surface: String,
}

#[derive(Debug, Clone)]
pub struct Thesis42SupportConfig {
    pub calibration_id: String,
    pub calibration_csv: PathBuf,
    pub dim: usize,
    pub theta_steps: usize,
    pub cp_phase_rad: f64,
    pub alpha_zd: f64,
    pub output_dir: PathBuf,
}

impl Default for Thesis42SupportConfig {
    fn default() -> Self {
        Self {
            calibration_id: DEFAULT_CALIBRATION_ID.to_string(),
            calibration_csv: PathBuf::from(DEFAULT_CALIBRATION_CSV),
            dim: 16,
            theta_steps: 20,
            cp_phase_rad: 0.0,
            alpha_zd: 0.01,
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Thesis42SupportLabels {
    pub nonlocal_metamaterial: String,
    pub majorana: String,
    pub dark_matter: String,
    pub gravastar: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetamaterialLaneReport {
    pub support_label: String,
    pub disposition: String,
    pub boundary: EvidenceBoundary,
    pub sources: Vec<EvidenceSource>,
    pub fixed_dim: usize,
    pub synthetic_coupling: SyntheticCouplingReport,
    pub benchmark: NonlocalBenchmarkReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MajoranaLaneReport {
    pub support_label: String,
    pub disposition: String,
    pub boundary: EvidenceBoundary,
    pub sources: Vec<EvidenceSource>,
    pub friction_sweep: MajoranaFrictionSweepReport,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DetectionThresholdRow {
    pub sample_label: String,
    pub n_curves: usize,
    pub sigma_v_frac: f64,
    pub alpha_zd_min: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HarmonicHaloReferenceRow {
    pub r_kpc: f64,
    pub v_circ_nfw_km_s: f64,
    pub v_circ_alpha_zero_km_s: f64,
    pub v_circ_probe_km_s: f64,
    pub delta_alpha_zero_percent: f64,
    pub delta_probe_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DarkMatterLaneReport {
    pub support_label: String,
    pub disposition: String,
    pub boundary: EvidenceBoundary,
    pub sources: Vec<EvidenceSource>,
    pub cd_dim: usize,
    pub n_modes: usize,
    pub alpha_zd_probe: f64,
    pub exact_nfw_recovery: bool,
    pub max_abs_delta_alpha_zero_percent: f64,
    pub max_abs_delta_probe_percent: f64,
    pub ska_forecast_note: String,
    pub analytic_thresholds: Vec<DetectionThresholdRow>,
    pub reference_curve: Vec<HarmonicHaloReferenceRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GravastarBridgeRow {
    pub coupling: f64,
    pub lambda: f64,
    pub solution_exists: bool,
    pub stable: bool,
    pub causal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GravastarBridgeModelReport {
    pub law_label: String,
    pub obstruction_norm: f64,
    pub coupling_min: f64,
    pub coupling_max: f64,
    pub n_steps: usize,
    pub zero_coupling_lambda: f64,
    pub stable_solution_count: usize,
    pub causal_solution_count: usize,
    pub stability_window_min: Option<f64>,
    pub stability_window_max: Option<f64>,
    pub rows: Vec<GravastarBridgeRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GravastarLaneReport {
    pub support_label: String,
    pub bridge_supported: bool,
    pub boundary: EvidenceBoundary,
    pub sources: Vec<EvidenceSource>,
    pub bridge_model: GravastarBridgeModelReport,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Thesis42SupportReport {
    pub generated_at_utc: String,
    pub evidence_posture: String,
    pub labels: Thesis42SupportLabels,
    pub metamaterial: MetamaterialLaneReport,
    pub majorana: MajoranaLaneReport,
    pub dark_matter: DarkMatterLaneReport,
    pub gravastar: GravastarLaneReport,
}

pub fn generate_thesis_42_support_report(
    config: &Thesis42SupportConfig,
) -> Result<Thesis42SupportReport> {
    if config.dim < 16 || !config.dim.is_power_of_two() {
        return Err(anyhow!("--dim must be a power of two >= 16"));
    }
    if config.theta_steps == 0 {
        return Err(anyhow!("--theta-steps must be > 0"));
    }

    let metamaterial = build_metamaterial_lane(config)?;
    let majorana = build_majorana_lane(config);
    let dark_matter = build_dark_matter_lane(config);
    let gravastar = build_gravastar_lane();

    Ok(Thesis42SupportReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        evidence_posture: "evidence_first".to_string(),
        labels: Thesis42SupportLabels {
            nonlocal_metamaterial: "design_stage_only".to_string(),
            majorana: "algebraic_not_physical".to_string(),
            dark_matter: "falsifiable_observable_lane".to_string(),
            gravastar: "gravastar_bridge_unsupported".to_string(),
        },
        metamaterial,
        majorana,
        dark_matter,
        gravastar,
    })
}

pub fn render_thesis_42_support_report_toml(report: &Thesis42SupportReport) -> Result<String> {
    toml::to_string_pretty(report).context("failed to serialize thesis-42 support report")
}

pub fn write_thesis_42_support_bundle(
    output_dir: &Path,
    report: &Thesis42SupportReport,
) -> Result<()> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let summary = render_thesis_42_support_report_toml(report)?;
    fs::write(output_dir.join("summary.toml"), summary)
        .with_context(|| format!("failed to write {}", output_dir.display()))?;

    write_nonlocal_topologies_csv(output_dir, &report.metamaterial.benchmark)?;
    write_majorana_sweep_csv(output_dir, &report.majorana.friction_sweep.rows)?;
    write_harmonic_reference_csv(output_dir, &report.dark_matter.reference_curve)?;
    write_gravastar_bridge_csv(output_dir, &report.gravastar.bridge_model.rows)?;
    Ok(())
}

fn build_metamaterial_lane(config: &Thesis42SupportConfig) -> Result<MetamaterialLaneReport> {
    let records = load_calibration_records(&config.calibration_csv)
        .map_err(|err| anyhow!(err))
        .with_context(|| format!("failed to load {}", config.calibration_csv.display()))?;
    let calibration = find_calibration_record(&records, &config.calibration_id)
        .map_err(|err| anyhow!(err))
        .with_context(|| format!("missing calibration {}", config.calibration_id))?;
    let model =
        SyntheticCouplingModel::build(&M3ProjectionConfig::c010_hybrid_default(), calibration)
            .map_err(|err| anyhow!(err))
            .context("failed to build nonlocal synthetic coupling model")?;
    let average_coupling = model.average_nonzero_coupling();
    let suite = vec![
        model.to_coupling_topology("thesis-42-nonlocal-floquet"),
        CouplingTopology::ring(42, average_coupling),
        CouplingTopology::chain(42, average_coupling),
        CouplingTopology::complete(42, average_coupling),
        CouplingTopology::cross_cluster_bridges(7, 6, average_coupling),
    ];
    let benchmark = run_benchmark(
        &suite,
        &benchmark_cavity(),
        1.0e9,
        &[0, 1, 2, 3, 4, 5],
        1.0e-3,
        1.0e-12,
        500,
    );
    let gate = ProjectionGate::c010_default().evaluate(&benchmark);
    let benchmark_report =
        build_nonlocal_benchmark_report(&model, &benchmark, &gate, average_coupling);

    Ok(MetamaterialLaneReport {
        support_label: "design_stage_only".to_string(),
        disposition: format!(
            "C-010 remains a closed local negative result; this lane only reports the cited non-local recovery design stage under {:?}.",
            gate.verdict
        ),
        boundary: EvidenceBoundary {
            physical_claim_status: "refuted_or_unsupported".to_string(),
            model_claim_status: "design_stage_only".to_string(),
            theorem_scope_status: "algebra_topology_only".to_string(),
            assumption_surface: "The repo only models a masked 7 x K6 assessor topology lifted through m3 into LC/Floquet-style synthetic couplers. No exact fabricated 42-assessor device is claimed.".to_string(),
        },
        sources: metamaterial_sources(),
        fixed_dim: 16,
        synthetic_coupling: model.report(),
        benchmark: benchmark_report,
    })
}

fn build_majorana_lane(config: &Thesis42SupportConfig) -> MajoranaLaneReport {
    let friction_sweep = majorana_friction_sweep(&MajoranaFrictionSweepConfig {
        dim: config.dim,
        theta_steps: config.theta_steps,
        cp_phase_bias_rad: config.cp_phase_rad,
    });
    MajoranaLaneReport {
        support_label: "algebraic_not_physical".to_string(),
        disposition: "Majorana fusion and complex-time friction are reported as Cayley-Dickson algebra invariants only; no Hamiltonian or antimatter simulation claim is made.".to_string(),
        boundary: EvidenceBoundary {
            physical_claim_status: "refuted_or_unsupported".to_string(),
            model_claim_status: "verified_model_under_assumptions".to_string(),
            theorem_scope_status: "algebra_and_bridge_law_only".to_string(),
            assumption_surface: "Braids are mapped from Clifford/Majorana generators into Cayley-Dickson basis rotations and associator bookkeeping. This is an algebraic analog lane, not a particle-antiparticle identification.".to_string(),
        },
        sources: majorana_sources(),
        friction_sweep,
        note: "cp_phase_rad is treated as an algebraic phase-bias control for phenomenological sweeps. It does not instantiate physical CP violation.".to_string(),
    }
}

fn build_dark_matter_lane(config: &Thesis42SupportConfig) -> DarkMatterLaneReport {
    let cd_params = CdDimensionParams::new(config.dim);
    let nfw = nfw_params_from_mass(HARMONIC_REFERENCE_M200_SOLAR, HARMONIC_REFERENCE_Z);
    let c200 = nfw.c200;
    let r_s_kpc = nfw.r200_kpc / c200;
    let zero_cfg = HarmonicHaloConfig::new_cd(0.0, cd_params.n_modes, r_s_kpc, config.dim);
    let probe_cfg =
        HarmonicHaloConfig::new_cd(config.alpha_zd, cd_params.n_modes, r_s_kpc, config.dim);
    let reference_curve = harmonic_reference_curve(c200, &zero_cfg, &probe_cfg);

    let max_abs_delta_alpha_zero_percent = reference_curve
        .iter()
        .map(|row| row.delta_alpha_zero_percent.abs())
        .fold(0.0_f64, f64::max);
    let max_abs_delta_probe_percent = reference_curve
        .iter()
        .map(|row| row.delta_probe_percent.abs())
        .fold(0.0_f64, f64::max);
    let analytic_thresholds = vec![
        DetectionThresholdRow {
            sample_label: "analytic_n175".to_string(),
            n_curves: 175,
            sigma_v_frac: 0.05,
            alpha_zd_min: detection_threshold(175, 0.05),
        },
        DetectionThresholdRow {
            sample_label: "analytic_n2500".to_string(),
            n_curves: 2500,
            sigma_v_frac: 0.05,
            alpha_zd_min: detection_threshold(2500, 0.05),
        },
        DetectionThresholdRow {
            sample_label: "analytic_n10000".to_string(),
            n_curves: 10_000,
            sigma_v_frac: 0.05,
            alpha_zd_min: detection_threshold(10_000, 0.05),
        },
    ];

    DarkMatterLaneReport {
        support_label: "falsifiable_observable_lane".to_string(),
        disposition: "The dark-matter-facing thesis support lane uses harmonic-halo/NFW observables. alpha_zd=0 must recover standard NFW exactly, while nonzero alpha_zd remains a falsifiable modulation probe rather than a verified detection.".to_string(),
        boundary: EvidenceBoundary {
            physical_claim_status: "refuted_or_unsupported".to_string(),
            model_claim_status: "falsifiable_observable_lane".to_string(),
            theorem_scope_status: "exact_recovery_and_model_law_only".to_string(),
            assumption_surface: "The repo treats harmonic halos as a modulation of standard NFW dynamics parameterized by alpha_zd. The model is observationally testable but does not establish algebraic dark-matter microphysics.".to_string(),
        },
        sources: dark_matter_sources(),
        cd_dim: config.dim,
        n_modes: cd_params.n_modes,
        alpha_zd_probe: config.alpha_zd,
        exact_nfw_recovery: max_abs_delta_alpha_zero_percent < 1.0e-12,
        max_abs_delta_alpha_zero_percent,
        max_abs_delta_probe_percent,
        ska_forecast_note: "SKAO is treated as a future forecast lane only. The analytic thresholds below describe the approximate alpha_zd floor needed for stacked rotation-curve detection; they are not present-day evidence claims.".to_string(),
        analytic_thresholds,
        reference_curve,
    }
}

fn build_gravastar_lane() -> GravastarLaneReport {
    let sweep = sweep_obstruction_coupling(&SweepConfig {
        r1: 5.0,
        m_target: 10.0,
        compactness: 0.6,
        gamma: 1.5,
        obstruction_norm: GRAVASTAR_OBSTRUCTION_NORM,
        coupling_min: GRAVASTAR_COUPLING_MIN,
        coupling_max: GRAVASTAR_COUPLING_MAX,
        n_steps: GRAVASTAR_SWEEP_STEPS,
    });
    let stability_window_min = sweep.stability_window.map(|(lo, _)| lo);
    let stability_window_max = sweep.stability_window.map(|(_, hi)| hi);
    let rows = sweep
        .couplings
        .iter()
        .enumerate()
        .map(|(idx, &coupling)| GravastarBridgeRow {
            coupling,
            lambda: homotopy_lambda(GRAVASTAR_OBSTRUCTION_NORM, coupling),
            solution_exists: sweep.solutions[idx].is_some(),
            stable: sweep.stable[idx],
            causal: sweep.causal[idx],
        })
        .collect::<Vec<_>>();
    let stable_solution_count = rows.iter().filter(|row| row.stable).count();
    let causal_solution_count = rows.iter().filter(|row| row.causal).count();

    GravastarLaneReport {
        support_label: "gravastar_bridge_unsupported".to_string(),
        bridge_supported: false,
        boundary: EvidenceBoundary {
            physical_claim_status: "closed_obstructed".to_string(),
            model_claim_status: "verified_model_under_assumptions".to_string(),
            theorem_scope_status: "explicit_bridge_law_only".to_string(),
            assumption_surface: "The current gravastar lane assumes the linear bridge law lambda = coupling * obstruction_norm and studies its TOV consequences. This does not derive a physical algebra-to-GR stress-energy bridge.".to_string(),
        },
        sources: gravastar_sources(),
        bridge_model: GravastarBridgeModelReport {
            law_label: "linear_homotopy_lambda".to_string(),
            obstruction_norm: GRAVASTAR_OBSTRUCTION_NORM,
            coupling_min: GRAVASTAR_COUPLING_MIN,
            coupling_max: GRAVASTAR_COUPLING_MAX,
            n_steps: GRAVASTAR_SWEEP_STEPS,
            zero_coupling_lambda: homotopy_lambda(GRAVASTAR_OBSTRUCTION_NORM, 0.0),
            stable_solution_count,
            causal_solution_count,
            stability_window_min,
            stability_window_max,
            rows,
        },
        note: "The repo currently supports an internal, assumption-labeled bridge-law audit. It does not support a derived physical gravastar bridge, so C-011 remains obstructed even when the linear model has stable solutions.".to_string(),
    }
}

fn harmonic_reference_curve(
    c200: f64,
    zero_cfg: &HarmonicHaloConfig,
    probe_cfg: &HarmonicHaloConfig,
) -> Vec<HarmonicHaloReferenceRow> {
    let log_r_min = 0.1_f64.ln();
    let log_r_max = 100.0_f64.ln();

    (0..HARMONIC_REFERENCE_POINTS)
        .map(|idx| {
            let frac = idx as f64 / (HARMONIC_REFERENCE_POINTS - 1) as f64;
            let r_kpc = (log_r_min + frac * (log_r_max - log_r_min)).exp();
            let v_nfw = v_circ_nfw(r_kpc, HARMONIC_REFERENCE_M200_SOLAR, HARMONIC_REFERENCE_Z);
            let v_zero = v_circ_with_halos(
                r_kpc,
                HARMONIC_REFERENCE_M200_SOLAR,
                c200,
                HARMONIC_REFERENCE_Z,
                zero_cfg,
            );
            let v_probe = v_circ_with_halos(
                r_kpc,
                HARMONIC_REFERENCE_M200_SOLAR,
                c200,
                HARMONIC_REFERENCE_Z,
                probe_cfg,
            );
            let delta_alpha_zero_percent = if v_nfw > 0.0 {
                (v_zero - v_nfw) / v_nfw * 100.0
            } else {
                0.0
            };
            let delta_probe_percent = if v_nfw > 0.0 {
                (v_probe - v_nfw) / v_nfw * 100.0
            } else {
                0.0
            };
            HarmonicHaloReferenceRow {
                r_kpc,
                v_circ_nfw_km_s: v_nfw,
                v_circ_alpha_zero_km_s: v_zero,
                v_circ_probe_km_s: v_probe,
                delta_alpha_zero_percent,
                delta_probe_percent,
            }
        })
        .collect()
}

fn benchmark_cavity() -> KerrCavity {
    KerrCavity::new(
        2.0 * std::f64::consts::PI * 193.0e12,
        1.0e6,
        5.0e5,
        1.45,
        2.6e-20,
        1.0e-18,
    )
}

fn majorana_sources() -> Vec<EvidenceSource> {
    vec![
        EvidenceSource {
            citation_key: "DasSarma2015".to_string(),
            citation_url: "https://doi.org/10.1038/npjqi.2015.1".to_string(),
            source_kind: "primary_review".to_string(),
            note: "Majorana zero modes are treated as condensed-matter topological objects and engineering targets for quantum computation, not as evidence for direct antimatter physics.".to_string(),
        },
        EvidenceSource {
            citation_key: "Ivanov2001".to_string(),
            citation_url: "https://doi.org/10.1103/PhysRevLett.86.268".to_string(),
            source_kind: "primary_theory".to_string(),
            note: "Non-Abelian statistics for Majorana-carrying vortices; used here as the braid/fusion analog anchor.".to_string(),
        },
        EvidenceSource {
            citation_key: "Kitaev2001".to_string(),
            citation_url: "https://doi.org/10.1070/1063-7869/44/10S/S29".to_string(),
            source_kind: "primary_theory".to_string(),
            note: "Foundational unpaired-Majorana/topological-computation reference already mirrored in the code comments.".to_string(),
        },
    ]
}

fn metamaterial_sources() -> Vec<EvidenceSource> {
    vec![
        EvidenceSource {
            citation_key: "Wang2020".to_string(),
            citation_url: "https://doi.org/10.1038/s41467-020-15940-3".to_string(),
            source_kind: "primary_experiment".to_string(),
            note: "Topolectrical realization of higher-dimensional synthetic topology; supports the non-local wiring lane rather than exact 42-assessor fabrication.".to_string(),
        },
        EvidenceSource {
            citation_key: "Yuan2017".to_string(),
            citation_url: "https://arxiv.org/abs/1710.01373".to_string(),
            source_kind: "primary_theory".to_string(),
            note: "Synthetic-frequency Floquet dimensions provide the modulation-side backend for the C-010 recovery lane.".to_string(),
        },
        EvidenceSource {
            citation_key: "Dutt2022".to_string(),
            citation_url: "https://doi.org/10.1038/s41467-022-31140-7".to_string(),
            source_kind: "primary_experiment".to_string(),
            note: "Boundary creation in synthetic frequency space supports explicit disconnected-sector engineering.".to_string(),
        },
        EvidenceSource {
            citation_key: "Chen2023".to_string(),
            citation_url: "https://doi.org/10.1002/adma.202209988".to_string(),
            source_kind: "primary_experiment".to_string(),
            note: "Cable-network metamaterials demonstrate genuine non-local interaction engineering and currently anchor the strongest passing thesis-bundle default.".to_string(),
        },
    ]
}

fn dark_matter_sources() -> Vec<EvidenceSource> {
    vec![
        EvidenceSource {
            citation_key: "NavarroFrenkWhite1997".to_string(),
            citation_url: "https://doi.org/10.1086/304888".to_string(),
            source_kind: "primary_theory".to_string(),
            note: "Standard NFW halo profile remains the exact baseline recovered when alpha_zd = 0.".to_string(),
        },
        EvidenceSource {
            citation_key: "Li2020SPARC".to_string(),
            citation_url: "https://doi.org/10.3847/1538-4365/ab700e".to_string(),
            source_kind: "primary_dataset".to_string(),
            note: "Repo-backed SPARC/NFW fits anchor the observational residual lane used for harmonic-halo stacking.".to_string(),
        },
        EvidenceSource {
            citation_key: "SKAOHIGalaxyScience".to_string(),
            citation_url: "https://www.skao.int/en/science-users/118/hi-galaxy-science".to_string(),
            source_kind: "science_case".to_string(),
            note: "SKAO is used as a future observability target for stacked rotation-curve forecasts, not as current evidence.".to_string(),
        },
    ]
}

fn gravastar_sources() -> Vec<EvidenceSource> {
    vec![
        EvidenceSource {
            citation_key: "MazurMottola2001".to_string(),
            citation_url: "https://arxiv.org/abs/gr-qc/0109035".to_string(),
            source_kind: "primary_theory".to_string(),
            note: "Original gravastar proposal; anchors the physical literature baseline rather than the repo bridge law.".to_string(),
        },
        EvidenceSource {
            citation_key: "MazurMottola2004".to_string(),
            citation_url: "https://doi.org/10.1073/pnas.0402717101".to_string(),
            source_kind: "primary_theory".to_string(),
            note: "Three-region gravastar structure used as the literature-grounded reference configuration.".to_string(),
        },
        EvidenceSource {
            citation_key: "VisserWiltshire2004".to_string(),
            citation_url: "https://arxiv.org/abs/gr-qc/0310107".to_string(),
            source_kind: "primary_theory".to_string(),
            note: "Thin-shell/dynamical stability benchmark for evaluating whether a repo bridge law is only phenomenological or physically admissible.".to_string(),
        },
        EvidenceSource {
            citation_key: "BowersLiang1974".to_string(),
            citation_url: "https://doi.org/10.1086/152638".to_string(),
            source_kind: "primary_theory".to_string(),
            note: "Anisotropic TOV framework underlying the current lambda bridge law.".to_string(),
        },
    ]
}

fn write_nonlocal_topologies_csv(
    output_dir: &Path,
    report: &NonlocalBenchmarkReport,
) -> Result<()> {
    let path = output_dir.join("nonlocal_topologies.csv");
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(&path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    writer.write_record([
        "name",
        "isolation_ratio",
        "crosstalk_db",
        "spectral_gap",
        "n_components",
    ])?;
    for row in &report.topologies {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_majorana_sweep_csv(output_dir: &Path, rows: &[MajoranaFrictionSweepRow]) -> Result<()> {
    let path = output_dir.join("majorana_friction_sweep.csv");
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(&path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    writer.write_record([
        "theta_rad",
        "tau_seconds",
        "raw_friction",
        "normalized_friction",
        "max_associator_norm",
        "phase_bias_factor",
    ])?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_harmonic_reference_csv(
    output_dir: &Path,
    rows: &[HarmonicHaloReferenceRow],
) -> Result<()> {
    let path = output_dir.join("harmonic_halo_reference.csv");
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(&path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    writer.write_record([
        "r_kpc",
        "v_circ_nfw_km_s",
        "v_circ_alpha_zero_km_s",
        "v_circ_probe_km_s",
        "delta_alpha_zero_percent",
        "delta_probe_percent",
    ])?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_gravastar_bridge_csv(output_dir: &Path, rows: &[GravastarBridgeRow]) -> Result<()> {
    let path = output_dir.join("gravastar_bridge_model.csv");
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(&path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    writer.write_record(["coupling", "lambda", "solution_exists", "stable", "causal"])?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}
