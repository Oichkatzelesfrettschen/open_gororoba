use std::{
    collections::{BTreeMap, BTreeSet},
    f64::consts::PI,
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use materials_core::{
    E_CHARGE, FiniteTemperatureOptions, HalfSpace, HighFrequencyCompletion, K_B_EV, Layer,
    LifshitzModel, Multilayer, ZeroTemperatureOptions, ev_to_omega, get_material, gold_drude,
    local_drude_finite_temperature_pressure, local_drude_zero_mode_pressure,
    silica_casimir_optical, zero_temperature_energy_per_area, zero_temperature_pressure,
};
use nalgebra::{DMatrix, DVector};
use quantum_core::{
    casimir::{
        C, DielectricModel, HBAR, LifshitzQuadratureOptions, lifshitz_energy_plates,
        lifshitz_energy_plates_with_options, lifshitz_pressure_plates_with_options,
    },
    channel_admissibility::{
        exponential_memory_depolarizing_eigenvalue, qubit_depolarizing_minimum_choi_eigenvalue,
    },
    lifshitz_force_sphere_plate, lifshitz_pressure_plates,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use statrs::distribution::{ContinuousCDF, Normal};
use stats_core::calibrated_discrimination::{
    NuisanceBound, bounded_profile_distance, certified_distance_lower_bound, efficient_information,
    fisher_information_with_pseudoinverse, unrestricted_residual,
};

const DEFAULT_OUTPUT_DIRECTORY: &str = "data/output/audit/casimir-optics-discrimination";
const GAP_METERS: f64 = 200.0e-9;
const SPHERE_RADIUS_METERS: f64 = 100.0e-6;
const APERY_CONSTANT: f64 = 1.202_056_903_159_594;
const PLANAR_CONVERGENCE_RELATIVE_TOLERANCE: f64 = 2.0e-9;
const MULTILAYER_ANGULAR_CONVERGENCE_RELATIVE_TOLERANCE: f64 = 1.0e-8;
const ANGULAR_CONVERGENCE_TARGET: &str = "Au20/SiO2-50/Al2O3-50/Si_vs_Au";
const PRODUCER_SOURCE: &str = include_str!("casimir_optics_discrimination_audit.rs");
const SOURCE_RETRIEVAL_MANIFEST: &str = "source-retrieval-manifest.toml";

struct SourceModelInput {
    path: &'static str,
    bytes: &'static [u8],
}

const SOURCE_MODEL_INPUTS: [SourceModelInput; 16] = [
    SourceModelInput {
        path: "crates/gororoba_cli_physics/src/bin/casimir_optics_discrimination_audit.rs",
        bytes: include_bytes!("casimir_optics_discrimination_audit.rs"),
    },
    SourceModelInput {
        path: "crates/materials_core/src/multilayer_lifshitz.rs",
        bytes: include_bytes!("../../../materials_core/src/multilayer_lifshitz.rs"),
    },
    SourceModelInput {
        path: "crates/materials_core/src/optical_database.rs",
        bytes: include_bytes!("../../../materials_core/src/optical_database.rs"),
    },
    SourceModelInput {
        path: "crates/materials_core/src/optical_database/metals_dl.rs",
        bytes: include_bytes!("../../../materials_core/src/optical_database/metals_dl.rs"),
    },
    SourceModelInput {
        path: "crates/materials_core/src/optical_database/oxides_tcos.rs",
        bytes: include_bytes!("../../../materials_core/src/optical_database/oxides_tcos.rs"),
    },
    SourceModelInput {
        path: "crates/materials_core/src/optical_database/semiconductors.rs",
        bytes: include_bytes!("../../../materials_core/src/optical_database/semiconductors.rs"),
    },
    SourceModelInput {
        path: "crates/materials_core/src/optical_database/thin_film_coating.rs",
        bytes: include_bytes!(
            "../../../materials_core/src/optical_database/thin_film_coating.rs"
        ),
    },
    SourceModelInput {
        path: "crates/materials_core/src/optical_database/tungstates.rs",
        bytes: include_bytes!("../../../materials_core/src/optical_database/tungstates.rs"),
    },
    SourceModelInput {
        path: "crates/materials_data/build.rs",
        bytes: include_bytes!("../../../materials_data/build.rs"),
    },
    SourceModelInput {
        path: "crates/materials_data/data/optical/drude_metals.toml",
        bytes: include_bytes!("../../../materials_data/data/optical/drude_metals.toml"),
    },
    SourceModelInput {
        path: "crates/materials_data/data/optical/lorentz_models.toml",
        bytes: include_bytes!("../../../materials_data/data/optical/lorentz_models.toml"),
    },
    SourceModelInput {
        path: "crates/materials_data/src/lib.rs",
        bytes: include_bytes!("../../../materials_data/src/lib.rs"),
    },
    SourceModelInput {
        path: "crates/quantum_core/src/casimir.rs",
        bytes: include_bytes!("../../../quantum_core/src/casimir.rs"),
    },
    SourceModelInput {
        path: "crates/quantum_core/src/casimir/lifshitz.rs",
        bytes: include_bytes!("../../../quantum_core/src/casimir/lifshitz.rs"),
    },
    SourceModelInput {
        path: "crates/quantum_core/src/channel_admissibility.rs",
        bytes: include_bytes!("../../../quantum_core/src/channel_admissibility.rs"),
    },
    SourceModelInput {
        path: "crates/stats_core/src/calibrated_discrimination.rs",
        bytes: include_bytes!("../../../stats_core/src/calibrated_discrimination.rs"),
    },
];

#[derive(Debug, Deserialize)]
struct SourceRetrievalManifest {
    schema_version: u32,
    source: Vec<RetainedSource>,
}

#[derive(Debug, Deserialize)]
struct RetainedSource {
    id: String,
    path: PathBuf,
    sha256: String,
}

#[derive(Debug, Parser)]
#[command(name = "casimir-optics-discrimination-audit")]
#[command(about = "Emit native Casimir, optics, channel, and inference audit evidence")]
struct Arguments {
    /// Destination for deterministic model outputs and their hash manifest.
    #[arg(long, default_value = DEFAULT_OUTPUT_DIRECTORY)]
    output_directory: PathBuf,
    /// Compare generated bytes with the retained artifacts without writing.
    #[arg(long)]
    check: bool,
    /// Write the expected generated bytes for CI artifact retrieval.
    #[arg(long, requires = "check")]
    expected_output_directory: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u32,
    scope: &'static str,
    planar: PlanarReport,
    thermal: ThermalReport,
    optics: OpticalReport,
    channel: ChannelReport,
    inference: InferenceReport,
    multilayer: MultilayerReport,
    physical_inference: PhysicalInferenceReport,
    controls: ControlReport,
}

#[derive(Debug, Serialize)]
struct PlanarReport {
    gap_m: f64,
    sphere_radius_m: f64,
    pressure_pa: f64,
    exact_pressure_pa: f64,
    pressure_relative_error: f64,
    energy_j_m2: f64,
    exact_energy_j_m2: f64,
    energy_relative_error: f64,
    sphere_force_n: f64,
    exact_sphere_force_n: f64,
    sphere_force_relative_error: f64,
    pressure_energy_identity_relative_error: f64,
}

#[derive(Debug, Serialize)]
struct ThermalReport {
    temperature_k: f64,
    drude_tm_zero_mode_energy_j_m2: f64,
    exact_drude_tm_zero_mode_energy_j_m2: f64,
    relative_error: f64,
}

#[derive(Debug, Serialize)]
struct OpticalReport {
    energy_ev: f64,
    wavelength_um: f64,
    wo3_imaginary_epsilon: f64,
    wo3_x_imaginary_epsilon: f64,
    halfspace_delta_reflectance: f64,
    film_100nm_delta_reflectance: f64,
}

#[derive(Debug, Serialize)]
struct ChannelReport {
    kernel_weight: f64,
    kernel_decay: f64,
    time: f64,
    depolarizing_eigenvalue: f64,
    minimum_normalized_choi_eigenvalue: f64,
}

#[derive(Debug, Serialize)]
struct InferenceReport {
    synthetic_uncalibrated_information: f64,
    synthetic_calibrated_information: f64,
    synthetic_unrestricted_distance: f64,
    synthetic_bounded_distance: f64,
    arithmetic_error_margin_lower_bound: f64,
    arithmetic_error_margin_hypothesis_0: f64,
    arithmetic_error_margin_hypothesis_1: f64,
    original_derived_feature_information: f64,
    augmented_derived_feature_information: f64,
}

#[derive(Debug, Serialize)]
struct MultilayerReport {
    au20_stack_pressure_200nm_pa: f64,
    au20_stack_energy_200nm_j_m2: f64,
    pressure_energy_derivative_relative_error: f64,
    maximum_common_cap_zero_mode_difference_pa: f64,
    maximum_uv_completion_relative_change: f64,
}

#[derive(Debug, Serialize)]
struct PhysicalInferenceReport {
    gap_count: usize,
    unrestricted_residual_fraction: f64,
    one_width_bounded_residual_fraction: f64,
    baseline_effective_information: f64,
    strongest_calibration_coordinate: String,
    strongest_calibration_relative_information_gain: f64,
}

#[derive(Debug, Serialize)]
struct ControlReport {
    classical_forward_loop: f64,
    classical_reverse_loop: f64,
    schedule_residual_fraction: f64,
    monte_carlo_seed: u64,
    trials_per_arm: usize,
    normal_critical_value: f64,
    null_false_positive_fraction: f64,
    simulated_power: f64,
    analytic_power: f64,
    matched_null_identity_error: f64,
}

#[derive(Clone, Copy)]
struct AuditRow {
    section: &'static str,
    quantity: &'static str,
    value: f64,
    unit: &'static str,
}

struct GeneratedAudit {
    report: AuditReport,
    tables: BTreeMap<&'static str, String>,
}

struct PhysicalBundle {
    gaps_nm: Vec<f64>,
    target_pa: DVector<f64>,
    mean_pa: DVector<f64>,
    nuisance: DMatrix<f64>,
}

const NUISANCE_NAMES: [&str; 8] = [
    "offset",
    "patch_amplitude_at_100nm",
    "full_differential_gap",
    "full_differential_gain",
    "roughness_variance_difference",
    "full_differential_au_thickness",
    "full_differential_silica_thickness",
    "full_differential_alumina_thickness",
];
const NUISANCE_UNITS: [&str; 8] = ["Pa", "Pa", "nm", "1", "nm^2", "nm", "nm", "nm"];
const NUISANCE_WIDTHS: [f64; 8] = [0.002, 0.010, 0.10, 0.001, 1.0, 0.10, 0.50, 0.50];

fn tsv(header: &str, rows: impl IntoIterator<Item = String>) -> String {
    let mut output = String::from(header);
    output.push('\n');
    for row in rows {
        output.push_str(&row);
        output.push('\n');
    }
    output
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn source_model_identity() -> String {
    let mut digest = Sha256::new();
    for input in &SOURCE_MODEL_INPUTS {
        digest.update(input.path.as_bytes());
        digest.update([0]);
        digest.update(input.bytes);
        digest.update([0]);
    }
    format!("sha256:{}", hex_encode(&digest.finalize()))
}

fn verify_source_retrieval_manifest_source(source: &str, repository_root: &Path) -> Result<()> {
    let manifest: SourceRetrievalManifest =
        toml::from_str(source).context("parsing retained source-retrieval manifest")?;
    ensure!(
        manifest.schema_version == 1,
        "unexpected source-retrieval manifest schema version {}",
        manifest.schema_version
    );
    ensure!(
        !manifest.source.is_empty(),
        "source-retrieval manifest declares zero sources"
    );

    let mut source_ids = BTreeSet::new();
    let mut source_paths = BTreeSet::new();
    for retained_source in manifest.source {
        ensure!(
            !retained_source.id.trim().is_empty(),
            "source-retrieval manifest declares an empty source id"
        );
        ensure!(
            source_ids.insert(retained_source.id.clone()),
            "source-retrieval manifest repeats source id {}",
            retained_source.id
        );
        ensure!(
            !retained_source.path.as_os_str().is_empty()
                && retained_source.path.is_relative()
                && retained_source
                    .path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "source {} declares unsafe repository-relative path {}",
            retained_source.id,
            retained_source.path.display()
        );
        ensure!(
            source_paths.insert(retained_source.path.clone()),
            "source-retrieval manifest repeats source path {}",
            retained_source.path.display()
        );
        ensure!(
            retained_source.sha256.len() == 64
                && retained_source
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "source {} declares malformed SHA-256 {}",
            retained_source.id,
            retained_source.sha256
        );

        let retained_path = repository_root.join(&retained_source.path);
        let retained_bytes = fs::read(&retained_path).with_context(|| {
            format!(
                "reading retained source {} at {}",
                retained_source.id,
                retained_path.display()
            )
        })?;
        let observed_sha256 = sha256_hex(&retained_bytes);
        ensure!(
            observed_sha256 == retained_source.sha256,
            "retained source {} SHA-256 mismatch at {}: declared {}, observed {}",
            retained_source.id,
            retained_source.path.display(),
            retained_source.sha256,
            observed_sha256
        );
    }
    Ok(())
}

fn verify_source_retrieval_manifest(manifest_path: &Path, repository_root: &Path) -> Result<()> {
    let source = fs::read_to_string(manifest_path).with_context(|| {
        format!(
            "reading retained source-retrieval manifest {}",
            manifest_path.display()
        )
    })?;
    verify_source_retrieval_manifest_source(&source, repository_root)
}

fn scaled_columns(matrix: &DMatrix<f64>, scales: &[f64]) -> DMatrix<f64> {
    let mut scaled = matrix.clone();
    for (mut column, scale) in scaled.column_iter_mut().zip(scales) {
        column.scale_mut(*scale);
    }
    scaled
}

fn normalized_least_squares_parameters(
    design: &DMatrix<f64>,
    target: &DVector<f64>,
) -> Result<DVector<f64>> {
    let scales: Vec<f64> = design
        .column_iter()
        .map(|column| column.norm().max(f64::MIN_POSITIVE))
        .collect();
    let normalized = scaled_columns(
        design,
        &scales.iter().map(|scale| 1.0 / scale).collect::<Vec<_>>(),
    );
    let scaled_solution = normalized
        .svd(true, true)
        .solve(target, f64::EPSILON.sqrt())
        .map_err(|_| anyhow!("physical nuisance least-squares solve failed"))?;
    Ok(DVector::from_iterator(
        scaled_solution.len(),
        scaled_solution
            .iter()
            .zip(scales)
            .map(|(value, scale)| value / scale),
    ))
}

#[derive(Clone, Copy)]
struct LifshitzMaterials<'a> {
    gold: &'a materials_core::DrudeLorentzParams,
    silica: &'a materials_core::DrudeLorentzParams,
    alumina: &'a materials_core::DrudeLorentzParams,
    silicon: &'a materials_core::DrudeLorentzParams,
}

#[derive(Clone, Copy)]
struct StackThicknesses {
    cap_nm: f64,
    silica_nm: f64,
    alumina_nm: f64,
}

fn stack_pair<'a>(
    materials: LifshitzMaterials<'a>,
    thicknesses: StackThicknesses,
    completion_ev: Option<f64>,
) -> (Multilayer<'a>, Multilayer<'a>) {
    let model = |params: &'a materials_core::DrudeLorentzParams| {
        let base = LifshitzModel::new(params);
        completion_ev.map_or(base, |energy| {
            base.with_high_frequency_completion(HighFrequencyCompletion::new(energy, 0.0))
        })
    };
    let first = Multilayer::new(
        vec![
            Layer::from_model(model(materials.gold), thicknesses.cap_nm * 1.0e-9),
            Layer::from_model(model(materials.silica), thicknesses.silica_nm * 1.0e-9),
            Layer::from_model(model(materials.alumina), thicknesses.alumina_nm * 1.0e-9),
        ],
        HalfSpace::Material(model(materials.silicon)),
    );
    let reversed = Multilayer::new(
        vec![
            Layer::from_model(model(materials.gold), thicknesses.cap_nm * 1.0e-9),
            Layer::from_model(model(materials.alumina), thicknesses.alumina_nm * 1.0e-9),
            Layer::from_model(model(materials.silica), thicknesses.silica_nm * 1.0e-9),
        ],
        HalfSpace::Material(model(materials.silicon)),
    );
    (first, reversed)
}

fn pair_pressures(
    gap_nm: f64,
    thicknesses: StackThicknesses,
    completion_ev: Option<f64>,
    materials: LifshitzMaterials<'_>,
    options: ZeroTemperatureOptions,
) -> Result<[f64; 2]> {
    let opposing = Multilayer::half_space(materials.gold);
    let (first, reversed) = stack_pair(materials, thicknesses, completion_ev);
    Ok([
        zero_temperature_pressure(gap_nm * 1.0e-9, &opposing, &first, options)?,
        zero_temperature_pressure(gap_nm * 1.0e-9, &opposing, &reversed, options)?,
    ])
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    ((actual - expected) / expected).abs()
}

fn cutoff_convergence_options() -> [LifshitzQuadratureOptions; 3] {
    [16.0, 20.0, 24.0]
        .map(|kappa_cutoff| LifshitzQuadratureOptions::new(kappa_cutoff, 256, 64))
}

fn radial_convergence_options() -> [LifshitzQuadratureOptions; 3] {
    [128, 256, 512]
        .map(|kappa_order| LifshitzQuadratureOptions::new(24.0, kappa_order, 64))
}

fn angular_convergence_options() -> [ZeroTemperatureOptions; 3] {
    [32, 64, 128].map(|angular_order| ZeroTemperatureOptions {
        radial_order: 128,
        angular_order,
    })
}

fn generate_convergence_table(materials: LifshitzMaterials<'_>) -> Result<String> {
    let perfect_conductor = DielectricModel::PerfectConductor;
    let mut rows = Vec::new();

    for (factor, configurations) in [
        ("cutoff", cutoff_convergence_options()),
        ("radial_order", radial_convergence_options()),
    ] {
        let observations = configurations.map(|options| {
            (
                options,
                lifshitz_pressure_plates_with_options(
                    GAP_METERS,
                    &perfect_conductor,
                    &perfect_conductor,
                    options,
                ),
                lifshitz_energy_plates_with_options(
                    GAP_METERS,
                    &perfect_conductor,
                    &perfect_conductor,
                    options,
                ),
            )
        });
        let reference_pressure = observations[2].1;
        let reference_energy = observations[2].2;
        let pressure_change = relative_error(observations[1].1, reference_pressure);
        let energy_change = relative_error(observations[1].2, reference_energy);
        ensure!(
            pressure_change < PLANAR_CONVERGENCE_RELATIVE_TOLERANCE
                && energy_change < PLANAR_CONVERGENCE_RELATIVE_TOLERANCE,
            "planar {factor} refinement exceeded relative tolerance {:.1e}: pressure={pressure_change:.17e}, energy={energy_change:.17e}",
            PLANAR_CONVERGENCE_RELATIVE_TOLERANCE,
        );

        for (options, observed_pressure, observed_energy) in observations {
            for (observable, value, reference) in [
                ("pressure_pa", observed_pressure, reference_pressure),
                ("energy_j_m2", observed_energy, reference_energy),
            ] {
                let relative_change = relative_error(value, reference);
                rows.push(format!(
                    "{factor}\tperfect_conductor_half_spaces\t{:.17e}\t{}\t{}\t{observable}\t{value:.17e}\t{reference:.17e}\t{relative_change:.17e}\t{:.17e}\t{}",
                    options.kappa_cutoff,
                    options.kappa_order,
                    options.angle_order,
                    PLANAR_CONVERGENCE_RELATIVE_TOLERANCE,
                    relative_change < PLANAR_CONVERGENCE_RELATIVE_TOLERANCE,
                ));
            }
        }
    }

    let opposing = Multilayer::half_space(materials.gold);
    let (dispersive_stack, _) = stack_pair(
        materials,
        StackThicknesses {
            cap_nm: 20.0,
            silica_nm: 50.0,
            alumina_nm: 50.0,
        },
        None,
    );
    let mut angular_observations = Vec::new();
    for options in angular_convergence_options() {
        let pressure =
            zero_temperature_pressure(GAP_METERS, &opposing, &dispersive_stack, options)?;
        angular_observations.push((options, pressure));
    }
    let angular_reference = angular_observations[2].1;
    let angular_change = relative_error(angular_observations[1].1, angular_reference);
    ensure!(
        angular_change < MULTILAYER_ANGULAR_CONVERGENCE_RELATIVE_TOLERANCE,
        "dispersive multilayer angular refinement exceeded relative tolerance {:.1e}: pressure={angular_change:.17e}",
        MULTILAYER_ANGULAR_CONVERGENCE_RELATIVE_TOLERANCE,
    );
    for (options, pressure) in angular_observations {
        let relative_change = relative_error(pressure, angular_reference);
        rows.push(format!(
            "angular_order\t{ANGULAR_CONVERGENCE_TARGET}\tna\t{}\t{}\tpressure_pa\t{pressure:.17e}\t{angular_reference:.17e}\t{relative_change:.17e}\t{:.17e}\t{}",
            options.radial_order,
            options.angular_order,
            MULTILAYER_ANGULAR_CONVERGENCE_RELATIVE_TOLERANCE,
            relative_change < MULTILAYER_ANGULAR_CONVERGENCE_RELATIVE_TOLERANCE,
        ));
    }

    Ok(tsv(
        "factor\ttarget\tkappa_cutoff\tradial_order\tangular_order\tobservable\tvalue\treference_value\trelative_change_from_reference\trelative_tolerance\twithin_tolerance",
        rows,
    ))
}

fn inference_error(
    operation: &str,
    error: stats_core::calibrated_discrimination::DiscriminationError,
) -> anyhow::Error {
    anyhow!("{operation} failed: {error:?}")
}

fn physical_bundle(
    cap_nm: f64,
    difference_step_nm: f64,
    options: ZeroTemperatureOptions,
    materials: LifshitzMaterials<'_>,
) -> Result<PhysicalBundle> {
    let gaps_nm: Vec<f64> = (0..=12).map(|index| 100.0 + 25.0 * index as f64).collect();
    let mut target = Vec::with_capacity(gaps_nm.len());
    let mut mean = Vec::with_capacity(gaps_nm.len());
    let mut columns = (0..8)
        .map(|_| Vec::with_capacity(gaps_nm.len()))
        .collect::<Vec<_>>();

    for gap_nm in &gaps_nm {
        let baseline = pair_pressures(
            *gap_nm,
            StackThicknesses {
                cap_nm,
                silica_nm: 50.0,
                alumina_nm: 50.0,
            },
            None,
            materials,
            options,
        )?;
        let gap_above = pair_pressures(
            gap_nm + difference_step_nm,
            StackThicknesses {
                cap_nm,
                silica_nm: 50.0,
                alumina_nm: 50.0,
            },
            None,
            materials,
            options,
        )?;
        let gap_below = pair_pressures(
            gap_nm - difference_step_nm,
            StackThicknesses {
                cap_nm,
                silica_nm: 50.0,
                alumina_nm: 50.0,
            },
            None,
            materials,
            options,
        )?;
        let baseline_mean = 0.5 * (baseline[0] + baseline[1]);
        let above_mean = 0.5 * (gap_above[0] + gap_above[1]);
        let below_mean = 0.5 * (gap_below[0] + gap_below[1]);
        target.push(baseline[0] - baseline[1]);
        mean.push(baseline_mean);
        columns[0].push(1.0);
        columns[1].push((100.0 / gap_nm).powi(2));
        columns[2].push((above_mean - below_mean) / (2.0 * difference_step_nm));
        columns[3].push(baseline_mean);
        columns[4].push(
            (above_mean - 2.0 * baseline_mean + below_mean) / (2.0 * difference_step_nm.powi(2)),
        );

        for (column_index, parameter_index) in (5..8).zip(0..3) {
            let mut above = [cap_nm, 50.0, 50.0];
            let mut below = above;
            above[parameter_index] += difference_step_nm;
            below[parameter_index] -= difference_step_nm;
            let pressure_above = pair_pressures(
                *gap_nm,
                StackThicknesses {
                    cap_nm: above[0],
                    silica_nm: above[1],
                    alumina_nm: above[2],
                },
                None,
                materials,
                options,
            )?;
            let pressure_below = pair_pressures(
                *gap_nm,
                StackThicknesses {
                    cap_nm: below[0],
                    silica_nm: below[1],
                    alumina_nm: below[2],
                },
                None,
                materials,
                options,
            )?;
            columns[column_index].push(
                ((pressure_above[0] - pressure_below[0]) + (pressure_above[1] - pressure_below[1]))
                    / (4.0 * difference_step_nm),
            );
        }
    }

    let mut nuisance = DMatrix::zeros(gaps_nm.len(), columns.len());
    for (column_index, column) in columns.iter().enumerate() {
        nuisance
            .column_mut(column_index)
            .copy_from(&DVector::from_column_slice(column));
    }
    Ok(PhysicalBundle {
        gaps_nm,
        target_pa: DVector::from_vec(target),
        mean_pa: DVector::from_vec(mean),
        nuisance,
    })
}

type ExtendedReports = (
    MultilayerReport,
    PhysicalInferenceReport,
    ControlReport,
    BTreeMap<&'static str, String>,
);

fn generate_extended_reports(materials: LifshitzMaterials<'_>) -> Result<ExtendedReports> {
    let reference_options = ZeroTemperatureOptions {
        radial_order: 128,
        angular_order: 64,
    };
    let physical_options = ZeroTemperatureOptions {
        radial_order: 64,
        angular_order: 32,
    };
    let opposing = Multilayer::half_space(materials.gold);
    let standard_stack = StackThicknesses {
        cap_nm: 20.0,
        silica_nm: 50.0,
        alumina_nm: 50.0,
    };
    let (au20_stack, _) = stack_pair(materials, standard_stack, None);
    let stack_pressure =
        zero_temperature_pressure(200.0e-9, &opposing, &au20_stack, reference_options)?;
    let stack_energy =
        zero_temperature_energy_per_area(200.0e-9, &opposing, &au20_stack, reference_options)?;
    let derivative_step_m = 0.02e-9;
    let energy_below = zero_temperature_energy_per_area(
        200.0e-9 - derivative_step_m,
        &opposing,
        &au20_stack,
        reference_options,
    )?;
    let energy_above = zero_temperature_energy_per_area(
        200.0e-9 + derivative_step_m,
        &opposing,
        &au20_stack,
        reference_options,
    )?;
    let derivative_pressure = -(energy_above - energy_below) / (2.0 * derivative_step_m);
    let derivative_error = relative_error(derivative_pressure, stack_pressure);
    let convergence_table = generate_convergence_table(materials)?;
    let multilayer_table = tsv(
        "gap_nm\tstack\tpressure_pa\tenergy_j_m2\tnegative_energy_derivative_pa\tderivative_relative_error",
        [format!(
            "200\tAu20/SiO2-50/Al2O3-50/Si\t{stack_pressure:.17e}\t{stack_energy:.17e}\t{derivative_pressure:.17e}\t{derivative_error:.17e}"
        )],
    );

    let mut finite_temperature_rows = Vec::new();
    let mut maximum_zero_mode_difference = 0.0_f64;
    for gap_nm in [100.0, 200.0, 400.0] {
        let (first, reversed) = stack_pair(materials, standard_stack, None);
        let zero_t_first =
            zero_temperature_pressure(gap_nm * 1.0e-9, &opposing, &first, reference_options)?;
        let zero_t_reversed =
            zero_temperature_pressure(gap_nm * 1.0e-9, &opposing, &reversed, reference_options)?;
        let minimum_q = 2.0 * PI * K_B_EV * E_CHARGE * 300.0 * gap_nm * 1.0e-9 / (HBAR * C);
        let thermal_options = FiniteTemperatureOptions {
            radial_order: 96,
            matsubara_terms: (18.0 / minimum_q).ceil() as usize,
        };
        let thermal_first = local_drude_finite_temperature_pressure(
            gap_nm * 1.0e-9,
            300.0,
            &opposing,
            &first,
            thermal_options,
        )?;
        let thermal_reversed = local_drude_finite_temperature_pressure(
            gap_nm * 1.0e-9,
            300.0,
            &opposing,
            &reversed,
            thermal_options,
        )?;
        let zero_mode_first =
            local_drude_zero_mode_pressure(gap_nm * 1.0e-9, 300.0, &opposing, &first, 96)?;
        let zero_mode_reversed =
            local_drude_zero_mode_pressure(gap_nm * 1.0e-9, 300.0, &opposing, &reversed, 96)?;
        let zero_mode_difference = zero_mode_first - zero_mode_reversed;
        maximum_zero_mode_difference = maximum_zero_mode_difference.max(zero_mode_difference.abs());
        let zero_t_difference = zero_t_first - zero_t_reversed;
        let thermal_difference = thermal_first - thermal_reversed;
        finite_temperature_rows.push(format!(
            "{gap_nm:.0}\t{zero_t_first:.17e}\t{zero_t_reversed:.17e}\t{zero_t_difference:.17e}\t{thermal_first:.17e}\t{thermal_reversed:.17e}\t{thermal_difference:.17e}\t{:.17e}\t{zero_mode_difference:.17e}",
            thermal_difference / zero_t_difference - 1.0,
        ));
    }
    let finite_temperature_table = tsv(
        "gap_nm\tpressure_a_zero_t_pa\tpressure_b_zero_t_pa\tdelta_zero_t_pa\tpressure_a_300k_pa\tpressure_b_300k_pa\tdelta_300k_pa\tdelta_relative_change\tzero_mode_delta_pa",
        finite_temperature_rows,
    );

    let mut uv_rows = Vec::new();
    let mut maximum_uv_relative_change = 0.0_f64;
    for gap_nm in [100.0, 200.0, 400.0] {
        let baseline = pair_pressures(gap_nm, standard_stack, None, materials, reference_options)?;
        let baseline_difference = baseline[0] - baseline[1];
        uv_rows.push(format!(
            "constant_background\t{gap_nm:.0}\t{baseline_difference:.17e}\t0.00000000000000000e0\thypothetical_model_sensitivity"
        ));
        for completion_ev in [8.0, 12.0, 20.0, 40.0] {
            let completed = pair_pressures(
                gap_nm,
                standard_stack,
                Some(completion_ev),
                materials,
                reference_options,
            )?;
            let difference = completed[0] - completed[1];
            let relative_change = difference / baseline_difference - 1.0;
            maximum_uv_relative_change = maximum_uv_relative_change.max(relative_change.abs());
            uv_rows.push(format!(
                "{completion_ev:.0}\t{gap_nm:.0}\t{difference:.17e}\t{relative_change:.17e}\thypothetical_model_sensitivity"
            ));
        }
    }
    let uv_table = tsv(
        "uv_completion_ev\tgap_nm\tdelta_pressure_pa\trelative_change\tevidence_class",
        uv_rows,
    );

    let bundle = physical_bundle(10.0, 0.1, physical_options, materials)?;
    let scaled_nuisance = scaled_columns(&bundle.nuisance, &NUISANCE_WIDTHS);
    let unrestricted_parameters =
        normalized_least_squares_parameters(&scaled_nuisance, &bundle.target_pa)?;
    let unrestricted_residual_vector =
        &bundle.target_pa - &scaled_nuisance * &unrestricted_parameters;
    let unrestricted_fraction = unrestricted_residual_vector.norm() / bundle.target_pa.norm();
    let mut bounded_rows = Vec::new();
    let mut bounded_parameter_rows = Vec::new();
    let mut one_width_fraction = f64::NAN;
    for multiplier in [1.0, 3.0, 10.0, 100.0] {
        let bounds: Vec<NuisanceBound> = NUISANCE_UNITS
            .iter()
            .map(|unit| NuisanceBound {
                lower: -multiplier,
                upper: multiplier,
                unit: format!("scenario width ({unit})"),
            })
            .collect();
        let result = bounded_profile_distance(&bundle.target_pa, &scaled_nuisance, &bounds)
            .map_err(|error| inference_error("physical bounded profile", error))?;
        let fraction = result.distance / bundle.target_pa.norm();
        if multiplier == 1.0 {
            one_width_fraction = fraction;
        }
        bounded_rows.push(format!(
            "{multiplier:.0}\t{fraction:.17e}\t{:.17e}\t{}\t{}",
            result.distance,
            result.active_lower_bounds.len(),
            result.active_upper_bounds.len(),
        ));
        for (index, parameter) in result.nuisance_parameters.iter().enumerate() {
            bounded_parameter_rows.push(format!(
                "{multiplier:.0}\t{}\t{}\t{parameter:.17e}\t{:.17e}",
                NUISANCE_NAMES[index],
                NUISANCE_UNITS[index],
                parameter * NUISANCE_WIDTHS[index],
            ));
        }
    }
    bounded_rows.push(format!(
        "unrestricted\t{unrestricted_fraction:.17e}\t{:.17e}\t0\t0",
        unrestricted_residual_vector.norm(),
    ));
    let bounded_table = tsv(
        "bound_multiplier\tresidual_fraction\tresidual_norm_pa\tactive_lower_bounds\tactive_upper_bounds",
        bounded_rows,
    );
    let bounded_parameter_table = tsv(
        "bound_multiplier\tparameter\tunit\tparameter_in_scenario_widths\tphysical_parameter",
        bounded_parameter_rows,
    );
    let coefficient_table = tsv(
        "parameter\tunit\tscenario_width\tmimic_in_scenario_widths\tphysical_mimic",
        NUISANCE_NAMES.iter().enumerate().map(|(index, name)| {
            format!(
                "{name}\t{}\t{:.17e}\t{:.17e}\t{:.17e}",
                NUISANCE_UNITS[index],
                NUISANCE_WIDTHS[index],
                unrestricted_parameters[index],
                unrestricted_parameters[index] * NUISANCE_WIDTHS[index],
            )
        }),
    );
    let jacobian_table = tsv(
        "gap_nm\ttarget_pa\tmean_pa\toffset_pa_per_pa\tpatch_pa_per_pa\tgap_pa_per_nm\tgain_pa_per_fraction\troughness_pa_per_nm2\tcap_pa_per_nm\tsilica_pa_per_nm\talumina_pa_per_nm",
        bundle.gaps_nm.iter().enumerate().map(|(row, gap_nm)| {
            format!(
                "{gap_nm:.0}\t{:.17e}\t{:.17e}\t{}",
                bundle.target_pa[row],
                bundle.mean_pa[row],
                (0..bundle.nuisance.ncols())
                    .map(|column| format!("{:.17e}", bundle.nuisance[(row, column)]))
                    .collect::<Vec<_>>()
                    .join("\t"),
            )
        }),
    );

    let target_whitened = &bundle.target_pa / 0.003;
    let nuisance_whitened = &scaled_nuisance / 0.003;
    let calibration = DMatrix::identity(NUISANCE_WIDTHS.len(), NUISANCE_WIDTHS.len());
    let baseline_information =
        efficient_information(&target_whitened, &nuisance_whitened, &calibration)
            .map_err(|error| inference_error("physical calibration information", error))?;
    let mut priority_rows = Vec::new();
    let mut strongest_coordinate = String::new();
    let mut strongest_gain = f64::NEG_INFINITY;
    for index in 0..NUISANCE_NAMES.len() {
        let mut improved_widths = NUISANCE_WIDTHS;
        improved_widths[index] *= 0.5;
        let improved_nuisance = scaled_columns(&bundle.nuisance, &improved_widths) / 0.003;
        let improved_information =
            efficient_information(&target_whitened, &improved_nuisance, &calibration)
                .map_err(|error| inference_error("improved calibration information", error))?;
        let gain = improved_information / baseline_information - 1.0;
        if gain > strongest_gain {
            strongest_gain = gain;
            strongest_coordinate = NUISANCE_NAMES[index].to_owned();
        }
        priority_rows.push(format!(
            "{}\t{}\t{:.17e}\t{:.17e}\t3.00000000000000006e-3\t{baseline_information:.17e}\t{improved_information:.17e}\t{gain:.17e}\tconditional_design_scenario",
            NUISANCE_NAMES[index],
            NUISANCE_UNITS[index],
            NUISANCE_WIDTHS[index],
            improved_widths[index],
        ));
    }
    let calibration_table = tsv(
        "parameter\tunit\tbaseline_width\timproved_width\tpoint_noise_pa\tinformation_before\tinformation_after\trelative_information_gain\tscope",
        priority_rows,
    );

    let mut step_rows = Vec::new();
    for step_nm in [0.025, 0.05, 0.1, 0.2, 0.5] {
        let step_bundle = physical_bundle(10.0, step_nm, physical_options, materials)?;
        let step_scaled = scaled_columns(&step_bundle.nuisance, &NUISANCE_WIDTHS);
        let step_residual = unrestricted_residual(&step_bundle.target_pa, &step_scaled)
            .map_err(|error| inference_error("derivative-step unrestricted profile", error))?;
        let step_bounds: Vec<NuisanceBound> = NUISANCE_UNITS
            .iter()
            .map(|unit| NuisanceBound {
                lower: -1.0,
                upper: 1.0,
                unit: format!("scenario width ({unit})"),
            })
            .collect();
        let step_bounded =
            bounded_profile_distance(&step_bundle.target_pa, &step_scaled, &step_bounds)
                .map_err(|error| inference_error("derivative-step bounded profile", error))?;
        step_rows.push(format!(
            "{step_nm:.3}\t{:.17e}\t{:.17e}",
            step_residual.norm() / step_bundle.target_pa.norm(),
            step_bounded.distance / step_bundle.target_pa.norm(),
        ));
    }
    let derivative_step_table = tsv(
        "central_difference_step_nm\tunrestricted_residual_fraction\tone_width_bounded_residual_fraction",
        step_rows,
    );
    let local_tolerance_table = tsv(
        "au_cap_nm\tgap_nm\ttarget_pressure_pa\tfull_gap_difference_nm_for_10pct\tfull_cap_difference_nm_for_10pct\tfull_gain_difference_for_10pct\troughness_variance_difference_nm2_for_10pct",
        [format!(
            "10\t100\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}",
            bundle.target_pa[0],
            (0.1 * bundle.target_pa[0] / bundle.nuisance[(0, 2)]).abs(),
            (0.1 * bundle.target_pa[0] / bundle.nuisance[(0, 5)]).abs(),
            (0.1 * bundle.target_pa[0] / bundle.nuisance[(0, 3)]).abs(),
            (0.1 * bundle.target_pa[0] / bundle.nuisance[(0, 4)]).abs(),
        )],
    );

    let identity = DMatrix::<f64>::identity(3, 3);
    let mut generator_a = DMatrix::<f64>::zeros(3, 3);
    generator_a[(0, 1)] = 1.0;
    let mut generator_b = DMatrix::<f64>::zeros(3, 3);
    generator_b[(1, 2)] = 1.0;
    let a = 0.2;
    let b = 0.3;
    let forward = (&identity + a * &generator_a)
        * (&identity + b * &generator_b)
        * (&identity - a * &generator_a)
        * (&identity - b * &generator_b);
    let reverse = (&identity + b * &generator_b)
        * (&identity + a * &generator_a)
        * (&identity - b * &generator_b)
        * (&identity - a * &generator_a);
    let initial = DVector::from_vec(vec![0.0, 0.0, 1.0]);
    let classical_forward = (&forward * &initial)[0];
    let classical_reverse = (&reverse * &initial)[0];

    let mut schedule_target = Vec::new();
    let mut schedule_rows = Vec::new();
    for control_a in [-1.0, -0.5, 0.5, 1.0] {
        for control_b in [-1.0, -0.5, 0.5, 1.0] {
            for orientation in [-1.0, 1.0] {
                for block in [-1.0, 0.0, 1.0] {
                    schedule_target.push(orientation * control_a * control_b);
                    schedule_rows.push([
                        1.0,
                        control_a,
                        control_b,
                        control_a * control_b,
                        control_a * control_a,
                        control_b * control_b,
                        block,
                        block * block,
                        orientation * control_a,
                        orientation * control_b,
                        orientation * (control_a * control_a - control_b * control_b),
                        orientation * block,
                        orientation * control_a * block,
                        orientation * control_b * block,
                    ]);
                }
            }
        }
    }
    let schedule_target = DVector::from_vec(schedule_target);
    let schedule_nuisance = DMatrix::from_fn(schedule_rows.len(), 14, |row, column| {
        schedule_rows[row][column]
    });
    let schedule_residual = unrestricted_residual(&schedule_target, &schedule_nuisance)
        .map_err(|error| inference_error("selected schedule projection", error))?;
    let schedule_fraction = schedule_residual.norm() / schedule_target.norm();
    let signal_to_noise = 0.05 * schedule_residual.norm() / 0.08;
    let standard_normal = Normal::new(0.0, 1.0).context("constructing standard normal")?;
    let critical_value = standard_normal.inverse_cdf(0.995);
    let monte_carlo_seed = 20_260_911_u64;
    let trials = 100_000_usize;
    let mut random = ChaCha8Rng::seed_from_u64(monte_carlo_seed);
    let mut false_positives = 0_usize;
    let mut detections = 0_usize;
    for _ in 0..trials {
        let null_draw: f64 = random.sample(StandardNormal);
        let alternative_draw: f64 = random.sample(StandardNormal);
        false_positives += usize::from(null_draw.abs() > critical_value);
        detections += usize::from((alternative_draw + signal_to_noise).abs() > critical_value);
    }
    let false_positive_fraction = false_positives as f64 / trials as f64;
    let simulated_power = detections as f64 / trials as f64;
    let analytic_power = 1.0 - standard_normal.cdf(critical_value - signal_to_noise)
        + standard_normal.cdf(-critical_value - signal_to_noise);

    let mut augmented_target = DVector::zeros(2 * schedule_target.len());
    augmented_target
        .rows_mut(0, schedule_target.len())
        .copy_from(&schedule_target);
    let mut augmented_nuisance =
        DMatrix::zeros(2 * schedule_nuisance.nrows(), schedule_nuisance.ncols());
    augmented_nuisance
        .view_mut((0, 0), schedule_nuisance.shape())
        .copy_from(&schedule_nuisance);
    augmented_nuisance
        .view_mut((schedule_nuisance.nrows(), 0), schedule_nuisance.shape())
        .copy_from(&schedule_nuisance);
    let augmented_residual = unrestricted_residual(&augmented_target, &augmented_nuisance)
        .map_err(|error| inference_error("matched active-sham projection", error))?;
    let matched_identity_reference =
        0.5 * (schedule_target.norm_squared() + schedule_residual.norm_squared());
    let matched_identity_error =
        (augmented_residual.norm_squared() - matched_identity_reference).abs();
    let controls_table = tsv(
        "control\torientation\tvalue\treference\terror\tseed\ttrials\tcritical_value\tfalse_positive_fraction\tsimulated_power\tanalytic_power\tresource_scope",
        [
            format!(
                "classical_commutator\tforward\t{classical_forward:.17e}\t{:.17e}\t{:.17e}\t\t\t\t\t\t\tordinary_classical_control",
                a * b,
                (classical_forward - a * b).abs()
            ),
            format!(
                "classical_commutator\treverse\t{classical_reverse:.17e}\t{:.17e}\t{:.17e}\t\t\t\t\t\t\tordinary_classical_control",
                -a * b,
                (classical_reverse + a * b).abs()
            ),
            format!(
                "source_branch_full_factorial_schedule\t\t{schedule_fraction:.17e}\t1.00000000000000000e0\t{:.17e}\t{monte_carlo_seed}\t{trials}\t{critical_value:.17e}\t{false_positive_fraction:.17e}\t{simulated_power:.17e}\t{analytic_power:.17e}\tindependent_fixed_threshold_validation",
                (schedule_fraction - 1.0).abs()
            ),
            "reported_selected_schedule_unreplayed\t\t9.95361060200000014e-1\t\t\t\t\t2.57582930354890039e0\t1.01500000000000006e-2\t7.57510000000000017e-1\t7.59588000000000019e-1\tmissing_design_commutator_csv".to_owned(),
            format!(
                "matched_active_sham_identity\t\t{:.17e}\t{matched_identity_reference:.17e}\t{matched_identity_error:.17e}\t\t\t\t\t\t\tequal_noise_additional_acquisition",
                augmented_residual.norm_squared()
            ),
        ],
    );

    let mut tables = BTreeMap::new();
    tables.insert("multilayer_pressure.tsv", multilayer_table);
    tables.insert("finite_temperature.tsv", finite_temperature_table);
    tables.insert("uv_model_sensitivity.tsv", uv_table);
    tables.insert("physical_jacobian.tsv", jacobian_table);
    tables.insert("bounded_mimic.tsv", bounded_table);
    tables.insert("bounded_mimic_coefficients.tsv", bounded_parameter_table);
    tables.insert("unbounded_mimic_coefficients.tsv", coefficient_table);
    tables.insert("calibration_priority.tsv", calibration_table);
    tables.insert("derivative_step_study.tsv", derivative_step_table);
    tables.insert("local_tolerances.tsv", local_tolerance_table);
    tables.insert("commutator_and_null_controls.tsv", controls_table);
    tables.insert("planar_convergence.tsv", convergence_table);

    Ok((
        MultilayerReport {
            au20_stack_pressure_200nm_pa: stack_pressure,
            au20_stack_energy_200nm_j_m2: stack_energy,
            pressure_energy_derivative_relative_error: derivative_error,
            maximum_common_cap_zero_mode_difference_pa: maximum_zero_mode_difference,
            maximum_uv_completion_relative_change: maximum_uv_relative_change,
        },
        PhysicalInferenceReport {
            gap_count: bundle.gaps_nm.len(),
            unrestricted_residual_fraction: unrestricted_fraction,
            one_width_bounded_residual_fraction: one_width_fraction,
            baseline_effective_information: baseline_information,
            strongest_calibration_coordinate: strongest_coordinate,
            strongest_calibration_relative_information_gain: strongest_gain,
        },
        ControlReport {
            classical_forward_loop: classical_forward,
            classical_reverse_loop: classical_reverse,
            schedule_residual_fraction: schedule_fraction,
            monte_carlo_seed,
            trials_per_arm: trials,
            normal_critical_value: critical_value,
            null_false_positive_fraction: false_positive_fraction,
            simulated_power,
            analytic_power,
            matched_null_identity_error: matched_identity_error,
        },
        tables,
    ))
}

fn generate_report() -> Result<GeneratedAudit> {
    let perfect_conductor = DielectricModel::PerfectConductor;
    let pressure =
        lifshitz_pressure_plates(GAP_METERS, &perfect_conductor, &perfect_conductor, 256, 64);
    let energy =
        lifshitz_energy_plates(GAP_METERS, &perfect_conductor, &perfect_conductor, 256, 64);
    let sphere_force = lifshitz_force_sphere_plate(
        SPHERE_RADIUS_METERS,
        GAP_METERS,
        &perfect_conductor,
        &perfect_conductor,
        256,
        64,
    );
    let exact_pressure = -PI.powi(2) * HBAR * C / (240.0 * GAP_METERS.powi(4));
    let exact_energy = -PI.powi(2) * HBAR * C / (720.0 * GAP_METERS.powi(3));
    let exact_sphere_force = 2.0 * PI * SPHERE_RADIUS_METERS * exact_energy;
    let gold = get_material("gold").context("gold optical model is unavailable")?;
    let temperature = 300.0;
    let zero_mode = materials_core::casimir_lifshitz_energy(
        &gold.optical,
        &gold.optical,
        GAP_METERS,
        temperature,
        0,
        32,
    );
    let exact_zero_mode =
        -K_B_EV * E_CHARGE * temperature * APERY_CONSTANT / (16.0 * PI * GAP_METERS.powi(2));

    let optical_energy_ev = 0.092_72;
    let optical_frequency = ev_to_omega(optical_energy_ev);
    let wo3 = get_material("wo3").context("WO3 optical model is unavailable")?;
    let wo3_x = get_material("wo3x").context("WO3-x optical model is unavailable")?;
    let silicon = get_material("silicon").context("silicon optical model is unavailable")?;
    let alumina = get_material("alumina").context("alumina optical model is unavailable")?;
    let wo3_epsilon = wo3.optical.epsilon(optical_frequency);
    let wo3_x_epsilon = wo3_x.optical.epsilon(optical_frequency);
    let halfspace_delta = wo3_x.optical.reflectivity_normal(optical_frequency)
        - wo3.optical.reflectivity_normal(optical_frequency);
    let film_delta = wo3_x.optical.thin_film_reflectance_on_material(
        optical_frequency,
        100.0e-9,
        &silicon.optical,
    ) - wo3.optical.thin_film_reflectance_on_material(
        optical_frequency,
        100.0e-9,
        &silicon.optical,
    );

    let kernel_weight = 4.0;
    let kernel_decay = 0.2;
    let memory_frequency = f64::sqrt(kernel_weight - kernel_decay * kernel_decay / 4.0);
    let channel_time = PI / memory_frequency;
    let depolarizing_eigenvalue =
        exponential_memory_depolarizing_eigenvalue(kernel_weight, kernel_decay, channel_time)
            .context("memory-kernel channel calculation failed")?;
    let minimum_choi = qubit_depolarizing_minimum_choi_eigenvalue(depolarizing_eigenvalue);

    let target = DVector::from_vec(vec![1.0, 1.0]);
    let nuisance = DMatrix::from_column_slice(2, 1, &[1.0, 0.0]);
    let absent_calibration = DMatrix::zeros(0, 1);
    let calibration = DMatrix::from_element(1, 1, 1.0);
    let uncalibrated_information =
        efficient_information(&target, &nuisance, &absent_calibration)
            .map_err(|error| inference_error("uncalibrated information calculation", error))?;
    let calibrated_information = efficient_information(&target, &nuisance, &calibration)
        .map_err(|error| inference_error("calibrated information calculation", error))?;

    let profile_target = DVector::from_vec(vec![2.0, 0.0]);
    let unrestricted_distance = unrestricted_residual(&profile_target, &nuisance)
        .map_err(|error| inference_error("unrestricted profile calculation", error))?
        .norm();
    let bounded_distance = bounded_profile_distance(
        &profile_target,
        &nuisance,
        &[NuisanceBound {
            lower: -1.0,
            upper: 1.0,
            unit: "target amplitude".to_owned(),
        }],
    )
    .map_err(|error| inference_error("bounded profile calculation", error))?
    .distance;
    let certified_distance = certified_distance_lower_bound(bounded_distance, 0.2, 0.3)
        .map_err(|error| inference_error("discrimination error margin", error))?;

    let derived_signal = DVector::from_vec(vec![1.0, -2.0, 0.5]);
    let derivative = DMatrix::from_row_slice(2, 3, &[-1.0, 1.0, 0.0, 0.0, -1.0, 1.0]);
    let mut augmentation = DMatrix::zeros(5, 3);
    augmentation
        .view_mut((0, 0), (3, 3))
        .copy_from(&DMatrix::identity(3, 3));
    augmentation.view_mut((3, 0), (2, 3)).copy_from(&derivative);
    let augmented_signal = &augmentation * &derived_signal;
    let augmented_covariance = &augmentation * augmentation.transpose();
    let original_information = derived_signal.norm_squared();
    let augmented_information =
        fisher_information_with_pseudoinverse(&augmented_signal, &augmented_covariance, 1.0e-12)
            .map_err(|error| inference_error("derived-feature covariance calculation", error))?;

    let casimir_gold = materials_core::DrudeLorentzParams {
        drude: Some(gold_drude()),
        oscillators: Vec::new(),
        eps_inf: 1.0,
        extended_drude: None,
    };
    let casimir_silica = silica_casimir_optical();
    let (multilayer, physical_inference, controls, tables) =
        generate_extended_reports(LifshitzMaterials {
            gold: &casimir_gold,
            silica: &casimir_silica,
            alumina: &alumina.optical,
            silicon: &silicon.optical,
        })?;

    Ok(GeneratedAudit {
        report: AuditReport {
            schema_version: 1,
            scope: "native Rust model audit; no laboratory observations",
            planar: PlanarReport {
                gap_m: GAP_METERS,
                sphere_radius_m: SPHERE_RADIUS_METERS,
                pressure_pa: pressure,
                exact_pressure_pa: exact_pressure,
                pressure_relative_error: relative_error(pressure, exact_pressure),
                energy_j_m2: energy,
                exact_energy_j_m2: exact_energy,
                energy_relative_error: relative_error(energy, exact_energy),
                sphere_force_n: sphere_force,
                exact_sphere_force_n: exact_sphere_force,
                sphere_force_relative_error: relative_error(sphere_force, exact_sphere_force),
                pressure_energy_identity_relative_error: relative_error(
                    pressure,
                    3.0 * energy / GAP_METERS,
                ),
            },
            thermal: ThermalReport {
                temperature_k: temperature,
                drude_tm_zero_mode_energy_j_m2: zero_mode,
                exact_drude_tm_zero_mode_energy_j_m2: exact_zero_mode,
                relative_error: relative_error(zero_mode, exact_zero_mode),
            },
            optics: OpticalReport {
                energy_ev: optical_energy_ev,
                wavelength_um: 2.0 * PI * C / optical_frequency * 1.0e6,
                wo3_imaginary_epsilon: wo3_epsilon.im,
                wo3_x_imaginary_epsilon: wo3_x_epsilon.im,
                halfspace_delta_reflectance: halfspace_delta,
                film_100nm_delta_reflectance: film_delta,
            },
            channel: ChannelReport {
                kernel_weight,
                kernel_decay,
                time: channel_time,
                depolarizing_eigenvalue,
                minimum_normalized_choi_eigenvalue: minimum_choi,
            },
            inference: InferenceReport {
                synthetic_uncalibrated_information: uncalibrated_information,
                synthetic_calibrated_information: calibrated_information,
                synthetic_unrestricted_distance: unrestricted_distance,
                synthetic_bounded_distance: bounded_distance,
                arithmetic_error_margin_lower_bound: certified_distance,
                arithmetic_error_margin_hypothesis_0: 0.2,
                arithmetic_error_margin_hypothesis_1: 0.3,
                original_derived_feature_information: original_information,
                augmented_derived_feature_information: augmented_information,
            },
            multilayer,
            physical_inference,
            controls,
        },
        tables,
    })
}

fn validate_report(report: &AuditReport) -> Result<()> {
    ensure!(
        report.schema_version == 1,
        "unexpected audit schema version"
    );
    ensure!(
        report.planar.pressure_relative_error < 1.0e-8,
        "ideal planar pressure normalization drifted"
    );
    ensure!(
        report.planar.energy_relative_error < 1.0e-8,
        "ideal planar energy normalization drifted"
    );
    ensure!(
        report.planar.sphere_force_relative_error < 1.0e-8,
        "ideal sphere-plane PFA normalization drifted"
    );
    ensure!(
        report.planar.pressure_energy_identity_relative_error < 1.0e-10,
        "ideal pressure-energy identity drifted"
    );
    ensure!(
        report.thermal.relative_error < 1.0e-12,
        "Drude TM zero-mode normalization drifted"
    );
    ensure!(
        report.optics.wo3_imaginary_epsilon >= 0.0 && report.optics.wo3_x_imaginary_epsilon >= 0.0,
        "passive optical model produced negative loss"
    );
    ensure!(
        (0.45..0.49).contains(&report.optics.halfspace_delta_reflectance),
        "WO3 half-space contrast left its audited bound"
    );
    ensure!(
        (0.09..0.13).contains(&report.optics.film_100nm_delta_reflectance),
        "WO3 finite-film contrast left its audited bound"
    );
    ensure!(
        report.channel.minimum_normalized_choi_eigenvalue < -0.39,
        "positive-kernel counterexample no longer violates complete positivity"
    );
    ensure!(
        report.inference.synthetic_calibrated_information
            > report.inference.synthetic_uncalibrated_information,
        "independent calibration did not increase efficient information"
    );
    ensure!(
        report.inference.synthetic_unrestricted_distance < 1.0e-12
            && (report.inference.synthetic_bounded_distance - 1.0).abs() < 1.0e-6,
        "bounded and mathematical nuisance imitation are no longer distinct"
    );
    ensure!(
        (report.inference.arithmetic_error_margin_lower_bound - 0.5).abs() < 1.0e-6,
        "numerical and model error margin drifted"
    );
    ensure!(
        relative_error(
            report.inference.augmented_derived_feature_information,
            report.inference.original_derived_feature_information,
        ) < 1.0e-10,
        "derived feature changed information under joint covariance"
    );
    ensure!(
        (-0.472..-0.469).contains(&report.multilayer.au20_stack_pressure_200nm_pa),
        "Au20 multilayer pressure {} left its independently audited interval",
        report.multilayer.au20_stack_pressure_200nm_pa
    );
    ensure!(
        report.multilayer.pressure_energy_derivative_relative_error < 2.0e-6,
        "multilayer pressure-energy derivative identity drifted"
    );
    ensure!(
        report.multilayer.maximum_common_cap_zero_mode_difference_pa == 0.0,
        "common conducting cap did not cancel the differential zero mode"
    );
    ensure!(
        (0.0064..0.0067).contains(&report.multilayer.maximum_uv_completion_relative_change),
        "hypothetical ultraviolet completion sensitivity left its audited interval"
    );
    ensure!(
        report.physical_inference.gap_count == 13
            && (4.9e-8..5.2e-8).contains(&report.physical_inference.unrestricted_residual_fraction)
            && report
                .physical_inference
                .one_width_bounded_residual_fraction
                > 0.6701
            && report
                .physical_inference
                .one_width_bounded_residual_fraction
                < 0.6703,
        "physical nuisance replay lost the bounded-versus-unrestricted distinction"
    );
    ensure!(
        (51.48..51.50).contains(&report.physical_inference.baseline_effective_information)
            && report.physical_inference.strongest_calibration_coordinate
                == "full_differential_gap"
            && (1.16..1.17).contains(
                &report
                    .physical_inference
                    .strongest_calibration_relative_information_gain,
            ),
        "physical calibration-priority replay drifted"
    );
    ensure!(
        report.controls.classical_forward_loop == 0.06
            && report.controls.classical_reverse_loop == -0.06,
        "classical commutator control drifted"
    );
    ensure!(
        relative_error(report.controls.schedule_residual_fraction, 1.0) < 1.0e-12,
        "source-branch commutator schedule fraction {} changed",
        report.controls.schedule_residual_fraction
    );
    ensure!(
        (report.controls.null_false_positive_fraction - 0.01).abs() < 0.002
            && (report.controls.simulated_power - report.controls.analytic_power).abs() < 0.005,
        "independent Monte Carlo replay left its predeclared sampling bounds"
    );
    ensure!(
        report.controls.matched_null_identity_error < 1.0e-10,
        "matched active-sham residual identity drifted"
    );
    Ok(())
}

fn summary_rows(report: &AuditReport) -> [AuditRow; 13] {
    [
        AuditRow {
            section: "planar",
            quantity: "pressure",
            value: report.planar.pressure_pa,
            unit: "Pa",
        },
        AuditRow {
            section: "planar",
            quantity: "energy_per_area",
            value: report.planar.energy_j_m2,
            unit: "J/m^2",
        },
        AuditRow {
            section: "planar",
            quantity: "sphere_pfa_force",
            value: report.planar.sphere_force_n,
            unit: "N",
        },
        AuditRow {
            section: "thermal",
            quantity: "drude_tm_zero_mode_energy_per_area",
            value: report.thermal.drude_tm_zero_mode_energy_j_m2,
            unit: "J/m^2",
        },
        AuditRow {
            section: "optics",
            quantity: "wo3_halfspace_delta_reflectance",
            value: report.optics.halfspace_delta_reflectance,
            unit: "1",
        },
        AuditRow {
            section: "optics",
            quantity: "wo3_100nm_film_delta_reflectance",
            value: report.optics.film_100nm_delta_reflectance,
            unit: "1",
        },
        AuditRow {
            section: "channel",
            quantity: "depolarizing_eigenvalue",
            value: report.channel.depolarizing_eigenvalue,
            unit: "1",
        },
        AuditRow {
            section: "channel",
            quantity: "minimum_normalized_choi_eigenvalue",
            value: report.channel.minimum_normalized_choi_eigenvalue,
            unit: "1",
        },
        AuditRow {
            section: "inference",
            quantity: "synthetic_uncalibrated_information",
            value: report.inference.synthetic_uncalibrated_information,
            unit: "1",
        },
        AuditRow {
            section: "inference",
            quantity: "synthetic_calibrated_information",
            value: report.inference.synthetic_calibrated_information,
            unit: "1",
        },
        AuditRow {
            section: "inference",
            quantity: "synthetic_unrestricted_profile_distance",
            value: report.inference.synthetic_unrestricted_distance,
            unit: "1",
        },
        AuditRow {
            section: "inference",
            quantity: "synthetic_bounded_profile_distance",
            value: report.inference.synthetic_bounded_distance,
            unit: "1",
        },
        AuditRow {
            section: "inference",
            quantity: "arithmetic_error_margin_lower_bound",
            value: report.inference.arithmetic_error_margin_lower_bound,
            unit: "whitened norm",
        },
    ]
}

fn render_outputs(generated: &GeneratedAudit) -> Result<BTreeMap<String, String>> {
    let mut outputs = BTreeMap::new();
    let summary = toml::to_string_pretty(&generated.report).context("serializing audit summary")?;
    outputs.insert("summary.toml".to_owned(), format!("{summary}\n"));
    let mut table = String::from("section\tquantity\tvalue\tunit\n");
    for row in summary_rows(&generated.report) {
        table.push_str(&format!(
            "{}\t{}\t{:.17e}\t{}\n",
            row.section, row.quantity, row.value, row.unit
        ));
    }
    outputs.insert("model_outputs.tsv".to_owned(), table);
    for (name, contents) in &generated.tables {
        outputs.insert((*name).to_owned(), contents.clone());
    }
    let producer_digest = sha256_hex(PRODUCER_SOURCE.as_bytes());
    let source_model_identity = source_model_identity();
    let mut manifest = format!(
        "schema_version = 1\nproducer = \"crates/gororoba_cli_physics/src/bin/casimir_optics_discrimination_audit.rs\"\nproducer_sha256 = \"{producer_digest}\"\nsource_model_identity = \"{source_model_identity}\"\nsource_model_identity_algorithm = \"SHA-256 over path UTF-8, NUL, file bytes, NUL for source_model_input entries in manifest order\"\ncommand = \"cargo run -p gororoba_cli_physics --bin casimir-optics-discrimination-audit --profile validation\"\nscope = \"native model outputs; no laboratory observations\"\n\n"
    );
    for input in &SOURCE_MODEL_INPUTS {
        let digest = sha256_hex(input.bytes);
        manifest.push_str(&format!(
            "[[source_model_input]]\npath = \"{}\"\nsha256 = \"{digest}\"\n\n",
            input.path
        ));
    }
    for (name, contents) in &outputs {
        let digest = sha256_hex(contents.as_bytes());
        manifest.push_str(&format!(
            "[[output]]\npath = \"{name}\"\nsha256 = \"{digest}\"\nbytes = {}\n\n",
            contents.len()
        ));
    }
    outputs.insert("native-output-manifest.toml".to_owned(), manifest);
    Ok(outputs)
}

fn write_report(output_directory: &Path, generated: &GeneratedAudit) -> Result<()> {
    fs::create_dir_all(output_directory)
        .with_context(|| format!("creating {}", output_directory.display()))?;
    for (name, contents) in render_outputs(generated)? {
        fs::write(output_directory.join(&name), contents)
            .with_context(|| format!("writing audit output {name}"))?;
    }
    Ok(())
}

fn check_report(output_directory: &Path, generated: &GeneratedAudit) -> Result<()> {
    let mut failures = Vec::new();
    if let Err(error) = verify_source_retrieval_manifest(
        &output_directory.join(SOURCE_RETRIEVAL_MANIFEST),
        &repo_root::resolve!(),
    ) {
        failures.push(format!("source retrieval manifest: {error:#}"));
    }
    for (name, expected) in render_outputs(generated)? {
        let path = output_directory.join(&name);
        match fs::read_to_string(&path) {
            Ok(retained) if retained == expected => {}
            Ok(_) => failures.push(format!("retained audit output is stale: {name}")),
            Err(error) => failures.push(format!(
                "reading retained audit output {}: {error}",
                path.display()
            )),
        }
    }
    ensure!(
        failures.is_empty(),
        "retained audit verification found {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

fn prepare_distinct_expected_output_directory(
    retained_output_directory: &Path,
    expected_output_directory: &Path,
) -> Result<()> {
    let retained_identity = fs::canonicalize(retained_output_directory).with_context(|| {
        format!(
            "resolving retained output directory {}",
            retained_output_directory.display()
        )
    })?;
    fs::create_dir_all(expected_output_directory).with_context(|| {
        format!(
            "creating expected output directory {}",
            expected_output_directory.display()
        )
    })?;
    let expected_identity = fs::canonicalize(expected_output_directory).with_context(|| {
        format!(
            "resolving expected output directory {}",
            expected_output_directory.display()
        )
    })?;
    ensure!(
        expected_identity != retained_identity,
        "expected output directory must differ from the retained output directory"
    );
    Ok(())
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let generated = generate_report()?;
    validate_report(&generated.report)?;
    if arguments.check {
        if let Some(expected_output_directory) = &arguments.expected_output_directory {
            prepare_distinct_expected_output_directory(
                &arguments.output_directory,
                expected_output_directory,
            )?;
            write_report(expected_output_directory, &generated)?;
        }
        check_report(&arguments.output_directory, &generated)
    } else {
        write_report(&arguments.output_directory, &generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_audit_satisfies_every_output_contract() {
        let report = generate_report().unwrap();
        validate_report(&report.report).unwrap();
        assert_eq!(summary_rows(&report.report).len(), 13);
    }

    #[test]
    fn stale_planar_and_optical_values_are_rejected() {
        let mut report = generate_report().unwrap();
        report.report.planar.pressure_relative_error = 0.099_503_7;
        assert!(validate_report(&report.report).is_err());

        let mut report = generate_report().unwrap();
        report.report.optics.film_100nm_delta_reflectance =
            report.report.optics.halfspace_delta_reflectance;
        assert!(validate_report(&report.report).is_err());
    }

    #[test]
    fn stale_channel_and_covariance_results_are_rejected() {
        let mut report = generate_report().unwrap();
        report.report.channel.minimum_normalized_choi_eigenvalue = 0.0;
        assert!(validate_report(&report.report).is_err());

        let mut report = generate_report().unwrap();
        report
            .report
            .inference
            .augmented_derived_feature_information *= 1.1;
        assert!(validate_report(&report.report).is_err());
    }

    #[test]
    fn serialization_is_byte_deterministic() {
        let first = render_outputs(&generate_report().unwrap()).unwrap();
        let second = render_outputs(&generate_report().unwrap()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn convergence_sweeps_vary_exactly_one_quadrature_control() {
        let cutoff_options = cutoff_convergence_options();
        assert_eq!(
            cutoff_options.map(|options| options.kappa_cutoff),
            [16.0, 20.0, 24.0]
        );
        assert!(
            cutoff_options
                .iter()
                .all(|options| options.kappa_order == 256 && options.angle_order == 64)
        );

        let radial_options = radial_convergence_options();
        assert_eq!(
            radial_options.map(|options| options.kappa_order),
            [128, 256, 512]
        );
        assert!(
            radial_options
                .iter()
                .all(|options| options.kappa_cutoff == 24.0 && options.angle_order == 64)
        );

        let angular_options = angular_convergence_options();
        assert_eq!(
            angular_options.map(|options| options.angular_order),
            [32, 64, 128]
        );
        assert!(
            angular_options
                .iter()
                .all(|options| options.radial_order == 128)
        );
    }

    #[test]
    fn convergence_table_declares_targets_and_tolerances() {
        let generated = generate_report().unwrap();
        let table = generated.tables.get("planar_convergence.tsv").unwrap();
        let rows = table.lines().skip(1).collect::<Vec<_>>();
        assert_eq!(rows.len(), 15);

        let mut factor_counts = BTreeMap::new();
        for row in rows {
            let columns = row.split('\t').collect::<Vec<_>>();
            assert_eq!(columns.len(), 11);
            *factor_counts.entry(columns[0]).or_insert(0) += 1;
            let tolerance: f64 = columns[9].parse().unwrap();
            match columns[0] {
                "cutoff" | "radial_order" => {
                    assert_eq!(columns[1], "perfect_conductor_half_spaces");
                    assert_eq!(tolerance, PLANAR_CONVERGENCE_RELATIVE_TOLERANCE);
                }
                "angular_order" => {
                    assert_eq!(columns[1], ANGULAR_CONVERGENCE_TARGET);
                    assert_eq!(columns[2], "na");
                    assert_eq!(
                        tolerance,
                        MULTILAYER_ANGULAR_CONVERGENCE_RELATIVE_TOLERANCE
                    );
                }
                factor => panic!("unexpected convergence factor {factor}"),
            }
        }
        assert_eq!(factor_counts.get("cutoff"), Some(&6));
        assert_eq!(factor_counts.get("radial_order"), Some(&6));
        assert_eq!(factor_counts.get("angular_order"), Some(&3));
    }

    #[test]
    fn retained_source_manifest_verifies_every_declared_digest() {
        let repository_root = repo_root::resolve!();
        let manifest_path = repository_root
            .join(DEFAULT_OUTPUT_DIRECTORY)
            .join(SOURCE_RETRIEVAL_MANIFEST);
        let manifest_source = fs::read_to_string(manifest_path).unwrap();
        verify_source_retrieval_manifest_source(&manifest_source, &repository_root).unwrap();

        let manifest: SourceRetrievalManifest = toml::from_str(&manifest_source).unwrap();
        let first_digest = &manifest.source.first().unwrap().sha256;
        let stale_manifest = manifest_source.replacen(first_digest.as_str(), &"0".repeat(64), 1);
        let error = verify_source_retrieval_manifest_source(&stale_manifest, &repository_root)
            .unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
    }

    #[test]
    fn retained_source_manifest_rejects_paths_outside_the_repository() {
        let repository_root = repo_root::resolve!();
        let manifest_path = repository_root
            .join(DEFAULT_OUTPUT_DIRECTORY)
            .join(SOURCE_RETRIEVAL_MANIFEST);
        let manifest_source = fs::read_to_string(manifest_path).unwrap();
        let manifest: SourceRetrievalManifest = toml::from_str(&manifest_source).unwrap();
        let first_path = manifest.source.first().unwrap().path.to_str().unwrap();
        let unsafe_manifest = manifest_source.replacen(first_path, "../outside.pdf", 1);
        let error = verify_source_retrieval_manifest_source(&unsafe_manifest, &repository_root)
            .unwrap_err();
        assert!(error.to_string().contains("unsafe repository-relative path"));
    }

    #[test]
    fn expected_output_directory_rejects_resolved_aliases() {
        let temporary_directory = tempfile::tempdir().unwrap();
        let retained_directory = temporary_directory.path().join("retained");
        fs::create_dir(&retained_directory).unwrap();
        let relative_alias = retained_directory.join("..").join("retained");

        let error = prepare_distinct_expected_output_directory(
            &retained_directory,
            &relative_alias,
        )
        .unwrap_err();
        assert!(error.to_string().contains("must differ"));
    }
}
