//! Conditional planar-response, local-mechanics and classical-noise composition.
//! Analyst-selected parameters establish reproducible models, not specimen evidence.
use anyhow::{Result, anyhow, ensure};
use clap::Parser;
use materials_core::{
    DrudeLorentzParams, gold_drude,
    multilayer_lifshitz::{
        Multilayer, ZeroTemperatureOptions, zero_temperature_energy_per_area,
        zero_temperature_pressure,
    },
};
use nalgebra::{DMatrix, DVector};
use quantum_core::vacuum_response::{
    HarmonicPulseSequence, MatrixTolerance, harmonic_mode_from_stiffness,
    sampled_gaussian_coherence, thermal_harmonic_coherence,
};
use rayon::prelude::*;
use serde::Serialize;
use stats_core::calibrated_discrimination::efficient_information;
use std::{io::Write, path::PathBuf};

const AREA_M2: f64 = 1e-10;
const SUPPORT_N_PER_M: f64 = 1.0;
const MASS_KG: f64 = 1e-12;
const TEMPERATURE_K: f64 = 0.02;
const DURATION_S: f64 = 1e-4;
const COUPLING_RAD_S_M: f64 = 1e15;
const T1_S: f64 = 1e-3;
const BOLTZMANN_J_K: f64 = 1.380649e-23;

#[derive(Parser)]
struct Arguments {
    /// Write the JSON report to a path; the default writes stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize, PartialEq)]
struct ModelRow {
    model: &'static str,
    gap_m: f64,
    pressure_pa: f64,
    energy_per_area_j_m2: f64,
    force_gradient_n_per_m: f64,
    derivative_step_m: f64,
    derivative_refinement_discrepancy_n_per_m: f64,
    quadrature_pressure_discrepancy_pa: f64,
    quadrature_energy_discrepancy_j_m2: f64,
    effective_stiffness_n_per_m: f64,
    omega_rad_s: f64,
    ordered_positive_frequency_m2: f64,
    ordered_negative_frequency_m2: f64,
    thermal_quasistatic_variance_m2: f64,
    ramsey_coherence: f64,
    echo_coherence: f64,
    thermal_harmonic_ramsey_coherence: f64,
    thermal_harmonic_echo_coherence: f64,
    thermal_harmonic_ramsey_phase_variance_rad2: f64,
    thermal_harmonic_echo_phase_variance_rad2: f64,
    covariance_symmetry_correction_relative: f64,
    covariance_psd_correction_relative: f64,
}

fn checked<T, E: std::fmt::Debug>(result: std::result::Result<T, E>) -> Result<T> {
    result.map_err(|error| anyhow!("{error:?}"))
}

fn classical_coherence(stiffness: f64, echo: bool) -> Result<(f64, f64, f64, f64)> {
    ensure!(
        stiffness.is_finite() && stiffness > 0.0,
        "unstable or invalid local stiffness"
    );
    let variance = BOLTZMANN_J_K * TEMPERATURE_K / stiffness;
    let weights = [
        DURATION_S / 2.0,
        if echo {
            -DURATION_S / 2.0
        } else {
            DURATION_S / 2.0
        },
    ];
    let response = checked(sampled_gaussian_coherence(
        &DMatrix::from_element(2, 2, variance),
        &weights,
        &DVector::from_element(1, COUPLING_RAD_S_M),
        DURATION_S,
        Some(T1_S),
        checked(MatrixTolerance::new(1e-12))?,
    ))?;
    Ok((
        variance,
        response.coherence,
        response.covariance_correction.symmetry_relative,
        response.covariance_correction.psd_projection_relative,
    ))
}

fn model_row(gap_m: f64, ideal: bool) -> Result<ModelRow> {
    ensure!(
        gap_m.is_finite() && gap_m > 0.0,
        "gap must be positive and finite"
    );
    let drude = gold_drude();
    let gold = DrudeLorentzParams {
        drude: Some(drude),
        oscillators: Vec::new(),
        eps_inf: drude.eps_inf,
        extended_drude: None,
    };
    let surface = if ideal {
        Multilayer::ideal_conductor()
    } else {
        Multilayer::half_space(&gold)
    };
    let fine = ZeroTemperatureOptions::default();
    let coarse = ZeroTemperatureOptions {
        radial_order: 48,
        angular_order: 24,
    };
    let pressure =
        |gap, options| checked(zero_temperature_pressure(gap, &surface, &surface, options));
    let energy = |options| {
        checked(zero_temperature_energy_per_area(
            gap_m, &surface, &surface, options,
        ))
    };
    let derivative = |step| -> Result<f64> {
        Ok(
            AREA_M2 * (pressure(gap_m + step, fine)? - pressure(gap_m - step, fine)?)
                / (2.0 * step),
        )
    };
    let step = gap_m * 1e-3;
    let gradient_coarse_step = derivative(step)?;
    let gradient = derivative(step / 2.0)?;
    let stiffness = SUPPORT_N_PER_M - gradient;
    let harmonic = checked(harmonic_mode_from_stiffness(
        MASS_KG,
        stiffness,
        TEMPERATURE_K,
    ))?;
    let (variance, ramsey, symmetry, psd) = classical_coherence(stiffness, false)?;
    let (_, echo, _, _) = classical_coherence(stiffness, true)?;
    let harmonic_ramsey = checked(thermal_harmonic_coherence(
        MASS_KG,
        stiffness,
        TEMPERATURE_K,
        COUPLING_RAD_S_M,
        DURATION_S,
        Some(T1_S),
        HarmonicPulseSequence::Ramsey,
    ))?;
    let harmonic_echo = checked(thermal_harmonic_coherence(
        MASS_KG,
        stiffness,
        TEMPERATURE_K,
        COUPLING_RAD_S_M,
        DURATION_S,
        Some(T1_S),
        HarmonicPulseSequence::HahnEcho,
    ))?;
    let pressure_pa = pressure(gap_m, fine)?;
    let energy_per_area = energy(fine)?;
    Ok(ModelRow {
        model: if ideal {
            "ideal_conductor"
        } else {
            "local_drude_gold_reference"
        },
        gap_m,
        pressure_pa,
        energy_per_area_j_m2: energy_per_area,
        force_gradient_n_per_m: gradient,
        derivative_step_m: step / 2.0,
        derivative_refinement_discrepancy_n_per_m: (gradient - gradient_coarse_step).abs(),
        quadrature_pressure_discrepancy_pa: (pressure_pa - pressure(gap_m, coarse)?).abs(),
        quadrature_energy_discrepancy_j_m2: (energy_per_area - energy(coarse)?).abs(),
        effective_stiffness_n_per_m: stiffness,
        omega_rad_s: harmonic.omega_rad_s,
        ordered_positive_frequency_m2: harmonic.line_weights.positive_frequency_m2,
        ordered_negative_frequency_m2: harmonic.line_weights.negative_frequency_m2,
        thermal_quasistatic_variance_m2: variance,
        ramsey_coherence: ramsey,
        echo_coherence: echo,
        thermal_harmonic_ramsey_coherence: harmonic_ramsey.coherence,
        thermal_harmonic_echo_coherence: harmonic_echo.coherence,
        thermal_harmonic_ramsey_phase_variance_rad2: harmonic_ramsey.phase_variance,
        thermal_harmonic_echo_phase_variance_rad2: harmonic_echo.phase_variance,
        covariance_symmetry_correction_relative: symmetry,
        covariance_psd_correction_relative: psd,
    })
}

fn rows(parallel: bool) -> Result<Vec<ModelRow>> {
    let cases: Vec<_> = [80e-9, 100e-9, 150e-9]
        .into_iter()
        .flat_map(|gap| [(gap, true), (gap, false)])
        .collect();
    if parallel {
        cases
            .par_iter()
            .map(|&(gap, ideal)| model_row(gap, ideal))
            .collect()
    } else {
        cases
            .iter()
            .map(|&(gap, ideal)| model_row(gap, ideal))
            .collect()
    }
}

#[derive(Serialize)]
struct Discrimination {
    observable: &'static str,
    target_parameter_count: usize,
    nuisance_parameter_count: usize,
    coherence_sigma_analyst_chosen: f64,
    nuisance_calibration_precision_analyst_chosen: f64,
    support_stiffness_mimic_shifts_n_per_m: Vec<f64>,
    support_stiffness_tangent_mode_n_per_m_per_amplitude: Vec<f64>,
    calibration_observable: &'static str,
    mimic_coherence_max_discrepancy: f64,
    efficient_information_without_calibration: f64,
    efficient_information_with_independent_calibration: f64,
}

fn coherence_stiffness_slope(stiffness: f64, coherence: f64) -> f64 {
    coherence * (COUPLING_RAD_S_M * DURATION_S).powi(2) * (0.5 * BOLTZMANN_J_K * TEMPERATURE_K)
        / stiffness.powi(2)
}

fn discrimination(models: &[ModelRow]) -> Result<Discrimination> {
    let sigma = 1e-4;
    let precision: f64 = 100.0;
    let target = DVector::from_iterator(
        3,
        models
            .chunks_exact(2)
            .map(|pair| (pair[0].ramsey_coherence - pair[1].ramsey_coherence) / sigma),
    );
    let tangent: Vec<f64> = models
        .chunks_exact(2)
        .map(|pair| {
            let derivative = coherence_stiffness_slope(
                pair[1].effective_stiffness_n_per_m,
                pair[1].ramsey_coherence,
            );
            (pair[0].ramsey_coherence - pair[1].ramsey_coherence) / derivative
        })
        .collect();
    let nuisance = DMatrix::from_fn(3, 1, |index, _| {
        let baseline = &models[2 * index + 1];
        let derivative = coherence_stiffness_slope(
            baseline.effective_stiffness_n_per_m,
            baseline.ramsey_coherence,
        );
        derivative * tangent[index] / sigma
    });
    let mut discrepancy: f64 = 0.0;
    let shifts = models
        .chunks_exact(2)
        .map(|pair| -> Result<f64> {
            let shift = pair[0].effective_stiffness_n_per_m - pair[1].effective_stiffness_n_per_m;
            let (_, mimic, _, _) =
                classical_coherence(pair[1].effective_stiffness_n_per_m + shift, false)?;
            discrepancy = discrepancy.max((mimic - pair[0].ramsey_coherence).abs());
            Ok(shift)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Discrimination {
        observable: "three_quasistatic_ramsey_coherences_local_linear_target_and_support_nuisance_mode",
        target_parameter_count: 1,
        nuisance_parameter_count: 1,
        coherence_sigma_analyst_chosen: sigma,
        nuisance_calibration_precision_analyst_chosen: precision,
        support_stiffness_mimic_shifts_n_per_m: shifts,
        support_stiffness_tangent_mode_n_per_m_per_amplitude: tangent,
        calibration_observable: "hypothetical_independent_dimensionless_stiffness_mode_amplitude",
        mimic_coherence_max_discrepancy: discrepancy,
        efficient_information_without_calibration: checked(efficient_information(
            &target,
            &nuisance,
            &DMatrix::zeros(0, 1),
        ))?,
        efficient_information_with_independent_calibration: checked(efficient_information(
            &target,
            &nuisance,
            &DMatrix::from_element(1, 1, precision.sqrt()),
        ))?,
    })
}

fn generate_report() -> Result<serde_json::Value> {
    let models = rows(true)?;
    let report = serde_json::json!({
        "evidence_class": "computed_model", "specimen_binding": "unprovided",
        "numerics": {"fine_radial_order":96,"fine_angular_order":48,"coarse_radial_order":48,"coarse_angular_order":24,"derivative_initial_step_over_gap":1e-3,"matrix_relative_tolerance":1e-12,"error_status":"refinement_estimates_not_certified_bounds"},
        "temporal_noise_models": {"quasistatic_fields": ["ramsey_coherence","echo_coherence"],"thermal_harmonic_fields": "thermal_harmonic_*", "harmonic_dynamics": "undamped classical thermal initial quadratures with exact continuous-time Ramsey and Hahn echo filters", "discrimination_model": "quasistatic"},
        "ordered_noise_fourier_convention":"exp(+i*omega*t); positive frequency is absorption",
        "parameter_account": {"fitted_parameters": 0,"analyst_selected_device_and_noise_inputs": 7,"analyst_selected_discrimination_inputs": 2,"gap_design_points": 3,"optical_parameters": "fixed_existing_gold_drude_reference"},
        "parameters": {"area_m2": AREA_M2,"support_stiffness_n_per_m": SUPPORT_N_PER_M,"mass_kg": MASS_KG,"temperature_k": TEMPERATURE_K,"duration_s": DURATION_S,"coupling_rad_s_m_analyst_chosen": COUPLING_RAD_S_M,"t1_s_analyst_chosen": T1_S,"boltzmann_j_k": BOLTZMANN_J_K,"time_samples_s": [0.0,DURATION_S]},
        "assumptions": ["Zero-temperature optical pressure and energy approximate cryogenic mechanics; thermodynamic temperatures differ.","Gold response is an existing reference local Drude model; material family identification supplies no specimen calibration.","Local harmonic equilibrium uses an external holding force to balance static attraction; global pull-in stability remains unassessed.","Classical quasistatic thermal covariance k_B*T/k_eff uses two fully correlated samples, an assumed static offset rather than the harmonic trajectory; ordered quantum noise remains separate.","Harmonic temporal filters use the composed mechanical frequency and exact continuous-time integrals. The selected duration spans many mechanical cycles; filter zeros prevent a universal direction of coherence change with stiffness. Harmonic damping, driven trajectories and quantum-bath coherence remain separate obligations.","Finite step and quadrature discrepancies are empirical refinement estimates, not certified error bounds.","Coupling, T1, observation sigma and independent calibration precision are analyst chosen; physical coherence and discrimination remain unmeasured.","Endpoint support shifts reproduce the alternate coherence exactly. A separate analyst-selected stiffness tangent mode uses the actual dC/dk Jacobian to match the coherence contrast locally; calibration precision applies to its hypothetical independent dimensionless amplitude measurement.","A matched thermodynamic cycle, energy harvesting, CPTP dynamics and device feasibility remain unestablished."],
        "source_owners": ["crates/materials_core/src/multilayer_lifshitz.rs","crates/materials_core/src/optical_database.rs","crates/quantum_core/src/vacuum_response.rs","crates/stats_core/src/calibrated_discrimination.rs"],
        "discrimination": discrimination(&models)?, "models": models,
    });
    Ok(report)
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let mut output = serde_json::to_vec_pretty(&generate_report()?)?;
    output.push(b'\n');
    if let Some(path) = arguments.output {
        std::fs::write(path, output)?;
    } else {
        std::io::stdout().lock().write_all(&output)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Refinement discrepancies near roundoff are estimates. Freshness checks
    // validate their finite nonnegative domain rather than their last bits.
    fn compare_reports(
        expected: &serde_json::Value,
        actual: &serde_json::Value,
        key: &str,
    ) -> Result<()> {
        use serde_json::Value;
        match (expected, actual) {
            (Value::Object(previous), Value::Object(current)) => {
                ensure!(
                    previous.len() == current.len(),
                    "object size changed at {key}"
                );
                for (name, value) in previous {
                    compare_reports(
                        value,
                        current
                            .get(name)
                            .ok_or_else(|| anyhow!("missing key {name}"))?,
                        name,
                    )?;
                }
            }
            (Value::Array(previous), Value::Array(current)) => {
                ensure!(
                    previous.len() == current.len(),
                    "array size changed at {key}"
                );
                for (first, second) in previous.iter().zip(current) {
                    compare_reports(first, second, key)?;
                }
            }
            (Value::Number(previous), Value::Number(current)) => {
                let first = previous
                    .as_f64()
                    .ok_or_else(|| anyhow!("invalid number {key}"))?;
                let second = current
                    .as_f64()
                    .ok_or_else(|| anyhow!("invalid number {key}"))?;
                ensure!(
                    first.is_finite() && second.is_finite(),
                    "nonfinite number {key}"
                );
                if key.contains("refinement_discrepancy")
                    || (key.starts_with("quadrature_") && key.contains("discrepancy"))
                    || key == "mimic_coherence_max_discrepancy"
                {
                    ensure!(first >= 0.0 && second >= 0.0, "negative estimate {key}");
                } else if key == "efficient_information_without_calibration"
                    && first.abs() <= 1e-20
                    && second.abs() <= 1e-20
                {
                    // The declared numerical null budget is dimensionless.
                } else {
                    let scale = first.abs().max(second.abs());
                    ensure!(
                        scale == 0.0 || (first / scale - second / scale).abs() <= 1e-9,
                        "numeric prediction changed at {key}: {first} versus {second}"
                    );
                }
            }
            _ => ensure!(expected == actual, "metadata or type changed at {key}"),
        }
        Ok(())
    }

    #[test]
    fn retained_report_matches_current_computation() {
        let root = repo_root::discover().expect("repository root");
        let bytes =
            std::fs::read(root.join("data/output/audit/vacuum-composition/model-report.json"))
                .unwrap();
        let retained = serde_json::from_slice(&bytes).unwrap();
        compare_reports(&retained, &generate_report().unwrap(), "report").unwrap();
    }

    #[test]
    fn scientific_prediction_mutations_reject() {
        let report = generate_report().unwrap();
        for field in ["pressure_pa", "thermal_quasistatic_variance_m2"] {
            let mut changed = report.clone();
            changed["models"][0][field] =
                serde_json::json!(report["models"][0][field].as_f64().unwrap() * 1.01);
            assert!(compare_reports(&report, &changed, "report").is_err());
        }
    }
    #[test]
    fn rejects_unstable_and_invalid_domains() {
        assert!(classical_coherence(-0.1, false).is_err());
        assert!(classical_coherence(f64::NAN, false).is_err());
        assert!(model_row(0.0, true).is_err());
        assert!(model_row(10e-9, true).is_err());
    }
    #[test]
    fn softening_increases_thermal_noise_and_echo_cancels() {
        let (baseline_variance, baseline_coherence, _, _) =
            classical_coherence(1.0, false).unwrap();
        let (soft_variance, soft_coherence, _, _) = classical_coherence(0.5, false).unwrap();
        assert!(soft_variance > baseline_variance);
        assert!(soft_coherence < baseline_coherence);
        let (_, echo, _, _) = classical_coherence(0.5, true).unwrap();
        assert!((echo - (-DURATION_S / (2.0 * T1_S)).exp()).abs() < 1e-14);
    }
    #[test]
    fn harmonic_temporal_filter_differs_from_quasistatic() {
        let row = model_row(80e-9, true).unwrap();
        for (coherence, variance) in [
            (
                row.thermal_harmonic_ramsey_coherence,
                row.thermal_harmonic_ramsey_phase_variance_rad2,
            ),
            (
                row.thermal_harmonic_echo_coherence,
                row.thermal_harmonic_echo_phase_variance_rad2,
            ),
        ] {
            assert!(coherence.is_finite() && (0.0..=1.0).contains(&coherence));
            assert!(variance.is_finite() && variance >= 0.0);
        }
        assert!(row.omega_rad_s * DURATION_S > 20.0 * std::f64::consts::PI);
        assert!(row.thermal_harmonic_ramsey_coherence - row.ramsey_coherence > 1e-4);
        assert!(row.thermal_harmonic_echo_coherence < row.echo_coherence);
    }

    #[test]
    fn parallel_serial_and_calibrated_nuisance() {
        let serial = rows(false).unwrap();
        assert_eq!(serial, rows(true).unwrap());
        let step = 1e-4;
        let (_, plus, _, _) = classical_coherence(1.0 + step, false).unwrap();
        let (_, minus, _, _) = classical_coherence(1.0 - step, false).unwrap();
        let (_, coherence, _, _) = classical_coherence(1.0, false).unwrap();
        assert!(
            ((plus - minus) / (2.0 * step) - coherence_stiffness_slope(1.0, coherence)).abs()
                < 1e-9
        );
        let result = discrimination(&serial).unwrap();
        assert!(result.mimic_coherence_max_discrepancy < 1e-14);
        assert!(result.efficient_information_without_calibration < 1e-20);
        assert!(result.efficient_information_with_independent_calibration > 0.0);
    }
}

#[cfg(test)]
#[path = "vacuum_response_audit/frontier_contract.rs"]
mod frontier_contract;
