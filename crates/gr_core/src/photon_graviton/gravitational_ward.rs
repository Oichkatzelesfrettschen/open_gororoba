//! Tensor-valued gravitational Ward comparison for the irreducible diagram.
//!
//! Equation 7.2 of arXiv:2601.23279 compares the graviton contraction of the
//! irreducible rank-three tensor with a one-photon term and a two-photon term.
//! The original photon index remains explicit as the four components of the
//! lower-point vectors in this module.

use num_complex::Complex64;

use super::{
    external_tensor::external_tensor_off_shell,
    irreducible_tensor::{irreducible_tensor_renormalized, irreducible_tensor_unrenormalized},
    one_photon::one_photon_amplitude,
    quadrature::QuadratureConfig,
    tadpole_tensor::tadpole_tensor_unrenormalized,
    tensor_integrands::{
        TensorEvaluationError, TensorLoopConfig, outer, rank_three_add, right_contract,
        validate_tensor_inputs,
    },
    tensor_types::{
        ComplexFourVector, ComplexLorentzMatrix, GravitationalWardResidual, RenormalizationState,
        ResidualTolerance, WardKinematics, vector_norm,
    },
    types::LoopType,
    vacuum_pol_tensor::vacuum_polarization_tensor_unrenormalized,
};

/// Complete off-shell lower-point comparison, including the distributional
/// disposition of the one-photon term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravitationalWardEvaluation {
    pub residual: GravitationalWardResidual,
    pub lower_point_momentum: ComplexFourVector,
    pub one_photon_delta_support: bool,
    pub one_photon_coefficient_components: ComplexFourVector,
    pub two_photon_effective_polarization: ComplexFourVector,
}

/// One finite-virtuality point used for the source-defined on-shell limit.
#[derive(Debug, Clone, PartialEq)]
pub struct OnShellCombinedSample {
    pub virtuality: Complex64,
    pub combined_components: ComplexFourVector,
    pub absolute_norm: f64,
}

/// Result of an explicit combined irreducible-plus-external extrapolation.
#[derive(Debug, Clone, PartialEq)]
pub struct OnShellCombinedLimit {
    pub samples: Vec<OnShellCombinedSample>,
    pub extrapolated_components: ComplexFourVector,
    pub extrapolated_norm: f64,
    pub convergence_estimate: f64,
    pub conditioning_scale: f64,
    pub tolerance: ResidualTolerance,
    pub passes: bool,
}

/// Separate tadpole result. The rank-one vector is retained before applying
/// the physical photon polarization required by the on-shell source identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TadpoleOnShellWardEvaluation {
    pub rank_one_residual: GravitationalWardResidual,
    pub photon_contracted_residual: Complex64,
    pub absolute_photon_contracted_residual: f64,
    pub normalized_photon_contracted_residual: f64,
    pub passes: bool,
}

/// Evaluate the off-shell identity with one common unrenormalized regulator.
///
/// The one-photon amplitude is distributional in its momentum. Its coefficient
/// and support flag are retained separately; the physical RHS component is
/// zero away from the delta support and is the retained coefficient on that
/// support. No nonzero expectation is imposed on either side of the identity.
pub fn gravitational_ward_off_shell(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
    tolerance: ResidualTolerance,
) -> Result<GravitationalWardEvaluation, TensorEvaluationError> {
    let (loop_config, _) = validate_tensor_inputs(kinematics, loop_config, quadrature)?;
    let irreducible =
        irreducible_tensor_unrenormalized(kinematics, loop_type, loop_config, quadrature)?;
    let lhs_components = irreducible.contract_graviton(&kinematics.graviton_variation());

    let lower_point_momentum = kinematics.k0 + kinematics.k;
    let mut one_photon_coefficient_components = ComplexFourVector::zeros();
    let mut one_photon_delta_support = true;
    for alpha in 0..4 {
        let mut basis = ComplexFourVector::zeros();
        basis[alpha] = Complex64::new(1.0, 0.0);
        let photon_field = outer(&kinematics.k, &basis) - outer(&basis, &kinematics.k);
        let tilde_epsilon = right_contract(&photon_field, &kinematics.zeta0)
            .map(|value| value * Complex64::new(loop_config.kappa, 0.0));
        let amplitude = one_photon_amplitude(
            &kinematics.field_strength,
            lower_point_momentum,
            tilde_epsilon,
            loop_type,
            loop_config,
            quadrature,
        )?;
        one_photon_delta_support &= amplitude.momentum_delta_support;
        one_photon_coefficient_components[alpha] = amplitude.coefficient;
    }
    let one_photon_rhs_components = if one_photon_delta_support {
        one_photon_coefficient_components
    } else {
        ComplexFourVector::zeros()
    };

    let two_photon_effective_polarization =
        right_contract(&kinematics.field_strength, &kinematics.zeta0)
            .map(|value| value * Complex64::new(0.0, -loop_config.kappa));
    let vacuum_polarization =
        vacuum_polarization_tensor_unrenormalized(kinematics, loop_type, loop_config, quadrature)?;
    let two_photon_rhs_components = ComplexFourVector::from_fn(|alpha, _| {
        (0..4)
            .map(|beta| {
                two_photon_effective_polarization[beta] * vacuum_polarization[(beta, alpha)]
            })
            .sum()
    });

    let conditioning_scale = uncancelled_scale(
        &lhs_components,
        &one_photon_rhs_components,
        &two_photon_rhs_components,
    )?;
    let residual = GravitationalWardResidual::from_components(
        lhs_components,
        one_photon_rhs_components,
        two_photon_rhs_components,
        kinematics.shell_mode,
        RenormalizationState::Unrenormalized,
        conditioning_scale,
        tolerance,
    )?;
    Ok(GravitationalWardEvaluation {
        residual,
        lower_point_momentum,
        one_photon_delta_support,
        one_photon_coefficient_components,
        two_photon_effective_polarization,
    })
}

/// Verify the separate on-shell tadpole variation without an external-leg
/// denominator. The lower-point RHS is zero for this separately invariant
/// diagram, so all four residual components remain in the result.
pub fn tadpole_gravitational_on_shell(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
    tolerance: ResidualTolerance,
) -> Result<TadpoleOnShellWardEvaluation, TensorEvaluationError> {
    let tensor = tadpole_tensor_unrenormalized(
        kinematics,
        ComplexLorentzMatrix::zeros(),
        loop_type,
        loop_config,
        quadrature,
    )?;
    let lhs_components = tensor.contract_graviton(&kinematics.graviton_variation());
    let conditioning_scale = tensor
        .components()
        .iter()
        .map(|component| component.norm())
        .sum::<f64>();
    if !conditioning_scale.is_finite() || conditioning_scale <= 0.0 {
        return Err(TensorEvaluationError::Kinematics(
            super::tensor_types::KinematicsError::InvalidConditioningScale,
        ));
    }
    let rank_one_residual = GravitationalWardResidual::from_components(
        lhs_components,
        ComplexFourVector::zeros(),
        ComplexFourVector::zeros(),
        kinematics.shell_mode,
        RenormalizationState::Unrenormalized,
        conditioning_scale,
        tolerance,
    )?;
    let photon_contracted_residual: Complex64 = rank_one_residual
        .defect_components
        .iter()
        .zip(kinematics.epsilon.iter())
        .map(|(component, polarization)| *component * *polarization)
        .sum();
    let absolute_photon_contracted_residual = photon_contracted_residual.norm();
    let normalized_photon_contracted_residual =
        absolute_photon_contracted_residual / conditioning_scale;
    Ok(TadpoleOnShellWardEvaluation {
        rank_one_residual,
        photon_contracted_residual,
        absolute_photon_contracted_residual,
        normalized_photon_contracted_residual,
        passes: tolerance.accepts(
            absolute_photon_contracted_residual,
            normalized_photon_contracted_residual,
        ),
    })
}

/// Evaluate the finite-virtuality combined irreducible-plus-external path.
///
/// The separate external tensor remains undefined at zero virtuality. This
/// function therefore accepts only nonzero-virtuality points and leaves the
/// extrapolation decision to `on_shell_combined_virtuality_ladder`.
pub fn combined_irr_external_gravitational_variation(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<ComplexFourVector, TensorEvaluationError> {
    let irreducible =
        irreducible_tensor_renormalized(kinematics, loop_type, loop_config, quadrature)?;
    let external = external_tensor_off_shell(kinematics, loop_type, loop_config, quadrature)?;
    let combined = rank_three_add(&irreducible, &external);
    Ok(combined.contract_graviton(&kinematics.graviton_variation()))
}

/// Run a declared nonzero-virtuality ladder and extrapolate the combined
/// gravitational variation to zero virtuality component by component.
pub fn on_shell_combined_virtuality_ladder(
    kinematics_ladder: &[WardKinematics],
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
    tolerance: ResidualTolerance,
) -> Result<OnShellCombinedLimit, TensorEvaluationError> {
    if kinematics_ladder.len() < 2 {
        return Err(TensorEvaluationError::InvalidQuadrature);
    }
    let mut samples = Vec::with_capacity(kinematics_ladder.len());
    for kinematics in kinematics_ladder {
        kinematics.validate()?;
        let virtuality = super::tensor_integrands::bilinear(
            &kinematics.k,
            &super::worldline_tensor::identity(),
            &kinematics.k,
        );
        if virtuality.norm() <= kinematics.validation_tolerance {
            return Err(TensorEvaluationError::ExternalOnShellSingularity);
        }
        let combined = combined_irr_external_gravitational_variation(
            kinematics,
            loop_type,
            loop_config,
            quadrature,
        )?;
        samples.push(OnShellCombinedSample {
            virtuality,
            absolute_norm: vector_norm(&combined),
            combined_components: combined,
        });
    }
    let last = samples.len() - 1;
    let (extrapolated_components, last_extrapolation) =
        extrapolate_pair(&samples[last - 1], &samples[last])?;
    let convergence_estimate = if samples.len() >= 3 {
        let previous = extrapolate_pair(&samples[last - 2], &samples[last - 1])?.0;
        vector_norm(&(extrapolated_components - previous))
    } else {
        f64::INFINITY
    };
    let conditioning_scale = samples
        .iter()
        .map(|sample| sample.absolute_norm)
        .fold(0.0, f64::max);
    let normalized_extrapolation = if conditioning_scale > 0.0 {
        last_extrapolation / conditioning_scale
    } else {
        f64::INFINITY
    };
    let passes = convergence_estimate.is_finite()
        && tolerance.accepts(last_extrapolation, normalized_extrapolation);
    Ok(OnShellCombinedLimit {
        samples,
        extrapolated_components,
        extrapolated_norm: last_extrapolation,
        convergence_estimate,
        conditioning_scale,
        tolerance,
        passes,
    })
}

fn extrapolate_pair(
    first: &OnShellCombinedSample,
    second: &OnShellCombinedSample,
) -> Result<(ComplexFourVector, f64), TensorEvaluationError> {
    let denominator = first.virtuality - second.virtuality;
    if denominator.norm() <= f64::EPSILON {
        return Err(TensorEvaluationError::InvalidQuadrature);
    }
    let extrapolated = (second.combined_components * first.virtuality
        - first.combined_components * second.virtuality)
        / denominator;
    let norm = vector_norm(&extrapolated);
    if !norm.is_finite() {
        return Err(TensorEvaluationError::NonFiniteResult);
    }
    Ok((extrapolated, norm))
}

fn uncancelled_scale(
    lhs: &ComplexFourVector,
    one_photon: &ComplexFourVector,
    two_photon: &ComplexFourVector,
) -> Result<f64, TensorEvaluationError> {
    let scale = vector_norm(lhs)
        .max(vector_norm(one_photon))
        .max(vector_norm(two_photon));
    if !scale.is_finite() || scale <= 0.0 {
        return Err(TensorEvaluationError::Kinematics(
            super::tensor_types::KinematicsError::InvalidConditioningScale,
        ));
    }
    Ok(scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photon_graviton::{
        tadpole_tensor::{photon_field_strength, tadpole_integrand},
        tensor_types::{
            ComplexFourVector, ComplexLorentzMatrix, GaugeWardResidual, MomentumRule, ShellMode,
        },
    };
    use std::{fmt::Write as _, fs};

    fn fixture() -> WardKinematics {
        let mut field = ComplexLorentzMatrix::zeros();
        field[(0, 1)] = Complex64::new(0.1, 0.0);
        field[(1, 0)] = Complex64::new(-0.1, 0.0);
        let k = ComplexFourVector::from([
            Complex64::new(0.15, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        WardKinematics::new(
            k,
            -k,
            ComplexFourVector::from([
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            ComplexLorentzMatrix::identity(),
            ComplexFourVector::from([
                Complex64::new(0.2, 0.0),
                Complex64::new(0.1, 0.0),
                Complex64::new(-0.3, 0.0),
                Complex64::new(0.4, 0.0),
            ]),
            field,
            Complex64::new(0.0625, 0.0),
            ShellMode::OffShell,
            MomentumRule::ConstantBackgroundConversion,
            false,
            1.0e-12,
        )
        .expect("valid gravitational fixture")
    }

    fn on_shell_fixture(virtuality: f64, shell_mode: ShellMode) -> WardKinematics {
        let mut field = ComplexLorentzMatrix::zeros();
        field[(0, 1)] = Complex64::new(0.1, 0.0);
        field[(1, 0)] = Complex64::new(-0.1, 0.0);
        let longitudinal_imaginary = (1.0 - virtuality).sqrt();
        let k = ComplexFourVector::from([
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, longitudinal_imaginary),
        ]);
        WardKinematics::new(
            k,
            -k,
            ComplexFourVector::from([
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            ComplexLorentzMatrix::identity(),
            ComplexFourVector::from([
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            field,
            Complex64::new(virtuality, 0.0),
            shell_mode,
            MomentumRule::ConstantBackgroundConversion,
            true,
            1.0e-12,
        )
        .expect("valid virtuality fixture")
    }

    fn grid_fixture(
        field_strength: f64,
        omega: f64,
        theta: f64,
        virtuality: f64,
        shell_mode: ShellMode,
    ) -> WardKinematics {
        let mut field = ComplexLorentzMatrix::zeros();
        field[(0, 1)] = Complex64::new(field_strength, 0.0);
        field[(1, 0)] = Complex64::new(-field_strength, 0.0);
        let temporal_factor = if virtuality == 0.0 {
            1.0
        } else {
            (1.0 - virtuality / (omega * omega)).sqrt()
        };
        let k = ComplexFourVector::from([
            Complex64::new(omega * theta.sin(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(omega * theta.cos(), 0.0),
            Complex64::new(0.0, omega * temporal_factor),
        ]);
        WardKinematics::new(
            k,
            -k,
            ComplexFourVector::from([
                Complex64::new(theta.cos(), 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-theta.sin(), 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            ComplexLorentzMatrix::identity(),
            ComplexFourVector::from([
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            field,
            Complex64::new(virtuality, 0.0),
            shell_mode,
            MomentumRule::ConstantBackgroundConversion,
            true,
            1.0e-12,
        )
        .expect("valid final-grid fixture")
    }

    fn complex_text(value: Complex64) -> String {
        format!("{:.17e}{:+.17e}i", value.re, value.im)
    }

    fn matrix_text(matrix: &ComplexLorentzMatrix) -> String {
        let mut values = Vec::with_capacity(16);
        for row in 0..4 {
            for column in 0..4 {
                values.push(format!("\"{}\"", complex_text(matrix[(row, column)])));
            }
        }
        format!("[{}]", values.join(", "))
    }

    fn vector_text(vector: &ComplexFourVector) -> String {
        let values = vector
            .iter()
            .map(|value| format!("\"{}\"", complex_text(*value)))
            .collect::<Vec<_>>();
        format!("[{}]", values.join(", "))
    }

    fn append_gauge_record(output: &mut String, name: &str, residual: Option<&GaugeWardResidual>) {
        match residual {
            Some(residual) => {
                writeln!(output, "{name}_status = \"evaluated\"").expect("write status");
                writeln!(
                    output,
                    "{name}_components = {}",
                    matrix_text(&residual.contracted_components)
                )
                .expect("write components");
                writeln!(
                    output,
                    "{name}_absolute_norm = {:.17e}",
                    residual.absolute_norm
                )
                .expect("write absolute norm");
                writeln!(
                    output,
                    "{name}_normalized_norm = {:.17e}",
                    residual.normalized_norm
                )
                .expect("write normalized norm");
                writeln!(output, "{name}_passes = {}", residual.passes).expect("write verdict");
            }
            None => {
                writeln!(output, "{name}_status = \"undefined_at_zero_virtuality\"")
                    .expect("write singular status");
                writeln!(output, "{name}_components = []").expect("write empty components");
            }
        }
    }

    #[test]
    fn off_shell_comparison_retains_all_lower_point_components() {
        let evaluation = gravitational_ward_off_shell(
            &fixture(),
            LoopType::Scalar,
            TensorLoopConfig::unit_natural(),
            &QuadratureConfig::fast(),
            ResidualTolerance::new(1.0e-12, 1.0e-6).expect("frozen tolerance"),
        )
        .expect("off-shell gravitational comparison");
        assert!(evaluation.one_photon_delta_support);
        assert_eq!(evaluation.lower_point_momentum, ComplexFourVector::zeros());
        assert!(evaluation.residual.passes);
        assert!(
            evaluation
                .residual
                .lhs_components
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert!(evaluation.residual.absolute_defect.is_finite());
        println!(
            "off_shell loop=Scalar lhs={:?} one_photon={:?} two_photon={:?} rhs={:?} defect={:?} absolute={:.16e} normalized={:.16e}",
            evaluation.residual.lhs_components,
            evaluation.residual.one_photon_rhs_components,
            evaluation.residual.two_photon_rhs_components,
            evaluation.residual.lower_point_rhs_components,
            evaluation.residual.defect_components,
            evaluation.residual.absolute_defect,
            evaluation.residual.normalized_defect,
        );
    }

    #[test]
    fn p1_calibration_records_scalar_identity_conditioning() {
        let tolerance = ResidualTolerance::new(1.0, 1.0).expect("characterization tolerance");
        let calibration_points = [
            (
                0.10,
                0.20,
                std::f64::consts::PI / 3.0,
                "field_010_omega_020_pi3",
            ),
            (
                0.05,
                0.15,
                std::f64::consts::PI / 4.0,
                "field_005_omega_015_pi4",
            ),
            (
                0.01,
                0.10,
                std::f64::consts::PI / 6.0,
                "field_001_omega_010_pi6",
            ),
        ];
        let quadratures = [
            ("fast", QuadratureConfig::fast()),
            ("default", QuadratureConfig::default()),
            ("high", QuadratureConfig::high_accuracy()),
            (
                "precision",
                QuadratureConfig {
                    n_u: 128,
                    n_t: 256,
                    t_max: 60.0,
                    t_min: 1.0e-10,
                },
            ),
        ];
        let mut output = String::new();
        writeln!(
            output,
            "# Scalar calibration points are selected before the final grid."
        )
        .expect("write calibration header");
        writeln!(output, "[meta]").expect("write calibration metadata");
        writeln!(output, "artifact_id = \"p1-photon-graviton-calibration\"")
            .expect("write calibration metadata");
        writeln!(output, "tolerance_absolute = 1.00000000000000000e-12")
            .expect("write calibration tolerance");
        writeln!(output, "tolerance_normalized = 1.00000000000000000e-06")
            .expect("write calibration tolerance");
        let mut scalar_max_absolute = 0.0_f64;
        let mut scalar_max_normalized = 0.0_f64;
        for &(field_strength, omega, theta, point_label) in &calibration_points {
            let virtuality = 0.2 * omega * omega;
            let kinematics = grid_fixture(
                field_strength,
                omega,
                theta,
                virtuality,
                ShellMode::OffShell,
            );
            for &(quadrature_label, quadrature) in &quadratures {
                for &loop_type in &[LoopType::Scalar, LoopType::Spinor] {
                    let evaluation = gravitational_ward_off_shell(
                        &kinematics,
                        loop_type,
                        TensorLoopConfig::unit_natural(),
                        &quadrature,
                        tolerance,
                    )
                    .expect("calibration identity");
                    if loop_type == LoopType::Scalar {
                        scalar_max_absolute =
                            scalar_max_absolute.max(evaluation.residual.absolute_defect);
                        scalar_max_normalized =
                            scalar_max_normalized.max(evaluation.residual.normalized_defect);
                    }
                    writeln!(output, "\n[[record]]").expect("write calibration record");
                    writeln!(output, "point = \"{point_label}\"").expect("write point");
                    writeln!(output, "loop_type = \"{loop_type:?}\"").expect("write loop");
                    writeln!(output, "quadrature = \"{quadrature_label}\"")
                        .expect("write quadrature");
                    writeln!(
                        output,
                        "absolute_defect = {:.17e}",
                        evaluation.residual.absolute_defect
                    )
                    .expect("write absolute defect");
                    writeln!(
                        output,
                        "normalized_defect = {:.17e}",
                        evaluation.residual.normalized_defect
                    )
                    .expect("write normalized defect");
                    writeln!(
                        output,
                        "components = {}",
                        vector_text(&evaluation.residual.defect_components)
                    )
                    .expect("write components");
                }
            }
        }
        assert!(scalar_max_absolute <= 1.0e-12);
        assert!(scalar_max_normalized <= 1.0e-6);
        if let Ok(output_path) = std::env::var("P1_CALIBRATION_OUTPUT") {
            fs::write(&output_path, output).expect("retain calibration evidence");
            println!(
                "p1_calibration_output={output_path} scalar_max_absolute={scalar_max_absolute:.17e} scalar_max_normalized={scalar_max_normalized:.17e}"
            );
        }
    }

    #[test]
    fn on_shell_tadpole_variation_retains_a_component_residual() {
        for loop_type in [LoopType::Scalar, LoopType::Spinor] {
            let residual = tadpole_gravitational_on_shell(
                &on_shell_fixture(0.0, ShellMode::OnShell),
                loop_type,
                TensorLoopConfig::unit_natural(),
                &QuadratureConfig::fast(),
                ResidualTolerance::new(1.0e6, 1.0e6).expect("wide characterization tolerance"),
            )
            .expect("on-shell tadpole variation");
            assert!(residual.absolute_photon_contracted_residual.is_finite());
            println!(
                "on_shell_tadpole loop={loop_type:?} rank_one={:?} photon_contracted={} absolute={:.16e} normalized={:.16e} passes={}",
                residual.rank_one_residual.defect_components,
                residual.photon_contracted_residual,
                residual.absolute_photon_contracted_residual,
                residual.normalized_photon_contracted_residual,
                residual.passes,
            );
        }
    }

    #[test]
    fn affine_virtuality_extrapolation_preserves_component_signs() {
        let intercept =
            ComplexFourVector::from_fn(|component, _| Complex64::new(component as f64 + 1.0, -0.5));
        let slope =
            ComplexFourVector::from_fn(|component, _| Complex64::new(0.2, component as f64));
        let sample = |virtuality: f64| {
            let combined_components = intercept + slope * Complex64::new(virtuality, 0.0);
            OnShellCombinedSample {
                virtuality: Complex64::new(virtuality, 0.0),
                absolute_norm: vector_norm(&combined_components),
                combined_components,
            }
        };
        let extracted = extrapolate_pair(&sample(0.1), &sample(0.05)).unwrap().0;
        assert!(vector_norm(&(extracted - intercept)) < 1e-14);
    }

    #[test]
    fn combined_on_shell_ladder_retains_limit_status_without_a_floor() {
        let ladder = [
            on_shell_fixture(0.10, ShellMode::OffShell),
            on_shell_fixture(0.05, ShellMode::OffShell),
            on_shell_fixture(0.01, ShellMode::OffShell),
            on_shell_fixture(0.005, ShellMode::OffShell),
        ];
        for loop_type in [LoopType::Scalar, LoopType::Spinor] {
            let limit = on_shell_combined_virtuality_ladder(
                &ladder,
                loop_type,
                TensorLoopConfig::unit_natural(),
                &QuadratureConfig::fast(),
                ResidualTolerance::new(1.0e-12, 1.0e-6).expect("frozen tolerance"),
            )
            .expect("finite-virtuality combined ladder");
            assert!(limit.extrapolated_norm.is_finite());
            assert!(limit.convergence_estimate.is_finite());
            assert!(!limit.passes);
            println!(
                "on_shell_combined loop={loop_type:?} samples={:?} extrapolated={:?} extrapolated_norm={:.16e} convergence={:.16e} passes={}",
                limit
                    .samples
                    .iter()
                    .map(|sample| (sample.virtuality, sample.absolute_norm))
                    .collect::<Vec<_>>(),
                limit.extrapolated_components,
                limit.extrapolated_norm,
                limit.convergence_estimate,
                limit.passes,
            );
        }
    }

    #[test]
    fn on_shell_omission_and_tadpole_trace_controls_break_the_invariant()
    -> Result<(), TensorEvaluationError> {
        let output_path = std::env::var("P1_ON_SHELL_CONTROLS_OUTPUT").ok();
        let mut retained_output = String::new();
        if output_path.is_some() {
            writeln!(
                retained_output,
                "# On-shell diagram omission and tadpole trace controls."
            )
            .expect("write on-shell control header");
            writeln!(retained_output, "[meta]").expect("write on-shell control metadata");
            writeln!(
                retained_output,
                "artifact_id = \"p1-photon-graviton-on-shell-controls\""
            )
            .expect("write on-shell control metadata");
            writeln!(
                retained_output,
                "tolerance_absolute = 1.00000000000000000e-12"
            )
            .expect("write on-shell control tolerance");
            writeln!(
                retained_output,
                "tolerance_normalized = 1.00000000000000000e-06"
            )
            .expect("write on-shell control tolerance");
        }
        let ladder = [
            on_shell_fixture(0.10, ShellMode::OffShell),
            on_shell_fixture(0.05, ShellMode::OffShell),
            on_shell_fixture(0.01, ShellMode::OffShell),
            on_shell_fixture(0.005, ShellMode::OffShell),
        ];
        for &loop_type in &[LoopType::Scalar, LoopType::Spinor] {
            let mut irreducible_samples = Vec::with_capacity(ladder.len());
            let mut external_samples = Vec::with_capacity(ladder.len());
            let mut combined_samples = Vec::with_capacity(ladder.len());
            for kinematics in &ladder {
                let irreducible = irreducible_tensor_renormalized(
                    kinematics,
                    loop_type,
                    TensorLoopConfig::unit_natural(),
                    &QuadratureConfig::fast(),
                )?;
                let external = external_tensor_off_shell(
                    kinematics,
                    loop_type,
                    TensorLoopConfig::unit_natural(),
                    &QuadratureConfig::fast(),
                )?;
                let irreducible_components =
                    irreducible.contract_graviton(&kinematics.graviton_variation());
                let external_components =
                    external.contract_graviton(&kinematics.graviton_variation());
                let combined_components = irreducible_components + external_components;
                let virtuality = super::super::tensor_integrands::bilinear(
                    &kinematics.k,
                    &super::super::worldline_tensor::identity(),
                    &kinematics.k,
                );
                irreducible_samples.push(OnShellCombinedSample {
                    virtuality,
                    absolute_norm: vector_norm(&irreducible_components),
                    combined_components: irreducible_components,
                });
                external_samples.push(OnShellCombinedSample {
                    virtuality,
                    absolute_norm: vector_norm(&external_components),
                    combined_components: external_components,
                });
                combined_samples.push(OnShellCombinedSample {
                    virtuality,
                    absolute_norm: vector_norm(&combined_components),
                    combined_components,
                });
            }
            let irreducible_limit = extrapolate_pair(
                &irreducible_samples[irreducible_samples.len() - 2],
                &irreducible_samples[irreducible_samples.len() - 1],
            )?
            .1;
            let external_limit = extrapolate_pair(
                &external_samples[external_samples.len() - 2],
                &external_samples[external_samples.len() - 1],
            )?
            .1;
            let combined_limit = extrapolate_pair(
                &combined_samples[combined_samples.len() - 2],
                &combined_samples[combined_samples.len() - 1],
            )?
            .1;
            let omission_external_detected = irreducible_limit > 1.0e-6;
            let omission_irreducible_detected = external_limit > 1.0e-6;
            if output_path.is_some() {
                writeln!(retained_output, "\n[[diagram_omission]]")
                    .expect("write diagram omission");
                writeln!(retained_output, "loop_type = \"{loop_type:?}\"")
                    .expect("write diagram loop");
                writeln!(
                    retained_output,
                    "irreducible_plus_external_extrapolated_norm = {:.17e}",
                    combined_limit
                )
                .expect("write combined norm");
                writeln!(
                    retained_output,
                    "omit_external_extrapolated_norm = {:.17e}",
                    irreducible_limit
                )
                .expect("write external omission norm");
                writeln!(
                    retained_output,
                    "omit_irreducible_extrapolated_norm = {:.17e}",
                    external_limit
                )
                .expect("write irreducible omission norm");
                writeln!(
                    retained_output,
                    "omit_external_detected = {omission_external_detected}"
                )
                .expect("write external omission verdict");
                writeln!(
                    retained_output,
                    "omit_irreducible_detected = {omission_irreducible_detected}"
                )
                .expect("write irreducible omission verdict");
            }
            // Keep the 1e-6 omission threshold: the scalar fixture falls below
            // it, while the spinor fixture discriminates both omissions.
            assert_eq!(omission_external_detected, loop_type == LoopType::Spinor);
            assert_eq!(omission_irreducible_detected, loop_type == LoopType::Spinor);
            assert!(combined_limit.is_finite());
        }

        for &loop_type in &[LoopType::Scalar, LoopType::Spinor] {
            let source_kinematics = on_shell_fixture(0.0, ShellMode::OnShell);
            let variation = source_kinematics.graviton_variation();
            let photon_field =
                photon_field_strength(&source_kinematics.k, &source_kinematics.epsilon);
            let mut invariant_kinematics = source_kinematics;
            invariant_kinematics.epsilon0 = variation;
            let invariant_trace = tadpole_integrand(
                &invariant_kinematics,
                photon_field,
                loop_type,
                0.8,
                TensorLoopConfig::unit_natural(),
            )?
            .unrenormalized_trace;
            let mut perturbed_kinematics = invariant_kinematics;
            perturbed_kinematics.epsilon0 =
                variation + ComplexLorentzMatrix::identity().map(|value| value * 0.37);
            let perturbed_trace = tadpole_integrand(
                &perturbed_kinematics,
                photon_field,
                loop_type,
                0.8,
                TensorLoopConfig::unit_natural(),
            )?
            .unrenormalized_trace;
            let trace_detected =
                invariant_trace.norm() < 1.0e-12 && perturbed_trace.norm() > 1.0e-12;
            if output_path.is_some() {
                writeln!(retained_output, "\n[[tadpole_trace_perturbation]]")
                    .expect("write tadpole trace control");
                writeln!(retained_output, "loop_type = \"{loop_type:?}\"")
                    .expect("write tadpole loop");
                writeln!(
                    retained_output,
                    "invariant_trace = \"{}\"",
                    complex_text(invariant_trace)
                )
                .expect("write invariant trace");
                writeln!(
                    retained_output,
                    "perturbed_trace = \"{}\"",
                    complex_text(perturbed_trace)
                )
                .expect("write perturbed trace");
                writeln!(retained_output, "detected = {trace_detected}")
                    .expect("write tadpole trace verdict");
            }
            assert!(trace_detected);
        }
        if let Some(output_path) = output_path {
            fs::write(&output_path, retained_output).expect("retain on-shell control evidence");
            println!("p1_on_shell_controls_output={output_path}");
        }
        Ok(())
    }

    #[test]
    fn p1_final_grid() {
        let Ok(output_path) = std::env::var("P1_FINAL_GRID_OUTPUT") else {
            return;
        };
        let tolerance =
            ResidualTolerance::new(1.0e-12, 1.0e-6).expect("frozen final-grid tolerance");
        let loop_types = [LoopType::Scalar, LoopType::Spinor];
        let fields = [(0.01, "0.01"), (0.10, "0.10")];
        let frequencies = [(0.05, "0.05"), (0.20, "0.20")];
        let angles = [
            (std::f64::consts::PI / 6.0, "pi/6"),
            (std::f64::consts::PI / 3.0, "pi/3"),
        ];
        let quadratures = [
            ("fast", QuadratureConfig::fast()),
            ("default", QuadratureConfig::default()),
            ("high", QuadratureConfig::high_accuracy()),
        ];
        let mut output = String::new();
        writeln!(output, "# Generated by the Rust p1_final_grid test.").expect("write header");
        writeln!(
            output,
            "# Complex arrays use row-major matrix order and no conjugation."
        )
        .expect("write header");
        writeln!(output, "[meta]").expect("write metadata");
        writeln!(output, "artifact_id = \"p1-photon-graviton-final-grid\"")
            .expect("write metadata");
        writeln!(
            output,
            "protocol = \"data/output/audit/2026-08-02/p1_numerical_protocol.toml\""
        )
        .expect("write metadata");
        writeln!(output, "tolerance_absolute = {:.17e}", tolerance.absolute)
            .expect("write tolerance");
        writeln!(
            output,
            "tolerance_normalized = {:.17e}",
            tolerance.normalized
        )
        .expect("write tolerance");
        let mut record_id = 0usize;
        for &(field_strength, field_label) in &fields {
            for &(omega, omega_label) in &frequencies {
                for &(theta, theta_label) in &angles {
                    let sentinel =
                        field_label == "0.10" && omega_label == "0.20" && theta_label == "pi/3";
                    let virtualities = if sentinel {
                        vec![
                            (0.2 * omega * omega, "primary"),
                            (0.1 * omega * omega, "sentinel"),
                        ]
                    } else {
                        vec![(0.2 * omega * omega, "primary")]
                    };
                    for &loop_type in &loop_types {
                        let loop_label = match loop_type {
                            LoopType::Scalar => "scalar",
                            LoopType::Spinor => "spinor",
                        };
                        for &(shell_mode, shell_label) in [
                            (ShellMode::OnShell, "on_shell"),
                            (ShellMode::OffShell, "off_shell"),
                        ]
                        .iter()
                        {
                            for &(virtuality, virtuality_label) in &virtualities {
                                if shell_mode == ShellMode::OnShell && virtuality_label != "primary"
                                {
                                    continue;
                                }
                                let actual_virtuality = if shell_mode == ShellMode::OnShell {
                                    0.0
                                } else {
                                    virtuality
                                };
                                let kinematics = grid_fixture(
                                    field_strength,
                                    omega,
                                    theta,
                                    actual_virtuality,
                                    shell_mode,
                                );
                                for &(quadrature_name, quadrature) in &quadratures {
                                    record_id += 1;
                                    writeln!(output, "\n[[record]]").expect("write record");
                                    writeln!(output, "record_id = {record_id}")
                                        .expect("write record id");
                                    writeln!(output, "field_ratio = {field_label}")
                                        .expect("write field");
                                    writeln!(output, "omega_over_m = {omega_label}")
                                        .expect("write frequency");
                                    writeln!(output, "theta = \"{theta_label}\"")
                                        .expect("write angle");
                                    writeln!(output, "loop_type = \"{loop_label}\"")
                                        .expect("write loop");
                                    writeln!(output, "shell_mode = \"{shell_label}\"")
                                        .expect("write shell");
                                    writeln!(output, "virtuality_label = \"{virtuality_label}\"")
                                        .expect("write virtuality label");
                                    writeln!(
                                        output,
                                        "declared_virtuality = \"{}\"",
                                        complex_text(kinematics.declared_virtuality)
                                    )
                                    .expect("write declared virtuality");
                                    writeln!(output, "quadrature = \"{quadrature_name}\"")
                                        .expect("write quadrature");
                                    writeln!(
                                        output,
                                        "k_components = {}",
                                        vector_text(&kinematics.k)
                                    )
                                    .expect("write k");
                                    writeln!(
                                        output,
                                        "k0_components = {}",
                                        vector_text(&kinematics.k0)
                                    )
                                    .expect("write k0");
                                    writeln!(
                                        output,
                                        "epsilon_components = {}",
                                        vector_text(&kinematics.epsilon)
                                    )
                                    .expect("write epsilon");
                                    writeln!(
                                        output,
                                        "zeta0_components = {}",
                                        vector_text(&kinematics.zeta0)
                                    )
                                    .expect("write zeta");
                                    writeln!(
                                        output,
                                        "graviton_variation_components = {}",
                                        matrix_text(&kinematics.graviton_variation())
                                    )
                                    .expect("write graviton variation");
                                    let gauge_results = if shell_mode == ShellMode::OffShell {
                                        let results = crate::photon_graviton::tensor_ward::gauge_ward_integrated_residuals(
                                            &kinematics,
                                            loop_type,
                                            TensorLoopConfig::unit_natural(),
                                            &quadrature,
                                            tolerance,
                                        )
                                        .expect("off-shell final-grid gauge residuals");
                                        [Some(results[0]), Some(results[1]), Some(results[2])]
                                    } else {
                                        let irreducible = crate::photon_graviton::irreducible_tensor::irreducible_tensor_unrenormalized(
                                            &kinematics,
                                            loop_type,
                                            TensorLoopConfig::unit_natural(),
                                            &quadrature,
                                        )
                                        .expect("on-shell irreducible final-grid tensor");
                                        let tadpole = crate::photon_graviton::tadpole_tensor::tadpole_tensor_unrenormalized(
                                            &kinematics,
                                            ComplexLorentzMatrix::zeros(),
                                            loop_type,
                                            TensorLoopConfig::unit_natural(),
                                            &quadrature,
                                        )
                                        .expect("on-shell tadpole final-grid tensor");
                                        let irr_residual = crate::photon_graviton::tensor_ward::residual_for_tensor(
                                            &irreducible,
                                            &kinematics,
                                            crate::photon_graviton::tensor_types::Diagram::Irreducible,
                                            tolerance,
                                        )
                                        .expect("on-shell irreducible residual");
                                        let tad_residual = crate::photon_graviton::tensor_ward::residual_for_tensor(
                                            &tadpole,
                                            &kinematics,
                                            crate::photon_graviton::tensor_types::Diagram::Tadpole,
                                            tolerance,
                                        )
                                        .expect("on-shell tadpole residual");
                                        [Some(irr_residual), Some(tad_residual), None]
                                    };
                                    append_gauge_record(
                                        &mut output,
                                        "gauge_irreducible",
                                        gauge_results[0].as_ref(),
                                    );
                                    append_gauge_record(
                                        &mut output,
                                        "gauge_tadpole",
                                        gauge_results[1].as_ref(),
                                    );
                                    append_gauge_record(
                                        &mut output,
                                        "gauge_external",
                                        gauge_results[2].as_ref(),
                                    );
                                    if shell_mode == ShellMode::OffShell {
                                        let evaluation = gravitational_ward_off_shell(
                                            &kinematics,
                                            loop_type,
                                            TensorLoopConfig::unit_natural(),
                                            &quadrature,
                                            tolerance,
                                        )
                                        .expect("off-shell gravitational final-grid identity");
                                        writeln!(output, "gravitational_status = \"evaluated\"")
                                            .expect("write gravitational status");
                                        writeln!(
                                            output,
                                            "gravitational_lower_point_momentum = {}",
                                            vector_text(&evaluation.lower_point_momentum)
                                        )
                                        .expect("write lower-point momentum");
                                        writeln!(
                                            output,
                                            "gravitational_one_photon_delta_support = {}",
                                            evaluation.one_photon_delta_support
                                        )
                                        .expect("write delta disposition");
                                        writeln!(
                                            output,
                                            "gravitational_one_photon_coefficients = {}",
                                            vector_text(
                                                &evaluation.one_photon_coefficient_components
                                            )
                                        )
                                        .expect("write one-photon coefficients");
                                        writeln!(
                                            output,
                                            "gravitational_lhs_components = {}",
                                            vector_text(&evaluation.residual.lhs_components)
                                        )
                                        .expect("write gravitational lhs");
                                        writeln!(
                                            output,
                                            "gravitational_one_photon_rhs_components = {}",
                                            vector_text(
                                                &evaluation.residual.one_photon_rhs_components
                                            )
                                        )
                                        .expect("write one-photon rhs");
                                        writeln!(
                                            output,
                                            "gravitational_two_photon_rhs_components = {}",
                                            vector_text(
                                                &evaluation.residual.two_photon_rhs_components
                                            )
                                        )
                                        .expect("write two-photon rhs");
                                        writeln!(
                                            output,
                                            "gravitational_rhs_components = {}",
                                            vector_text(
                                                &evaluation.residual.lower_point_rhs_components
                                            )
                                        )
                                        .expect("write gravitational rhs");
                                        writeln!(
                                            output,
                                            "gravitational_defect_components = {}",
                                            vector_text(&evaluation.residual.defect_components)
                                        )
                                        .expect("write gravitational defect");
                                        writeln!(
                                            output,
                                            "gravitational_absolute_defect = {:.17e}",
                                            evaluation.residual.absolute_defect
                                        )
                                        .expect("write gravitational absolute");
                                        writeln!(
                                            output,
                                            "gravitational_normalized_defect = {:.17e}",
                                            evaluation.residual.normalized_defect
                                        )
                                        .expect("write gravitational normalized");
                                        writeln!(
                                            output,
                                            "gravitational_passes = {}",
                                            evaluation.residual.passes
                                        )
                                        .expect("write gravitational verdict");
                                    } else {
                                        writeln!(
                                            output,
                                            "gravitational_status = \"off_shell_identity_not_applied_to_on_shell_record\""
                                        )
                                        .expect("write on-shell gravitational status");
                                        let tadpole = tadpole_gravitational_on_shell(
                                            &kinematics,
                                            loop_type,
                                            TensorLoopConfig::unit_natural(),
                                            &quadrature,
                                            tolerance,
                                        )
                                        .expect("on-shell tadpole final-grid identity");
                                        writeln!(
                                            output,
                                            "tadpole_gravitational_rank_one_components = {}",
                                            vector_text(
                                                &tadpole.rank_one_residual.defect_components
                                            )
                                        )
                                        .expect("write tadpole rank-one residual");
                                        writeln!(
                                            output,
                                            "tadpole_gravitational_photon_contracted = \"{}\"",
                                            complex_text(tadpole.photon_contracted_residual)
                                        )
                                        .expect("write tadpole scalar residual");
                                        writeln!(
                                            output,
                                            "tadpole_gravitational_absolute = {:.17e}",
                                            tadpole.absolute_photon_contracted_residual
                                        )
                                        .expect("write tadpole absolute");
                                        writeln!(
                                            output,
                                            "tadpole_gravitational_normalized = {:.17e}",
                                            tadpole.normalized_photon_contracted_residual
                                        )
                                        .expect("write tadpole normalized");
                                        writeln!(
                                            output,
                                            "tadpole_gravitational_passes = {}",
                                            tadpole.passes
                                        )
                                        .expect("write tadpole verdict");
                                        let ladder =
                                            [0.4, 0.2, 0.1, 0.05].map(|relative_virtuality| {
                                                grid_fixture(
                                                    field_strength,
                                                    omega,
                                                    theta,
                                                    relative_virtuality * omega * omega,
                                                    ShellMode::OffShell,
                                                )
                                            });
                                        let combined = on_shell_combined_virtuality_ladder(
                                            &ladder,
                                            loop_type,
                                            TensorLoopConfig::unit_natural(),
                                            &quadrature,
                                            tolerance,
                                        )
                                        .expect("on-shell combined final-grid ladder");
                                        writeln!(
                                            output,
                                            "combined_on_shell_virtualities = [{}]",
                                            combined
                                                .samples
                                                .iter()
                                                .map(|sample| format!(
                                                    "\"{}\"",
                                                    complex_text(sample.virtuality)
                                                ))
                                                .collect::<Vec<_>>()
                                                .join(", ")
                                        )
                                        .expect("write combined virtualities");
                                        writeln!(
                                            output,
                                            "combined_on_shell_sample_norms = [{}]",
                                            combined
                                                .samples
                                                .iter()
                                                .map(|sample| format!(
                                                    "{:.17e}",
                                                    sample.absolute_norm
                                                ))
                                                .collect::<Vec<_>>()
                                                .join(", ")
                                        )
                                        .expect("write combined sample norms");
                                        writeln!(
                                            output,
                                            "combined_on_shell_extrapolated_components = {}",
                                            vector_text(&combined.extrapolated_components)
                                        )
                                        .expect("write combined extrapolation");
                                        writeln!(
                                            output,
                                            "combined_on_shell_extrapolated_norm = {:.17e}",
                                            combined.extrapolated_norm
                                        )
                                        .expect("write combined extrapolated norm");
                                        writeln!(
                                            output,
                                            "combined_on_shell_convergence = {:.17e}",
                                            combined.convergence_estimate
                                        )
                                        .expect("write combined convergence");
                                        writeln!(
                                            output,
                                            "combined_on_shell_passes = {}",
                                            combined.passes
                                        )
                                        .expect("write combined verdict");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        fs::write(&output_path, output).expect("retain final-grid evidence");
        println!("p1_final_grid_output={output_path} records={record_id}");
    }
}
