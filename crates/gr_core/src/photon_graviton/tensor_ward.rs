//! Rank-preserving electromagnetic gauge Ward residuals.
//!
//! The source states the electromagnetic gauge identity separately for the
//! irreducible, tadpole, and external diagrams. These checks retain the full
//! sixteen graviton-index components after contracting the photon index.

use num_complex::Complex64;

use super::{
    external_tensor::{
        external_tensor_from_vacuum_polarization, external_tensor_unrenormalized_off_shell,
    },
    irreducible_tensor::{
        IrreducibleMutation, irreducible_integrand, irreducible_tensor_unrenormalized_with_mutation,
    },
    quadrature::QuadratureConfig,
    tadpole_tensor::{tadpole_integrand, tadpole_tensor_unrenormalized},
    tensor_integrands::{TensorEvaluationError, TensorLoopConfig, magnetic_field_strength},
    tensor_types::{
        ComplexLorentzMatrix, ComplexRankThreeTensor, Diagram, GaugeWardResidual,
        RenormalizationState, ResidualTolerance, WardKinematics, rank_three_frobenius_norm,
    },
    types::LoopType,
    vacuum_pol_tensor::vacuum_polarization_integrand,
};

pub fn gauge_ward_integrand_residuals(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    proper_time: f64,
    u: f64,
    loop_config: TensorLoopConfig,
    tolerance: ResidualTolerance,
    mutation: IrreducibleMutation,
) -> Result<[GaugeWardResidual; 3], TensorEvaluationError> {
    let irreducible =
        irreducible_integrand(kinematics, loop_type, proper_time, u, loop_config, mutation)?;
    let tadpole = tadpole_integrand(
        kinematics,
        ComplexLorentzMatrix::zeros(),
        loop_type,
        proper_time,
        loop_config,
    )?;
    let vacuum = vacuum_polarization_integrand(kinematics, loop_type, proper_time, u, loop_config)?;
    let _magnetic_field = magnetic_field_strength(&kinematics.field_strength)?;
    let k_squared = super::tensor_integrands::bilinear(
        &kinematics.k,
        &super::worldline_tensor::identity(),
        &kinematics.k,
    );
    if k_squared.norm() <= kinematics.validation_tolerance {
        return Err(TensorEvaluationError::ExternalOnShellSingularity);
    }
    let vacuum_tensor = vacuum
        .symmetric_tensor
        .map(|value| value * Complex64::new(vacuum.determinant_factor, 0.0) * vacuum.exponent);
    let external = external_tensor_from_vacuum_polarization(
        kinematics,
        loop_config,
        k_squared,
        &vacuum_tensor,
    );
    Ok([
        residual_for_tensor(
            &irreducible.total,
            kinematics,
            Diagram::Irreducible,
            tolerance,
        )?,
        residual_for_tensor(&tadpole.tensor, kinematics, Diagram::Tadpole, tolerance)?,
        residual_for_tensor(&external, kinematics, Diagram::External, tolerance)?,
    ])
}

pub fn gauge_ward_integrated_residuals(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
    tolerance: ResidualTolerance,
) -> Result<[GaugeWardResidual; 3], TensorEvaluationError> {
    gauge_ward_integrated_residuals_with_mutation(
        kinematics,
        loop_type,
        loop_config,
        quadrature,
        tolerance,
        IrreducibleMutation::None,
    )
}

pub fn gauge_ward_integrated_residuals_with_mutation(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
    tolerance: ResidualTolerance,
    mutation: IrreducibleMutation,
) -> Result<[GaugeWardResidual; 3], TensorEvaluationError> {
    let irreducible = irreducible_tensor_unrenormalized_with_mutation(
        kinematics,
        loop_type,
        loop_config,
        quadrature,
        mutation,
    )?;
    let tadpole = tadpole_tensor_unrenormalized(
        kinematics,
        ComplexLorentzMatrix::zeros(),
        loop_type,
        loop_config,
        quadrature,
    )?;
    let external =
        external_tensor_unrenormalized_off_shell(kinematics, loop_type, loop_config, quadrature)?;
    Ok([
        residual_for_tensor(&irreducible, kinematics, Diagram::Irreducible, tolerance)?,
        residual_for_tensor(&tadpole, kinematics, Diagram::Tadpole, tolerance)?,
        residual_for_tensor(&external, kinematics, Diagram::External, tolerance)?,
    ])
}

pub fn residual_for_tensor(
    tensor: &ComplexRankThreeTensor,
    kinematics: &WardKinematics,
    diagram: Diagram,
    tolerance: ResidualTolerance,
) -> Result<GaugeWardResidual, TensorEvaluationError> {
    let conditioning_scale = rank_three_frobenius_norm(tensor);
    if !conditioning_scale.is_finite() || conditioning_scale <= 0.0 {
        return Err(TensorEvaluationError::Kinematics(
            super::tensor_types::KinematicsError::InvalidConditioningScale,
        ));
    }
    Ok(GaugeWardResidual::from_tensor(
        tensor,
        kinematics,
        diagram,
        RenormalizationState::Unrenormalized,
        conditioning_scale,
        tolerance,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use std::{fmt::Write as _, fs};

    fn fixture() -> WardKinematics {
        let mut field = ComplexLorentzMatrix::zeros();
        field[(0, 1)] = Complex64::new(0.1, 0.0);
        field[(1, 0)] = Complex64::new(-0.1, 0.0);
        let k = super::super::tensor_types::ComplexFourVector::from([
            Complex64::new(0.15, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        WardKinematics::new(
            k,
            -k,
            super::super::tensor_types::ComplexFourVector::from([
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            ComplexLorentzMatrix::identity(),
            super::super::tensor_types::ComplexFourVector::from([
                Complex64::new(0.2, 0.0),
                Complex64::new(0.1, 0.0),
                Complex64::new(-0.3, 0.0),
                Complex64::new(0.4, 0.0),
            ]),
            field,
            Complex64::new(0.0625, 0.0),
            super::super::tensor_types::ShellMode::OffShell,
            super::super::tensor_types::MomentumRule::ConstantBackgroundConversion,
            false,
            1.0e-12,
        )
        .expect("valid gauge fixture")
    }

    #[test]
    fn tadpole_gauge_residual_retains_a_zero_matrix_not_a_scalar() {
        let results = gauge_ward_integrand_residuals(
            &fixture(),
            LoopType::Scalar,
            0.8,
            0.27,
            TensorLoopConfig::unit_natural(),
            ResidualTolerance::new(1.0e-12, 1.0e-12).expect("valid tolerance"),
            IrreducibleMutation::None,
        )
        .expect("gauge node");
        assert!(
            results[1]
                .contracted_components
                .iter()
                .all(|value| value.norm() < 1.0e-14)
        );
    }

    #[test]
    fn wrong_contraction_momentum_changes_the_rank_two_residual() {
        let kinematics = fixture();
        let tensor = irreducible_integrand(
            &kinematics,
            LoopType::Spinor,
            0.8,
            0.27,
            TensorLoopConfig::unit_natural(),
            IrreducibleMutation::None,
        )
        .expect("irreducible node")
        .total;
        let bad_k = kinematics.k.map(|value| value + Complex64::new(0.07, 0.0));
        let bad_kinematics = WardKinematics::new(
            bad_k,
            -bad_k,
            kinematics.epsilon,
            kinematics.epsilon0,
            kinematics.zeta0,
            kinematics.field_strength,
            super::super::tensor_types::bilinear_dot(&bad_k, &bad_k),
            super::super::tensor_types::ShellMode::OffShell,
            super::super::tensor_types::MomentumRule::ConstantBackgroundConversion,
            false,
            1.0e-12,
        )
        .expect("bad contraction fixture remains valid kinematics");
        let baseline = residual_for_tensor(
            &tensor,
            &kinematics,
            Diagram::Irreducible,
            ResidualTolerance::new(1.0, 1.0).expect("valid tolerance"),
        )
        .expect("baseline residual");
        let changed = residual_for_tensor(
            &tensor,
            &bad_kinematics,
            Diagram::Irreducible,
            ResidualTolerance::new(1.0, 1.0).expect("valid tolerance"),
        )
        .expect("changed residual");
        assert!((baseline.absolute_norm - changed.absolute_norm).abs() > 1.0e-12);
    }

    #[test]
    fn scalar_integrated_gauge_residuals_pass_the_frozen_protocol() {
        let tolerance = ResidualTolerance::new(1.0e-12, 1.0e-8).expect("frozen tolerance");
        for quadrature in [
            QuadratureConfig::fast(),
            QuadratureConfig::default(),
            QuadratureConfig::high_accuracy(),
        ] {
            let results = gauge_ward_integrated_residuals(
                &fixture(),
                LoopType::Scalar,
                TensorLoopConfig::unit_natural(),
                &quadrature,
                tolerance,
            )
            .expect("scalar integrated gauge residuals");
            for result in results {
                assert!(
                    result.passes,
                    "scalar {} residual exceeded the frozen protocol: absolute={:.16e}, normalized={:.16e}",
                    result.diagram.as_str(),
                    result.absolute_norm,
                    result.normalized_norm
                );
            }
        }
    }

    #[test]
    fn spinor_irreducible_residual_retains_the_contact_boundary_gap() {
        let tolerance = ResidualTolerance::new(1.0e-12, 1.0e-8).expect("frozen tolerance");
        let mut residuals = Vec::new();
        for quadrature in [
            QuadratureConfig::fast(),
            QuadratureConfig::default(),
            QuadratureConfig::high_accuracy(),
        ] {
            let results = gauge_ward_integrated_residuals(
                &fixture(),
                LoopType::Spinor,
                TensorLoopConfig::unit_natural(),
                &quadrature,
                tolerance,
            )
            .expect("spinor integrated gauge residuals");
            assert!(results[1].passes, "spinor tadpole gauge residual failed");
            assert!(results[2].passes, "spinor external gauge residual failed");
            assert!(!results[0].passes);
            assert!(results[0].absolute_norm > 1.0e-6);
            assert!(results[0].normalized_norm > 1.0e-3);
            residuals.push(results[0].absolute_norm);
        }
        let minimum = residuals.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = residuals.iter().copied().fold(0.0, f64::max);
        assert!(minimum.is_finite() && maximum.is_finite());
        assert!(maximum / minimum < 2.0);
    }

    #[test]
    fn declared_integrated_mutations_change_the_rank_three_tensor() {
        let output_path = std::env::var("P1_MUTATION_OUTPUT").ok();
        let mut retained_output = String::new();
        if output_path.is_some() {
            writeln!(
                retained_output,
                "# Tensor Ward mutation and input controls."
            )
            .expect("write mutation header");
            writeln!(retained_output, "[meta]").expect("write mutation metadata");
            writeln!(
                retained_output,
                "artifact_id = \"p1-photon-graviton-mutation-controls\""
            )
            .expect("write mutation metadata");
            writeln!(
                retained_output,
                "detection_threshold = 1.00000000000000004e-10"
            )
            .expect("write mutation threshold");
            writeln!(
                retained_output,
                "mutation_source = \"irreducible J1, J2, J3 assembly and Appendix B projector inputs\""
            )
            .expect("write mutation metadata");
        }
        for loop_type in [LoopType::Scalar, LoopType::Spinor] {
            let baseline = super::super::irreducible_tensor::irreducible_tensor_unrenormalized(
                &fixture(),
                loop_type,
                TensorLoopConfig::unit_natural(),
                &QuadratureConfig::fast(),
            )
            .expect("baseline tensor");
            for mutation in [
                IrreducibleMutation::FlipJ1Sign,
                IrreducibleMutation::FlipJ2Sign,
                IrreducibleMutation::FlipJ3Sign,
                IrreducibleMutation::OmitJ1,
                IrreducibleMutation::OmitJ2,
                IrreducibleMutation::OmitJ3,
                IrreducibleMutation::TransposeAntisymmetricProjector,
            ] {
                let mutated = irreducible_tensor_unrenormalized_with_mutation(
                    &fixture(),
                    loop_type,
                    TensorLoopConfig::unit_natural(),
                    &QuadratureConfig::fast(),
                    mutation,
                )
                .expect("mutated tensor");
                let difference = baseline
                    .components()
                    .iter()
                    .zip(mutated.components().iter())
                    .map(|(left, right)| (*left - *right).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                if output_path.is_some() {
                    writeln!(retained_output, "\n[[mutation]]").expect("write mutation record");
                    writeln!(retained_output, "loop_type = \"{loop_type:?}\"")
                        .expect("write mutation loop");
                    writeln!(retained_output, "mutation = \"{mutation:?}\"")
                        .expect("write mutation name");
                    writeln!(
                        retained_output,
                        "rank_three_l2_difference = {:.17e}",
                        difference
                    )
                    .expect("write mutation difference");
                    writeln!(retained_output, "detected = {}", difference > 1.0e-10)
                        .expect("write mutation verdict");
                }
                assert!(
                    difference > 1.0e-10,
                    "mutation {mutation:?} was not detected in the rank-three tensor for {loop_type:?}: {difference}"
                );
            }
        }

        let kinematics = fixture();
        let tensor = irreducible_integrand(
            &kinematics,
            LoopType::Spinor,
            0.8,
            0.27,
            TensorLoopConfig::unit_natural(),
            IrreducibleMutation::None,
        )
        .expect("baseline tensor for input controls")
        .total;
        let bad_k = kinematics.k.map(|value| value + Complex64::new(0.07, 0.0));
        let bad_kinematics = WardKinematics::new(
            bad_k,
            -bad_k,
            kinematics.epsilon,
            kinematics.epsilon0,
            kinematics.zeta0,
            kinematics.field_strength,
            super::super::tensor_types::bilinear_dot(&bad_k, &bad_k),
            super::super::tensor_types::ShellMode::OffShell,
            super::super::tensor_types::MomentumRule::ConstantBackgroundConversion,
            false,
            1.0e-12,
        )
        .expect("bad momentum fixture");
        let baseline_residual = residual_for_tensor(
            &tensor,
            &kinematics,
            Diagram::Irreducible,
            ResidualTolerance::new(1.0, 1.0).expect("wide input-control tolerance"),
        )
        .expect("baseline input-control residual");
        let bad_momentum_residual = residual_for_tensor(
            &tensor,
            &bad_kinematics,
            Diagram::Irreducible,
            ResidualTolerance::new(1.0, 1.0).expect("wide input-control tolerance"),
        )
        .expect("bad-momentum residual");
        let momentum_difference =
            (baseline_residual.absolute_norm - bad_momentum_residual.absolute_norm).abs();
        let mut non_antisymmetric_field = kinematics.field_strength;
        non_antisymmetric_field[(1, 0)] = non_antisymmetric_field[(0, 1)];
        let invalid_field = WardKinematics::new(
            kinematics.k,
            kinematics.k0,
            kinematics.epsilon,
            kinematics.epsilon0,
            kinematics.zeta0,
            non_antisymmetric_field,
            kinematics.declared_virtuality,
            kinematics.shell_mode,
            kinematics.momentum_rule,
            kinematics.require_zeta_transversality,
            kinematics.validation_tolerance,
        );
        let field_rejected = matches!(
            invalid_field,
            Err(super::super::tensor_types::KinematicsError::FieldStrengthNotAntisymmetric)
        );
        if output_path.is_some() {
            writeln!(retained_output, "\n[input_control]").expect("write input control");
            writeln!(
                retained_output,
                "control = \"nonmatching_contraction_momentum\""
            )
            .expect("write momentum control");
            writeln!(
                retained_output,
                "absolute_norm_difference = {:.17e}",
                momentum_difference
            )
            .expect("write momentum difference");
            writeln!(
                retained_output,
                "detected = {}",
                momentum_difference > 1.0e-12
            )
            .expect("write momentum verdict");
            writeln!(retained_output, "\n[input_control]").expect("write field control");
            writeln!(
                retained_output,
                "control = \"non_antisymmetric_field_strength\""
            )
            .expect("write field control");
            writeln!(
                retained_output,
                "rejected_by_kinematic_validation = {field_rejected}"
            )
            .expect("write field verdict");
            writeln!(retained_output, "\n[p0_control]").expect("write P0 control");
            writeln!(
                retained_output,
                "control = \"audit_c820_on_shell_factor_hides_nonzero_integrand\""
            )
            .expect("write P0 control");
            writeln!(retained_output, "status = \"retained_and_passing\"")
                .expect("write P0 status");
        }
        assert!(momentum_difference > 1.0e-12);
        assert!(field_rejected);
        if let Some(output_path) = output_path {
            fs::write(&output_path, retained_output).expect("retain mutation evidence");
            println!("p1_mutation_output={output_path}");
        }
    }
}
