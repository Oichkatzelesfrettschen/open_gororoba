//! Unprojected vacuum-polarization tensors for the Ward campaign.
//!
//! Equations 3.14, 3.20, 3.22, and 3.23 are evaluated with explicit Lorentz
//! indices. The legacy parallel and perpendicular scalar channels are not
//! used by these functions.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::{
    tensor_integrands::{
        TensorEvaluationError, TensorLoopConfig, bilinear, double_integrate_matrix, even,
        matrix_is_finite, outer, scalar_determinant, source_worldline_node, validate_tensor_inputs,
    },
    tensor_types::{ComplexFourVector, ComplexLorentzMatrix, WardKinematics},
    types::LoopType,
};

use super::quadrature::QuadratureConfig;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VacuumPolarizationNode {
    pub raw_tensor: ComplexLorentzMatrix,
    pub symmetric_tensor: ComplexLorentzMatrix,
    pub exponent: Complex64,
    pub determinant_factor: f64,
    pub renormalization_counterterm: ComplexLorentzMatrix,
}

/// Build the full unintegrated tensor from the source worldline matrices.
pub fn vacuum_polarization_integrand(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    proper_time: f64,
    u: f64,
    loop_config: TensorLoopConfig,
) -> Result<VacuumPolarizationNode, TensorEvaluationError> {
    let (loop_config, magnetic_field) = validate_tensor_inputs(
        kinematics,
        loop_config,
        &QuadratureConfig {
            n_u: 1,
            n_t: 1,
            t_min: proper_time,
            t_max: proper_time + 1.0,
        },
    )?;
    if !proper_time.is_finite() || proper_time <= 0.0 || !u.is_finite() || !(0.0..=1.0).contains(&u)
    {
        return Err(TensorEvaluationError::InvalidQuadrature);
    }
    let node = source_worldline_node(magnetic_field, loop_config.charge, proper_time, u)?;
    let k = &kinematics.k;
    let raw_tensor = match loop_type {
        LoopType::Scalar => scalar_vacuum_integrand(&node, k),
        LoopType::Spinor => spinor_vacuum_integrand(&node, k),
    };
    let symmetric_tensor = even(&raw_tensor);
    let exponent = -bilinear(k, &node.bar_g_b12, k);
    let exponent = exponent.exp();
    let determinant_factor =
        scalar_determinant(loop_type, loop_config.charge * magnetic_field * proper_time);
    let k_squared = bilinear(k, &super::worldline_tensor::identity(), k);
    let transverse =
        super::worldline_tensor::identity().map(|value| value * k_squared) - outer(k, k);
    let v = 1.0 - 2.0 * u;
    let renormalization_counterterm = match loop_type {
        LoopType::Scalar => transverse.map(|value| value * (-v * v)),
        LoopType::Spinor => transverse.map(|value| value * (4.0 * u * (1.0 - u))),
    };
    if !matrix_is_finite(&symmetric_tensor) || !exponent.re.is_finite() || !exponent.im.is_finite()
    {
        return Err(TensorEvaluationError::NonFiniteResult);
    }
    Ok(VacuumPolarizationNode {
        raw_tensor,
        symmetric_tensor,
        exponent,
        determinant_factor,
        renormalization_counterterm,
    })
}

/// Integrate the unrenormalized rank-two tensor in the common regulated domain.
pub fn vacuum_polarization_tensor_unrenormalized(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<ComplexLorentzMatrix, TensorEvaluationError> {
    let (loop_config, magnetic_field) =
        validate_tensor_inputs(kinematics, loop_config, quadrature)?;
    let kinematics = *kinematics;
    let tensor = double_integrate_matrix(
        |proper_time, u| {
            let node = source_worldline_node(magnetic_field, loop_config.charge, proper_time, u)?;
            let raw = match loop_type {
                LoopType::Scalar => scalar_vacuum_integrand(&node, &kinematics.k),
                LoopType::Spinor => spinor_vacuum_integrand(&node, &kinematics.k),
            };
            let tensor = even(&raw);
            let exponent = (-bilinear(&kinematics.k, &node.bar_g_b12, &kinematics.k)).exp();
            let determinant =
                scalar_determinant(loop_type, loop_config.charge * magnetic_field * proper_time);
            Ok(tensor.map(|value| value * Complex64::new(determinant, 0.0) * exponent))
        },
        loop_config.mass,
        1.0,
        quadrature,
    )?;
    let prefactor = match loop_type {
        LoopType::Scalar => -loop_config.charge.powi(2) / (16.0 * PI * PI),
        LoopType::Spinor => loop_config.charge.powi(2) / (8.0 * PI * PI),
    };
    Ok(tensor.map(|value| value * Complex64::new(prefactor, 0.0)))
}

/// Integrate the renormalized tensor using the source's zero-field subtraction.
pub fn vacuum_polarization_tensor_renormalized(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<ComplexLorentzMatrix, TensorEvaluationError> {
    let (loop_config, _magnetic_field) =
        validate_tensor_inputs(kinematics, loop_config, quadrature)?;
    let kinematics = *kinematics;
    let tensor = double_integrate_matrix(
        |proper_time, u| {
            let source =
                vacuum_polarization_integrand(&kinematics, loop_type, proper_time, u, loop_config)?;
            let source_term = source.symmetric_tensor.map(|value| {
                value * Complex64::new(source.determinant_factor, 0.0) * source.exponent
            });
            Ok(source_term + source.renormalization_counterterm)
        },
        loop_config.mass,
        1.0,
        quadrature,
    )?;
    let prefactor = match loop_type {
        LoopType::Scalar => -loop_config.charge.powi(2) / (16.0 * PI * PI),
        LoopType::Spinor => loop_config.charge.powi(2) / (8.0 * PI * PI),
    };
    Ok(tensor.map(|value| value * Complex64::new(prefactor, 0.0)))
}

fn scalar_vacuum_integrand(
    node: &super::tensor_integrands::SourceWorldlineNode,
    k: &ComplexFourVector,
) -> ComplexLorentzMatrix {
    let k_dot_dot_g_b_k = bilinear(k, &node.full.dot_g_b, k);
    let mut tensor = ComplexLorentzMatrix::zeros();
    for mu in 0..4 {
        for nu in 0..4 {
            let mut value = node.full.dot_g_b[(mu, nu)] * k_dot_dot_g_b_k;
            for lambda in 0..4 {
                for kappa in 0..4 {
                    value += node.bar_dot_g_b12[(mu, lambda)]
                        * node.bar_dot_g_b21[(nu, kappa)]
                        * k[lambda]
                        * k[kappa];
                }
            }
            tensor[(mu, nu)] = value;
        }
    }
    tensor
}

fn spinor_vacuum_integrand(
    node: &super::tensor_integrands::SourceWorldlineNode,
    k: &ComplexFourVector,
) -> ComplexLorentzMatrix {
    let k_dot_dot_g_b_k = bilinear(k, &node.full.dot_g_b, k);
    let k_dot_g_f_k = bilinear(k, &node.full.g_f, k);
    let gf21 = -node.full.g_f.transpose();
    let left_factor = node.coincidence.dot_g_b - node.coincidence.g_f - node.full.dot_g_b;
    let right_factor = -node.full.dot_g_b.transpose() - node.coincidence.dot_g_b + node.full.g_f;
    let mut tensor = ComplexLorentzMatrix::zeros();
    for mu in 0..4 {
        for nu in 0..4 {
            let mut value = node.full.dot_g_b[(mu, nu)] * k_dot_dot_g_b_k
                - node.full.g_f[(mu, nu)] * k_dot_g_f_k;
            for lambda in 0..4 {
                for kappa in 0..4 {
                    value -= (left_factor[(mu, lambda)] * right_factor[(nu, kappa)]
                        + node.full.g_f[(mu, lambda)] * gf21[(nu, kappa)])
                        * k[lambda]
                        * k[kappa];
                }
            }
            tensor[(mu, nu)] = value;
        }
    }
    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photon_graviton::tensor_types::{
        ComplexLorentzMatrix, MomentumRule, ShellMode, WardKinematics,
    };

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
        .expect("valid tensor fixture")
    }

    #[test]
    fn source_tensor_has_full_lorentz_components() {
        let node = vacuum_polarization_integrand(
            &fixture(),
            LoopType::Spinor,
            0.8,
            0.27,
            TensorLoopConfig::unit_natural(),
        )
        .expect("valid node");
        assert!(matrix_is_finite(&node.raw_tensor));
        assert!(node.raw_tensor.iter().any(|value| value.norm() > 1.0e-14));
        assert!(node.symmetric_tensor[(0, 2)].norm() > 0.0);
    }

    #[test]
    fn renormalized_and_unrenormalized_paths_are_distinct() {
        let node = vacuum_polarization_integrand(
            &fixture(),
            LoopType::Scalar,
            0.8,
            0.27,
            TensorLoopConfig::unit_natural(),
        )
        .expect("valid node");
        assert!(node.renormalization_counterterm.norm() > 0.0);
    }

    #[test]
    fn tensor_vacuum_polarization_integrates_deterministically() {
        let quadrature = QuadratureConfig::fast();
        let first = vacuum_polarization_tensor_renormalized(
            &fixture(),
            LoopType::Scalar,
            TensorLoopConfig::unit_natural(),
            &quadrature,
        )
        .expect("first integration");
        let second = vacuum_polarization_tensor_renormalized(
            &fixture(),
            LoopType::Scalar,
            TensorLoopConfig::unit_natural(),
            &quadrature,
        )
        .expect("second integration");
        assert_eq!(first, second);
    }

    #[test]
    fn transverse_subtraction_is_not_a_projector_guess() {
        let node = vacuum_polarization_integrand(
            &fixture(),
            LoopType::Scalar,
            0.8,
            0.27,
            TensorLoopConfig::unit_natural(),
        )
        .expect("valid node");
        let k = fixture().k;
        let k2 = bilinear(&k, &super::super::worldline_tensor::identity(), &k);
        let expected =
            super::super::worldline_tensor::identity().map(|value| value * k2) - outer(&k, &k);
        let v = 1.0 - 2.0 * 0.27;
        let expected = expected.map(|value| value * (-v * v));
        assert_eq!(node.renormalization_counterterm, expected);
    }
}
