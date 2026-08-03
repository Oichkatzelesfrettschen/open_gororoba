//! Rank-three tadpole tensor from the full trace input.
//!
//! Equations 5.2 and 5.5 retain the arbitrary symmetric graviton
//! polarization and the complete photon field-strength matrix. The control
//! epsilon = k is represented by the same field-strength constructor, so its
//! vanishing is a consequence of antisymmetry rather than a returned zero.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::{
    quadrature::QuadratureConfig,
    tensor_integrands::{
        TensorEvaluationError, TensorLoopConfig, double_integrate_rank_three, left_contract,
        matrix_is_finite, rank_three_scale, scalar_determinant, validate_tensor_inputs,
    },
    tensor_types::{
        ComplexFourVector, ComplexLorentzMatrix, ComplexRankThreeTensor, WardKinematics,
    },
    types::LoopType,
    worldline_tensor::r_plus,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TadpoleNode {
    pub photon_field_strength: ComplexLorentzMatrix,
    pub kernel: ComplexLorentzMatrix,
    pub renormalization_kernel: ComplexLorentzMatrix,
    pub trace_input: ComplexLorentzMatrix,
    pub unrenormalized_trace: Complex64,
    pub renormalized_trace: Complex64,
    pub tensor: ComplexRankThreeTensor,
    pub determinant_factor: f64,
}

pub fn photon_field_strength(
    photon_momentum: &ComplexFourVector,
    photon_polarization: &ComplexFourVector,
) -> ComplexLorentzMatrix {
    ComplexLorentzMatrix::from_fn(|row, column| {
        photon_momentum[row] * photon_polarization[column]
            - photon_polarization[row] * photon_momentum[column]
    })
}

pub fn tadpole_integrand(
    kinematics: &WardKinematics,
    photon_field_strength: ComplexLorentzMatrix,
    loop_type: LoopType,
    proper_time: f64,
    loop_config: TensorLoopConfig,
) -> Result<TadpoleNode, TensorEvaluationError> {
    let (_, magnetic_field) = validate_tensor_inputs(
        kinematics,
        loop_config,
        &QuadratureConfig {
            n_u: 1,
            n_t: 1,
            t_min: proper_time,
            t_max: proper_time + 1.0,
        },
    )?;
    if !proper_time.is_finite() || proper_time <= 0.0 {
        return Err(TensorEvaluationError::InvalidQuadrature);
    }
    let z = loop_config.charge * magnetic_field * proper_time;
    let kernel = tadpole_kernel(loop_type, z);
    let z_matrix = r_plus().map(|value| value * Complex64::new(z, 0.0));
    let renormalization_kernel = match loop_type {
        LoopType::Scalar => z_matrix.map(|value| value * Complex64::new(-1.0 / 3.0, 0.0)),
        LoopType::Spinor => z_matrix.map(|value| value * Complex64::new(2.0 / 3.0, 0.0)),
    };
    let trace_input = kernel * kinematics.epsilon0 * photon_field_strength;
    let unrenormalized_trace = trace(&trace_input);
    let renormalized_trace = trace(
        &(kernel.map(|value| value * Complex64::new(scalar_determinant(loop_type, z), 0.0))
            * kinematics.epsilon0
            * photon_field_strength
            - renormalization_kernel * kinematics.epsilon0 * photon_field_strength),
    );
    let tensor = tadpole_tensor_from_kernel(&kernel, &kinematics.k);
    let determinant_factor = scalar_determinant(loop_type, z);
    if !matrix_is_finite(&trace_input)
        || !unrenormalized_trace.re.is_finite()
        || !renormalized_trace.re.is_finite()
    {
        return Err(TensorEvaluationError::NonFiniteResult);
    }
    Ok(TadpoleNode {
        photon_field_strength,
        kernel,
        renormalization_kernel,
        trace_input,
        unrenormalized_trace,
        renormalized_trace,
        tensor,
        determinant_factor,
    })
}

pub fn tadpole_tensor_unrenormalized(
    kinematics: &WardKinematics,
    _photon_field_strength: ComplexLorentzMatrix,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<ComplexRankThreeTensor, TensorEvaluationError> {
    let (loop_config, magnetic_field) =
        validate_tensor_inputs(kinematics, loop_config, quadrature)?;
    let kinematics = *kinematics;
    let tensor = double_integrate_rank_three(
        |proper_time, _u| {
            let z = loop_config.charge * magnetic_field * proper_time;
            let kernel = tadpole_kernel(loop_type, z);
            let determinant = scalar_determinant(loop_type, z);
            Ok(rank_three_scale(
                &tadpole_tensor_from_kernel(&kernel, &kinematics.k),
                Complex64::new(determinant, 0.0),
            ))
        },
        loop_config.mass,
        2.0,
        quadrature,
    )?;
    Ok(rank_three_scale(
        &tensor,
        tadpole_prefactor(loop_type, loop_config),
    ))
}

pub fn tadpole_tensor_renormalized(
    kinematics: &WardKinematics,
    _photon_field_strength: ComplexLorentzMatrix,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<ComplexRankThreeTensor, TensorEvaluationError> {
    let (loop_config, magnetic_field) =
        validate_tensor_inputs(kinematics, loop_config, quadrature)?;
    let kinematics = *kinematics;
    let tensor = double_integrate_rank_three(
        |proper_time, _u| {
            let z = loop_config.charge * magnetic_field * proper_time;
            let kernel = tadpole_kernel(loop_type, z);
            let z_matrix = r_plus().map(|value| value * Complex64::new(z, 0.0));
            let renormalization_kernel = match loop_type {
                LoopType::Scalar => z_matrix.map(|value| value * Complex64::new(-1.0 / 3.0, 0.0)),
                LoopType::Spinor => z_matrix.map(|value| value * Complex64::new(2.0 / 3.0, 0.0)),
            };
            let determinant = scalar_determinant(loop_type, z);
            let effective_kernel = kernel.map(|value| value * Complex64::new(determinant, 0.0))
                - renormalization_kernel;
            Ok(rank_three_scale(
                &tadpole_tensor_from_kernel(&effective_kernel, &kinematics.k),
                Complex64::new(1.0, 0.0),
            ))
        },
        loop_config.mass,
        2.0,
        quadrature,
    )?;
    Ok(rank_three_scale(
        &tensor,
        tadpole_prefactor(loop_type, loop_config),
    ))
}

fn tadpole_prefactor(loop_type: LoopType, loop_config: TensorLoopConfig) -> Complex64 {
    let spinor = Complex64::new(
        0.0,
        loop_config.charge * loop_config.kappa / (16.0 * PI * PI),
    );
    match loop_type {
        LoopType::Spinor => spinor,
        LoopType::Scalar => spinor * Complex64::new(-0.5, 0.0),
    }
}

fn tadpole_kernel(loop_type: LoopType, z: f64) -> ComplexLorentzMatrix {
    let coefficient = match loop_type {
        LoopType::Scalar => {
            if z.abs() < 1.0e-6 {
                -z / 3.0 + z.powi(3) / 45.0 - 2.0 * z.powi(5) / 945.0
            } else {
                -(coth(z) - 1.0 / z)
            }
        }
        LoopType::Spinor => {
            if z.abs() < 1.0e-6 {
                2.0 * z / 3.0 - 14.0 * z.powi(3) / 45.0 + 124.0 * z.powi(5) / 945.0
            } else {
                -coth(z) + tanh(z) + 1.0 / z
            }
        }
    };
    r_plus().map(|value| value * Complex64::new(coefficient, 0.0))
}

fn tadpole_tensor_from_kernel(
    kernel: &ComplexLorentzMatrix,
    k: &ComplexFourVector,
) -> ComplexRankThreeTensor {
    let k_left = left_contract(k, kernel);
    let tensor = ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        kernel[(alpha, mu)] * k[nu]
            - if nu == alpha {
                k_left[mu]
            } else {
                Complex64::new(0.0, 0.0)
            }
    });
    symmetrize(tensor)
}

fn symmetrize(tensor: ComplexRankThreeTensor) -> ComplexRankThreeTensor {
    ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        (tensor.get(mu, nu, alpha) + tensor.get(nu, mu, alpha)) * Complex64::new(0.5, 0.0)
    })
}

fn trace(matrix: &ComplexLorentzMatrix) -> Complex64 {
    (0..4).map(|index| matrix[(index, index)]).sum()
}

fn coth(value: f64) -> f64 {
    if value.abs() < 1.0e-6 {
        1.0 / value + value / 3.0 - value.powi(3) / 45.0 + 2.0 * value.powi(5) / 945.0
    } else {
        1.0 / value.tanh()
    }
}

fn tanh(value: f64) -> f64 {
    if value.abs() < 1.0e-6 {
        value - value.powi(3) / 3.0 + 2.0 * value.powi(5) / 15.0
    } else {
        value.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn fixture() -> (WardKinematics, ComplexFourVector) {
        let mut field = ComplexLorentzMatrix::zeros();
        field[(0, 1)] = Complex64::new(0.1, 0.0);
        field[(1, 0)] = Complex64::new(-0.1, 0.0);
        let k = ComplexFourVector::from([
            Complex64::new(0.15, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let kinematics = WardKinematics::new(
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
            super::super::tensor_types::ShellMode::OffShell,
            super::super::tensor_types::MomentumRule::ConstantBackgroundConversion,
            false,
            1.0e-12,
        )
        .expect("valid tensor fixture");
        let epsilon = ComplexFourVector::from([
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        (kinematics, epsilon)
    }

    #[test]
    fn epsilon_equals_k_gives_zero_field_strength() {
        let (kinematics, _) = fixture();
        let field = photon_field_strength(&kinematics.k, &kinematics.k);
        assert!(field.iter().all(|component| component.norm() == 0.0));
    }

    #[test]
    fn tadpole_node_retains_trace_inputs_and_rank() {
        let (kinematics, epsilon) = fixture();
        let field = photon_field_strength(&kinematics.k, &epsilon);
        let node = tadpole_integrand(
            &kinematics,
            field,
            LoopType::Spinor,
            0.8,
            TensorLoopConfig::unit_natural(),
        )
        .expect("valid tadpole node");
        assert!(node.trace_input.iter().any(|value| value.norm() > 0.0));
        assert!(
            node.tensor
                .components()
                .iter()
                .any(|value| value.norm() > 0.0)
        );
    }

    #[test]
    fn tadpole_gauge_contraction_is_structural() {
        let (kinematics, _) = fixture();
        let field = photon_field_strength(&kinematics.k, &kinematics.k);
        let node = tadpole_integrand(
            &kinematics,
            field,
            LoopType::Scalar,
            0.8,
            TensorLoopConfig::unit_natural(),
        )
        .expect("valid tadpole node");
        let residual = node.tensor.contract_photon(&kinematics.k);
        assert!(residual.iter().all(|value| value.norm() < 1.0e-14));
    }
}
