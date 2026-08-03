//! Source one-photon amplitude used by the gravitational Ward identity.
//!
//! Equations 3.10 and 3.11 are distributional in momentum. The returned
//! value retains the coefficient of the momentum delta function and marks
//! whether the requested momentum is its support; it does not turn the delta
//! function into an undocumented finite scalar.

use super::{
    quadrature::{QuadratureConfig, gl_integrate_complex},
    tensor_integrands::{
        TensorEvaluationError, TensorLoopConfig, bilinear, magnetic_field_strength,
        scalar_determinant,
    },
    tensor_types::{ComplexFourVector, ComplexLorentzMatrix},
    types::LoopType,
    worldline_tensor::{g_minus, g_plus, r_plus},
};
use num_complex::Complex64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnePhotonAmplitude {
    pub momentum: ComplexFourVector,
    pub polarization: ComplexFourVector,
    pub coefficient: Complex64,
    pub momentum_delta_support: bool,
}

pub fn one_photon_amplitude(
    field_strength: &ComplexLorentzMatrix,
    momentum: ComplexFourVector,
    polarization: ComplexFourVector,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<OnePhotonAmplitude, TensorEvaluationError> {
    let magnetic_field = magnetic_field_strength(field_strength)?;
    let loop_config = loop_config.validate()?;
    if quadrature.n_t == 0
        || !quadrature.t_min.is_finite()
        || !quadrature.t_max.is_finite()
        || quadrature.t_min <= 0.0
        || quadrature.t_max <= quadrature.t_min
    {
        return Err(TensorEvaluationError::InvalidQuadrature);
    }
    let coefficient = gl_integrate_complex(
        |proper_time| {
            let z = loop_config.charge * magnetic_field * proper_time;
            let q_kernel = one_photon_kernel(loop_type, z);
            let exponent_matrix = bosonic_exponent_matrix(z, proper_time);
            let exponent =
                (proper_time * 0.25 * bilinear(&momentum, &exponent_matrix, &momentum)).exp();
            let prefactor = match loop_type {
                LoopType::Scalar => Complex64::new(0.0, -loop_config.charge),
                LoopType::Spinor => Complex64::new(0.0, 2.0 * loop_config.charge),
            };
            let contraction = bilinear(&polarization, &q_kernel, &momentum);
            let determinant = scalar_determinant(loop_type, z);
            let measure = proper_time.powi(-2) * (-loop_config.mass.powi(2) * proper_time).exp();
            prefactor * contraction * exponent * Complex64::new(determinant * measure, 0.0)
        },
        quadrature.t_min,
        quadrature.t_max,
        quadrature.n_t,
    );
    if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
        return Err(TensorEvaluationError::NonFiniteResult);
    }
    Ok(OnePhotonAmplitude {
        momentum_delta_support: momentum.norm() <= 1.0e-12,
        momentum,
        polarization,
        coefficient,
    })
}

fn one_photon_kernel(loop_type: LoopType, z: f64) -> ComplexLorentzMatrix {
    let coefficient = match loop_type {
        LoopType::Scalar => scalar_kernel_coefficient(z),
        LoopType::Spinor => spinor_kernel_coefficient(z),
    };
    r_plus().map(|value| value * Complex64::new(coefficient, 0.0))
}

fn bosonic_exponent_matrix(z: f64, _proper_time: f64) -> ComplexLorentzMatrix {
    let plus_coefficient = if z.abs() < 1.0e-6 {
        -1.0 / 3.0 + z.powi(2) / 45.0 - 2.0 * z.powi(4) / 945.0
    } else {
        1.0 / z.powi(2) - coth(z) / z
    };
    g_minus().map(|value| value * Complex64::new(-1.0 / 3.0, 0.0))
        + g_plus().map(|value| value * Complex64::new(plus_coefficient, 0.0))
}

fn scalar_kernel_coefficient(z: f64) -> f64 {
    if z.abs() < 1.0e-6 {
        -z / 3.0 + z.powi(3) / 45.0 - 2.0 * z.powi(5) / 945.0
    } else {
        -coth(z) + 1.0 / z
    }
}

fn spinor_kernel_coefficient(z: f64) -> f64 {
    if z.abs() < 1.0e-6 {
        2.0 * z / 3.0 - 14.0 * z.powi(3) / 45.0 + 124.0 * z.powi(5) / 945.0
    } else {
        -coth(z) + tanh(z) + 1.0 / z
    }
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

    fn field() -> ComplexLorentzMatrix {
        let mut field = ComplexLorentzMatrix::zeros();
        field[(0, 1)] = Complex64::new(0.1, 0.0);
        field[(1, 0)] = Complex64::new(-0.1, 0.0);
        field
    }

    #[test]
    fn zero_momentum_retains_distributional_disposition() {
        let momentum = ComplexFourVector::zeros();
        let polarization = ComplexFourVector::from([
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let result = one_photon_amplitude(
            &field(),
            momentum,
            polarization,
            LoopType::Spinor,
            TensorLoopConfig::unit_natural(),
            &QuadratureConfig::fast(),
        )
        .expect("zero-momentum coefficient");
        assert!(result.momentum_delta_support);
        assert!(result.coefficient.norm() < 1.0e-14);
    }

    #[test]
    fn one_photon_coefficient_is_finite_away_from_zero_momentum() {
        let momentum = ComplexFourVector::from([
            Complex64::new(0.2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.1, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let polarization = ComplexFourVector::from([
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let result = one_photon_amplitude(
            &field(),
            momentum,
            polarization,
            LoopType::Scalar,
            TensorLoopConfig::unit_natural(),
            &QuadratureConfig::fast(),
        )
        .expect("finite coefficient");
        assert!(!result.momentum_delta_support);
        assert!(result.coefficient.re.is_finite() && result.coefficient.im.is_finite());
    }

    #[test]
    fn weak_field_exponent_has_the_source_zero_field_limit() {
        let matrix = bosonic_exponent_matrix(0.0, 1.0);
        let expected =
            ComplexLorentzMatrix::identity().map(|value| value * Complex64::new(-1.0 / 3.0, 0.0));
        assert_eq!(matrix, expected);
    }
}
