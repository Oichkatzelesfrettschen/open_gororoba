//! Full external-leg tensor and its explicit off-shell boundary.
//!
//! Equations 6.1 through 6.3 use the complete renormalized
//! vacuum-polarization tensor and the effective vector
//! `v_F = -i*kappa*{epsilon0,F}*k/k^2`. A zero virtuality is reported as a
//! singular evaluation; no denominator floor is introduced.

use num_complex::Complex64;

use super::{
    quadrature::QuadratureConfig,
    tensor_integrands::{
        TensorEvaluationError, TensorLoopConfig, bilinear, right_contract, validate_tensor_inputs,
    },
    tensor_types::{
        ComplexFourVector, ComplexLorentzMatrix, ComplexRankThreeTensor, WardKinematics,
    },
    types::LoopType,
    vacuum_pol_tensor::vacuum_polarization_tensor_renormalized,
    worldline_tensor::identity,
};

pub fn effective_polarization_vector(
    kinematics: &WardKinematics,
    loop_config: TensorLoopConfig,
) -> Result<ComplexFourVector, TensorEvaluationError> {
    let (loop_config, _) = validate_tensor_inputs(
        kinematics,
        loop_config,
        &QuadratureConfig {
            n_u: 1,
            n_t: 1,
            t_min: 1.0e-4,
            t_max: 1.0,
        },
    )?;
    let k_squared = bilinear(&kinematics.k, &identity(), &kinematics.k);
    if k_squared.norm() <= kinematics.validation_tolerance {
        return Err(TensorEvaluationError::ExternalOnShellSingularity);
    }
    let anticommutator = kinematics.epsilon0 * kinematics.field_strength
        + kinematics.field_strength * kinematics.epsilon0;
    let numerator = right_contract(&anticommutator, &kinematics.k);
    Ok(numerator.map(|value| value * Complex64::new(0.0, -loop_config.kappa) / k_squared))
}

pub fn external_tensor_off_shell(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<ComplexRankThreeTensor, TensorEvaluationError> {
    let (loop_config, _) = validate_tensor_inputs(kinematics, loop_config, quadrature)?;
    let k_squared = bilinear(&kinematics.k, &identity(), &kinematics.k);
    if k_squared.norm() <= kinematics.validation_tolerance {
        return Err(TensorEvaluationError::ExternalOnShellSingularity);
    }
    let vacuum_polarization =
        vacuum_polarization_tensor_renormalized(kinematics, loop_type, loop_config, quadrature)?;
    let mut tensor = ComplexRankThreeTensor::from_fn(|_, _, _| Complex64::new(0.0, 0.0));
    for mu in 0..4 {
        for nu in 0..4 {
            let mut basis = ComplexLorentzMatrix::zeros();
            basis[(mu, nu)] = Complex64::new(1.0, 0.0);
            let anticommutator =
                basis * kinematics.field_strength + kinematics.field_strength * basis;
            let effective_numerator = right_contract(&anticommutator, &kinematics.k);
            for alpha in 0..4 {
                let mut component = Complex64::new(0.0, 0.0);
                for beta in 0..4 {
                    component += effective_numerator[beta] * vacuum_polarization[(beta, alpha)];
                }
                tensor.set(
                    mu,
                    nu,
                    alpha,
                    component * Complex64::new(0.0, -loop_config.kappa) / k_squared,
                );
            }
        }
    }
    Ok(symmetrize(tensor))
}

pub fn external_amplitude_off_shell(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<Complex64, TensorEvaluationError> {
    let tensor = external_tensor_off_shell(kinematics, loop_type, loop_config, quadrature)?;
    let gravitational_contraction = tensor.contract_graviton(&kinematics.epsilon0);
    Ok(gravitational_contraction
        .iter()
        .zip(kinematics.epsilon.iter())
        .map(|(left, right)| *left * *right)
        .sum())
}

fn symmetrize(tensor: ComplexRankThreeTensor) -> ComplexRankThreeTensor {
    ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        (tensor.get(mu, nu, alpha) + tensor.get(nu, mu, alpha)) * Complex64::new(0.5, 0.0)
    })
}

#[cfg(test)]
mod tests {
    use super::super::tensor_types::{MomentumRule, ShellMode};
    use super::*;
    use num_complex::Complex64;

    fn fixture(on_shell: bool) -> WardKinematics {
        let mut field = ComplexLorentzMatrix::zeros();
        field[(0, 1)] = Complex64::new(0.1, 0.0);
        field[(1, 0)] = Complex64::new(-0.1, 0.0);
        let k = if on_shell {
            ComplexFourVector::from([
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 1.0),
            ])
        } else {
            ComplexFourVector::from([
                Complex64::new(0.15, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(0.0, 0.0),
            ])
        };
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
            if on_shell {
                Complex64::new(0.0, 0.0)
            } else {
                Complex64::new(0.0625, 0.0)
            },
            if on_shell {
                ShellMode::OnShell
            } else {
                ShellMode::OffShell
            },
            MomentumRule::ConstantBackgroundConversion,
            false,
            1.0e-12,
        )
        .expect("valid external fixture")
    }

    #[test]
    fn effective_vector_rejects_zero_virtuality_without_a_floor() {
        let result =
            effective_polarization_vector(&fixture(true), TensorLoopConfig::unit_natural());
        assert_eq!(
            result,
            Err(TensorEvaluationError::ExternalOnShellSingularity)
        );
    }

    #[test]
    fn external_tensor_is_finite_off_shell() {
        let result = external_tensor_off_shell(
            &fixture(false),
            LoopType::Scalar,
            TensorLoopConfig::unit_natural(),
            &QuadratureConfig::fast(),
        )
        .expect("off-shell external tensor");
        assert!(
            result
                .components()
                .iter()
                .all(|component| component.re.is_finite() && component.im.is_finite())
        );
    }
}
