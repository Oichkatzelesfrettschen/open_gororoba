// SPDX-License-Identifier: MIT
//! Coupled classical mechanical storage and sampled Gaussian dephasing.
//!
//! These finite-dimensional models supply conditional predictions, rather than
//! material calibration, integration error bounds, or quantum-bath evidence.
//! Ordered oscillator line weights retain a separate quantum-noise contract.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
#[path = "vacuum_response/harmonic.rs"]
mod harmonic;
pub use harmonic::{HarmonicPulseSequence, thermal_harmonic_coherence};

use crate::vacuum_audit::{OscillatorLineWeights, oscillator_line_weights};

/// Invalid model domains or unrepresentable arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseError {
    Shape,
    Domain,
    Asymmetric,
    Indefinite,
    Arithmetic,
}

/// Relative tolerance applied after dividing a matrix by its largest entry.
///
/// The tolerance bounds normalized entry asymmetry and negative eigenvalues.
/// Accepted matrices are symmetrized; accepted negative eigenvalues are set to
/// zero. Reported corrections identify that changed numerical model explicitly.
/// Positive-definite forms require minimum normalized eigenvalue > tolerance.
#[derive(Debug, Clone, Copy)]
pub struct MatrixTolerance(f64);

impl MatrixTolerance {
    pub fn new(relative: f64) -> Result<Self, ResponseError> {
        if relative.is_finite() && (0.0..=1e-6).contains(&relative) {
            Ok(Self(relative))
        } else {
            Err(ResponseError::Domain)
        }
    }
}

/// Largest entrywise change divided by the original largest absolute entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixCorrection {
    pub symmetry_relative: f64,
    pub psd_projection_relative: f64,
}

#[derive(Debug, Clone)]
struct QuadraticForm {
    normalized: DMatrix<f64>,
    scale: f64,
    correction: MatrixCorrection,
}

fn finite(value: f64) -> Result<f64, ResponseError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ResponseError::Arithmetic)
    }
}

fn preserve_nonzero(value: f64, original: f64) -> Result<f64, ResponseError> {
    let value = finite(value)?;
    if value == 0.0 && original != 0.0 {
        Err(ResponseError::Arithmetic)
    } else {
        Ok(value)
    }
}

fn checked_vector_division(
    vector: &DVector<f64>,
    divisor: f64,
) -> Result<DVector<f64>, ResponseError> {
    let mut divided = vector / divisor;
    for (value, original) in divided.iter_mut().zip(vector) {
        *value = preserve_nonzero(*value, *original)?;
    }
    Ok(divided)
}

fn half_storage(form: &QuadraticForm, coordinate: &DVector<f64>) -> Result<f64, ResponseError> {
    let quadratic = form.quadratic(coordinate)?;
    preserve_nonzero(0.5 * quadratic, quadratic)
}

fn checked_product(left: f64, right: f64) -> Result<f64, ResponseError> {
    let product = finite(left * right)?;
    if product == 0.0 && left != 0.0 && right != 0.0 {
        Err(ResponseError::Arithmetic)
    } else {
        Ok(product)
    }
}

fn checked_dot(left: &DVector<f64>, right: &DVector<f64>) -> Result<f64, ResponseError> {
    let mut sum = 0.0_f64;
    let mut correction = 0.0;
    for (left_value, right_value) in left.iter().zip(right) {
        let term = checked_product(*left_value, *right_value)?;
        let next = finite(sum + term)?;
        let roundoff = if sum.abs() >= term.abs() {
            finite(finite(sum - next)? + term)?
        } else {
            finite(finite(term - next)? + sum)?
        };
        correction = finite(correction + roundoff)?;
        sum = next;
    }
    finite(sum + correction)
}

fn positive(value: f64) -> Result<(), ResponseError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ResponseError::Domain)
    }
}

fn finite_vector(vector: &DVector<f64>) -> Result<(), ResponseError> {
    if vector.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ResponseError::Arithmetic)
    }
}

impl QuadraticForm {
    fn new(
        matrix: DMatrix<f64>,
        definite: bool,
        tolerance: MatrixTolerance,
    ) -> Result<Self, ResponseError> {
        if matrix.nrows() == 0 || !matrix.is_square() {
            return Err(ResponseError::Shape);
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(ResponseError::Domain);
        }
        let scale = matrix.amax();
        if scale == 0.0 {
            if definite {
                return Err(ResponseError::Indefinite);
            }
            return Ok(Self {
                normalized: matrix,
                scale: 0.0,
                correction: MatrixCorrection {
                    symmetry_relative: 0.0,
                    psd_projection_relative: 0.0,
                },
            });
        }
        let normalized = &matrix / scale;
        if matrix
            .iter()
            .zip(normalized.iter())
            .any(|(original, scaled)| *original != 0.0 && *scaled == 0.0)
        {
            return Err(ResponseError::Arithmetic);
        }
        let asymmetry = (&normalized - normalized.transpose()).amax();
        if asymmetry > tolerance.0 {
            return Err(ResponseError::Asymmetric);
        }
        let mut symmetric = (&normalized + normalized.transpose()) * 0.5;
        let symmetry_correction = (&symmetric - &normalized).amax();
        let eigen = SymmetricEigen::new(symmetric.clone());
        if eigen.eigenvalues.iter().any(|value| !value.is_finite())
            || eigen.eigenvectors.iter().any(|value| !value.is_finite())
        {
            return Err(ResponseError::Arithmetic);
        }
        let minimum = eigen.eigenvalues.min();
        if (definite && minimum <= tolerance.0) || minimum < -tolerance.0 {
            return Err(ResponseError::Indefinite);
        }
        let mut projection = 0.0;
        if minimum < 0.0 {
            let nonnegative = eigen.eigenvalues.map(|value| value.max(0.0));
            let projected = &eigen.eigenvectors
                * DMatrix::from_diagonal(&nonnegative)
                * eigen.eigenvectors.transpose();
            projection = (&projected - &symmetric).amax();
            symmetric = projected;
        }
        Ok(Self {
            normalized: symmetric,
            scale,
            correction: MatrixCorrection {
                symmetry_relative: symmetry_correction,
                psd_projection_relative: projection,
            },
        })
    }

    fn normalized_action(&self, vector: &DVector<f64>) -> Result<DVector<f64>, ResponseError> {
        let mut result = DVector::zeros(vector.len());
        for (index, row) in self.normalized.row_iter().enumerate() {
            result[index] = checked_dot(&row.transpose(), vector)?;
        }
        Ok(result)
    }

    fn apply(&self, vector: &DVector<f64>) -> Result<DVector<f64>, ResponseError> {
        let mut result = self.normalized_action(vector)?;
        for coordinate in result.iter_mut() {
            *coordinate = checked_product(*coordinate, self.scale)?;
        }
        Ok(result)
    }

    fn quadratic(&self, vector: &DVector<f64>) -> Result<f64, ResponseError> {
        let coordinate_scale = vector.amax();
        if coordinate_scale == 0.0 || self.scale == 0.0 {
            return Ok(0.0);
        }
        let coordinates = checked_vector_division(vector, coordinate_scale)?;
        let normalized_value = checked_dot(&coordinates, &self.normalized_action(&coordinates)?)?;
        if normalized_value < 0.0 {
            return Err(ResponseError::Arithmetic);
        }
        if normalized_value == 0.0 {
            return Ok(0.0);
        }
        let factors = [
            normalized_value,
            self.scale,
            coordinate_scale,
            coordinate_scale,
        ];
        let mut value: f64 = factors.iter().product();
        if !value.is_finite() || value == 0.0 {
            value = factors.iter().map(|factor| factor.ln()).sum::<f64>().exp();
        }
        finite(value)?;
        if value == 0.0 {
            return Err(ResponseError::Arithmetic);
        }
        Ok(value)
    }
}

/// One classical memory branch with auxiliary displacement q in metres.
///
/// qdot = v-q/tau; restoring force = K*q; K may be semidefinite. Every branch
/// uses the same generalized displacement coordinates as its parent mechanics.
#[derive(Debug, Clone)]
pub struct MemoryBranch {
    stiffness: QuadraticForm,
    relaxation_s: f64,
}

impl MemoryBranch {
    pub fn new(
        stiffness_n_per_m: DMatrix<f64>,
        relaxation_s: f64,
        tolerance: MatrixTolerance,
    ) -> Result<Self, ResponseError> {
        positive(relaxation_s)?;
        Ok(Self {
            stiffness: QuadraticForm::new(stiffness_n_per_m, false, tolerance)?,
            relaxation_s,
        })
    }

    pub fn correction(&self) -> MatrixCorrection {
        self.stiffness.correction
    }
}

/// Validated finite-dimensional local mechanical model, using SI coordinates.
///
/// M*vdot + Gamma*v + K*x + sum(K_j*q_j) = drive. M and K are positive
/// definite at the declared tolerance; Gamma and K_j are positive semidefinite.
/// The support is a local quadratic approximation, including any independently
/// supplied interaction curvature. The model establishes neither global
/// stability of a singular Casimir potential nor a CPTP quantum channel.
#[derive(Debug, Clone)]
pub struct QuadraticMechanics {
    mass: QuadraticForm,
    stiffness: QuadraticForm,
    damping: QuadraticForm,
    branches: Vec<MemoryBranch>,
}

/// Instantaneous model storage, derivatives and powers; integration is separate.
#[derive(Debug, Clone)]
pub struct MechanicalPower {
    pub stored_energy_j: f64,
    pub drive_power_w: f64,
    pub dissipation_w: f64,
    pub energy_rate_w: f64,
    pub balance_residual_w: f64,
    pub acceleration_m_per_s2: DVector<f64>,
    pub memory_rates_m_per_s: Vec<DVector<f64>>,
}

impl QuadraticMechanics {
    pub fn new(
        mass_kg: DMatrix<f64>,
        stiffness_n_per_m: DMatrix<f64>,
        damping_n_s_per_m: DMatrix<f64>,
        branches: Vec<MemoryBranch>,
        tolerance: MatrixTolerance,
    ) -> Result<Self, ResponseError> {
        let dimension = mass_kg.nrows();
        if stiffness_n_per_m.shape() != mass_kg.shape()
            || damping_n_s_per_m.shape() != mass_kg.shape()
            || branches
                .iter()
                .any(|branch| branch.stiffness.normalized.nrows() != dimension)
        {
            return Err(ResponseError::Shape);
        }
        Ok(Self {
            mass: QuadraticForm::new(mass_kg, true, tolerance)?,
            stiffness: QuadraticForm::new(stiffness_n_per_m, true, tolerance)?,
            damping: QuadraticForm::new(damping_n_s_per_m, false, tolerance)?,
            branches,
        })
    }

    /// Corrections in mass, support-stiffness and damping order.
    pub fn corrections(&self) -> [MatrixCorrection; 3] {
        [
            self.mass.correction,
            self.stiffness.correction,
            self.damping.correction,
        ]
    }

    /// Corrections for each auxiliary memory stiffness, in branch order.
    pub fn memory_corrections(&self) -> Vec<MatrixCorrection> {
        self.branches.iter().map(MemoryBranch::correction).collect()
    }

    pub fn power(
        &self,
        position_m: &DVector<f64>,
        velocity_m_per_s: &DVector<f64>,
        memory_displacements_m: &[DVector<f64>],
        drive_n: &DVector<f64>,
    ) -> Result<MechanicalPower, ResponseError> {
        let dimension = self.mass.normalized.nrows();
        if [position_m, velocity_m_per_s, drive_n]
            .iter()
            .any(|vector| vector.len() != dimension)
            || memory_displacements_m.len() != self.branches.len()
            || memory_displacements_m
                .iter()
                .any(|vector| vector.len() != dimension)
        {
            return Err(ResponseError::Shape);
        }
        for vector in [position_m, velocity_m_per_s, drive_n]
            .into_iter()
            .chain(memory_displacements_m)
        {
            finite_vector(vector)?;
        }
        let support_force = self.stiffness.apply(position_m)?;
        let mut force = drive_n - &support_force - self.damping.apply(velocity_m_per_s)?;
        let mut stored_energy = finite(
            half_storage(&self.mass, velocity_m_per_s)?
                + half_storage(&self.stiffness, position_m)?,
        )?;
        let mut dissipation = self.damping.quadratic(velocity_m_per_s)?;
        let mut memory_rate = 0.0;
        let mut rates = Vec::with_capacity(self.branches.len());
        for (branch, displacement) in self.branches.iter().zip(memory_displacements_m) {
            let restoring = branch.stiffness.apply(displacement)?;
            let quadratic = branch.stiffness.quadratic(displacement)?;
            stored_energy = finite(stored_energy + preserve_nonzero(0.5 * quadratic, quadratic)?)?;
            dissipation = finite(
                dissipation + preserve_nonzero(quadratic / branch.relaxation_s, quadratic)?,
            )?;
            let rate =
                velocity_m_per_s - checked_vector_division(displacement, branch.relaxation_s)?;
            finite_vector(&rate)?;
            memory_rate = finite(memory_rate + checked_dot(&restoring, &rate)?)?;
            force -= restoring;
            rates.push(rate);
        }
        finite_vector(&force)?;
        let cholesky = self
            .mass
            .normalized
            .clone()
            .cholesky()
            .ok_or(ResponseError::Arithmetic)?;
        let acceleration = cholesky.solve(&checked_vector_division(&force, self.mass.scale)?);
        finite_vector(&acceleration)?;
        let kinetic_rate = checked_dot(velocity_m_per_s, &self.mass.apply(&acceleration)?)?;
        let support_rate = checked_dot(&support_force, velocity_m_per_s)?;
        let energy_rate = finite(finite(kinetic_rate + support_rate)? + memory_rate)?;
        let drive_power = checked_dot(drive_n, velocity_m_per_s)?;
        let balance = crate::vacuum_audit::energy_closure_residual(
            drive_power.max(0.0),
            0.0,
            (-drive_power).max(0.0),
            dissipation,
            energy_rate,
        )
        .map_err(|_| ResponseError::Arithmetic)?;
        Ok(MechanicalPower {
            stored_energy_j: stored_energy,
            drive_power_w: drive_power,
            dissipation_w: dissipation,
            energy_rate_w: energy_rate,
            balance_residual_w: balance,
            acceleration_m_per_s2: acceleration,
            memory_rates_m_per_s: rates,
        })
    }
}

/// Scalar local harmonic mode and its ordered thermal displacement line weights.
#[derive(Debug, Clone, Copy)]
pub struct HarmonicMode {
    pub omega_rad_s: f64,
    pub line_weights: OscillatorLineWeights,
}

pub fn harmonic_mode_from_stiffness(
    mass_kg: f64,
    effective_stiffness_n_per_m: f64,
    temperature_k: f64,
) -> Result<HarmonicMode, ResponseError> {
    positive(mass_kg)?;
    positive(effective_stiffness_n_per_m)?;
    let omega = finite(effective_stiffness_n_per_m.sqrt() / mass_kg.sqrt())?;
    let line_weights = oscillator_line_weights(mass_kg, omega, temperature_k)
        .map_err(|_| ResponseError::Domain)?;
    Ok(HarmonicMode {
        omega_rad_s: omega,
        line_weights,
    })
}

/// Gaussian phase variance and relaxation-inclusive coherence, dimensionless.
#[derive(Debug, Clone, Copy)]
pub struct GaussianCoherence {
    pub phase_variance: f64,
    pub coherence: f64,
    pub covariance_correction: MatrixCorrection,
}

/// Sampled zero-mean classical Gaussian dephasing with explicit cross-covariance.
///
/// Covariance rows use time-major order `(time, coordinate)` and units m^2.
/// Signed weights combine time quadrature (seconds) and pulse switching (+/-1).
/// Coupling supplies angular-frequency/displacement derivatives (rad/s/m).
/// Empirical prediction requires independent calibration; investigator-selected
/// values instead define a conditional model. Phase variance is w^T C w with
/// w[time,coordinate] = signed_weight[time]*coupling[coordinate].
/// Duration is seconds and bounds sum(abs(weights)); T1, when supplied, is a
/// positive relaxation time in seconds; empirical use requires independent
/// measurement. Coherence is
/// exp(-duration/(2*T1)-phase_variance/2). Omitted T1 declares relaxation absent.
/// Covariance PSD is necessary for the sampled Gaussian model; sampling and
/// quadrature fidelity need separate evidence. Ordered quantum noise, pulse
/// imperfections and non-Gaussian corrections require different contracts.
pub fn sampled_gaussian_coherence(
    covariance_m2: &DMatrix<f64>,
    signed_weights_s: &[f64],
    coupling_rad_per_s_per_m: &DVector<f64>,
    duration_s: f64,
    t1_s: Option<f64>,
    tolerance: MatrixTolerance,
) -> Result<GaussianCoherence, ResponseError> {
    if signed_weights_s.is_empty() || coupling_rad_per_s_per_m.is_empty() {
        return Err(ResponseError::Shape);
    }
    let dimension = signed_weights_s
        .len()
        .checked_mul(coupling_rad_per_s_per_m.len())
        .ok_or(ResponseError::Shape)?;
    if covariance_m2.shape() != (dimension, dimension) {
        return Err(ResponseError::Shape);
    }
    if !duration_s.is_finite()
        || duration_s < 0.0
        || signed_weights_s.iter().any(|weight| !weight.is_finite())
    {
        return Err(ResponseError::Domain);
    }
    finite_vector(coupling_rad_per_s_per_m)?;
    let weight_total = finite(signed_weights_s.iter().map(|weight| weight.abs()).sum())?;
    if weight_total > duration_s && weight_total - duration_s > tolerance.0 * duration_s {
        return Err(ResponseError::Domain);
    }
    let relaxation_exponent = if let Some(t1) = t1_s {
        positive(t1)?;
        finite((duration_s / t1) * 0.5)?
    } else {
        0.0
    };
    let covariance = QuadraticForm::new(covariance_m2.clone(), false, tolerance)?;
    let mut weights = DVector::zeros(dimension);
    for (time, weight) in signed_weights_s.iter().enumerate() {
        for (coordinate, coupling) in coupling_rad_per_s_per_m.iter().enumerate() {
            weights[time * coupling_rad_per_s_per_m.len() + coordinate] =
                checked_product(*weight, *coupling)?;
        }
    }
    let variance = covariance.quadratic(&weights)?;
    let exponent = finite(relaxation_exponent + variance * 0.5)?;
    Ok(GaussianCoherence {
        phase_variance: variance,
        coherence: (-exponent).exp(),
        covariance_correction: covariance.correction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn matrix(values: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, values)
    }

    #[test]
    fn coupled_storage_and_multiple_memory_branches_close_power() {
        let tolerance = MatrixTolerance::new(1e-12).unwrap();
        let branch = MemoryBranch::new(matrix(&[2.0, 1.0, 1.0, 2.0]), 2.0, tolerance).unwrap();
        let null_branch = MemoryBranch::new(DMatrix::zeros(2, 2), 3.0, tolerance).unwrap();
        let model = QuadraticMechanics::new(
            matrix(&[2.0, 1.0, 1.0, 2.0]),
            matrix(&[4.0, 1.0, 1.0, 3.0]),
            matrix(&[1.0, 0.5, 0.5, 1.0]),
            vec![branch, null_branch],
            tolerance,
        )
        .unwrap();
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let v = DVector::from_vec(vec![2.0, -1.0]);
        let q = vec![
            DVector::from_vec(vec![1.0, -1.0]),
            DVector::from_vec(vec![4.0, 7.0]),
        ];
        let result = model
            .power(&x, &v, &q, &DVector::from_vec(vec![3.0, 2.0]))
            .unwrap();
        assert!((result.stored_energy_j - 14.0).abs() < 1e-11);
        assert!((result.dissipation_w - 4.0).abs() < 1e-11);
        assert!((result.drive_power_w - 4.0).abs() < 1e-11);
        assert!(result.balance_residual_w.abs() < 1e-11);
        assert!((result.energy_rate_w).abs() < 1e-11);
        let resting = model
            .power(
                &DVector::zeros(2),
                &DVector::zeros(2),
                &q,
                &DVector::zeros(2),
            )
            .unwrap();
        assert!((resting.energy_rate_w + 1.0).abs() < 1e-11);
        assert_eq!(resting.drive_power_w, 0.0);
    }

    #[test]
    fn matrix_domains_reject_indefinite_asymmetric_and_dimension_mismatch() {
        let tolerance = MatrixTolerance::new(1e-12).unwrap();
        assert!(MemoryBranch::new(matrix(&[1.0, 2.0, 2.0, 1.0]), 1.0, tolerance).is_err());
        assert!(MemoryBranch::new(matrix(&[1.0, 0.2, 0.0, 1.0]), 1.0, tolerance).is_err());
        assert!(MemoryBranch::new(DMatrix::identity(2, 2), 0.0, tolerance).is_err());
        assert!(
            QuadraticMechanics::new(
                DMatrix::identity(2, 2),
                DMatrix::zeros(2, 2),
                DMatrix::zeros(2, 2),
                vec![],
                tolerance
            )
            .is_err()
        );
        assert!(MatrixTolerance::new(f64::NAN).is_err());
        let rounded = MemoryBranch::new(matrix(&[1.0, 0.0, 0.0, -1e-13]), 1.0, tolerance).unwrap();
        assert!(rounded.correction().psd_projection_relative > 0.0);
        let asymmetric =
            MemoryBranch::new(matrix(&[1.0, 5e-13, 0.0, 1.0]), 1.0, tolerance).unwrap();
        assert!(asymmetric.correction().symmetry_relative > 0.0);
    }

    #[test]
    fn quasistatic_covariance_distinguishes_ramsey_echo_and_zero_coupling() {
        let covariance = DMatrix::from_element(2, 2, 4.0);
        let coupling = DVector::from_element(1, 3.0);
        let tolerance = MatrixTolerance::new(1e-12).unwrap();
        let ramsey =
            sampled_gaussian_coherence(&covariance, &[0.5, 0.5], &coupling, 1.0, None, tolerance)
                .unwrap();
        let echo =
            sampled_gaussian_coherence(&covariance, &[0.5, -0.5], &coupling, 1.0, None, tolerance)
                .unwrap();
        assert!((ramsey.phase_variance - 36.0).abs() < 1e-10);
        assert!((ramsey.coherence - (-18.0_f64).exp()).abs() < 1e-16);
        assert!(echo.phase_variance < 1e-12);
        assert!((echo.coherence - 1.0).abs() < 1e-12);
        let zero = sampled_gaussian_coherence(
            &covariance,
            &[0.5, 0.5],
            &DVector::zeros(1),
            1.0,
            Some(2.0),
            tolerance,
        )
        .unwrap();
        assert_eq!(zero.phase_variance, 0.0);
        assert_eq!(zero.coherence, (-0.25_f64).exp());
    }

    #[test]
    fn covariance_cross_coordinate_terms_and_failures_are_visible() {
        let tolerance = MatrixTolerance::new(1e-12).unwrap();
        let covariance = matrix(&[4.0, -2.0, -2.0, 4.0]);
        let result = sampled_gaussian_coherence(
            &covariance,
            &[1.0],
            &DVector::from_vec(vec![1.0, 1.0]),
            1.0,
            None,
            tolerance,
        )
        .unwrap();
        assert!((result.phase_variance - 4.0).abs() < 1e-11);
        assert!(
            sampled_gaussian_coherence(
                &matrix(&[1.0, 2.0, 2.0, 1.0]),
                &[1.0],
                &DVector::from_vec(vec![1.0, 1.0]),
                1.0,
                None,
                tolerance
            )
            .is_err()
        );
        assert!(
            sampled_gaussian_coherence(
                &covariance,
                &[1.0, 1.0],
                &DVector::from_element(2, 1.0),
                1.0,
                None,
                tolerance
            )
            .is_err()
        );
    }

    #[test]
    fn stiffness_constructor_reuses_ordered_oscillator_and_rejects_instability() {
        let mode = harmonic_mode_from_stiffness(2.0, 8.0, 0.0).unwrap();
        assert_eq!(mode.omega_rad_s, 2.0);
        assert_eq!(mode.line_weights.negative_frequency_m2, 0.0);
        assert!(harmonic_mode_from_stiffness(2.0, -1.0, 0.0).is_err());
    }

    #[test]
    fn tiny_nonzero_variance_rejects_underflow_and_large_coordinates_rescale() {
        let tolerance = MatrixTolerance::new(1e-12).unwrap();
        let covariance = DMatrix::from_element(1, 1, 1e-300);
        assert!(
            sampled_gaussian_coherence(
                &covariance,
                &[1.0],
                &DVector::from_element(1, 1e-100),
                1.0,
                None,
                tolerance
            )
            .is_err()
        );
        let result = sampled_gaussian_coherence(
            &covariance,
            &[1.0],
            &DVector::from_element(1, 1e200),
            1.0,
            None,
            tolerance,
        )
        .unwrap();
        assert!((result.phase_variance / 1e100 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn negative_drive_power_extracts_mechanical_energy() {
        let unit = DMatrix::identity(1, 1);
        let zero = DVector::zeros(1);
        let model = QuadraticMechanics::new(
            unit.clone(),
            unit,
            DMatrix::zeros(1, 1),
            vec![],
            MatrixTolerance::new(1e-12).unwrap(),
        )
        .unwrap();
        let result = model
            .power(
                &zero,
                &DVector::from_element(1, 1.0),
                &[],
                &DVector::from_element(1, -1.0),
            )
            .unwrap();
        assert_eq!(result.drive_power_w, -1.0);
        assert_eq!(result.energy_rate_w, -1.0);
        assert_eq!(result.balance_residual_w, 0.0);
    }

    #[test]
    fn nonzero_drive_power_rejects_underflow() {
        let unit = DMatrix::identity(1, 1);
        let model = QuadraticMechanics::new(
            unit.clone(),
            unit,
            DMatrix::zeros(1, 1),
            vec![],
            MatrixTolerance::new(1e-12).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            model.power(
                &DVector::zeros(1),
                &DVector::from_element(1, 1e-160),
                &[],
                &DVector::from_element(1, 1e-200)
            ),
            Err(ResponseError::Arithmetic)
        ));
    }

    #[test]
    fn nonzero_memory_storage_and_dissipation_reject_underflow() {
        let tolerance = MatrixTolerance::new(1e-12).unwrap();
        for (stiffness, relaxation) in [(f64::from_bits(1), 1.0), (1e-300, 1e100)] {
            let branch = MemoryBranch::new(
                DMatrix::from_element(1, 1, stiffness),
                relaxation,
                tolerance,
            )
            .unwrap();
            let model = QuadraticMechanics::new(
                DMatrix::identity(1, 1),
                DMatrix::identity(1, 1),
                DMatrix::zeros(1, 1),
                vec![branch],
                tolerance,
            )
            .unwrap();
            let zero = DVector::zeros(1);
            assert!(matches!(
                model.power(&zero, &zero, &[DVector::from_element(1, 1.0)], &zero),
                Err(ResponseError::Arithmetic)
            ));
        }
    }

    #[test]
    fn matrix_normalization_preserves_nonzero_coefficients_or_rejects() {
        let tolerance = MatrixTolerance::new(0.0).unwrap();
        let extreme = matrix(&[1e300, 0.0, 0.0, 1e-300]);
        assert!(matches!(
            sampled_gaussian_coherence(
                &extreme,
                &[1.0],
                &DVector::from_vec(vec![0.0, 1e150]),
                1.0,
                None,
                tolerance
            ),
            Err(ResponseError::Arithmetic)
        ));
        assert!(matches!(
            MemoryBranch::new(extreme, 1.0, tolerance),
            Err(ResponseError::Arithmetic)
        ));
    }

    #[test]
    fn coordinate_normalization_rejects_nonzero_component_loss() {
        let covariance = matrix(&[0.0, 0.0, 0.0, 1e300]);
        assert!(matches!(
            sampled_gaussian_coherence(
                &covariance,
                &[1.0],
                &DVector::from_vec(vec![1e300, 1e-300]),
                1.0,
                None,
                MatrixTolerance::new(0.0).unwrap()
            ),
            Err(ResponseError::Arithmetic)
        ));
    }

    #[test]
    fn mass_solve_scaling_rejects_nonzero_force_loss() {
        let model = QuadraticMechanics::new(
            matrix(&[1e300, 0.0, 0.0, 1.0]),
            DMatrix::identity(2, 2),
            DMatrix::zeros(2, 2),
            vec![],
            MatrixTolerance::new(0.0).unwrap(),
        )
        .unwrap();
        let zero = DVector::zeros(2);
        assert!(matches!(
            model.power(&zero, &zero, &[], &DVector::from_vec(vec![0.0, 1e-100])),
            Err(ResponseError::Arithmetic)
        ));
    }

    #[test]
    fn quadratic_matrix_action_rejects_nonzero_intermediate_loss() {
        let covariance = DMatrix::from_diagonal(&DVector::from_vec(vec![1e300, 1e100, 0.0]));
        let coupling = DVector::from_vec(vec![0.0, 1e-50, 1e150]);
        assert!(matches!(
            sampled_gaussian_coherence(
                &covariance,
                &[1.0],
                &coupling,
                1.0,
                None,
                MatrixTolerance::new(0.0).unwrap()
            ),
            Err(ResponseError::Arithmetic)
        ));
    }

    #[test]
    fn arithmetic_overflow_is_a_failure() {
        let tolerance = MatrixTolerance::new(1e-12).unwrap();
        assert!(
            sampled_gaussian_coherence(
                &DMatrix::from_element(1, 1, f64::MAX),
                &[1.0],
                &DVector::from_element(1, f64::MAX),
                1.0,
                None,
                tolerance
            )
            .is_err()
        );
    }
}
