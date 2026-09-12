//! Calibrated mechanism discrimination in whitened measurement coordinates.
//!
//! The functions in this module separate unrestricted linear span tests from
//! nuisance explanations constrained by independent calibration or physical
//! bounds. Callers own the whitening transform and the physical metadata that
//! maps each nuisance coordinate to a specimen or instrument parameter.

use clarabel::{algebra::CscMatrix, solver::*};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

const SOLVER_CERTIFICATE_TOLERANCE: f64 = 1e-7;

/// Input or numerical failure in a discrimination calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscriminationError {
    /// Matrix and vector dimensions disagree.
    DimensionMismatch,
    /// An input contains a non-finite value.
    NonFiniteInput,
    /// A lower bound exceeds its corresponding upper bound.
    InvalidBounds,
    /// The least-squares or quadratic solve did not produce a finite result.
    NumericalFailure,
}

/// A physical nuisance interval with caller-declared units.
#[derive(Debug, Clone, PartialEq)]
pub struct NuisanceBound {
    pub lower: f64,
    pub upper: f64,
    pub unit: String,
}

/// Result of a bounded whitened profile calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedProfileResult {
    pub distance: f64,
    pub nuisance_parameters: DVector<f64>,
    pub active_lower_bounds: Vec<usize>,
    pub active_upper_bounds: Vec<usize>,
}

#[derive(Clone, Copy)]
struct SolverCertificate {
    status: SolverStatus,
    primal_residual: f64,
    dual_residual: f64,
    absolute_gap: f64,
    relative_gap: f64,
}

fn validate_bounded_solution(
    certificate: SolverCertificate,
    target: &DVector<f64>,
    nuisance: &DMatrix<f64>,
    bounds: &[NuisanceBound],
    parameters: &DVector<f64>,
) -> Result<(), DiscriminationError> {
    if certificate.status != SolverStatus::Solved
        || !certificate.primal_residual.is_finite()
        || !certificate.dual_residual.is_finite()
        || !certificate.absolute_gap.is_finite()
        || !certificate.relative_gap.is_finite()
        || certificate.primal_residual > SOLVER_CERTIFICATE_TOLERANCE
        || certificate.dual_residual > SOLVER_CERTIFICATE_TOLERANCE
        || (certificate.absolute_gap > SOLVER_CERTIFICATE_TOLERANCE
            && certificate.relative_gap > SOLVER_CERTIFICATE_TOLERANCE)
    {
        return Err(DiscriminationError::NumericalFailure);
    }

    for (parameter, bound) in parameters.iter().zip(bounds) {
        let parameter_scale = parameter
            .abs()
            .max(bound.lower.abs())
            .max(bound.upper.abs())
            .max(1.0);
        let feasibility_tolerance = SOLVER_CERTIFICATE_TOLERANCE * parameter_scale;
        if *parameter < bound.lower - feasibility_tolerance
            || *parameter > bound.upper + feasibility_tolerance
        {
            return Err(DiscriminationError::NumericalFailure);
        }
    }

    let gradient = 2.0 * nuisance.transpose() * (nuisance * parameters - target);
    let gradient_scale = (2.0 * nuisance.transpose() * (nuisance * parameters))
        .amax()
        .max((2.0 * nuisance.transpose() * target).amax())
        .max(1.0);
    let optimality_tolerance = SOLVER_CERTIFICATE_TOLERANCE * gradient_scale;
    for ((parameter, gradient_component), bound) in
        parameters.iter().zip(gradient.iter()).zip(bounds)
    {
        let parameter_scale = parameter
            .abs()
            .max(bound.lower.abs())
            .max(bound.upper.abs())
            .max(1.0);
        let projected = (*parameter - *gradient_component).clamp(bound.lower, bound.upper);
        let projected_gradient_tolerance = SOLVER_CERTIFICATE_TOLERANCE
            .sqrt()
            .max(optimality_tolerance)
            * parameter_scale;
        if (*parameter - projected).abs() > projected_gradient_tolerance {
            return Err(DiscriminationError::NumericalFailure);
        }
    }
    Ok(())
}

fn all_finite_matrix(matrix: &DMatrix<f64>) -> bool {
    matrix.iter().all(|value| value.is_finite())
}

fn all_finite_vector(vector: &DVector<f64>) -> bool {
    vector.iter().all(|value| value.is_finite())
}

fn classify_active_bounds(
    parameters: &DVector<f64>,
    bounds: &[NuisanceBound],
) -> (Vec<usize>, Vec<usize>) {
    let mut active_lower_bounds = Vec::new();
    let mut active_upper_bounds = Vec::new();
    for (index, (parameter, bound)) in parameters.iter().zip(bounds).enumerate() {
        let bound_scale = (bound.upper - bound.lower)
            .abs()
            .max(bound.lower.abs())
            .max(bound.upper.abs())
            .max(f64::MIN_POSITIVE);
        let active_tolerance = SOLVER_CERTIFICATE_TOLERANCE * bound_scale;
        if (*parameter - bound.lower).abs() <= active_tolerance {
            active_lower_bounds.push(index);
        }
        if (*parameter - bound.upper).abs() <= active_tolerance {
            active_upper_bounds.push(index);
        }
    }
    (active_lower_bounds, active_upper_bounds)
}

fn validate_linear_model(
    target: &DVector<f64>,
    nuisance: &DMatrix<f64>,
) -> Result<(), DiscriminationError> {
    if nuisance.nrows() != target.len() {
        return Err(DiscriminationError::DimensionMismatch);
    }
    if !all_finite_vector(target) || !all_finite_matrix(nuisance) {
        return Err(DiscriminationError::NonFiniteInput);
    }
    Ok(())
}

fn least_squares(
    design: &DMatrix<f64>,
    response: &DVector<f64>,
) -> Result<DVector<f64>, DiscriminationError> {
    let scales: Vec<f64> = design
        .column_iter()
        .map(|column| {
            let norm = column.norm();
            if norm > 0.0 { norm } else { 1.0 }
        })
        .collect();
    let mut normalized = design.clone();
    for (mut column, scale) in normalized.column_iter_mut().zip(&scales) {
        column.scale_mut(1.0 / scale);
    }
    let scaled_solution = normalized
        .svd(true, true)
        .solve(response, f64::EPSILON.sqrt())
        .map_err(|_| DiscriminationError::NumericalFailure)?;
    Ok(DVector::from_iterator(
        scaled_solution.len(),
        scaled_solution
            .iter()
            .zip(scales)
            .map(|(value, scale)| value / scale),
    ))
}

/// Efficient information after independent calibration constrains nuisances.
///
/// Target and nuisance are already whitened by the target-measurement
/// covariance. Calibration is the independently whitened response K. The
/// function solves one augmented least-squares problem instead of subtracting
/// nearly equal quadratic forms.
pub fn efficient_information(
    target: &DVector<f64>,
    nuisance: &DMatrix<f64>,
    calibration: &DMatrix<f64>,
) -> Result<f64, DiscriminationError> {
    validate_linear_model(target, nuisance)?;
    if calibration.ncols() != nuisance.ncols() {
        return Err(DiscriminationError::DimensionMismatch);
    }
    if !all_finite_matrix(calibration) {
        return Err(DiscriminationError::NonFiniteInput);
    }

    let mut augmented_design =
        DMatrix::zeros(nuisance.nrows() + calibration.nrows(), nuisance.ncols());
    augmented_design
        .view_mut((0, 0), nuisance.shape())
        .copy_from(nuisance);
    augmented_design
        .view_mut(
            (nuisance.nrows(), 0),
            (calibration.nrows(), calibration.ncols()),
        )
        .copy_from(calibration);
    let mut augmented_target = DVector::zeros(target.len() + calibration.nrows());
    augmented_target.rows_mut(0, target.len()).copy_from(target);
    let fitted = least_squares(&augmented_design, &augmented_target)?;
    let residual = augmented_target - augmented_design * fitted;
    Ok(residual.norm_squared())
}

/// Local derivative of efficient information with respect to added diagonal
/// calibration precision on nuisance coordinate.
pub fn calibration_precision_sensitivity(
    target: &DVector<f64>,
    nuisance: &DMatrix<f64>,
    calibration: &DMatrix<f64>,
    coordinate: usize,
) -> Result<f64, DiscriminationError> {
    validate_linear_model(target, nuisance)?;
    if calibration.ncols() != nuisance.ncols() || coordinate >= nuisance.ncols() {
        return Err(DiscriminationError::DimensionMismatch);
    }
    if !all_finite_matrix(calibration) {
        return Err(DiscriminationError::NonFiniteInput);
    }
    let normal = nuisance.transpose() * nuisance + calibration.transpose() * calibration;
    let cross = nuisance.transpose() * target;
    let solution = least_squares(&normal, &cross)?;
    Ok(solution[coordinate].powi(2))
}

/// Unrestricted nuisance projection residual using an SVD pseudoinverse.
pub fn unrestricted_residual(
    target: &DVector<f64>,
    nuisance: &DMatrix<f64>,
) -> Result<DVector<f64>, DiscriminationError> {
    validate_linear_model(target, nuisance)?;
    let parameters = least_squares(nuisance, target)?;
    Ok(target - nuisance * parameters)
}

/// Minimize the whitened residual norm subject to caller-declared box bounds.
///
/// Clarabel solves the convex quadratic program globally. The returned active
/// sets use a tolerance scaled to each interval.
pub fn bounded_profile_distance(
    target: &DVector<f64>,
    nuisance: &DMatrix<f64>,
    bounds: &[NuisanceBound],
) -> Result<BoundedProfileResult, DiscriminationError> {
    validate_linear_model(target, nuisance)?;
    if bounds.len() != nuisance.ncols() {
        return Err(DiscriminationError::DimensionMismatch);
    }
    if bounds.iter().any(|bound| {
        !bound.lower.is_finite()
            || !bound.upper.is_finite()
            || bound.lower > bound.upper
            || bound.unit.trim().is_empty()
    }) {
        return Err(DiscriminationError::InvalidBounds);
    }

    let parameter_count = nuisance.ncols();
    let objective_scale = target.amax().max(nuisance.amax()).max(f64::MIN_POSITIVE);
    let scaled_target = target / objective_scale;
    let scaled_nuisance = nuisance / objective_scale;
    let column_scales: Vec<f64> = scaled_nuisance
        .column_iter()
        .map(|column| column.norm().max(f64::MIN_POSITIVE))
        .collect();
    let mut solver_nuisance = scaled_nuisance.clone();
    for (mut column, scale) in solver_nuisance.column_iter_mut().zip(&column_scales) {
        column.scale_mut(1.0 / scale);
    }
    let mut quadratic_data = Vec::new();
    let mut quadratic_indices = Vec::new();
    let mut quadratic_indptr = vec![0];
    for column in 0..parameter_count {
        for row in 0..=column {
            quadratic_data.push(
                2.0 * solver_nuisance
                    .column(row)
                    .dot(&solver_nuisance.column(column)),
            );
            quadratic_indices.push(row);
        }
        quadratic_indptr.push(quadratic_data.len());
    }
    let quadratic = CscMatrix::new(
        parameter_count,
        parameter_count,
        quadratic_indptr,
        quadratic_indices,
        quadratic_data,
    );
    let linear: Vec<f64> = (-2.0 * solver_nuisance.transpose() * &scaled_target)
        .iter()
        .copied()
        .collect();

    let mut constraint_data = Vec::with_capacity(2 * parameter_count);
    let mut constraint_indices = Vec::with_capacity(2 * parameter_count);
    let mut constraint_indptr = vec![0];
    for column in 0..parameter_count {
        constraint_data.extend([1.0, -1.0]);
        constraint_indices.extend([column, parameter_count + column]);
        constraint_indptr.push(constraint_data.len());
    }
    let constraints = CscMatrix::new(
        2 * parameter_count,
        parameter_count,
        constraint_indptr,
        constraint_indices,
        constraint_data,
    );
    let mut right_hand_side: Vec<f64> = bounds
        .iter()
        .zip(&column_scales)
        .map(|(bound, scale)| bound.upper * scale)
        .collect();
    right_hand_side.extend(
        bounds
            .iter()
            .zip(&column_scales)
            .map(|(bound, scale)| -bound.lower * scale),
    );
    let cones = [NonnegativeConeT::<f64>(2 * parameter_count)];
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .max_iter(500)
        .tol_gap_abs(1.0e-10)
        .tol_gap_rel(1.0e-10)
        .tol_feas(1.0e-10)
        .build()
        .map_err(|_| DiscriminationError::NumericalFailure)?;
    let mut solver = DefaultSolver::new(
        &quadratic,
        &linear,
        &constraints,
        &right_hand_side,
        &cones,
        settings,
    )
    .map_err(|_| DiscriminationError::NumericalFailure)?;
    solver.solve();
    let scaled_parameters = DVector::from_column_slice(&solver.solution.x);
    let parameters = DVector::from_iterator(
        scaled_parameters.len(),
        scaled_parameters
            .iter()
            .zip(&column_scales)
            .map(|(parameter, scale)| parameter / scale),
    );
    if !all_finite_vector(&parameters) {
        return Err(DiscriminationError::NumericalFailure);
    }
    validate_bounded_solution(
        SolverCertificate {
            status: solver.solution.status,
            primal_residual: solver.solution.r_prim,
            dual_residual: solver.solution.r_dual,
            absolute_gap: solver.info.gap_abs,
            relative_gap: solver.info.gap_rel,
        },
        &scaled_target,
        &scaled_nuisance,
        bounds,
        &parameters,
    )?;
    let distance = (target - nuisance * &parameters).norm();
    let (active_lower_bounds, active_upper_bounds) = classify_active_bounds(&parameters, bounds);
    Ok(BoundedProfileResult {
        distance,
        nuisance_parameters: parameters,
        active_lower_bounds,
        active_upper_bounds,
    })
}

/// Fisher information with a symmetric positive-semidefinite covariance.
///
/// Eigenvalues at or below the relative threshold are deterministic null
/// directions. A signal component in such a direction makes the information
/// infinite because the supplied covariance declares zero noise there.
pub fn fisher_information_with_pseudoinverse(
    signal: &DVector<f64>,
    covariance: &DMatrix<f64>,
    relative_tolerance: f64,
) -> Result<f64, DiscriminationError> {
    if covariance.nrows() != covariance.ncols() || covariance.nrows() != signal.len() {
        return Err(DiscriminationError::DimensionMismatch);
    }
    if !all_finite_vector(signal)
        || !all_finite_matrix(covariance)
        || !relative_tolerance.is_finite()
        || relative_tolerance < 0.0
    {
        return Err(DiscriminationError::NonFiniteInput);
    }
    let symmetry_error = covariance - covariance.transpose();
    if symmetry_error.norm() > 1e-10 * covariance.norm().max(1.0) {
        return Err(DiscriminationError::NumericalFailure);
    }
    let symmetric_covariance = (covariance + covariance.transpose()) * 0.5;
    let decomposition = SymmetricEigen::new(symmetric_covariance);
    let spectral_scale = decomposition
        .eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let effective_relative_tolerance = relative_tolerance.max(64.0 * f64::EPSILON);
    let eigenvalue_tolerance = effective_relative_tolerance * spectral_scale;
    if decomposition
        .eigenvalues
        .iter()
        .any(|eigenvalue| *eigenvalue < -eigenvalue_tolerance)
    {
        return Err(DiscriminationError::NumericalFailure);
    }
    let maximum = decomposition
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let threshold = effective_relative_tolerance * maximum;
    let coordinates = decomposition.eigenvectors.transpose() * signal;
    let null_coordinate_tolerance = effective_relative_tolerance.sqrt() * signal.norm();
    let mut information = 0.0;
    for (eigenvalue, coordinate) in decomposition.eigenvalues.iter().zip(coordinates.iter()) {
        if *eigenvalue > threshold {
            information += coordinate * coordinate / eigenvalue;
        } else if coordinate.abs() > null_coordinate_tolerance {
            return Ok(f64::INFINITY);
        }
    }
    Ok(information)
}

/// Lower bound on true whitened separation after numerical and model errors.
pub fn certified_distance_lower_bound(
    computed_distance: f64,
    hypothesis_0_error: f64,
    hypothesis_1_error: f64,
) -> Result<f64, DiscriminationError> {
    if !computed_distance.is_finite()
        || !hypothesis_0_error.is_finite()
        || !hypothesis_1_error.is_finite()
        || computed_distance < 0.0
        || hypothesis_0_error < 0.0
        || hypothesis_1_error < 0.0
    {
        return Err(DiscriminationError::NonFiniteInput);
    }
    Ok((computed_distance - hypothesis_0_error - hypothesis_1_error).max(0.0))
}

/// Whether two continuous effect parameters satisfy a predeclared minimum
/// physically meaningful separation.
pub fn effect_separation_is_admissible(
    theta_0: f64,
    theta_1: f64,
    minimum_effect: f64,
) -> Result<bool, DiscriminationError> {
    if !theta_0.is_finite()
        || !theta_1.is_finite()
        || !minimum_effect.is_finite()
        || minimum_effect <= 0.0
    {
        return Err(DiscriminationError::NonFiniteInput);
    }
    Ok((theta_1 - theta_0).abs() >= minimum_effect)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn exact_unrestricted_mimic_collapses_residual() {
        let nuisance = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let target = DVector::from_vec(vec![2.0, -1.0, 1.0]);
        assert!(unrestricted_residual(&target, &nuisance).unwrap().norm() < 1e-12);
    }

    #[test]
    fn nuisance_rescaling_preserves_unrestricted_residual() {
        let target = DVector::from_vec(vec![1.0, 2.0, -1.0, 0.5]);
        let nuisance = DMatrix::from_row_slice(4, 2, &[1.0, 0.2, 0.0, 1.0, 1.0, -1.0, 0.4, 0.8]);
        let mut rescaled = nuisance.clone();
        rescaled.column_mut(0).scale_mut(1e-8);
        rescaled.column_mut(1).scale_mut(1e8);
        let baseline = unrestricted_residual(&target, &nuisance).unwrap();
        let transformed = unrestricted_residual(&target, &rescaled).unwrap();
        assert_relative_eq!(baseline, transformed, epsilon = 1e-8);
    }

    #[test]
    fn independent_calibration_increases_information() {
        let target = DVector::from_vec(vec![1.0, 1.0]);
        let nuisance = DMatrix::from_column_slice(2, 1, &[1.0, 1.0]);
        let no_calibration = DMatrix::zeros(0, 1);
        let calibrated = DMatrix::from_element(1, 1, 2.0);
        let uncalibrated_information =
            efficient_information(&target, &nuisance, &no_calibration).unwrap();
        let calibrated_information =
            efficient_information(&target, &nuisance, &calibrated).unwrap();
        assert!(uncalibrated_information < 1e-12);
        assert!(calibrated_information > uncalibrated_information);
        let derivative =
            calibration_precision_sensitivity(&target, &nuisance, &calibrated, 0).unwrap();
        assert!(derivative >= 0.0);
    }

    #[test]
    fn bounded_profile_distinguishes_physical_from_unbounded_imitation() {
        let target = DVector::from_vec(vec![2.0, 0.0]);
        let nuisance = DMatrix::identity(2, 2);
        let bounds = vec![
            NuisanceBound {
                lower: -1.0,
                upper: 1.0,
                unit: "Pa".to_owned(),
            },
            NuisanceBound {
                lower: -1.0,
                upper: 1.0,
                unit: "Pa".to_owned(),
            },
        ];
        let bounded = bounded_profile_distance(&target, &nuisance, &bounds).unwrap();
        assert_relative_eq!(bounded.distance, 1.0, epsilon = 1e-7);
        assert_relative_eq!(bounded.nuisance_parameters[0], 1.0, epsilon = 1e-7);
        assert_eq!(bounded.active_upper_bounds, vec![0]);
        assert!(unrestricted_residual(&target, &nuisance).unwrap().norm() < 1e-12);
    }

    #[test]
    fn active_bound_classification_respects_small_parameter_units() {
        let parameters = DVector::from_vec(vec![1e-10, 2e-10, 1.5e-10]);
        let bounds = vec![
            NuisanceBound {
                lower: 1e-10,
                upper: 2e-10,
                unit: "m".to_owned(),
            };
            3
        ];
        let (active_lower_bounds, active_upper_bounds) =
            classify_active_bounds(&parameters, &bounds);

        assert_eq!(active_lower_bounds, vec![0]);
        assert_eq!(active_upper_bounds, vec![1]);
    }

    #[test]
    fn bounded_solution_requires_termination_feasibility_and_optimality() {
        let target = DVector::from_vec(vec![2.0]);
        let nuisance = DMatrix::identity(1, 1);
        let bounds = vec![NuisanceBound {
            lower: -1.0,
            upper: 1.0,
            unit: "Pa".to_owned(),
        }];
        let optimum = DVector::from_vec(vec![1.0]);
        assert!(
            validate_bounded_solution(
                SolverCertificate {
                    status: SolverStatus::Solved,
                    primal_residual: 1e-9,
                    dual_residual: 1e-9,
                    absolute_gap: 1e-9,
                    relative_gap: 1e-9,
                },
                &target,
                &nuisance,
                &bounds,
                &optimum,
            )
            .is_ok()
        );
        assert!(
            validate_bounded_solution(
                SolverCertificate {
                    status: SolverStatus::MaxIterations,
                    primal_residual: 1e-9,
                    dual_residual: 1e-9,
                    absolute_gap: 1e-9,
                    relative_gap: 1e-9,
                },
                &target,
                &nuisance,
                &bounds,
                &optimum,
            )
            .is_err()
        );
        assert!(
            validate_bounded_solution(
                SolverCertificate {
                    status: SolverStatus::Solved,
                    primal_residual: 1e-9,
                    dual_residual: 1e-9,
                    absolute_gap: 1e-9,
                    relative_gap: 1e-9,
                },
                &target,
                &nuisance,
                &bounds,
                &DVector::from_vec(vec![1.1]),
            )
            .is_err()
        );
        assert!(
            validate_bounded_solution(
                SolverCertificate {
                    status: SolverStatus::Solved,
                    primal_residual: 1e-9,
                    dual_residual: 1e-9,
                    absolute_gap: 1e-9,
                    relative_gap: 1e-9,
                },
                &target,
                &nuisance,
                &bounds,
                &DVector::from_vec(vec![0.0]),
            )
            .is_err()
        );
    }

    #[test]
    fn deterministic_derived_feature_does_not_add_information() {
        let signal = DVector::from_vec(vec![1.0, -2.0, 0.5]);
        let covariance = DMatrix::identity(3, 3);
        let derivative = DMatrix::from_row_slice(2, 3, &[-1.0, 1.0, 0.0, 0.0, -1.0, 1.0]);
        let mut augmentation = DMatrix::zeros(5, 3);
        augmentation
            .view_mut((0, 0), (3, 3))
            .copy_from(&DMatrix::identity(3, 3));
        augmentation.view_mut((3, 0), (2, 3)).copy_from(&derivative);
        let augmented_signal = &augmentation * &signal;
        let augmented_covariance = &augmentation * covariance * augmentation.transpose();
        let original = signal.norm_squared();
        let augmented =
            fisher_information_with_pseudoinverse(&augmented_signal, &augmented_covariance, 1e-12)
                .unwrap();
        assert_relative_eq!(augmented, original, epsilon = 1e-10);
    }

    #[test]
    fn deterministic_information_is_invariant_under_signal_rescaling() {
        let covariance = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.0]));
        for signal_scale in [1e-12, 1e-7, 1.0, 1e7] {
            let signal = DVector::from_vec(vec![0.0, signal_scale]);
            let information =
                fisher_information_with_pseudoinverse(&signal, &covariance, 1e-12).unwrap();
            assert!(information.is_infinite());
        }
    }

    #[test]
    fn covariance_rejects_indefinite_input_but_accepts_roundoff() {
        let signal = DVector::from_vec(vec![0.0, 1.0]);
        let indefinite = DMatrix::from_diagonal(&DVector::from_vec(vec![-1e-3, 1.0]));
        assert!(fisher_information_with_pseudoinverse(&signal, &indefinite, 1e-12).is_err());

        let roundoff = DMatrix::from_diagonal(&DVector::from_vec(vec![-1e-14, 1.0]));
        assert_relative_eq!(
            fisher_information_with_pseudoinverse(&signal, &roundoff, 1e-12).unwrap(),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn error_margin_and_minimum_effect_boundaries_are_explicit() {
        assert_eq!(
            certified_distance_lower_bound(2.0, 0.5, 0.25).unwrap(),
            1.25
        );
        assert_eq!(certified_distance_lower_bound(0.5, 0.5, 0.25).unwrap(), 0.0);
        assert!(!effect_separation_is_admissible(0.0, 0.1, 0.2).unwrap());
        assert!(effect_separation_is_admissible(0.0, 0.2, 0.2).unwrap());
    }
}
