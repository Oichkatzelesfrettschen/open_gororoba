//! Calibrated mechanism discrimination in whitened measurement coordinates.
//!
//! The functions in this module separate unrestricted linear span tests from
//! nuisance explanations constrained by independent calibration or physical
//! bounds. Callers own the whitening transform and the physical metadata that
//! maps each nuisance coordinate to a specimen or instrument parameter.

use clarabel::{algebra::CscMatrix, solver::*};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

const SOLVER_CERTIFICATE_TOLERANCE: f64 = 1e-7;

type ColumnNormalizationScale = (f64, f64);
type NormalizedDesign = (DMatrix<f64>, Vec<ColumnNormalizationScale>);

/// Input or numerical failure in a discrimination calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscriminationError {
    /// Matrix and vector dimensions disagree.
    DimensionMismatch,
    /// An input contains a non-finite value.
    NonFiniteInput,
    /// A lower bound exceeds its corresponding upper bound.
    InvalidBounds,
    /// A coordinate-specific calibration derivative has no unique nuisance basis.
    RankDeficientDesign,
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
        let feasibility_tolerance = bound_coordinate_tolerance(bound);
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

fn finite_euclidean_norm(vector: &DVector<f64>) -> Result<f64, DiscriminationError> {
    if !all_finite_vector(vector) {
        return Err(DiscriminationError::NumericalFailure);
    }
    let direct_norm = vector.norm();
    if direct_norm.is_finite()
        && (direct_norm > 0.0 || vector.iter().all(|component| *component == 0.0))
    {
        return Ok(direct_norm);
    }
    let maximum_component = vector
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if maximum_component == 0.0 {
        return Ok(0.0);
    }
    let scaled_norm = (vector / maximum_component).norm();
    let norm = maximum_component * scaled_norm;
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(DiscriminationError::NumericalFailure)
    }
}

fn bound_coordinate_tolerance(bound: &NuisanceBound) -> f64 {
    let interval_width = if bound.lower.is_sign_negative() != bound.upper.is_sign_negative() {
        bound.lower.abs().max(bound.upper.abs())
    } else {
        (bound.upper - bound.lower).abs()
    };
    let width_tolerance = SOLVER_CERTIFICATE_TOLERANCE * interval_width;
    let floating_tolerance =
        8.0 * endpoint_spacing(bound.lower).max(endpoint_spacing(bound.upper));
    let tolerance = width_tolerance.max(floating_tolerance);
    if interval_width > 0.0 {
        tolerance.min(0.25 * interval_width)
    } else {
        tolerance
    }
}

fn endpoint_spacing(value: f64) -> f64 {
    [value.next_up() - value, value - value.next_down()]
        .into_iter()
        .filter(|spacing| spacing.is_finite())
        .fold(0.0_f64, f64::max)
}

fn classify_active_bounds(
    parameters: &DVector<f64>,
    bounds: &[NuisanceBound],
) -> (Vec<usize>, Vec<usize>) {
    let mut active_lower_bounds = Vec::new();
    let mut active_upper_bounds = Vec::new();
    for (index, (parameter, bound)) in parameters.iter().zip(bounds).enumerate() {
        let active_tolerance = bound_coordinate_tolerance(bound);
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
    let (normalized, scales) = normalize_design_columns(design)?;
    let scaled_solution = normalized
        .svd(true, true)
        .solve(response, f64::EPSILON.sqrt())
        .map_err(|_| DiscriminationError::NumericalFailure)?;
    let solution = DVector::from_iterator(
        scaled_solution.len(),
        scaled_solution
            .iter()
            .zip(scales)
            .map(|(value, (maximum_component, scaled_norm))| {
                value / scaled_norm / maximum_component
            }),
    );
    if !all_finite_vector(&solution) {
        return Err(DiscriminationError::NumericalFailure);
    }
    Ok(solution)
}

fn normalize_design_columns(
    design: &DMatrix<f64>,
) -> Result<NormalizedDesign, DiscriminationError> {
    let mut normalized = design.clone();
    let mut scales = Vec::with_capacity(normalized.ncols());
    for mut column in normalized.column_iter_mut() {
        let direct_norm = column.norm();
        if direct_norm.is_finite() && direct_norm > 0.0 {
            column.scale_mut(1.0 / direct_norm);
            scales.push((direct_norm, 1.0));
            continue;
        }
        let maximum_component = column
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        if maximum_component == 0.0 {
            scales.push((1.0, 1.0));
            continue;
        }
        for value in column.iter_mut() {
            *value /= maximum_component;
        }
        let scaled_norm = column.norm();
        if !scaled_norm.is_finite() || scaled_norm == 0.0 {
            return Err(DiscriminationError::NumericalFailure);
        }
        column.scale_mut(1.0 / scaled_norm);
        scales.push((maximum_component, scaled_norm));
    }
    Ok((normalized, scales))
}

fn has_full_column_rank(design: &DMatrix<f64>) -> Result<bool, DiscriminationError> {
    let (normalized, _) = normalize_design_columns(design)?;
    if normalized.column_iter().any(|column| column.amax() == 0.0) {
        return Ok(false);
    }
    let singular_values = normalized.svd(false, false).singular_values;
    if !all_finite_vector(&singular_values) {
        return Err(DiscriminationError::NumericalFailure);
    }
    Ok(singular_values
        .iter()
        .filter(|singular_value| **singular_value > f64::EPSILON.sqrt())
        .count()
        == design.ncols())
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
    let information = residual.norm_squared();
    if !information.is_finite() {
        return Err(DiscriminationError::NumericalFailure);
    }
    Ok(information)
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
    if !has_full_column_rank(&augmented_design)? {
        return Err(DiscriminationError::RankDeficientDesign);
    }
    let mut augmented_target = DVector::zeros(target.len() + calibration.nrows());
    augmented_target.rows_mut(0, target.len()).copy_from(target);
    let solution = least_squares(&augmented_design, &augmented_target)?;
    let sensitivity = solution[coordinate].powi(2);
    if !sensitivity.is_finite() {
        return Err(DiscriminationError::NumericalFailure);
    }
    Ok(sensitivity)
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

    if bounds.iter().all(|bound| bound.lower == bound.upper) {
        let parameters = DVector::from_iterator(
            bounds.len(),
            bounds.iter().map(|bound| bound.lower),
        );
        let distance = finite_euclidean_norm(&(target - nuisance * &parameters))?;
        let active_bounds = (0..bounds.len()).collect::<Vec<_>>();
        return Ok(BoundedProfileResult {
            distance,
            nuisance_parameters: parameters,
            active_lower_bounds: active_bounds.clone(),
            active_upper_bounds: active_bounds,
        });
    }

    let fixed_coordinates: Vec<_> = bounds
        .iter()
        .enumerate()
        .filter_map(|(index, bound)| (bound.lower == bound.upper).then_some(index))
        .collect();
    if !fixed_coordinates.is_empty() {
        let free_coordinates: Vec<_> = bounds
            .iter()
            .enumerate()
            .filter_map(|(index, bound)| (bound.lower != bound.upper).then_some(index))
            .collect();
        let mut residual_target = target.clone();
        for &coordinate in &fixed_coordinates {
            residual_target -= nuisance.column(coordinate) * bounds[coordinate].lower;
        }
        if !all_finite_vector(&residual_target) {
            return Err(DiscriminationError::NumericalFailure);
        }
        let free_nuisance = DMatrix::from_fn(target.len(), free_coordinates.len(), |row, column| {
            nuisance[(row, free_coordinates[column])]
        });
        let free_bounds: Vec<_> = free_coordinates
            .iter()
            .map(|&coordinate| bounds[coordinate].clone())
            .collect();
        let free_result = bounded_profile_distance(&residual_target, &free_nuisance, &free_bounds)?;
        let mut parameters = DVector::zeros(bounds.len());
        let mut active_lower_bounds = fixed_coordinates.clone();
        let mut active_upper_bounds = fixed_coordinates;
        for &coordinate in &active_lower_bounds {
            parameters[coordinate] = bounds[coordinate].lower;
        }
        for (free_index, &coordinate) in free_coordinates.iter().enumerate() {
            parameters[coordinate] = free_result.nuisance_parameters[free_index];
            if free_result.active_lower_bounds.contains(&free_index) {
                active_lower_bounds.push(coordinate);
            }
            if free_result.active_upper_bounds.contains(&free_index) {
                active_upper_bounds.push(coordinate);
            }
        }
        active_lower_bounds.sort_unstable();
        active_upper_bounds.sort_unstable();
        let distance = finite_euclidean_norm(&(target - nuisance * &parameters))?;
        return Ok(BoundedProfileResult {
            distance,
            nuisance_parameters: parameters,
            active_lower_bounds,
            active_upper_bounds,
        });
    }

    let parameter_count = nuisance.ncols();
    let objective_scale = target
        .amax()
        .max(nuisance.amax())
        .max(f64::MIN_POSITIVE);
    let scaled_target = target / objective_scale;
    let column_scales: Vec<(f64, f64)> = nuisance
        .column_iter()
        .map(|column| {
            let maximum_component = column.amax();
            if maximum_component == 0.0 {
                (1.0, 1.0)
            } else {
                let normalized_norm = (column / maximum_component).norm();
                (maximum_component, normalized_norm)
            }
        })
        .collect();
    let mut solver_nuisance = nuisance.clone();
    for (mut column, (maximum_component, normalized_norm)) in
        solver_nuisance.column_iter_mut().zip(&column_scales)
    {
        column.scale_mut(1.0 / maximum_component);
        column.scale_mut(1.0 / normalized_norm);
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
    let mut right_hand_side = Vec::with_capacity(2 * parameter_count);
    for (bound, (maximum_component, normalized_norm)) in bounds.iter().zip(&column_scales) {
        let scaled_upper =
            ((bound.upper * maximum_component) * normalized_norm) / objective_scale;
        if !scaled_upper.is_finite() {
            return Err(DiscriminationError::NumericalFailure);
        }
        right_hand_side.push(scaled_upper);
    }
    for (bound, (maximum_component, normalized_norm)) in bounds.iter().zip(&column_scales) {
        let scaled_lower =
            ((-bound.lower * maximum_component) * normalized_norm) / objective_scale;
        if !scaled_lower.is_finite() {
            return Err(DiscriminationError::NumericalFailure);
        }
        right_hand_side.push(scaled_lower);
    }
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
            .map(|(parameter, (maximum_component, normalized_norm))| {
                ((parameter * objective_scale) / normalized_norm) / maximum_component
            }),
    );
    if !all_finite_vector(&parameters) {
        return Err(DiscriminationError::NumericalFailure);
    }
    let solver_bounds: Vec<_> = bounds
        .iter()
        .zip(&column_scales)
        .map(|(bound, (maximum_component, normalized_norm))| NuisanceBound {
            lower: ((bound.lower * maximum_component) * normalized_norm) / objective_scale,
            upper: ((bound.upper * maximum_component) * normalized_norm) / objective_scale,
            unit: bound.unit.clone(),
        })
        .collect();
    validate_bounded_solution(
        SolverCertificate {
            status: solver.solution.status,
            primal_residual: solver.solution.r_prim,
            dual_residual: solver.solution.r_dual,
            absolute_gap: solver.info.gap_abs,
            relative_gap: solver.info.gap_rel,
        },
        &scaled_target,
        &solver_nuisance,
        &solver_bounds,
        &scaled_parameters,
    )?;
    let distance = finite_euclidean_norm(&(target - nuisance * &parameters))?;
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
    let covariance_scale = covariance
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if covariance_scale > 0.0 {
        let scaled_covariance = covariance / covariance_scale;
        let scaled_symmetry_error = &scaled_covariance - scaled_covariance.transpose();
        if scaled_symmetry_error.norm() > 1e-10 * scaled_covariance.norm() {
            return Err(DiscriminationError::NumericalFailure);
        }
    }
    if (0..covariance.nrows()).any(|index| covariance[(index, index)] < 0.0) {
        return Err(DiscriminationError::NumericalFailure);
    }
    // Positive diagonal congruence preserves inertia. Normalize through the
    // standard deviations so a large coordinate cannot hide an invalid mode
    // elsewhere. A zero-variance coordinate must have zero covariance with
    // every other coordinate.
    let mut correlation = DMatrix::identity(covariance.nrows(), covariance.ncols());
    let effective_relative_tolerance = relative_tolerance.max(64.0 * f64::EPSILON);
    for row in 0..covariance.nrows() {
        if covariance[(row, row)] == 0.0 {
            correlation[(row, row)] = 0.0;
        }
        for column in (row + 1)..covariance.ncols() {
            let row_deviation = covariance[(row, row)].sqrt();
            let column_deviation = covariance[(column, column)].sqrt();
            let upper = covariance[(row, column)];
            let lower = covariance[(column, row)];
            if row_deviation == 0.0 || column_deviation == 0.0 {
                if upper != 0.0 || lower != 0.0 {
                    return Err(DiscriminationError::NumericalFailure);
                }
                correlation[(row, column)] = 0.0;
                correlation[(column, row)] = 0.0;
                continue;
            }
            let larger_deviation = row_deviation.max(column_deviation);
            let smaller_deviation = row_deviation.min(column_deviation);
            let upper_correlation = upper / larger_deviation / smaller_deviation;
            let lower_correlation = lower / larger_deviation / smaller_deviation;
            if !upper_correlation.is_finite()
                || !lower_correlation.is_finite()
                || (upper_correlation - lower_correlation).abs()
                    > effective_relative_tolerance
            {
                return Err(DiscriminationError::NumericalFailure);
            }
            let symmetric_correlation = 0.5 * (upper_correlation + lower_correlation);
            if symmetric_correlation.abs() > 1.0 + effective_relative_tolerance {
                return Err(DiscriminationError::NumericalFailure);
            }
            correlation[(row, column)] = symmetric_correlation;
            correlation[(column, row)] = symmetric_correlation;
        }
    }
    let coordinate_scaled_decomposition = SymmetricEigen::new(correlation);
    let coordinate_scaled_spectral_scale = coordinate_scaled_decomposition
        .eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let eigenvalue_tolerance = effective_relative_tolerance * coordinate_scaled_spectral_scale;
    if coordinate_scaled_decomposition
        .eigenvalues
        .iter()
        .any(|eigenvalue| *eigenvalue < -eigenvalue_tolerance)
    {
        return Err(DiscriminationError::NumericalFailure);
    }
    let symmetric_covariance = covariance * 0.5 + covariance.transpose() * 0.5;
    let decomposition = SymmetricEigen::new(symmetric_covariance);
    let maximum = decomposition
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let threshold = effective_relative_tolerance * maximum;
    let coordinates = decomposition.eigenvectors.transpose() * signal;
    let mut information = 0.0;
    for (coordinate_index, (eigenvalue, coordinate)) in decomposition
        .eigenvalues
        .iter()
        .zip(coordinates.iter())
        .enumerate()
    {
        if *eigenvalue > threshold {
            // Division before squaring preserves quotients whose unscaled
            // coordinate square underflows even though the Fisher term does not.
            let standardized_coordinate = coordinate / eigenvalue.sqrt();
            information += standardized_coordinate * standardized_coordinate;
            if !information.is_finite() {
                return Err(DiscriminationError::NumericalFailure);
            }
        } else {
            let projection_scale: f64 = decomposition
                .eigenvectors
                .column(coordinate_index)
                .iter()
                .zip(signal.iter())
                .map(|(basis_component, signal_component)| {
                    (basis_component * signal_component).abs()
                })
                .sum();
            let roundoff_tolerance = signal.len() as f64 * f64::EPSILON * projection_scale;
            if coordinate.abs() > roundoff_tolerance {
                return Ok(f64::INFINITY);
            }
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
    fn unrestricted_residual_preserves_extreme_finite_column_scales() {
        for scale in [1e-200, 1e200] {
            let target = DVector::from_element(1, 1.0);
            let nuisance = DMatrix::from_element(1, 1, scale);
            assert!(unrestricted_residual(&target, &nuisance).unwrap().norm() < 1e-12);
        }
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
    fn efficient_information_rejects_finite_inputs_with_overflowing_result() {
        let target = DVector::from_element(1, 1e200);
        let nuisance = DMatrix::zeros(1, 1);
        let calibration = DMatrix::zeros(0, 1);

        assert_eq!(
            efficient_information(&target, &nuisance, &calibration),
            Err(DiscriminationError::NumericalFailure)
        );
    }

    #[test]
    fn calibration_sensitivity_rejects_finite_inputs_with_overflowing_result() {
        let target = DVector::from_element(1, 1e200);
        let nuisance = DMatrix::identity(1, 1);
        let calibration = DMatrix::zeros(0, 1);

        assert_eq!(
            calibration_precision_sensitivity(&target, &nuisance, &calibration, 0),
            Err(DiscriminationError::NumericalFailure)
        );
    }

    #[test]
    fn calibration_sensitivity_preserves_underflow_scale_information() {
        let target = DVector::from_element(1, 1e-200);
        let nuisance = DMatrix::from_element(1, 1, 1e-200);
        let calibration = DMatrix::zeros(0, 1);

        let sensitivity =
            calibration_precision_sensitivity(&target, &nuisance, &calibration, 0).unwrap();
        assert_relative_eq!(sensitivity, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn calibration_sensitivity_rejects_rank_deficient_nuisance_coordinates() {
        let target = DVector::from_element(1, 1.0);
        let nuisance = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let calibration = DMatrix::zeros(0, 2);

        assert_eq!(
            calibration_precision_sensitivity(&target, &nuisance, &calibration, 0),
            Err(DiscriminationError::RankDeficientDesign)
        );
        assert_relative_eq!(
            efficient_information(&target, &nuisance, &calibration).unwrap(),
            0.0,
            epsilon = 1e-12
        );
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
    fn bounded_profile_rejects_an_overflowed_distance() {
        let target = DVector::from_vec(vec![1.3e308, 1.3e308]);
        let nuisance = DMatrix::from_column_slice(2, 1, &[1.3e308, 1.3e308]);
        let bounds = vec![NuisanceBound {
            lower: 0.0,
            upper: 0.0,
            unit: "Pa".to_owned(),
        }];

        assert_eq!(
            bounded_profile_distance(&target, &nuisance, &bounds),
            Err(DiscriminationError::NumericalFailure)
        );
    }

    #[test]
    fn bounded_profile_recovers_a_near_limit_finite_distance() {
        let target = DVector::from_vec(vec![1e308, 1e308]);
        let nuisance = DMatrix::from_column_slice(2, 1, &[1e308, 1e308]);
        let bounds = vec![NuisanceBound {
            lower: 0.0,
            upper: 0.0,
            unit: "Pa".to_owned(),
        }];

        let result = bounded_profile_distance(&target, &nuisance, &bounds).unwrap();
        assert_relative_eq!(result.distance / 1e308, 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn bounded_profile_recovers_a_large_finite_distance() {
        let target = DVector::from_vec(vec![1e200, 1e200]);
        let nuisance = DMatrix::from_column_slice(2, 1, &[1e200, 1e200]);
        let bounds = vec![NuisanceBound {
            lower: 0.0,
            upper: 0.0,
            unit: "Pa".to_owned(),
        }];

        let result = bounded_profile_distance(&target, &nuisance, &bounds).unwrap();
        assert_relative_eq!(result.distance / 1e200, 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn bounded_profile_preserves_a_tiny_nonzero_distance() {
        let target = DVector::from_element(1, 1e-200);
        let nuisance = DMatrix::identity(1, 1);
        let bounds = vec![NuisanceBound {
            lower: 0.0,
            upper: 0.0,
            unit: "Pa".to_owned(),
        }];

        let result = bounded_profile_distance(&target, &nuisance, &bounds).unwrap();
        assert!(result.distance > 0.0);
        assert_relative_eq!(result.distance / 1e-200, 1.0, epsilon = 1e-12);
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
    fn active_bound_classification_handles_a_wide_finite_interval() {
        let parameters = DVector::from_vec(vec![0.0]);
        let bounds = vec![NuisanceBound {
            lower: -1e308,
            upper: 1e308,
            unit: "Pa".to_owned(),
        }];

        let (active_lower_bounds, active_upper_bounds) =
            classify_active_bounds(&parameters, &bounds);

        assert!(active_lower_bounds.is_empty());
        assert!(active_upper_bounds.is_empty());
    }

    #[test]
    fn active_bound_classification_uses_interval_width_at_large_offsets() {
        let lower = 1e12;
        let upper = lower + 1.0;
        let parameters = DVector::from_vec(vec![lower, lower + 0.5, upper]);
        let bounds = vec![
            NuisanceBound {
                lower,
                upper,
                unit: "Pa".to_owned(),
            };
            3
        ];

        let (active_lower_bounds, active_upper_bounds) =
            classify_active_bounds(&parameters, &bounds);

        assert_eq!(active_lower_bounds, vec![0]);
        assert_eq!(active_upper_bounds, vec![2]);
    }

    #[test]
    fn active_bound_classification_preserves_tiny_interval_interior() {
        let parameters = DVector::from_vec(vec![0.0, 5e-21, 1e-20]);
        let bounds = vec![
            NuisanceBound {
                lower: 0.0,
                upper: 1e-20,
                unit: "Pa".to_owned(),
            };
            3
        ];

        let (active_lower_bounds, active_upper_bounds) =
            classify_active_bounds(&parameters, &bounds);

        assert_eq!(active_lower_bounds, vec![0]);
        assert_eq!(active_upper_bounds, vec![2]);
    }

    #[test]
    fn active_bound_classification_preserves_subnormal_endpoint_spacing() {
        let lower = 0.0_f64;
        let upper = 1e-320_f64;
        let parameters = DVector::from_vec(vec![lower.next_up(), 5e-321, upper.next_down()]);
        let bounds = vec![
            NuisanceBound {
                lower,
                upper,
                unit: "Pa".to_owned(),
            };
            3
        ];

        let (active_lower_bounds, active_upper_bounds) =
            classify_active_bounds(&parameters, &bounds);

        assert_eq!(active_lower_bounds, vec![0]);
        assert_eq!(active_upper_bounds, vec![2]);
    }

    #[test]
    fn bounded_profile_preserves_a_tiny_nuisance_column() {
        let target = DVector::from_vec(vec![1.0]);
        let nuisance = DMatrix::from_vec(1, 1, vec![1e-200]);
        let bounds = vec![NuisanceBound {
            lower: -1e200,
            upper: 1e200,
            unit: "Pa".to_owned(),
        }];

        let result = bounded_profile_distance(&target, &nuisance, &bounds).unwrap();
        let active_solution_tolerance = SOLVER_CERTIFICATE_TOLERANCE.sqrt();
        assert!(result.distance <= active_solution_tolerance);
        assert_relative_eq!(
            result.nuisance_parameters[0] / 1e200,
            1.0,
            epsilon = active_solution_tolerance
        );
    }

    #[test]
    fn bounded_profile_preserves_columns_across_extreme_finite_scales() {
        let target = DVector::from_vec(vec![1e200, 1.0]);
        let nuisance = DMatrix::from_diagonal(&DVector::from_vec(vec![1e200, 1e-200]));
        let bounds = vec![
            NuisanceBound {
                lower: 1.0,
                upper: 1.0,
                unit: "fixed".to_owned(),
            },
            NuisanceBound {
                lower: 0.0,
                upper: 2e200,
                unit: "free".to_owned(),
            },
        ];

        let result = bounded_profile_distance(&target, &nuisance, &bounds).unwrap();
        assert!(result.distance <= SOLVER_CERTIFICATE_TOLERANCE);
        assert_eq!(result.nuisance_parameters[0], 1.0);
        assert_relative_eq!(result.nuisance_parameters[1] / 1e200, 1.0, epsilon = 1e-7);
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
    fn noisy_signal_component_does_not_hide_deterministic_information() {
        let covariance = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.0]));
        for signal_scale in [1e-12, 1.0, 1e12] {
            let signal = DVector::from_vec(vec![signal_scale, 1e-7 * signal_scale]);
            let information =
                fisher_information_with_pseudoinverse(&signal, &covariance, 1e-12).unwrap();
            assert!(information.is_infinite());
        }
    }

    #[test]
    fn bounded_solution_feasibility_tracks_small_parameter_units() {
        let target = DVector::from_vec(vec![0.0]);
        let nuisance = DMatrix::identity(1, 1);
        let bounds = vec![NuisanceBound {
            lower: 1e-10,
            upper: 2e-10,
            unit: "m".to_owned(),
        }];
        let outside = DVector::from_vec(vec![0.0]);
        assert_eq!(
            validate_bounded_solution(
                SolverCertificate {
                    status: SolverStatus::Solved,
                    primal_residual: 0.0,
                    dual_residual: 0.0,
                    absolute_gap: 0.0,
                    relative_gap: 0.0,
                },
                &target,
                &nuisance,
                &bounds,
                &outside,
            ),
            Err(DiscriminationError::NumericalFailure)
        );
    }

    #[test]
    fn bounded_solution_feasibility_uses_width_at_large_offsets() {
        let target = DVector::from_vec(vec![0.0]);
        let nuisance = DMatrix::zeros(1, 1);
        let bounds = vec![NuisanceBound {
            lower: 1e12,
            upper: 1e12 + 1.0,
            unit: "Pa".to_owned(),
        }];
        let outside = DVector::from_vec(vec![1e12 + 1001.0]);
        assert_eq!(
            validate_bounded_solution(
                SolverCertificate {
                    status: SolverStatus::Solved,
                    primal_residual: 0.0,
                    dual_residual: 0.0,
                    absolute_gap: 0.0,
                    relative_gap: 0.0,
                },
                &target,
                &nuisance,
                &bounds,
                &outside,
            ),
            Err(DiscriminationError::NumericalFailure)
        );
    }

    #[test]
    fn covariance_rejects_negative_variance_but_accepts_symmetric_roundoff() {
        let signal = DVector::from_vec(vec![0.0, 1.0]);
        let indefinite = DMatrix::from_diagonal(&DVector::from_vec(vec![-1e-3, 1.0]));
        assert!(fisher_information_with_pseudoinverse(&signal, &indefinite, 1e-12).is_err());

        let negative_roundoff = DMatrix::from_diagonal(&DVector::from_vec(vec![-1e-14, 1.0]));
        assert!(
            fisher_information_with_pseudoinverse(&signal, &negative_roundoff, 1e-12).is_err()
        );

        let symmetric_roundoff = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        assert!(
            fisher_information_with_pseudoinverse(&signal, &symmetric_roundoff, 1e-12).is_ok()
        );
    }

    #[test]
    fn covariance_psd_validation_scales_each_coordinate() {
        let signal = DVector::from_vec(vec![1.0, 1.0]);
        let indefinite = DMatrix::from_diagonal(&DVector::from_vec(vec![1e200, -1.0]));
        let tiny_negative_variance =
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-300, -1e-320]));
        let zero_variance_with_covariance =
            DMatrix::from_row_slice(2, 2, &[1e200, 1e90, 1e90, 0.0]);

        assert_eq!(
            fisher_information_with_pseudoinverse(&signal, &indefinite, 1e-12),
            Err(DiscriminationError::NumericalFailure)
        );
        assert_eq!(
            fisher_information_with_pseudoinverse(&signal, &tiny_negative_variance, 1e-12),
            Err(DiscriminationError::NumericalFailure)
        );
        assert_eq!(
            fisher_information_with_pseudoinverse(
                &signal,
                &zero_variance_with_covariance,
                1e-12
            ),
            Err(DiscriminationError::NumericalFailure)
        );
    }

    #[test]
    fn fisher_information_scales_tiny_signal_before_squaring() {
        let signal = DVector::from_element(1, 1e-200);
        let covariance = DMatrix::from_element(1, 1, 1e-300);

        let information =
            fisher_information_with_pseudoinverse(&signal, &covariance, 1e-12).unwrap();
        assert_relative_eq!(information, 1e-100, max_relative = 1e-12);
    }

    #[test]
    fn covariance_symmetry_tolerance_tracks_covariance_scale() {
        let signal = DVector::from_vec(vec![1.0, 0.0]);
        let asymmetric = DMatrix::from_row_slice(2, 2, &[1e-20, 1e-20, 0.0, 1e-20]);
        assert_eq!(
            fisher_information_with_pseudoinverse(&signal, &asymmetric, 1e-12),
            Err(DiscriminationError::NumericalFailure)
        );
    }

    #[test]
    fn covariance_symmetry_check_survives_large_finite_entries() {
        let signal = DVector::from_vec(vec![1.0, 0.0]);
        let asymmetric = DMatrix::from_row_slice(2, 2, &[1e200, 1e200, 0.0, 1e200]);
        assert_eq!(
            fisher_information_with_pseudoinverse(&signal, &asymmetric, 1e-12),
            Err(DiscriminationError::NumericalFailure)
        );
    }

    #[test]
    fn fisher_information_rejects_an_overflowed_noisy_result() {
        let signal = DVector::from_element(1, 1e200);
        let covariance = DMatrix::identity(1, 1);
        assert_eq!(
            fisher_information_with_pseudoinverse(&signal, &covariance, 1e-12),
            Err(DiscriminationError::NumericalFailure)
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
