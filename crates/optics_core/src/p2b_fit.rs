//! Deterministic held-out fitting for the Ruan-Fan single-resonance model.
//!
//! The fitting code receives coordinates and complex Mie amplitudes from the
//! caller, but never sees a test coordinate. The optimizer uses fixed starts,
//! fixed bounds, bounded Nelder-Mead updates, and a fixed iteration budget.
//! Complex residuals remain the objective; scalar observables are diagnostics.

use num_complex::Complex64;
use thiserror::Error;

/// One complex channel observation at a declared real frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexSample {
    /// Real angular frequency.
    pub omega: f64,
    /// Complex S_l or R_l value.
    pub value: Complex64,
}

/// Parameters of the source-constrained TCMT reflection model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitParameters {
    /// Resonance frequency.
    pub omega_0: f64,
    /// Radiative rate.
    pub gamma: f64,
    /// Intrinsic loss rate.
    pub gamma_0: f64,
    /// Background phase.
    pub phi: f64,
}

/// Parameters for an unconstrained complex one-pole comparison model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnePoleParameters {
    /// Constant complex background.
    pub background: Complex64,
    /// Complex pole residue.
    pub residue: Complex64,
    /// Real pole coordinate.
    pub omega_0: f64,
    /// Positive pole half-width.
    pub gamma: f64,
}

/// One optimizer start and its held-out-free diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct FitStartResult {
    /// Starting point.
    pub start: Vec<f64>,
    /// Final point.
    pub result: Vec<f64>,
    /// Training sum of squared complex residuals.
    pub training_error: f64,
    /// Validation sum of squared complex residuals.
    pub validation_error: f64,
    /// Whether the objective spread and parameter spread met the convergence
    /// criterion before the iteration limit.
    pub converged: bool,
    /// Iterations used.
    pub iterations: usize,
}

/// A selected TCMT fit with every deterministic start retained.
#[derive(Debug, Clone, PartialEq)]
pub struct TcmtFitReport {
    /// Selected parameters.
    pub parameters: FitParameters,
    /// Training error of the selected parameters.
    pub training_error: f64,
    /// Validation error of the selected parameters.
    pub validation_error: f64,
    /// Every fixed start and optimizer result.
    pub starts: Vec<FitStartResult>,
}

/// A selected unconstrained one-pole fit with every start retained.
#[derive(Debug, Clone, PartialEq)]
pub struct OnePoleFitReport {
    /// Selected parameters.
    pub parameters: OnePoleParameters,
    /// Training error of the selected parameters.
    pub training_error: f64,
    /// Validation error of the selected parameters.
    pub validation_error: f64,
    /// Every fixed start and optimizer result.
    pub starts: Vec<FitStartResult>,
}

/// Local fitting failures.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum FitError {
    #[error("the fitting data set is empty")]
    EmptyData,
    #[error("the fitting data contain a non-finite value")]
    NonFiniteData,
    #[error("the fitting bounds are invalid")]
    InvalidBounds,
    #[error("the optimizer received no fixed starts")]
    EmptyStarts,
    #[error("the optimizer produced a non-finite objective")]
    NonFiniteObjective,
}

fn finite_complex(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn validate_samples(samples: &[ComplexSample]) -> Result<(), FitError> {
    if samples.is_empty() {
        return Err(FitError::EmptyData);
    }
    if samples
        .iter()
        .any(|sample| !sample.omega.is_finite() || !finite_complex(sample.value))
    {
        return Err(FitError::NonFiniteData);
    }
    Ok(())
}

fn validate_bounds<const N: usize>(bounds: &[[f64; 2]; N]) -> Result<(), FitError> {
    if bounds
        .iter()
        .any(|bound| !bound[0].is_finite() || !bound[1].is_finite() || bound[0] > bound[1])
    {
        return Err(FitError::InvalidBounds);
    }
    Ok(())
}

fn clamp<const N: usize>(point: [f64; N], bounds: &[[f64; 2]; N]) -> [f64; N] {
    let mut output = point;
    for (value, bound) in output.iter_mut().zip(bounds) {
        *value = value.clamp(bound[0], bound[1]);
    }
    output
}

fn squared_complex_error(left: Complex64, right: Complex64) -> f64 {
    (left - right).norm_sqr()
}

/// Evaluate the source Eq. 21 reflection amplitude.
pub fn tcmt_reflection(parameters: FitParameters, omega: f64) -> Complex64 {
    let numerator = Complex64::new(
        parameters.gamma_0 - parameters.gamma,
        parameters.omega_0 - omega,
    );
    let denominator = Complex64::new(
        parameters.gamma_0 + parameters.gamma,
        parameters.omega_0 - omega,
    );
    Complex64::from_polar(1.0, parameters.phi) * numerator / denominator
}

/// Evaluate the source Eq. 8 scattering amplitude.
pub fn tcmt_scattering(parameters: FitParameters, omega: f64) -> Complex64 {
    (tcmt_reflection(parameters, omega) - Complex64::new(1.0, 0.0)) / 2.0
}

/// Evaluate the unconstrained complex one-pole model.
pub fn one_pole_value(parameters: OnePoleParameters, omega: f64) -> Complex64 {
    let denominator = Complex64::new(omega - parameters.omega_0, parameters.gamma);
    parameters.background + parameters.residue / denominator
}

fn tcmt_objective(parameters: FitParameters, samples: &[ComplexSample]) -> f64 {
    samples
        .iter()
        .map(|sample| {
            squared_complex_error(tcmt_scattering(parameters, sample.omega), sample.value)
        })
        .sum()
}

fn one_pole_objective(parameters: OnePoleParameters, samples: &[ComplexSample]) -> f64 {
    samples
        .iter()
        .map(|sample| squared_complex_error(one_pole_value(parameters, sample.omega), sample.value))
        .sum()
}

fn bounded_nelder_mead<const N: usize, F>(
    start: [f64; N],
    steps: [f64; N],
    bounds: [[f64; 2]; N],
    max_iterations: usize,
    objective: F,
) -> Result<([f64; N], f64, bool, usize), FitError>
where
    F: Fn([f64; N]) -> f64,
{
    let mut simplex = Vec::with_capacity(N + 1);
    simplex.push(clamp(start, &bounds));
    for dimension in 0..N {
        let mut point = start;
        point[dimension] += steps[dimension];
        simplex.push(clamp(point, &bounds));
    }
    let mut values: Vec<f64> = simplex.iter().map(|point| objective(*point)).collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FitError::NonFiniteObjective);
    }

    let mut converged = false;
    let mut iterations = 0;
    for iteration in 0..max_iterations {
        iterations = iteration + 1;
        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
        let ordered_simplex: Vec<[f64; N]> = order.iter().map(|index| simplex[*index]).collect();
        let ordered_values: Vec<f64> = order.iter().map(|index| values[*index]).collect();
        simplex = ordered_simplex;
        values = ordered_values;

        let objective_spread = values[N] - values[0];
        let parameter_spread = (0..N)
            .map(|dimension| (simplex[N][dimension] - simplex[0][dimension]).abs())
            .fold(0.0, f64::max);
        if objective_spread <= 1e-24 * (1.0 + values[0].abs()) && parameter_spread <= 1e-10 {
            converged = true;
            break;
        }

        let mut centroid = [0.0; N];
        for point in simplex.iter().take(N) {
            for (dimension, value) in point.iter().enumerate() {
                centroid[dimension] += value / N as f64;
            }
        }
        let worst = simplex[N];
        let reflected = clamp(
            std::array::from_fn(|dimension| {
                centroid[dimension] + (centroid[dimension] - worst[dimension])
            }),
            &bounds,
        );
        let reflected_value = objective(reflected);
        if !reflected_value.is_finite() {
            return Err(FitError::NonFiniteObjective);
        }

        if reflected_value < values[0] {
            let expanded = clamp(
                std::array::from_fn(|dimension| {
                    centroid[dimension] + 2.0 * (reflected[dimension] - centroid[dimension])
                }),
                &bounds,
            );
            let expanded_value = objective(expanded);
            if !expanded_value.is_finite() {
                return Err(FitError::NonFiniteObjective);
            }
            if expanded_value < reflected_value {
                simplex[N] = expanded;
                values[N] = expanded_value;
            } else {
                simplex[N] = reflected;
                values[N] = reflected_value;
            }
        } else if reflected_value < values[N - 1] {
            simplex[N] = reflected;
            values[N] = reflected_value;
        } else {
            let contraction = if reflected_value < values[N] {
                std::array::from_fn(|dimension| {
                    centroid[dimension] + 0.5 * (reflected[dimension] - centroid[dimension])
                })
            } else {
                std::array::from_fn(|dimension| {
                    centroid[dimension] + 0.5 * (worst[dimension] - centroid[dimension])
                })
            };
            let contraction = clamp(contraction, &bounds);
            let contraction_value = objective(contraction);
            if !contraction_value.is_finite() {
                return Err(FitError::NonFiniteObjective);
            }
            if contraction_value < values[N] {
                simplex[N] = contraction;
                values[N] = contraction_value;
            } else {
                let best = simplex[0];
                for index in 1..=N {
                    simplex[index] = clamp(
                        std::array::from_fn(|dimension| {
                            best[dimension] + 0.5 * (simplex[index][dimension] - best[dimension])
                        }),
                        &bounds,
                    );
                    values[index] = objective(simplex[index]);
                    if !values[index].is_finite() {
                        return Err(FitError::NonFiniteObjective);
                    }
                }
            }
        }
    }
    let mut best_index = 0;
    for index in 1..simplex.len() {
        if values[index] < values[best_index] {
            best_index = index;
        }
    }
    Ok((
        simplex[best_index],
        values[best_index],
        converged,
        iterations,
    ))
}

fn fit_start_result(
    start: &[f64],
    result: &[f64],
    training_error: f64,
    validation_error: f64,
    converged: bool,
    iterations: usize,
) -> FitStartResult {
    FitStartResult {
        start: start.to_vec(),
        result: result.to_vec(),
        training_error,
        validation_error,
        converged,
        iterations,
    }
}

/// Fit the source-constrained TCMT model using fixed starts and bounds.
pub fn fit_tcmt(
    training: &[ComplexSample],
    validation: &[ComplexSample],
    starts: &[[f64; 4]],
    bounds: [[f64; 2]; 4],
    fixed_phase: Option<f64>,
    max_iterations: usize,
) -> Result<TcmtFitReport, FitError> {
    validate_samples(training)?;
    validate_samples(validation)?;
    validate_bounds(&bounds)?;
    if starts.is_empty() {
        return Err(FitError::EmptyStarts);
    }
    let mut results = Vec::with_capacity(starts.len());
    for start in starts {
        let start = clamp(*start, &bounds);
        let steps = std::array::from_fn(|dimension| {
            if bounds[dimension][0] == bounds[dimension][1] {
                0.0
            } else {
                (bounds[dimension][1] - bounds[dimension][0]) * 0.02
            }
        });
        let objective = |point: [f64; 4]| {
            let parameters = FitParameters {
                omega_0: point[0],
                gamma: point[1],
                gamma_0: point[2],
                phi: fixed_phase.unwrap_or(point[3]),
            };
            tcmt_objective(parameters, training)
        };
        let (point, training_error, converged, iterations) =
            bounded_nelder_mead(start, steps, bounds, max_iterations, objective)?;
        let parameters = FitParameters {
            omega_0: point[0],
            gamma: point[1],
            gamma_0: point[2],
            phi: fixed_phase.unwrap_or(point[3]),
        };
        let validation_error = tcmt_objective(parameters, validation);
        if !validation_error.is_finite() {
            return Err(FitError::NonFiniteObjective);
        }
        results.push(fit_start_result(
            &start,
            &point,
            training_error,
            validation_error,
            converged,
            iterations,
        ));
    }
    let selected_index = results
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            left.validation_error
                .total_cmp(&right.validation_error)
                .then_with(|| left.training_error.total_cmp(&right.training_error))
        })
        .map(|(index, _)| index)
        .ok_or(FitError::EmptyStarts)?;
    let point = &results[selected_index].result;
    let parameters = FitParameters {
        omega_0: point[0],
        gamma: point[1],
        gamma_0: point[2],
        phi: fixed_phase.unwrap_or(point[3]),
    };
    Ok(TcmtFitReport {
        parameters,
        training_error: results[selected_index].training_error,
        validation_error: results[selected_index].validation_error,
        starts: results,
    })
}

/// Fit an unconstrained complex one-pole model using fixed starts and bounds.
pub fn fit_one_pole(
    training: &[ComplexSample],
    validation: &[ComplexSample],
    starts: &[[f64; 6]],
    bounds: [[f64; 2]; 6],
    max_iterations: usize,
) -> Result<OnePoleFitReport, FitError> {
    validate_samples(training)?;
    validate_samples(validation)?;
    validate_bounds(&bounds)?;
    if starts.is_empty() {
        return Err(FitError::EmptyStarts);
    }
    let mut results = Vec::with_capacity(starts.len());
    for start in starts {
        let start = clamp(*start, &bounds);
        let steps =
            std::array::from_fn(|dimension| (bounds[dimension][1] - bounds[dimension][0]) * 0.02);
        let objective = |point: [f64; 6]| {
            one_pole_objective(
                OnePoleParameters {
                    background: Complex64::new(point[0], point[1]),
                    residue: Complex64::new(point[2], point[3]),
                    omega_0: point[4],
                    gamma: point[5],
                },
                training,
            )
        };
        let (point, training_error, converged, iterations) =
            bounded_nelder_mead(start, steps, bounds, max_iterations, objective)?;
        let parameters = OnePoleParameters {
            background: Complex64::new(point[0], point[1]),
            residue: Complex64::new(point[2], point[3]),
            omega_0: point[4],
            gamma: point[5],
        };
        let validation_error = one_pole_objective(parameters, validation);
        if !validation_error.is_finite() {
            return Err(FitError::NonFiniteObjective);
        }
        results.push(fit_start_result(
            &start,
            &point,
            training_error,
            validation_error,
            converged,
            iterations,
        ));
    }
    let selected_index = results
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            left.validation_error
                .total_cmp(&right.validation_error)
                .then_with(|| left.training_error.total_cmp(&right.training_error))
        })
        .map(|(index, _)| index)
        .ok_or(FitError::EmptyStarts)?;
    let point = &results[selected_index].result;
    let parameters = OnePoleParameters {
        background: Complex64::new(point[0], point[1]),
        residue: Complex64::new(point[2], point[3]),
        omega_0: point[4],
        gamma: point[5],
    };
    Ok(OnePoleFitReport {
        parameters,
        training_error: results[selected_index].training_error,
        validation_error: results[selected_index].validation_error,
        starts: results,
    })
}

/// Return the constant complex background fitted only from training samples.
pub fn background_only(samples: &[ComplexSample]) -> Result<Complex64, FitError> {
    validate_samples(samples)?;
    let sum = samples
        .iter()
        .fold(Complex64::new(0.0, 0.0), |accumulator, sample| {
            accumulator + sample.value
        });
    Ok(sum / samples.len() as f64)
}

/// Compute the sum of squared complex residuals for a TCMT parameter set.
pub fn tcmt_error(parameters: FitParameters, samples: &[ComplexSample]) -> f64 {
    tcmt_objective(parameters, samples)
}

/// Compute the sum of squared complex residuals for a one-pole parameter set.
pub fn one_pole_error(parameters: OnePoleParameters, samples: &[ComplexSample]) -> f64 {
    one_pole_objective(parameters, samples)
}

/// Return the largest complex residual over a sample set.
pub fn tcmt_max_error(parameters: FitParameters, samples: &[ComplexSample]) -> f64 {
    samples
        .iter()
        .map(|sample| (tcmt_scattering(parameters, sample.omega) - sample.value).norm())
        .fold(0.0, f64::max)
}

/// Return the largest complex residual for an unconstrained pole.
pub fn one_pole_max_error(parameters: OnePoleParameters, samples: &[ComplexSample]) -> f64 {
    samples
        .iter()
        .map(|sample| (one_pole_value(parameters, sample.omega) - sample.value).norm())
        .fold(0.0, f64::max)
}

/// Compute singular values of the TCMT complex-amplitude Jacobian.
pub fn tcmt_jacobian_singular_values(
    parameters: FitParameters,
    samples: &[ComplexSample],
) -> [f64; 4] {
    let mut gram = [[0.0_f64; 4]; 4];
    for sample in samples {
        let mut derivatives = [Complex64::new(0.0, 0.0); 4];
        let point = [
            parameters.omega_0,
            parameters.gamma,
            parameters.gamma_0,
            parameters.phi,
        ];
        for dimension in 0..4 {
            let step = 1e-7 * point[dimension].abs().max(1.0);
            let mut plus = point;
            let mut minus = point;
            plus[dimension] += step;
            minus[dimension] -= step;
            let plus_parameters = FitParameters {
                omega_0: plus[0],
                gamma: plus[1],
                gamma_0: plus[2],
                phi: plus[3],
            };
            let minus_parameters = FitParameters {
                omega_0: minus[0],
                gamma: minus[1],
                gamma_0: minus[2],
                phi: minus[3],
            };
            derivatives[dimension] = (tcmt_scattering(plus_parameters, sample.omega)
                - tcmt_scattering(minus_parameters, sample.omega))
                / (2.0 * step);
        }
        let rows = [
            derivatives.map(|derivative| derivative.re),
            derivatives.map(|derivative| derivative.im),
        ];
        for row in rows {
            for left in 0..4 {
                for right in 0..4 {
                    gram[left][right] += row[left] * row[right];
                }
            }
        }
    }
    let eigenvalues = symmetric_eigenvalues(gram);
    [
        eigenvalues[0].max(0.0).sqrt(),
        eigenvalues[1].max(0.0).sqrt(),
        eigenvalues[2].max(0.0).sqrt(),
        eigenvalues[3].max(0.0).sqrt(),
    ]
}

fn symmetric_eigenvalues(mut matrix: [[f64; 4]; 4]) -> [f64; 4] {
    for _ in 0..100 {
        let mut pivot = (0, 1);
        let mut maximum = matrix[0][1].abs();
        for left in 0..4 {
            for right in (left + 1)..4 {
                if matrix[left][right].abs() > maximum {
                    maximum = matrix[left][right].abs();
                    pivot = (left, right);
                }
            }
        }
        if maximum <= 1e-14 {
            break;
        }
        let (left, right) = pivot;
        let angle =
            0.5 * (2.0 * matrix[left][right]).atan2(matrix[left][left] - matrix[right][right]);
        let cosine = angle.cos();
        let sine = angle.sin();
        for index in 0..4 {
            let left_value = matrix[left][index];
            let right_value = matrix[right][index];
            matrix[left][index] = cosine * left_value + sine * right_value;
            matrix[right][index] = -sine * left_value + cosine * right_value;
        }
        for index in 0..4 {
            let left_value = matrix[index][left];
            let right_value = matrix[index][right];
            matrix[index][left] = cosine * left_value + sine * right_value;
            matrix[index][right] = -sine * left_value + cosine * right_value;
        }
    }
    let mut values = [matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][3]];
    values.sort_by(|left, right| right.total_cmp(left));
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> Vec<ComplexSample> {
        let parameters = FitParameters {
            omega_0: 0.23,
            gamma: 0.002,
            gamma_0: 0.001,
            phi: -0.4,
        };
        (0..20)
            .map(|index| {
                let omega = 0.22 + index as f64 * 0.0007;
                ComplexSample {
                    omega,
                    value: tcmt_scattering(parameters, omega),
                }
            })
            .collect()
    }

    #[test]
    fn fixed_start_fit_recovers_source_model() {
        let data = samples();
        let report = fit_tcmt(
            &data[..12],
            &data[12..],
            &[[0.23, 0.001, 0.001, -0.4], [0.225, 0.004, 0.004, 0.0]],
            [
                [0.20, 0.25],
                [1e-5, 0.02],
                [0.0, 0.02],
                [-std::f64::consts::PI, std::f64::consts::PI],
            ],
            None,
            800,
        )
        .expect("deterministic fit succeeds");
        assert!(report.validation_error < 1e-12);
        assert!((report.parameters.omega_0 - 0.23).abs() < 1e-5);
    }

    #[test]
    fn jacobian_reports_four_finite_singular_values() {
        let data = samples();
        let values = tcmt_jacobian_singular_values(
            FitParameters {
                omega_0: 0.23,
                gamma: 0.002,
                gamma_0: 0.001,
                phi: -0.4,
            },
            &data,
        );
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values[0] >= values[1]);
    }
}
