//! Complex-frequency outgoing-mode determinants for the Ruan-Fan MDM cylinder.
//!
//! The real-frequency scattering path and the complex-frequency pole path are
//! separate on purpose. The pole path assembles the six boundary equations
//! for regular J_l fields in the core, J_l/Y_l fields in finite layers, and an
//! outgoing H_l^(1) field in air. It does not use the heuristic peak extractor.
//!
//! The source convention is exp(-i*omega*t), H_z TM polarization, and the
//! continuous interface state [H_z, (1/epsilon)*dH_z/drho].

use crate::{
    bessel::{bessel_j, bessel_j_prime, bessel_y, bessel_y_prime, hankel_1, hankel_1_prime},
    fano_tcmt::FanoDrudeParams,
    mie_cylinder::CylindricalPolarization,
};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

const MATRIX_SIZE: usize = 6;
const ROOT_DEDUPE_TOLERANCE: f64 = 1e-8;

/// A complex frequency with explicit finite-value validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexFrequency {
    /// Dimensionless or dimensional complex angular frequency.
    pub value: Complex64,
}

impl ComplexFrequency {
    /// Construct a finite complex frequency.
    pub fn new(value: Complex64) -> Result<Self, PoleError> {
        if !finite_complex(value) {
            return Err(PoleError::NonFinite {
                field: "complex frequency",
            });
        }
        Ok(Self { value })
    }
}

/// Source geometry used by the complex-frequency determinant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoleGeometry {
    /// Interface radii in increasing order.
    pub radii: [f64; 3],
    /// Dielectric relative permittivity in the middle layer.
    pub dielectric_epsilon: f64,
    /// Plasma frequency in the same units as the complex frequency.
    pub omega_p: f64,
    /// Longitudinal polarization used by the boundary state.
    pub polarization: CylindricalPolarization,
}

impl PoleGeometry {
    /// Build a source-normalized MDM geometry from radii in lambda_p units.
    pub fn source_mdm(
        omega_p: f64,
        inner_radius_over_lambda_p: f64,
        dielectric_radius_over_lambda_p: f64,
        outer_radius_over_lambda_p: f64,
    ) -> Result<Self, PoleError> {
        if !omega_p.is_finite() || omega_p <= 0.0 {
            return Err(PoleError::NonFinite { field: "omega_p" });
        }
        let lambda_p = 2.0 * PI / omega_p;
        let geometry = Self {
            radii: [
                inner_radius_over_lambda_p * lambda_p,
                dielectric_radius_over_lambda_p * lambda_p,
                outer_radius_over_lambda_p * lambda_p,
            ],
            dielectric_epsilon: 12.96,
            omega_p,
            polarization: CylindricalPolarization::HzTm,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    /// Validate the source geometry before determinant evaluation.
    pub fn validate(&self) -> Result<(), PoleError> {
        if !self.omega_p.is_finite() || self.omega_p <= 0.0 {
            return Err(PoleError::NonFinite { field: "omega_p" });
        }
        if !self.dielectric_epsilon.is_finite() || self.dielectric_epsilon <= 0.0 {
            return Err(PoleError::NonFinite {
                field: "dielectric_epsilon",
            });
        }
        if self.polarization != CylindricalPolarization::HzTm {
            return Err(PoleError::UnsupportedPolarization);
        }
        let mut previous = 0.0;
        for radius in self.radii {
            if !radius.is_finite() || radius <= previous {
                return Err(PoleError::InvalidRadii);
            }
            previous = radius;
        }
        Ok(())
    }
}

/// Complex rectangular search contour in the frequency plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootRectangle {
    /// Minimum real frequency.
    pub re_min: f64,
    /// Maximum real frequency.
    pub re_max: f64,
    /// Minimum imaginary frequency.
    pub im_min: f64,
    /// Maximum imaginary frequency.
    pub im_max: f64,
}

impl RootRectangle {
    /// Validate the rectangle and its ordering.
    pub fn validate(&self) -> Result<(), PoleError> {
        if [self.re_min, self.re_max, self.im_min, self.im_max]
            .iter()
            .any(|value| !value.is_finite())
            || self.re_min >= self.re_max
            || self.im_min >= self.im_max
        {
            return Err(PoleError::InvalidRectangle);
        }
        Ok(())
    }

    fn contains(&self, value: Complex64) -> bool {
        value.re >= self.re_min
            && value.re <= self.re_max
            && value.im >= self.im_min
            && value.im <= self.im_max
    }
}

/// Stable argument-principle root count over a declared rectangle.
#[derive(Debug, Clone, PartialEq)]
pub struct StableRootCount {
    /// Edge sample refinements used for the count.
    pub refinements: Vec<usize>,
    /// Winding estimates before integer rounding.
    pub winding_numbers: Vec<f64>,
    /// Integer root counts at each refinement.
    pub counts: Vec<usize>,
    /// Minimum determinant modulus observed on each contour.
    pub minimum_moduli: Vec<f64>,
}

/// One refined complex pole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexPole {
    /// Initial Newton point.
    pub start: Complex64,
    /// Refined complex frequency.
    pub omega: Complex64,
    /// Determinant at the refined frequency.
    pub determinant: Complex64,
    /// Symmetric secant derivative at the refined frequency.
    pub determinant_derivative: Complex64,
    /// Number of Newton iterations used.
    pub iterations: usize,
    /// Root residual magnitude.
    pub residual: f64,
    /// Decay rate -Im(omega).
    pub decay_rate: f64,
}

/// One deterministic root-refinement attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct RootAttempt {
    /// Initial point.
    pub start: Complex64,
    /// Refined result when convergence succeeds.
    pub root: Option<Complex64>,
    /// Iterations completed.
    pub iterations: usize,
    /// Final residual magnitude.
    pub residual: f64,
    /// Error text for a failed attempt.
    pub error: Option<String>,
}

/// All roots found after the root count gate and deterministic seed sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct RootSearch {
    /// Argument-principle count record.
    pub count: StableRootCount,
    /// Every retained unique root.
    pub roots: Vec<ComplexPole>,
    /// Every seed attempt, including failures.
    pub attempts: Vec<RootAttempt>,
}

/// Errors from complex pole construction and refinement.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum PoleError {
    #[error("{field} must be finite")]
    NonFinite { field: &'static str },
    #[error("source geometry radii must be positive and strictly increasing")]
    InvalidRadii,
    #[error("the complex pole path supports H_z TM polarization only")]
    UnsupportedPolarization,
    #[error("the root rectangle is invalid")]
    InvalidRectangle,
    #[error("the determinant is non-finite")]
    NonFiniteDeterminant,
    #[error("the contour passes through a zero or non-finite determinant")]
    SingularContour,
    #[error("root count is not stable under contour refinement")]
    UnstableRootCount,
    #[error("the secant derivative is singular")]
    SingularSecant,
    #[error("the root refinement did not converge")]
    RootDidNotConverge,
    #[error("the complex Drude denominator is singular")]
    SingularDrudeDenominator,
    #[error("the boundary matrix is non-finite")]
    NonFiniteBoundaryMatrix,
}

type Matrix6 = [[Complex64; MATRIX_SIZE]; MATRIX_SIZE];

#[derive(Debug, Clone, Copy)]
enum BasisFunction {
    BesselJ,
    BesselY,
    HankelOutgoing,
    HankelIncoming,
}

fn finite_complex(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn passive_sqrt(value: Complex64) -> Result<Complex64, PoleError> {
    if !finite_complex(value) {
        return Err(PoleError::NonFinite {
            field: "permittivity",
        });
    }
    let mut root = value.sqrt();
    if root.im < 0.0 || (root.im == 0.0 && root.re < 0.0) {
        root = -root;
    }
    if !finite_complex(root) {
        return Err(PoleError::NonFinite {
            field: "square root",
        });
    }
    Ok(root)
}

/// Evaluate the Drude permittivity at a complex frequency.
pub fn complex_drude_epsilon(
    frequency: ComplexFrequency,
    params: &FanoDrudeParams,
) -> Result<Complex64, PoleError> {
    params.validate().map_err(|_| PoleError::NonFinite {
        field: "Drude parameters",
    })?;
    let omega = frequency.value;
    let denominator = omega * omega + Complex64::i() * params.gamma_d * omega;
    if denominator.norm_sqr() == 0.0 {
        return Err(PoleError::SingularDrudeDenominator);
    }
    let epsilon = Complex64::new(1.0, 0.0)
        - Complex64::new(params.omega_p * params.omega_p, 0.0) / denominator;
    if !finite_complex(epsilon) {
        return Err(PoleError::NonFinite {
            field: "Drude epsilon",
        });
    }
    Ok(epsilon)
}

fn basis_value(basis: BasisFunction, l: i32, argument: Complex64) -> (Complex64, Complex64) {
    match basis {
        BasisFunction::BesselJ => (bessel_j(l, argument), bessel_j_prime(l, argument)),
        BasisFunction::BesselY => (bessel_y(l, argument), bessel_y_prime(l, argument)),
        BasisFunction::HankelOutgoing => (hankel_1(l, argument), hankel_1_prime(l, argument)),
        BasisFunction::HankelIncoming => {
            let value = bessel_j(l, argument) - Complex64::i() * bessel_y(l, argument);
            let derivative =
                bessel_j_prime(l, argument) - Complex64::i() * bessel_y_prime(l, argument);
            (value, derivative)
        }
    }
}

fn state_value(
    basis: BasisFunction,
    l: i32,
    frequency: Complex64,
    epsilon: Complex64,
    radius: f64,
) -> Result<[Complex64; 2], PoleError> {
    if epsilon.norm_sqr() == 0.0 {
        return Err(PoleError::NonFinite {
            field: "zero permittivity",
        });
    }
    let wave_number = frequency * passive_sqrt(epsilon)?;
    let argument = wave_number * radius;
    let (field, derivative) = basis_value(basis, l, argument);
    let flux = wave_number * derivative / epsilon;
    if !finite_complex(field) || !finite_complex(flux) {
        return Err(PoleError::NonFiniteBoundaryMatrix);
    }
    Ok([field, flux])
}

fn zero_matrix() -> Matrix6 {
    [[Complex64::new(0.0, 0.0); MATRIX_SIZE]; MATRIX_SIZE]
}

fn determinant(mut matrix: Matrix6) -> Result<Complex64, PoleError> {
    let mut result = Complex64::new(1.0, 0.0);
    for column in 0..MATRIX_SIZE {
        let pivot_row = (column..MATRIX_SIZE)
            .max_by(|left, right| {
                matrix[*left][column]
                    .norm_sqr()
                    .total_cmp(&matrix[*right][column].norm_sqr())
            })
            .ok_or(PoleError::NonFiniteBoundaryMatrix)?;
        let pivot = matrix[pivot_row][column];
        if pivot.norm_sqr() == 0.0 {
            return Ok(Complex64::new(0.0, 0.0));
        }
        if pivot_row != column {
            matrix.swap(pivot_row, column);
            result = -result;
        }
        result *= pivot;
        let pivot_values = matrix[column];
        for row_values in matrix.iter_mut().skip(column + 1) {
            let factor = row_values[column] / pivot;
            for (index, value) in row_values.iter_mut().enumerate().skip(column + 1) {
                *value -= factor * pivot_values[index];
            }
        }
    }
    if !finite_complex(result) {
        return Err(PoleError::NonFiniteDeterminant);
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct BoundaryContext {
    l: i32,
    frequency: Complex64,
    epsilons: [Complex64; 4],
    radii: [f64; 3],
}

fn fill_interface_row(
    matrix: &mut Matrix6,
    row: usize,
    region: usize,
    sign: f64,
    context: BoundaryContext,
    radius_index: usize,
) -> Result<(), PoleError> {
    let radius = context.radii[radius_index];
    let basis = match region {
        0 => [Some(BasisFunction::BesselJ), None],
        1 | 2 => [Some(BasisFunction::BesselJ), Some(BasisFunction::BesselY)],
        3 => [Some(BasisFunction::HankelOutgoing), None],
        _ => return Err(PoleError::NonFiniteBoundaryMatrix),
    };
    let columns = match region {
        0 => [0, 0],
        1 => [1, 2],
        2 => [3, 4],
        3 => [5, 5],
        _ => return Err(PoleError::NonFiniteBoundaryMatrix),
    };
    for (offset, function) in basis.into_iter().enumerate() {
        if let Some(function) = function {
            let state = state_value(
                function,
                context.l,
                context.frequency,
                context.epsilons[region],
                radius,
            )?;
            matrix[row][columns[offset]] = Complex64::new(sign, 0.0) * state[0];
            matrix[row + 1][columns[offset]] = Complex64::new(sign, 0.0) * state[1];
        }
    }
    Ok(())
}

fn boundary_matrix(
    geometry: &PoleGeometry,
    l: i32,
    frequency: Complex64,
    gamma_d: f64,
) -> Result<Matrix6, PoleError> {
    geometry.validate()?;
    if !gamma_d.is_finite() || gamma_d < 0.0 {
        return Err(PoleError::NonFinite { field: "gamma_d" });
    }
    let drude = FanoDrudeParams {
        omega_p: geometry.omega_p,
        gamma_d,
    };
    let metal = complex_drude_epsilon(ComplexFrequency::new(frequency)?, &drude)?;
    let epsilons = [
        metal,
        Complex64::new(geometry.dielectric_epsilon, 0.0),
        metal,
        Complex64::new(1.0, 0.0),
    ];
    let context = BoundaryContext {
        l,
        frequency,
        epsilons,
        radii: geometry.radii,
    };
    let mut matrix = zero_matrix();
    for (interface, (inner, outer)) in [(0usize, (0usize, 1usize)), (1, (1, 2)), (2, (2, 3))] {
        let row = 2 * interface;
        fill_interface_row(&mut matrix, row, inner, 1.0, context, interface)?;
        fill_interface_row(&mut matrix, row, outer, -1.0, context, interface)?;
    }
    if matrix.iter().flatten().any(|value| !finite_complex(*value)) {
        return Err(PoleError::NonFiniteBoundaryMatrix);
    }
    Ok(matrix)
}

/// Evaluate the six-equation outgoing-mode determinant.
pub fn outgoing_determinant(
    geometry: &PoleGeometry,
    l: i32,
    frequency: ComplexFrequency,
    gamma_d: f64,
) -> Result<Complex64, PoleError> {
    determinant(boundary_matrix(geometry, l, frequency.value, gamma_d)?)
}

fn contour_points(rectangle: RootRectangle, edge_samples: usize) -> Vec<Complex64> {
    let n = edge_samples.max(2);
    let mut points = Vec::with_capacity(4 * n);
    for index in 0..n {
        let t = index as f64 / n as f64;
        points.push(Complex64::new(
            rectangle.re_min + (rectangle.re_max - rectangle.re_min) * t,
            rectangle.im_min,
        ));
    }
    for index in 1..=n {
        let t = index as f64 / n as f64;
        points.push(Complex64::new(
            rectangle.re_max,
            rectangle.im_min + (rectangle.im_max - rectangle.im_min) * t,
        ));
    }
    for index in 1..=n {
        let t = index as f64 / n as f64;
        points.push(Complex64::new(
            rectangle.re_max - (rectangle.re_max - rectangle.re_min) * t,
            rectangle.im_max,
        ));
    }
    for index in 1..n {
        let t = index as f64 / n as f64;
        points.push(Complex64::new(
            rectangle.re_min,
            rectangle.im_max - (rectangle.im_max - rectangle.im_min) * t,
        ));
    }
    points
}

fn contour_count(
    geometry: &PoleGeometry,
    l: i32,
    gamma_d: f64,
    rectangle: RootRectangle,
    edge_samples: usize,
) -> Result<(f64, usize, f64), PoleError> {
    rectangle.validate()?;
    let points = contour_points(rectangle, edge_samples);
    let mut values = Vec::with_capacity(points.len());
    let mut minimum_modulus = f64::INFINITY;
    for point in points {
        let value = outgoing_determinant(geometry, l, ComplexFrequency::new(point)?, gamma_d)?;
        if !finite_complex(value) || value.norm_sqr() == 0.0 {
            return Err(PoleError::SingularContour);
        }
        minimum_modulus = minimum_modulus.min(value.norm());
        values.push(value);
    }
    let mut winding = 0.0;
    for index in 0..values.len() {
        let ratio = values[(index + 1) % values.len()] / values[index];
        if !finite_complex(ratio) {
            return Err(PoleError::SingularContour);
        }
        winding += ratio.arg();
    }
    let rounded = winding / (2.0 * PI);
    let count = rounded.round();
    if count < 0.0 || (rounded - count).abs() > 0.25 {
        return Err(PoleError::UnstableRootCount);
    }
    Ok((rounded, count as usize, minimum_modulus))
}

/// Count roots at several contour refinements and require agreement.
pub fn stable_root_count(
    geometry: &PoleGeometry,
    l: i32,
    gamma_d: f64,
    rectangle: RootRectangle,
    refinements: &[usize],
) -> Result<StableRootCount, PoleError> {
    if refinements.is_empty() {
        return Err(PoleError::UnstableRootCount);
    }
    let mut winding_numbers = Vec::with_capacity(refinements.len());
    let mut counts = Vec::with_capacity(refinements.len());
    let mut minimum_moduli = Vec::with_capacity(refinements.len());
    for &refinement in refinements {
        let (winding, count, minimum_modulus) =
            contour_count(geometry, l, gamma_d, rectangle, refinement)?;
        winding_numbers.push(winding);
        counts.push(count);
        minimum_moduli.push(minimum_modulus);
    }
    if counts.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(PoleError::UnstableRootCount);
    }
    Ok(StableRootCount {
        refinements: refinements.to_vec(),
        winding_numbers,
        counts,
        minimum_moduli,
    })
}

/// Build a deterministic rectangular seed grid inside a root contour.
pub fn root_seed_grid(
    rectangle: RootRectangle,
    real_points: usize,
    imaginary_points: usize,
) -> Vec<Complex64> {
    let real_points = real_points.max(1);
    let imaginary_points = imaginary_points.max(1);
    let mut seeds = Vec::with_capacity(real_points * imaginary_points);
    for re_index in 0..real_points {
        let re_fraction = (re_index as f64 + 0.5) / real_points as f64;
        let re = rectangle.re_min + (rectangle.re_max - rectangle.re_min) * re_fraction;
        for im_index in 0..imaginary_points {
            let im_fraction = (im_index as f64 + 0.5) / imaginary_points as f64;
            let im = rectangle.im_min + (rectangle.im_max - rectangle.im_min) * im_fraction;
            seeds.push(Complex64::new(re, im));
        }
    }
    seeds
}

/// Refine one root with a symmetric complex secant derivative.
pub fn refine_root(
    geometry: &PoleGeometry,
    l: i32,
    gamma_d: f64,
    start: Complex64,
    max_iterations: usize,
    root_tolerance: f64,
    residual_tolerance: f64,
) -> Result<ComplexPole, PoleError> {
    let mut omega = ComplexFrequency::new(start)?.value;
    for iteration in 0..max_iterations {
        let value = outgoing_determinant(geometry, l, ComplexFrequency::new(omega)?, gamma_d)?;
        let residual = value.norm();
        let scale = 1e-7 * omega.norm().max(1.0);
        let step = Complex64::new(scale, 0.0);
        let forward =
            outgoing_determinant(geometry, l, ComplexFrequency::new(omega + step)?, gamma_d)?;
        let backward =
            outgoing_determinant(geometry, l, ComplexFrequency::new(omega - step)?, gamma_d)?;
        let derivative = (forward - backward) / (2.0 * scale);
        if derivative.norm_sqr() == 0.0 {
            return Err(PoleError::SingularSecant);
        }
        let update = value / derivative;
        omega -= update;
        if !finite_complex(omega) {
            return Err(PoleError::RootDidNotConverge);
        }
        if update.norm() <= root_tolerance * omega.norm().max(1.0) && residual <= residual_tolerance
        {
            let final_value =
                outgoing_determinant(geometry, l, ComplexFrequency::new(omega)?, gamma_d)?;
            let final_residual = final_value.norm();
            if final_residual <= residual_tolerance {
                return Ok(ComplexPole {
                    start,
                    omega,
                    determinant: final_value,
                    determinant_derivative: derivative,
                    iterations: iteration + 1,
                    residual: final_residual,
                    decay_rate: -omega.im,
                });
            }
        }
    }
    Err(PoleError::RootDidNotConverge)
}

/// Count roots, refine every deterministic seed, and retain unique roots.
pub fn search_roots(
    geometry: &PoleGeometry,
    l: i32,
    gamma_d: f64,
    rectangle: RootRectangle,
    refinements: &[usize],
    seeds: &[Complex64],
) -> Result<RootSearch, PoleError> {
    let count = stable_root_count(geometry, l, gamma_d, rectangle, refinements)?;
    let mut roots = Vec::new();
    let mut attempts = Vec::with_capacity(seeds.len());
    for &start in seeds {
        match refine_root(geometry, l, gamma_d, start, 80, 1e-12, 1e-10) {
            Ok(root) if rectangle.contains(root.omega) && root.decay_rate > 0.0 => {
                if roots.iter().all(|existing: &ComplexPole| {
                    (existing.omega - root.omega).norm() > ROOT_DEDUPE_TOLERANCE
                }) {
                    roots.push(root);
                }
                attempts.push(RootAttempt {
                    start,
                    root: Some(root.omega),
                    iterations: root.iterations,
                    residual: root.residual,
                    error: None,
                });
            }
            Ok(root) => {
                attempts.push(RootAttempt {
                    start,
                    root: Some(root.omega),
                    iterations: root.iterations,
                    residual: root.residual,
                    error: Some(
                        "converged root is outside the declared decaying rectangle".to_owned(),
                    ),
                });
            }
            Err(error) => attempts.push(RootAttempt {
                start,
                root: None,
                iterations: 80,
                residual: f64::INFINITY,
                error: Some(error.to_string()),
            }),
        }
    }
    roots.sort_by(|left, right| {
        left.omega
            .re
            .total_cmp(&right.omega.re)
            .then_with(|| left.omega.im.total_cmp(&right.omega.im))
    });
    Ok(RootSearch {
        count,
        roots,
        attempts,
    })
}

/// Compute the uniform-metal-cylinder reflection amplitude independently of
/// the layered real-frequency transfer path.
pub fn uniform_metal_reflection(
    geometry: &PoleGeometry,
    l: i32,
    frequency: f64,
    gamma_d: f64,
) -> Result<Complex64, PoleError> {
    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(PoleError::NonFinite {
            field: "real frequency",
        });
    }
    geometry.validate()?;
    let drude = FanoDrudeParams {
        omega_p: geometry.omega_p,
        gamma_d,
    };
    let metal = complex_drude_epsilon(
        ComplexFrequency::new(Complex64::new(frequency, 0.0))?,
        &drude,
    )?;
    let radius = geometry.radii[2];
    let inner = state_value(
        BasisFunction::BesselJ,
        l,
        Complex64::new(frequency, 0.0),
        metal,
        radius,
    )?;
    let outgoing = state_value(
        BasisFunction::HankelOutgoing,
        l,
        Complex64::new(frequency, 0.0),
        Complex64::new(1.0, 0.0),
        radius,
    )?;
    let incoming = state_value(
        BasisFunction::HankelIncoming,
        l,
        Complex64::new(frequency, 0.0),
        Complex64::new(1.0, 0.0),
        radius,
    )?;
    let matrix = [[inner[0], -outgoing[0]], [inner[1], -outgoing[1]]];
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if determinant.norm_sqr() == 0.0 {
        return Err(PoleError::SingularSecant);
    }
    let output = (matrix[0][0] * incoming[1] - incoming[0] * matrix[1][0]) / determinant;
    if !finite_complex(output) {
        return Err(PoleError::NonFiniteDeterminant);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fig4_geometry() -> PoleGeometry {
        PoleGeometry::source_mdm(1.0, 0.285, 1.0, 1.5).expect("valid Figure 4 geometry")
    }

    #[test]
    fn source_determinant_reproduces_fig4_lossless_pole() {
        let geometry = fig4_geometry();
        let root = refine_root(
            &geometry,
            0,
            0.0,
            Complex64::new(0.1552, -0.000019166),
            80,
            1e-12,
            1e-10,
        )
        .expect("Figure 4 root converges");
        assert!((root.omega.re - 0.155215136).abs() < 5e-8);
        assert!((root.decay_rate - 0.000019166366).abs() < 5e-10);
        assert!(root.residual < 1e-10);
    }

    #[test]
    fn contour_count_is_stable_for_fig4() {
        let geometry = fig4_geometry();
        let rectangle = RootRectangle {
            re_min: 0.154,
            re_max: 0.1565,
            im_min: -0.0002,
            im_max: 0.00005,
        };
        let count = stable_root_count(&geometry, 0, 0.0, rectangle, &[32, 64, 128])
            .expect("stable Figure 4 root count");
        assert_eq!(count.counts, vec![1, 1, 1]);
    }

    #[test]
    fn uniform_metal_background_has_unit_modulus_when_lossless() {
        let geometry = fig4_geometry();
        let reflection = uniform_metal_reflection(&geometry, 0, 0.155215136, 0.0)
            .expect("uniform metal reflection");
        assert!((reflection.norm() - 1.0).abs() < 1e-10);
        assert!((reflection.arg() / PI + 0.488221).abs() < 2e-5);
    }
}
