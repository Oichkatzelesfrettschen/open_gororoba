//! Shared source-owned operations for tensor-valued photon-graviton paths.
//!
//! The functions in this module keep Euclidean index order explicit. They do
//! not call the projected legacy channel implementations in `vacuum_pol` or
//! `irreducible`.

use gauss_quad::GaussLegendre;
use num_complex::Complex64;
use std::{cell::Cell, num::NonZeroUsize};

use super::{
    quadrature::QuadratureConfig,
    tensor_types::{
        ComplexFourVector, ComplexLorentzMatrix, ComplexRankThreeTensor, KinematicsError,
        LORENTZ_DIMENSION,
    },
    worldline_tensor::{
        CoincidenceLimits, PureMagneticWorldline, WorldlineInputError, pure_magnetic_coincidence,
        pure_magnetic_worldline,
    },
};

const FIELD_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorLoopConfig {
    pub mass: f64,
    pub charge: f64,
    pub kappa: f64,
    pub dimension: usize,
}

impl TensorLoopConfig {
    pub const fn unit_natural() -> Self {
        Self {
            mass: 1.0,
            charge: 1.0,
            kappa: 1.0,
            dimension: LORENTZ_DIMENSION,
        }
    }

    pub fn validate(self) -> Result<Self, TensorEvaluationError> {
        if self.dimension != LORENTZ_DIMENSION {
            return Err(TensorEvaluationError::DimensionMustBeFour);
        }
        if !self.mass.is_finite() || self.mass <= 0.0 {
            return Err(TensorEvaluationError::InvalidMass);
        }
        if !self.charge.is_finite() || self.charge == 0.0 {
            return Err(TensorEvaluationError::InvalidCharge);
        }
        if !self.kappa.is_finite() {
            return Err(TensorEvaluationError::InvalidCoupling);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorEvaluationError {
    DimensionMustBeFour,
    InvalidMass,
    InvalidCharge,
    InvalidCoupling,
    InvalidQuadrature,
    NonMagneticField,
    ComplexMagneticField,
    Worldline(WorldlineInputError),
    NonFiniteResult,
    ExternalOnShellSingularity,
    Kinematics(KinematicsError),
}

impl From<WorldlineInputError> for TensorEvaluationError {
    fn from(error: WorldlineInputError) -> Self {
        Self::Worldline(error)
    }
}

impl From<KinematicsError> for TensorEvaluationError {
    fn from(error: KinematicsError) -> Self {
        Self::Kinematics(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SourceWorldlineNode {
    pub full: PureMagneticWorldline,
    pub coincidence: CoincidenceLimits,
    pub bar_g_b12: ComplexLorentzMatrix,
    pub bar_dot_g_b12: ComplexLorentzMatrix,
    pub bar_dot_g_b21: ComplexLorentzMatrix,
    pub s_b12: ComplexLorentzMatrix,
    pub a_b12: ComplexLorentzMatrix,
    pub dot_s_b12: ComplexLorentzMatrix,
    pub dot_a_b12: ComplexLorentzMatrix,
    pub s_f12: ComplexLorentzMatrix,
    pub a_f12: ComplexLorentzMatrix,
    pub dot_s_f12: ComplexLorentzMatrix,
    pub dot_a_f12: ComplexLorentzMatrix,
}

pub(crate) fn validate_tensor_inputs(
    kinematics: &super::tensor_types::WardKinematics,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<(TensorLoopConfig, f64), TensorEvaluationError> {
    kinematics.validate()?;
    let loop_config = loop_config.validate()?;
    if quadrature.n_u == 0
        || quadrature.n_t == 0
        || !quadrature.t_min.is_finite()
        || !quadrature.t_max.is_finite()
        || quadrature.t_min <= 0.0
        || quadrature.t_max <= quadrature.t_min
    {
        return Err(TensorEvaluationError::InvalidQuadrature);
    }
    let magnetic_field = magnetic_field_strength(&kinematics.field_strength)?;
    Ok((loop_config, magnetic_field))
}

pub(crate) fn magnetic_field_strength(
    field_strength: &ComplexLorentzMatrix,
) -> Result<f64, TensorEvaluationError> {
    let magnetic_entry = field_strength[(0, 1)];
    if magnetic_entry.im.abs() > FIELD_TOLERANCE {
        return Err(TensorEvaluationError::ComplexMagneticField);
    }
    if (field_strength[(1, 0)] + magnetic_entry).norm() > FIELD_TOLERANCE {
        return Err(TensorEvaluationError::NonMagneticField);
    }
    for row in 0..LORENTZ_DIMENSION {
        for column in 0..LORENTZ_DIMENSION {
            let supported_plane = (row == 0 && column == 1) || (row == 1 && column == 0);
            if !supported_plane && field_strength[(row, column)].norm() > FIELD_TOLERANCE {
                return Err(TensorEvaluationError::NonMagneticField);
            }
        }
    }
    Ok(magnetic_entry.re)
}

pub(crate) fn source_worldline_node(
    magnetic_field: f64,
    charge: f64,
    proper_time: f64,
    u: f64,
) -> Result<SourceWorldlineNode, TensorEvaluationError> {
    let z = charge * magnetic_field * proper_time;
    let full = pure_magnetic_worldline(z, u, proper_time)?;
    let coincidence = pure_magnetic_coincidence(z, proper_time)?;
    let bar_g_b12 = full.g_b - coincidence.g_b;
    let bar_dot_g_b12 = full.dot_g_b - coincidence.dot_g_b;
    let dot_g_b21 = -full.dot_g_b.transpose();
    let bar_dot_g_b21 = dot_g_b21 - coincidence.dot_g_b;
    Ok(SourceWorldlineNode {
        full,
        coincidence,
        bar_g_b12,
        bar_dot_g_b12,
        bar_dot_g_b21,
        s_b12: even(&full.g_b),
        a_b12: odd(&full.g_b),
        dot_s_b12: even(&full.dot_g_b),
        dot_a_b12: odd(&full.dot_g_b),
        s_f12: even(&full.g_f),
        a_f12: odd(&full.g_f),
        dot_s_f12: even(&full.dot_g_f),
        dot_a_f12: odd(&full.dot_g_f),
    })
}

pub(crate) fn even(matrix: &ComplexLorentzMatrix) -> ComplexLorentzMatrix {
    (matrix + matrix.transpose()) * Complex64::new(0.5, 0.0)
}

pub(crate) fn odd(matrix: &ComplexLorentzMatrix) -> ComplexLorentzMatrix {
    (matrix - matrix.transpose()) * Complex64::new(0.5, 0.0)
}

pub(crate) fn left_contract(
    vector: &ComplexFourVector,
    matrix: &ComplexLorentzMatrix,
) -> ComplexFourVector {
    let mut result = ComplexFourVector::zeros();
    for row in 0..LORENTZ_DIMENSION {
        for column in 0..LORENTZ_DIMENSION {
            result[column] += vector[row] * matrix[(row, column)];
        }
    }
    result
}

pub(crate) fn right_contract(
    matrix: &ComplexLorentzMatrix,
    vector: &ComplexFourVector,
) -> ComplexFourVector {
    let mut result = ComplexFourVector::zeros();
    for row in 0..LORENTZ_DIMENSION {
        for column in 0..LORENTZ_DIMENSION {
            result[row] += matrix[(row, column)] * vector[column];
        }
    }
    result
}

pub(crate) fn bilinear(
    left: &ComplexFourVector,
    matrix: &ComplexLorentzMatrix,
    right: &ComplexFourVector,
) -> Complex64 {
    left_contract(left, matrix)
        .iter()
        .zip(right.iter())
        .map(|(left_component, right_component)| *left_component * *right_component)
        .sum()
}

pub(crate) fn outer(left: &ComplexFourVector, right: &ComplexFourVector) -> ComplexLorentzMatrix {
    ComplexLorentzMatrix::from_fn(|row, column| left[row] * right[column])
}

pub(crate) fn matrix_is_finite(matrix: &ComplexLorentzMatrix) -> bool {
    matrix
        .iter()
        .all(|component| component.re.is_finite() && component.im.is_finite())
}

pub(crate) fn scalar_determinant(loop_type: super::types::LoopType, z: f64) -> f64 {
    match loop_type {
        super::types::LoopType::Scalar => super::worldline_tensor::scalar_determinant_factor(z),
        super::types::LoopType::Spinor => super::worldline_tensor::spinor_determinant_factor(z),
    }
}

pub(crate) fn double_integrate_matrix<F>(
    function: F,
    mass: f64,
    power: f64,
    quadrature: &QuadratureConfig,
) -> Result<ComplexLorentzMatrix, TensorEvaluationError>
where
    F: Fn(f64, f64) -> Result<ComplexLorentzMatrix, TensorEvaluationError>,
{
    if quadrature.n_u == 0 || quadrature.n_t == 0 {
        return Err(TensorEvaluationError::InvalidQuadrature);
    }
    let u_quad = GaussLegendre::new(
        NonZeroUsize::new(quadrature.n_u).ok_or(TensorEvaluationError::InvalidQuadrature)?,
    );
    let u_nodes = u_quad.as_node_weight_pairs().to_vec();
    let error = Cell::new(None);
    let result = integrate_matrix(
        |proper_time| {
            let mut u_sum = ComplexLorentzMatrix::zeros();
            for &(node, weight) in &u_nodes {
                let u = 0.5 + 0.5 * node;
                match function(proper_time, u) {
                    Ok(value) => u_sum += value * Complex64::new(0.5 * weight, 0.0),
                    Err(value) => error.set(Some(value)),
                }
            }
            u_sum
                * Complex64::new(
                    proper_time.powf(-power) * (-mass * mass * proper_time).exp(),
                    0.0,
                )
        },
        quadrature.t_min,
        quadrature.t_max,
        quadrature.n_t,
    );
    if let Some(value) = error.get() {
        return Err(value);
    }
    if !matrix_is_finite(&result) {
        return Err(TensorEvaluationError::NonFiniteResult);
    }
    Ok(result)
}

pub(crate) fn double_integrate_rank_three<F>(
    function: F,
    mass: f64,
    power: f64,
    quadrature: &QuadratureConfig,
) -> Result<ComplexRankThreeTensor, TensorEvaluationError>
where
    F: Fn(f64, f64) -> Result<ComplexRankThreeTensor, TensorEvaluationError>,
{
    double_integrate_rank_three_with_contact(
        function,
        |_| Ok(ComplexRankThreeTensor::zero()),
        mass,
        power,
        quadrature,
    )
}

/// Integrate regular insertion nodes and an analytically integrated contact.
/// The contact is added once per proper-time node, outside open-interval
/// Gauss-Legendre sampling, with its rescaled delta Jacobian already applied.
pub(crate) fn double_integrate_rank_three_with_contact<F, C>(
    function: F,
    contact: C,
    mass: f64,
    power: f64,
    quadrature: &QuadratureConfig,
) -> Result<ComplexRankThreeTensor, TensorEvaluationError>
where
    F: Fn(f64, f64) -> Result<ComplexRankThreeTensor, TensorEvaluationError>,
    C: Fn(f64) -> Result<ComplexRankThreeTensor, TensorEvaluationError>,
{
    if quadrature.n_u == 0 || quadrature.n_t == 0 {
        return Err(TensorEvaluationError::InvalidQuadrature);
    }
    let u_quad = GaussLegendre::new(
        NonZeroUsize::new(quadrature.n_u).ok_or(TensorEvaluationError::InvalidQuadrature)?,
    );
    let u_nodes = u_quad.as_node_weight_pairs().to_vec();
    let error = Cell::new(None);
    let result = integrate_rank_three(
        |proper_time| {
            let mut u_sum = ComplexRankThreeTensor::zero();
            for &(node, weight) in &u_nodes {
                let u = 0.5 + 0.5 * node;
                match function(proper_time, u) {
                    Ok(value) => {
                        u_sum = rank_three_add(
                            &u_sum,
                            &rank_three_scale(&value, Complex64::new(0.5 * weight, 0.0)),
                        )
                    }
                    Err(value) => error.set(Some(value)),
                }
            }
            match contact(proper_time) {
                Ok(value) => u_sum = rank_three_add(&u_sum, &value),
                Err(value) => error.set(Some(value)),
            }
            rank_three_scale(
                &u_sum,
                Complex64::new(
                    proper_time.powf(-power) * (-mass * mass * proper_time).exp(),
                    0.0,
                ),
            )
        },
        quadrature.t_min,
        quadrature.t_max,
        quadrature.n_t,
    );
    if let Some(value) = error.get() {
        return Err(value);
    }
    if !result
        .components()
        .iter()
        .all(|component| component.re.is_finite() && component.im.is_finite())
    {
        return Err(TensorEvaluationError::NonFiniteResult);
    }
    Ok(result)
}

fn integrate_rank_three<F>(
    function: F,
    lower: f64,
    upper: f64,
    degree: usize,
) -> ComplexRankThreeTensor
where
    F: Fn(f64) -> ComplexRankThreeTensor,
{
    let quad = GaussLegendre::new(NonZeroUsize::new(degree).expect("validated quadrature degree"));
    let half_length = 0.5 * (upper - lower);
    let midpoint = 0.5 * (upper + lower);
    let mut result = ComplexRankThreeTensor::zero();
    for &(node, weight) in quad.as_node_weight_pairs() {
        let proper_time = midpoint + half_length * node;
        result = rank_three_add(
            &result,
            &rank_three_scale(
                &function(proper_time),
                Complex64::new(half_length * weight, 0.0),
            ),
        );
    }
    result
}

pub(crate) fn rank_three_add(
    left: &ComplexRankThreeTensor,
    right: &ComplexRankThreeTensor,
) -> ComplexRankThreeTensor {
    ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        left.get(mu, nu, alpha) + right.get(mu, nu, alpha)
    })
}

pub(crate) fn rank_three_scale(
    tensor: &ComplexRankThreeTensor,
    factor: Complex64,
) -> ComplexRankThreeTensor {
    ComplexRankThreeTensor::from_fn(|mu, nu, alpha| tensor.get(mu, nu, alpha) * factor)
}

fn integrate_matrix<F>(function: F, lower: f64, upper: f64, degree: usize) -> ComplexLorentzMatrix
where
    F: Fn(f64) -> ComplexLorentzMatrix,
{
    let quad = GaussLegendre::new(NonZeroUsize::new(degree).expect("validated quadrature degree"));
    let half_length = 0.5 * (upper - lower);
    let midpoint = 0.5 * (upper + lower);
    let mut result = ComplexLorentzMatrix::zeros();
    for &(node, weight) in quad.as_node_weight_pairs() {
        let proper_time = midpoint + half_length * node;
        result += function(proper_time) * Complex64::new(half_length * weight, 0.0);
    }
    result
}
