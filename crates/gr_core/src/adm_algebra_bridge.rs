// Tensor loops use multi-index access across several arrays; iterators obscure the math.
#![allow(clippy::needless_range_loop)]

//! ADM-Algebraic bridge: Cayley-Dickson coupling to 3+1 decomposition.
//!
//! Non-paper-canonical module connecting the ADM 3+1 infrastructure
//! to Cayley-Dickson algebraic structure. Tests novel physical hypotheses:
//!
//! - Quaternion frame fields parameterize hypersurface orientations
//! - Octonion non-associativity couples to trace-free extrinsic curvature
//! - Sedenion imbalance sources effective stress-energy in constraints
//! - Cayley-Dickson dimension maps to nacelle segmentation symmetry
//!
//! The key principle: zero coupling constants recover standard GR exactly.
//! The algebraic corrections are perturbative with dimensionless coupling
//! alpha, and the vacuum imbalance attractor (3/8) gives zero correction.
//!
//! # References
//! - This module (open_gororoba non-canonical exploration)
//! - Arnowitt, Deser, Misner (1962): ADM formalism
//! - de Marrais (2000): Sedenion zero-divisor structure

use crate::adm::{AdmDecomposition, ExtrinsicCurvatureData};
use crate::spatial::{
    SDIM, SpatialMetric, SpatialSymmetricTensor, trace_symmetric,
};

/// Combined ADM + algebraic state at a spatial point.
#[derive(Debug, Clone)]
pub struct AlgebraicAdmState {
    /// Standard GR ADM decomposition.
    pub adm: AdmDecomposition,
    /// Extrinsic curvature data.
    pub extrinsic: ExtrinsicCurvatureData,
    /// Quaternion encoding the spatial frame orientation.
    pub quaternion_frame: [f64; 4],
    /// Local octonion field value (8 components).
    pub octonion_field: [f64; 8],
    /// Sedenion imbalance density at this point (in [0, 0.5]).
    pub sedenion_imbalance: f64,
    /// Associator norm of the local algebraic field.
    pub associator_norm: f64,
}

/// Algebraic coupling model selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlgebraicCouplingModel {
    /// Quaternion frame orientation modulates the shift vector.
    QuaternionFrame,
    /// Octonion imbalance couples to K_ij trace-free part.
    OctonionBivector,
    /// Sedenion F(x) -> effective stress-energy T_{ab}.
    SedenionViscosity,
    /// All three coupled simultaneously.
    FullTower,
}

/// The vacuum imbalance attractor value (3/8).
pub const IMBALANCE_ATTRACTOR: f64 = 0.375;

// ============================================================================
// Frame extraction
// ============================================================================

/// Extract a quaternion encoding the principal axes of gamma_{ij}.
///
/// Eigendecompose the 3x3 spatial metric to find the rotation from
/// the identity metric (flat space). For flat space: q = (1,0,0,0).
///
/// Uses the fact that gamma_{ij} = R^T D R where D is diagonal
/// (eigenvalues) and R is the rotation matrix of eigenvectors.
/// The quaternion encodes R.
pub fn quaternion_frame_from_spatial_metric(gamma: &SpatialMetric) -> [f64; 4] {
    let mat = nalgebra::Matrix3::new(
        gamma[0][0], gamma[0][1], gamma[0][2],
        gamma[1][0], gamma[1][1], gamma[1][2],
        gamma[2][0], gamma[2][1], gamma[2][2],
    );

    // Eigendecompose (symmetric -> real eigenvalues, orthogonal eigenvectors)
    let eigen = mat.symmetric_eigen();
    let rot = eigen.eigenvectors;

    // Extract rotation matrix
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = rot[(i, j)];
        }
    }

    // Ensure proper rotation (det = +1)
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det < 0.0 {
        // Flip one column to make it a proper rotation
        for i in 0..3 {
            m[i][0] = -m[i][0];
        }
    }

    algebra_core::physics::quat_rotation::quat_from_rotation_matrix(&m)
}

// ============================================================================
// Algebraic coupling terms
// ============================================================================

/// Octonion bivector coupling to trace-free extrinsic curvature.
///
/// Maps octonion non-associativity (associator norm) to a correction
/// for the trace-free K_{ij}:
///   delta_A_{ij} = alpha_oct * ||[a,b,c]|| * A_{ij} / ||A||
///
/// where alpha_oct is a dimensionless coupling constant.
/// At zero coupling or zero associator: delta = 0.
pub fn octonion_bivector_coupling(
    associator_norm: f64,
    tracefree_k: &SpatialSymmetricTensor,
    alpha_oct: f64,
) -> SpatialSymmetricTensor {
    // Compute ||A|| = sqrt(A_{ij} A^{ij}) (using flat metric for simplicity)
    let mut a_norm_sq = 0.0;
    for i in 0..SDIM {
        for j in 0..SDIM {
            a_norm_sq += tracefree_k[i][j] * tracefree_k[i][j];
        }
    }
    let a_norm = a_norm_sq.sqrt();

    let mut delta = [[0.0; SDIM]; SDIM];
    if a_norm > 1e-30 {
        let factor = alpha_oct * associator_norm / a_norm;
        for i in 0..SDIM {
            for j in 0..SDIM {
                delta[i][j] = factor * tracefree_k[i][j];
            }
        }
    }
    delta
}

/// Sedenion imbalance -> effective energy density.
///
/// Maps imbalance density F(x) to an effective energy density via the
/// viscous dissipation analogy:
///   rho_eff = (1/2) * nu(F) * |K|^2
///
/// where nu is a viscosity coupling function.
/// Common models: Exponential, Linear, PowerLaw (from sign_imbalance::bridge).
///
/// For simplicity, uses exponential coupling: nu = nu_0 * exp(beta * (F - F_vac))
/// where F_vac = 3/8 is the imbalance attractor.
pub fn sedenion_stress_energy(
    imbalance: f64,
    k_trace_sq: f64,
    nu_0: f64,
    beta_coupling: f64,
) -> f64 {
    let nu = nu_0 * (beta_coupling * (imbalance - IMBALANCE_ATTRACTOR)).exp();
    0.5 * nu * k_trace_sq
}

/// Algebraic correction to York time.
///
/// delta_theta = alpha_s * (F - 3/8) * theta_GR
///
/// where alpha_s is a dimensionless coupling constant.
/// At F = 3/8 (vacuum): delta_theta = 0 exactly.
/// F > 3/8 (over-frustrated): enhances expansion/contraction.
/// F < 3/8 (under-frustrated): suppresses expansion/contraction.
pub fn algebraic_york_time_correction(
    imbalance: f64,
    theta_gr: f64,
    alpha_s: f64,
) -> f64 {
    alpha_s * (imbalance - IMBALANCE_ATTRACTOR) * theta_gr
}

/// Map Cayley-Dickson algebra dimension to nacelle count.
///
/// dim=1 (real) -> n=1, dim=2 (complex) -> n=2, dim=4 (quaternion) -> n=4,
/// dim=8 (octonion) -> n=8, dim=16 (sedenion) -> n=16.
///
/// The hypothesis: algebraic dimension determines natural segmentation
/// symmetry of the warp bubble geometry.
pub fn cd_dimension_nacelle_map(cd_dim: usize) -> Option<usize> {
    match cd_dim {
        1 => Some(1),
        2 => Some(2),
        4 => Some(4),
        8 => Some(8),
        16 => Some(16),
        _ => None,
    }
}

/// Compute the full algebraic ADM state at a point.
///
/// Combines standard ADM decomposition with algebraic data.
pub fn algebraic_adm_state(
    adm: AdmDecomposition,
    extrinsic: ExtrinsicCurvatureData,
    octonion_field: [f64; 8],
    sedenion_imbalance: f64,
    associator_norm: f64,
) -> AlgebraicAdmState {
    let quaternion_frame = quaternion_frame_from_spatial_metric(&adm.spatial_metric);

    AlgebraicAdmState {
        adm,
        extrinsic,
        quaternion_frame,
        octonion_field,
        sedenion_imbalance,
        associator_norm,
    }
}

/// Compute total corrected York time including algebraic contribution.
pub fn total_york_time(state: &AlgebraicAdmState, alpha_s: f64) -> f64 {
    state.extrinsic.york_time
        + algebraic_york_time_correction(state.sedenion_imbalance, state.extrinsic.york_time, alpha_s)
}

/// Compute total effective energy density including algebraic source.
pub fn total_energy_density(
    state: &AlgebraicAdmState,
    gr_energy_density: f64,
    nu_0: f64,
    beta_coupling: f64,
) -> f64 {
    let k_trace = trace_symmetric(&state.adm.spatial_metric_inv, &state.extrinsic.k_ij);
    let k_trace_sq = k_trace * k_trace;
    gr_energy_density
        + sedenion_stress_energy(state.sedenion_imbalance, k_trace_sq, nu_0, beta_coupling)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::SPATIAL_IDENTITY;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_flat_metric_gives_identity_quaternion() {
        let q = quaternion_frame_from_spatial_metric(&SPATIAL_IDENTITY);
        // Should be close to identity (1,0,0,0) or (-1,0,0,0)
        let w_abs = q[0].abs();
        assert!(
            w_abs > 0.99,
            "flat metric quaternion scalar part = {} (expected ~1)",
            q[0]
        );
        assert!(q[1].abs() < 0.01, "q_x = {}", q[1]);
        assert!(q[2].abs() < 0.01, "q_y = {}", q[2]);
        assert!(q[3].abs() < 0.01, "q_z = {}", q[3]);
    }

    #[test]
    fn test_zero_coupling_gives_zero_correction() {
        let theta_gr = 1.5;
        let delta = algebraic_york_time_correction(0.4, theta_gr, 0.0);
        assert!(delta.abs() < TOL, "zero coupling: delta = {delta}");
    }

    #[test]
    fn test_imbalance_attractor_gives_zero_correction() {
        let delta = algebraic_york_time_correction(IMBALANCE_ATTRACTOR, 2.0, 1.0);
        assert!(
            delta.abs() < TOL,
            "imbalance attractor: delta = {delta}"
        );
    }

    #[test]
    fn test_sedenion_stress_energy_at_vacuum() {
        let rho = sedenion_stress_energy(IMBALANCE_ATTRACTOR, 1.0, 1.0, 0.0);
        // nu = nu_0 * exp(0) = nu_0 = 1.0, rho = 0.5 * 1.0 * 1.0 = 0.5
        assert!((rho - 0.5).abs() < TOL, "rho at vacuum = {rho}");
    }

    #[test]
    fn test_cd_dimension_map() {
        assert_eq!(cd_dimension_nacelle_map(1), Some(1));
        assert_eq!(cd_dimension_nacelle_map(2), Some(2));
        assert_eq!(cd_dimension_nacelle_map(4), Some(4));
        assert_eq!(cd_dimension_nacelle_map(8), Some(8));
        assert_eq!(cd_dimension_nacelle_map(16), Some(16));
        assert_eq!(cd_dimension_nacelle_map(32), None);
        assert_eq!(cd_dimension_nacelle_map(3), None);
    }

    #[test]
    fn test_octonion_coupling_zero_alpha() {
        let tf_k = [[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]];
        let delta = octonion_bivector_coupling(1.0, &tf_k, 0.0);
        for i in 0..SDIM {
            for j in 0..SDIM {
                assert!(
                    delta[i][j].abs() < TOL,
                    "zero alpha: delta[{i}][{j}] = {}",
                    delta[i][j]
                );
            }
        }
    }

    #[test]
    fn test_octonion_coupling_zero_associator() {
        let tf_k = [[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]];
        let delta = octonion_bivector_coupling(0.0, &tf_k, 1.0);
        for i in 0..SDIM {
            for j in 0..SDIM {
                assert!(
                    delta[i][j].abs() < TOL,
                    "zero assoc: delta[{i}][{j}] = {}",
                    delta[i][j]
                );
            }
        }
    }

    #[test]
    fn test_over_frustrated_enhances() {
        let theta_gr = 1.0;
        let alpha_s = 0.1;
        let delta_over = algebraic_york_time_correction(0.5, theta_gr, alpha_s);
        let delta_under = algebraic_york_time_correction(0.2, theta_gr, alpha_s);
        // Over-frustrated (0.5 > 3/8) should give positive correction
        assert!(
            delta_over > 0.0,
            "over-frustrated should enhance: {delta_over}"
        );
        // Under-frustrated (0.2 < 3/8) should give negative correction
        assert!(
            delta_under < 0.0,
            "under-frustrated should suppress: {delta_under}"
        );
    }
}
