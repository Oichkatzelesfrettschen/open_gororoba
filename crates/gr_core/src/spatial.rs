//! 3D spatial metric types for ADM 3+1 decomposition.
//!
//! Provides spatial (3-dimensional) counterparts to the 4D types in `metric.rs`.
//! These types represent objects living on a spacelike hypersurface Sigma_t:
//!
//! - gamma_{ij}: the induced 3-metric (SpatialMetric)
//! - K_{ij}: extrinsic curvature (SpatialSymmetricTensor)
//! - beta^i: shift vector (SpatialVector)
//! - Gamma^i_{jk}: spatial Christoffel symbols (SpatialChristoffel)
//!
//! # Conventions
//! - Indices i,j,k run over {0,1,2} corresponding to spatial coordinates.
//! - For Boyer-Lindquist-type coords: 0=r, 1=theta, 2=phi
//! - For Cartesian coords: 0=x, 1=y, 2=z
//!
//! # References
//! - Arnowitt, Deser, Misner (1962): The Dynamics of General Relativity
//! - Gourgoulhon (2012): 3+1 Formalism in General Relativity
//! - York (1979): Kinematics and dynamics of general relativity

/// Spatial dimension.
pub const SDIM: usize = 3;

/// Spatial coordinate indices (Cartesian labeling).
pub const X1: usize = 0;
pub const X2: usize = 1;
pub const X3: usize = 2;

/// The induced 3-metric gamma_{ij} on a spacelike hypersurface.
pub type SpatialMetric = [[f64; SDIM]; SDIM];

/// A symmetric 3x3 spatial tensor (extrinsic curvature K_{ij}, shear, etc.).
pub type SpatialSymmetricTensor = [[f64; SDIM]; SDIM];

/// A spatial 3-vector (shift beta^i, momentum density S^i, etc.).
pub type SpatialVector = [f64; SDIM];

/// Spatial Christoffel symbols Gamma^i_{jk}, stored as `[i]``[j]``[k]`.
pub type SpatialChristoffel = [[[f64; SDIM]; SDIM]; SDIM];

/// Identity 3-metric (flat Euclidean space).
pub const SPATIAL_IDENTITY: SpatialMetric = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Zero spatial vector.
pub const SPATIAL_ZERO_VECTOR: SpatialVector = [0.0, 0.0, 0.0];

/// Invert a 3x3 symmetric matrix using nalgebra.
///
/// For spatial metrics on hypersurfaces, the matrix is always positive-definite
/// (Riemannian signature) and thus invertible. Panics if singular.
pub fn invert_3x3_symmetric(m: &SpatialMetric) -> SpatialMetric {
    let mat = nalgebra::Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    );

    let inv = mat
        .try_inverse()
        .expect("spatial metric must be invertible (degenerate hypersurface?)");

    let mut result = [[0.0; SDIM]; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            result[i][j] = inv[(i, j)];
        }
    }
    result
}

/// Determinant of a 3x3 matrix (spatial metric determinant).
///
/// det(gamma) = gamma_{00}(gamma_{11}*gamma_{22} - gamma_{12}^2)
///            - gamma_{01}(gamma_{01}*gamma_{22} - gamma_{12}*gamma_{02})
///            + gamma_{02}(gamma_{01}*gamma_{12} - gamma_{11}*gamma_{02})
pub fn spatial_determinant(m: &SpatialMetric) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Raise the index of a spatial covector using the inverse metric.
///
/// v^i = gamma^{ij} v_j
pub fn raise_index_vector(gamma_inv: &SpatialMetric, v_lower: &SpatialVector) -> SpatialVector {
    let mut v_upper = [0.0; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            v_upper[i] += gamma_inv[i][j] * v_lower[j];
        }
    }
    v_upper
}

/// Lower the index of a spatial vector using the metric.
///
/// v_i = gamma_{ij} v^j
pub fn lower_index_vector(gamma: &SpatialMetric, v_upper: &SpatialVector) -> SpatialVector {
    let mut v_lower = [0.0; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            v_lower[i] += gamma[i][j] * v_upper[j];
        }
    }
    v_lower
}

/// Trace of a symmetric tensor with respect to the inverse metric.
///
/// tr(A) = gamma^{ij} A_{ij}
pub fn trace_symmetric(gamma_inv: &SpatialMetric, a: &SpatialSymmetricTensor) -> f64 {
    let mut tr = 0.0;
    for i in 0..SDIM {
        for j in 0..SDIM {
            tr += gamma_inv[i][j] * a[i][j];
        }
    }
    tr
}

/// Contract two symmetric tensors: A^{ij} B_{ij}.
///
/// Uses the inverse metric to raise indices of A:
/// result = gamma^{ik} gamma^{jl} A_{kl} B_{ij}
///
/// When A_upper is already the contravariant form, use `contract_upper_lower`.
pub fn contract_symmetric_tensors(
    gamma_inv: &SpatialMetric,
    a: &SpatialSymmetricTensor,
    b: &SpatialSymmetricTensor,
) -> f64 {
    let mut result = 0.0;
    for i in 0..SDIM {
        for j in 0..SDIM {
            for k in 0..SDIM {
                for l in 0..SDIM {
                    result += gamma_inv[i][k] * gamma_inv[j][l] * a[k][l] * b[i][j];
                }
            }
        }
    }
    result
}

/// Contract an upper-index tensor with a lower-index tensor: A^{ij} B_{ij}.
pub fn contract_upper_lower(
    a_upper: &SpatialSymmetricTensor,
    b_lower: &SpatialSymmetricTensor,
) -> f64 {
    let mut result = 0.0;
    for i in 0..SDIM {
        for j in 0..SDIM {
            result += a_upper[i][j] * b_lower[i][j];
        }
    }
    result
}

/// Raise both indices of a symmetric tensor.
///
/// A^{ij} = gamma^{ik} gamma^{jl} A_{kl}
pub fn raise_symmetric_tensor(
    gamma_inv: &SpatialMetric,
    a_lower: &SpatialSymmetricTensor,
) -> SpatialSymmetricTensor {
    let mut a_upper = [[0.0; SDIM]; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            for k in 0..SDIM {
                for l in 0..SDIM {
                    a_upper[i][j] += gamma_inv[i][k] * gamma_inv[j][l] * a_lower[k][l];
                }
            }
        }
    }
    a_upper
}

/// Compute the trace-free part of a symmetric tensor.
///
/// A^TF_{ij} = A_{ij} - (1/3) gamma_{ij} gamma^{kl} A_{kl}
pub fn tracefree_part(
    gamma: &SpatialMetric,
    gamma_inv: &SpatialMetric,
    a: &SpatialSymmetricTensor,
) -> SpatialSymmetricTensor {
    let tr = trace_symmetric(gamma_inv, a);
    let mut tf = [[0.0; SDIM]; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            tf[i][j] = a[i][j] - (1.0 / 3.0) * gamma[i][j] * tr;
        }
    }
    tf
}

/// Inner product of two spatial vectors using the metric.
///
/// <u, v> = gamma_{ij} u^i v^j
pub fn spatial_inner_product(gamma: &SpatialMetric, u: &SpatialVector, v: &SpatialVector) -> f64 {
    let mut result = 0.0;
    for i in 0..SDIM {
        for j in 0..SDIM {
            result += gamma[i][j] * u[i] * v[j];
        }
    }
    result
}

/// Norm of a spatial vector: ||v|| = sqrt(gamma_{ij} v^i v^j).
pub fn spatial_norm(gamma: &SpatialMetric, v: &SpatialVector) -> f64 {
    spatial_inner_product(gamma, v, v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    #[allow(clippy::needless_range_loop)] // 2D matrix indexing is clearer with indices
    fn test_identity_inverse() {
        let inv = invert_3x3_symmetric(&SPATIAL_IDENTITY);
        for i in 0..SDIM {
            for j in 0..SDIM {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i][j] - expected).abs() < TOL,
                    "inv[{i}][{j}] = {}, expected {expected}",
                    inv[i][j]
                );
            }
        }
    }

    #[test]
    fn test_diagonal_inverse() {
        let gamma = [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]];
        let inv = invert_3x3_symmetric(&gamma);
        assert!((inv[0][0] - 0.25).abs() < TOL);
        assert!((inv[1][1] - 1.0 / 9.0).abs() < TOL);
        assert!((inv[2][2] - 1.0 / 16.0).abs() < TOL);
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // 2D matrix product with triple index i,j,k
    fn test_inverse_round_trip() {
        // A non-diagonal positive-definite metric
        let gamma = [[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 4.0]];
        let inv = invert_3x3_symmetric(&gamma);
        // gamma * gamma_inv should be identity
        for i in 0..SDIM {
            for j in 0..SDIM {
                let mut product = 0.0;
                for k in 0..SDIM {
                    product += gamma[i][k] * inv[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product - expected).abs() < TOL,
                    "(gamma * inv)[{i}][{j}] = {product}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_determinant_identity() {
        let det = spatial_determinant(&SPATIAL_IDENTITY);
        assert!((det - 1.0).abs() < TOL);
    }

    #[test]
    fn test_determinant_diagonal() {
        let gamma = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]];
        let det = spatial_determinant(&gamma);
        assert!((det - 30.0).abs() < TOL);
    }

    #[test]
    fn test_determinant_general() {
        let gamma = [[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 4.0]];
        // Compute via nalgebra for cross-check
        let mat = nalgebra::Matrix3::new(
            gamma[0][0],
            gamma[0][1],
            gamma[0][2],
            gamma[1][0],
            gamma[1][1],
            gamma[1][2],
            gamma[2][0],
            gamma[2][1],
            gamma[2][2],
        );
        let expected = mat.determinant();
        let det = spatial_determinant(&gamma);
        assert!(
            (det - expected).abs() < TOL,
            "det = {det}, expected {expected}"
        );
    }

    #[test]
    fn test_raise_lower_round_trip() {
        let gamma = [[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 4.0]];
        let gamma_inv = invert_3x3_symmetric(&gamma);
        let v = [1.0, 2.0, 3.0];
        let v_low = lower_index_vector(&gamma, &v);
        let v_back = raise_index_vector(&gamma_inv, &v_low);
        for i in 0..SDIM {
            assert!(
                (v_back[i] - v[i]).abs() < TOL,
                "round-trip failed at {i}: {} vs {}",
                v_back[i],
                v[i]
            );
        }
    }

    #[test]
    fn test_trace_identity() {
        let a = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];
        let tr = trace_symmetric(&SPATIAL_IDENTITY, &a);
        assert!(
            (tr - 11.0).abs() < TOL,
            "trace of A with identity metric: {tr}"
        );
    }

    #[test]
    fn test_tracefree_has_zero_trace() {
        let gamma = [[2.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let gamma_inv = invert_3x3_symmetric(&gamma);
        let a = [[1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let tf = tracefree_part(&gamma, &gamma_inv, &a);
        let tr_tf = trace_symmetric(&gamma_inv, &tf);
        assert!(
            tr_tf.abs() < TOL,
            "trace of trace-free part should be zero: {tr_tf}"
        );
    }

    #[test]
    fn test_spatial_norm_euclidean() {
        let v = [3.0, 4.0, 0.0];
        let norm = spatial_norm(&SPATIAL_IDENTITY, &v);
        assert!((norm - 5.0).abs() < TOL);
    }
}
