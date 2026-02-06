//! Bridge to qua_ten_net for optimized tensor operations.
//!
//! This module provides integration with the qua_ten_net crate for:
//! - Optimized tensor contraction ordering
//! - Truncated SVD for bond dimension control
//! - Efficient tensor dot products
//!
//! # Literature
//!
//! - Gray & Kourtis (2021): Hyper-optimized tensor network contraction
//! - Schollwoeck (2011): DMRG review with SVD-based truncation

use ndarray::{Array2, ArrayD, s};
use qua_ten_net::tencon::contract;
use qua_ten_net::tendot::tensor_dot;
use qua_ten_net::tensor::svd;

/// Result of an SVD truncation operation.
#[derive(Debug, Clone)]
pub struct TruncatedSVD {
    /// Left unitary matrix (truncated).
    pub u: Array2<f64>,
    /// Singular values (truncated).
    pub s: Vec<f64>,
    /// Right unitary matrix (truncated).
    pub vt: Array2<f64>,
    /// Number of singular values kept.
    pub rank: usize,
    /// Truncation error (sum of discarded squared singular values).
    pub truncation_error: f64,
}

/// Perform truncated SVD on a 2D array.
///
/// Keeps the largest `max_rank` singular values, or all if `max_rank` is None.
///
/// # Arguments
///
/// * `matrix` - 2D array to decompose
/// * `max_rank` - Maximum number of singular values to keep
/// * `min_singular_value` - Minimum singular value to keep (defaults to 1e-14)
///
/// # Returns
///
/// `TruncatedSVD` with truncated U, S, V^T matrices and error estimate.
pub fn truncated_svd(
    matrix: Array2<f64>,
    max_rank: Option<usize>,
    min_singular_value: f64,
) -> Result<TruncatedSVD, String> {
    let svd_result = svd(matrix)?;

    // Determine truncation rank
    let full_rank = svd_result.sigma.len();
    let max_r = max_rank.unwrap_or(full_rank);

    // Find cutoff based on min_singular_value
    let sv_cutoff = svd_result
        .sigma
        .iter()
        .take_while(|&&sv| sv >= min_singular_value)
        .count();

    let rank = max_r.min(sv_cutoff).min(full_rank);

    // Compute truncation error (sum of squared discarded singular values)
    let truncation_error: f64 = svd_result
        .sigma
        .iter()
        .skip(rank)
        .map(|sv| sv * sv)
        .sum();

    // Truncate matrices
    let u_trunc = svd_result.u.slice(s![.., ..rank]).to_owned();
    let s_trunc: Vec<f64> = svd_result.sigma.iter().take(rank).cloned().collect();
    let vt_trunc = svd_result.vt.slice(s![..rank, ..]).to_owned();

    Ok(TruncatedSVD {
        u: u_trunc,
        s: s_trunc,
        vt: vt_trunc,
        rank,
        truncation_error,
    })
}

/// Contract a tensor network given tensors and their leg orderings.
///
/// This wraps qua_ten_net::tencon::contract with a more ergonomic interface.
///
/// # Arguments
///
/// * `tensors` - List of tensors to contract
/// * `orders` - For each tensor, list of leg indices (shared indices are contracted)
///
/// # Returns
///
/// Contracted result tensor
pub fn contract_network(
    tensors: &[ArrayD<f64>],
    orders: &[Vec<i32>],
) -> Result<ArrayD<f64>, String> {
    if tensors.len() != orders.len() {
        return Err("Number of tensors must match number of orderings".to_string());
    }
    if tensors.is_empty() {
        return Err("Cannot contract empty tensor list".to_string());
    }

    // Convert Vec<i32> to &[i32] for the qua_ten_net API
    let order_refs: Vec<&[i32]> = orders.iter().map(|v| v.as_slice()).collect();
    contract(tensors, &order_refs)
}

/// Compute tensor dot product along specified axes.
///
/// Wraps qua_ten_net::tendot::tensor_dot for MPS-style contractions.
///
/// The axes are specified as pairs: (axis_a, axis_b) indicating which
/// axis of tensor a contracts with which axis of tensor b.
pub fn tensor_contract(
    a: &ArrayD<f64>,
    b: &ArrayD<f64>,
    axes_a: &[usize],
    axes_b: &[usize],
) -> Result<ArrayD<f64>, String> {
    if axes_a.len() != axes_b.len() {
        return Err("Axis lists must have same length".to_string());
    }
    // qua_ten_net expects interleaved axes: [a0, b0, a1, b1, ...]
    let mut axis_vec: Vec<usize> = Vec::with_capacity(axes_a.len() * 2);
    for (&ax_a, &ax_b) in axes_a.iter().zip(axes_b.iter()) {
        axis_vec.push(ax_a);
        axis_vec.push(ax_b);
    }
    tensor_dot(a, b, axis_vec)
}

/// Estimate contraction cost for a given ordering.
///
/// Returns the estimated number of floating point operations.
pub fn estimate_contraction_cost(shapes: &[Vec<usize>], _orders: &[Vec<i32>]) -> usize {
    // Simple estimate: product of all dimensions involved
    // A proper implementation would use optimal contraction path finding
    let mut cost = 1usize;
    for shape in shapes {
        for &dim in shape {
            cost = cost.saturating_mul(dim);
        }
    }
    cost
}

/// Apply truncated SVD to an MPS tensor bond.
///
/// This is the core operation for MPS canonicalization and compression.
///
/// # Arguments
///
/// * `tensor` - MPS tensor reshaped as 2D matrix (left_dim * phys_dim, right_dim)
/// * `chi_max` - Maximum bond dimension after truncation
///
/// # Returns
///
/// (left_tensor, right_tensor, truncation_error)
pub fn truncate_mps_bond(
    tensor: Array2<f64>,
    chi_max: usize,
) -> Result<(Array2<f64>, Array2<f64>, f64), String> {
    let result = truncated_svd(tensor, Some(chi_max), 1e-14)?;

    // Absorb singular values into left tensor: L = U * diag(S)
    let mut left = result.u;
    for (i, &sv) in result.s.iter().enumerate() {
        for j in 0..left.nrows() {
            left[[j, i]] *= sv;
        }
    }

    Ok((left, result.vt, result.truncation_error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_truncated_svd_full_rank() {
        let matrix: Array2<f64> = array![[1.0, 0.0], [0.0, 2.0]];
        let result = truncated_svd(matrix, None, 1e-14).unwrap();

        assert_eq!(result.rank, 2);
        assert!(result.truncation_error < 1e-10);
    }

    #[test]
    fn test_truncated_svd_with_cutoff() {
        // Matrix with singular values 3, 2, 1
        let matrix: Array2<f64> = array![
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let result = truncated_svd(matrix, Some(2), 1e-14).unwrap();

        assert_eq!(result.rank, 2);
        // Truncation error = 1^2 = 1
        assert!((result.truncation_error - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_truncate_mps_bond() {
        // Simple 2x2 identity-like tensor
        let tensor: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]];
        let (left, right, error) = truncate_mps_bond(tensor, 2).unwrap();

        // For identity, no truncation should occur
        assert!(error < 1e-10);
        assert_eq!(left.ncols(), 2);
        assert_eq!(right.nrows(), 2);
    }

    #[test]
    fn test_contraction_cost_estimate() {
        let shapes = vec![vec![2, 3], vec![3, 4]];
        let orders = vec![vec![0, 1], vec![1, 2]];
        let cost = estimate_contraction_cost(&shapes, &orders);
        // Simple estimate: 2*3*3*4 = 72
        assert!(cost > 0);
    }
}
