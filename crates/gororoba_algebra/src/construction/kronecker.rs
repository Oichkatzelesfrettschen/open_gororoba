//! Kronecker product utilities for complex matrices.
//!
//! Provides the 2x2 -> 4x4 Kronecker (tensor) product used throughout the
//! spin tomography and QGP bridge layers for constructing two-qubit operators
//! from single-qubit Pauli matrices.

use nalgebra::{Matrix2, Matrix4};
use num_complex::Complex64;

/// Kronecker product of two 2x2 complex matrices, yielding a 4x4 matrix.
///
/// Given matrices A and B, the Kronecker product A (x) B is the 4x4 block
/// matrix where block (i,j) equals A_{ij} * B.  This is the standard
/// construction for two-qubit operator spaces from single-qubit operators.
pub fn kron2(a: &Matrix2<Complex64>, b: &Matrix2<Complex64>) -> Matrix4<Complex64> {
    let mut m = Matrix4::zeros();
    for i in 0..2 {
        for j in 0..2 {
            let val_a = a[(i, j)];
            for k in 0..2 {
                for l in 0..2 {
                    m[(2 * i + k, 2 * j + l)] = val_a * b[(k, l)];
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye2() -> Matrix2<Complex64> {
        Matrix2::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        )
    }

    #[test]
    fn kron2_identity_yields_identity4() {
        let e = eye2();
        let result = kron2(&e, &e);
        let expected = Matrix4::<Complex64>::identity();
        for i in 0..4 {
            for j in 0..4 {
                let diff = (result[(i, j)] - expected[(i, j)]).norm();
                assert!(diff < 1e-14, "kron2(I,I)[{i},{j}] diff={diff}");
            }
        }
    }

    #[test]
    fn kron2_sigma_z_tensor_identity() {
        let sz = Matrix2::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        );
        let e = eye2();
        let result = kron2(&sz, &e);
        // sigma_z (x) I = diag(1, 1, -1, -1)
        assert!((result[(0, 0)].re - 1.0).abs() < 1e-14);
        assert!((result[(1, 1)].re - 1.0).abs() < 1e-14);
        assert!((result[(2, 2)].re - (-1.0)).abs() < 1e-14);
        assert!((result[(3, 3)].re - (-1.0)).abs() < 1e-14);
        // Off-diagonals zero
        assert!((result[(0, 1)].norm()) < 1e-14);
        assert!((result[(2, 3)].norm()) < 1e-14);
    }

    #[test]
    fn kron2_trace_is_product_of_traces() {
        let a = Matrix2::new(
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(3.0, 0.0),
        );
        let b = Matrix2::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(4.0, 0.0),
        );
        let result = kron2(&a, &b);
        let tr_a = a[(0, 0)] + a[(1, 1)];
        let tr_b = b[(0, 0)] + b[(1, 1)];
        let tr_kron = result.trace();
        assert!(
            (tr_kron - tr_a * tr_b).norm() < 1e-12,
            "tr(A (x) B) must equal tr(A) * tr(B)"
        );
    }
}
