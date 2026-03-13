//! x87-backed Jacobi eigensolver wired through the shared flat-array scaffold.

#![cfg(target_arch = "x86_64")]

use crate::jacobi_shared::symmetric_eigenvalues_with_backend;
use cd_kernel::{
    AlgebraError, AlgebraResult,
    x87_jacobi_kernels::{
        x87_givens_diagonal_update as diagonal_update_f64, x87_givens_sincos as givens_sincos_f64,
    },
};

// ── Flat-array Jacobi solver ────────────────────────────────────────────────

/// Compute eigenvalues of a symmetric matrix using Jacobi iteration with
/// x87 80-bit precision for Givens rotations.
pub fn symmetric_eigenvalues_x87(
    matrix: &[f64],
    n: usize,
    max_iterations: usize,
    tolerance: f64,
) -> AlgebraResult<Vec<f64>> {
    symmetric_eigenvalues_with_backend(
        matrix,
        n,
        max_iterations,
        tolerance,
        x87_rotation_f64,
        diagonal_update_f64,
        "x87",
    )
}

fn x87_rotation_f64(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
    let y = 2.0 * apq;
    let x = app - aqq;
    givens_sincos_f64(y, x)
}

/// Full symmetric Jacobi eigensystem with x87 80-bit precision.
///
/// Returns `(eigenvalues, v_flat)` sorted by descending |eigenvalue|.
/// `v_flat` is the row-major eigenvector matrix: column k is the eigenvector
/// for `eigenvalues[k]`, accessed as `v_flat[i*n + k]` for row i.
///
/// Both the Givens angle (via x87 atan2 in cd_kernel) and the diagonal update
/// (via x87 polynomial assembly) execute in 80-bit extended precision.
/// The same sin_t/cos_t values drive V accumulation, so eigenvectors inherit
/// x87 precision at no additional cost.
pub fn symmetric_eigensystem_x87(
    matrix: &[f64],
    n: usize,
    max_iterations: usize,
    tolerance: f64,
) -> AlgebraResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if matrix.len() != n * n {
        return Err(AlgebraError::DimensionMismatch {
            expected: n * n,
            found: matrix.len(),
        });
    }

    let mut a = matrix.to_vec();
    // V starts as the identity: V[:,i] = e_i
    let mut v: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let mut converged = false;

    for _ in 0..max_iterations {
        // Find the off-diagonal element with largest absolute value
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v_abs = a[i * n + j].abs();
                if v_abs > max_val {
                    max_val = v_abs;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tolerance {
            converged = true;
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        // Degenerate diagonal case: use 45-degree rotation
        let (sin_t, cos_t) = if (app - aqq).abs() < 1.0e-15 {
            let scale = std::f64::consts::FRAC_1_SQRT_2;
            (scale, scale)
        } else {
            x87_rotation_f64(app, aqq, apq)
        };

        // Off-diagonal A update: skip rows p and q (handled by diagonal update)
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            let aip = a[i * n + p];
            let aiq = a[i * n + q];
            let new_ip = cos_t * aip + sin_t * aiq;
            let new_iq = -sin_t * aip + cos_t * aiq;
            a[i * n + p] = new_ip;
            a[p * n + i] = new_ip;
            a[i * n + q] = new_iq;
            a[q * n + i] = new_iq;
        }

        // Diagonal A update via x87 polynomial
        let (new_pp, new_qq) = diagonal_update_f64(sin_t, cos_t, app, apq, aqq);
        a[p * n + p] = new_pp;
        a[q * n + q] = new_qq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // V accumulation: V_new = V * G for ALL rows (no row skipping)
        // G[p,p]=cos_t, G[q,q]=cos_t, G[p,q]=-sin_t, G[q,p]=sin_t
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = cos_t * vip + sin_t * viq;
            v[i * n + q] = -sin_t * vip + cos_t * viq;
        }
    }

    if !converged {
        return Err(AlgebraError::NumericalError(format!(
            "x87 Jacobi eigensystem failed to converge after {max_iterations} steps"
        )));
    }

    // Sort eigenvalues by descending |lambda|
    let mut indexed: Vec<(usize, f64)> = (0..n).map(|i| (i, a[i * n + i])).collect();
    indexed.sort_by(|(_, l), (_, r)| r.abs().total_cmp(&l.abs()).then_with(|| r.total_cmp(l)));

    let eigenvalues: Vec<f64> = indexed.iter().map(|(_, ev)| *ev).collect();
    // Permute V columns: output column j gets input column indexed[j].0
    let v_out: Vec<f64> = (0..n * n)
        .map(|idx| v[(idx / n) * n + indexed[idx % n].0])
        .collect();

    Ok((eigenvalues, v_out))
}

#[cfg(test)]
mod tests {
    use super::{symmetric_eigensystem_x87, symmetric_eigenvalues_x87};

    #[test]
    fn test_x87_jacobi_diagonal_matrix() {
        let matrix = [5.0, 0.0, 0.0, 3.0];
        let eigs = symmetric_eigenvalues_x87(&matrix, 2, 64, 1.0e-14).unwrap();

        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] - 5.0).abs() < 1.0e-14);
        assert!((eigs[1] - 3.0).abs() < 1.0e-14);
    }

    #[test]
    fn test_x87_jacobi_known_spectrum() {
        let matrix = [
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let flat: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
        let eigs = symmetric_eigenvalues_x87(&flat, 3, 500, 1.0e-14).unwrap();

        assert_eq!(eigs.len(), 3);
        assert!(eigs[0].abs() >= eigs[1].abs());
        assert!(eigs[1].abs() >= eigs[2].abs());
    }

    // ── Eigensystem tests ────────────────────────────────────────────────────

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn max_off_diagonal_vtv(v: &[f64], n: usize) -> f64 {
        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let col_i: Vec<f64> = (0..n).map(|r| v[r * n + i]).collect();
                let col_j: Vec<f64> = (0..n).map(|r| v[r * n + j]).collect();
                let val = dot(&col_i, &col_j);
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (val - expected).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        max_err
    }

    #[test]
    fn test_eigensystem_x87_2x2_known() {
        // [[3,1],[1,3]] has eigenvalues 4, 2 with eigenvectors [1,1]/sqrt(2), [1,-1]/sqrt(2)
        let matrix = [3.0_f64, 1.0, 1.0, 3.0];
        let (eigs, v) = symmetric_eigensystem_x87(&matrix, 2, 200, 1.0e-14).unwrap();

        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] - 4.0).abs() < 1.0e-12, "lambda_0={}", eigs[0]);
        assert!((eigs[1] - 2.0).abs() < 1.0e-12, "lambda_1={}", eigs[1]);

        // V^T V = I
        assert!(max_off_diagonal_vtv(&v, 2) < 1.0e-12, "orthogonality error");

        // Reconstruct: V diag(lambda) V^T should equal original matrix
        let a_rec: Vec<f64> = (0..4)
            .map(|idx| {
                let i = idx / 2;
                let j = idx % 2;
                (0..2)
                    .map(|k| v[i * 2 + k] * eigs[k] * v[j * 2 + k])
                    .sum::<f64>()
            })
            .collect();
        for (orig, rec) in matrix.iter().zip(a_rec.iter()) {
            assert!(
                (orig - rec).abs() < 1.0e-10,
                "reconstruct diff {}",
                (orig - rec).abs()
            );
        }
    }

    #[test]
    fn test_eigensystem_x87_orthonormal() {
        // 4x4 symmetric matrix; verify V^T V ~ I
        let rows = [
            [4.0_f64, 1.2, 0.3, -0.5],
            [1.2, 3.0, 0.8, 0.1],
            [0.3, 0.8, 2.5, -0.4],
            [-0.5, 0.1, -0.4, 1.0],
        ];
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let (_, v) = symmetric_eigensystem_x87(&flat, 4, 10_000, 1.0e-14).unwrap();

        let max_err = max_off_diagonal_vtv(&v, 4);
        assert!(max_err < 1.0e-11, "max V^T V off-diagonal = {max_err}");
    }

    #[test]
    fn test_eigensystem_x87_reconstruct() {
        // 3x3 Hilbert matrix: H[i,j] = 1/(i+j+1), ill-conditioned but exact eigenstructure
        let n = 3_usize;
        let flat: Vec<f64> = (0..n)
            .flat_map(|i| (0..n).map(move |j| 1.0 / (i + j + 1) as f64))
            .collect();
        let (eigs, v) = symmetric_eigensystem_x87(&flat, n, 10_000, 1.0e-14).unwrap();

        // Reconstruct A = V diag(lambda) V^T
        let a_rec: Vec<f64> = (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                (0..n)
                    .map(|k| v[i * n + k] * eigs[k] * v[j * n + k])
                    .sum::<f64>()
            })
            .collect();

        let max_err = flat
            .iter()
            .zip(a_rec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1.0e-10,
            "Hilbert reconstruction error = {max_err}"
        );
    }

    #[test]
    fn test_eigensystem_x87_identity() {
        // Identity matrix: all eigenvalues = 1, V^T V = I
        let n = 3_usize;
        let flat: Vec<f64> = (0..n * n)
            .map(|idx| if idx / n == idx % n { 1.0 } else { 0.0 })
            .collect();
        let (eigs, v) = symmetric_eigensystem_x87(&flat, n, 1_000, 1.0e-14).unwrap();

        for (k, &e) in eigs.iter().enumerate() {
            assert!((e - 1.0).abs() < 1.0e-14, "eig[{k}]={e}");
        }
        assert!(max_off_diagonal_vtv(&v, n) < 1.0e-12);
    }
}
