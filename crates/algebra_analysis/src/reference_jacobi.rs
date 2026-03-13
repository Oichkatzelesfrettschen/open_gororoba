//! Reference f64 Jacobi eigensolver wired through the shared flat-array scaffold.

use crate::jacobi_shared::{
    flatten_square_matrix, recommended_jacobi_iterations, symmetric_eigenvalues_with_backend,
};
use cd_kernel::error::AlgebraResult;

#[inline]
fn f64_rotation(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
    let theta = 0.5 * (2.0 * apq).atan2(app - aqq);
    theta.sin_cos()
}

#[inline]
fn f64_diagonal_update(sin_t: f64, cos_t: f64, app: f64, apq: f64, aqq: f64) -> (f64, f64) {
    (
        cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq,
        sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq,
    )
}

/// Full symmetric Jacobi eigensystem in standard f64 precision.
///
/// Returns `(eigenvalues, v_flat)` sorted by descending |eigenvalue|.
/// `v_flat` is the row-major eigenvector matrix: column k is the eigenvector
/// for `eigenvalues[k]`, accessed as `v_flat[i*n + k]`.
/// Use this as the reference/oracle backend when x87 or DD precision is not required.
pub fn symmetric_eigensystem_f64(matrix: &[Vec<f64>]) -> AlgebraResult<(Vec<f64>, Vec<f64>)> {
    use cd_kernel::error::AlgebraError;

    let n = matrix.len();
    let flat = flatten_square_matrix(matrix)?;
    let max_iterations = recommended_jacobi_iterations(n);
    let tolerance = 1.0e-12_f64;

    if n == 0 {
        return Ok((vec![], vec![]));
    }

    let mut a = flat;
    let mut v: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let mut converged = false;

    for _ in 0..max_iterations {
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

        let (sin_t, cos_t) = if (app - aqq).abs() < 1.0e-15 {
            let scale = std::f64::consts::FRAC_1_SQRT_2;
            (scale, scale)
        } else {
            f64_rotation(app, aqq, apq)
        };

        // Off-diagonal A update (skip p and q rows)
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

        // Diagonal A update
        let (new_pp, new_qq) = f64_diagonal_update(sin_t, cos_t, app, apq, aqq);
        a[p * n + p] = new_pp;
        a[q * n + q] = new_qq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // V accumulation: V_new = V * G for ALL rows
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = cos_t * vip + sin_t * viq;
            v[i * n + q] = -sin_t * vip + cos_t * viq;
        }
    }

    if !converged {
        return Err(AlgebraError::NumericalError(format!(
            "f64 Jacobi eigensystem failed to converge after {max_iterations} steps"
        )));
    }

    // Sort by descending |eigenvalue|
    let mut indexed: Vec<(usize, f64)> = (0..n).map(|i| (i, a[i * n + i])).collect();
    indexed.sort_by(|(_, l), (_, r)| r.abs().total_cmp(&l.abs()).then_with(|| r.total_cmp(l)));

    let eigenvalues: Vec<f64> = indexed.iter().map(|(_, ev)| *ev).collect();
    let v_out: Vec<f64> = (0..n * n)
        .map(|idx| v[(idx / n) * n + indexed[idx % n].0])
        .collect();

    Ok((eigenvalues, v_out))
}

/// Compute eigenvalues of a symmetric matrix using the plain f64 Jacobi lane.
///
/// This is the reference/oracle backend for cross-checking the higher-precision
/// x87 and double-double lanes through the same shared scaffold.
pub fn symmetric_eigenvalues_f64(matrix: &[Vec<f64>]) -> AlgebraResult<Vec<f64>> {
    let n = matrix.len();
    let flat = flatten_square_matrix(matrix)?;
    symmetric_eigenvalues_with_backend(
        &flat,
        n,
        recommended_jacobi_iterations(n),
        1.0e-12,
        f64_rotation,
        f64_diagonal_update,
        "f64",
    )
}

#[cfg(test)]
mod tests {
    use super::{symmetric_eigensystem_f64, symmetric_eigenvalues_f64};

    #[test]
    fn test_reference_jacobi_diagonal_matrix() {
        let matrix = vec![vec![5.0, 0.0], vec![0.0, 3.0]];
        let eigs = symmetric_eigenvalues_f64(&matrix).unwrap();

        assert_eq!(eigs, vec![5.0, 3.0]);
    }

    #[test]
    fn test_reference_jacobi_known_spectrum() {
        let matrix = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let eigs = symmetric_eigenvalues_f64(&matrix).unwrap();

        assert_eq!(eigs.len(), 3);
        assert!(eigs[0].abs() >= eigs[1].abs());
        assert!(eigs[1].abs() >= eigs[2].abs());
    }

    // ── Eigensystem tests ────────────────────────────────────────────────────

    fn max_off_diagonal_vtv(v: &[f64], n: usize) -> f64 {
        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let inner: f64 = (0..n).map(|r| v[r * n + i] * v[r * n + j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (inner - expected).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        max_err
    }

    #[test]
    fn test_eigensystem_f64_2x2_known() {
        let matrix = vec![vec![3.0_f64, 1.0], vec![1.0, 3.0]];
        let (eigs, v) = symmetric_eigensystem_f64(&matrix).unwrap();

        assert!((eigs[0] - 4.0).abs() < 1.0e-11, "lambda_0={}", eigs[0]);
        assert!((eigs[1] - 2.0).abs() < 1.0e-11, "lambda_1={}", eigs[1]);
        assert!(
            max_off_diagonal_vtv(&v, 2) < 1.0e-11,
            "orthogonality violated"
        );

        let a_rec: Vec<f64> = (0..4)
            .map(|idx| {
                let i = idx / 2;
                let j = idx % 2;
                (0..2)
                    .map(|k| v[i * 2 + k] * eigs[k] * v[j * 2 + k])
                    .sum::<f64>()
            })
            .collect();
        let orig = [3.0_f64, 1.0, 1.0, 3.0];
        for (a, b) in orig.iter().zip(a_rec.iter()) {
            assert!((a - b).abs() < 1.0e-11);
        }
    }

    #[test]
    fn test_eigensystem_f64_orthonormal() {
        let matrix = vec![
            vec![4.0_f64, 1.2, 0.3, -0.5],
            vec![1.2, 3.0, 0.8, 0.1],
            vec![0.3, 0.8, 2.5, -0.4],
            vec![-0.5, 0.1, -0.4, 1.0],
        ];
        let (_, v) = symmetric_eigensystem_f64(&matrix).unwrap();
        let max_err = max_off_diagonal_vtv(&v, 4);
        assert!(max_err < 1.0e-11, "max V^T V off-diagonal = {max_err}");
    }

    #[test]
    fn test_eigensystem_f64_reconstruct() {
        let n = 3_usize;
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| 1.0 / (i + j + 1) as f64).collect())
            .collect();
        let flat: Vec<f64> = matrix.iter().flat_map(|r| r.iter().copied()).collect();
        let (eigs, v) = symmetric_eigensystem_f64(&matrix).unwrap();

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
        assert!(max_err < 1.0e-9, "Hilbert reconstruction error = {max_err}");
    }
}
