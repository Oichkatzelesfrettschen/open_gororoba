//! Shared flat-array Jacobi solver scaffold for precision-specific backends.

use cd_kernel::error::{AlgebraError, AlgebraResult};

const DEGENERATE_DIAGONAL_THRESHOLD: f64 = 1.0e-15;

pub(crate) const fn recommended_jacobi_iterations(n: usize) -> usize {
    100usize.saturating_mul(n).saturating_mul(n)
}

pub(crate) fn flatten_square_matrix(matrix: &[Vec<f64>]) -> AlgebraResult<Vec<f64>> {
    let n = matrix.len();
    let mut flat = Vec::with_capacity(n * n);

    for row in matrix {
        if row.len() != n {
            return Err(AlgebraError::DimensionMismatch {
                expected: n,
                found: row.len(),
            });
        }
        flat.extend_from_slice(row);
    }

    Ok(flat)
}

pub(crate) fn symmetric_eigenvalues_with_backend<Rotation, DiagonalUpdate>(
    matrix: &[f64],
    n: usize,
    max_iterations: usize,
    tolerance: f64,
    rotation_kernel: Rotation,
    diagonal_update_kernel: DiagonalUpdate,
    backend_name: &str,
) -> AlgebraResult<Vec<f64>>
where
    Rotation: Fn(f64, f64, f64) -> (f64, f64),
    DiagonalUpdate: Fn(f64, f64, f64, f64, f64) -> (f64, f64),
{
    if n == 0 {
        return Ok(Vec::new());
    }

    if matrix.len() != n * n {
        return Err(AlgebraError::DimensionMismatch {
            expected: n * n,
            found: matrix.len(),
        });
    }

    let mut a = matrix.to_vec();
    let mut converged = false;

    for _ in 0..max_iterations {
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in (i + 1)..n {
                let v = a[i * n + j].abs();
                if v > max_val {
                    max_val = v;
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

        let (sin_t, cos_t) = if (app - aqq).abs() < DEGENERATE_DIAGONAL_THRESHOLD {
            let scale = std::f64::consts::FRAC_1_SQRT_2;
            (scale, scale)
        } else {
            rotation_kernel(app, aqq, apq)
        };

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

        let (new_pp, new_qq) = diagonal_update_kernel(sin_t, cos_t, app, apq, aqq);
        a[p * n + p] = new_pp;
        a[q * n + q] = new_qq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;
    }

    if !converged {
        return Err(AlgebraError::NumericalError(format!(
            "{backend_name} Jacobi iteration failed to converge after {max_iterations} steps"
        )));
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eigenvalues.sort_by(|lhs, rhs| compare_eigenvalue_abs_desc(*lhs, *rhs));
    Ok(eigenvalues)
}

fn compare_eigenvalue_abs_desc(lhs: f64, rhs: f64) -> std::cmp::Ordering {
    rhs.abs()
        .total_cmp(&lhs.abs())
        .then_with(|| rhs.total_cmp(&lhs))
}

#[cfg(test)]
mod tests {
    use super::{
        compare_eigenvalue_abs_desc, flatten_square_matrix, symmetric_eigenvalues_with_backend,
    };
    use std::cmp::Ordering;

    fn f64_rotation(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
        let theta = 0.5 * (2.0 * apq).atan2(app - aqq);
        theta.sin_cos()
    }

    fn f64_diagonal_update(sin_t: f64, cos_t: f64, app: f64, apq: f64, aqq: f64) -> (f64, f64) {
        (
            cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq,
            sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq,
        )
    }

    #[test]
    fn flatten_square_matrix_rejects_non_square_input() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0]];
        let err = flatten_square_matrix(&matrix).unwrap_err();
        match err {
            cd_kernel::error::AlgebraError::DimensionMismatch { expected, found } => {
                assert_eq!(expected, 2);
                assert_eq!(found, 1);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn shared_jacobi_scaffold_solves_diagonal_matrix() {
        let matrix = vec![vec![5.0, 0.0], vec![0.0, 3.0]];
        let flat = flatten_square_matrix(&matrix).unwrap();
        let eigs = symmetric_eigenvalues_with_backend(
            &flat,
            2,
            32,
            1.0e-14,
            f64_rotation,
            f64_diagonal_update,
            "f64-test",
        )
        .unwrap();

        assert_eq!(eigs, vec![5.0, 3.0]);
    }

    #[test]
    fn shared_jacobi_scaffold_recovers_known_spectrum() {
        let matrix = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let flat = flatten_square_matrix(&matrix).unwrap();
        let eigs = symmetric_eigenvalues_with_backend(
            &flat,
            3,
            256,
            1.0e-14,
            f64_rotation,
            f64_diagonal_update,
            "f64-test",
        )
        .unwrap();

        assert_eq!(eigs.len(), 3);
        assert!(eigs[0] >= eigs[1]);
    }

    #[test]
    fn eigenvalue_sort_order_is_total_for_nan_and_tied_magnitudes() {
        assert_eq!(compare_eigenvalue_abs_desc(2.0, -2.0), Ordering::Less);
        assert_eq!(compare_eigenvalue_abs_desc(-2.0, 2.0), Ordering::Greater);
        assert_eq!(compare_eigenvalue_abs_desc(f64::NAN, 1.0), Ordering::Less);
    }
}
