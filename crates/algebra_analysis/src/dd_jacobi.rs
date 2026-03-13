//! Double-double precision Jacobi eigenvalue solver (non-x86_64 fallback).
//!
//! Uses the DD arithmetic from `double_double.rs` (~31 decimal digits) for
//! Givens rotation computations. This is the software fallback lane when the
//! x87-specific backend is unavailable and is intended for the small matrices
//! used in obstruction spectrum analysis.
//!
//! This module is always compiled (no cfg gate) so that its tests run on
//! x86_64 CI as well. The cfg gate is on the dispatch in homotopy_algebra.rs.

use crate::{
    double_double::{DD, dd_mul_add_pair},
    jacobi_shared::{flatten_square_matrix, recommended_jacobi_iterations},
};
use cd_kernel::error::{AlgebraError, AlgebraResult};

const DEGENERATE_DIAGONAL_THRESHOLD: f64 = 1.0e-15;

/// Compute eigenvalues of a symmetric matrix using Jacobi iteration with
/// double-double precision for Givens rotations.
///
/// Uses a dedicated DD matrix iteration so the evolving state stays in the
/// software extended-precision lane instead of collapsing back to f64 after
/// every rotation.
#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
pub fn symmetric_eigenvalues_dd(matrix: &[Vec<f64>]) -> AlgebraResult<Vec<f64>> {
    let n = matrix.len();
    let flat = flatten_square_matrix(matrix)?;
    symmetric_eigenvalues_dd_flat(&flat, n, recommended_jacobi_iterations(n), 1.0e-14)
}

#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
fn symmetric_eigenvalues_dd_flat(
    matrix: &[f64],
    n: usize,
    max_iterations: usize,
    tolerance: f64,
) -> AlgebraResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    if matrix.len() != n * n {
        return Err(AlgebraError::DimensionMismatch {
            expected: n * n,
            found: matrix.len(),
        });
    }

    let mut a: Vec<DD> = matrix.iter().copied().map(DD::from_f64).collect();
    let mut converged = false;

    for _ in 0..max_iterations {
        let (max_val, p, q) = dd_find_pivot(&a, n);

        if max_val < tolerance {
            converged = true;
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let (sin_t, cos_t) = if app.sub(aqq).abs().to_f64() < DEGENERATE_DIAGONAL_THRESHOLD {
            let scale = DD::from_f64(std::f64::consts::FRAC_1_SQRT_2);
            (scale, scale)
        } else {
            dd_rotation(app, aqq, apq)
        };

        dd_apply_plane_rotation(&mut a, n, p, q, sin_t, cos_t);

        let (new_pp, new_qq) = dd_diagonal_update(sin_t, cos_t, app, apq, aqq);
        a[p * n + p] = new_pp;
        a[q * n + q] = new_qq;
        a[p * n + q] = DD::ZERO;
        a[q * n + p] = DD::ZERO;
    }

    if !converged {
        return Err(AlgebraError::NumericalError(format!(
            "DD Jacobi iteration failed to converge after {max_iterations} steps"
        )));
    }

    Ok(dd_collect_sorted_eigenvalues(&a, n))
}

#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
fn dd_find_pivot(a: &[DD], n: usize) -> (f64, usize, usize) {
    let mut max_hi = 0.0_f64;
    let mut max_lo = 0.0_f64;
    let mut p = 0;
    let mut q = 1;

    for i in 0..n {
        for j in (i + 1)..n {
            let entry = a[i * n + j];
            let hi = entry.hi.abs();
            let lo = entry.lo.abs();
            if hi > max_hi || (hi == max_hi && lo > max_lo) {
                max_hi = hi;
                max_lo = lo;
                p = i;
                q = j;
            }
        }
    }

    (max_hi + max_lo, p, q)
}

#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
fn dd_rotation(app: DD, aqq: DD, apq: DD) -> (DD, DD) {
    if apq.abs().to_f64() < f64::EPSILON {
        return (DD::ZERO, DD::ONE);
    }

    // Stable Jacobi rotation formula matching the shared scaffold's
    // angle convention `theta = 0.5 * atan2(2*apq, app - aqq)`.
    // tau = (app - aqq) / (2 * apq)
    // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
    let tau = app.sub(aqq).div(DD::from_f64(2.0).mul(apq));
    let root = DD::ONE.add(tau.mul(tau)).sqrt();
    let t_mag = DD::ONE.div(tau.abs().add(root));
    let t = if tau.hi < 0.0 || (tau.hi == 0.0 && tau.lo < 0.0) {
        DD {
            hi: -t_mag.hi,
            lo: -t_mag.lo,
        }
    } else {
        t_mag
    };
    let cos_t = DD::ONE.div(DD::ONE.add(t.mul(t)).sqrt());
    let sin_t = t.mul(cos_t);
    (sin_t, cos_t)
}

#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
fn dd_apply_plane_rotation(a: &mut [DD], n: usize, p: usize, q: usize, sin_t: DD, cos_t: DD) {
    let neg_sin_t = DD {
        hi: -sin_t.hi,
        lo: -sin_t.lo,
    };

    for i in 0..n {
        if i == p || i == q {
            continue;
        }

        let api = a[i * n + p];
        let aqi = a[i * n + q];
        let new_pi = dd_mul_add_pair(cos_t, api, sin_t, aqi);
        let new_qi = dd_mul_add_pair(cos_t, aqi, neg_sin_t, api);

        a[i * n + p] = new_pi;
        a[p * n + i] = new_pi;
        a[i * n + q] = new_qi;
        a[q * n + i] = new_qi;
    }
}

#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
fn dd_diagonal_update(sin_t: DD, cos_t: DD, app: DD, apq: DD, aqq: DD) -> (DD, DD) {
    let c2 = cos_t.mul(cos_t);
    let s2 = sin_t.mul(sin_t);
    let two_sc = DD::from_f64(2.0).mul(sin_t).mul(cos_t);

    (
        c2.mul(app).add(two_sc.mul(apq)).add(s2.mul(aqq)),
        s2.mul(app).sub(two_sc.mul(apq)).add(c2.mul(aqq)),
    )
}

/// Apply a Givens rotation to the eigenvector accumulator V.
///
/// Updates columns p and q for ALL n rows (no row skipping).
/// This accumulates V_new = V * G where G is the Givens rotation matrix.
#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
fn dd_apply_eigenvec_rotation(v: &mut [DD], n: usize, p: usize, q: usize, sin_t: DD, cos_t: DD) {
    let neg_sin = DD {
        hi: -sin_t.hi,
        lo: -sin_t.lo,
    };
    for i in 0..n {
        let vpi = v[i * n + p];
        let vqi = v[i * n + q];
        // new_p = cos_t * vpi + sin_t * vqi
        v[i * n + p] = dd_mul_add_pair(cos_t, vpi, sin_t, vqi);
        // new_q = cos_t * vqi + (-sin_t) * vpi
        v[i * n + q] = dd_mul_add_pair(cos_t, vqi, neg_sin, vpi);
    }
}

/// Full symmetric Jacobi eigensystem with double-double precision.
///
/// Returns `(eigenvalues, v_flat)` sorted by descending |eigenvalue|.
/// `v_flat` is the row-major eigenvector matrix (converted to f64):
/// column k is the eigenvector for `eigenvalues[k]`.
pub fn symmetric_eigensystem_dd(matrix: &[Vec<f64>]) -> AlgebraResult<(Vec<f64>, Vec<f64>)> {
    let n = matrix.len();
    let flat = flatten_square_matrix(matrix)?;
    symmetric_eigensystem_dd_flat(&flat, n, recommended_jacobi_iterations(n), 1.0e-14)
}

#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
fn symmetric_eigensystem_dd_flat(
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

    let mut a: Vec<DD> = matrix.iter().copied().map(DD::from_f64).collect();
    // V starts as the identity in DD precision
    let mut v: Vec<DD> = (0..n * n)
        .map(|idx| if idx / n == idx % n { DD::ONE } else { DD::ZERO })
        .collect();
    let mut converged = false;

    for _ in 0..max_iterations {
        let (max_val, p, q) = dd_find_pivot(&a, n);

        if max_val < tolerance {
            converged = true;
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let (sin_t, cos_t) = if app.sub(aqq).abs().to_f64() < DEGENERATE_DIAGONAL_THRESHOLD {
            let scale = DD::from_f64(std::f64::consts::FRAC_1_SQRT_2);
            (scale, scale)
        } else {
            dd_rotation(app, aqq, apq)
        };

        dd_apply_plane_rotation(&mut a, n, p, q, sin_t, cos_t);
        dd_apply_eigenvec_rotation(&mut v, n, p, q, sin_t, cos_t);

        let (new_pp, new_qq) = dd_diagonal_update(sin_t, cos_t, app, apq, aqq);
        a[p * n + p] = new_pp;
        a[q * n + q] = new_qq;
        a[p * n + q] = DD::ZERO;
        a[q * n + p] = DD::ZERO;
    }

    if !converged {
        return Err(AlgebraError::NumericalError(format!(
            "DD Jacobi eigensystem failed to converge after {max_iterations} steps"
        )));
    }

    // Sort by descending |eigenvalue|
    let mut indexed: Vec<(usize, f64)> = (0..n).map(|i| (i, a[i * n + i].to_f64())).collect();
    indexed.sort_by(|(_, l), (_, r)| r.abs().total_cmp(&l.abs()).then_with(|| r.total_cmp(l)));

    let eigenvalues: Vec<f64> = indexed.iter().map(|(_, ev)| *ev).collect();
    // Permute V columns and convert to f64: output column j = input column indexed[j].0
    let v_out: Vec<f64> = (0..n * n)
        .map(|idx| v[(idx / n) * n + indexed[idx % n].0].to_f64())
        .collect();

    Ok((eigenvalues, v_out))
}

#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
fn dd_collect_sorted_eigenvalues(a: &[DD], n: usize) -> Vec<f64> {
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i].to_f64()).collect();
    sort_by_abs_desc(&mut eigenvalues);
    eigenvalues
}

#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
fn sort_by_abs_desc(values: &mut [f64]) {
    values.sort_by(|lhs, rhs| {
        rhs.abs()
            .total_cmp(&lhs.abs())
            .then_with(|| rhs.total_cmp(lhs))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dd_find_pivot_prefers_largest_off_diagonal_magnitude() {
        let n = 3;
        let matrix = [
            DD::from_f64(0.0),
            DD::from_f64(1.0),
            DD::from_f64(-3.0),
            DD::from_f64(1.0),
            DD::from_f64(0.0),
            DD::from_f64(2.0),
            DD::from_f64(-3.0),
            DD::from_f64(2.0),
            DD::from_f64(0.0),
        ];

        let (max_val, p, q) = dd_find_pivot(&matrix, n);
        assert!((max_val - 3.0).abs() < 1e-14);
        assert_eq!((p, q), (0, 2));
    }

    #[test]
    fn test_dd_jacobi_diagonal_matrix() {
        let diag = vec![
            vec![5.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let eigs = symmetric_eigenvalues_dd(&diag).unwrap();
        assert_eq!(eigs.len(), 3);
        assert!((eigs[0] - 5.0).abs() < 1e-14);
        assert!((eigs[1] - 3.0).abs() < 1e-14);
        assert!((eigs[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_dd_jacobi_known_spectrum() {
        let n = 4;
        let diag = [500.0, 100.0, 0.0, 0.0];
        let angle = std::f64::consts::FRAC_PI_4;
        let (s, c) = angle.sin_cos();

        let mut r = vec![vec![0.0; n]; n];
        r[0][0] = c;
        r[0][1] = -s;
        r[1][0] = s;
        r[1][1] = c;
        r[2][2] = 1.0;
        r[3][3] = 1.0;

        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i][k] * diag[k] * r[j][k];
                }
                matrix[i][j] = sum;
            }
        }

        let eigs = symmetric_eigenvalues_dd(&matrix).unwrap();
        assert!((eigs[0] - 500.0).abs() < 1e-10);
        assert!((eigs[1] - 100.0).abs() < 1e-10);
        assert!(eigs[2].abs() < 1e-13, "null eig[2] = {}", eigs[2]);
        assert!(eigs[3].abs() < 1e-13, "null eig[3] = {}", eigs[3]);
    }

    #[test]
    fn test_dd_jacobi_vs_nalgebra() {
        let matrix = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let eigs_dd = symmetric_eigenvalues_dd(&matrix).unwrap();

        let na_mat = nalgebra::DMatrix::from_fn(3, 3, |i, j| matrix[i][j]);
        let na_eig = na_mat.symmetric_eigen();
        let mut na_eigs: Vec<f64> = na_eig.eigenvalues.iter().copied().collect();
        sort_by_abs_desc(&mut na_eigs);

        for (dd_e, na_e) in eigs_dd.iter().zip(na_eigs.iter()) {
            assert!((dd_e - na_e).abs() < 1e-12, "dd={dd_e}, nalgebra={na_e}");
        }
    }

    fn obstruction_plateau_matrix(size: usize) -> Vec<Vec<f64>> {
        let bulk = (size as f64) * 0.8;
        let tail = (size as f64) * 0.16;
        let mut diagonal = Vec::with_capacity(size);
        diagonal.push((size as f64) * 3.1);
        if size > 1 {
            diagonal.push((size as f64) * 0.9);
        }
        while diagonal.len() + 1 < size {
            if diagonal.len() < 2 + size / 2 {
                diagonal.push(-bulk);
            } else {
                diagonal.push(-tail);
            }
        }
        while diagonal.len() < size {
            diagonal.push(0.0);
        }

        let n = diagonal.len();
        let inv_n = 1.0 / n as f64;
        let mut matrix = vec![vec![0.0; n]; n];
        for (i, row) in matrix.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let mut value = 0.0;
                for (k, &lambda) in diagonal.iter().enumerate() {
                    let h_ik = if i == k { 1.0 } else { 0.0 } - 2.0 * inv_n;
                    let h_jk = if j == k { 1.0 } else { 0.0 } - 2.0 * inv_n;
                    value += h_ik * lambda * h_jk;
                }
                *cell = value;
            }
        }
        matrix
    }

    fn quantized_obstruction_graph(size: usize) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; size]; size];
        for i in 0..size {
            let (head, tail) = matrix.split_at_mut(i + 1);
            let row = &mut head[i];
            for (offset, other_row) in tail.iter_mut().enumerate() {
                let j = i + 1 + offset;
                let value = if i == 0 || j == 0 {
                    0.0
                } else if i == 1 {
                    if j.is_multiple_of(2) { 112.0 } else { 80.0 }
                } else if i == 2 {
                    if j.is_multiple_of(3) { 80.0 } else { 48.0 }
                } else {
                    let bucket = (i + j) % 4;
                    16.0 * (2 * bucket + 1) as f64
                };
                row[j] = value;
                other_row[i] = value;
            }
        }
        matrix
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
    fn test_eigensystem_dd_2x2_known() {
        let matrix = vec![vec![3.0_f64, 1.0], vec![1.0, 3.0]];
        let (eigs, v) = symmetric_eigensystem_dd(&matrix).unwrap();

        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] - 4.0).abs() < 1.0e-13, "lambda_0={}", eigs[0]);
        assert!((eigs[1] - 2.0).abs() < 1.0e-13, "lambda_1={}", eigs[1]);

        assert!(max_off_diagonal_vtv(&v, 2) < 1.0e-13, "orthogonality violated");

        // Reconstruct
        let flat = [3.0_f64, 1.0, 1.0, 3.0];
        let a_rec: Vec<f64> = (0..4)
            .map(|idx| {
                let i = idx / 2;
                let j = idx % 2;
                (0..2).map(|k| v[i * 2 + k] * eigs[k] * v[j * 2 + k]).sum::<f64>()
            })
            .collect();
        for (orig, rec) in flat.iter().zip(a_rec.iter()) {
            assert!((orig - rec).abs() < 1.0e-13);
        }
    }

    #[test]
    fn test_eigensystem_dd_orthonormal() {
        let matrix = vec![
            vec![4.0_f64, 1.2, 0.3, -0.5],
            vec![1.2, 3.0, 0.8, 0.1],
            vec![0.3, 0.8, 2.5, -0.4],
            vec![-0.5, 0.1, -0.4, 1.0],
        ];
        let (_, v) = symmetric_eigensystem_dd(&matrix).unwrap();
        let max_err = max_off_diagonal_vtv(&v, 4);
        assert!(max_err < 1.0e-13, "max V^T V off-diagonal = {max_err}");
    }

    #[test]
    fn test_eigensystem_dd_reconstruct() {
        // 3x3 Hilbert matrix
        let n = 3_usize;
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| 1.0 / (i + j + 1) as f64).collect())
            .collect();
        let flat: Vec<f64> = matrix.iter().flat_map(|r| r.iter().copied()).collect();
        let (eigs, v) = symmetric_eigensystem_dd(&matrix).unwrap();

        let a_rec: Vec<f64> = (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                (0..n).map(|k| v[i * n + k] * eigs[k] * v[j * n + k]).sum::<f64>()
            })
            .collect();
        let max_err = flat
            .iter()
            .zip(a_rec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1.0e-13, "Hilbert reconstruction error = {max_err}");
    }

    fn nalgebra_sorted_eigenvalues(matrix: &[Vec<f64>]) -> Vec<f64> {
        let dense = nalgebra::DMatrix::from_fn(matrix.len(), matrix.len(), |i, j| matrix[i][j]);
        let eig = dense.symmetric_eigen();
        let mut eigs: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        sort_by_abs_desc(&mut eigs);
        eigs
    }

    fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn test_dd_jacobi_obstruction_plateau_regression() {
        let matrix = obstruction_plateau_matrix(32);
        let eigs_dd = symmetric_eigenvalues_dd(&matrix).unwrap();
        let eigs_ref = nalgebra_sorted_eigenvalues(&matrix);

        assert!(max_abs_diff(&eigs_dd, &eigs_ref) < 1.0e-8);
    }

    #[test]
    fn test_dd_jacobi_quantized_obstruction_regression() {
        let matrix = quantized_obstruction_graph(32);
        let eigs_dd = symmetric_eigenvalues_dd(&matrix).unwrap();
        let eigs_ref = nalgebra_sorted_eigenvalues(&matrix);

        assert!(max_abs_diff(&eigs_dd, &eigs_ref) < 1.0e-8);
    }
}
