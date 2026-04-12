//! Pseudospectrum computation for fractional Laplacian operators.
//!
//! A pseudospectrum visualizes how much a matrix deviates from normality by
//! mapping the minimum singular value of `(zI - A)` over a grid of complex
//! points `z`.  Regions where this minimum is small indicate that `z` is
//! "nearly" an eigenvalue even when it is not exactly one.
//!
//! # Algorithm
//!
//! Given operator size `n` and a set of (power, coefficient) pairs defining
//!   A = sum_k  coeffs[k] * L^{powers[k]}
//! where L is the n x n Dirichlet Laplacian, this module:
//!   1. Builds L numerically (tridiagonal, Dirichlet BCs).
//!   2. Computes the symmetric eigendecomposition L = U diag(w) U^T via faer.
//!   3. Constructs each fractional power L^s = U diag(w^s) U^T.
//!   4. Assembles A from the weighted sum.
//!   5. Sweeps a (n_re x n_im) complex grid, building (zI - A) as a complex
//!      matrix and computing its smallest singular value via faer SVD.
//!   6. Returns log10(sigma_min) at each grid point.
//!
//! CLI binaries are responsible only for rendering the returned grid (colors,
//! image encoding) and are free of all `faer` dependencies.

use faer::{Mat, Side, c64};

/// Output of a pseudospectrum sweep.
pub struct PseudospectrumResult {
    /// log10(sigma_min(zI - A)) at each grid point.
    /// Indexed as `log_smin[im_idx][re_idx]`.
    pub log_smin: Vec<Vec<f64>>,
    /// Real-axis sample points (length n_re).
    pub re_grid: Vec<f64>,
    /// Imaginary-axis sample points (length n_im).
    pub im_grid: Vec<f64>,
}

/// Compute the pseudospectrum of a fractional Laplacian composite operator.
///
/// # Parameters
/// - `n`: Interior dimension of the 1-D Dirichlet Laplacian (mesh size).
/// - `powers_coeffs`: Slice of `(s, c)` pairs.  The operator is
///   `A = sum_k c_k * L^{s_k}`.  The effective coefficient for power `s_k` is
///   multiplied by `lambda^k` where the exponent is the slice index (0, 1, 2,
///   ...).  Pass pre-scaled coefficients if you need a different weighting.
/// - `lambda`: Scaling parameter applied as `c_k * lambda^k` to the k-th term.
/// - `re_range`: `(min, max, n_re)` -- real-axis grid.
/// - `im_range`: `(min, max, n_im)` -- imaginary-axis grid.
pub fn fractional_laplacian_pseudospectrum(
    n: usize,
    powers_coeffs: &[(f64, f64)],
    lambda: f64,
    re_range: (f64, f64, usize),
    im_range: (f64, f64, usize),
) -> PseudospectrumResult {
    // -----------------------------------------------------------------
    // 1. Build the n x n Dirichlet Laplacian (tridiagonal).
    // -----------------------------------------------------------------
    let h = 1.0 / (n as f64 + 1.0);
    let diag_val = 2.0 / (h * h);
    let off_val = -1.0 / (h * h);

    let mut l_mat = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        l_mat[(i, i)] = diag_val;
        if i > 0 {
            l_mat[(i, i - 1)] = off_val;
        }
        if i < n - 1 {
            l_mat[(i, i + 1)] = off_val;
        }
    }

    // -----------------------------------------------------------------
    // 2. Symmetric eigen-decomposition L = U diag(w) U^T.
    // -----------------------------------------------------------------
    let evd = l_mat.self_adjoint_eigen(Side::Lower).unwrap();
    let s_vec = evd.S();
    let u = evd.U();

    // Clamp eigenvalues to be non-negative (numerical safety).
    let w_vals: Vec<f64> = (0..n).map(|i| s_vec.column_vector()[i].max(0.0)).collect();

    // -----------------------------------------------------------------
    // 3 & 4. Build composite operator A = sum_k (c_k * lambda^k) * L^{s_k}.
    // -----------------------------------------------------------------
    let mut a_mat = Mat::<f64>::zeros(n, n);
    for (k, &(power, base_coeff)) in powers_coeffs.iter().enumerate() {
        let coeff = base_coeff * lambda.powi(k as i32);
        // L^power = U * diag(w^power) * U^T  (add into a_mat in-place)
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..n {
                    sum += u[(i, kk)] * w_vals[kk].powf(power) * u[(j, kk)];
                }
                a_mat[(i, j)] += coeff * sum;
            }
        }
    }

    // -----------------------------------------------------------------
    // 5. Sweep the complex grid.
    // -----------------------------------------------------------------
    let (re_min, re_max, n_re) = re_range;
    let (im_min, im_max, n_im) = im_range;

    let re_grid: Vec<f64> = (0..n_re)
        .map(|j| re_min + (re_max - re_min) * (j as f64) / ((n_re - 1).max(1) as f64))
        .collect();
    let im_grid: Vec<f64> = (0..n_im)
        .map(|i| im_min + (im_max - im_min) * (i as f64) / ((n_im - 1).max(1) as f64))
        .collect();

    let mut log_smin = vec![vec![0.0_f64; n_re]; n_im];

    for (i, &im_z) in im_grid.iter().enumerate() {
        for (j, &re_z) in re_grid.iter().enumerate() {
            // Build complex matrix (zI - A).
            let mut a_c = Mat::<c64>::zeros(n, n);
            for r in 0..n {
                for c in 0..n {
                    let re_part = if r == c {
                        re_z - a_mat[(r, c)]
                    } else {
                        -a_mat[(r, c)]
                    };
                    let im_part = if r == c { im_z } else { 0.0 };
                    a_c[(r, c)] = c64::new(re_part, im_part);
                }
            }

            let svd = a_c.svd().unwrap();
            let sing_vals = svd.S();
            let s_min = (0..n).map(|k| sing_vals[k].re).fold(f64::INFINITY, f64::min);

            log_smin[i][j] = (s_min + 1e-14).log10();
        }
    }

    PseudospectrumResult { log_smin, re_grid, im_grid }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pseudospectrum_shape() {
        let result = fractional_laplacian_pseudospectrum(
            10,
            &[(1.0, 1.0), (0.7, 0.5), (0.4, 0.3)],
            0.7,
            (0.0, 20.0, 5),
            (-5.0, 5.0, 5),
        );
        assert_eq!(result.log_smin.len(), 5);
        assert_eq!(result.log_smin[0].len(), 5);
        assert_eq!(result.re_grid.len(), 5);
        assert_eq!(result.im_grid.len(), 5);
        // All values must be finite log10 of a small positive number.
        for row in &result.log_smin {
            for &v in row {
                assert!(v.is_finite(), "log_smin must be finite");
            }
        }
    }
}
