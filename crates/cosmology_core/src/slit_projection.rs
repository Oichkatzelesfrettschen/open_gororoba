//! Pseudo-slit PSF projection and Tikhonov de-projection for rotation curves.
//!
//! A pseudo-slit rotation curve (e.g., from an IFU or radio observation) is
//! not the intrinsic circular velocity: it is the projection of v_circ through
//! a Gaussian PSF beam and a finite slit aperture. This module provides:
//!
//! 1. `compute_projection_matrix`: builds the N x N PSF projection matrix M,
//!    where M[i][j] describes how emission at radius r[j] contributes to the
//!    observed velocity at radius r[i].
//!
//! 2. `deproject_rotation_curve`: Tikhonov-regularized inversion of M to
//!    recover the intrinsic circular velocity v_circ from v_obs.
//!
//! # References
//!
//! - Begeman (1989): ring-average de-projection for HI velocity fields
//! - Swaters et al. (2009): beam smearing correction for pseudo-slit data
//! - E-194: Slit de-projection implementation (this module)

use std::f64::consts::{LN_2, PI};

/// Build the N x N Gaussian PSF projection matrix for a pseudo-slit rotation curve.
///
/// `M[i][j]` gives the contribution of ring `j` (at radius `r_kpc[j]`) to the
/// observed velocity at radius `r_kpc[i]`, folded through the Gaussian PSF and
/// a first-order finite-slit aperture correction.
///
/// Formula:
/// ```text
/// sigma = psf_fwhm_kpc / (2 * sqrt(2 * ln(2)))
/// G_psf(r_i, r_j) = exp(-(r_i - r_j)^2 / (2 * sigma^2))
/// correction(r_j, incl, w) = 1 - (w / (r_j * sin(incl_rad)))^2 / 6
/// M[i][j] = G_psf * correction(r_j)
/// ```
/// Each row is L1-normalized so `sum_j M[i][j] = 1`.
///
/// # Arguments
/// - `r_kpc`: radial grid in kpc (must be non-empty and strictly positive)
/// - `psf_fwhm_kpc`: PSF FWHM in kpc
/// - `incl_deg`: disk inclination in degrees (0 = face-on, 90 = edge-on)
/// - `slit_half_width_kpc`: half-width of the slit aperture in kpc
pub fn compute_projection_matrix(
    r_kpc: &[f64],
    psf_fwhm_kpc: f64,
    incl_deg: f64,
    slit_half_width_kpc: f64,
) -> Vec<Vec<f64>> {
    let n = r_kpc.len();
    if n == 0 {
        return vec![];
    }

    // sigma of the Gaussian PSF
    let sigma = psf_fwhm_kpc / (2.0 * (2.0 * LN_2).sqrt());
    let two_sig2 = 2.0 * sigma * sigma;

    let incl_rad = incl_deg * PI / 180.0;
    let sin_incl = incl_rad.sin().max(1.0e-6); // avoid division by zero at face-on

    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    // Gaussian PSF smearing between rings i and j
                    let dr = r_kpc[i] - r_kpc[j];
                    let g_psf = (-dr * dr / two_sig2).exp();

                    // First-order finite-slit aperture correction (Swaters 2009 eq. 3)
                    let r_sin = r_kpc[j] * sin_incl;
                    let correction = if r_sin > slit_half_width_kpc {
                        let ratio = slit_half_width_kpc / r_sin;
                        1.0 - ratio * ratio / 6.0
                    } else {
                        // Slit much wider than ring: full coverage, correction ~ 1
                        1.0
                    };

                    g_psf * correction
                })
                .collect()
        })
        .collect();

    // L1-normalize each row
    for row in &mut m {
        let row_sum: f64 = row.iter().sum();
        if row_sum > 1.0e-30 {
            for v in row.iter_mut() {
                *v /= row_sum;
            }
        }
    }

    m
}

/// Tikhonov-regularized de-projection of a pseudo-slit rotation curve.
///
/// Solves `v_circ = (M^T M + lambda I)^{-1} M^T v_obs` via spectral decomposition
/// of the symmetric matrix `A = M^T M + lambda I`.
///
/// Also propagates the observational error covariance `diag(sigma_obs^2)` through
/// the inversion to give per-bin uncertainties on the recovered `v_circ`.
///
/// # Returns
/// `(v_circ, v_circ_err)` -- the de-projected circular velocity and its 1-sigma
/// uncertainty at each radial grid point.
///
/// # Arguments
/// - `v_obs`: observed velocities (length N)
/// - `sigma_obs`: observational uncertainties (length N, must be same length as v_obs)
/// - `projection_matrix`: N x N matrix from `compute_projection_matrix`
/// - `lambda`: Tikhonov regularization parameter (larger = smoother result)
pub fn deproject_rotation_curve(
    v_obs: &[f64],
    sigma_obs: &[f64],
    projection_matrix: &[Vec<f64>],
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = v_obs.len();
    if n == 0 || projection_matrix.is_empty() {
        return (vec![], vec![]);
    }

    // Step 1: A = M^T M  (N x N symmetric)
    let mut a_flat: Vec<f64> = vec![0.0; n * n];
    for row_k in projection_matrix.iter().take(n) {
        for i in 0..n {
            for j in 0..n {
                a_flat[i * n + j] += row_k[i] * row_k[j];
            }
        }
    }

    // Step 2: A[i,i] += lambda  (Tikhonov regularization)
    for i in 0..n {
        a_flat[i * n + i] += lambda;
    }

    // Step 3: eigendecompose A = V Lambda V^T
    let (eigenvalues, v_flat) = eigensystem_dispatch(&a_flat, n);

    // Step 4: A_inv = V diag(1/lambda_k) V^T
    // Step 5: b = M^T v_obs
    let mut b: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for k in 0..n {
            b[i] += projection_matrix[k][i] * v_obs[k];
        }
    }

    // Step 6: v_circ = A_inv * b  = V diag(1/lk) V^T b
    // Compute alpha_k = (V^T b)[k] / lambda_k
    let vt_b: Vec<f64> = (0..n)
        .map(|k| (0..n).map(|i| v_flat[i * n + k] * b[i]).sum::<f64>())
        .collect();

    // Regularized inverse: skip near-zero eigenvalues
    let lk_min = eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .abs()
        * 1.0e-10;

    let alpha: Vec<f64> = eigenvalues
        .iter()
        .zip(vt_b.iter())
        .map(
            |(&lk, &vt_b_k)| {
                if lk.abs() > lk_min { vt_b_k / lk } else { 0.0 }
            },
        )
        .collect();

    let v_circ: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|k| v_flat[i * n + k] * alpha[k]).sum::<f64>())
        .collect();

    // Step 7: error propagation  sigma_circ^2[i] = sum_j (A_inv M^T)^2[i,j] * sigma_obs^2[j]
    // (A_inv M^T)[i,j] = sum_k v[i,k] * (1/lk) * (M^T)[k,j]
    //                  = sum_k v[i,k] * (1/lk) * M[j,k]   (M is row-indexed as M[row][col])
    let v_circ_err: Vec<f64> = (0..n)
        .map(|i| {
            let sigma2: f64 = (0..n)
                .map(|j| {
                    // (A_inv M^T)[i,j]
                    let coeff: f64 = (0..n)
                        .map(|k| {
                            let inv_lk = if eigenvalues[k].abs() > lk_min {
                                1.0 / eigenvalues[k]
                            } else {
                                0.0
                            };
                            // M^T[k,j] = M[j][k]
                            v_flat[i * n + k] * inv_lk * projection_matrix[j][k]
                        })
                        .sum::<f64>();
                    coeff * coeff * sigma_obs[j] * sigma_obs[j]
                })
                .sum();
            sigma2.sqrt()
        })
        .collect();

    (v_circ, v_circ_err)
}

/// Dispatch to the appropriate eigensystem backend based on target architecture.
///
/// On x86_64, uses the x87 80-bit Jacobi solver for maximum precision.
/// On other targets, falls back to the double-double software lane.
fn eigensystem_dispatch(a_flat: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let max_iter = 100 * n * n;

    #[cfg(target_arch = "x86_64")]
    {
        use algebra_analysis::x87_jacobi::symmetric_eigensystem_x87;
        match symmetric_eigensystem_x87(a_flat, n, max_iter, 1.0e-12) {
            Ok(result) => result,
            Err(_) => {
                // Fallback: return diagonal approximation
                let eigenvalues = (0..n).map(|i| a_flat[i * n + i]).collect();
                let mut v = vec![0.0_f64; n * n];
                for i in 0..n {
                    v[i * n + i] = 1.0;
                }
                (eigenvalues, v)
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        use algebra_analysis::dd_jacobi::symmetric_eigensystem_dd;
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| a_flat[i * n..(i + 1) * n].to_vec())
            .collect();
        match symmetric_eigensystem_dd(&matrix) {
            Ok(result) => result,
            Err(_) => {
                let eigenvalues = (0..n).map(|i| a_flat[i * n + i]).collect();
                let mut v = vec![0.0_f64; n * n];
                for i in 0..n {
                    v[i * n + i] = 1.0;
                }
                (eigenvalues, v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_projection_matrix, deproject_rotation_curve};

    #[test]
    fn test_projection_narrow_psf_near_identity() {
        // A very narrow PSF (1 micro-kpc) should give M ~ I
        let r_kpc: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let m = compute_projection_matrix(&r_kpc, 1.0e-6, 60.0, 0.5);
        for (i, row) in m.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (*value - expected).abs();
                assert!(err < 1.0e-5, "M[{i}][{j}] = {}, expected {expected}", value);
            }
        }
    }

    #[test]
    fn test_deproject_recovers_linear_curve() {
        // v_circ = 100 + 20*r km/s; forward-project and deproject
        let r_kpc: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let n = r_kpc.len();
        let v_true: Vec<f64> = r_kpc.iter().map(|&r| 100.0 + 20.0 * r).collect();
        let sigma_obs = vec![1.0_f64; n];

        // Build a mild PSF projection
        let m = compute_projection_matrix(&r_kpc, 0.5, 60.0, 0.3);

        // Forward project: v_obs = M * v_true
        let v_obs: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| m[i][j] * v_true[j]).sum::<f64>())
            .collect();

        let (v_circ, _) = deproject_rotation_curve(&v_obs, &sigma_obs, &m, 0.01);

        // Check relative error < 5%
        for (k, (&v_t, &v_c)) in v_true.iter().zip(v_circ.iter()).enumerate() {
            let rel_err = (v_c - v_t).abs() / v_t;
            assert!(
                rel_err < 0.05,
                "bin {k}: v_true={v_t:.1}, v_circ={v_c:.1}, rel_err={rel_err:.3}"
            );
        }
    }

    #[test]
    fn test_deproject_handles_flat_curve() {
        // Constant v_circ = 150 km/s; deprojected result should be ~flat
        let r_kpc: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let n = r_kpc.len();
        let v_true = vec![150.0_f64; n];
        let sigma_obs = vec![1.0_f64; n];

        let m = compute_projection_matrix(&r_kpc, 0.5, 60.0, 0.3);
        let v_obs: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| m[i][j] * v_true[j]).sum::<f64>())
            .collect();

        let (v_circ, _) = deproject_rotation_curve(&v_obs, &sigma_obs, &m, 0.01);

        // (max - min) / mean < 2%
        let mean = v_circ.iter().sum::<f64>() / n as f64;
        let max = v_circ.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = v_circ.iter().copied().fold(f64::INFINITY, f64::min);
        let variation = (max - min) / mean;
        assert!(
            variation < 0.02,
            "flat curve variation = {variation:.4} (max={max:.2}, min={min:.2}, mean={mean:.2})"
        );
    }
}
