//! MaNGA IFU analysis functions: WLS baselines, harmonic stacking, multi-algebra DFT.
//!
//! These routines operate on `MangaDataBundle` from `manga_sim` and use ndarray
//! for intermediate computations. The CLI thin-wrapper `manga_zd_null` delegates
//! all analysis here; no ndarray types cross the CLI crate boundary.

use crate::manga_sim::MangaDataBundle;
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use std::f64::consts::PI;

/// Result of a rotation-curve fit or signal analysis.
pub struct MangaFitResult {
    pub snr: f64,
    pub residuals: Array2<f64>,
}

/// Single-amplitude NFW fit via Weighted Least Squares.
pub fn run_nfw_baseline(data: &MangaDataBundle) -> MangaFitResult {
    let n = data.v_los.nrows();
    let n_bins = data.v_los.ncols();
    let t = &data.nfw_template;
    let mut residuals = Array2::zeros((n, n_bins));

    for i in 0..n {
        let y = data.v_corrected.row(i);
        let err = data.err_corrected.row(i);
        let w = err.mapv(|e| 1.0 / (e * e + 1e-20));

        let num = (y.to_owned() * t * &w).sum();
        let den = (t * t * &w).sum();
        let a = num / (den + 1e-20);

        let model = t * a;
        let scale = y.mapv(|v| v.abs()).mean().unwrap_or(1.0).max(1e-6);
        for j in 0..n_bins {
            residuals[[i, j]] = (y[j] - model[j]) / scale;
        }
    }

    let snr = compute_stacked_snr(&residuals);
    MangaFitResult { snr, residuals }
}

/// 2-parameter (NFW + Baryonic) linear WLS fit.
pub fn run_baryonic_baseline(data: &MangaDataBundle) -> MangaFitResult {
    let n = data.v_los.nrows();
    let n_bins = data.v_los.ncols();
    let t0 = &data.nfw_template;
    let t1 = &data.baryonic_template;
    let mut residuals = Array2::zeros((n, n_bins));

    for i in 0..n {
        let y = data.v_corrected.row(i);
        let err = data.err_corrected.row(i);
        let w = err.mapv(|e| 1.0 / (e * e + 1e-20));

        let a00 = (t0 * t0 * &w).sum();
        let a01 = (t0 * t1 * &w).sum();
        let a11 = (t1 * t1 * &w).sum();
        let b0 = (t0 * &y * &w).sum();
        let b1 = (t1 * &y * &w).sum();

        let det = (a00 * a11 - a01 * a01).max(1e-20);
        let a_nfw = ((a11 * b0 - a01 * b1) / det).max(0.0);
        let a_bar = ((a00 * b1 - a01 * b0) / det).max(0.0);

        let model = t0 * a_nfw + t1 * a_bar;
        let scale = y.mapv(|v| v.abs()).mean().unwrap_or(1.0).max(1e-6);
        for j in 0..n_bins {
            residuals[[i, j]] = (y[j] - model[j]) / scale;
        }
    }

    let snr = compute_stacked_snr(&residuals);
    MangaFitResult { snr, residuals }
}

/// Compute binwise t-statistic on stacked residuals.
pub fn compute_stacked_snr(residuals: &Array2<f64>) -> f64 {
    let n = residuals.nrows() as f64;
    let mean_r = residuals.mean_axis(Axis(0)).unwrap();
    let mut max_snr = 0.0_f64;

    for j in 0..residuals.ncols() {
        let col = residuals.column(j);
        let var = col.fold(0.0, |acc, &x| acc + (x - mean_r[j]).powi(2)) / (n - 1.0).max(1.0);
        let sem = (var / n).sqrt().max(1e-20);
        let t = mean_r[j].abs() / sem;
        if t > max_snr {
            max_snr = t;
        }
    }
    max_snr
}

/// Orthonormal harmonic mode projection with bootstrap uncertainty.
pub fn run_harmonic_stacking(data: &MangaDataBundle, n_boot: usize) -> MangaFitResult {
    let res_obj = run_baryonic_baseline(data);
    let residuals = &res_obj.residuals;
    let x_grid = &data.x_grid;
    let x_max = data.x_grid.fold(0.0_f64, |a, &b| a.max(b));

    // D=16 modes (k=1..4)
    let n_modes = 4;
    let mut basis = Array2::<f64>::zeros((n_modes, x_grid.len()));
    for k in 0..n_modes {
        basis
            .row_mut(k)
            .assign(&x_grid.mapv(|x| (2.0 * PI * (k + 1) as f64 * x / x_max).cos()));
    }
    basis = gram_schmidt_ortho(basis);

    let mean_res = residuals.mean_axis(Axis(0)).unwrap();
    let coeffs = basis.dot(&mean_res);

    let mut rng = StdRng::seed_from_u64(137);
    let mut boot_coeffs = Array2::<f64>::zeros((n_boot, n_modes));
    for b in 0..n_boot {
        let mut b_sum = Array1::<f64>::zeros(x_grid.len());
        for _ in 0..residuals.nrows() {
            let idx = rng.random_range(0..residuals.nrows());
            b_sum += &residuals.row(idx);
        }
        let b_mean = b_sum / residuals.nrows() as f64;
        boot_coeffs.row_mut(b).assign(&basis.dot(&b_mean));
    }

    let mut max_snr = 0.0_f64;
    for k in 0..n_modes {
        let col = boot_coeffs.column(k);
        let std = col.std(0.0).max(1e-20);
        let t = coeffs[k].abs() / std;
        if t > max_snr {
            max_snr = t;
        }
    }

    MangaFitResult {
        snr: max_snr,
        residuals: residuals.clone(),
    }
}

/// Multi-Algebra restricted DFT analysis (G2 modes k=1..7).
pub fn run_multi_algebra_dft(data: &MangaDataBundle, n_boot: usize) -> MangaFitResult {
    let res_obj = run_baryonic_baseline(data);
    let residuals = &res_obj.residuals;
    let n_bins = residuals.ncols();

    let g2_modes = vec![1, 2, 3, 4, 5, 6, 7];

    let mean_res = residuals.mean_axis(Axis(0)).unwrap();
    let fft_obs = compute_rfft_abs(&mean_res);

    let mut rng = StdRng::seed_from_u64(777);
    let n_freq = fft_obs.len();
    let mut boot_amps = Array2::<f64>::zeros((n_boot, n_freq));
    for b in 0..n_boot {
        let mut b_sum = Array1::<f64>::zeros(n_bins);
        for _ in 0..residuals.nrows() {
            let idx = rng.random_range(0..residuals.nrows());
            b_sum += &residuals.row(idx);
        }
        let b_mean = b_sum / residuals.nrows() as f64;
        boot_amps.row_mut(b).assign(&compute_rfft_abs(&b_mean));
    }

    let mut max_snr = 0.0_f64;
    for &k in &g2_modes {
        if k < n_freq {
            let std = boot_amps.column(k).std(0.0).max(1e-20);
            let t = fft_obs[k] / std;
            if t > max_snr {
                max_snr = t;
            }
        }
    }

    MangaFitResult {
        snr: max_snr,
        residuals: residuals.clone(),
    }
}

/// Gram-Schmidt orthonormalization of a row-basis matrix.
pub fn gram_schmidt_ortho(mut basis: Array2<f64>) -> Array2<f64> {
    let n = basis.nrows();
    for i in 0..n {
        for j in 0..i {
            let dot = basis.row(i).dot(&basis.row(j));
            let proj = basis.row(j).to_owned() * dot;
            // Separate immutable read from mutable write to satisfy the borrow checker.
            let new_row = basis.row(i).to_owned() - proj;
            basis.row_mut(i).assign(&new_row);
        }
        let norm = basis.row(i).fold(0.0, |acc, &x| acc + x * x).sqrt();
        if norm > 1e-10 {
            basis.row_mut(i).mapv_inplace(|x| x / norm);
        }
    }
    basis
}

/// Real-valued FFT magnitude spectrum (length n/2+1).
pub fn compute_rfft_abs(data: &Array1<f64>) -> Array1<f64> {
    use rustfft::{num_complex::Complex, FftPlanner};
    let n = data.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);

    let n_out = n / 2 + 1;
    let mut result = Array1::<f64>::zeros(n_out);
    for i in 0..n_out {
        result[i] = buffer[i].norm();
    }
    result
}
