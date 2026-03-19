//! MaNGA IFU Rotation Curve Null Result Experiment (C-010).
//!
//! Investigates synthetic galaxy rotation curves for Zero-Divisor (ZD) signals.
//! Implements NFW and Baryonic baseline fits, harmonic stacking, and 
//! algebra-specific DFT mode analysis.
//!
//! Migrated from main.py.

use anyhow::Result;
use clap::Parser;
use cosmology_core::manga_sim::{
    generate_synthetic_manga, MangaDataBundle, MangaSimParams,
    compute_nfw_v_template, compute_baryonic_v_template,
};
use ndarray::{Array1, Array2, Axis, Zip};
use rand::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "manga-zd-null")]
#[command(about = "MaNGA IFU Rotation Curve Null Result Experiment")]
struct Args {
    /// Number of galaxies
    #[arg(short, long, default_value_t = 1000)]
    n_galaxies: usize,

    /// Number of bootstrap iterations
    #[arg(short, long, default_value_t = 100)]
    n_bootstrap: usize,

    /// Random seed
    #[arg(short, long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();

    println!("--- MaNGA IFU Null Experiment [RUST-FIRST] ---");
    
    // 1. Setup & Data Generation
    let mut sim_params = MangaSimParams::default();
    sim_params.n_galaxies = args.n_galaxies;
    sim_params.seed = args.seed;
    
    println!("Generating synthetic MaNGA data (N={})...", args.n_galaxies);
    let data = generate_synthetic_manga(&sim_params);

    // 2. Baselines
    let nfw_res = run_nfw_baseline(&data);
    println!("NFW Baseline SNR: {:.4}", nfw_res.snr);

    let baryonic_res = run_baryonic_baseline(&data);
    println!("Baryonic Baseline SNR: {:.4}", baryonic_res.snr);

    // 3. Test Conditions
    let harmonic_res = run_harmonic_stacking(&data, args.n_bootstrap);
    println!("Harmonic Halo Stack SNR: {:.4}", harmonic_res.snr);

    let dft_res = run_multi_algebra_dft(&data, args.n_bootstrap);
    println!("Multi-Algebra DFT SNR: {:.4}", dft_res.snr);

    println!("\n--- FINAL SUMMARY ---");
    println!("Total Time: {:.2?}", start.elapsed());
    
    Ok(())
}

struct FitResult {
    snr: f64,
    residuals: Array2<f64>,
}

/// Single-amplitude NFW fit via Weighted Least Squares.
fn run_nfw_baseline(data: &MangaDataBundle) -> FitResult {
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
    FitResult { snr, residuals }
}

/// 2-parameter (NFW + Baryonic) linear WLS fit.
fn run_baryonic_baseline(data: &MangaDataBundle) -> FitResult {
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
    FitResult { snr, residuals }
}

/// Compute binwise t-statistic on stacked residuals.
fn compute_stacked_snr(residuals: &Array2<f64>) -> f64 {
    let n = residuals.nrows() as f64;
    let mean_r = residuals.mean_axis(Axis(0)).unwrap();
    let mut max_snr = 0.0;

    for j in 0..residuals.ncols() {
        let col = residuals.column(j);
        let var = col.fold(0.0, |acc, &x| acc + (x - mean_r[j]).powi(2)) / (n - 1.0).max(1.0);
        let sem = (var / n).sqrt().max(1e-20);
        let t = (mean_r[j].abs() / sem);
        if t > max_snr { max_snr = t; }
    }
    max_snr
}

/// Orthonormal harmonic mode projection.
fn run_harmonic_stacking(data: &MangaDataBundle, n_boot: usize) -> FitResult {
    let res_obj = run_baryonic_baseline(data);
    let residuals = &res_obj.residuals;
    let x_grid = &data.x_grid;
    let x_max = data.x_grid.fold(0.0, |a, &b| a.max(b));

    // D=16 modes (k=1..4)
    let n_modes = 4;
    let mut basis = Array2::<f64>::zeros((n_modes, x_grid.len()));
    for k in 0..n_modes {
        basis.row_mut(k).assign(&x_grid.mapv(|x| (2.0 * PI * (k + 1) as f64 * x / x_max).cos()));
    }
    basis = gram_schmidt_ortho(basis);

    let mean_res = residuals.mean_axis(Axis(0)).unwrap();
    let coeffs = basis.dot(&mean_res);

    // Bootstrap for SE
    let mut rng = StdRng::seed_from_u64(137);
    let mut boot_coeffs = Array2::<f64>::zeros((n_boot, n_modes));
    for b in 0..n_boot {
        let mut b_sum = Array1::<f64>::zeros(x_grid.len());
        for _ in 0..residuals.nrows() {
            let idx = rng.gen_range(0..residuals.nrows());
            b_sum += &residuals.row(idx);
        }
        let b_mean = b_sum / residuals.nrows() as f64;
        boot_coeffs.row_mut(b).assign(&basis.dot(&b_mean));
    }

    let mut max_snr = 0.0;
    for k in 0..n_modes {
        let col = boot_coeffs.column(k);
        let std = col.std(0.0).max(1e-20);
        let t = coeffs[k].abs() / std;
        if t > max_snr { max_snr = t; }
    }

    FitResult { snr: max_snr, residuals: residuals.clone() }
}

/// Multi-Algebra restricted DFT analysis.
fn run_multi_algebra_dft(data: &MangaDataBundle, n_boot: usize) -> FitResult {
    let res_obj = run_baryonic_baseline(data);
    let residuals = &res_obj.residuals;
    let n_bins = residuals.ncols();
    
    // G2 modes k=1..7
    let g2_modes = vec![1, 2, 3, 4, 5, 6, 7];
    
    let mean_res = residuals.mean_axis(Axis(0)).unwrap();
    let fft_obs = compute_rfft_abs(&mean_res);

    // Bootstrap
    let mut rng = StdRng::seed_from_u64(777);
    let n_freq = fft_obs.len();
    let mut boot_amps = Array2::<f64>::zeros((n_boot, n_freq));
    for b in 0..n_boot {
        let mut b_sum = Array1::<f64>::zeros(n_bins);
        for _ in 0..residuals.nrows() {
            let idx = rng.gen_range(0..residuals.nrows());
            b_sum += &residuals.row(idx);
        }
        let b_mean = b_sum / residuals.nrows() as f64;
        boot_amps.row_mut(b).assign(&compute_rfft_abs(&b_mean));
    }

    let mut max_snr = 0.0;
    for &k in &g2_modes {
        if k < n_freq {
            let std = boot_amps.column(k).std(0.0).max(1e-20);
            let t = fft_obs[k] / std;
            if t > max_snr { max_snr = t; }
        }
    }

    FitResult { snr: max_snr, residuals: residuals.clone() }
}

fn gram_schmidt_ortho(mut basis: Array2<f64>) -> Array2<f64> {
    let n = basis.nrows();
    for i in 0..n {
        for j in 0..i {
            let dot = basis.row(i).dot(&basis.row(j));
            let proj = basis.row(j).to_owned() * dot;
            basis.row_mut(i).assign(&(basis.row(i).to_owned() - proj));
        }
        let norm = basis.row(i).fold(0.0, |acc, &x| acc + x * x).sqrt();
        if norm > 1e-10 {
            basis.row_mut(i).mapv_inplace(|x| x / norm);
        }
    }
    basis
}

fn compute_rfft_abs(data: &Array1<f64>) -> Array1<f64> {
    use rustfft::{FftPlanner, num_complex::Complex};
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
