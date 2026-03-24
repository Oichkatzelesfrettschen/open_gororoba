//! MaNGA IFU Rotation Curve ZD Null Result Experiment.
//!
//! Synthetic null experiment: N=6992 galaxies, x=0.5..1.35 r/r_s, 20 radial bins.
//! Tests whether ZD harmonic signals are detectable over baryonic systematics.
//! Primary metric: SNR (signal-to-noise ratio of ZD detection statistic).
//!
//! Reimplements experiments/manga_zd_null/main.py in pure Rust.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::{collections::HashMap, f64::consts::PI};

/// Experiment configuration.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub n_galaxies: usize,
    pub n_bins: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub cd_dims: Vec<usize>,
    pub n_seeds: usize,
    pub data_seed: u64,
    pub rs_min_kpc: f64,
    pub rs_max_kpc: f64,
    pub inc_min_deg: f64,
    pub inc_max_deg: f64,
    pub vel_disp_frac: f64,
    pub disk_amp_mean: f64,
    pub disk_amp_std: f64,
    pub bulge_amp_mean: f64,
    pub bulge_amp_std: f64,
    pub zd_signal_amp: f64,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            n_galaxies: 6992,
            n_bins: 20,
            x_min: 0.5,
            x_max: 1.35,
            cd_dims: vec![16, 32, 64, 128, 256, 512, 1024],
            n_seeds: 12,
            data_seed: 999,
            rs_min_kpc: 8.0,
            rs_max_kpc: 20.0,
            inc_min_deg: 30.0,
            inc_max_deg: 70.0,
            vel_disp_frac: 0.05,
            disk_amp_mean: 0.3,
            disk_amp_std: 0.1,
            bulge_amp_mean: 0.1,
            bulge_amp_std: 0.05,
            zd_signal_amp: 0.0,
        }
    }
}

/// Per-seed noise realization for all galaxies.
pub struct DataBundle {
    pub x_bins: Vec<f64>,
    pub v_nfw: Vec<Vec<f64>>,
    pub v_disk: Vec<Vec<f64>>,
    pub v_bulge: Vec<Vec<f64>>,
    pub v_obs: Vec<Vec<f64>>,
    pub err_obs: Vec<Vec<f64>>,
    pub inclinations: Vec<f64>,
}

fn nfw_circular_velocity(x: f64) -> f64 {
    let log_term = (1.0 + x).ln() - x / (1.0 + x);
    let norm = 2.0_f64.ln() - 0.5;
    (log_term / (norm * x)).max(1e-8).sqrt()
}

fn disk_template(x: f64) -> f64 {
    let y = x / 2.0;
    x * (-y).exp() * (1.0 + 0.5 * y)
}

fn bulge_template(x: f64) -> f64 {
    x / (1.0 + x * x).powf(0.75)
}

/// Generate a synthetic data bundle for one seed.
pub fn generate_data_bundle(config: &ExperimentConfig, seed: u64) -> DataBundle {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = config.n_galaxies;
    let b = config.n_bins;

    let x_bins: Vec<f64> = (0..b)
        .map(|i| config.x_min + (config.x_max - config.x_min) * i as f64 / (b - 1) as f64)
        .collect();

    let inc_dist = Uniform::new(config.inc_min_deg, config.inc_max_deg);
    let rs_dist = Uniform::new(config.rs_min_kpc, config.rs_max_kpc);
    let disk_dist = Normal::new(config.disk_amp_mean, config.disk_amp_std).unwrap();
    let bulge_dist = Normal::new(config.bulge_amp_mean, config.bulge_amp_std).unwrap();
    let noise_dist = Normal::new(0.0, 1.0).unwrap();

    let inclinations: Vec<f64> = (0..n).map(|_| inc_dist.sample(&mut rng)).collect();
    let _rs_kpc: Vec<f64> = (0..n).map(|_| rs_dist.sample(&mut rng)).collect();
    let disk_amps: Vec<f64> = (0..n)
        .map(|_| disk_dist.sample(&mut rng).clamp(0.05, 0.8))
        .collect();
    let bulge_amps: Vec<f64> = (0..n)
        .map(|_| bulge_dist.sample(&mut rng).clamp(0.01, 0.3))
        .collect();

    let sin_min = config.inc_min_deg.to_radians().sin();
    let sin_i: Vec<f64> = inclinations
        .iter()
        .map(|inc| inc.to_radians().sin().max(sin_min))
        .collect();

    let v_nfw_template: Vec<f64> = x_bins.iter().map(|&x| nfw_circular_velocity(x)).collect();
    let disk_t: Vec<f64> = x_bins.iter().map(|&x| disk_template(x)).collect();
    let bulge_t: Vec<f64> = x_bins.iter().map(|&x| bulge_template(x)).collect();

    let mut v_nfw = Vec::with_capacity(n);
    let mut v_disk = Vec::with_capacity(n);
    let mut v_bulge = Vec::with_capacity(n);
    let mut v_obs = Vec::with_capacity(n);
    let mut err_obs = Vec::with_capacity(n);

    for i in 0..n {
        let nfw_row: Vec<f64> = v_nfw_template.clone();
        let disk_row: Vec<f64> = disk_t.iter().map(|&v| disk_amps[i] * v).collect();
        let bulge_row: Vec<f64> = bulge_t.iter().map(|&v| bulge_amps[i] * v).collect();

        let v_true: Vec<f64> = (0..b)
            .map(|j| nfw_row[j] + disk_row[j] + bulge_row[j])
            .collect();

        let noise_std: Vec<f64> = v_true.iter().map(|&v| config.vel_disp_frac * v).collect();
        let noise: Vec<f64> = noise_std
            .iter()
            .map(|&ns| noise_dist.sample(&mut rng) * ns)
            .collect();

        let obs: Vec<f64> = (0..b)
            .map(|j| (v_true[j] * sin_i[i] + noise[j]) / sin_i[i])
            .collect();
        let err: Vec<f64> = noise_std.iter().map(|&ns| ns / sin_i[i]).collect();

        v_nfw.push(nfw_row);
        v_disk.push(disk_row);
        v_bulge.push(bulge_row);
        v_obs.push(obs);
        err_obs.push(err);
    }

    DataBundle {
        x_bins,
        v_nfw,
        v_disk,
        v_bulge,
        v_obs,
        err_obs,
        inclinations,
    }
}

/// Build orthonormal ZD cosine basis via Gram-Schmidt. n_modes = floor(log2(cd_dim)).
pub fn build_zd_basis(n_bins: usize, cd_dim: usize) -> Vec<Vec<f64>> {
    let n_modes = (cd_dim as f64).log2().floor() as usize;
    let n_modes = n_modes.max(1).min(n_bins / 2);

    let t: Vec<f64> = (0..n_bins)
        .map(|i| PI * i as f64 / (n_bins - 1) as f64)
        .collect();

    let mut basis: Vec<Vec<f64>> = Vec::new();
    for k in 1..=n_modes {
        let v: Vec<f64> = t.iter().map(|&ti| (k as f64 * ti).cos()).collect();
        let mut w = v;
        for b in &basis {
            let dot: f64 = w.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
            for j in 0..w.len() {
                w[j] -= dot * b[j];
            }
        }
        let nrm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 1e-12 {
            for x in &mut w {
                *x /= nrm;
            }
            basis.push(w);
        }
    }
    basis
}

/// WLS fit: NFW + disk + bulge templates via 3x3 normal equations.
/// Returns residuals for each galaxy.
fn wls_baryonic(bundle: &DataBundle) -> Vec<Vec<f64>> {
    let n = bundle.v_obs.len();
    let b = bundle.x_bins.len();
    let mut residuals = Vec::with_capacity(n);

    for i in 0..n {
        let w: Vec<f64> = bundle.err_obs[i]
            .iter()
            .map(|&e| 1.0 / (e * e + 1e-12))
            .collect();
        let templates = [&bundle.v_nfw[i], &bundle.v_disk[i], &bundle.v_bulge[i]];

        // Build 3x3 normal equations
        let mut a = [[0.0f64; 3]; 3];
        let mut rhs = [0.0f64; 3];
        for ti in 0..3 {
            for j in 0..b {
                rhs[ti] += w[j] * bundle.v_obs[i][j] * templates[ti][j];
            }
            for tj in 0..3 {
                for j in 0..b {
                    a[ti][tj] += w[j] * templates[ti][j] * templates[tj][j];
                }
            }
        }
        // Add regularization
        #[allow(clippy::needless_range_loop)]
        for k in 0..3 {
            a[k][k] += 1e-8;
        }

        // Solve 3x3 by Cramer's rule (small system)
        let coeffs = solve_3x3(&a, &rhs);
        let mut res = vec![0.0; b];
        for j in 0..b {
            let v_fit = coeffs[0] * templates[0][j]
                + coeffs[1] * templates[1][j]
                + coeffs[2] * templates[2][j];
            res[j] = bundle.v_obs[i][j] - v_fit;
        }
        residuals.push(res);
    }
    residuals
}

fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-30 {
        return [0.0; 3];
    }
    let inv_det = 1.0 / det;
    [
        inv_det
            * (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
                + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2])),
        inv_det
            * (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
                - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0])),
        inv_det
            * (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
                - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
                + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])),
    ]
}

/// Compute RMS of fractional residuals.
fn compute_rms(residuals: &[Vec<f64>], v_obs: &[Vec<f64>]) -> f64 {
    let n = residuals.len();
    let mut total_rms = 0.0;
    for i in 0..n {
        let v_scale: f64 = v_obs[i].iter().map(|v| v.abs()).sum::<f64>() / v_obs[i].len() as f64;
        let v_scale = v_scale.max(1e-6);
        let b = residuals[i].len();
        let mse: f64 = residuals[i]
            .iter()
            .map(|r| (r / v_scale).powi(2))
            .sum::<f64>()
            / b as f64;
        total_rms += mse.sqrt();
    }
    total_rms / n as f64
}

/// Compute ZD detection SNR: max over modes of |mean(coeffs)| / std(coeffs).
fn compute_snr(residuals: &[Vec<f64>], v_obs: &[Vec<f64>], basis: &[Vec<f64>]) -> f64 {
    let n = residuals.len();
    let n_modes = basis.len();
    if n_modes == 0 || n < 2 {
        return 0.0;
    }

    // Compute per-galaxy projection coefficients
    let mut coeffs = vec![vec![0.0; n_modes]; n];
    for i in 0..n {
        let v_scale: f64 = v_obs[i].iter().map(|v| v.abs()).sum::<f64>() / v_obs[i].len() as f64;
        let v_scale = v_scale.max(1e-6);
        for k in 0..n_modes {
            coeffs[i][k] = residuals[i]
                .iter()
                .zip(basis[k].iter())
                .map(|(r, b)| (r / v_scale) * b)
                .sum();
        }
    }

    // SNR per mode: |mean| / std
    let mut max_snr = 0.0f64;
    for k in 0..n_modes {
        let mean: f64 = coeffs.iter().map(|c| c[k]).sum::<f64>() / n as f64;
        let var: f64 = coeffs.iter().map(|c| (c[k] - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt().max(1e-12);
        let snr = (mean / std).abs();
        if snr > max_snr {
            max_snr = snr;
        }
    }
    max_snr
}

/// Detection threshold: 3 * max_bin(SEM of per-galaxy fractional residuals).
fn compute_threshold(residuals: &[Vec<f64>], v_obs: &[Vec<f64>]) -> f64 {
    let n = residuals.len();
    let b = residuals[0].len();

    // Compute fractional residuals per bin
    let mut frac = vec![vec![0.0; b]; n];
    for i in 0..n {
        let v_scale: f64 = v_obs[i].iter().map(|v| v.abs()).sum::<f64>() / v_obs[i].len() as f64;
        let v_scale = v_scale.max(1e-6);
        for j in 0..b {
            frac[i][j] = residuals[i][j] / v_scale;
        }
    }

    // SEM per bin
    let mut max_sem = 0.0f64;
    for j in 0..b {
        let mean: f64 = frac.iter().map(|f| f[j]).sum::<f64>() / n as f64;
        let var: f64 = frac.iter().map(|f| (f[j] - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let sem = var.sqrt() / (n as f64).sqrt();
        if sem > max_sem {
            max_sem = sem;
        }
    }
    3.0 * max_sem
}

/// Metrics from one condition evaluation.
#[derive(Debug, Clone)]
pub struct ConditionMetrics {
    pub snr: f64,
    pub rms: f64,
    pub threshold: f64,
}

/// Run the baryonic baseline condition on a data bundle.
pub fn evaluate_baryonic_baseline(bundle: &DataBundle, cd_dim: usize) -> ConditionMetrics {
    let residuals = wls_baryonic(bundle);
    let basis = build_zd_basis(bundle.x_bins.len(), cd_dim);
    let snr = compute_snr(&residuals, &bundle.v_obs, &basis);
    let rms = compute_rms(&residuals, &bundle.v_obs);
    let threshold = compute_threshold(&residuals, &bundle.v_obs);
    ConditionMetrics {
        snr,
        rms,
        threshold,
    }
}

/// Assess null result: SNR < 3 means no ZD detection.
pub fn assess_null_result(metrics: &[ConditionMetrics]) -> bool {
    metrics.iter().all(|m| m.snr < 3.0)
}

/// Run the full experiment over multiple seeds.
pub fn run_full_experiment(config: &ExperimentConfig) -> HashMap<String, Vec<ConditionMetrics>> {
    let seeds: Vec<u64> = (0..config.n_seeds as u64)
        .map(|i| config.data_seed + i)
        .collect();

    let mut results: HashMap<String, Vec<ConditionMetrics>> = HashMap::new();

    for &seed in &seeds {
        let bundle = generate_data_bundle(config, seed);
        let metrics = evaluate_baryonic_baseline(&bundle, 64);
        results
            .entry("baryonic_baseline".to_string())
            .or_default()
            .push(metrics);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfw_velocity_profile() {
        // v(1) should be close to 1.0 (normalized).
        let v1 = nfw_circular_velocity(1.0);
        assert!((v1 - 1.0).abs() < 0.01, "v(1.0) = {v1}, expected ~1.0");
    }

    #[test]
    fn test_zd_basis_orthonormal() {
        let basis = build_zd_basis(20, 64);
        for i in 0..basis.len() {
            let norm_sq: f64 = basis[i].iter().map(|x| x * x).sum();
            assert!((norm_sq - 1.0).abs() < 1e-10, "basis[{i}] norm = {norm_sq}");
            for j in (i + 1)..basis.len() {
                let dot: f64 = basis[i]
                    .iter()
                    .zip(basis[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(dot.abs() < 1e-10, "basis[{i}] . basis[{j}] = {dot}");
            }
        }
    }

    #[test]
    fn test_null_experiment_small() {
        let config = ExperimentConfig {
            n_galaxies: 100,
            n_bins: 10,
            n_seeds: 2,
            ..Default::default()
        };
        let results = run_full_experiment(&config);
        let metrics = &results["baryonic_baseline"];
        assert_eq!(metrics.len(), 2);
        // With zd_signal_amp=0, should be null.
        assert!(assess_null_result(metrics));
    }

    #[test]
    fn test_solve_3x3() {
        let a = [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let b = [1.0, 2.0, 3.0];
        let x = solve_3x3(&a, &b);
        // Verify Ax = b
        for i in 0..3 {
            let ax_i: f64 = (0..3).map(|j| a[i][j] * x[j]).sum();
            assert!(
                (ax_i - b[i]).abs() < 1e-10,
                "Ax[{i}]={ax_i}, b[{i}]={}",
                b[i]
            );
        }
    }
}
