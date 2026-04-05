//! Synthetic MaNGA-like Rotation Curve Generation.
//!
//! Generates synthetic galaxy rotation curves matching MaNGA IFU
//! statistics (E-183) for null-result validation of Zero-Divisor (ZD) signals.
//!
//! Migrated from main.py.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;

/// MaNGA-like data bundle for rotation curve experiments.
#[derive(Debug, Clone)]
pub struct MangaDataBundle {
    /// Line-of-sight velocities (N_galaxies x N_bins)
    pub v_los: Array2<f64>,
    /// Inclination-corrected velocities
    pub v_corrected: Array2<f64>,
    /// Statistical errors in corrected frame
    pub err_corrected: Array2<f64>,
    /// Galaxy inclinations in degrees
    pub inclinations: Array1<f64>,
    /// Log10 stellar masses
    pub log_mass: Array1<f64>,
    /// Normalized NFW template used for generation
    pub nfw_template: Array1<f64>,
    /// Normalized baryonic template
    pub baryonic_template: Array1<f64>,
    /// Radial grid (r/r_s)
    pub x_grid: Array1<f64>,
}

/// Compute the NFW velocity profile template: V(x) = sqrt((ln(1+x) - x/(1+x))/x).
pub fn compute_nfw_v_template(x_grid: &Array1<f64>) -> Array1<f64> {
    let mut v = x_grid.mapv(|x| {
        let val = (x.ln_1p() - x / (1.0 + x)) / x;
        if val > 0.0 && val.is_finite() {
            val.sqrt()
        } else {
            0.0
        }
    });
    let vmax: f64 = v.fold(0.0, |a, &b| a.max(b));
    if vmax > 0.0 {
        v /= vmax;
    }
    v
}

/// Compute a generic baryonic disk velocity profile template: V(x) = x * exp(-x/0.3).
pub fn compute_baryonic_v_template(x_grid: &Array1<f64>) -> Array1<f64> {
    let mut v = x_grid.mapv(|x| x * (-x / 0.3).exp());
    let vmax: f64 = v.fold(0.0, |a, &b| a.max(b));
    if vmax > 0.0 {
        v /= vmax;
    }
    v
}

/// Parameters for synthetic MaNGA generation.
pub struct MangaSimParams {
    pub n_galaxies: usize,
    pub n_bins: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub noise_frac: f64,
    pub baryonic_frac: f64,
    pub seed: u64,
}

impl Default for MangaSimParams {
    fn default() -> Self {
        Self {
            n_galaxies: 6992,
            n_bins: 20,
            x_min: 0.5,
            x_max: 1.35,
            noise_frac: 0.075,
            baryonic_frac: 0.30,
            seed: 999,
        }
    }
}

/// Generate a synthetic MaNGA data bundle.
pub fn generate_synthetic_manga(params: &MangaSimParams) -> MangaDataBundle {
    let mut rng = StdRng::seed_from_u64(params.seed);
    let x_grid = Array1::linspace(params.x_min, params.x_max, params.n_bins);
    let nfw_template = compute_nfw_v_template(&x_grid);
    let baryonic_template = compute_baryonic_v_template(&x_grid);

    let n = params.n_galaxies;
    let mut v_los = Array2::zeros((n, params.n_bins));
    let mut v_corrected = Array2::zeros((n, params.n_bins));
    let mut err_corrected = Array2::zeros((n, params.n_bins));
    let mut inclinations = Array1::zeros(n);
    let mut log_mass = Array1::zeros(n);

    let incl_dist = Uniform::new(30.0, 70.0).expect("valid inclination range");
    let mass_dist = Normal::new(10.5, 0.5).unwrap();
    let nfw_amp_dist = Normal::new(1.0, 0.15).unwrap();
    let bar_amp_dist = Normal::new(1.0, 0.2).unwrap();
    let noise_dist = Normal::new(0.0, 1.0).unwrap();

    for i in 0..n {
        let inc = incl_dist.sample(&mut rng);
        let lm: f64 = mass_dist.sample(&mut rng);
        let lm = lm.clamp(8.5_f64, 12.5_f64);
        inclinations[i] = inc;
        log_mass[i] = lm;

        let v_peak = 10.0_f64.powf(0.32 * lm - 1.7);
        let a_nfw: f64 = nfw_amp_dist.sample(&mut rng);
        let a_nfw = v_peak * a_nfw.clamp(0.4_f64, 2.5_f64);
        let a_bar: f64 = bar_amp_dist.sample(&mut rng);
        let a_bar = v_peak * params.baryonic_frac * a_bar.max(0.0_f64);

        let sin_i = (inc * PI / 180.0).sin();
        let noise_sigma = a_nfw * params.noise_frac;

        for j in 0..params.n_bins {
            let v_circ = a_nfw * nfw_template[j] + a_bar * baryonic_template[j];
            let noise = noise_dist.sample(&mut rng) * noise_sigma;
            let v_l = v_circ * sin_i + noise;

            v_los[[i, j]] = v_l;
            v_corrected[[i, j]] = v_l / sin_i.max(0.1);
            err_corrected[[i, j]] = noise_sigma.max(1e-6) / sin_i.max(0.1);
        }
    }

    MangaDataBundle {
        v_los,
        v_corrected,
        err_corrected,
        inclinations,
        log_mass,
        nfw_template,
        baryonic_template,
        x_grid,
    }
}
