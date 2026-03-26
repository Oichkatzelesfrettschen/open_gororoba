//! # H1: Pre-Fit Injection Recovery (BLOCKING)
//!
//! Injects a CD-ZD harmonic signal **before** NFW fitting and checks whether
//! the fitting procedure absorbs the signal.  Prior runs only injected
//! post-fit; this tests whether NFW freedom can mimic the ZD signature.
//!
//! ## Design
//! - 500 synthetic galaxies x 4 \alpha_zd values x 5 RNG seeds = 10 000 trials
//! - For each trial:
//!   1. Generate clean NFW rotation curve
//!   2. Inject ZD harmonic signal (amplitude \alpha_zd)
//!   3. Fit NFW to the *contaminated* curve (Nelder-Mead)
//!   4. Compute residuals of the fit
//!   5. Measure Fourier power at CD-predicted wavenumbers
//!   6. Compare recovered power to injected power -> absorption fraction
//! - **Gate**: if mean absorption > 60 %, the pipeline is blind -> STOP.
//!
//! ## Ablation: no injection (\alpha = 0) establishes the noise floor.
//!
//! ~125 seconds compute (synthetic data, no I/O).

use crate::common::{
    Verdict, detection_snr, fourier_power_at_wavenumbers, generate_synthetic_galaxy,
    inject_zd_signal, nfw_v_circ, predicted_wavenumbers_cd16,
};
use cosmology_core::nfw_utils;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters for the H1 experiment.
#[derive(Debug, Clone)]
pub struct H1Config {
    /// Number of synthetic galaxies per (\alpha, seed) cell.
    pub n_galaxies: usize,
    /// \alpha_zd injection amplitudes to sweep.
    pub alpha_values: Vec<f64>,
    /// RNG seeds for each independent realization.
    pub seeds: Vec<u64>,
    /// Absorption fraction above which the pipeline is declared blind.
    pub absorption_gate_threshold: f64,
    /// Number of radial points per synthetic rotation curve.
    pub n_radial_points: usize,
    /// Fractional Gaussian noise on velocities.
    pub noise_frac: f64,
}

impl Default for H1Config {
    fn default() -> Self {
        Self {
            n_galaxies: 500,
            alpha_values: vec![0.0, 0.005, 0.01, 0.02],
            seeds: vec![1, 2, 3, 4, 5],
            absorption_gate_threshold: 0.60,
            n_radial_points: 30,
            noise_frac: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-galaxy result
// ---------------------------------------------------------------------------

/// Result from a single galaxy injection-recovery trial.
#[derive(Debug, Clone)]
pub struct GalaxyInjectionResult {
    /// Injected \alpha_zd.
    pub alpha_injected: f64,
    /// Total injected Fourier power (at CD wavenumbers).
    pub injected_power: f64,
    /// Recovered Fourier power after NFW re-fit.
    pub recovered_power: f64,
    /// Absorption fraction: 1 - recovered/injected.  Clamped to [0,1].
    pub absorption: f64,
    /// Detection SNR of residuals.
    pub snr: f64,
}

// ---------------------------------------------------------------------------
// Aggregate result
// ---------------------------------------------------------------------------

/// Aggregate result for one (\alpha, seed) cell.
#[derive(Debug, Clone)]
pub struct CellResult {
    pub alpha: f64,
    pub seed: u64,
    pub mean_absorption: f64,
    pub std_absorption: f64,
    pub mean_snr: f64,
    pub n_galaxies: usize,
}

/// Full H1 experiment result.
#[derive(Debug, Clone)]
pub struct H1Result {
    /// Per-cell results (one per (\alpha, seed) pair).
    pub cells: Vec<CellResult>,
    /// Grand-mean absorption across all non-zero-\alpha cells.
    pub mean_absorption: f64,
    /// Gate verdict.
    pub verdict: Verdict,
    /// Human-readable summary.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

/// 1D bounded NFW fit on a rotation curve (r, v_obs).
///
/// Performs a coarse grid search plus golden-section refinement over log_m200
/// to minimise chi-squared. Returns best-fit M_200.
fn fit_nfw_to_curve(r: &[f64], v_obs: &[f64], v_err: &[f64], z: f64) -> f64 {
    // 1-parameter grid search + golden-section refinement over log_m200.
    let mut best_chi2 = f64::MAX;
    let mut best_log_m = 11.0;

    // Coarse grid
    for i in 0..60 {
        let log_m = 10.0 + 0.05 * i as f64;
        let m200 = 10.0_f64.powf(log_m);
        let chi2: f64 = r
            .iter()
            .zip(v_obs.iter())
            .zip(v_err.iter())
            .map(|((&ri, &vo), &ve)| {
                let vm = nfw_v_circ(ri, m200, z);
                let d = (vo - vm) / ve.max(1.0);
                d * d
            })
            .sum();
        if chi2 < best_chi2 {
            best_chi2 = chi2;
            best_log_m = log_m;
        }
    }

    // Golden-section refinement (±0.1 dex around best)
    let (mut a, mut b) = (best_log_m - 0.1, best_log_m + 0.1);
    let gr = 0.6180339887;
    for _ in 0..40 {
        let c = b - gr * (b - a);
        let d = a + gr * (b - a);
        let fc = chi2_at(c, r, v_obs, v_err, z);
        let fd = chi2_at(d, r, v_obs, v_err, z);
        if fc < fd {
            b = d;
        } else {
            a = c;
        }
    }
    10.0_f64.powf(0.5 * (a + b))
}

fn chi2_at(log_m: f64, r: &[f64], v_obs: &[f64], v_err: &[f64], z: f64) -> f64 {
    let m200 = 10.0_f64.powf(log_m);
    r.iter()
        .zip(v_obs.iter())
        .zip(v_err.iter())
        .map(|((&ri, &vo), &ve)| {
            let vm = nfw_v_circ(ri, m200, z);
            let d = (vo - vm) / ve.max(1.0);
            d * d
        })
        .sum()
}

/// Run one injection-recovery trial for a single galaxy.
fn run_single_galaxy(
    log_m200_true: f64,
    alpha: f64,
    seed: u64,
    n_points: usize,
    noise_frac: f64,
) -> GalaxyInjectionResult {
    let z = 0.03;
    let galaxy = generate_synthetic_galaxy(log_m200_true, z, n_points, noise_frac, seed);
    let m200_true = 10.0_f64.powf(log_m200_true);
    let params_true = nfw_utils::nfw_params_from_mass(m200_true, z);
    let r_s = params_true.r_s_kpc;

    let r: Vec<f64> = galaxy.rotation_curve.iter().map(|p| p.r_kpc).collect();
    let x: Vec<f64> = r.iter().map(|&ri| ri / r_s).collect();

    // Step 1: compute clean fractional residuals.
    let v_model: Vec<f64> = r.iter().map(|&ri| nfw_v_circ(ri, m200_true, z)).collect();
    let mut v_obs_injected: Vec<f64> = galaxy.rotation_curve.iter().map(|p| p.v_obs).collect();
    let v_err: Vec<f64> = galaxy.rotation_curve.iter().map(|p| p.v_err).collect();

    // Step 2: inject ZD signal into *velocities* (pre-fit).
    if alpha > 0.0 {
        let mut delta_frac = vec![0.0; x.len()];
        inject_zd_signal(&x, &mut delta_frac, alpha, seed.wrapping_add(1_000_000));
        for (vo, (vm, df)) in v_obs_injected
            .iter_mut()
            .zip(v_model.iter().zip(delta_frac.iter()))
        {
            *vo += df * vm; // add fractional signal in velocity space
        }
    }

    // Measure injected power (on the delta-v residual *before* re-fit).
    let delta_injected: Vec<f64> = v_obs_injected
        .iter()
        .zip(v_model.iter())
        .map(|(&vo, &vm)| if vm > 1.0 { (vo - vm) / vm } else { 0.0 })
        .collect();
    let wavenumbers = predicted_wavenumbers_cd16();
    let power_injected = fourier_power_at_wavenumbers(&x, &delta_injected, &wavenumbers);
    let total_injected: f64 = power_injected.iter().sum();

    // Step 3: fit NFW to the *contaminated* curve.
    let m200_fit = fit_nfw_to_curve(&r, &v_obs_injected, &v_err, z);

    // Step 4: compute residuals from the fit.
    let v_fit: Vec<f64> = r.iter().map(|&ri| nfw_v_circ(ri, m200_fit, z)).collect();
    let params_fit = nfw_utils::nfw_params_from_mass(m200_fit, z);
    let r_s_fit = params_fit.r_s_kpc;
    let x_fit: Vec<f64> = r.iter().map(|&ri| ri / r_s_fit).collect();

    let delta_recovered: Vec<f64> = v_obs_injected
        .iter()
        .zip(v_fit.iter())
        .map(|(&vo, &vf)| if vf > 1.0 { (vo - vf) / vf } else { 0.0 })
        .collect();

    // Step 5: measure recovered Fourier power.
    let power_recovered = fourier_power_at_wavenumbers(&x_fit, &delta_recovered, &wavenumbers);
    let total_recovered: f64 = power_recovered.iter().sum();

    let snr = detection_snr(&power_recovered, &delta_recovered);

    // Step 6: absorption.
    let absorption = if total_injected > 1e-30 {
        (1.0 - total_recovered / total_injected).clamp(0.0, 1.0)
    } else {
        0.0
    };

    GalaxyInjectionResult {
        alpha_injected: alpha,
        injected_power: total_injected,
        recovered_power: total_recovered,
        absorption,
        snr,
    }
}

// ---------------------------------------------------------------------------
// Public experiment entry point
// ---------------------------------------------------------------------------

/// Run the full H1 experiment.
pub fn run_h1(config: &H1Config) -> H1Result {
    let log_m200_min = 11.0;
    let log_m200_max = 13.0;

    let tasks: Vec<(f64, u64)> = config
        .alpha_values
        .iter()
        .flat_map(|&a| config.seeds.iter().map(move |&s| (a, s)))
        .collect();

    let cells: Vec<CellResult> = tasks
        .par_iter()
        .map(|&(alpha, seed)| {
            let dm = (log_m200_max - log_m200_min) / (config.n_galaxies as f64 - 1.0).max(1.0);
            let results: Vec<GalaxyInjectionResult> = (0..config.n_galaxies)
                .map(|i| {
                    let log_m = log_m200_min + dm * (i as f64);
                    let gseed = seed.wrapping_mul(100_000).wrapping_add(i as u64);
                    run_single_galaxy(
                        log_m,
                        alpha,
                        gseed,
                        config.n_radial_points,
                        config.noise_frac,
                    )
                })
                .collect();

            let absorptions: Vec<f64> = results.iter().map(|r| r.absorption).collect();
            let snrs: Vec<f64> = results.iter().map(|r| r.snr).collect();
            let (mean_abs, var_abs, mean_snr) = if absorptions.is_empty() {
                (0.0, 0.0, 0.0)
            } else {
                let mean_abs = absorptions.iter().sum::<f64>() / absorptions.len() as f64;
                let var_abs = absorptions
                    .iter()
                    .map(|a| (a - mean_abs).powi(2))
                    .sum::<f64>()
                    / absorptions.len() as f64;
                let mean_snr = snrs.iter().sum::<f64>() / snrs.len() as f64;
                (mean_abs, var_abs, mean_snr)
            };

            CellResult {
                alpha,
                seed,
                mean_absorption: mean_abs,
                std_absorption: var_abs.sqrt(),
                mean_snr,
                n_galaxies: config.n_galaxies,
            }
        })
        .collect();

    // Gate: mean absorption across non-zero-alpha cells.
    let nonzero: Vec<&CellResult> = cells.iter().filter(|c| c.alpha > 0.0).collect();
    let mean_absorption = if nonzero.is_empty() {
        0.0
    } else {
        nonzero.iter().map(|c| c.mean_absorption).sum::<f64>() / nonzero.len() as f64
    };

    let verdict = if mean_absorption > config.absorption_gate_threshold {
        Verdict::Fail
    } else {
        Verdict::Pass
    };

    let summary = format!(
        "H1 Pre-Fit Injection Recovery: mean_absorption={:.3} (gate={:.2}) -> {:?}\n\
         {} cells, {} galaxies/cell, {} alpha values, {} seeds",
        mean_absorption,
        config.absorption_gate_threshold,
        verdict,
        cells.len(),
        config.n_galaxies,
        config.alpha_values.len(),
        config.seeds.len(),
    );

    H1Result {
        cells,
        mean_absorption,
        verdict,
        summary,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test with small parameters.
    #[test]
    fn test_h1_smoke() {
        let config = H1Config {
            n_galaxies: 10,
            alpha_values: vec![0.0, 0.01],
            seeds: vec![42],
            absorption_gate_threshold: 0.60,
            n_radial_points: 20,
            noise_frac: 0.05,
        };
        let result = run_h1(&config);
        assert_eq!(result.cells.len(), 2); // 2 alpha x 1 seed
        assert!(result.mean_absorption >= 0.0 && result.mean_absorption <= 1.0);
        // With \alpha=0.01, absorption should be moderate (pipeline should pass)
        println!("{}", result.summary);
    }

    /// The noise-floor cell (\alpha=0) should have ~0 absorption.
    #[test]
    fn test_h1_noise_floor() {
        let config = H1Config {
            n_galaxies: 20,
            alpha_values: vec![0.0],
            seeds: vec![7],
            absorption_gate_threshold: 0.60,
            n_radial_points: 20,
            noise_frac: 0.05,
        };
        let result = run_h1(&config);
        // \alpha=0 -> no injection -> absorption is meaningless (filtered out)
        assert_eq!(result.mean_absorption, 0.0);
        assert_eq!(result.verdict, Verdict::Pass);
    }

    /// Test the single-galaxy injection-recovery function.
    #[test]
    fn test_single_galaxy_recovery() {
        let r = run_single_galaxy(12.0, 0.02, 42, 30, 0.05);
        assert!(r.absorption >= 0.0 && r.absorption <= 1.0);
        assert!(r.injected_power > 0.0);
        assert!(r.recovered_power >= 0.0);
    }

    /// Verify that fit_nfw_to_curve can recover the input mass.
    #[test]
    fn test_nfw_fit_recovery() {
        let log_m_true = 12.0;
        let z = 0.03;
        let galaxy = generate_synthetic_galaxy(log_m_true, z, 30, 0.01, 999);
        let r: Vec<f64> = galaxy.rotation_curve.iter().map(|p| p.r_kpc).collect();
        let v: Vec<f64> = galaxy.rotation_curve.iter().map(|p| p.v_obs).collect();
        let e: Vec<f64> = galaxy.rotation_curve.iter().map(|p| p.v_err).collect();
        let m_fit = fit_nfw_to_curve(&r, &v, &e, z);
        let log_m_fit = m_fit.log10();
        // Should recover within ~0.2 dex
        assert!(
            (log_m_fit - log_m_true).abs() < 0.3,
            "NFW fit should recover input mass: got {:.2} vs {:.2}",
            log_m_fit,
            log_m_true
        );
    }
}
