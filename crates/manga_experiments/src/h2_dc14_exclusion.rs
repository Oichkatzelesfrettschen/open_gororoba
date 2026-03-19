//! # H2: DC14 Phase-Shift Exclusion Surface
//!
//! Replaces the NFW profile with the feedback-modified DC14 cored profile
//! (Di Cintio et al. 2014) on all galaxies, then builds an exclusion surface
//! over a 21-point phase scan δx ∈ [−0.5, +0.5].
//!
//! ## Design
//! - For each galaxy:
//!   1. Compute DC14 profile from (M_star, M_halo)
//!   2. Compute velocity residual: δv = (v_obs − v_DC14) / v_DC14
//!   3. For each phase shift δx (21 values in [−0.5, +0.5]):
//!      - Shift the CD wavenumbers: k_n → k_n + δx
//!      - Compute Fourier power at shifted wavenumbers
//!   4. Build 2D exclusion surface (δx, α_zd) from injection-recovery
//!
//! ## Ablations
//! - **NFW-only (no phase scan)**: Use NFW residuals, no δx sweep.
//! - **NFW + phase scan (no DC14)**: Use NFW residuals with δx sweep.
//!
//! ~100 seconds compute.

use crate::common::{
    detection_snr, fourier_power_at_wavenumbers, generate_galaxy_sample,
    inject_zd_signal, nfw_v_circ, predicted_wavenumbers_cd16, G_KPC_KMS2,
    SyntheticGalaxy,
};
use cosmology_core::nfw_utils::{
    dc14_enclosed_mass, dc14_shape_params, nfw_params_from_mass,
};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Profile type for the experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileType {
    /// Standard NFW cuspy profile.
    Nfw,
    /// DC14 feedback-modified cored profile.
    Dc14,
}

/// Parameters for the H2 experiment.
#[derive(Debug, Clone)]
pub struct H2Config {
    /// Number of galaxies.
    pub n_galaxies: usize,
    /// Phase-shift scan values δx.
    pub delta_x_values: Vec<f64>,
    /// α_zd injection amplitudes for exclusion-surface construction.
    pub alpha_values: Vec<f64>,
    /// Profile for the primary analysis.
    pub profile: ProfileType,
    /// Whether to perform the phase scan.
    pub phase_scan: bool,
    /// Number of radial points per galaxy.
    pub n_radial_points: usize,
    /// Noise level.
    pub noise_frac: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for H2Config {
    fn default() -> Self {
        let delta_x: Vec<f64> = (0..21).map(|i| -0.5 + 0.05 * i as f64).collect();
        Self {
            n_galaxies: 200,
            delta_x_values: delta_x,
            alpha_values: vec![0.0, 0.005, 0.01, 0.02, 0.05],
            profile: ProfileType::Dc14,
            phase_scan: true,
            n_radial_points: 30,
            noise_frac: 0.05,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// DC14 circular velocity
// ---------------------------------------------------------------------------

/// Compute DC14 circular velocity at radius r_kpc.
fn dc14_v_circ(r_kpc: f64, m200_solar: f64, log_mstar: f64, z: f64) -> f64 {
    let p = nfw_params_from_mass(m200_solar, z);
    let log_ratio = log_mstar - m200_solar.log10();
    let shape = dc14_shape_params(log_ratio);
    let m_enc = dc14_enclosed_mass(r_kpc, p.r_s_kpc, p.rho_s_solar_per_kpc3, &shape);
    if m_enc <= 0.0 || r_kpc <= 0.0 {
        return 0.0;
    }
    (G_KPC_KMS2 * m_enc / r_kpc).sqrt()
}

// ---------------------------------------------------------------------------
// Per-galaxy residual computation
// ---------------------------------------------------------------------------

/// Fractional residuals for one galaxy under a given profile.
pub fn compute_residuals(
    galaxy: &SyntheticGalaxy,
    profile: ProfileType,
) -> (Vec<f64>, Vec<f64>) {
    let m200 = 10.0_f64.powf(galaxy.meta.log_m200);
    let z = galaxy.meta.z;
    let p = nfw_params_from_mass(m200, z);
    let r_s = p.r_s_kpc;

    let x: Vec<f64> = galaxy
        .rotation_curve
        .iter()
        .map(|pt| pt.r_kpc / r_s)
        .collect();

    let delta_v: Vec<f64> = galaxy
        .rotation_curve
        .iter()
        .map(|pt| {
            let v_model = match profile {
                ProfileType::Nfw => nfw_v_circ(pt.r_kpc, m200, z),
                ProfileType::Dc14 => dc14_v_circ(pt.r_kpc, m200, galaxy.meta.log_mstar, z),
            };
            if v_model > 1.0 {
                (pt.v_obs - v_model) / v_model
            } else {
                0.0
            }
        })
        .collect();

    (x, delta_v)
}

// ---------------------------------------------------------------------------
// Exclusion surface point
// ---------------------------------------------------------------------------

/// One point on the exclusion surface.
#[derive(Debug, Clone, Copy)]
pub struct ExclusionPoint {
    /// Phase shift applied.
    pub delta_x: f64,
    /// Injected alpha.
    pub alpha: f64,
    /// Mean Fourier power (averaged over galaxies).
    pub mean_power: f64,
    /// Detection SNR.
    pub mean_snr: f64,
    /// Fraction of galaxies where SNR > 2.
    pub detection_fraction: f64,
}

/// Full H2 experiment result.
#[derive(Debug, Clone)]
pub struct H2Result {
    /// Exclusion surface: grid of (δx, α) points.
    pub exclusion_surface: Vec<ExclusionPoint>,
    /// Profile used.
    pub profile: ProfileType,
    /// Whether phase scan was active.
    pub phase_scan: bool,
    /// Human-readable summary.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Public experiment entry point
// ---------------------------------------------------------------------------

/// Run the H2 experiment.
pub fn run_h2(config: &H2Config) -> H2Result {
    let galaxies = generate_galaxy_sample(
        config.n_galaxies,
        11.0,
        13.0,
        config.seed,
    );

    // Pre-compute residuals for all galaxies.
    let residuals: Vec<(Vec<f64>, Vec<f64>)> = galaxies
        .iter()
        .map(|g| compute_residuals(g, config.profile))
        .collect();

    let base_wavenumbers = predicted_wavenumbers_cd16();
    let snr_threshold = 2.0;

    // Determine effective δx values.
    let delta_x_values: Vec<f64> = if config.phase_scan {
        config.delta_x_values.clone()
    } else {
        vec![0.0] // no scan: only δx = 0
    };

    // Build exclusion surface.
    let tasks: Vec<(f64, f64)> = delta_x_values
        .iter()
        .flat_map(|&dx| config.alpha_values.iter().map(move |&a| (dx, a)))
        .collect();

    let exclusion_surface: Vec<ExclusionPoint> = tasks
        .par_iter()
        .map(|&(delta_x, alpha)| {
            // Shift wavenumbers.
            let shifted_k: Vec<f64> = base_wavenumbers.iter().map(|&k| k + delta_x).collect();

            let mut total_power = 0.0;
            let mut total_snr = 0.0;
            let mut n_detected = 0usize;

            for (x, dv) in &residuals {
                // Inject signal (if alpha > 0).
                let mut dv_inj = dv.clone();
                if alpha > 0.0 {
                    inject_zd_signal(x, &mut dv_inj, alpha, config.seed);
                }

                let power = fourier_power_at_wavenumbers(x, &dv_inj, &shifted_k);
                let snr = detection_snr(&power, &dv_inj);
                let total_p: f64 = power.iter().sum();

                total_power += total_p;
                total_snr += snr;
                if snr > snr_threshold {
                    n_detected += 1;
                }
            }

            let n = residuals.len() as f64;
            ExclusionPoint {
                delta_x,
                alpha,
                mean_power: total_power / n,
                mean_snr: total_snr / n,
                detection_fraction: n_detected as f64 / n,
            }
        })
        .collect();

    let summary = format!(
        "H2 DC14 Phase-Shift Exclusion: profile={:?}, phase_scan={}, {} galaxies, \
         {} δx × {} α = {} surface points",
        config.profile,
        config.phase_scan,
        config.n_galaxies,
        delta_x_values.len(),
        config.alpha_values.len(),
        exclusion_surface.len(),
    );

    H2Result {
        exclusion_surface,
        profile: config.profile,
        phase_scan: config.phase_scan,
        summary,
    }
}

// ---------------------------------------------------------------------------
// Ablation runners
// ---------------------------------------------------------------------------

/// Ablation 1: NFW-only, no phase scan.
pub fn run_h2_ablation_nfw_only(n_galaxies: usize, seed: u64) -> H2Result {
    let config = H2Config {
        n_galaxies,
        profile: ProfileType::Nfw,
        phase_scan: false,
        seed,
        ..Default::default()
    };
    run_h2(&config)
}

/// Ablation 2: NFW with phase scan (no DC14).
pub fn run_h2_ablation_nfw_phase_scan(n_galaxies: usize, seed: u64) -> H2Result {
    let config = H2Config {
        n_galaxies,
        profile: ProfileType::Nfw,
        phase_scan: true,
        seed,
        ..Default::default()
    };
    run_h2(&config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h2_smoke() {
        let config = H2Config {
            n_galaxies: 10,
            delta_x_values: vec![-0.25, 0.0, 0.25],
            alpha_values: vec![0.0, 0.01],
            n_radial_points: 20,
            seed: 42,
            ..Default::default()
        };
        let result = run_h2(&config);
        assert_eq!(result.exclusion_surface.len(), 6); // 3 δx × 2 α
        assert!(result.phase_scan);
        assert_eq!(result.profile, ProfileType::Dc14);
        println!("{}", result.summary);
    }

    #[test]
    fn test_h2_ablation_nfw_only() {
        let result = run_h2_ablation_nfw_only(10, 42);
        assert_eq!(result.profile, ProfileType::Nfw);
        assert!(!result.phase_scan);
        // Should have n_alpha surface points (only δx=0).
        assert!(result.exclusion_surface.len() >= 1);
    }

    #[test]
    fn test_h2_ablation_nfw_phase_scan() {
        let result = run_h2_ablation_nfw_phase_scan(10, 42);
        assert_eq!(result.profile, ProfileType::Nfw);
        assert!(result.phase_scan);
        assert!(result.exclusion_surface.len() > 5);
    }

    #[test]
    fn test_dc14_v_circ_positive() {
        let v = dc14_v_circ(10.0, 1e12, 10.5, 0.03);
        assert!(v > 0.0, "DC14 v_circ should be positive: got {v}");
    }

    #[test]
    fn test_exclusion_surface_power_increases_with_alpha() {
        let config = H2Config {
            n_galaxies: 20,
            delta_x_values: vec![0.0],
            alpha_values: vec![0.0, 0.05],
            n_radial_points: 20,
            seed: 77,
            ..Default::default()
        };
        let result = run_h2(&config);
        let p0 = result
            .exclusion_surface
            .iter()
            .find(|e| e.alpha == 0.0)
            .unwrap()
            .mean_power;
        let p1 = result
            .exclusion_surface
            .iter()
            .find(|e| (e.alpha - 0.05).abs() < 1e-10)
            .unwrap()
            .mean_power;
        assert!(
            p1 > p0,
            "Power should increase with injected α: p0={p0}, p1={p1}"
        );
    }
}
