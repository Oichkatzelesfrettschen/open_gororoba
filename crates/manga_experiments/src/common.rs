//! Shared types and synthetic-data helpers used by all four experiments.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants (Planck 2018, h = 1)
// ---------------------------------------------------------------------------

/// Gravitational constant in kpc (km/s)^2 / M_sun.
pub const G_KPC_KMS2: f64 = 4.301e-3;

/// Planck 2018 matter fraction.
pub const OMEGA_M: f64 = 0.315;

// ---------------------------------------------------------------------------
// Experiment-level configuration
// ---------------------------------------------------------------------------

/// Verdict returned by a statistical gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    Uncertain,
}

/// A single radial data point on a rotation curve.
#[derive(Debug, Clone, Copy)]
pub struct RotationPoint {
    /// Galactocentric radius (kpc).
    pub r_kpc: f64,
    /// Observed circular velocity (km/s).
    pub v_obs: f64,
    /// Measurement uncertainty on v_obs (km/s).
    pub v_err: f64,
}

/// Minimal galaxy metadata needed by the experiments.
#[derive(Debug, Clone)]
pub struct GalaxyMeta {
    /// log10(M_star / M_sun).
    pub log_mstar: f64,
    /// Estimated log10(M_200 / M_sun) from SMHM relation.
    pub log_m200: f64,
    /// Redshift (typically ~0 for MaNGA).
    pub z: f64,
    /// Inclination in degrees (for beam corrections).
    pub inclination_deg: f64,
}

/// A complete synthetic galaxy for the experiments.
#[derive(Debug, Clone)]
pub struct SyntheticGalaxy {
    pub meta: GalaxyMeta,
    pub rotation_curve: Vec<RotationPoint>,
}

// ---------------------------------------------------------------------------
// NFW helpers (thin wrappers for clarity inside experiments)
// ---------------------------------------------------------------------------

/// NFW scale radius from M_200 using Dutton & Maccio (2014).
pub fn nfw_rs_from_m200(m200_solar: f64, z: f64) -> f64 {
    let p = cosmology_core::nfw_utils::nfw_params_from_mass(m200_solar, z);
    p.r_s_kpc
}

/// NFW circular velocity at radius r for a given virial mass.
pub fn nfw_v_circ(r_kpc: f64, m200_solar: f64, z: f64) -> f64 {
    let p = cosmology_core::nfw_utils::nfw_params_from_mass(m200_solar, z);
    let m_enc = cosmology_core::nfw_utils::nfw_enclosed_mass_from_params(r_kpc, &p);
    if m_enc <= 0.0 || r_kpc <= 0.0 {
        return 0.0;
    }
    (G_KPC_KMS2 * m_enc / r_kpc).sqrt()
}

// ---------------------------------------------------------------------------
// ZD harmonic signal model
// ---------------------------------------------------------------------------

/// Predicted CD-ZD wavenumbers for 16-dimensional (sedenion) algebra.
///
/// In the Cayley-Dickson zero-divisor (CD-ZD) framework, a dim-D algebra
/// has `N = D/2 − 1` independent modes.  For the sedenion (D = 16),
/// `N = 7`.  The predicted wavenumbers are `k_n = 2π n / N` for
/// `n = 1..N`, corresponding to harmonic partners in the projective
/// geometry PG(2, 2).
pub fn predicted_wavenumbers_cd16() -> Vec<f64> {
    let n_modes = 7; // 16/2 - 1
    (1..=n_modes)
        .map(|n| 2.0 * PI * (n as f64) / (n_modes as f64))
        .collect()
}

/// Inject a ZD harmonic signal into a set of fractional residuals.
///
/// `delta_v[i] += alpha * sum_n sin(k_n * x[i] + phi_n)`
///
/// where k_n are the CD-predicted wavenumbers and phi_n are random phases.
pub fn inject_zd_signal(x: &[f64], delta_v: &mut [f64], alpha: f64, seed: u64) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let phase_dist = rand_distr::Uniform::new(0.0, 2.0 * PI);
    let wavenumbers = predicted_wavenumbers_cd16();
    let phases: Vec<f64> = wavenumbers
        .iter()
        .map(|_| phase_dist.sample(&mut rng))
        .collect();

    for (i, xi) in x.iter().enumerate() {
        let mut signal = 0.0;
        for (k, phi) in wavenumbers.iter().zip(phases.iter()) {
            signal += (k * xi + phi).sin();
        }
        delta_v[i] += alpha * signal / (wavenumbers.len() as f64).sqrt();
    }
}

// ---------------------------------------------------------------------------
// Synthetic galaxy generator
// ---------------------------------------------------------------------------

/// Generate a synthetic galaxy rotation curve with NFW profile + Gaussian noise.
pub fn generate_synthetic_galaxy(
    log_m200: f64,
    z: f64,
    n_points: usize,
    noise_frac: f64,
    seed: u64,
) -> SyntheticGalaxy {
    let m200 = 10.0_f64.powf(log_m200);
    let params = cosmology_core::nfw_utils::nfw_params_from_mass(m200, z);
    let r_s = params.r_s_kpc;

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Radial range: 0.5*r_s .. 10*r_s
    let r_min = 0.5 * r_s;
    let r_max = 10.0 * r_s;
    let dr = (r_max - r_min) / (n_points as f64 - 1.0).max(1.0);

    let mut curve = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let r = r_min + dr * (i as f64);
        let v_model = nfw_v_circ(r, m200, z);
        let noise = noise_frac * v_model * normal.sample(&mut rng);
        let v_obs = (v_model + noise).max(1.0);
        let v_err = (noise_frac * v_model).max(1.0);
        curve.push(RotationPoint {
            r_kpc: r,
            v_obs,
            v_err,
        });
    }

    let log_mstar = log_m200 - 1.5; // approximate SMHM offset

    SyntheticGalaxy {
        meta: GalaxyMeta {
            log_mstar,
            log_m200,
            z,
            inclination_deg: 30.0, // typical face-on
        },
        rotation_curve: curve,
    }
}

/// Generate a batch of synthetic galaxies spanning a mass range.
pub fn generate_galaxy_sample(
    n_galaxies: usize,
    log_m200_min: f64,
    log_m200_max: f64,
    z: f64,
    n_points: usize,
    noise_frac: f64,
    seed: u64,
) -> Vec<SyntheticGalaxy> {
    let dm = (log_m200_max - log_m200_min) / (n_galaxies as f64 - 1.0).max(1.0);
    (0..n_galaxies)
        .map(|i| {
            let log_m200 = log_m200_min + dm * (i as f64);
            generate_synthetic_galaxy(
                log_m200,
                z,
                n_points,
                noise_frac,
                seed.wrapping_add(i as u64),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Fourier analysis helpers
// ---------------------------------------------------------------------------

/// Compute discrete Fourier power at specified wavenumbers on irregularly
/// sampled (x, y) data using a direct (non-FFT) evaluation.
pub fn fourier_power_at_wavenumbers(x: &[f64], y: &[f64], wavenumbers: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    wavenumbers
        .iter()
        .map(|&k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (&xi, &yi) in x.iter().zip(y.iter()) {
                let phase = k * xi;
                re += yi * phase.cos();
                im += yi * phase.sin();
            }
            (re * re + im * im) / (n * n)
        })
        .collect()
}

/// Root-mean-square of a slice.
pub fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let ss: f64 = v.iter().map(|x| x * x).sum();
    (ss / v.len() as f64).sqrt()
}

/// Detection SNR = sqrt(max_power) / rms_residual.
pub fn detection_snr(power: &[f64], residuals: &[f64]) -> f64 {
    let max_p = power.iter().cloned().fold(0.0_f64, f64::max);
    let noise = rms(residuals);
    if noise < 1e-30 {
        return 0.0;
    }
    max_p.sqrt() / noise
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicted_wavenumbers_cd16() {
        let k = predicted_wavenumbers_cd16();
        assert_eq!(k.len(), 7);
        assert!((k[0] - 2.0 * PI / 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_generate_synthetic_galaxy() {
        let g = generate_synthetic_galaxy(12.0, 0.03, 30, 0.05, 42);
        assert_eq!(g.rotation_curve.len(), 30);
        assert!(g.rotation_curve.iter().all(|p| p.v_obs > 0.0));
    }

    #[test]
    fn test_inject_zd_signal_nonzero() {
        let x: Vec<f64> = (0..100).map(|i| 0.5 + 0.1 * i as f64).collect();
        let mut dv = vec![0.0; 100];
        inject_zd_signal(&x, &mut dv, 0.01, 42);
        let energy: f64 = dv.iter().map(|v| v * v).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_fourier_power_detects_injected() {
        let n = 200;
        let x: Vec<f64> = (0..n)
            .map(|i| 0.5 + 9.5 * (i as f64) / (n as f64 - 1.0))
            .collect();
        let mut y = vec![0.0; n];
        inject_zd_signal(&x, &mut y, 0.1, 99);
        let k = predicted_wavenumbers_cd16();
        let power = fourier_power_at_wavenumbers(&x, &y, &k);
        let max_p = power.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_p > 1e-6,
            "Injected signal should produce measurable Fourier power"
        );
    }

    #[test]
    fn test_rms() {
        let v = vec![1.0, -1.0, 1.0, -1.0];
        assert!((rms(&v) - 1.0).abs() < 1e-12);
    }
}
