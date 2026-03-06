//! Radial density profile extraction and NFW fitting for simulated DM halos.
//!
//! Given a 3D density field rho(x,y,z), extracts an azimuthally averaged
//! radial profile rho(r), then fits the NFW profile:
//!   rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2]
//!
//! Uses the Nelder-Mead optimizer from the optimizer module.
//!
//! # References
//! - Navarro, Frenk, White (1997): ApJ 490, 493
//! - Dutton & Maccio (2014): MNRAS 441, 3359 (c-M relation)

use crate::optimizer::{NelderMeadConfig, bounded_nelder_mead};

/// A single radial profile bin.
#[derive(Debug, Clone)]
pub struct RadialBin {
    /// Mean radius of this bin.
    pub r: f64,
    /// Mean density in this bin.
    pub rho: f64,
    /// Number of cells contributing to this bin.
    pub count: usize,
}

/// Result of NFW profile fitting.
#[derive(Debug, Clone)]
pub struct NfwFitResult {
    /// Best-fit scale radius r_s.
    pub r_s: f64,
    /// Best-fit characteristic density rho_s.
    pub rho_s: f64,
    /// Chi-squared of the fit.
    pub chi2: f64,
    /// Concentration parameter c = r_200 / r_s (if r_200 provided).
    pub concentration: Option<f64>,
}

/// Extract azimuthally averaged radial density profile from a 3D field.
///
/// `rho_field`: flattened 3D density array [nx * ny * nz], row-major.
/// `nx, ny, nz`: grid dimensions.
/// `center`: center of the profile (grid coordinates).
/// `n_bins`: number of radial bins.
/// `r_max`: maximum radius (in grid units).
pub fn extract_radial_profile(
    rho_field: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    center: [f64; 3],
    n_bins: usize,
    r_max: f64,
) -> Vec<RadialBin> {
    let dr = r_max / n_bins as f64;
    let mut bins: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); n_bins]; // (sum_r, sum_rho, count)

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dx = ix as f64 - center[0];
                let dy = iy as f64 - center[1];
                let dz = iz as f64 - center[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                if r >= r_max {
                    continue;
                }

                let bin_idx = (r / dr) as usize;
                if bin_idx >= n_bins {
                    continue;
                }

                let flat_idx = ix + nx * (iy + ny * iz);
                let rho_val = rho_field[flat_idx] as f64;

                bins[bin_idx].0 += r;
                bins[bin_idx].1 += rho_val;
                bins[bin_idx].2 += 1;
            }
        }
    }

    bins.iter()
        .map(|&(sum_r, sum_rho, count)| {
            if count > 0 {
                RadialBin {
                    r: sum_r / count as f64,
                    rho: sum_rho / count as f64,
                    count,
                }
            } else {
                RadialBin {
                    r: 0.0,
                    rho: 0.0,
                    count: 0,
                }
            }
        })
        .collect()
}

/// NFW density profile: rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2].
pub fn nfw_density(r: f64, r_s: f64, rho_s: f64) -> f64 {
    let x = r / r_s;
    if x < 1e-10 {
        return rho_s; // Central limit
    }
    rho_s / (x * (1.0 + x).powi(2))
}

/// Fit NFW profile to radial density data using Nelder-Mead.
///
/// Returns the best-fit (r_s, rho_s) and chi-squared.
pub fn fit_nfw_profile(profile: &[RadialBin]) -> NfwFitResult {
    let valid_bins: Vec<&RadialBin> = profile.iter().filter(|b| b.count > 0 && b.r > 0.0).collect();

    if valid_bins.len() < 2 {
        return NfwFitResult {
            r_s: 1.0,
            rho_s: 1.0,
            chi2: f64::INFINITY,
            concentration: None,
        };
    }

    // Estimate initial parameters
    let rho_max = valid_bins.iter().map(|b| b.rho).fold(0.0f64, f64::max);
    let r_min_data = valid_bins[0].r;
    let r_max_data = valid_bins.last().map(|b| b.r).unwrap_or(10.0);

    let config = NelderMeadConfig {
        bounds: vec![
            (r_min_data * 0.1, r_max_data * 2.0), // r_s range
            (rho_max * 0.001, rho_max * 1000.0),   // rho_s range
        ],
        max_iter: 5000,
        tol: 1e-14,
        ..Default::default()
    };

    let chi2_fn = |params: &[f64]| -> f64 {
        let r_s = params[0];
        let rho_s = params[1];

        valid_bins
            .iter()
            .map(|bin| {
                let rho_model = nfw_density(bin.r, r_s, rho_s);
                let log_data = (bin.rho + 1e-30).ln();
                let log_model = (rho_model + 1e-30).ln();
                let diff = log_data - log_model;
                diff * diff
            })
            .sum()
    };

    // Multi-start simplex spanning the full r_s range (log-spaced)
    // NFW scale radius often sits well below the data midpoint
    let guesses: Vec<Vec<f64>> = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        .iter()
        .map(|&frac| {
            let r_s_guess = r_min_data + frac * (r_max_data - r_min_data);
            // Estimate rho_s from innermost bin: rho_s ~ rho(r1) * x * (1+x)^2
            let x = valid_bins[0].r / r_s_guess;
            let rho_s_guess = valid_bins[0].rho * x * (1.0 + x).powi(2);
            vec![r_s_guess, rho_s_guess.max(rho_max * 0.001)]
        })
        .collect();
    let (best, chi2) = bounded_nelder_mead(chi2_fn, &guesses, &config);

    NfwFitResult {
        r_s: best[0],
        rho_s: best[1],
        chi2,
        concentration: None,
    }
}

/// Dutton-Maccio (2014) concentration-mass relation: log10(c) = a + b * log10(M / (1e12 Msun/h)).
///
/// Fit for z=0, relaxed halos: a = 0.905, b = -0.101.
pub fn dutton_maccio_concentration(m200_solar: f64) -> f64 {
    let a = 0.905;
    let b = -0.101;
    let log_c = a + b * (m200_solar / 1e12).log10();
    10.0_f64.powf(log_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfw_density_decreases_with_r() {
        let r_s = 10.0;
        let rho_s = 1.0;
        let rho_1 = nfw_density(1.0, r_s, rho_s);
        let rho_10 = nfw_density(10.0, r_s, rho_s);
        let rho_100 = nfw_density(100.0, r_s, rho_s);

        assert!(rho_1 > rho_10);
        assert!(rho_10 > rho_100);
    }

    #[test]
    fn nfw_fit_recovers_known_profile() {
        // Generate synthetic NFW profile
        let r_s_true = 5.0;
        let rho_s_true = 100.0;

        let profile: Vec<RadialBin> = (1..50)
            .map(|i| {
                let r = i as f64;
                RadialBin {
                    r,
                    rho: nfw_density(r, r_s_true, rho_s_true),
                    count: 100,
                }
            })
            .collect();

        let result = fit_nfw_profile(&profile);

        assert!(
            (result.r_s - r_s_true).abs() / r_s_true < 0.1,
            "r_s recovery: expected ~{}, got {}",
            r_s_true,
            result.r_s
        );
        assert!(
            (result.rho_s - rho_s_true).abs() / rho_s_true < 0.1,
            "rho_s recovery: expected ~{}, got {}",
            rho_s_true,
            result.rho_s
        );
    }

    #[test]
    fn dutton_maccio_reasonable_values() {
        // MW-mass halo ~ 1e12 Msun: c ~ 10
        let c_mw = dutton_maccio_concentration(1e12);
        assert!(c_mw > 5.0 && c_mw < 15.0, "MW c = {}", c_mw);

        // Cluster ~ 1e15 Msun: c ~ 5
        let c_cluster = dutton_maccio_concentration(1e15);
        assert!(c_cluster > 2.0 && c_cluster < 10.0, "cluster c = {}", c_cluster);

        // c decreases with mass
        assert!(c_mw > c_cluster);
    }

    #[test]
    fn extract_profile_from_uniform_field() {
        let n = 32;
        let rho_field = vec![1.0f32; n * n * n];
        let center = [n as f64 / 2.0; 3];

        let profile = extract_radial_profile(&rho_field, n, n, n, center, 10, 12.0);

        // All bins should have rho ~ 1.0
        for bin in &profile {
            if bin.count > 0 {
                assert!(
                    (bin.rho - 1.0).abs() < 1e-5,
                    "Uniform field should give rho=1.0, got {}",
                    bin.rho
                );
            }
        }
    }
}
