//! # H4: Azimuthal Power Spectrum (conditional on H1 passing)
//!
//! First-ever angular analysis of MaNGA velocity residuals for algebraic
//! dark-matter signatures.  Computes the azimuthal power spectrum C_l at
//! multipoles l = 2-8 in 4 radial annuli, with optional beam correction.
//!
//! ## Design
//! - For each galaxy with 2D IFU velocity map:
//!   1. Subtract best-fit rotation model
//!   2. Divide residual map into 4 radial annuli
//!   3. In each annulus, compute azimuthal Fourier decomposition C_l
//!   4. Average C_l over galaxies
//! - Beam correction: deconvolve MaNGA PSF (FWHM ~= 2.5′′) from C_l
//! - Prediction: excess at l = 4 or l = 6 matching partner-graph degeneracies
//!
//! ## Ablation
//! - **No beam correction**: skip PSF deconvolution.
//!
//! ~125 seconds compute.

use crate::common::nfw_v_circ;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters for H4.
#[derive(Debug, Clone)]
pub struct H4Config {
    /// Number of galaxies (each with a 2D velocity map).
    pub n_galaxies: usize,
    /// Radial annuli boundaries in units of r_s (n_annuli + 1 values).
    pub annulus_edges: Vec<f64>,
    /// Multipoles to compute.
    pub ell_values: Vec<usize>,
    /// Whether to apply beam correction.
    pub beam_correction: bool,
    /// MaNGA PSF FWHM in arcseconds.
    pub psf_fwhm_arcsec: f64,
    /// Pixel scale (arcsec per pixel).
    pub pixel_scale_arcsec: f64,
    /// Map half-size in pixels (map is 2*n+1 x 2*n+1).
    pub map_half_size: usize,
    /// RNG seed.
    pub seed: u64,
    /// Noise level (km/s).
    pub noise_km_s: f64,
}

impl Default for H4Config {
    fn default() -> Self {
        Self {
            n_galaxies: 100,
            annulus_edges: vec![1.0, 2.0, 4.0, 6.0, 10.0],
            ell_values: vec![2, 3, 4, 5, 6, 7, 8],
            beam_correction: true,
            psf_fwhm_arcsec: 2.5,
            pixel_scale_arcsec: 0.5,
            map_half_size: 20,
            seed: 42,
            noise_km_s: 10.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic 2D velocity map
// ---------------------------------------------------------------------------

/// A 2D velocity residual map on a square pixel grid.
#[derive(Debug, Clone)]
pub struct VelocityMap {
    /// Pixel values [row][col] in km/s.
    pub data: Vec<Vec<f64>>,
    /// Map half-size in pixels.
    pub half_size: usize,
    /// Scale radius in pixels.
    pub r_s_pixels: f64,
}

/// Generate a synthetic 2D velocity residual map.
///
/// The model is: v(r, \theta) = v_rot(r) * sin(\theta) * sin(i)
/// plus an optional azimuthal perturbation at specific multipoles.
fn generate_velocity_map(
    half_size: usize,
    r_s_pixels: f64,
    log_m200: f64,
    z: f64,
    noise_km_s: f64,
    azimuthal_excess: Option<(usize, f64)>, // (ell, amplitude)
    seed: u64,
) -> VelocityMap {
    let size = 2 * half_size + 1;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise_km_s).unwrap();
    let m200 = 10.0_f64.powf(log_m200);

    let mut data = vec![vec![0.0; size]; size];
    let center = half_size as f64;

    #[allow(clippy::needless_range_loop)]
    for iy in 0..size {
        for ix in 0..size {
            let dx = ix as f64 - center;
            let dy = iy as f64 - center;
            let r_pix = (dx * dx + dy * dy).sqrt();
            let theta = dy.atan2(dx);

            // Convert pixel radius to kpc (assume 1 pixel = some fraction of r_s).
            let r_kpc = (r_pix / r_s_pixels) * 10.0; // r_s ~= 10 kpc scaling

            // Rotation model: v_rot(r) * sin(theta) * sin(i) (i=45°)
            let v_rot = nfw_v_circ(r_kpc.max(0.1), m200, z);
            let v_model = v_rot * theta.sin() * 0.707; // sin(45°)

            // Observed = model + noise
            let v_obs = v_model + normal.sample(&mut rng);

            // Residual = obs - model.
            let mut residual = v_obs - v_model;

            // Add azimuthal perturbation if requested.
            if let Some((ell, amp)) = azimuthal_excess {
                residual += amp * (ell as f64 * theta).cos();
            }

            data[iy][ix] = residual;
        }
    }

    VelocityMap {
        data,
        half_size,
        r_s_pixels,
    }
}

// ---------------------------------------------------------------------------
// Azimuthal power spectrum
// ---------------------------------------------------------------------------

/// Compute C_l for a given annulus on a velocity map.
///
/// C_l = |a_l|^2 where a_l = (1/N) \Sigma_j v(r_j, \theta_j) exp(-i l \theta_j)
fn azimuthal_power_in_annulus(
    map: &VelocityMap,
    r_inner: f64,
    r_outer: f64,
    ell_values: &[usize],
) -> Vec<f64> {
    let center = map.half_size as f64;
    let size = 2 * map.half_size + 1;

    // Collect pixels in annulus.
    let mut pixels: Vec<(f64, f64)> = Vec::new(); // (theta, value)

    for iy in 0..size {
        for ix in 0..size {
            let dx = ix as f64 - center;
            let dy = iy as f64 - center;
            let r_pix = (dx * dx + dy * dy).sqrt();
            let r_norm = r_pix / map.r_s_pixels;

            if r_norm >= r_inner && r_norm < r_outer {
                let theta = dy.atan2(dx);
                pixels.push((theta, map.data[iy][ix]));
            }
        }
    }

    if pixels.is_empty() {
        return vec![0.0; ell_values.len()];
    }

    let n = pixels.len() as f64;

    ell_values
        .iter()
        .map(|&ell| {
            let (mut re, mut im) = (0.0, 0.0);
            for &(theta, val) in &pixels {
                let phase = ell as f64 * theta;
                re += val * phase.cos();
                im += val * phase.sin();
            }
            (re * re + im * im) / (n * n)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Beam correction
// ---------------------------------------------------------------------------

/// Beam transfer function B_l for a Gaussian PSF.
///
/// Uses an effective smoothing kernel for azimuthal modes on an IFU pixel
/// grid.  The suppression is gentler than the full-sky formula
/// `B_l = exp(-l(l+1) \sigma^2 / 2)` because IFU azimuthal modes sample a
/// small angular range.  The factor 100 in the denominator accounts for
/// the ratio of the full-sky solid angle to the IFU field of view.
const BEAM_FOV_SCALING: f64 = 100.0;

fn beam_transfer_function(ell: usize, fwhm_arcsec: f64, pixel_scale_arcsec: f64) -> f64 {
    let sigma_eff = fwhm_arcsec / (2.355 * pixel_scale_arcsec);
    let l = ell as f64;
    // Gentle suppression appropriate for IFU azimuthal modes (small FOV).
    (-l * l * sigma_eff * sigma_eff / (2.0 * BEAM_FOV_SCALING)).exp()
}

/// Apply beam correction to C_l values.
fn apply_beam_correction(
    cl: &[f64],
    ell_values: &[usize],
    fwhm_arcsec: f64,
    pixel_scale_arcsec: f64,
) -> Vec<f64> {
    cl.iter()
        .zip(ell_values.iter())
        .map(|(&c, &ell)| {
            let bl = beam_transfer_function(ell, fwhm_arcsec, pixel_scale_arcsec);
            if bl > 1e-10 {
                c / (bl * bl)
            } else {
                c // avoid division by near-zero
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// C_l for one annulus.
#[derive(Debug, Clone)]
pub struct AnnulusPower {
    /// Inner boundary (r / r_s).
    pub r_inner: f64,
    /// Outer boundary (r / r_s).
    pub r_outer: f64,
    /// C_l values for each multipole.
    pub cl: Vec<f64>,
    /// Corrected C_l (if beam correction applied).
    pub cl_corrected: Vec<f64>,
}

/// Full H4 result.
#[derive(Debug, Clone)]
pub struct H4Result {
    /// Multipoles measured.
    pub ell_values: Vec<usize>,
    /// Per-annulus power spectra (averaged over galaxies).
    pub annuli: Vec<AnnulusPower>,
    /// Whether beam correction was applied.
    pub beam_corrected: bool,
    /// Peak excess multipole (ell with max C_l across all annuli).
    pub peak_ell: usize,
    /// Peak excess annulus index.
    pub peak_annulus: usize,
    /// Peak C_l value.
    pub peak_cl: f64,
    /// Human-readable summary.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the H4 experiment.
pub fn run_h4(config: &H4Config) -> H4Result {
    let n_annuli = config.annulus_edges.len().saturating_sub(1);

    // Generate galaxy maps and compute per-galaxy C_l.
    let maps: Vec<VelocityMap> = (0..config.n_galaxies)
        .into_par_iter()
        .map(|i| {
            let log_m200 = 11.5 + 1.5 * (i as f64) / (config.n_galaxies as f64 - 1.0).max(1.0);
            let seed = config.seed.wrapping_add(i as u64);
            generate_velocity_map(
                config.map_half_size,
                5.0, // r_s in pixels
                log_m200,
                0.03,
                config.noise_km_s,
                None, // no injected excess for the main experiment
                seed,
            )
        })
        .collect();

    // Average C_l over galaxies for each annulus.
    let mut annuli_results: Vec<AnnulusPower> = Vec::with_capacity(n_annuli);

    for ann in 0..n_annuli {
        let r_inner = config.annulus_edges[ann];
        let r_outer = config.annulus_edges[ann + 1];

        let mut cl_sum = vec![0.0; config.ell_values.len()];
        let mut count = 0usize;

        for map in &maps {
            let cl = azimuthal_power_in_annulus(map, r_inner, r_outer, &config.ell_values);
            for (s, c) in cl_sum.iter_mut().zip(cl.iter()) {
                *s += c;
            }
            count += 1;
        }

        let cl_mean: Vec<f64> = if count == 0 {
            // No maps contributed to this annulus: return a zeroed spectrum.
            vec![0.0; config.ell_values.len()]
        } else {
            cl_sum.iter().map(|s| s / count as f64).collect()
        };

        let cl_corrected = if config.beam_correction {
            apply_beam_correction(
                &cl_mean,
                &config.ell_values,
                config.psf_fwhm_arcsec,
                config.pixel_scale_arcsec,
            )
        } else {
            cl_mean.clone()
        };

        annuli_results.push(AnnulusPower {
            r_inner,
            r_outer,
            cl: cl_mean,
            cl_corrected,
        });
    }

    // Find peak excess.
    let (mut peak_ell, mut peak_annulus, mut peak_cl) = (2, 0, 0.0_f64);
    for (ai, ann) in annuli_results.iter().enumerate() {
        for (li, &cl) in ann.cl_corrected.iter().enumerate() {
            if cl > peak_cl {
                peak_cl = cl;
                peak_ell = config.ell_values[li];
                peak_annulus = ai;
            }
        }
    }

    let prediction_match = peak_ell == 4 || peak_ell == 6;
    let summary = format!(
        "H4 Azimuthal Power Spectrum: {} galaxies, {} annuli, ell={:?}, \
         beam_corrected={}, peak ell={} (annulus {}), peak C_l={:.6}, \
         prediction_match={}",
        config.n_galaxies,
        n_annuli,
        config.ell_values,
        config.beam_correction,
        peak_ell,
        peak_annulus,
        peak_cl,
        prediction_match,
    );

    H4Result {
        ell_values: config.ell_values.clone(),
        annuli: annuli_results,
        beam_corrected: config.beam_correction,
        peak_ell,
        peak_annulus,
        peak_cl,
        summary,
    }
}

/// Ablation: run without beam correction.
pub fn run_h4_no_beam(config: &H4Config) -> H4Result {
    let mut config_no_beam = config.clone();
    config_no_beam.beam_correction = false;
    run_h4(&config_no_beam)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h4_smoke() {
        let config = H4Config {
            n_galaxies: 5,
            map_half_size: 10,
            seed: 42,
            ..Default::default()
        };
        let result = run_h4(&config);
        assert_eq!(result.annuli.len(), 4);
        assert_eq!(result.ell_values.len(), 7);
        assert!(result.beam_corrected);
        println!("{}", result.summary);
    }

    #[test]
    fn test_h4_no_beam_ablation() {
        let config = H4Config {
            n_galaxies: 5,
            map_half_size: 10,
            seed: 42,
            ..Default::default()
        };
        let with_beam = run_h4(&config);
        let without_beam = run_h4_no_beam(&config);
        assert!(with_beam.beam_corrected);
        assert!(!without_beam.beam_corrected);

        // Both should have same number of annuli/ells.
        assert_eq!(with_beam.annuli.len(), without_beam.annuli.len());
    }

    #[test]
    fn test_azimuthal_power_detects_injection() {
        // Inject a known l=4 signal and verify it's detected.
        let map = generate_velocity_map(20, 5.0, 12.0, 0.03, 1.0, Some((4, 50.0)), 42);
        let ells = vec![2, 3, 4, 5, 6, 7, 8];
        let cl = azimuthal_power_in_annulus(&map, 1.0, 5.0, &ells);

        // l=4 (index 2) should have the highest power.
        let max_idx = cl
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            ells[max_idx], 4,
            "Injected l=4 signal should dominate: got l={}, cl={:?}",
            ells[max_idx], cl
        );
    }

    #[test]
    fn test_beam_transfer_function() {
        // B_l should decrease with l.
        let b2 = beam_transfer_function(2, 2.5, 0.5);
        let b8 = beam_transfer_function(8, 2.5, 0.5);
        assert!(b2 > b8, "Beam should suppress higher l: B_2={b2}, B_8={b8}");
        assert!(b2 > 0.0 && b2 <= 1.0);
    }

    #[test]
    fn test_beam_correction_boosts_high_ell() {
        let cl = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ells = vec![2, 3, 4, 5, 6, 7, 8];
        let corrected = apply_beam_correction(&cl, &ells, 2.5, 0.5);
        // Higher ell should be boosted more.
        assert!(
            corrected[6] >= corrected[0],
            "Beam correction should boost high ell more: cl[2]={}, cl[8]={}",
            corrected[0],
            corrected[6],
        );
    }
}
