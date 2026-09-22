//! Algebraic amplitude diagnostics with p-adic and spectral weights.
//!
//! Closed wavevector triples are classified using scalar amplitudes and
//! user-chosen weights. These scores discard Fourier phase and polarization;
//! they do not implement the Navier-Stokes nonlinearity or signed transfer.
//! E7 root overlays are algebraic classifications, with no established
//! equivalence to the Fourier operator. Applying these weights or filters
//! during time evolution defines modified equations unless an exact
//! change-of-basis equivalence is separately proved.

use gororoba_algebra::construction::padic::vp_int;
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Configuration for a warp-ring spectral analysis.
#[derive(Debug, Clone)]
pub struct WarpRingConfig {
    /// Prime for the algebraic p-adic weighting (2 for dyadic indices).
    pub prime: u64,
    /// Fractional Laplacian exponent (alpha < 0 for negative dimension)
    pub alpha: f64,
    /// Regularization parameter for negative-dim kernel
    pub epsilon: f64,
    /// Physical domain size
    pub domain_size: f64,
}

impl Default for WarpRingConfig {
    fn default() -> Self {
        Self {
            prime: 2,
            alpha: -0.5,
            epsilon: 0.01,
            domain_size: 2.0 * PI,
        }
    }
}

/// P-adic modulation weight for a spectral triad.
///
/// Given wavevector magnitudes |k|, |p|, |q|, compute the p-adic weight:
///   w = p^{-max(v_p(k_int), v_p(p_int), v_p(q_int))}
///
/// Modes at dyadic indices k = 2^n receive weight 2^{-n}.
/// This is a chosen classification weight, not a measured interaction rate.
pub fn padic_triad_weight(k_mag: f64, p_mag: f64, q_mag: f64, prime: u64) -> f64 {
    // Convert to integer wavenumber indices (nearest nonzero integer)
    let k_int = (k_mag.round() as i64).max(1);
    let p_int = (p_mag.round() as i64).max(1);
    let q_int = (q_mag.round() as i64).max(1);

    let vk = vp_int(k_int, prime).unwrap_or(0);
    let vp = vp_int(p_int, prime).unwrap_or(0);
    let vq = vp_int(q_int, prime).unwrap_or(0);

    let v_max = vk.max(vp).max(vq);
    (prime as f64).powi(-v_max)
}

/// Apply negative-dimension spectral kernel to a 2D Fourier-space field.
///
/// Multiplies each Fourier coefficient by (|k| + epsilon)^alpha where
/// alpha < 0 provides the "negative-dimension" regularized smoothing.
///
/// This is the spectral realization of the operator T from neg_dim.rs,
/// applied directly in k-space without a separate FFT round-trip.
pub fn apply_neg_dim_kernel(
    field_hat: &Array2<Complex64>,
    config: &WarpRingConfig,
) -> Array2<Complex64> {
    let (nx, ny) = field_hat.dim();
    let l = config.domain_size;
    let alpha = config.alpha;
    let eps = config.epsilon;

    Array2::from_shape_fn((nx, ny), |(i, j)| {
        let kx = if i <= nx / 2 {
            i as f64
        } else {
            i as f64 - nx as f64
        };
        let ky = if j <= ny / 2 {
            j as f64
        } else {
            j as f64 - ny as f64
        };
        let kx_phys = 2.0 * PI * kx / l;
        let ky_phys = 2.0 * PI * ky / l;
        let k_mag = (kx_phys * kx_phys + ky_phys * ky_phys).sqrt();

        let multiplier = (k_mag + eps).powf(alpha);
        field_hat[[i, j]] * multiplier
    })
}

/// Compute p-adic modulated power spectrum from a 2D Fourier field.
///
/// For each radial wavenumber bin, weights the power by the p-adic valuation
/// of the bin index. Bins at power-of-p scales get suppressed, revealing
/// the non-dyadic index classes in this algebraic diagnostic.
pub fn padic_power_spectrum(
    field_hat: &Array2<Complex64>,
    domain_size: f64,
    prime: u64,
) -> (Vec<f64>, Vec<f64>) {
    let (nx, ny) = field_hat.dim();
    let k_max = (nx / 2).min(ny / 2);
    if k_max == 0 {
        return (vec![], vec![]);
    }

    let dk = 2.0 * PI / domain_size;
    let mut power = vec![0.0_f64; k_max];
    let mut counts = vec![0usize; k_max];

    for ((i, j), val) in field_hat.indexed_iter() {
        let kx = if i <= nx / 2 {
            i as f64
        } else {
            i as f64 - nx as f64
        };
        let ky = if j <= ny / 2 {
            j as f64
        } else {
            j as f64 - ny as f64
        };
        let k = (kx * kx + ky * ky).sqrt();
        let bin = k.floor() as usize;
        if bin > 0 && bin < k_max {
            // Weight by p-adic valuation of the bin index
            let vp = vp_int(bin as i64, prime).unwrap_or(0);
            let weight = (prime as f64).powi(-vp);
            power[bin] += val.norm_sqr() * weight;
            counts[bin] += 1;
        }
    }

    let k_bins: Vec<f64> = (0..k_max).map(|b| (b as f64 + 0.5) * dk).collect();
    for (p, &c) in power.iter_mut().zip(counts.iter()) {
        if c > 0 {
            *p /= c as f64;
        }
    }

    (k_bins, power)
}

/// Spectral triad with p-adic and negative-dimension weights.
#[derive(Debug, Clone)]
pub struct WarpTriad {
    /// Wavevector indices (signed)
    pub k: [i32; 2],
    pub p: [i32; 2],
    pub q: [i32; 2],
    /// Scalar three-mode amplitude product, without phase or polarization.
    pub triad_amplitude_proxy: f64,
    /// P-adic modulation weight
    pub padic_weight: f64,
    /// Negative-dimension kernel weight at mode k
    pub neg_dim_weight: f64,
    /// Amplitude product multiplied by p-adic and spectral weights.
    pub triad_weighted_amplitude: f64,
}

/// Extract warp-ring triads from a 2D turbulent field.
///
/// Combines spectral triad extraction with p-adic modulation and
/// negative-dimension kernel weighting to produce the full warp-ring
/// triad set. Amplitudes retain the normalization of field_hat; supply F/N
/// for Fourier-series coefficients. Phase invariance is an intentional
/// limitation of this proxy, not evidence of phase-insensitive dynamics.
pub fn extract_warp_triads(
    field_hat: &Array2<Complex64>,
    config: &WarpRingConfig,
    amplitude_threshold: f64,
) -> Vec<WarpTriad> {
    let (nx, ny) = field_hat.dim();
    assert!(amplitude_threshold.is_finite() && amplitude_threshold >= 0.0);
    assert!(
        field_hat
            .iter()
            .all(|z| z.re.is_finite() && z.im.is_finite())
    );
    let half_x = (nx / 2) as i32;
    let half_y = (ny / 2) as i32;
    let l = config.domain_size;
    let alpha = config.alpha;
    let eps = config.epsilon;

    let mut triads = Vec::new();

    // Iterate over pairs (k, p) and check if q = -(k+p) exists in the grid
    for kx in -half_x..half_x {
        for ky in -half_y..half_y {
            if kx == 0 && ky == 0 {
                continue;
            }
            for px in -half_x..half_x {
                for py in -half_y..half_y {
                    if px == 0 && py == 0 {
                        continue;
                    }
                    let qx = -(kx + px);
                    let qy = -(ky + py);

                    // Check bounds
                    if qx < -half_x || qx >= half_x || qy < -half_y || qy >= half_y {
                        continue;
                    }
                    if qx == 0 && qy == 0 {
                        continue;
                    }

                    // Canonical ordering to avoid duplicates
                    let k_idx = (kx, ky);
                    let p_idx = (px, py);
                    let q_idx = (qx, qy);
                    if !(k_idx < p_idx && p_idx < q_idx) {
                        continue;
                    }

                    // Fetch Fourier coefficients
                    let ki = if kx < 0 {
                        (kx + nx as i32) as usize
                    } else {
                        kx as usize
                    };
                    let kj = if ky < 0 {
                        (ky + ny as i32) as usize
                    } else {
                        ky as usize
                    };
                    let pi = if px < 0 {
                        (px + nx as i32) as usize
                    } else {
                        px as usize
                    };
                    let pj = if py < 0 {
                        (py + ny as i32) as usize
                    } else {
                        py as usize
                    };

                    let uk = field_hat[[ki, kj]];
                    let up = field_hat[[pi, pj]];

                    let qi = qx.rem_euclid(nx as i32) as usize;
                    let qj = qy.rem_euclid(ny as i32) as usize;
                    let uq = field_hat[[qi, qj]];
                    let amplitude = uk.norm() * up.norm() * uq.norm();
                    if amplitude <= amplitude_threshold {
                        continue;
                    }

                    // P-adic weight
                    let k_mag = ((kx * kx + ky * ky) as f64).sqrt();
                    let p_mag = ((px * px + py * py) as f64).sqrt();
                    let q_mag = ((qx * qx + qy * qy) as f64).sqrt();
                    let pw = padic_triad_weight(k_mag, p_mag, q_mag, config.prime);

                    // Negative-dim kernel weight at mode k
                    let kx_phys = 2.0 * PI * kx as f64 / l;
                    let ky_phys = 2.0 * PI * ky as f64 / l;
                    let k_phys = (kx_phys * kx_phys + ky_phys * ky_phys).sqrt();
                    let ndw = (k_phys + eps).powf(alpha);

                    let triad_weighted_amplitude = amplitude * pw * ndw;

                    triads.push(WarpTriad {
                        k: [kx, ky],
                        p: [px, py],
                        q: [qx, qy],
                        triad_amplitude_proxy: amplitude,
                        padic_weight: pw,
                        neg_dim_weight: ndw,
                        triad_weighted_amplitude,
                    });
                }
            }
        }
    }

    // Sort by warp weight descending
    triads.sort_by(|a, b| {
        b.triad_weighted_amplitude
            .partial_cmp(&a.triad_weighted_amplitude)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    triads
}

/// Compute the warp-ring spectral density: power spectrum weighted by
/// the negative-dimension kernel.
///
/// P_warp(k) = P(k) * (k + epsilon)^alpha
///
/// For alpha < 0, this enhances low-k (large-scale) modes and suppresses
/// high-k modes. This spectral reweighting does not establish a physical cascade.
pub fn warp_spectral_density(k_bins: &[f64], power: &[f64], config: &WarpRingConfig) -> Vec<f64> {
    k_bins
        .iter()
        .zip(power.iter())
        .map(|(&k, &p)| {
            let weight = (k + config.epsilon).powf(config.alpha);
            p * weight
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padic_triad_weight_dyadic() {
        // k=4=2^2, p=8=2^3, q=12: max valuation is 3 (from p=8)
        let w = padic_triad_weight(4.0, 8.0, 12.0, 2);
        let expected = 2.0_f64.powi(-3); // 0.125
        assert!(
            (w - expected).abs() < 1e-14,
            "Expected {}, got {}",
            expected,
            w
        );
    }

    #[test]
    fn test_padic_triad_weight_odd() {
        // k=3, p=5, q=7: all odd, v_2 = 0 for all
        let w = padic_triad_weight(3.0, 5.0, 7.0, 2);
        assert!((w - 1.0).abs() < 1e-14, "Odd modes should have weight 1.0");
    }

    #[test]
    fn test_padic_triad_weight_prime_3() {
        // k=9=3^2: v_3(9) = 2
        let w = padic_triad_weight(9.0, 5.0, 7.0, 3);
        let expected = 3.0_f64.powi(-2); // 1/9
        assert!((w - expected).abs() < 1e-14);
    }

    #[test]
    fn test_neg_dim_kernel_dc_enhanced() {
        // DC mode should be enhanced (large multiplier for small k)
        let config = WarpRingConfig {
            alpha: -1.0,
            epsilon: 0.01,
            ..Default::default()
        };
        let field = Array2::from_elem((8, 8), Complex64::new(1.0, 0.0));
        let result = apply_neg_dim_kernel(&field, &config);

        // DC mode: k=0, multiplier = (0 + 0.01)^(-1) = 100
        assert!(result[[0, 0]].re > 50.0, "DC should be strongly enhanced");

        // High-k mode: multiplier should be much smaller
        assert!(result[[4, 0]].re < result[[0, 0]].re);
    }

    #[test]
    fn test_padic_power_spectrum_nonempty() {
        let field = Array2::from_shape_fn((16, 16), |(i, j)| {
            Complex64::new(
                (2.0 * PI * i as f64 / 16.0).sin() + (4.0 * PI * j as f64 / 16.0).cos(),
                0.0,
            )
        });
        let (k_bins, power) = padic_power_spectrum(&field, 2.0 * PI, 2);
        assert!(!k_bins.is_empty());
        assert!(!power.is_empty());
        // At least some bins should have nonzero power
        assert!(power.iter().any(|&p| p > 0.0));
    }

    #[test]
    fn test_warp_spectral_density_enhances_low_k() {
        let config = WarpRingConfig {
            alpha: -0.5,
            epsilon: 0.1,
            ..Default::default()
        };
        let k_bins = vec![0.1, 1.0, 10.0, 100.0];
        let power = vec![1.0, 1.0, 1.0, 1.0]; // Flat spectrum

        let warp = warp_spectral_density(&k_bins, &power, &config);

        // Should be monotonically decreasing (low-k enhanced)
        for i in 1..warp.len() {
            assert!(
                warp[i] < warp[i - 1],
                "Warp density should decrease with k: warp[{}]={} >= warp[{}]={}",
                i,
                warp[i],
                i - 1,
                warp[i - 1]
            );
        }
    }

    #[test]
    fn test_extract_warp_triads_closure() {
        // Create a simple field and verify triad closure k+p+q=0
        let field = Array2::from_shape_fn((8, 8), |(i, j)| {
            Complex64::new(
                (2.0 * PI * i as f64 / 8.0).sin() * (2.0 * PI * j as f64 / 8.0).cos(),
                0.0,
            )
        });
        let config = WarpRingConfig::default();
        let triads = extract_warp_triads(&field, &config, 0.0);

        for t in &triads {
            assert_eq!(t.k[0] + t.p[0] + t.q[0], 0, "k+p+q x-component must be 0");
            assert_eq!(t.k[1] + t.p[1] + t.q[1], 0, "k+p+q y-component must be 0");
        }
    }

    #[test]
    fn test_extract_warp_triads_sorted() {
        let field = Array2::from_shape_fn((8, 8), |(i, j)| {
            Complex64::new((i as f64 + j as f64 * 0.3).sin(), 0.0)
        });
        let config = WarpRingConfig::default();
        let triads = extract_warp_triads(&field, &config, 0.0);

        // Should be sorted by triad_weighted_amplitude descending
        for i in 1..triads.len() {
            assert!(
                triads[i].triad_weighted_amplitude <= triads[i - 1].triad_weighted_amplitude,
                "Triads should be sorted by triad_weighted_amplitude descending"
            );
        }
    }

    #[test]
    fn test_warp_triad_weights_positive() {
        let field = Array2::from_shape_fn((8, 8), |(i, j)| {
            Complex64::new((i as f64 * 0.5 + j as f64 * 0.7).cos(), 0.0)
        });
        let config = WarpRingConfig::default();
        let triads = extract_warp_triads(&field, &config, 0.0);

        for t in &triads {
            assert!(t.padic_weight > 0.0, "P-adic weight must be positive");
            assert!(t.neg_dim_weight > 0.0, "Neg-dim weight must be positive");
            assert!(
                t.triad_weighted_amplitude >= 0.0,
                "Warp weight must be non-negative"
            );
        }
    }

    #[test]
    fn test_amplitude_proxy_requires_three_modes_and_ignores_phase() {
        let mut field = Array2::zeros((8, 8));
        field[[1, 0]] = Complex64::new(2.0, 0.0);
        field[[0, 1]] = Complex64::new(3.0, 0.0);
        let config = WarpRingConfig::default();
        assert!(extract_warp_triads(&field, &config, 0.0).is_empty());
        field[[7, 7]] = Complex64::new(4.0, 0.0);
        let before = extract_warp_triads(&field, &config, 0.0);
        assert!(!before.is_empty());
        assert!(before.iter().all(|t| t.triad_amplitude_proxy == 24.0));
        field[[7, 7]] *= Complex64::new(0.0, 1.0);
        let after = extract_warp_triads(&field, &config, 0.0);
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.triad_weighted_amplitude, b.triad_weighted_amplitude);
        }
        field.mapv_inplace(|z| z * 2.0);
        let scaled = extract_warp_triads(&field, &config, 0.0);
        assert!(scaled.iter().all(|t| t.triad_amplitude_proxy == 192.0));
    }

    #[test]
    fn test_default_config() {
        let config = WarpRingConfig::default();
        assert_eq!(config.prime, 2);
        assert!(config.alpha < 0.0);
        assert!(config.epsilon > 0.0);
    }
}
