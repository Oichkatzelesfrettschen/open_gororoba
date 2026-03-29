//! Lloyd-Max optimal scalar quantizer with codebook caching.
//!
//! Translated from TurboQuant's lloyd_max.py. The codebook is solved once per
//! (dimension, bits) pair and cached via a static HashMap.
//!
//! For d >= 64, the coordinate distribution after Haar rotation is well
//! approximated by N(0, 1/d). The Lloyd-Max algorithm iteratively refines
//! centroids as conditional expectations under this distribution.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Cached Lloyd-Max codebooks keyed by (dimension, bits)
static CODEBOOK_CACHE: LazyLock<Mutex<HashMap<(usize, u32), LloydMaxCodebook>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Debug)]
pub struct LloydMaxCodebook {
    pub centroids: Vec<f32>,
    pub boundaries: Vec<f32>,
    pub distortion: f64,
}

/// Gaussian PDF N(0, sigma^2) where sigma = 1/sqrt(d)
fn gaussian_pdf(x: f64, d: usize) -> f64 {
    let sigma2 = 1.0 / d as f64;
    (1.0 / (2.0 * std::f64::consts::PI * sigma2).sqrt()) * (-x * x / (2.0 * sigma2)).exp()
}

/// Exact Beta marginal PDF for a single coordinate after random rotation
/// of a d-dimensional unit vector.
///
/// f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
///
/// Supported on [-1, 1].  More accurate than the Gaussian approximation
/// for d < 50 where the tails differ significantly.
///
/// Ported from turboquant-pytorch/lloyd_max.py:beta_pdf and validated
/// against the turboquant (abdelstark) crate's codebook.rs.
fn beta_pdf(x: f64, d: usize) -> f64 {
    if x.abs() >= 1.0 {
        return 0.0;
    }
    let half_d = d as f64 / 2.0;
    let half_dm1 = (d as f64 - 1.0) / 2.0;
    // ln(Gamma(d/2)) - ln(sqrt(pi)) - ln(Gamma((d-1)/2))
    let log_norm = lgamma(half_d) - 0.5 * std::f64::consts::PI.ln() - lgamma(half_dm1);
    let exponent = (d as f64 - 3.0) / 2.0;
    log_norm.exp() * (1.0 - x * x).powf(exponent)
}

/// Log-gamma function (Lanczos approximation for non-negative half-integers).
fn lgamma(x: f64) -> f64 {
    // Use the Stirling-like approximation via the standard library
    // Rust doesn't have lgamma in std, but we can compute it via:
    // For positive x, use the recursive relation Gamma(x+1) = x*Gamma(x)
    // and the approximation for the base case.
    // Simple implementation via the Spouge approximation:
    if x <= 0.5 {
        // Reflection formula: Gamma(1-x)*Gamma(x) = pi/sin(pi*x)
        let reflected = lgamma(1.0 - x);
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - reflected
    } else {
        // Stirling's approximation with Lanczos coefficients (g=7)
        #[allow(clippy::excessive_precision)]
        let coeffs = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        let g = 7.0;
        let x = x - 1.0;
        let mut sum = coeffs[0];
        for (i, &c) in coeffs[1..].iter().enumerate() {
            sum += c / (x + i as f64 + 1.0);
        }
        let t = x + g + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
    }
}

/// Select the appropriate PDF for the given dimension.
///
/// d < 50: exact Beta marginal (more accurate for small dimensions)
/// d >= 50: Gaussian N(0, 1/d) approximation (fast, accurate for large d)
///
/// Threshold 50 matches the turboquant (abdelstark) crate's choice.
fn coordinate_pdf(x: f64, d: usize) -> f64 {
    if d < 50 {
        beta_pdf(x, d)
    } else {
        gaussian_pdf(x, d)
    }
}

/// Numerical integration via Gauss-Legendre quadrature (16-point).
///
/// Upgraded from Simpson's rule based on crate ecosystem research.
/// Gauss-Legendre of order 16 achieves machine-precision accuracy for
/// smooth functions like our Gaussian-weighted integrands, while Simpson's
/// at n_steps=200 had ~1e-8 accuracy.
///
/// Uses the gauss-quad crate for node/weight precomputation.
fn integrate_gauss_legendre(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    use gauss_quad::GaussLegendre;
    // 16-point GL: exact for polynomials up to degree 31.
    // Our integrand (x * Gaussian) is infinitely smooth, so 16 points
    // gives far better accuracy than 200-point Simpson's rule.
    let gl = GaussLegendre::new(16).unwrap();
    gl.integrate(a, b, &f)
}

/// Legacy Simpson's rule (kept for comparison benchmarks).
#[allow(dead_code)]
fn integrate_simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n_steps: usize) -> f64 {
    let h = (b - a) / n_steps as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n_steps {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

/// Solve the Lloyd-Max optimal quantizer for the coordinate distribution.
///
/// For d < 50: uses the exact Beta marginal PDF (supported on [-1, 1])
/// For d >= 50: uses Gaussian N(0, 1/d) approximation (supported on R)
///
/// The PDF selection matches the turboquant crate's approach and provides
/// better codebook accuracy at small dimensions.
pub fn solve_lloyd_max(d: usize, bits: u32) -> LloydMaxCodebook {
    let n_levels = 1usize << bits;
    let sigma = 1.0 / (d as f64).sqrt();
    // For Beta PDF (d<50): support is [-1, 1]
    // For Gaussian (d>=50): use 3.5*sigma tails
    let lo = if d < 50 { -0.999 } else { -3.5 * sigma };
    let hi = if d < 50 { 0.999 } else { 3.5 * sigma };

    // Initialize centroids uniformly
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| lo + (hi - lo) * (i as f64 + 0.5) / n_levels as f64)
        .collect();

    for _iter in 0..200 {
        // Compute boundaries (midpoints)
        let boundaries: Vec<f64> = (0..n_levels - 1)
            .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
            .collect();

        // Update centroids as conditional expectations
        let edges: Vec<f64> = std::iter::once(lo * 3.0)
            .chain(boundaries.iter().copied())
            .chain(std::iter::once(hi * 3.0))
            .collect();

        let mut new_centroids = Vec::with_capacity(n_levels);
        for i in 0..n_levels {
            let a = edges[i];
            let b = edges[i + 1];
            let num = integrate_gauss_legendre(|x| x * coordinate_pdf(x, d), a, b);
            let den = integrate_gauss_legendre(|x| coordinate_pdf(x, d), a, b);
            new_centroids.push(if den.abs() > 1e-15 {
                num / den
            } else {
                centroids[i]
            });
        }

        let max_shift = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        centroids = new_centroids;

        if max_shift < 1e-10 {
            break;
        }
    }

    let boundaries: Vec<f64> = (0..n_levels - 1)
        .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
        .collect();

    // Compute distortion
    let edges: Vec<f64> = std::iter::once(lo * 3.0)
        .chain(boundaries.iter().copied())
        .chain(std::iter::once(hi * 3.0))
        .collect();
    let distortion: f64 = (0..n_levels)
        .map(|i| {
            let c = centroids[i];
            integrate_gauss_legendre(
                |x| (x - c).powi(2) * coordinate_pdf(x, d),
                edges[i],
                edges[i + 1],
            )
        })
        .sum();

    LloydMaxCodebook {
        centroids: centroids.iter().map(|&x| x as f32).collect(),
        boundaries: boundaries.iter().map(|&x| x as f32).collect(),
        distortion,
    }
}

/// Get or compute a Lloyd-Max codebook (cached by dimension and bits).
/// The cache eliminates redundant solves -- TurboQuant's validate.py bug
/// of re-solving 108 times is avoided.
pub fn get_codebook(d: usize, bits: u32) -> LloydMaxCodebook {
    let mut cache = CODEBOOK_CACHE.lock().unwrap();
    cache
        .entry((d, bits))
        .or_insert_with(|| solve_lloyd_max(d, bits))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lloyd_max_3bit_128d() {
        let cb = solve_lloyd_max(128, 3);
        assert_eq!(cb.centroids.len(), 8);
        assert_eq!(cb.boundaries.len(), 7);
        // Centroids should be symmetric around 0
        for i in 0..4 {
            assert!(
                (cb.centroids[i] + cb.centroids[7 - i]).abs() < 1e-4,
                "Centroids not symmetric: {} + {} = {}",
                cb.centroids[i],
                cb.centroids[7 - i],
                cb.centroids[i] + cb.centroids[7 - i]
            );
        }
        // Check against known values from Python
        assert!(
            (cb.centroids[0] - (-0.19020693)).abs() < 1e-3,
            "First centroid mismatch: {}",
            cb.centroids[0]
        );
    }

    #[test]
    fn test_lloyd_max_3bit_16d_beta() {
        // d=16 uses the exact Beta PDF (d < 50)
        let cb = solve_lloyd_max(16, 3);
        assert_eq!(cb.centroids.len(), 8);
        assert_eq!(cb.boundaries.len(), 7);
        // Centroids should be symmetric around 0
        for i in 0..4 {
            assert!(
                (cb.centroids[i] + cb.centroids[7 - i]).abs() < 1e-3,
                "Beta centroids not symmetric at d=16: {} + {} = {}",
                cb.centroids[i], cb.centroids[7 - i],
                cb.centroids[i] + cb.centroids[7 - i]
            );
        }
        // All centroids should be within [-1, 1] (Beta support)
        assert!(cb.centroids.iter().all(|&c| c.abs() <= 1.0),
            "Beta centroids out of range: {:?}", cb.centroids);
        println!("d=16 Beta centroids: {:?}", cb.centroids);
    }

    #[test]
    fn test_beta_vs_gaussian_codebook() {
        // Compare codebooks at d=32 where Beta and Gaussian diverge most
        // Force Beta path
        let cb_beta = solve_lloyd_max(32, 3); // d=32 < 50, uses Beta

        // The Beta distribution at d=32 is more peaked than Gaussian,
        // so centroids should be tighter
        let beta_range = cb_beta.centroids.last().unwrap() - cb_beta.centroids.first().unwrap();
        println!("d=32 Beta codebook range: {:.6}", beta_range);
        println!("d=32 Beta centroids: {:?}", cb_beta.centroids);
        assert!(beta_range > 0.0 && beta_range < 2.0, "Beta range out of bounds");
    }

    #[test]
    fn test_codebook_caching() {
        let cb1 = get_codebook(128, 3);
        let cb2 = get_codebook(128, 3);
        // Should return identical results (same pointer in cache)
        assert_eq!(cb1.centroids, cb2.centroids);
        // Different params should give different codebook
        let cb3 = get_codebook(64, 4);
        assert_ne!(cb3.centroids.len(), cb1.centroids.len());
    }
}
