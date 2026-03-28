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

/// Numerical integration via Simpson's rule (no scipy dependency)
fn integrate_simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n_steps: usize) -> f64 {
    let h = (b - a) / n_steps as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n_steps {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

/// Solve the Lloyd-Max optimal quantizer for N(0, 1/d)
pub fn solve_lloyd_max(d: usize, bits: u32) -> LloydMaxCodebook {
    let n_levels = 1usize << bits;
    let sigma = 1.0 / (d as f64).sqrt();
    let lo = -3.5 * sigma;
    let hi = 3.5 * sigma;
    let n_simpson = 200;

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
            let num = integrate_simpson(|x| x * gaussian_pdf(x, d), a, b, n_simpson);
            let den = integrate_simpson(|x| gaussian_pdf(x, d), a, b, n_simpson);
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
            integrate_simpson(
                |x| (x - c).powi(2) * gaussian_pdf(x, d),
                edges[i],
                edges[i + 1],
                n_simpson,
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
