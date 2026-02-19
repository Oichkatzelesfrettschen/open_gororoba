//! Cayley-Dickson associator entropy census.
//!
//! Computes Shannon entropy of the associator-norm distribution for CD algebras
//! at each dimension.  The key prediction: a sharp entropy jump at dim=16
//! (octonion -> sedenion transition) because loss of the alternative property
//! introduces zero divisors and a non-trivial associator-norm distribution.

use crate::construction::cayley_dickson::{
    batch_associator_norms_parallel, find_zero_divisors,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Shannon entropy of the associator-norm distribution at a given CD dimension.
///
/// 1. Samples `n_samples` random unit triples (a, b, c) in CD(dim).
/// 2. Computes associator norms via parallel batch.
/// 3. Histograms into `n_bins` bins over [0, max_norm].
/// 4. Returns H = -sum(p_i * ln(p_i)) where p_i = count_i / total.
///
/// For dim <= 8 (associative or alternative): all norms cluster near 0, H ~ 0.
/// For dim >= 16: non-trivial distribution, H > 0.
pub fn associator_entropy(dim: usize, n_samples: usize, n_bins: usize, seed: u64) -> f64 {
    assert!(dim >= 2 && dim.is_power_of_two(), "dim must be power of 2 >= 2");
    assert!(n_bins >= 2, "need at least 2 bins");
    assert!(n_samples >= 1, "need at least 1 sample");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate random triples and normalize to unit vectors
    let mut a_flat = Vec::with_capacity(dim * n_samples);
    let mut b_flat = Vec::with_capacity(dim * n_samples);
    let mut c_flat = Vec::with_capacity(dim * n_samples);

    for _ in 0..n_samples {
        // Generate random vector and normalize
        let a_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let b_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let c_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();

        let a_norm = a_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let b_norm = b_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let c_norm = c_raw.iter().map(|x| x * x).sum::<f64>().sqrt();

        for &x in &a_raw {
            a_flat.push(x / a_norm);
        }
        for &x in &b_raw {
            b_flat.push(x / b_norm);
        }
        for &x in &c_raw {
            c_flat.push(x / c_norm);
        }
    }

    let norms = batch_associator_norms_parallel(&a_flat, &b_flat, &c_flat, dim, n_samples);

    // Find max norm for bin range
    let max_norm = norms.iter().cloned().fold(0.0_f64, f64::max);
    if max_norm < 1e-15 {
        // All norms effectively zero -> entropy is 0
        return 0.0;
    }

    // Histogram
    let bin_width = max_norm / n_bins as f64;
    let mut counts = vec![0usize; n_bins];
    for &n in &norms {
        let bin = ((n / bin_width) as usize).min(n_bins - 1);
        counts[bin] += 1;
    }

    // Shannon entropy
    let total = n_samples as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            h -= p * p.ln();
        }
    }

    h
}

/// Fraction of 2-blade basis-pair products that are zero divisors.
///
/// Uses `find_zero_divisors` with atol=1e-10.
/// For dim <= 8: returns 0.0 (no ZDs).
/// For dim >= 16: returns positive fraction.
pub fn zero_divisor_density(dim: usize) -> f64 {
    if dim < 16 {
        return 0.0;
    }
    let zds = find_zero_divisors(dim, 1e-10);
    // Total possible 2-blade pairs: C(dim,2) * C(dim,2) * 2 (positive and negative blade)
    let n_pairs = dim * (dim - 1) / 2;
    let total = n_pairs * n_pairs * 2;
    zds.len() as f64 / total as f64
}

/// Phase transition analysis across CD dimensions.
///
/// Returns Vec of (dim, entropy, zd_density, delta_H) tuples.
/// delta_H is the entropy jump from the previous dimension in the list.
pub fn phase_transition_analysis(
    dims: &[usize],
    n_samples: usize,
    n_bins: usize,
    seed: u64,
) -> Vec<(usize, f64, f64, f64)> {
    let mut results = Vec::with_capacity(dims.len());
    let mut prev_h = 0.0;

    for (i, &dim) in dims.iter().enumerate() {
        let h = associator_entropy(dim, n_samples, n_bins, seed.wrapping_add(dim as u64));
        let zd = if dim <= 32 {
            // ZD search is O(dim^4); only feasible for small dims
            zero_divisor_density(dim)
        } else {
            f64::NAN
        };
        let delta_h = if i == 0 { 0.0 } else { h - prev_h };
        results.push((dim, h, zd, delta_h));
        prev_h = h;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_entropy_near_zero() {
        // Quaternions are associative -> all associator norms are 0
        let h = associator_entropy(4, 500, 50, 42);
        assert!(h.abs() < 1e-10, "quaternion entropy should be 0, got {}", h);
    }

    #[test]
    fn test_octonion_entropy_positive() {
        // Octonions are alternative but NOT associative.
        // Random triples have nonzero associator -> H(8) > 0.
        let h = associator_entropy(8, 500, 50, 42);
        assert!(h > 0.1, "octonion entropy should be positive (non-associative), got {}", h);
    }

    #[test]
    fn test_sedenion_entropy_positive() {
        // Sedenions are NOT alternative -> non-trivial associator distribution
        let h = associator_entropy(16, 1000, 50, 42);
        assert!(h > 0.1, "sedenion entropy should be positive, got {}", h);
    }

    #[test]
    fn test_phase_transition_at_dim_8() {
        // The primary phase transition is 4->8: associative -> non-associative.
        // H(4) = 0 (quaternions are associative), H(8) > 0 (octonions are not).
        let dims = vec![4, 8, 16];
        let results = phase_transition_analysis(&dims, 1000, 50, 42);
        let (_, h4, _, _) = results[0];
        let (_, h8, _, _) = results[1];
        let (_, h16, _, _) = results[2];
        assert!(h4.abs() < 1e-10, "H(4) should be 0 (associative)");
        assert!(h8 > 0.1, "H(8) should be positive (non-associative)");
        assert!(h16 > 0.1, "H(16) should be positive");
    }

    #[test]
    fn test_zd_density_zero_for_octonions() {
        assert_eq!(zero_divisor_density(8), 0.0);
    }

    #[test]
    fn test_zd_density_positive_for_sedenions() {
        let d = zero_divisor_density(16);
        assert!(d > 0.0, "sedenion ZD density should be positive, got {}", d);
    }
}
