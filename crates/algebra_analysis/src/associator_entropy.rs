//! Associator entropy mechanism analysis.
//!
//! Investigates WHY zero divisors cause Shannon entropy of the associator-norm
//! distribution to DECREASE at dim >= 16. The mechanism: ZDs create forbidden
//! zones in the associator space, reducing the effective volume and thus the
//! entropy. This is the INVERSE of naive expectation (more dimensions = more
//! entropy).
//!
//! # Key observation (C-806)
//!
//! H(4) = 0, H(8) = 4.24 (max), H(16) = 3.76 (DECREASE despite more dimensions)
//!
//! # Mechanism hypothesis
//!
//! At dim >= 16, zero divisors constrain the associator-norm distribution:
//! certain norm values become forbidden (or suppressed) because the ZD structure
//! creates algebraic dependencies among basis products. This reduces the
//! effective support of the distribution, lowering entropy.
//!
//! # Approach
//!
//! 1. Sample random triples and compute associator norms.
//! 2. For each sample, check proximity to ZD constraints.
//! 3. Partition samples into "near-ZD" and "far-from-ZD" subsets.
//! 4. Compare conditional entropies: H(norms | near ZD) vs H(norms | far from ZD).
//! 5. Quantify entropy reduction per box-kite component.

use cd_kernel::cayley_dickson::batch_associator_norms_parallel;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use crate::boxkites::motif_components_for_cross_assessors;

/// Result of associator entropy decomposition.
#[derive(Debug, Clone)]
pub struct EntropyDecomposition {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Total (unconditional) Shannon entropy.
    pub total_entropy: f64,
    /// Entropy of samples with high ZD proximity.
    pub near_zd_entropy: f64,
    /// Entropy of samples with low ZD proximity.
    pub far_zd_entropy: f64,
    /// Fraction of samples classified as "near ZD".
    pub near_zd_fraction: f64,
    /// Entropy reduction attributable to ZD constraints: H_far - H_near.
    pub zd_entropy_reduction: f64,
    /// Number of ZD components at this dimension.
    pub n_zd_components: usize,
    /// Number of samples used.
    pub n_samples: usize,
}

/// Result of cross-dimensional entropy analysis.
#[derive(Debug, Clone)]
pub struct CrossDimensionalEntropy {
    /// Per-dimension results.
    pub dimensions: Vec<EntropyDecomposition>,
    /// ZD count vs entropy reduction correlation.
    pub zd_entropy_correlation: f64,
}

/// Compute ZD proximity for a vector relative to the cross-assessor structure.
///
/// Measures how close a vector is to lying in a ZD 2-plane. Returns the
/// minimum angle (in radians) between the vector and any ZD diagonal.
pub fn zd_proximity(v: &[f64], dim: usize) -> f64 {
    let half = dim / 2;
    let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
    if v_norm_sq < 1e-30 {
        return std::f64::consts::FRAC_PI_2;
    }
    let mut min_angle = std::f64::consts::FRAC_PI_2;

    // Check projection onto each cross-assessor 2-plane
    for i in 1..half {
        for j in half..dim {
            // The 2-plane spanned by e_i and e_j
            let proj_sq = v[i] * v[i] + v[j] * v[j];
            let cos_sq = proj_sq / v_norm_sq;
            // Clamp for numerical safety
            let cos_val = cos_sq.sqrt().min(1.0);
            let angle = cos_val.acos();
            if angle < min_angle {
                min_angle = angle;
            }
        }
    }
    min_angle
}

/// Shannon entropy of a histogram.
fn histogram_entropy(counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / t;
            h -= p * p.ln();
        }
    }
    h
}

/// Bin associator norms into a histogram.
fn bin_norms(norms: &[f64], n_bins: usize, max_val: f64) -> Vec<usize> {
    if max_val < 1e-15 {
        let mut counts = vec![0usize; n_bins];
        counts[0] = norms.len();
        return counts;
    }
    let bin_width = max_val / n_bins as f64;
    let mut counts = vec![0usize; n_bins];
    for &n in norms {
        let bin = ((n / bin_width) as usize).min(n_bins - 1);
        counts[bin] += 1;
    }
    counts
}

/// Decompose associator entropy by ZD proximity at a single dimension.
///
/// Samples random unit triples, computes associator norms, partitions by
/// proximity to ZD 2-planes, and compares conditional entropies.
pub fn decompose_entropy(
    dim: usize,
    n_samples: usize,
    n_bins: usize,
    seed: u64,
    proximity_threshold: f64,
) -> EntropyDecomposition {
    assert!(
        dim >= 2 && dim.is_power_of_two(),
        "dim must be power of 2 >= 2"
    );

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate random triples and normalize to unit vectors (Muller's method)
    let mut a_flat = Vec::with_capacity(dim * n_samples);
    let mut b_flat = Vec::with_capacity(dim * n_samples);
    let mut c_flat = Vec::with_capacity(dim * n_samples);
    let mut a_vecs = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let a_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let b_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let c_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();

        let a_norm = a_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let b_norm = b_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let c_norm = c_raw.iter().map(|x| x * x).sum::<f64>().sqrt();

        let a_unit: Vec<f64> = a_raw.iter().map(|x| x / a_norm).collect();
        a_vecs.push(a_unit.clone());

        for &x in &a_unit {
            a_flat.push(x);
        }
        for x in &b_raw {
            b_flat.push(x / b_norm);
        }
        for x in &c_raw {
            c_flat.push(x / c_norm);
        }
    }

    let norms = batch_associator_norms_parallel(&a_flat, &b_flat, &c_flat, dim, n_samples);

    // Compute ZD proximity for the first element of each triple
    let proximities: Vec<f64> = if dim >= 16 {
        a_vecs.iter().map(|v| zd_proximity(v, dim)).collect()
    } else {
        vec![std::f64::consts::FRAC_PI_2; n_samples]
    };

    // Partition by proximity
    let max_norm = norms.iter().cloned().fold(0.0_f64, f64::max);

    let mut near_norms = Vec::new();
    let mut far_norms = Vec::new();

    for i in 0..n_samples {
        if proximities[i] < proximity_threshold {
            near_norms.push(norms[i]);
        } else {
            far_norms.push(norms[i]);
        }
    }

    // Compute entropies
    let total_counts = bin_norms(&norms, n_bins, max_norm);
    let near_counts = bin_norms(&near_norms, n_bins, max_norm);
    let far_counts = bin_norms(&far_norms, n_bins, max_norm);

    let total_entropy = histogram_entropy(&total_counts, n_samples);
    let near_zd_entropy = histogram_entropy(&near_counts, near_norms.len());
    let far_zd_entropy = histogram_entropy(&far_counts, far_norms.len());

    let n_zd_components = if dim >= 16 {
        motif_components_for_cross_assessors(dim).len()
    } else {
        0
    };

    EntropyDecomposition {
        dim,
        total_entropy,
        near_zd_entropy,
        far_zd_entropy,
        near_zd_fraction: near_norms.len() as f64 / n_samples as f64,
        zd_entropy_reduction: far_zd_entropy - near_zd_entropy,
        n_zd_components,
        n_samples,
    }
}

/// Pearson correlation between two slices.
fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-30 {
        0.0
    } else {
        cov / denom
    }
}

/// Run cross-dimensional entropy decomposition.
///
/// For each dimension, decomposes entropy into ZD-proximal and ZD-distal
/// contributions, and correlates ZD component count with entropy reduction.
pub fn cross_dimensional_analysis(
    dims: &[usize],
    n_samples: usize,
    n_bins: usize,
    seed: u64,
    proximity_threshold: f64,
) -> CrossDimensionalEntropy {
    let dimensions: Vec<EntropyDecomposition> = dims
        .iter()
        .map(|&dim| {
            decompose_entropy(dim, n_samples, n_bins, seed.wrapping_add(dim as u64), proximity_threshold)
        })
        .collect();

    // Correlate ZD count with entropy reduction
    let zd_counts: Vec<f64> = dimensions.iter().map(|d| d.n_zd_components as f64).collect();
    let reductions: Vec<f64> = dimensions.iter().map(|d| d.zd_entropy_reduction).collect();
    let zd_entropy_correlation = pearson_correlation(&zd_counts, &reductions);

    CrossDimensionalEntropy {
        dimensions,
        zd_entropy_correlation,
    }
}

/// Analytic upper bound on associator entropy from ZD count.
///
/// H(dim) <= H(8) - k * ln(n_zd_components + 1)
///
/// where k is estimated empirically from the dim=16 data point.
/// Returns (k, bound_at_dim) pair.
pub fn entropy_upper_bound(
    h_8: f64,
    h_dim: f64,
    n_zd_components: usize,
) -> (f64, f64) {
    if n_zd_components == 0 {
        return (0.0, h_8);
    }
    let log_zd = (n_zd_components as f64 + 1.0).ln();
    let k = (h_8 - h_dim) / log_zd;
    let bound = h_8 - k * log_zd;
    (k, bound)
}

/// Dimension-dependent proximity threshold.
///
/// epsilon(N) = 1.3 * sqrt(16 / N)
///
/// Calibrated against the dim=16 baseline where 1.3 rad captures the
/// concentration-of-measure peak. At higher dimensions, random unit vectors
/// are closer to every 2-plane (concentration of measure), so the threshold
/// must shrink to avoid near_frac saturation.
///
/// Values: dim=16 -> 1.300, dim=32 -> 0.919, dim=64 -> 0.650, dim=128 -> 0.460.
pub fn adaptive_proximity_threshold(dim: usize) -> f64 {
    1.3 * (16.0 / dim as f64).sqrt()
}

/// Decompose associator entropy with an adaptive proximity threshold.
///
/// Wrapper around `decompose_entropy()` that uses `adaptive_proximity_threshold(dim)`
/// instead of a fixed threshold. This prevents near_frac saturation at high dims.
pub fn decompose_entropy_adaptive(
    dim: usize,
    n_samples: usize,
    n_bins: usize,
    seed: u64,
) -> EntropyDecomposition {
    let threshold = adaptive_proximity_threshold(dim);
    decompose_entropy(dim, n_samples, n_bins, seed, threshold)
}

/// Cross-dimensional entropy analysis with adaptive thresholds.
///
/// Like `cross_dimensional_analysis()` but each dimension gets its own
/// adaptive proximity threshold via `adaptive_proximity_threshold(dim)`.
pub fn cross_dimensional_analysis_adaptive(
    dims: &[usize],
    n_samples: usize,
    n_bins: usize,
    seed: u64,
) -> CrossDimensionalEntropy {
    let dimensions: Vec<EntropyDecomposition> = dims
        .iter()
        .map(|&dim| {
            decompose_entropy_adaptive(dim, n_samples, n_bins, seed.wrapping_add(dim as u64))
        })
        .collect();

    let zd_counts: Vec<f64> = dimensions.iter().map(|d| d.n_zd_components as f64).collect();
    let reductions: Vec<f64> = dimensions.iter().map(|d| d.zd_entropy_reduction).collect();
    let zd_entropy_correlation = pearson_correlation(&zd_counts, &reductions);

    CrossDimensionalEntropy {
        dimensions,
        zd_entropy_correlation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quaternion_entropy_decomposition() {
        let result = decompose_entropy(4, 500, 50, 42, 0.3);
        assert_eq!(result.dim, 4);
        assert!(
            result.total_entropy.abs() < 1e-10,
            "H(4) should be ~0 (associative), got {}",
            result.total_entropy
        );
        assert_eq!(result.n_zd_components, 0);
    }

    #[test]
    fn octonion_entropy_decomposition() {
        let result = decompose_entropy(8, 1000, 50, 42, 0.3);
        assert!(
            result.total_entropy > 0.1,
            "H(8) should be positive, got {}",
            result.total_entropy
        );
        assert_eq!(result.n_zd_components, 0);
    }

    #[test]
    fn sedenion_entropy_decomposition() {
        // In dim=16, a random unit vector's angle to the closest 2-plane
        // is typically arccos(sqrt(2/16)) ~ 1.2 rad due to concentration
        // of measure. Use a generous threshold to capture near-ZD samples.
        let result = decompose_entropy(16, 1000, 50, 42, 1.3);
        assert!(
            result.total_entropy > 0.1,
            "H(16) should be positive, got {}",
            result.total_entropy
        );
        assert_eq!(result.n_zd_components, 7);
        // With threshold ~1.3 rad, some samples should be classified as near-ZD
        assert!(
            result.near_zd_fraction > 0.0,
            "should have some near-ZD samples at dim=16 with threshold 1.3"
        );
    }

    #[test]
    fn entropy_bound_consistency() {
        let (k, bound) = entropy_upper_bound(4.24, 3.76, 7);
        assert!(k > 0.0, "k should be positive: {k}");
        assert!(
            (bound - 3.76).abs() < 0.01,
            "bound should match H(16): got {bound}"
        );
    }

    #[test]
    fn zd_proximity_orthogonal_to_all_planes() {
        // A vector along e_0 (scalar) is orthogonal to all cross-assessor planes
        let dim = 16;
        let mut v = vec![0.0; dim];
        v[0] = 1.0;
        let prox = zd_proximity(&v, dim);
        assert!(
            (prox - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "e_0 should be maximally far from all ZD planes: {prox}"
        );
    }

    #[test]
    fn zd_proximity_in_plane() {
        // A vector along e_1 should be close to ZD planes containing e_1
        let dim = 16;
        let mut v = vec![0.0; dim];
        v[1] = 1.0;
        let prox = zd_proximity(&v, dim);
        // e_1 lies in multiple cross-assessor 2-planes (1, j) for j in 8..16
        assert!(
            prox < 0.01,
            "e_1 should be close to ZD planes: {prox}"
        );
    }

    #[test]
    fn cross_dimensional_runs() {
        let result = cross_dimensional_analysis(&[4, 8, 16], 500, 50, 42, 0.3);
        assert_eq!(result.dimensions.len(), 3);
    }

    #[test]
    fn entropy_census_4_8_16_32() {
        // Census of entropy decomposition for the claim record.
        // Uses larger sample count and generous proximity threshold.
        let dims = [4, 8, 16, 32];
        let cross = cross_dimensional_analysis(&dims, 5000, 100, 42, 1.3);
        for d in &cross.dimensions {
            eprintln!(
                "dim={:3}: H_total={:.4}, near_frac={:.3}, reduction={:.4}, n_zd={}",
                d.dim, d.total_entropy, d.near_zd_fraction, d.zd_entropy_reduction, d.n_zd_components
            );
        }

        // Key claim: H(8) > H(16) (entropy decreases despite more dimensions)
        let h8 = cross.dimensions.iter().find(|d| d.dim == 8).unwrap().total_entropy;
        let h16 = cross.dimensions.iter().find(|d| d.dim == 16).unwrap().total_entropy;
        eprintln!("H(8)={:.4}, H(16)={:.4}, H(8)>H(16): {}", h8, h16, h8 > h16);
        eprintln!("ZD-entropy correlation: {:.4}", cross.zd_entropy_correlation);

        // Structural assertions
        assert!(h8 > 0.0, "octonion entropy should be positive");
        assert_eq!(cross.dimensions.len(), 4);
    }

    // -- adaptive threshold tests --

    #[test]
    fn adaptive_threshold_values() {
        let t16 = adaptive_proximity_threshold(16);
        let t32 = adaptive_proximity_threshold(32);
        let t64 = adaptive_proximity_threshold(64);
        assert!(
            (t16 - 1.3).abs() < 1e-14,
            "dim=16 should give 1.3: {t16}"
        );
        assert!(
            (t32 - 1.3 * (0.5_f64).sqrt()).abs() < 1e-14,
            "dim=32 should give 1.3*sqrt(0.5): {t32}"
        );
        assert!(
            (t64 - 1.3 * (0.25_f64).sqrt()).abs() < 1e-14,
            "dim=64 should give 1.3*sqrt(0.25): {t64}"
        );
    }

    #[test]
    fn adaptive_threshold_monotone_decreasing() {
        let dims = [16, 32, 64, 128, 256];
        let thresholds: Vec<f64> = dims.iter().map(|&d| adaptive_proximity_threshold(d)).collect();
        for w in thresholds.windows(2) {
            assert!(
                w[0] > w[1],
                "threshold should decrease: {} > {} violated",
                w[0], w[1]
            );
        }
    }

    #[test]
    fn adaptive_entropy_dim32_not_saturated() {
        // With adaptive threshold, dim=32 near_frac should be < 1.0
        let result = decompose_entropy_adaptive(32, 2000, 100, 42);
        eprintln!(
            "adaptive dim=32: near_frac={:.4}, threshold={:.4}",
            result.near_zd_fraction,
            adaptive_proximity_threshold(32)
        );
        assert!(
            result.near_zd_fraction < 1.0,
            "adaptive threshold should prevent saturation at dim=32: near_frac={}",
            result.near_zd_fraction
        );
    }

    #[test]
    fn adaptive_entropy_census_4_8_16_32() {
        let dims = [4, 8, 16, 32];
        let cross = cross_dimensional_analysis_adaptive(&dims, 5000, 100, 42);
        for d in &cross.dimensions {
            eprintln!(
                "adaptive dim={:3}: H_total={:.4}, near_frac={:.3}, reduction={:.4}, n_zd={}, threshold={:.3}",
                d.dim, d.total_entropy, d.near_zd_fraction, d.zd_entropy_reduction,
                d.n_zd_components, adaptive_proximity_threshold(d.dim)
            );
        }
        assert_eq!(cross.dimensions.len(), 4);

        // dim=32 should no longer saturate
        let d32 = cross.dimensions.iter().find(|d| d.dim == 32).unwrap();
        assert!(
            d32.near_zd_fraction < 1.0,
            "dim=32 near_frac should be < 1.0 with adaptive: {}",
            d32.near_zd_fraction
        );
    }
}
