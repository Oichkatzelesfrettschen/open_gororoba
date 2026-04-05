//! E8 rotation vs Haar-random: KS test for decorrelation quality.
//!
//! Thesis T6: E8-structured rotation achieves decorrelation comparable
//! to Haar-random at d=128.
//!
//! Falsification: KS test rejects equality (p < 0.01) with E8 having
//! higher pairwise correlations.

use super::{
    e8_rotation::{e8_block_rotate, generate_e8_roots, select_diverse_roots},
    rotation::{generate_haar_rotation, rotate},
};

/// Compute pairwise correlation matrix for rotated coordinates.
///
/// For n rotated vectors of dimension d, compute the Pearson correlation
/// between each pair of coordinate dimensions (i, j).
/// Returns the distribution of |correlation| values as a sorted Vec.
fn pairwise_correlations(rotated: &[Vec<f64>], d: usize) -> Vec<f64> {
    let n = rotated.len();
    if n < 3 {
        return vec![];
    }

    let mut correlations = Vec::new();

    // Compute mean and std for each coordinate
    let mut means = vec![0.0f64; d];
    let mut stds = vec![0.0f64; d];

    for v in rotated {
        for (i, &val) in v.iter().enumerate() {
            means[i] += val;
        }
    }
    for m in means.iter_mut() {
        *m /= n as f64;
    }

    for v in rotated {
        for (i, &val) in v.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2);
        }
    }
    for s in stds.iter_mut() {
        *s = (*s / (n - 1) as f64).sqrt();
    }

    // Compute pairwise correlations for a sample of coordinate pairs
    // (full d*d is expensive; sample 500 random pairs)
    let n_pairs = 500.min(d * (d - 1) / 2);
    let mut pair_idx = 0;

    for i in 0..d {
        for j in (i + 1)..d {
            if pair_idx >= n_pairs {
                break;
            }
            if stds[i] < 1e-10 || stds[j] < 1e-10 {
                continue;
            }

            let mut cov = 0.0f64;
            for v in rotated {
                cov += (v[i] - means[i]) * (v[j] - means[j]);
            }
            cov /= (n - 1) as f64;
            let corr = cov / (stds[i] * stds[j]);
            correlations.push(corr.abs());
            pair_idx += 1;
        }
        if pair_idx >= n_pairs {
            break;
        }
    }

    correlations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    correlations
}

/// Two-sample Kolmogorov-Smirnov test statistic.
///
/// Returns the KS statistic D = max|F1(x) - F2(x)| and approximate p-value.
/// Both samples must be sorted in ascending order.
fn ks_test(sample1: &[f64], sample2: &[f64]) -> (f64, f64) {
    let n1 = sample1.len();
    let n2 = sample2.len();
    if n1 == 0 || n2 == 0 {
        return (0.0, 1.0);
    }

    let mut d_max = 0.0f64;
    let mut i = 0usize;
    let mut j = 0usize;

    while i < n1 && j < n2 {
        let f1 = (i + 1) as f64 / n1 as f64;
        let f2 = (j + 1) as f64 / n2 as f64;

        if sample1[i] <= sample2[j] {
            d_max = d_max.max((f1 - j as f64 / n2 as f64).abs());
            i += 1;
        } else {
            d_max = d_max.max((i as f64 / n1 as f64 - f2).abs());
            j += 1;
        }
    }

    // Handle remaining elements
    while i < n1 {
        let f1 = (i + 1) as f64 / n1 as f64;
        d_max = d_max.max((f1 - 1.0).abs());
        i += 1;
    }
    while j < n2 {
        let f2 = (j + 1) as f64 / n2 as f64;
        d_max = d_max.max((1.0 - f2).abs());
        j += 1;
    }

    // Approximate p-value (large-sample asymptotic)
    let en = ((n1 * n2) as f64 / (n1 + n2) as f64).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d_max;
    // Kolmogorov distribution approximation
    let p_value = 2.0 * (-2.0 * lambda * lambda).exp();
    let p_value = p_value.clamp(0.0, 1.0);

    (d_max, p_value)
}

/// Result of the E8 vs Haar decorrelation comparison.
#[derive(Clone, Debug)]
pub struct DecorrelationResult {
    /// Mean |correlation| for E8 rotation.
    pub e8_mean_corr: f64,
    /// Mean |correlation| for Haar rotation.
    pub haar_mean_corr: f64,
    /// KS test statistic (D).
    pub ks_statistic: f64,
    /// KS test p-value.
    pub ks_p_value: f64,
    /// Whether E8 passes (p > 0.01, cannot reject equality).
    pub e8_passes: bool,
    /// Number of vectors tested.
    pub n_vectors: usize,
    /// Dimension.
    pub dim: usize,
}

/// Run the E8 vs Haar decorrelation KS test (Thesis T6).
///
/// 1. Generate n random d-dimensional vectors
/// 2. Rotate all with E8-block rotation and with Haar rotation
/// 3. Compute pairwise |correlation| distributions for both
/// 4. Two-sample KS test on the distributions
/// 5. E8 passes if p > 0.01
pub fn e8_vs_haar_ks_test(d: usize, n_vectors: usize, seed: u64) -> DecorrelationResult {
    assert_eq!(d, 128, "E8 block rotation requires d=128");

    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;

    // Generate random vectors
    let vectors: Vec<Vec<f64>> = (0..n_vectors)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    // E8 rotation
    let all_roots = generate_e8_roots();
    let roots = select_diverse_roots(&all_roots, seed);
    let e8_rotated: Vec<Vec<f64>> = vectors
        .iter()
        .map(|v| {
            let mut out = vec![0.0f64; d];
            e8_block_rotate(v, &roots, &mut out);
            out
        })
        .collect();

    // Haar rotation
    let pi = generate_haar_rotation(d, seed + 1000);
    let haar_rotated: Vec<Vec<f64>> = vectors
        .iter()
        .map(|v| {
            let mut out = vec![0.0f64; d];
            rotate(v, &pi, d, &mut out);
            out
        })
        .collect();

    // Compute correlation distributions
    let e8_corrs = pairwise_correlations(&e8_rotated, d);
    let haar_corrs = pairwise_correlations(&haar_rotated, d);

    let e8_mean = if e8_corrs.is_empty() {
        0.0
    } else {
        e8_corrs.iter().sum::<f64>() / e8_corrs.len() as f64
    };
    let haar_mean = if haar_corrs.is_empty() {
        0.0
    } else {
        haar_corrs.iter().sum::<f64>() / haar_corrs.len() as f64
    };

    // KS test
    let (ks_d, ks_p) = ks_test(&e8_corrs, &haar_corrs);

    DecorrelationResult {
        e8_mean_corr: e8_mean,
        haar_mean_corr: haar_mean,
        ks_statistic: ks_d,
        ks_p_value: ks_p,
        e8_passes: ks_p > 0.01,
        n_vectors,
        dim: d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks_test_identical_samples() {
        let s: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let (d, p) = ks_test(&s, &s);
        assert!(
            d < 0.05,
            "Identical samples should have D near 0, got {}",
            d
        );
        assert!(
            p > 0.5,
            "Identical samples should have high p-value, got {}",
            p
        );
    }

    #[test]
    fn test_ks_test_different_samples() {
        let s1: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let s2: Vec<f64> = (0..100).map(|i| (i as f64 / 100.0).powi(2)).collect();
        let (d, _p) = ks_test(&s1, &s2);
        assert!(
            d > 0.1,
            "Different distributions should have D > 0.1, got {}",
            d
        );
    }

    #[test]
    fn test_e8_vs_haar_decorrelation() {
        // This is the core Thesis T6 validation.
        // Use a modest sample (500 vectors) for test speed.
        let result = e8_vs_haar_ks_test(128, 500, 42);

        println!("=== E8 vs Haar Decorrelation (Thesis T6) ===");
        println!("  E8 mean |corr|:   {:.6}", result.e8_mean_corr);
        println!("  Haar mean |corr|: {:.6}", result.haar_mean_corr);
        println!("  KS statistic D:   {:.6}", result.ks_statistic);
        println!("  KS p-value:       {:.6}", result.ks_p_value);
        println!("  E8 passes (p>0.01): {}", result.e8_passes);

        // Document result either way (honest science)
        if result.e8_passes {
            println!("  RESULT: Cannot reject H0 -- E8 decorrelation comparable to Haar");
        } else {
            println!("  RESULT: Reject H0 -- E8 decorrelation significantly different from Haar");
            println!("  (This is an expected possible outcome for the speculative T6 hypothesis)");
        }
    }
}
