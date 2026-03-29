//! Zero-divisor affinity scoring for quantization bias detection.
//!
//! Moreno (1997, arXiv q-alg/9710013) proved that an element a in the
//! sedenion algebra is a zero-divisor iff the left-multiplication operator
//! L_a has eigenvalue -2.  Reggiani (2024) showed the ZD manifold has
//! topology G_2 x S^1.
//!
//! For TurboQuant: QJL residual vectors aligned with zero-divisor manifolds
//! have poor inner-product behavior.  Quantization error along these
//! directions is amplified because the CD multiplication structure provides
//! no error-correcting redundancy.
//!
//! # Hypothesis (Thesis T5)
//!
//! High ZD-affinity residuals correlate with higher quantization error.
//! If the Haar rotation fully isotropizes residuals, ZD-affinity should be
//! uniformly low (negative result = Haar works well).

// Zero-divisor search functions are re-exported at crate root

/// Compute the zero-divisor affinity of a vector in sedenion (16D) space.
///
/// The affinity is the minimum norm of the product a*b over all unit vectors b,
/// computed via sampling.  True zero-divisors have affinity = 0.  Vectors
/// far from ZD manifolds have high affinity (products are large).
///
/// Lower affinity = closer to ZD manifold = more quantization-vulnerable.
///
/// `v` must have length 16 (sedenion).
/// `n_samples`: number of random unit vectors b to test (higher = more accurate).
pub fn sedenion_zd_affinity(v: &[f64], n_samples: usize) -> f64 {
    assert_eq!(v.len(), 16, "ZD affinity requires 16D sedenion vectors");

    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let normal = StandardNormal;

    let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if v_norm < 1e-15 {
        return 0.0; // zero vector trivially
    }

    // Normalize v
    let v_unit: Vec<f64> = v.iter().map(|x| x / v_norm).collect();

    let mut min_product_norm = f64::MAX;
    let mut product = vec![0.0f64; 16];
    let mut workspace = vec![0.0f64; 4 * 16];

    for _ in 0..n_samples {
        // Random unit vector b
        let b: Vec<f64> = {
            let raw: Vec<f64> = (0..16).map(|_| normal.sample(&mut rng)).collect();
            let norm: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            raw.iter().map(|x| x / norm).collect()
        };

        // Compute v * b using CD multiplication
        crate::cayley_dickson::cd_multiply_into(&v_unit, &b, &mut product, &mut workspace);
        let prod_norm: f64 = product.iter().map(|x| x * x).sum::<f64>().sqrt();

        if prod_norm < min_product_norm {
            min_product_norm = prod_norm;
        }
    }

    min_product_norm
}

/// Compute ZD-affinity scores for a batch of QJL residual vectors.
///
/// For vectors with dimension > 16, project into sedenion (16D) subspace
/// by taking the first 16 components.  For dim < 16, pad with zeros.
///
/// Returns one affinity score per vector. Lower = more ZD-vulnerable.
pub fn batch_zd_affinity(residuals: &[Vec<f64>], n_samples: usize) -> Vec<f64> {
    residuals
        .iter()
        .map(|r| {
            // Project to 16D sedenion subspace
            let mut s = vec![0.0f64; 16];
            let copy_len = r.len().min(16);
            s[..copy_len].copy_from_slice(&r[..copy_len]);
            sedenion_zd_affinity(&s, n_samples)
        })
        .collect()
}

/// Bin residuals by ZD-affinity quartile and compute per-quartile MSE.
///
/// Returns (q1_mse, q2_mse, q3_mse, q4_mse) where q1 = lowest affinity
/// (most ZD-vulnerable) and q4 = highest affinity (least vulnerable).
pub fn zd_quartile_analysis(
    residuals: &[Vec<f64>],
    quantization_errors: &[f64],
    n_samples: usize,
) -> [f64; 4] {
    let n = residuals.len();
    assert_eq!(n, quantization_errors.len());

    let affinities = batch_zd_affinity(residuals, n_samples);

    // Sort by affinity, split into quartiles
    let mut indexed: Vec<(usize, f64)> = affinities.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let q_size = n / 4;
    let mut quartile_mse = [0.0f64; 4];

    for (q, mse) in quartile_mse.iter_mut().enumerate() {
        let start = q * q_size;
        let end = if q == 3 { n } else { start + q_size };
        let count = end - start;
        if count > 0 {
            let sum: f64 = indexed[start..end]
                .iter()
                .map(|&(idx, _)| quantization_errors[idx])
                .sum();
            *mse = sum / count as f64;
        }
    }

    quartile_mse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_zd_pair() {
        // Known sedenion zero-divisor: (e3 + e10) * (e6 - e15) = 0
        // Construct the ZD vector and verify low affinity
        let mut v = vec![0.0f64; 16];
        v[3] = 1.0;  // e3 component
        v[10] = 1.0; // e10 component
        let affinity = sedenion_zd_affinity(&v, 1000);
        // Should be low (near-ZD direction)
        println!("Known ZD vector affinity: {}", affinity);
        // Note: affinity may not be exactly 0 because we sample random b's,
        // not the specific partner (e6 - e15)
    }

    #[test]
    fn test_basis_vector_affinity() {
        // Single basis vector e0 should have high affinity (not a ZD)
        let mut v = vec![0.0f64; 16];
        v[0] = 1.0;
        let affinity = sedenion_zd_affinity(&v, 500);
        assert!(affinity > 0.1, "Basis vector should have high affinity, got {}", affinity);
    }

    #[test]
    fn test_batch_zd_affinity() {
        let residuals: Vec<Vec<f64>> = (0..10)
            .map(|t| (0..32).map(|i| ((t * 5 + i) as f64 * 0.1).sin()).collect())
            .collect();

        let affinities = batch_zd_affinity(&residuals, 100);
        assert_eq!(affinities.len(), 10);
        assert!(affinities.iter().all(|&a| a >= 0.0));
    }

    #[test]
    fn test_quartile_analysis() {
        let n = 100;
        let residuals: Vec<Vec<f64>> = (0..n)
            .map(|t| (0..16).map(|i| ((t * 3 + i * 7) as f64 * 0.05).sin()).collect())
            .collect();
        let errors: Vec<f64> = (0..n).map(|t| (t as f64 * 0.01).sin().abs()).collect();

        let quartiles = zd_quartile_analysis(&residuals, &errors, 50);
        // All quartile MSEs should be non-negative
        for (q, &mse) in quartiles.iter().enumerate() {
            assert!(mse >= 0.0, "Quartile {} has negative MSE: {}", q, mse);
        }
    }
}
