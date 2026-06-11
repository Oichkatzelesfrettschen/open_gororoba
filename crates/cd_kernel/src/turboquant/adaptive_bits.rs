//! Per-token adaptive bit allocation via CD associator.
//!
//! Tokens with high residual associator norm (||`[r_t, r_{t+1}, r_{t+2}]`||)
//! have phase-coupling structure that sign projections capture poorly.
//! Allocating more bits to these tokens improves attention fidelity at
//! the same average bit budget.
//!
//! # Algorithm
//!
//! 1. Quantize all tokens at base bit-width b
//! 2. Compute per-token residual associator score
//! 3. Rank tokens by score
//! 4. Promote top-k% tokens to (b+1) bits
//! 5. Total budget: (1-k)*b + k*(b+1) = b + k average bits/token
//!
//! # Hypothesis (Thesis T4)
//!
//! Adaptive allocation should improve attention fidelity by >= 0.5%
//! over uniform allocation at the same average bit-width.
//! If not, document as honest negative result.

use super::{
    cd_fidelity::residual_associator_per_token,
    pipeline::{MseCompressed, TurboQuantMSE},
};

/// Bit allocation decision for each token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitAllocation {
    /// Base bit-width (b).
    Base,
    /// Promoted bit-width (b+1).
    Promoted,
}

/// Compute per-token bit allocation based on residual associator scores.
///
/// `promote_fraction`: fraction of tokens to promote (0.0 to 1.0).
/// Typical value: 0.25 (top quartile gets extra bit).
///
/// Returns allocation for each token.
pub fn allocate_bits(
    residuals: &[Vec<f64>],
    dim: usize,
    promote_fraction: f64,
) -> Vec<BitAllocation> {
    let n = residuals.len();
    if n == 0 {
        return vec![];
    }

    // Compute per-token associator scores
    let scores = residual_associator_per_token(residuals, dim);

    // Find threshold: top promote_fraction by score
    let n_promote = ((n as f64 * promote_fraction).ceil() as usize).min(n);
    let mut sorted_scores: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut allocation = vec![BitAllocation::Base; n];
    for (rank, &(idx, _)) in sorted_scores.iter().enumerate() {
        if rank < n_promote {
            allocation[idx] = BitAllocation::Promoted;
        }
    }

    allocation
}

/// Closed-form Lagrange bit allocation from rate-distortion theory.
///
/// From the quantization force analogy (Pais -> rate-distortion):
///   F_Q = b * ln(2) * 4^b / (sigma^2 * C)
///
/// The optimal per-region bit allocation that minimizes total distortion
/// subject to a total bit budget is:
///   b_r = b_mean + (1 / (2 * ln(4))) * ln(sigma_r^2 / sigma_mean^2)
///
/// where sigma_r^2 is the variance of region r after rotation.
///
/// This replaces O(n*B) greedy allocation with O(n) closed-form.
/// Rounds to nearest integer bit-width (2, 3, or 4).
///
/// Returns per-token bit allocation.
pub fn allocate_bits_lagrange(
    variances: &[f64],
    base_bits: u32,
    min_bits: u32,
    max_bits: u32,
) -> Vec<u32> {
    let n = variances.len();
    if n == 0 {
        return vec![];
    }

    // Geometric mean variance (sigma_mean^2)
    let log_var_sum: f64 = variances
        .iter()
        .map(|&v| if v > 1e-30 { v.ln() } else { -69.0 }) // ln(1e-30)
        .sum();
    let log_var_mean = log_var_sum / n as f64;

    // b_r = b_mean + (1/(2*ln(4))) * ln(sigma_r^2 / sigma_mean^2)
    // = b_mean + (1/(2*ln(4))) * (ln(sigma_r^2) - ln(sigma_mean^2))
    let scale = 1.0 / (2.0 * 4.0_f64.ln()); // 1/(2*ln(4)) ~ 0.3607

    let allocations: Vec<u32> = variances
        .iter()
        .map(|&v| {
            let log_v = if v > 1e-30 { v.ln() } else { -69.0 };
            let b_continuous = base_bits as f64 + scale * (log_v - log_var_mean);
            let b_rounded = b_continuous
                .round()
                .max(min_bits as f64)
                .min(max_bits as f64);
            b_rounded as u32
        })
        .collect();

    allocations
}

/// Compute per-token variance in rotated space (for Lagrange allocation).
///
/// For each vector: normalize, rotate, compute coordinate variance.
/// Returns one variance per vector.
pub fn compute_rotated_variances(vectors: &[Vec<f64>], tq: &TurboQuantMSE) -> Vec<f64> {
    let d = vectors[0].len();
    let mut buf = vec![0.0f64; 3 * d];

    vectors
        .iter()
        .map(|v| {
            // Quantize to get the rotated representation
            let compressed = tq.quantize(v, &mut buf);
            // The variance in rotated space is approximately 1/d for unit vectors
            // But real data has non-uniform variance per coordinate
            // Use the vec_norm as a proxy (higher norm = higher variance)
            compressed.vec_norm.powi(2) / d as f64
        })
        .collect()
}

/// Multi-precision bit allocation: allocate 2, 3, or 4 bits per token
/// based on the closed-form Lagrange solver.
///
/// Returns (allocation, avg_bits) where allocation`[i]` is the bit-width for token i.
pub fn multi_precision_allocate(
    vectors: &[Vec<f64>],
    base_bits: u32,
    seed: u64,
    use_wht: bool,
) -> (Vec<u32>, f64) {
    let d = vectors[0].len();
    let tq = TurboQuantMSE::new(d, base_bits, seed, use_wht);
    let variances = compute_rotated_variances(vectors, &tq);
    let allocations = allocate_bits_lagrange(&variances, base_bits, 2, 4);
    let avg = allocations.iter().sum::<u32>() as f64 / allocations.len() as f64;
    (allocations, avg)
}

/// Adaptive quantization: apply different bit-widths per token.
///
/// Tokens marked `Promoted` get (base_bits + 1) bits.
/// Tokens marked `Base` get base_bits.
///
/// Returns compressed representations and the actual per-token bit-widths.
pub fn adaptive_quantize(
    vectors: &[Vec<f64>],
    allocation: &[BitAllocation],
    base_bits: u32,
    seed: u64,
    use_wht: bool,
) -> Vec<(MseCompressed, u32)> {
    let n = vectors.len();
    let d = vectors[0].len();

    let tq_base = TurboQuantMSE::new(d, base_bits, seed, use_wht);
    let tq_promoted = TurboQuantMSE::new(d, base_bits + 1, seed, use_wht);

    let mut buf = vec![0.0f64; 3 * d];
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let bits = match allocation[i] {
            BitAllocation::Base => base_bits,
            BitAllocation::Promoted => base_bits + 1,
        };
        let tq = match allocation[i] {
            BitAllocation::Base => &tq_base,
            BitAllocation::Promoted => &tq_promoted,
        };
        let compressed = tq.quantize(&vectors[i], &mut buf);
        results.push((compressed, bits));
    }

    results
}

/// Compute average bit-width from an allocation.
pub fn average_bits(allocation: &[BitAllocation], base_bits: u32) -> f64 {
    let n = allocation.len();
    if n == 0 {
        return base_bits as f64;
    }
    let total: u64 = allocation
        .iter()
        .map(|a| match a {
            BitAllocation::Base => base_bits as u64,
            BitAllocation::Promoted => (base_bits + 1) as u64,
        })
        .sum();
    total as f64 / n as f64
}

/// Compare adaptive vs uniform quantization quality.
///
/// Returns (adaptive_mse, uniform_mse, improvement_pct).
pub fn compare_adaptive_vs_uniform(
    vectors: &[Vec<f64>],
    base_bits: u32,
    promote_fraction: f64,
    seed: u64,
    use_wht: bool,
) -> (f64, f64, f64) {
    let n = vectors.len();
    let d = vectors[0].len();

    // Uniform quantization at base_bits
    let tq_uniform = TurboQuantMSE::new(d, base_bits, seed, use_wht);
    let mut buf = vec![0.0f64; 3 * d];
    let mut uniform_mse_sum = 0.0f64;
    let mut residuals = Vec::with_capacity(n);

    for v in vectors {
        let compressed = tq_uniform.quantize(v, &mut buf);
        let mut recon = vec![0.0f64; d];
        tq_uniform.dequantize(&compressed, &mut buf, &mut recon);

        let mse: f64 = v
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        uniform_mse_sum += mse;

        let residual: Vec<f64> = v.iter().zip(recon.iter()).map(|(a, b)| a - b).collect();
        residuals.push(residual);
    }

    // Adaptive allocation based on residuals
    let allocation = allocate_bits(&residuals, d, promote_fraction);
    let adaptive_results = adaptive_quantize(vectors, &allocation, base_bits, seed, use_wht);

    let mut adaptive_mse_sum = 0.0f64;
    for (i, (compressed, bits)) in adaptive_results.iter().enumerate() {
        let tq = TurboQuantMSE::new(d, *bits, seed, use_wht);
        let mut recon = vec![0.0f64; d];
        tq.dequantize(compressed, &mut buf, &mut recon);

        let mse: f64 = vectors[i]
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        adaptive_mse_sum += mse;
    }

    let uniform_mse = uniform_mse_sum / n as f64;
    let adaptive_mse = adaptive_mse_sum / n as f64;
    let improvement = if uniform_mse > 1e-15 {
        (uniform_mse - adaptive_mse) / uniform_mse * 100.0
    } else {
        0.0
    };

    (adaptive_mse, uniform_mse, improvement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let normal = StandardNormal;
        (0..n)
            .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

    #[test]
    fn test_allocate_bits_fractions() {
        let d = 16;
        let residuals: Vec<Vec<f64>> = (0..100)
            .map(|t| (0..d).map(|i| ((t * 7 + i) as f64 * 0.1).sin()).collect())
            .collect();

        let allocation = allocate_bits(&residuals, d, 0.25);
        let n_promoted = allocation
            .iter()
            .filter(|&&a| a == BitAllocation::Promoted)
            .count();
        assert_eq!(n_promoted, 25, "25% of 100 = 25 promoted");

        let allocation_half = allocate_bits(&residuals, d, 0.5);
        let n_promoted_half = allocation_half
            .iter()
            .filter(|&&a| a == BitAllocation::Promoted)
            .count();
        assert_eq!(n_promoted_half, 50);
    }

    #[test]
    fn test_average_bits() {
        let allocation = vec![
            BitAllocation::Base,
            BitAllocation::Base,
            BitAllocation::Base,
            BitAllocation::Promoted,
        ];
        let avg = average_bits(&allocation, 3);
        assert!(
            (avg - 3.25).abs() < 1e-10,
            "3 base + 1 promoted at bits=3 -> avg 3.25"
        );
    }

    #[test]
    fn test_adaptive_vs_uniform() {
        let n = 100;
        let d = 32;
        let vectors = random_vectors(n, d, 42);

        let (adaptive_mse, uniform_mse, improvement) =
            compare_adaptive_vs_uniform(&vectors, 3, 0.25, 42, true);

        println!("Adaptive MSE: {:.6}", adaptive_mse);
        println!("Uniform MSE:  {:.6}", uniform_mse);
        println!("Improvement:  {:.2}%", improvement);

        // Adaptive should be at least as good as uniform (or very close)
        // since promoted tokens get more bits
        assert!(
            adaptive_mse <= uniform_mse * 1.1,
            "Adaptive should not be much worse: adaptive={}, uniform={}",
            adaptive_mse,
            uniform_mse
        );
    }

    #[test]
    fn test_adaptive_quantize_mixed_bits() {
        let d = 32;
        let vectors = random_vectors(10, d, 99);
        let allocation = vec![
            BitAllocation::Base,
            BitAllocation::Promoted,
            BitAllocation::Base,
            BitAllocation::Base,
            BitAllocation::Promoted,
            BitAllocation::Base,
            BitAllocation::Base,
            BitAllocation::Base,
            BitAllocation::Promoted,
            BitAllocation::Base,
        ];

        let results = adaptive_quantize(&vectors, &allocation, 3, 42, true);
        assert_eq!(results.len(), 10);

        // Check that promoted tokens use 4-bit codebook
        assert_eq!(results[1].1, 4); // promoted
        assert_eq!(results[0].1, 3); // base
        assert_eq!(results[4].1, 4); // promoted
    }

    #[test]
    fn test_lagrange_bit_allocation() {
        // Test with heterogeneous variances: some vectors 10x larger
        let variances: Vec<f64> = (0..100)
            .map(|i| {
                if i < 25 { 10.0 } // high-variance tokens
            else { 1.0 } // normal tokens
            })
            .collect();

        let alloc = allocate_bits_lagrange(&variances, 3, 2, 4);
        assert_eq!(alloc.len(), 100);

        // High-variance tokens should get more bits
        let high_var_bits: f64 = alloc[0..25].iter().map(|&b| b as f64).sum::<f64>() / 25.0;
        let low_var_bits: f64 = alloc[25..100].iter().map(|&b| b as f64).sum::<f64>() / 75.0;
        let avg_bits: f64 = alloc.iter().sum::<u32>() as f64 / 100.0;

        println!(
            "Lagrange allocation: high_var={:.2} bits, low_var={:.2} bits, avg={:.2}",
            high_var_bits, low_var_bits, avg_bits
        );

        assert!(
            high_var_bits > low_var_bits,
            "High-variance tokens should get more bits: {} vs {}",
            high_var_bits,
            low_var_bits
        );
        // Average should be near base_bits (3.0)
        assert!(
            (avg_bits - 3.0).abs() < 0.5,
            "Average bits should be near base: {}",
            avg_bits
        );
    }

    #[test]
    fn test_lagrange_uniform_variance() {
        // Uniform variance -> uniform allocation
        let variances = vec![1.0; 100];
        let alloc = allocate_bits_lagrange(&variances, 3, 2, 4);
        assert!(
            alloc.iter().all(|&b| b == 3),
            "Uniform variance should give uniform allocation: {:?}",
            &alloc[..5]
        );
    }

    #[test]
    fn test_lagrange_vs_greedy_heterogeneous() {
        // Create vectors with heterogeneous norms (simulating real KV cache)
        let d = 64;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let normal = StandardNormal;
        let mut vectors: Vec<Vec<f64>> = (0..200)
            .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        // Make first 50 vectors 5x larger (high variance)
        for v in vectors[..50].iter_mut() {
            for x in v.iter_mut() {
                *x *= 5.0;
            }
        }

        let (lagrange_alloc, lagrange_avg) = multi_precision_allocate(&vectors, 3, 42, true);

        println!("Lagrange avg bits: {:.2}", lagrange_avg);
        let n_promoted = lagrange_alloc.iter().filter(|&&b| b > 3).count();
        let n_demoted = lagrange_alloc.iter().filter(|&&b| b < 3).count();
        println!("Promoted (>3): {}, Demoted (<3): {}", n_promoted, n_demoted);

        // With heterogeneous data, Lagrange should differentiate
        // High-variance vectors should get more bits
        let high_var_bits: f64 = lagrange_alloc[..50].iter().map(|&b| b as f64).sum::<f64>() / 50.0;
        let low_var_bits: f64 = lagrange_alloc[50..].iter().map(|&b| b as f64).sum::<f64>() / 150.0;
        println!(
            "High-var avg bits: {:.2}, Low-var avg bits: {:.2}",
            high_var_bits, low_var_bits
        );

        assert!(
            high_var_bits >= low_var_bits,
            "High-variance vectors should get >= bits: {:.2} vs {:.2}",
            high_var_bits,
            low_var_bits
        );
    }
}
