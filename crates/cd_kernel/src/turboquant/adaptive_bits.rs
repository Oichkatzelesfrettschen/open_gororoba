//! Per-token adaptive bit allocation via CD associator.
//!
//! Tokens with high residual associator norm (||[r_t, r_{t+1}, r_{t+2}]||)
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

use super::cd_fidelity::residual_associator_per_token;
use super::pipeline::{TurboQuantMSE, MseCompressed};

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

    let mut buf = vec![0.0f64; 2 * d];
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
    let total: u64 = allocation.iter().map(|a| match a {
        BitAllocation::Base => base_bits as u64,
        BitAllocation::Promoted => (base_bits + 1) as u64,
    }).sum();
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
    let mut buf = vec![0.0f64; 2 * d];
    let mut uniform_mse_sum = 0.0f64;
    let mut residuals = Vec::with_capacity(n);

    for v in vectors {
        let compressed = tq_uniform.quantize(v, &mut buf);
        let mut recon = vec![0.0f64; d];
        tq_uniform.dequantize(&compressed, &mut buf, &mut recon);

        let mse: f64 = v.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
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

        let mse: f64 = vectors[i].iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
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
        (0..n).map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect()).collect()
    }

    #[test]
    fn test_allocate_bits_fractions() {
        let d = 16;
        let residuals: Vec<Vec<f64>> = (0..100)
            .map(|t| (0..d).map(|i| ((t * 7 + i) as f64 * 0.1).sin()).collect())
            .collect();

        let allocation = allocate_bits(&residuals, d, 0.25);
        let n_promoted = allocation.iter().filter(|&&a| a == BitAllocation::Promoted).count();
        assert_eq!(n_promoted, 25, "25% of 100 = 25 promoted");

        let allocation_half = allocate_bits(&residuals, d, 0.5);
        let n_promoted_half = allocation_half.iter().filter(|&&a| a == BitAllocation::Promoted).count();
        assert_eq!(n_promoted_half, 50);
    }

    #[test]
    fn test_average_bits() {
        let allocation = vec![
            BitAllocation::Base, BitAllocation::Base,
            BitAllocation::Base, BitAllocation::Promoted,
        ];
        let avg = average_bits(&allocation, 3);
        assert!((avg - 3.25).abs() < 1e-10, "3 base + 1 promoted at bits=3 -> avg 3.25");
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
            adaptive_mse, uniform_mse
        );
    }

    #[test]
    fn test_adaptive_quantize_mixed_bits() {
        let d = 32;
        let vectors = random_vectors(10, d, 99);
        let allocation = vec![
            BitAllocation::Base, BitAllocation::Promoted,
            BitAllocation::Base, BitAllocation::Base,
            BitAllocation::Promoted, BitAllocation::Base,
            BitAllocation::Base, BitAllocation::Base,
            BitAllocation::Promoted, BitAllocation::Base,
        ];

        let results = adaptive_quantize(&vectors, &allocation, 3, 42, true);
        assert_eq!(results.len(), 10);

        // Check that promoted tokens use 4-bit codebook
        assert_eq!(results[1].1, 4); // promoted
        assert_eq!(results[0].1, 3); // base
        assert_eq!(results[4].1, 4); // promoted
    }
}
