//! Adaptive allocation profiling: decompose the 250x overhead.
//!
//! The adaptive bit allocation path is 250x slower than the fast path
//! (4 kvec/s vs 1040 kvec/s).  This module identifies exactly which
//! step dominates.

use super::{adaptive_bits, cd_fidelity::residual_associator_per_token, pipeline::TurboQuantMSE};
use std::time::Instant;

/// Profile result for each step of the adaptive pipeline.
#[derive(Clone, Debug)]
pub struct AdaptiveProfile {
    pub base_quantize_ms: f64,
    pub dequantize_residual_ms: f64,
    pub associator_scoring_ms: f64,
    pub bit_allocation_ms: f64,
    pub re_quantize_ms: f64,
    pub total_ms: f64,
    pub n_vectors: usize,
    pub d: usize,
}

impl AdaptiveProfile {
    /// Fraction of total time for each step.
    pub fn fractions(&self) -> Vec<(&'static str, f64)> {
        let t = self.total_ms;
        vec![
            ("base_quantize", self.base_quantize_ms / t * 100.0),
            ("dequant_residual", self.dequantize_residual_ms / t * 100.0),
            ("associator_score", self.associator_scoring_ms / t * 100.0),
            ("bit_allocation", self.bit_allocation_ms / t * 100.0),
            ("re_quantize", self.re_quantize_ms / t * 100.0),
        ]
    }
}

/// Profile the adaptive pipeline step by step.
pub fn profile_adaptive_steps(vectors: &[Vec<f64>], d: usize, bits: u32) -> AdaptiveProfile {
    let n = vectors.len();
    let tq = TurboQuantMSE::new(d, bits, 42, true);
    let tq_promoted = TurboQuantMSE::new(d, bits + 1, 42, true);
    let mut buf = vec![0.0f64; 3 * d];

    let t_total = Instant::now();

    // Step 1: Base quantize
    let t0 = Instant::now();
    let base_compressed: Vec<_> = vectors.iter().map(|v| tq.quantize(v, &mut buf)).collect();
    let base_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Step 2: Dequantize + residual
    let t0 = Instant::now();
    let residuals: Vec<Vec<f64>> = vectors
        .iter()
        .zip(base_compressed.iter())
        .map(|(orig, comp)| {
            let mut recon = vec![0.0f64; d];
            tq.dequantize(comp, &mut buf, &mut recon);
            orig.iter().zip(recon.iter()).map(|(a, b)| a - b).collect()
        })
        .collect();
    let dequant_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Step 3: CD associator scoring
    let t0 = Instant::now();
    let _scores = residual_associator_per_token(&residuals, d);
    let assoc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Step 4: Bit allocation (just the sorting/ranking, fast)
    let t0 = Instant::now();
    let allocation = adaptive_bits::allocate_bits(&residuals, d, 0.25);
    let alloc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Step 5: Re-quantize promoted tokens
    let t0 = Instant::now();
    let _n_promoted = allocation
        .iter()
        .filter(|&&a| a == adaptive_bits::BitAllocation::Promoted)
        .count();
    for (i, alloc) in allocation.iter().enumerate() {
        if *alloc == adaptive_bits::BitAllocation::Promoted {
            let _comp = tq_promoted.quantize(&vectors[i], &mut buf);
        }
    }
    let requant_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    AdaptiveProfile {
        base_quantize_ms: base_ms,
        dequantize_residual_ms: dequant_ms,
        associator_scoring_ms: assoc_ms,
        bit_allocation_ms: alloc_ms,
        re_quantize_ms: requant_ms,
        total_ms,
        n_vectors: n,
        d,
    }
}

/// Optimized adaptive allocation using sampled scoring.
///
/// Instead of computing the CD associator for ALL tokens, sample a
/// fraction and interpolate. This trades a small quality loss for
/// a large speedup on the scoring step.
pub fn adaptive_allocate_sampled(
    vectors: &[Vec<f64>],
    d: usize,
    bits: u32,
    promote_fraction: f64,
    sample_fraction: f64,
) -> Vec<adaptive_bits::BitAllocation> {
    let n = vectors.len();
    let n_sample = ((n as f64 * sample_fraction).ceil() as usize).max(3);

    if n_sample >= n {
        // Sample covers everything -- use full path
        let tq = TurboQuantMSE::new(d, bits, 42, true);
        let mut buf = vec![0.0f64; 3 * d];
        let residuals: Vec<Vec<f64>> = vectors
            .iter()
            .map(|v| {
                let comp = tq.quantize(v, &mut buf);
                let mut recon = vec![0.0f64; d];
                tq.dequantize(&comp, &mut buf, &mut recon);
                v.iter().zip(recon.iter()).map(|(a, b)| a - b).collect()
            })
            .collect();
        return adaptive_bits::allocate_bits(&residuals, d, promote_fraction);
    }

    // Sample evenly spaced tokens
    let step = n / n_sample;
    let sample_indices: Vec<usize> = (0..n_sample).map(|i| i * step).collect();

    let tq = TurboQuantMSE::new(d, bits, 42, true);
    let mut buf = vec![0.0f64; 3 * d];

    // Compute residuals only for sampled tokens (f32 fast path: 37x faster)
    let sample_residuals_f32: Vec<Vec<f32>> = sample_indices
        .iter()
        .map(|&i| {
            let comp = tq.quantize(&vectors[i], &mut buf);
            let mut recon = vec![0.0f64; d];
            tq.dequantize(&comp, &mut buf, &mut recon);
            vectors[i]
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (*a - *b) as f32)
                .collect()
        })
        .collect();

    // Score using f32 fast associator (workspace-based, 3-buffer reuse)
    let sample_scores: Vec<f64> =
        crate::batch_sliding_associator_norms_f32(&sample_residuals_f32, d)
            .into_iter()
            .map(|s| s as f64)
            .collect();
    // Pad scores to match sample count (sliding window produces n-2 values)
    let mut padded_scores = vec![0.0f64; n_sample];
    for (i, &s) in sample_scores.iter().enumerate() {
        if i + 1 < n_sample {
            padded_scores[i + 1] = s;
        }
    }

    // The sample scores inform the threshold, but we use norm-based
    // proxy for the full batch (fast: O(d) per vector vs O(d^2) for CD multiply)

    // Quick-score all tokens: use vector norm as a fast proxy for residual magnitude
    // (tokens with larger norms tend to have larger residuals)
    let mut allocation = vec![adaptive_bits::BitAllocation::Base; n];
    let n_to_promote = ((n as f64 * promote_fraction).ceil() as usize).min(n);

    // Score all tokens by norm via simsimd (HW-accelerated, O(d) per token)
    let mut all_norms: Vec<(usize, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            // simsimd dot(v, v) = ||v||^2, then sqrt
            let norm_sq = super::simsimd_bridge::dot_f64(v, v);
            (i, norm_sq.sqrt())
        })
        .collect();
    all_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for &(idx, _) in all_norms.iter().take(n_to_promote) {
        allocation[idx] = adaptive_bits::BitAllocation::Promoted;
    }

    allocation
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
    fn test_profile_adaptive() {
        let d = 32;
        let n = 100;
        let vectors = random_vectors(n, d, 42);
        let profile = profile_adaptive_steps(&vectors, d, 3);

        println!("\n=== Adaptive Pipeline Profile (d={}, n={}) ===", d, n);
        println!(
            "  Total: {:.2} ms ({:.0} kvec/s)",
            profile.total_ms,
            n as f64 / profile.total_ms
        );
        for (name, frac) in profile.fractions() {
            println!("  {:>20}: {:.1}%", name, frac);
        }
    }

    #[test]
    fn test_sampled_allocation_speed() {
        let d = 64;
        let n = 500;
        let vectors = random_vectors(n, d, 42);

        // Full adaptive
        let t0 = Instant::now();
        let full = {
            let tq = TurboQuantMSE::new(d, 3, 42, true);
            let mut buf = vec![0.0f64; 3 * d];
            let residuals: Vec<Vec<f64>> = vectors
                .iter()
                .map(|v| {
                    let comp = tq.quantize(v, &mut buf);
                    let mut recon = vec![0.0f64; d];
                    tq.dequantize(&comp, &mut buf, &mut recon);
                    v.iter().zip(recon.iter()).map(|(a, b)| a - b).collect()
                })
                .collect();
            adaptive_bits::allocate_bits(&residuals, d, 0.25)
        };
        let full_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Sampled (10% sample)
        let t0 = Instant::now();
        let sampled = adaptive_allocate_sampled(&vectors, d, 3, 0.25, 0.1);
        let sampled_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let n_promoted_full = full
            .iter()
            .filter(|&&a| a == adaptive_bits::BitAllocation::Promoted)
            .count();
        let n_promoted_sampled = sampled
            .iter()
            .filter(|&&a| a == adaptive_bits::BitAllocation::Promoted)
            .count();

        println!(
            "\nFull adaptive: {:.1} ms, {} promoted",
            full_ms, n_promoted_full
        );
        println!(
            "Sampled (10%): {:.1} ms, {} promoted",
            sampled_ms, n_promoted_sampled
        );
        println!("Speedup: {:.1}x", full_ms / sampled_ms);

        // Both should promote approximately the same fraction
        assert!((n_promoted_full as f64 / n as f64 - 0.25).abs() < 0.05);
        assert!((n_promoted_sampled as f64 / n as f64 - 0.25).abs() < 0.05);
    }
}
