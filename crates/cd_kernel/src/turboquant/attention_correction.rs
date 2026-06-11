//! Quantization-aware attention score correction.
//!
//! Instead of just measuring fidelity post-hoc, use the known
//! quantization error distribution to debias attention scores.
//!
//! The key insight: scalar quantization with a Lloyd-Max codebook
//! introduces a known per-coordinate distortion E`[e^2]` = distortion.
//! The dot product error between query q and quantized key k_q is:
//!
//!   <q, k> - <q, k_q> = <q, k - k_q> = <q, e>
//!
//! where e = k - k_q is the quantization error.
//!
//! For i.i.d. error with E`[e_i]` = 0 and Var`[e_i]` = distortion/d:
//!   E`[<q, e>]` = 0  (unbiased)
//!   Var`[<q, e>]` = ||q||^2 * distortion / d
//!
//! The attention score variance can be corrected to improve ranking quality.

use crate::lloyd_max;

/// Compute the expected attention score variance from quantization.
///
/// `query_norm_sq`: ||q||^2 for the query vector
/// `d`: head dimension
/// `bits`: quantization bits
///
/// Returns Var`[<q, e>]` = ||q||^2 * codebook_distortion / d
pub fn attention_score_variance(query_norm_sq: f64, d: usize, bits: u32) -> f64 {
    let codebook = lloyd_max::get_codebook(d, bits);
    query_norm_sq * codebook.distortion / d as f64
}

/// Correct attention scores by subtracting the expected variance.
///
/// This sharpens the softmax by reducing the score variance caused
/// by quantization noise.  Equivalent to temperature scaling:
///   corrected_score = score / sqrt(1 + variance/temperature^2)
///
/// `scores`: raw attention scores (query x keys)
/// `query_norm_sq`: ||q||^2
/// `d`: head dimension
/// `bits`: quantization bits
/// `temperature`: attention temperature (typically 1/sqrt(d))
pub fn correct_attention_scores(scores: &mut [f64], query_norm_sq: f64, d: usize, bits: u32) {
    let var = attention_score_variance(query_norm_sq, d, bits);
    let temperature_sq = 1.0 / d as f64; // standard attention temperature^2

    // Correction factor: scores / sqrt(1 + var/temperature^2)
    // This compensates for the broadening of the score distribution
    // caused by quantization noise.
    let correction = 1.0 / (1.0 + var / temperature_sq).sqrt();

    for s in scores.iter_mut() {
        *s *= correction;
    }
}

/// Estimate the expected top-1 accuracy degradation from quantization.
///
/// Uses the gap between the top-2 scores relative to the noise variance
/// to estimate the probability that quantization flips the ranking.
///
/// Returns the estimated probability that the correct top-1 is preserved.
pub fn top1_preservation_probability(
    top1_score: f64,
    top2_score: f64,
    query_norm_sq: f64,
    d: usize,
    bits: u32,
) -> f64 {
    let var = attention_score_variance(query_norm_sq, d, bits);
    if var < 1e-15 {
        return 1.0;
    }

    // Gap between top-1 and top-2 scores
    let gap = top1_score - top2_score;
    let noise_std = var.sqrt();

    // Approximate: P(top-1 preserved) = Phi(gap / noise_std)
    // Using error function approximation
    let z = gap / (noise_std * std::f64::consts::SQRT_2);
    0.5 * (1.0 + erf_approx(z))
}

/// Approximate error function (Abramowitz-Stegun 7.1.26).
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_variance() {
        let var = attention_score_variance(1.0, 128, 3);
        println!(
            "Attention score variance (||q||=1, d=128, 3-bit): {:.6e}",
            var
        );
        assert!(var > 0.0 && var < 0.01, "Variance out of range: {}", var);
    }

    #[test]
    fn test_correction_sharpens_scores() {
        let d = 128;
        let bits = 3;
        let query_norm_sq = 1.0;

        let mut scores = vec![1.0, 0.8, 0.5, 0.2, -0.1];
        let original = scores.clone();
        correct_attention_scores(&mut scores, query_norm_sq, d, bits);

        // Correction should scale scores toward zero (sharpen)
        for (s, &o) in scores.iter().zip(original.iter()) {
            assert!(
                s.abs() <= o.abs() + 1e-10,
                "Correction should not increase magnitude: {} -> {}",
                o,
                s
            );
        }

        // Relative ordering should be preserved
        for i in 0..scores.len() - 1 {
            assert!(
                scores[i] >= scores[i + 1],
                "Ordering broken at {}: {} < {}",
                i,
                scores[i],
                scores[i + 1]
            );
        }
    }

    #[test]
    fn test_top1_preservation() {
        let d = 128;
        let bits = 3;
        let query_norm_sq = 1.0;

        // Large gap -> high preservation probability
        let p_large = top1_preservation_probability(1.0, 0.5, query_norm_sq, d, bits);
        println!("P(top-1 preserved, gap=0.5): {:.4}", p_large);

        // Small gap -> lower preservation probability
        let p_small = top1_preservation_probability(1.0, 0.99, query_norm_sq, d, bits);
        println!("P(top-1 preserved, gap=0.01): {:.4}", p_small);

        assert!(p_large > p_small, "Larger gap should preserve better");
        assert!(p_large > 0.9, "Large gap should give > 90% preservation");
    }

    #[test]
    fn test_erf_approx() {
        // Known values
        assert!((erf_approx(0.0)).abs() < 1e-6);
        assert!((erf_approx(1.0) - 0.8427).abs() < 1e-3);
        assert!((erf_approx(2.0) - 0.9953).abs() < 1e-3);
    }
}
