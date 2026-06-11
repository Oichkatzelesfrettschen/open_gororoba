//! Per-head mixed-precision bit allocation.
//!
//! Different attention heads have different quantization sensitivity.
//! Some heads are highly structured (low associator -> easy to quantize),
//! others are chaotic (high associator -> need more bits).
//!
//! This module extends the per-token adaptive allocation to per-head
//! granularity: score each head's sensitivity, then allocate bits
//! to meet a total budget.
//!
//! # Algorithm
//!
//! 1. For each head h, compute mean CD associator over its key vectors
//! 2. Rank heads by sensitivity (higher associator = more sensitive)
//! 3. Allocate: sensitive heads get (base_bits + 1), others get base_bits
//! 4. Total budget: n_heads * base_bits + n_promoted (within budget)
//!
//! # Connection to RaanA (arXiv 2504.03717)
//!
//! RaanA's AllocateBits solves mixed-precision as an integer programming
//! problem. Our CD-based approach is a greedy heuristic that uses the
//! associator as a proxy for quantization sensitivity.

use super::cd_fidelity::residual_associator_per_token;

/// Per-head sensitivity score.
#[derive(Clone, Debug)]
pub struct HeadSensitivity {
    /// Head index.
    pub head_idx: usize,
    /// Mean CD associator across all tokens in this head.
    pub mean_associator: f64,
    /// Allocated bit-width for this head.
    pub bits: u32,
}

/// Score each head's quantization sensitivity using CD associator.
///
/// `residuals_per_head`: residuals`[head_idx]` = Vec of per-token residual vectors.
/// Returns per-head sensitivity scores (sorted by sensitivity, descending).
pub fn score_head_sensitivity(
    residuals_per_head: &[Vec<Vec<f64>>],
    dim: usize,
) -> Vec<HeadSensitivity> {
    let mut scores: Vec<HeadSensitivity> = residuals_per_head
        .iter()
        .enumerate()
        .map(|(h, residuals)| {
            let token_scores = residual_associator_per_token(residuals, dim);
            let mean = if token_scores.is_empty() {
                0.0
            } else {
                token_scores.iter().sum::<f64>() / token_scores.len() as f64
            };
            HeadSensitivity {
                head_idx: h,
                mean_associator: mean,
                bits: 0, // filled by allocator
            }
        })
        .collect();

    // Sort by sensitivity (most sensitive first)
    scores.sort_by(|a, b| {
        b.mean_associator
            .partial_cmp(&a.mean_associator)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores
}

/// Allocate bits per head to meet a total budget.
///
/// `base_bits`: minimum bits per head.
/// `budget_total_bits`: total bit budget (n_heads * base_bits + surplus).
/// `n_heads`: number of attention heads.
///
/// Returns the allocated bits per head (indexed by head_idx).
pub fn allocate_per_head_bits(
    sensitivities: &mut [HeadSensitivity],
    base_bits: u32,
    max_bits: u32,
    budget_extra_bits: usize,
) -> Vec<u32> {
    let n_heads = sensitivities.len();

    // Start all at base
    for s in sensitivities.iter_mut() {
        s.bits = base_bits;
    }

    // Promote the most sensitive heads (already sorted by sensitivity)
    let mut extra_used = 0;
    for s in sensitivities.iter_mut() {
        if extra_used >= budget_extra_bits {
            break;
        }
        if s.bits < max_bits {
            s.bits += 1;
            extra_used += 1;
        }
    }

    // Build result indexed by head_idx
    let mut bits_per_head = vec![base_bits; n_heads];
    for s in sensitivities.iter() {
        bits_per_head[s.head_idx] = s.bits;
    }
    bits_per_head
}

/// Summary of per-head allocation.
#[derive(Clone, Debug)]
pub struct AllocationSummary {
    /// Number of heads at each bit-width.
    pub heads_at_bits: Vec<(u32, usize)>,
    /// Average bits per head.
    pub avg_bits: f64,
    /// Most sensitive head index.
    pub most_sensitive_head: usize,
    /// Least sensitive head index.
    pub least_sensitive_head: usize,
}

/// Summarize per-head bit allocation.
pub fn summarize_allocation(
    sensitivities: &[HeadSensitivity],
    bits_per_head: &[u32],
) -> AllocationSummary {
    let n = bits_per_head.len();
    let avg = bits_per_head.iter().map(|&b| b as f64).sum::<f64>() / n as f64;

    let mut bit_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &b in bits_per_head {
        *bit_counts.entry(b).or_insert(0) += 1;
    }
    let mut heads_at_bits: Vec<(u32, usize)> = bit_counts.into_iter().collect();
    heads_at_bits.sort();

    let most = sensitivities.first().map(|s| s.head_idx).unwrap_or(0);
    let least = sensitivities.last().map(|s| s.head_idx).unwrap_or(0);

    AllocationSummary {
        heads_at_bits,
        avg_bits: avg,
        most_sensitive_head: most,
        least_sensitive_head: least,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_head_allocation() {
        let dim = 16;
        let n_heads = 8;

        // Create synthetic residuals with varying complexity per head
        let residuals_per_head: Vec<Vec<Vec<f64>>> = (0..n_heads)
            .map(|h| {
                (0..20)
                    .map(|t| {
                        let scale = (h + 1) as f64 * 0.1; // head 7 is most complex
                        (0..dim)
                            .map(|i| ((t * 3 + i * 7 + h * 11) as f64 * scale * 0.05).sin())
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let mut sensitivities = score_head_sensitivity(&residuals_per_head, dim);
        println!("Head sensitivities (sorted by sensitivity):");
        for s in &sensitivities {
            println!("  Head {}: mean_assoc={:.6}", s.head_idx, s.mean_associator);
        }

        // Allocate: base=3, promote 2 heads to 4-bit
        let bits = allocate_per_head_bits(&mut sensitivities, 3, 4, 2);
        assert_eq!(bits.len(), n_heads);

        let summary = summarize_allocation(&sensitivities, &bits);
        println!("\nAllocation summary:");
        for (b, count) in &summary.heads_at_bits {
            println!("  {}-bit: {} heads", b, count);
        }
        println!("  Avg bits: {:.2}", summary.avg_bits);
        println!("  Most sensitive: head {}", summary.most_sensitive_head);
        println!("  Least sensitive: head {}", summary.least_sensitive_head);

        // Should have 2 promoted and 6 base
        let n_promoted = bits.iter().filter(|&&b| b == 4).count();
        assert_eq!(n_promoted, 2, "Should promote 2 heads");
        assert!((summary.avg_bits - 3.25).abs() < 0.01);
    }

    #[test]
    fn test_sensitivity_ordering() {
        let dim = 16;
        let n_heads = 4;

        // Head 3 has large residuals (high sensitivity)
        // Head 0 has small residuals (low sensitivity)
        let residuals_per_head: Vec<Vec<Vec<f64>>> = (0..n_heads)
            .map(|h| {
                let scale = if h == 3 { 10.0 } else { 0.1 };
                (0..10)
                    .map(|t| {
                        (0..dim)
                            .map(|i| ((t + i * h) as f64 * scale * 0.1).sin())
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let sensitivities = score_head_sensitivity(&residuals_per_head, dim);

        // Head 3 should be most sensitive (first in sorted list)
        assert_eq!(
            sensitivities[0].head_idx, 3,
            "Head 3 should be most sensitive, got head {}",
            sensitivities[0].head_idx
        );
    }
}
