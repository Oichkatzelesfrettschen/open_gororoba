//! Hierarchical CD tower quantization: per-level bit allocation.
//!
//! The CD tower provides a natural multi-scale decomposition:
//!   Level 0: quaternion (4D) subalgebra -- fully associative, most robust
//!   Level 1: octonion (8D) extension -- alternative, robust
//!   Level 2: sedenion (16D) extension -- 84+ ZD pairs, moderately vulnerable
//!   Level 3: pathion (32D) extension -- proliferating ZDs, most vulnerable
//!   Level 4: chingon (64D) extension
//!   Level 5: routon (128D) extension
//!
//! # MERA-CD duality (Novel Connection N1)
//!
//! The CD tower doubling (4D -> 128D) is the algebraic dual of MERA
//! coarse-graining (128D -> 4D).  Each level adds a "scale" of phase
//! structure.  Quantization is truncation in the tower sense: coarser
//! levels are robust (need fewer bits), finer levels are fragile (need
//! more bits).
//!
//! # Algorithm
//!
//! 1. Decompose 128D vector into tower-level components via projection
//!    onto nested subalgebra orthogonal complements
//! 2. Quantize each level at its assigned bit-width
//! 3. Reconstruct by summing quantized components
//!
//! The decomposition is lossless: sum of components = original vector.

use super::pipeline::TurboQuantMSE;

/// Tower-level specification for hierarchical quantization.
#[derive(Clone, Debug)]
pub struct TowerLevel {
    /// CD algebra dimension at this level.
    pub dim: usize,
    /// Bit-width for this level.
    pub bits: u32,
    /// Component start index in the full vector.
    pub start: usize,
    /// Component end index (exclusive).
    pub end: usize,
}

/// Hierarchical tower decomposition of a 128D vector.
///
/// The CD doubling construction builds 128D as:
///   (((((R^2)^2)^2)^2)^2)^2)^2 = R^128
///
/// At each doubling, the new components are the "detail" coefficients.
/// The decomposition splits the vector into components at each scale:
///   - coords [0..4]:   quaternion core (most robust)
///   - coords [4..8]:   octonion extension
///   - coords [8..16]:  sedenion extension
///   - coords [16..32]: pathion extension
///   - coords [32..64]: chingon extension
///   - coords [64..128]: routon extension (most fragile)
///
/// This is a coordinate-aligned decomposition matching the CD tower
/// structure, NOT a random projection.
pub fn tower_levels(d: usize) -> Vec<TowerLevel> {
    assert!(d.is_power_of_two() && d >= 4);
    let n_levels = d.trailing_zeros() - 1; // 128 -> 6 levels

    let mut levels = Vec::new();
    let mut level_dim = 4; // quaternion base

    // Level 0: quaternion core (first 4 components)
    levels.push(TowerLevel {
        dim: 4,
        bits: 0, // to be filled by allocator
        start: 0,
        end: 4,
    });
    let mut start = 4;

    // Subsequent levels: extensions
    for _ in 1..n_levels as usize {
        let end = (start + level_dim).min(d);
        levels.push(TowerLevel {
            dim: end - start,
            bits: 0,
            start,
            end,
        });
        start = end;
        level_dim *= 2;
    }

    // If there are remaining coordinates, add them to the last level
    if start < d {
        levels.push(TowerLevel {
            dim: d - start,
            bits: 0,
            start,
            end: d,
        });
    }

    levels
}

/// Assign bits per level to minimize total MSE at a fixed budget.
///
/// Uses a simple greedy approach: start all levels at `min_bits`,
/// then promote levels with highest per-coordinate distortion until
/// the budget is exhausted.
///
/// `budget_bits`: total bits per vector (e.g., 3 * 128 = 384)
/// `min_bits`: minimum bits per level (1 or 2)
/// `max_bits`: maximum bits per level (4 or 5)
pub fn allocate_bits_to_levels(
    levels: &mut [TowerLevel],
    _d: usize,
    budget_bits: usize,
    min_bits: u32,
    max_bits: u32,
) {
    // Start at minimum
    for level in levels.iter_mut() {
        level.bits = min_bits;
    }

    // Compute current total bits
    let current_total = |levels: &[TowerLevel]| -> usize {
        levels.iter().map(|l| l.dim * l.bits as usize).sum()
    };

    // Greedy promotion: add 1 bit to the level with highest improvement potential
    // Higher tower levels (more fragile) get priority
    while current_total(levels) < budget_bits {
        // Find the level that benefits most from an extra bit
        // Heuristic: higher levels (larger start index) are more fragile
        let mut best_level = None;
        let mut best_score = f64::MIN;

        for (i, level) in levels.iter().enumerate() {
            if level.bits >= max_bits {
                continue; // already at max
            }
            if current_total(levels) + level.dim > budget_bits {
                continue; // would exceed budget
            }
            // Score: fragility * dim (higher tower levels are more fragile)
            let fragility = (level.start as f64 + 1.0).ln();
            let score = fragility * level.dim as f64;
            if score > best_score {
                best_score = score;
                best_level = Some(i);
            }
        }

        match best_level {
            Some(idx) => levels[idx].bits += 1,
            None => break, // no more promotions possible
        }
    }
}

/// Quantize a vector using hierarchical tower decomposition.
///
/// Each tower level is quantized independently at its assigned bit-width.
/// Returns the per-level quantized indices and the reconstructed vector.
pub fn hierarchical_quantize(
    v: &[f64],
    levels: &[TowerLevel],
    seed: u64,
    use_wht: bool,
) -> (Vec<Vec<u8>>, Vec<f64>) {
    let d = v.len();
    let mut all_indices = Vec::with_capacity(levels.len());
    let mut reconstructed = vec![0.0f64; d];

    for (li, level) in levels.iter().enumerate() {
        if level.bits == 0 {
            // Zero bits: drop this component (maximum compression)
            all_indices.push(vec![0u8; level.dim]);
            continue;
        }

        // Extract this level's component
        let component = &v[level.start..level.end];

        // Quantize using a per-level TurboQuantMSE
        let tq = TurboQuantMSE::new(level.dim, level.bits, seed + li as u64, use_wht);
        let mut buf = vec![0.0f64; 2 * level.dim];
        let compressed = tq.quantize(component, &mut buf);

        // Dequantize and store
        let mut recon = vec![0.0f64; level.dim];
        tq.dequantize(&compressed, &mut buf, &mut recon);
        reconstructed[level.start..level.end].copy_from_slice(&recon);

        all_indices.push(compressed.indices);
    }

    (all_indices, reconstructed)
}

/// Compare hierarchical vs uniform quantization at the same total budget.
///
/// Returns (hierarchical_mse, uniform_mse, per_level_mse).
pub fn compare_hierarchical_vs_uniform(
    vectors: &[Vec<f64>],
    uniform_bits: u32,
    min_level_bits: u32,
    max_level_bits: u32,
    seed: u64,
    use_wht: bool,
) -> (f64, f64, Vec<f64>) {
    let d = vectors[0].len();
    let n = vectors.len();
    let budget = d * uniform_bits as usize;

    // Hierarchical allocation
    let mut levels = tower_levels(d);
    allocate_bits_to_levels(&mut levels, d, budget, min_level_bits, max_level_bits);

    let mut hier_mse_sum = 0.0f64;
    let mut per_level_mse = vec![0.0f64; levels.len()];

    for v in vectors {
        let (_, recon) = hierarchical_quantize(v, &levels, seed, use_wht);
        let mse: f64 = v.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
        hier_mse_sum += mse;

        // Per-level MSE
        for (li, level) in levels.iter().enumerate() {
            let level_mse: f64 = v[level.start..level.end].iter()
                .zip(recon[level.start..level.end].iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / level.dim as f64;
            per_level_mse[li] += level_mse;
        }
    }

    // Uniform quantization
    let tq_uniform = TurboQuantMSE::new(d, uniform_bits, seed, use_wht);
    let mut uniform_mse_sum = 0.0f64;
    let mut buf = vec![0.0f64; 2 * d];

    for v in vectors {
        let compressed = tq_uniform.quantize(v, &mut buf);
        let mut recon = vec![0.0f64; d];
        tq_uniform.dequantize(&compressed, &mut buf, &mut recon);
        let mse: f64 = v.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
        uniform_mse_sum += mse;
    }

    let hier_mse = hier_mse_sum / n as f64;
    let uniform_mse = uniform_mse_sum / n as f64;
    let per_level_avg: Vec<f64> = per_level_mse.iter().map(|m| m / n as f64).collect();

    (hier_mse, uniform_mse, per_level_avg)
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
    fn test_tower_levels_128d() {
        let levels = tower_levels(128);
        // 128 = 2^7, so 6 levels (2->4->8->16->32->64->128)
        // Level 0: [0,4), Level 1: [4,8), Level 2: [8,16), Level 3: [16,32),
        // Level 4: [32,64), Level 5: [64,128)
        assert_eq!(levels.len(), 6);
        assert_eq!(levels[0].start, 0);
        assert_eq!(levels[0].end, 4);
        assert_eq!(levels[0].dim, 4);
        assert_eq!(levels[5].start, 64);
        assert_eq!(levels[5].end, 128);
        assert_eq!(levels[5].dim, 64);

        // Total dimension should equal 128
        let total: usize = levels.iter().map(|l| l.dim).sum();
        assert_eq!(total, 128);
    }

    #[test]
    fn test_tower_levels_64d() {
        let levels = tower_levels(64);
        let total: usize = levels.iter().map(|l| l.dim).sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn test_allocate_bits() {
        let mut levels = tower_levels(128);
        let budget = 128 * 3; // 3 bits per coord average
        allocate_bits_to_levels(&mut levels, 128, budget, 2, 5);

        // All levels should have >= 2 bits
        assert!(levels.iter().all(|l| l.bits >= 2));
        // Total should not exceed budget
        let total_bits: usize = levels.iter().map(|l| l.dim * l.bits as usize).sum();
        assert!(total_bits <= budget, "Total {} exceeds budget {}", total_bits, budget);

        // Higher levels should generally have more bits (fragility heuristic)
        // (not strictly guaranteed but likely)
        println!("Bit allocation:");
        for (i, level) in levels.iter().enumerate() {
            println!("  Level {} ({}D, [{},{})): {} bits",
                i, level.dim, level.start, level.end, level.bits);
        }
    }

    #[test]
    fn test_hierarchical_quantize_roundtrip() {
        let d = 64;
        let mut levels = tower_levels(d);
        let budget = d * 3;
        allocate_bits_to_levels(&mut levels, d, budget, 2, 4);

        let v: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let (indices, recon) = hierarchical_quantize(&v, &levels, 42, true);

        assert_eq!(indices.len(), levels.len());
        assert_eq!(recon.len(), d);

        let mse: f64 = v.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
        assert!(mse < 0.5, "Hierarchical MSE too high: {}", mse);
    }

    #[test]
    fn test_hierarchical_vs_uniform() {
        let d = 64;
        let n = 50;
        let vectors = random_vectors(n, d, 42);

        let (hier_mse, uniform_mse, per_level_mse) =
            compare_hierarchical_vs_uniform(&vectors, 3, 2, 5, 42, true);

        println!("Hierarchical MSE: {:.6}", hier_mse);
        println!("Uniform MSE:      {:.6}", uniform_mse);
        println!("Per-level MSE: {:?}", per_level_mse.iter()
            .map(|m| format!("{:.4}", m)).collect::<Vec<_>>());

        // Both should be finite and positive
        assert!(hier_mse > 0.0 && hier_mse.is_finite());
        assert!(uniform_mse > 0.0 && uniform_mse.is_finite());
    }
}
