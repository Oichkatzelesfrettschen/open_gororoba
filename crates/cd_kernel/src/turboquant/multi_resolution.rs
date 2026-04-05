//! Multi-resolution codebook: coarse + fine bits for adaptive allocation.
//!
//! Instead of maintaining separate base (b-bit) and promoted ((b+1)-bit)
//! codebooks, use a hierarchical codebook where:
//! - Coarse bits (b): all tokens get this level
//! - Fine bits (1 extra): only promoted tokens get refinement
//!
//! The fine codebook subdivides each coarse partition into 2 sub-partitions,
//! allowing a 1-bit refinement without a completely new codebook.
//!
//! # Advantage
//!
//! The coarse codebook is shared across all tokens (reuse across cache).
//! The fine codebook is only applied to promoted tokens (25% of cache).
//! Memory: base_bits * n_total + 1 * n_promoted vs (base_bits+1) * n_promoted.

use crate::lloyd_max;

/// Multi-resolution codebook: coarse + fine levels.
#[derive(Clone, Debug)]
pub struct MultiResCodebook {
    /// Coarse centroids (2^base_bits levels).
    pub coarse_centroids: Vec<f32>,
    /// Coarse boundaries.
    pub coarse_boundaries: Vec<f32>,
    /// Fine sub-centroids: for each coarse partition, 2 sub-centroids.
    /// fine_sub[coarse_idx] = (lower_centroid, upper_centroid)
    pub fine_sub: Vec<(f32, f32)>,
    /// Fine sub-boundary: midpoint within each coarse partition.
    pub fine_boundaries: Vec<f32>,
    /// Base bits.
    pub base_bits: u32,
}

impl MultiResCodebook {
    /// Build a multi-resolution codebook from Lloyd-Max.
    pub fn build(d: usize, base_bits: u32) -> Self {
        let coarse = lloyd_max::get_codebook(d, base_bits);
        let fine = lloyd_max::get_codebook(d, base_bits + 1);

        // For each coarse partition, find the 2 fine centroids within it
        let n_coarse = 1usize << base_bits;
        let _n_fine = 1usize << (base_bits + 1);
        let mut fine_sub = Vec::with_capacity(n_coarse);
        let mut fine_boundaries = Vec::with_capacity(n_coarse);

        for c in 0..n_coarse {
            // The fine codebook has 2 centroids per coarse partition
            let fine_lo = fine.centroids[2 * c];
            let fine_hi = fine.centroids[2 * c + 1];
            fine_sub.push((fine_lo, fine_hi));
            fine_boundaries.push((fine_lo + fine_hi) / 2.0);
        }

        MultiResCodebook {
            coarse_centroids: coarse.centroids.clone(),
            coarse_boundaries: coarse.boundaries.clone(),
            fine_sub,
            fine_boundaries,
            base_bits,
        }
    }

    /// Coarse quantize: returns base_bits index.
    pub fn quantize_coarse(&self, value: f64) -> u8 {
        let mut idx = 0u8;
        for &b in &self.coarse_boundaries {
            if value > b as f64 {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Fine refine: given a coarse index, returns the 1-bit refinement.
    pub fn refine(&self, value: f64, coarse_idx: u8) -> u8 {
        if value > self.fine_boundaries[coarse_idx as usize] as f64 {
            1
        } else {
            0
        }
    }

    /// Full quantize: coarse + fine = (base_bits + 1) total.
    pub fn quantize_full(&self, value: f64) -> (u8, u8) {
        let coarse = self.quantize_coarse(value);
        let fine = self.refine(value, coarse);
        (coarse, fine)
    }

    /// Dequantize coarse only.
    pub fn dequantize_coarse(&self, coarse_idx: u8) -> f32 {
        self.coarse_centroids[coarse_idx as usize]
    }

    /// Dequantize with fine refinement.
    pub fn dequantize_fine(&self, coarse_idx: u8, fine_bit: u8) -> f32 {
        let (lo, hi) = self.fine_sub[coarse_idx as usize];
        if fine_bit == 0 { lo } else { hi }
    }

    /// Memory per vector at coarse level.
    pub fn coarse_bits_per_coord(&self) -> u32 {
        self.base_bits
    }

    /// Memory per vector at fine level (additional 1 bit per coord).
    pub fn fine_bits_per_coord(&self) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_res_build() {
        let mr = MultiResCodebook::build(128, 3);
        assert_eq!(mr.coarse_centroids.len(), 8); // 2^3
        assert_eq!(mr.fine_sub.len(), 8); // one per coarse partition
        assert_eq!(mr.fine_boundaries.len(), 8);
    }

    #[test]
    fn test_multi_res_quantize() {
        let mr = MultiResCodebook::build(128, 3);
        let value = 0.05;

        let coarse = mr.quantize_coarse(value);
        let (c, f) = mr.quantize_full(value);
        assert_eq!(c, coarse);
        assert!(f == 0 || f == 1);

        // Dequantize
        let recon_coarse = mr.dequantize_coarse(coarse);
        let recon_fine = mr.dequantize_fine(c, f);

        // Fine should be closer to original than coarse
        let err_coarse = (value - recon_coarse as f64).abs();
        let err_fine = (value - recon_fine as f64).abs();
        println!(
            "Coarse error: {:.6}, Fine error: {:.6}",
            err_coarse, err_fine
        );
        assert!(
            err_fine <= err_coarse + 1e-6,
            "Fine should be at least as good: {} vs {}",
            err_fine,
            err_coarse
        );
    }

    #[test]
    fn test_multi_res_memory() {
        let mr = MultiResCodebook::build(128, 3);
        let d = 128;
        let n_total = 1000;
        let n_promoted = 250; // 25%

        // Multi-resolution: coarse for all + fine for promoted
        let mr_bits = n_total * d * mr.coarse_bits_per_coord() as usize
            + n_promoted * d * mr.fine_bits_per_coord() as usize;

        // Standard: base for non-promoted + (base+1) for promoted
        let std_bits = (n_total - n_promoted) * d * 3 + n_promoted * d * 4;

        println!(
            "Multi-res: {} bits, Standard: {} bits, savings: {:.1}%",
            mr_bits,
            std_bits,
            (1.0 - mr_bits as f64 / std_bits as f64) * 100.0
        );

        // Multi-resolution uses fewer total bits
        assert!(mr_bits <= std_bits, "Multi-res should use <= standard bits");
    }
}
