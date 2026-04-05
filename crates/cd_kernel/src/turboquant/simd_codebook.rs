//! SIMD-accelerated Lloyd-Max codebook lookup.
//!
//! For b-bit quantization with 2^b centroids, the quantization operation is:
//!   index = argmin_j |x - centroid[j]|
//!
//! The boundary-search variant (sort centroids, count comparisons) is already
//! faster than argmin.  This module adds true SIMD parallelism via the `wide`
//! crate's f32x8 type: at 3-bit (8 centroids), one SIMD register holds the
//! entire codebook boundary set.
//!
//! # Approach: boundary-count via SIMD
//!
//! For sorted boundaries b[0] < b[1] < ... < b[2^bits - 2]:
//!   index = count(b[j] < x for all j)
//!
//! With f32x8, we broadcast x into all 8 lanes, compare against boundaries
//! packed in a single register, and popcount the comparison mask.
//!
//! # Measured results (from turboquant-simd-bench at d=128, 100K vectors)
//!
//! | Method          | Throughput (Mval/s) | Speedup vs scalar |
//! |-----------------|---------------------|-------------------|
//! | Scalar argmin   | baseline            | 1.0x              |
//! | Boundary search | ~3.5x               | 3.5x              |
//! | SIMD-style      | ~3.1x               | 3.1x              |
//! | Batched (8-wide)| ~3.0x               | 3.0x              |
//!
//! Hardware pattern: steinmarder INT8 SoA LBM achieves 5643 MLUPS via the same
//! broadcast-compare-count approach on GPU (CUDA kernels_int8_soa.cu).

use wide::{CmpGt, f32x8};

/// Precomputed SIMD-ready boundary set for a given codebook.
///
/// For b-bit quantization with n_boundaries = 2^b - 1 boundaries:
/// - If n_boundaries <= 8: pack into a single f32x8 (pad with +INF)
/// - If n_boundaries <= 16: use two f32x8 registers
/// - Otherwise: fall back to scalar boundary search
#[derive(Clone, Debug)]
pub struct SimdBoundaries {
    /// Packed boundary registers.  Unused lanes filled with f32::MAX.
    words: Vec<f32x8>,
    /// Number of actual boundaries (2^bits - 1).
    n_boundaries: usize,
    /// Quantization bits.
    bits: u32,
}

impl SimdBoundaries {
    /// Create SIMD-packed boundaries from a sorted boundary slice.
    pub fn from_boundaries(boundaries: &[f32], bits: u32) -> Self {
        let n = boundaries.len();
        let n_words = n.div_ceil(8);
        let mut words = Vec::with_capacity(n_words);

        for w in 0..n_words {
            let mut vals = [f32::MAX; 8]; // pad with +INF (always > x, so no false count)
            for (i, val) in vals.iter_mut().enumerate() {
                let idx = w * 8 + i;
                if idx < n {
                    *val = boundaries[idx];
                }
            }
            words.push(f32x8::new(vals));
        }

        SimdBoundaries {
            words,
            n_boundaries: n,
            bits,
        }
    }

    /// Quantize a single value using SIMD boundary comparison.
    ///
    /// Returns the centroid index in [0, 2^bits).
    #[inline]
    pub fn quantize_one(&self, x: f32) -> u8 {
        let x_broadcast = f32x8::splat(x);
        let mut count = 0u32;
        for word in &self.words {
            // Compare: x > boundary[j] for each lane
            // cmp_gt returns f32x8 with all-ones for true lanes
            // move_mask extracts the sign bit of each lane into an i32 bitmask
            let mask = x_broadcast.simd_gt(*word);
            // Each set bit in move_mask = one boundary that x exceeds
            count += mask.to_bitmask().count_ones();
        }
        count as u8
    }

    /// Quantize a slice of values, writing indices into `out`.
    ///
    /// This is the hot loop for TurboQuant Stage 1.
    pub fn quantize_batch(&self, values: &[f32], out: &mut [u8]) {
        debug_assert_eq!(values.len(), out.len());
        for (i, &v) in values.iter().enumerate() {
            out[i] = self.quantize_one(v);
        }
    }

    /// Quantize a slice with 8-value batching for better ILP.
    ///
    /// Processes 8 values simultaneously against each boundary word,
    /// allowing the CPU to pipeline comparison operations.
    pub fn quantize_batch_8wide(&self, values: &[f32], out: &mut [u8]) {
        debug_assert_eq!(values.len(), out.len());
        let n = values.len();
        let chunks = n / 8;

        // Process in 8-element chunks
        for chunk in 0..chunks {
            let base = chunk * 8;
            // Initialize counts to zero
            let mut counts = [0u8; 8];

            for word in &self.words {
                // Compare each of the 8 values against this boundary word
                for j in 0..8 {
                    let x_broadcast = f32x8::splat(values[base + j]);
                    let mask = x_broadcast.simd_gt(*word);
                    counts[j] += mask.to_bitmask().count_ones() as u8;
                }
            }

            out[base..base + 8].copy_from_slice(&counts);
        }

        // Handle remainder
        for i in (chunks * 8)..n {
            out[i] = self.quantize_one(values[i]);
        }
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn n_boundaries(&self) -> usize {
        self.n_boundaries
    }
}

/// Scalar boundary-search quantizer (baseline, no SIMD).
///
/// For comparison benchmarks.  Counts how many boundaries are less than x.
pub fn quantize_scalar_boundary(value: f32, boundaries: &[f32]) -> u8 {
    let mut idx = 0u8;
    for &b in boundaries {
        if value > b {
            idx += 1;
        } else {
            break;
        }
    }
    idx
}

/// Dequantize: map index to centroid value.
#[inline]
pub fn dequantize_one(index: u8, centroids: &[f32]) -> f32 {
    centroids[index as usize]
}

/// Batch dequantize: map indices to centroid values.
pub fn dequantize_batch(indices: &[u8], centroids: &[f32], out: &mut [f32]) {
    debug_assert_eq!(indices.len(), out.len());
    for (i, &idx) in indices.iter().enumerate() {
        out[i] = centroids[idx as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lloyd_max;

    #[test]
    fn test_simd_matches_scalar_3bit() {
        let cb = lloyd_max::get_codebook(128, 3);
        let boundaries = &cb.boundaries;
        let simd = SimdBoundaries::from_boundaries(boundaries, 3);

        // Test 1000 random values
        let sigma = 1.0 / (128.0f32).sqrt();
        for i in 0..1000 {
            let x = ((i as f32 * 0.618) % 1.0 - 0.5) * 7.0 * sigma;
            let scalar_idx = quantize_scalar_boundary(x, boundaries);
            let simd_idx = simd.quantize_one(x);
            assert_eq!(
                scalar_idx, simd_idx,
                "Mismatch at x={}: scalar={}, simd={}",
                x, scalar_idx, simd_idx
            );
        }
    }

    #[test]
    fn test_simd_matches_scalar_2bit() {
        let cb = lloyd_max::get_codebook(128, 2);
        let boundaries = &cb.boundaries;
        let simd = SimdBoundaries::from_boundaries(boundaries, 2);

        for i in 0..500 {
            let x = ((i as f32 * 0.31415) % 1.0 - 0.5) * 0.5;
            let scalar_idx = quantize_scalar_boundary(x, boundaries);
            let simd_idx = simd.quantize_one(x);
            assert_eq!(scalar_idx, simd_idx);
        }
    }

    #[test]
    fn test_simd_matches_scalar_4bit() {
        let cb = lloyd_max::get_codebook(128, 4);
        let boundaries = &cb.boundaries;
        let simd = SimdBoundaries::from_boundaries(boundaries, 4);

        // 4-bit has 15 boundaries -> 2 SIMD words
        assert_eq!(simd.words.len(), 2);

        for i in 0..500 {
            let x = ((i as f32 * 0.7071) % 1.0 - 0.5) * 0.5;
            let scalar_idx = quantize_scalar_boundary(x, boundaries);
            let simd_idx = simd.quantize_one(x);
            assert_eq!(scalar_idx, simd_idx);
        }
    }

    #[test]
    fn test_batch_quantize() {
        let cb = lloyd_max::get_codebook(128, 3);
        let simd = SimdBoundaries::from_boundaries(&cb.boundaries, 3);

        let values: Vec<f32> = (0..256).map(|i| ((i as f32 * 0.1).sin()) * 0.3).collect();
        let mut out_single = vec![0u8; 256];
        let mut out_batch = vec![0u8; 256];

        for (i, &v) in values.iter().enumerate() {
            out_single[i] = simd.quantize_one(v);
        }
        simd.quantize_batch(&values, &mut out_batch);

        assert_eq!(out_single, out_batch);
    }

    #[test]
    fn test_batch_8wide_matches() {
        let cb = lloyd_max::get_codebook(128, 3);
        let simd = SimdBoundaries::from_boundaries(&cb.boundaries, 3);

        let values: Vec<f32> = (0..1024).map(|i| ((i as f32 * 0.07).cos()) * 0.2).collect();
        let mut out_batch = vec![0u8; 1024];
        let mut out_8wide = vec![0u8; 1024];

        simd.quantize_batch(&values, &mut out_batch);
        simd.quantize_batch_8wide(&values, &mut out_8wide);

        assert_eq!(out_batch, out_8wide);
    }

    #[test]
    fn test_dequantize_roundtrip() {
        let cb = lloyd_max::get_codebook(128, 3);
        let simd = SimdBoundaries::from_boundaries(&cb.boundaries, 3);

        let values: Vec<f32> = (0..64).map(|i| ((i as f32 * 0.2).sin()) * 0.15).collect();
        let mut indices = vec![0u8; 64];
        simd.quantize_batch(&values, &mut indices);

        let mut reconstructed = vec![0.0f32; 64];
        dequantize_batch(&indices, &cb.centroids, &mut reconstructed);

        // Each reconstructed value should be a valid centroid
        for &r in &reconstructed {
            assert!(
                cb.centroids.contains(&r),
                "Reconstructed value {} is not a valid centroid",
                r
            );
        }
    }

    #[test]
    fn test_extreme_values() {
        let cb = lloyd_max::get_codebook(128, 3);
        let simd = SimdBoundaries::from_boundaries(&cb.boundaries, 3);

        // Very large positive -> should be max index
        assert_eq!(simd.quantize_one(100.0), 7);
        // Very large negative -> should be index 0
        assert_eq!(simd.quantize_one(-100.0), 0);
        // Zero -> should be middle index (3 or 4 for symmetric codebook)
        let zero_idx = simd.quantize_one(0.0);
        assert!(
            zero_idx == 3 || zero_idx == 4,
            "Zero maps to index {}",
            zero_idx
        );
    }
}
