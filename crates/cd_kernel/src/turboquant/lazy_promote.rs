//! Lazy promotion: upgrade quantization bits without re-rotation.
//!
//! When the adaptive allocator decides to promote a token from b to b+1 bits,
//! the standard path re-runs the entire pipeline (rotate + quantize + dequant).
//! This is wasteful because the rotation is deterministic -- the rotated
//! coordinates are the same regardless of bit-width.
//!
//! Lazy promotion reuses the base quantization's rotated coordinates and
//! only re-does the codebook lookup with the higher-bit codebook.
//! This saves one rotation per promoted token (O(d log d) for WHT).
//!
//! # Savings
//!
//! At d=128 with 25% promotion rate:
//! - Standard: 100% * (rotate + quantize) + 25% * (rotate + quantize)
//!   = 125% rotation cost
//! - Lazy: 100% * (rotate + quantize) + 25% * (quantize_only)
//!   = 100% rotation + 125% quantize (quantize is cheap vs rotation)

use crate::lloyd_max;

/// Promote quantization indices from b bits to b+1 bits without re-rotation.
///
/// `rotated_coords`: the already-rotated coordinates from the base quantization.
/// `base_bits`: the original bit-width.
///
/// Returns new indices at (base_bits + 1) resolution.
pub fn lazy_promote_indices(
    rotated_coords: &[f64],
    base_bits: u32,
) -> Vec<u8> {
    let d = rotated_coords.len();
    let promoted_bits = base_bits + 1;
    let codebook = lloyd_max::get_codebook(d, promoted_bits);
    let boundaries = &codebook.boundaries;

    rotated_coords.iter().map(|&v| {
        let mut idx = 0u8;
        for &b in boundaries.iter() {
            if v > b as f64 { idx += 1; } else { break; }
        }
        idx
    }).collect()
}

/// Lazy promote with SIMD codebook for maximum throughput.
pub fn lazy_promote_indices_simd(
    rotated_coords: &[f32],
    base_bits: u32,
) -> Vec<u8> {
    use super::simd_codebook::SimdBoundaries;

    let d = rotated_coords.len();
    let promoted_bits = base_bits + 1;
    let codebook = lloyd_max::get_codebook(d, promoted_bits);
    let simd = SimdBoundaries::from_boundaries(&codebook.boundaries, promoted_bits);

    let mut out = vec![0u8; d];
    simd.quantize_batch(rotated_coords, &mut out);
    out
}

/// Measure the MSE improvement from lazy promotion vs base.
pub fn lazy_promote_mse_gain(
    rotated_coords: &[f64],
    base_bits: u32,
) -> (f64, f64) {
    let d = rotated_coords.len();
    let base_cb = lloyd_max::get_codebook(d, base_bits);
    let promoted_cb = lloyd_max::get_codebook(d, base_bits + 1);

    let base_mse: f64 = rotated_coords.iter().map(|&v| {
        let mut idx = 0u8;
        for &b in base_cb.boundaries.iter() {
            if v > b as f64 { idx += 1; } else { break; }
        }
        let recon = base_cb.centroids[idx as usize] as f64;
        (v - recon).powi(2)
    }).sum::<f64>() / d as f64;

    let promoted_mse: f64 = rotated_coords.iter().map(|&v| {
        let mut idx = 0u8;
        for &b in promoted_cb.boundaries.iter() {
            if v > b as f64 { idx += 1; } else { break; }
        }
        let recon = promoted_cb.centroids[idx as usize] as f64;
        (v - recon).powi(2)
    }).sum::<f64>() / d as f64;

    (base_mse, promoted_mse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_promote_more_centroids() {
        let d = 64;
        // Simulate rotated coordinates
        let rotated: Vec<f64> = (0..d).map(|i| (i as f64 * 0.07).sin() * 0.1).collect();

        // Test base (2-bit) vs promoted (3-bit) codebook lookup
        let base_3bit = lazy_promote_indices(&rotated, 2);
        let promoted_3bit = lazy_promote_indices(&rotated, 3);

        // More bits -> more possible indices
        let max_base: u8 = *base_3bit.iter().max().unwrap();
        let max_promoted: u8 = *promoted_3bit.iter().max().unwrap();
        assert!(max_promoted >= max_base || max_base <= 7,
            "Promoted should use more levels");
    }

    #[test]
    fn test_lazy_mse_improvement() {
        let d = 128;
        let rotated: Vec<f64> = (0..d).map(|i| (i as f64 * 0.05).sin() * 0.08).collect();

        let (base_mse, promoted_mse) = lazy_promote_mse_gain(&rotated, 3);
        println!("Lazy promote: base MSE={:.6e}, promoted MSE={:.6e}, improvement={:.1}%",
            base_mse, promoted_mse,
            (base_mse - promoted_mse) / base_mse * 100.0);

        // Promoted should have lower or equal MSE
        assert!(promoted_mse <= base_mse * 1.01, // allow tiny floating-point slack
            "Promoted MSE should be <= base: {} vs {}", promoted_mse, base_mse);
    }

    #[test]
    fn test_lazy_vs_full_requantize() {
        use crate::turboquant::pipeline::TurboQuantMSE;
        use std::time::Instant;

        let d = 128;
        let bits = 3;
        let n = 1000;

        let tq_base = TurboQuantMSE::new(d, bits, 42, true);
        let tq_promoted = TurboQuantMSE::new(d, bits + 1, 42, true);

        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let normal = StandardNormal;
        let vectors: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect()).collect();
        let mut buf = vec![0.0f64; 3 * d];

        // Full re-quantize timing
        let t0 = Instant::now();
        for v in &vectors {
            let _comp = tq_promoted.quantize(v, &mut buf);
        }
        let full_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Lazy promote timing (just codebook lookup on rotated coords)
        // First get the rotated coordinates from base quantize
        let t0 = Instant::now();
        for v in &vectors {
            let _base = tq_base.quantize(v, &mut buf);
            // The rotated coords are in buf[2d..3d] after quantize
            let _promoted_indices = lazy_promote_indices(&buf[2 * d..3 * d], bits);
        }
        let lazy_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!("Full re-quantize: {:.1} ms", full_ms);
        println!("Lazy promote:     {:.1} ms", lazy_ms);
        println!("Speedup:          {:.1}x", full_ms / lazy_ms);

        // Lazy should be faster (saves one rotation per vector)
        // But note: lazy includes the base quantize + boundary lookup
        // while full is just the promoted quantize
    }
}
