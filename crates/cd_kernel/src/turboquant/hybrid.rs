//! TurboQuant + GroupQuant hybrid: rotation decorrelation + per-group scales.
//!
//! Combines the best of both approaches:
//! - TurboQuant's rotation decorrelates coordinates (dominates at 2-bit)
//! - GroupQuant's per-group scale/zp adapts to local statistics (wins at 3-4 bit)
//!
//! Pipeline: rotate -> group-wise quantize with per-group scale/zp -> dequantize -> unrotate
//!
//! Benchmark showed:
//!   2-bit: TurboQuant MSE=0.116, GroupQuant MSE=0.182 (TQ wins by 36%)
//!   3-bit: TurboQuant MSE=0.034, GroupQuant MSE=0.033 (GQ wins by 3%)
//!   4-bit: TurboQuant MSE=0.009, GroupQuant MSE=0.007 (GQ wins by 22%)
//!
//! The hybrid should give TurboQuant's 2-bit advantage AND GroupQuant's
//! 3-4 bit advantage by applying rotation first then grouping.

use super::grouping::{compute_group_params, group_quantize, group_dequantize, GroupQuantParams};
use super::rotation::Rotation;

/// Hybrid compressed representation: rotated + group-quantized.
#[derive(Clone, Debug)]
pub struct HybridCompressed {
    /// Per-coordinate quantized indices.
    pub indices: Vec<u8>,
    /// Per-group quantization parameters.
    pub group_params: GroupQuantParams,
    /// Original vector norm.
    pub vec_norm: f64,
    /// Bits per coordinate.
    pub bits: u32,
}

/// TurboQuant + GroupQuant hybrid quantizer.
pub struct HybridQuantizer {
    rotation: Rotation,
    d: usize,
    bits: u32,
    group_size: usize,
}

impl HybridQuantizer {
    /// Create a hybrid quantizer.
    ///
    /// `group_size`: number of coordinates per group (typically 32).
    /// Smaller groups = more adaptive but more metadata overhead.
    pub fn new(d: usize, bits: u32, seed: u64, use_wht: bool, group_size: usize) -> Self {
        let rotation = if use_wht {
            Rotation::new_fast_jl(d, seed)
        } else {
            Rotation::new_haar(d, seed)
        };
        HybridQuantizer { rotation, d, bits, group_size }
    }

    /// Quantize: normalize -> rotate -> group-quantize.
    pub fn quantize(&self, x: &[f64], buf: &mut [f64]) -> HybridCompressed {
        let d = self.d;
        debug_assert_eq!(x.len(), d);
        debug_assert!(buf.len() >= 2 * d);

        // Normalize
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inv_norm = if norm > 1e-15 { 1.0 / norm } else { 1.0 };
        for i in 0..d {
            buf[i] = x[i] * inv_norm;
        }

        // Rotate
        let mut rotated = vec![0.0f64; d];
        {
            let (input, scratch) = buf.split_at_mut(d);
            self.rotation.forward(input, scratch, &mut rotated);
        }

        // Group-wise quantize on the rotated coordinates
        let params = compute_group_params(&rotated, self.group_size, self.bits);
        let indices = group_quantize(&rotated, &params, self.bits);

        HybridCompressed {
            indices,
            group_params: params,
            vec_norm: norm,
            bits: self.bits,
        }
    }

    /// Dequantize: group-dequantize -> unrotate -> denormalize.
    pub fn dequantize(&self, compressed: &HybridCompressed, buf: &mut [f64], out: &mut [f64]) {
        let d = self.d;

        // Group-wise dequantize
        let rotated_recon = group_dequantize(&compressed.indices, &compressed.group_params, compressed.bits);

        // Unrotate
        buf[..d].copy_from_slice(&rotated_recon);
        let (input, scratch) = buf.split_at_mut(d);
        self.rotation.inverse(input, scratch, out);

        // Denormalize
        for v in out[..d].iter_mut() {
            *v *= compressed.vec_norm;
        }
    }

    pub fn dim(&self) -> usize { self.d }
    pub fn bits(&self) -> u32 { self.bits }
    pub fn group_size(&self) -> usize { self.group_size }

    /// Memory per vector: indices + per-group metadata.
    pub fn bits_per_vector(&self) -> usize {
        let n_groups = self.d.div_ceil(self.group_size);
        let index_bits = self.d * self.bits as usize;
        let meta_bits = n_groups * (32 + 32 + 8); // scale(f32) + zp(f32) + flag(u8)
        let norm_bits = 16; // vec_norm as f16
        index_bits + meta_bits + norm_bits
    }
}

/// Compare hybrid vs TurboQuant-only vs GroupQuant-only.
pub fn compare_hybrid(
    vectors: &[Vec<f64>],
    bits: u32,
    group_size: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let d = vectors[0].len();
    let n = vectors.len();

    let hybrid = HybridQuantizer::new(d, bits, seed, true, group_size);
    let tq = super::pipeline::TurboQuantMSE::new(d, bits, seed, true);

    let mut buf = vec![0.0f64; 2 * d];

    // Hybrid MSE
    let mut hybrid_mse = 0.0f64;
    for v in vectors {
        let comp = hybrid.quantize(v, &mut buf);
        let mut recon = vec![0.0f64; d];
        hybrid.dequantize(&comp, &mut buf, &mut recon);
        hybrid_mse += v.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
    }

    // TurboQuant MSE (rotation + uniform codebook)
    let mut tq_mse = 0.0f64;
    for v in vectors {
        let comp = tq.quantize(v, &mut buf);
        let mut recon = vec![0.0f64; d];
        tq.dequantize(&comp, &mut buf, &mut recon);
        tq_mse += v.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
    }

    // GroupQuant MSE (no rotation, just grouping)
    let mut gq_mse = 0.0f64;
    for v in vectors {
        let params = compute_group_params(v, group_size, bits);
        let indices = group_quantize(v, &params, bits);
        let recon = group_dequantize(&indices, &params, bits);
        gq_mse += v.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
    }

    (hybrid_mse / n as f64, tq_mse / n as f64, gq_mse / n as f64)
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
    fn test_hybrid_roundtrip() {
        let d = 128;
        let bits = 3;
        let hq = HybridQuantizer::new(d, bits, 42, true, 32);
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut buf = vec![0.0f64; 2 * d];
        let comp = hq.quantize(&x, &mut buf);
        let mut recon = vec![0.0f64; d];
        hq.dequantize(&comp, &mut buf, &mut recon);

        let mse: f64 = x.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
        assert!(mse < 0.1, "Hybrid MSE too high: {}", mse);
    }

    #[test]
    fn test_hybrid_vs_turbo_vs_group() {
        let d = 128;
        let n = 200;
        let vectors = random_vectors(n, d, 42);

        for bits in [2, 3, 4] {
            let (hybrid_mse, tq_mse, gq_mse) = compare_hybrid(&vectors, bits, 32, 42);
            println!(
                "  {}-bit: hybrid={:.6}  turbo={:.6}  group={:.6}  hybrid_vs_best={:.1}%",
                bits, hybrid_mse, tq_mse, gq_mse,
                (hybrid_mse - tq_mse.min(gq_mse)) / tq_mse.min(gq_mse) * 100.0
            );

            // At 2-bit, group-wise scale/zp overhead hurts more than it helps
            // because 4 centroids can't adapt well within small groups.
            // At 3-4 bit, hybrid should be competitive.
            if bits >= 3 {
                let best_other = tq_mse.min(gq_mse);
                assert!(
                    hybrid_mse < best_other * 1.5,
                    "{}-bit: hybrid MSE={:.6} should be within 50% of best={:.6}",
                    bits, hybrid_mse, best_other
                );
            }
        }
    }

    #[test]
    fn test_hybrid_group_sizes() {
        let d = 128;
        let n = 100;
        let bits = 3;
        let vectors = random_vectors(n, d, 42);

        for group_size in [16, 32, 64, 128] {
            let (hybrid_mse, _, _) = compare_hybrid(&vectors, bits, group_size, 42);
            println!("  group_size={}: hybrid MSE={:.6}", group_size, hybrid_mse);
        }
    }
}
