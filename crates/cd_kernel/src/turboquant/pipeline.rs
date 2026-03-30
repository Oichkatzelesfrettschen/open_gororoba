//! TurboQuant quantization pipeline: TurboQuantMSE and TurboQuantProd.
//!
//! TurboQuantMSE (Stage 1 only): rotate -> scalar quantize -> dequantize -> unrotate
//! TurboQuantProd (Stage 1 + 2): TurboQuantMSE + QJL sign sketch on residual
//!
//! Ported from `turboquant.py:TurboQuantMSE` and `turboquant.py:TurboQuantProd`.

use crate::lloyd_max::{self, LloydMaxCodebook};
use super::qjl;
use super::rotation::Rotation;

/// Compressed representation from TurboQuantMSE (Stage 1 only).
#[derive(Clone, Debug)]
pub struct MseCompressed {
    /// Quantization indices, one per coordinate. Range [0, 2^bits).
    pub indices: Vec<u8>,
    /// Original vector norm (for denormalization).
    pub vec_norm: f64,
    /// Dense-and-sparse outlier retention: (coordinate_index, original_f32_value)
    /// for the top-k coordinates by quantization error magnitude.
    /// Empty when outlier_keep_frac is 0 (default).
    pub outliers: Vec<(u16, f32)>,
}

/// Compressed representation from TurboQuantProd (Stage 1 + 2).
#[derive(Clone, Debug)]
pub struct ProdCompressed {
    /// MSE stage indices.
    pub mse_indices: Vec<u8>,
    /// QJL sign sketch of the residual. +1 or -1 per projected dimension.
    pub qjl_signs: Vec<i8>,
    /// L2 norm of the quantization residual.
    pub residual_norm: f64,
    /// Original vector norm (for denormalization).
    pub vec_norm: f64,
}

/// Stage 1: MSE-optimal scalar quantization after random rotation.
///
/// Pipeline: normalize -> rotate -> per-coordinate Lloyd-Max -> dequantize -> unrotate -> denormalize
pub struct TurboQuantMSE {
    rotation: Rotation,
    codebook: LloydMaxCodebook,
    d: usize,
    bits: u32,
    /// Fraction of coordinates to keep as outliers (0.0 = none, 0.01 = 1%).
    /// Each outlier costs 16 bits (f16 value) + 16 bits (index) = 32 bits.
    outlier_keep_frac: f64,
}

impl TurboQuantMSE {
    pub fn new(d: usize, bits: u32, seed: u64, use_wht: bool) -> Self {
        let rotation = if use_wht {
            Rotation::new_fast_jl(d, seed)
        } else {
            Rotation::new_haar(d, seed)
        };
        let codebook = lloyd_max::get_codebook(d, bits);
        TurboQuantMSE { rotation, codebook, d, bits, outlier_keep_frac: 0.0 }
    }

    /// Create from a TurboQuantConfig, using the configured rotation method.
    ///
    /// This is the recommended constructor: it selects E8 for d=128,
    /// FastJL for d>=64, and Haar for smaller dimensions.
    pub fn from_config(config: &super::config::TurboQuantConfig, bits: u32, seed: u64) -> Self {
        use super::config::RotationMethod;
        let d = config.dim;
        let rotation = match &config.rotation {
            RotationMethod::E8Block => Rotation::new_e8(d, seed),
            RotationMethod::E8Wht => Rotation::new_e8_wht(d, seed),
            RotationMethod::F4Block => Rotation::new_f4(d, seed),
            RotationMethod::FastJL => Rotation::new_fast_jl(d, seed),
            RotationMethod::Haar => Rotation::new_haar(d, seed),
        };
        let codebook = lloyd_max::get_codebook(d, bits);
        TurboQuantMSE { rotation, codebook, d, bits, outlier_keep_frac: 0.0 }
    }

    /// Create with a specific distribution-aware codebook.
    ///
    /// For real LLM KV cache data (non-Gaussian, heavy-tailed), use
    /// `DistributionFamily::GeneralizedGaussian` with beta=0.9.
    pub fn with_codebook(d: usize, bits: u32, seed: u64, use_wht: bool, codebook: LloydMaxCodebook) -> Self {
        let rotation = if use_wht {
            Rotation::new_fast_jl(d, seed)
        } else {
            Rotation::new_haar(d, seed)
        };
        TurboQuantMSE { rotation, codebook, d, bits, outlier_keep_frac: 0.0 }
    }

    /// Enable dense-and-sparse outlier retention.
    ///
    /// `frac` is the fraction of coordinates to keep in fp32 (e.g. 0.01 = 1%).
    /// Each outlier costs 32 bits (16-bit index + 16-bit value).
    /// At d=128, 1% = 1.28 outliers -> effectively 2.25 bits/coord at 2-bit.
    ///
    /// Published KVQuant-1% uses 0.5-1% and achieves +5.8% PPL at 2-bit
    /// vs our baseline of +9.0%.
    pub fn with_outlier_retention(mut self, frac: f64) -> Self {
        self.outlier_keep_frac = frac;
        self
    }

    /// Quantize a single vector.  Returns compressed indices + vec_norm.
    ///
    /// `x` has length d.
    /// `buf` is scratch space (>= 2*d elements).
    pub fn quantize(&self, x: &[f64], buf: &mut [f64]) -> MseCompressed {
        debug_assert_eq!(x.len(), self.d);
        debug_assert!(buf.len() >= 3 * self.d); // 3*d: normalized + scratch + rotated
        let d = self.d;

        // Normalize to unit vector (store at buf[0..d])
        let norm: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let inv_norm = if norm > 1e-15 { 1.0 / norm } else { 1.0 };
        for i in 0..d {
            buf[i] = x[i] * inv_norm;
        }

        // Rotate: buf[0..d] -> buf[2d..3d] using buf[d..2d] as scratch
        // ZERO ALLOCATION -- all in caller's buffer
        {
            let (first_two_d, rot_out) = buf.split_at_mut(2 * d);
            let (input, scratch) = first_two_d.split_at_mut(d);
            self.rotation.forward(input, scratch, &mut rot_out[..d]);
        }

        // Per-coordinate quantization via boundary search on buf[2d..3d]
        let boundaries = &self.codebook.boundaries;
        let centroids = &self.codebook.centroids;
        let rotated = &buf[2 * d..3 * d];
        let indices: Vec<u8> = rotated
            .iter()
            .map(|&v| {
                let mut idx = 0u8;
                for &b in boundaries.iter() {
                    if v > b as f64 {
                        idx += 1;
                    } else {
                        break;
                    }
                }
                idx
            })
            .collect();

        // Dense-and-sparse outlier retention: keep top-k by quantization error
        let outliers = if self.outlier_keep_frac > 0.0 {
            let n_keep = (d as f64 * self.outlier_keep_frac).ceil().max(1.0) as usize;
            // Compute per-coordinate quantization error
            let mut errors: Vec<(usize, f64)> = indices.iter().enumerate()
                .map(|(i, &idx)| {
                    let orig = rotated[i];
                    let quant = centroids[idx as usize] as f64;
                    (i, (orig - quant).abs())
                })
                .collect();
            // Partial sort: find top-k by error magnitude
            errors.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            errors.truncate(n_keep);
            errors.iter()
                .map(|&(i, _)| (i as u16, rotated[i] as f32))
                .collect()
        } else {
            Vec::new()
        };

        MseCompressed {
            indices,
            vec_norm: norm,
            outliers,
        }
    }

    /// Dequantize: indices -> centroid values -> unrotate -> denormalize.
    ///
    /// `compressed` is from `quantize()`.
    /// `out` has length d, receives the reconstructed vector.
    /// `buf` is scratch space (>= 2*d).
    pub fn dequantize(&self, compressed: &MseCompressed, buf: &mut [f64], out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.d);
        debug_assert!(buf.len() >= 2 * self.d);
        let (buf1, buf2) = buf.split_at_mut(self.d);

        // Map indices to centroid values (in rotated space)
        let centroids = &self.codebook.centroids;
        for i in 0..self.d {
            buf1[i] = centroids[compressed.indices[i] as usize] as f64;
        }

        // Overwrite outlier positions with their stored fp32 values
        for &(idx, val) in &compressed.outliers {
            buf1[idx as usize] = val as f64;
        }

        // Unrotate
        self.rotation.inverse(buf1, buf2, out);

        // Denormalize
        for v in out.iter_mut() {
            *v *= compressed.vec_norm;
        }
    }

    /// Reconstruct in rotated space (no unrotation). Used by TurboQuantProd
    /// to compute residuals in rotated space.
    pub fn dequantize_rotated(&self, compressed: &MseCompressed, out: &mut [f64]) {
        let centroids = &self.codebook.centroids;
        for i in 0..self.d {
            out[i] = centroids[compressed.indices[i] as usize] as f64;
        }
    }

    pub fn dim(&self) -> usize { self.d }
    pub fn bits(&self) -> u32 { self.bits }
    pub fn codebook(&self) -> &crate::lloyd_max::LloydMaxCodebook { &self.codebook }
    pub fn rotation(&self) -> &super::rotation::Rotation { &self.rotation }
}

/// Stage 1 + 2: MSE quantization plus QJL sign-sketch residual correction.
///
/// Keys use TurboQuantProd (need inner product estimation).
/// Values use TurboQuantMSE (need reconstruction only).
pub struct TurboQuantProd {
    mse: TurboQuantMSE,
    s_matrix: Vec<f64>,
    d: usize,
    m: usize,
}

impl TurboQuantProd {
    /// Create a TurboQuantProd quantizer.
    ///
    /// `bits` is the total bit budget per coordinate.
    /// MSE stage uses `max(bits-1, 1)` bits; QJL uses 1 bit.
    /// `qjl_dim` defaults to d if None.
    pub fn new(d: usize, bits: u32, seed: u64, use_wht: bool, qjl_dim: Option<usize>) -> Self {
        let mse_bits = bits.saturating_sub(1).max(1);
        let m = qjl_dim.unwrap_or(d);
        let mse = TurboQuantMSE::new(d, mse_bits, seed, use_wht);
        // QJL projection matrix uses seed + 1 (matching turboquant.py convention)
        let s_matrix = qjl::generate_projection_matrix(d, m, seed + 1);
        TurboQuantProd { mse, s_matrix, d, m }
    }

    /// Quantize a single vector with MSE + QJL.
    ///
    /// `buf` needs >= 3*d elements of scratch space.
    pub fn quantize(&self, x: &[f64], buf: &mut [f64]) -> ProdCompressed {
        debug_assert_eq!(x.len(), self.d);
        debug_assert!(buf.len() >= 3 * self.d);

        // Stage 1: MSE quantization
        let mse_compressed = self.mse.quantize(x, buf);

        // Reconstruct MSE result to compute residual
        let mut x_mse = vec![0.0; self.d];
        self.mse.dequantize(&mse_compressed, buf, &mut x_mse);

        // Compute residual: r = x - x_mse
        let residual: Vec<f64> = x.iter().zip(x_mse.iter()).map(|(a, b)| a - b).collect();

        // Stage 2: QJL sign quantization of residual
        let mut qjl_signs = vec![0i8; self.m];
        let mut residual_norm = 0.0;
        qjl::sign_quantize(&residual, &self.s_matrix, self.d, self.m, &mut qjl_signs, &mut residual_norm);

        ProdCompressed {
            mse_indices: mse_compressed.indices,
            qjl_signs,
            residual_norm,
            vec_norm: mse_compressed.vec_norm,
        }
    }

    /// Estimate inner product <query, original> using compressed representation.
    ///
    /// `buf` needs >= 2*d elements of scratch space.
    pub fn inner_product(&self, query: &[f64], compressed: &ProdCompressed, buf: &mut [f64]) -> f64 {
        debug_assert_eq!(query.len(), self.d);

        // Reconstruct x_mse for the dot product
        let mse_compressed = MseCompressed {
            indices: compressed.mse_indices.clone(),
            vec_norm: compressed.vec_norm,
            outliers: Vec::new(),
        };
        let mut x_mse = vec![0.0; self.d];
        self.mse.dequantize(&mse_compressed, buf, &mut x_mse);

        qjl::asymmetric_inner_product(
            query,
            &x_mse,
            &self.s_matrix,
            &compressed.qjl_signs,
            compressed.residual_norm,
            self.d,
            self.m,
        )
    }

    /// Memory usage in bits for a single compressed vector.
    pub fn bits_per_vector(&self) -> usize {
        let mse_bits = self.mse.bits() as usize * self.d;  // indices
        let qjl_bits = self.m;                               // 1 bit per sign
        let norm_bits = 16;                                   // fp16 for residual norm
        let vec_norm_bits = 16;                               // fp16 for vec norm
        mse_bits + qjl_bits + norm_bits + vec_norm_bits
    }

    /// Compression ratio vs fp16 storage.
    pub fn compression_ratio(&self) -> f64 {
        let fp16_bits = self.d * 16;
        fp16_bits as f64 / self.bits_per_vector() as f64
    }

    pub fn dim(&self) -> usize { self.d }
    pub fn mse_bits(&self) -> u32 { self.mse.bits() }
    pub fn qjl_dim(&self) -> usize { self.m }
    pub fn s_matrix(&self) -> &[f64] { &self.s_matrix }
    pub fn mse(&self) -> &TurboQuantMSE { &self.mse }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_roundtrip_quality() {
        let d = 128;
        let bits = 3;
        let tq = TurboQuantMSE::new(d, bits, 42, false);

        // Random unit vector
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let x_unit: Vec<f64> = x.iter().map(|v| v / norm).collect();

        let mut buf = vec![0.0; 3 * d];
        let compressed = tq.quantize(&x_unit, &mut buf);
        let mut reconstructed = vec![0.0; d];
        tq.dequantize(&compressed, &mut buf, &mut reconstructed);

        // MSE should be bounded
        let mse: f64 = x_unit
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;

        assert!(
            mse < 0.01,
            "MSE too high: {} (expected < 0.01 for 3-bit at d=128)",
            mse
        );
    }

    #[test]
    fn test_prod_inner_product_statistical() {
        // Statistical test: over many random pairs, mean error should be small.
        // Single-pair tests are unreliable due to QJL variance.
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        use rand_distr::{Distribution, StandardNormal};

        let d = 64;
        let bits = 3;
        let tq = TurboQuantProd::new(d, bits, 42, false, None);
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let normal = StandardNormal;

        let n_trials = 200;
        let mut total_error = 0.0;
        let mut total_abs_true = 0.0;

        for _trial in 0..n_trials {
            let x: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
            let q: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
            let true_ip: f64 = x.iter().zip(q.iter()).map(|(a, b)| a * b).sum();

            let mut buf = vec![0.0; 3 * d];
            let compressed = tq.quantize(&x, &mut buf);
            let est_ip = tq.inner_product(&q, &compressed, &mut buf);

            total_error += est_ip - true_ip;
            total_abs_true += true_ip.abs();
        }

        let mean_error = total_error / n_trials as f64;
        let mean_abs_true = total_abs_true / n_trials as f64;
        // Mean error should be small relative to typical IP magnitude
        assert!(
            mean_error.abs() < mean_abs_true * 0.3,
            "TurboQuantProd IP biased: mean_error={:.4}, mean_|true|={:.4}",
            mean_error, mean_abs_true
        );
    }

    #[test]
    fn test_compression_ratio() {
        let d = 128;
        let bits = 3;
        let tq = TurboQuantProd::new(d, bits, 42, false, None);
        let ratio = tq.compression_ratio();
        // At 3 bits total (2 MSE + 1 QJL), with d=128:
        // MSE: 128 * 2 = 256 bits
        // QJL: 128 bits
        // Norms: 32 bits
        // Total: 416 bits vs fp16: 2048 bits
        // Ratio ~= 4.9x
        assert!(ratio > 3.0 && ratio < 7.0, "Unexpected compression ratio: {}", ratio);
    }

    #[test]
    fn test_wht_vs_haar() {
        let d = 64;
        let bits = 3;

        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.15).sin()).collect();

        // Both methods should produce similar quality
        let tq_haar = TurboQuantMSE::new(d, bits, 42, false);
        let tq_wht = TurboQuantMSE::new(d, bits, 42, true);

        let mut buf = vec![0.0; 3 * d];

        let comp_haar = tq_haar.quantize(&x, &mut buf);
        let comp_wht = tq_wht.quantize(&x, &mut buf);

        let mut recon_haar = vec![0.0; d];
        let mut recon_wht = vec![0.0; d];
        tq_haar.dequantize(&comp_haar, &mut buf, &mut recon_haar);
        tq_wht.dequantize(&comp_wht, &mut buf, &mut recon_wht);

        let mse_haar: f64 = x.iter().zip(recon_haar.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
        let mse_wht: f64 = x.iter().zip(recon_wht.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;

        // Both should be reasonable (< 0.01 for 3-bit at d=64)
        assert!(mse_haar < 0.05, "Haar MSE too high: {}", mse_haar);
        assert!(mse_wht < 0.05, "WHT MSE too high: {}", mse_wht);
    }

    #[test]
    fn test_outlier_retention_improves_2bit() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        use rand_distr::{Distribution, StandardNormal};

        let d = 128;
        let bits = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let normal = StandardNormal;

        // Without outlier retention
        let tq_base = TurboQuantMSE::new(d, bits, 42, true);
        // With 1% outlier retention
        let tq_outlier = TurboQuantMSE::new(d, bits, 42, true).with_outlier_retention(0.01);

        let mut buf = vec![0.0f64; 3 * d];
        let mut base_mse_sum = 0.0;
        let mut outlier_mse_sum = 0.0;
        let n = 500;

        for _ in 0..n {
            let x: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();

            let comp_base = tq_base.quantize(&x, &mut buf);
            let comp_outlier = tq_outlier.quantize(&x, &mut buf);

            assert!(comp_base.outliers.is_empty());
            assert!(!comp_outlier.outliers.is_empty());

            let mut recon_base = vec![0.0; d];
            let mut recon_outlier = vec![0.0; d];
            tq_base.dequantize(&comp_base, &mut buf, &mut recon_base);
            tq_outlier.dequantize(&comp_outlier, &mut buf, &mut recon_outlier);

            let mse_base: f64 = x.iter().zip(recon_base.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
            let mse_outlier: f64 = x.iter().zip(recon_outlier.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;

            base_mse_sum += mse_base;
            outlier_mse_sum += mse_outlier;
        }

        let base_mse = base_mse_sum / n as f64;
        let outlier_mse = outlier_mse_sum / n as f64;
        let improvement = (1.0 - outlier_mse / base_mse) * 100.0;

        println!("2-bit d=128: base MSE={:.6}, outlier(1%) MSE={:.6}, improvement={:.1}%",
            base_mse, outlier_mse, improvement);
        assert!(outlier_mse < base_mse,
            "Outlier retention should improve MSE: base={:.6} vs outlier={:.6}",
            base_mse, outlier_mse);
        assert!(improvement > 5.0,
            "Expected >5% improvement from 1% outlier retention, got {:.1}%", improvement);
    }
}
