//! High-level compressor API for KV cache quantization.
//!
//! Handles batched (B, H, S, D) tensor layouts with per-vector normalization,
//! matching `compressors.py:TurboQuantCompressorV2` (keys with QJL) and
//! `compressors.py:TurboQuantCompressorMSE` (values, MSE-only).

use super::config::TurboQuantConfig;
use super::pipeline::{MseCompressed, TurboQuantMSE, TurboQuantProd};
use super::sign_pack::BitPackedSigns;

/// Compressed key cache entry (one vector, Stage 1 + 2).
#[derive(Clone, Debug)]
pub struct CompressedKey {
    /// MSE reconstruction in original space (d elements, f32).
    pub k_mse: Vec<f32>,
    /// Bit-packed QJL signs (d bits packed into ceil(d/64) u64 words).
    pub signs: BitPackedSigns,
    /// L2 norm of the quantization residual.
    pub residual_norm: f32,
}

/// Compressed value cache entry (one vector, Stage 1 only).
#[derive(Clone, Debug)]
pub struct CompressedValue {
    /// Quantization indices (d elements, u8).
    pub indices: Vec<u8>,
    /// Original vector norm (for denormalization).
    pub vec_norm: f32,
}

/// Batch-compressed keys for one layer: (B*H*S) compressed vectors.
#[derive(Clone, Debug)]
pub struct CompressedKeyBatch {
    pub keys: Vec<CompressedKey>,
    pub shape: (usize, usize, usize, usize), // (B, H, S, D)
}

/// Batch-compressed values for one layer: (B*H*S) compressed vectors.
#[derive(Clone, Debug)]
pub struct CompressedValueBatch {
    pub values: Vec<CompressedValue>,
    pub shape: (usize, usize, usize, usize), // (B, H, S, D)
}

/// Key compressor (TurboQuantProd: MSE + QJL).
///
/// Port of `compressors.py:TurboQuantCompressorV2`.
pub struct KeyCompressor {
    quantizer: TurboQuantProd,
    d: usize,
    apply_qjl: bool,
}

impl KeyCompressor {
    /// Create from explicit parameters (legacy API).
    pub fn new(head_dim: usize, bits: u32, seed: u64, use_wht: bool) -> Self {
        let quantizer = TurboQuantProd::new(head_dim, bits, seed, use_wht, None);
        KeyCompressor {
            quantizer,
            d: head_dim,
            apply_qjl: bits <= 3, // auto: on for low bits, off for high
        }
    }

    /// Create from a TurboQuantConfig (recommended API).
    ///
    /// Respects all config settings: rotation method, QJL toggle, etc.
    pub fn from_config(config: &TurboQuantConfig) -> Self {
        let quantizer = TurboQuantProd::new(
            config.dim,
            config.bits,
            config.seed,
            config.use_wht(),
            config.qjl_dim,
        );
        KeyCompressor {
            quantizer,
            d: config.dim,
            apply_qjl: config.should_apply_qjl(),
        }
    }

    /// Compress a batch of key vectors.
    ///
    /// `states` is a flat slice of B*H*S*D f64 values in row-major order.
    /// `shape` is (B, H, S, D).
    pub fn compress(&self, states: &[f64], shape: (usize, usize, usize, usize)) -> CompressedKeyBatch {
        let (b, h, s, d) = shape;
        debug_assert_eq!(d, self.d);
        debug_assert_eq!(states.len(), b * h * s * d);

        let n_vectors = b * h * s;
        let mut keys = Vec::with_capacity(n_vectors);
        let mut buf = vec![0.0f64; 3 * d];

        for v in 0..n_vectors {
            let offset = v * d;
            let x = &states[offset..offset + d];

            let compressed = self.quantizer.quantize(x, &mut buf);

            // Reconstruct MSE result for storage (Python CompressorV2 stores k_mse as f16)
            let mse_repr = MseCompressed {
                indices: compressed.mse_indices.clone(),
                vec_norm: compressed.vec_norm,
                outliers: Vec::new(),
            };
            let mut x_mse = vec![0.0f64; d];
            self.quantizer.mse().dequantize(&mse_repr, &mut buf, &mut x_mse);

            let packed_signs = BitPackedSigns::pack(&compressed.qjl_signs);

            keys.push(CompressedKey {
                k_mse: x_mse.iter().map(|&v| v as f32).collect(),
                signs: packed_signs,
                residual_norm: compressed.residual_norm as f32,
            });
        }

        CompressedKeyBatch { keys, shape }
    }

    /// Compute asymmetric attention scores: queries x compressed_keys.
    ///
    /// Full TurboQuant estimator (matching `compressors.py:asymmetric_attention_scores`):
    ///   score = <q, k_mse> + ||r|| * sqrt(pi/2) / m * <S@q, signs>
    ///
    /// Term 1 uses the stored MSE reconstruction.
    /// Term 2 (QJL correction) projects the query through the QJL matrix S,
    /// then computes the inner product with bit-packed signs via POPCNT.
    ///
    /// The QJL correction is most valuable at low bit-widths (2-bit) where
    /// MSE residuals are large.  At higher bit-widths (4+), MSE-only can
    /// be more accurate because the 1-bit sign noise exceeds the residual.
    ///
    /// `queries` is (n_queries, d) flat f64 array.
    /// Returns (n_queries, n_keys) attention scores.
    pub fn attention_scores(&self, queries: &[f64], n_queries: usize, batch: &CompressedKeyBatch) -> Vec<f64> {
        let d = self.d;
        let m = self.quantizer.qjl_dim();
        let s_matrix = self.quantizer.s_matrix();
        let correction_scale = (std::f64::consts::FRAC_PI_2).sqrt() / m as f64;
        let n_keys = batch.keys.len();
        let mut scores = vec![0.0f64; n_queries * n_keys];

        for qi in 0..n_queries {
            let q = &queries[qi * d..(qi + 1) * d];

            // Project query through QJL matrix: s_q[j] = sum_i S[j*d + i] * q[i]
            let mut s_q = vec![0.0f64; m];
            for (j, sq_val) in s_q.iter_mut().enumerate() {
                let row_start = j * d;
                let mut dot = 0.0f64;
                for i in 0..d {
                    dot += s_matrix[row_start + i] * q[i];
                }
                *sq_val = dot;
            }

            for (ki, key) in batch.keys.iter().enumerate() {
                // Term 1: <q, k_mse>
                let term1: f64 = q.iter()
                    .zip(key.k_mse.iter())
                    .map(|(&qv, &kv)| qv * kv as f64)
                    .sum();

                // Term 2: QJL correction (only when beneficial)
                // On at bits<=3 where MSE residuals are large.
                // Off at bits>=4 where 1-bit sign noise exceeds residual.
                let term2 = if self.apply_qjl {
                    let sign_dot = key.signs.inner_product(&s_q);
                    key.residual_norm as f64 * correction_scale * sign_dot
                } else {
                    0.0
                };

                scores[qi * n_keys + ki] = term1 + term2;
            }
        }

        scores
    }
}

/// Value compressor (TurboQuantMSE: MSE-only, no QJL).
///
/// Port of `compressors.py:TurboQuantCompressorMSE`.
pub struct ValueCompressor {
    quantizer: TurboQuantMSE,
    d: usize,
}

impl ValueCompressor {
    pub fn new(head_dim: usize, bits: u32, seed: u64, use_wht: bool) -> Self {
        let quantizer = TurboQuantMSE::new(head_dim, bits, seed, use_wht);
        ValueCompressor {
            quantizer,
            d: head_dim,
        }
    }

    /// Compress a batch of value vectors.
    pub fn compress(&self, states: &[f64], shape: (usize, usize, usize, usize)) -> CompressedValueBatch {
        let (b, h, s, d) = shape;
        debug_assert_eq!(d, self.d);
        debug_assert_eq!(states.len(), b * h * s * d);

        let n_vectors = b * h * s;
        let mut values = Vec::with_capacity(n_vectors);
        let mut buf = vec![0.0f64; 3 * d];

        for v in 0..n_vectors {
            let offset = v * d;
            let x = &states[offset..offset + d];
            let compressed = self.quantizer.quantize(x, &mut buf);
            values.push(CompressedValue {
                indices: compressed.indices,
                vec_norm: compressed.vec_norm as f32,
            });
        }

        CompressedValueBatch { values, shape }
    }

    /// Decompress all values back to f64.
    ///
    /// Returns flat (B*H*S*D) f64 array.
    pub fn decompress(&self, batch: &CompressedValueBatch) -> Vec<f64> {
        let d = self.d;
        let n_vectors = batch.values.len();
        let mut result = vec![0.0f64; n_vectors * d];
        let mut buf = vec![0.0f64; 3 * d];

        for (v, cv) in batch.values.iter().enumerate() {
            let mse_compressed = MseCompressed {
                indices: cv.indices.clone(),
                vec_norm: cv.vec_norm as f64,
                outliers: Vec::new(),
            };
            self.quantizer.dequantize(&mse_compressed, &mut buf, &mut result[v * d..(v + 1) * d]);
        }

        result
    }
}

/// Memory usage statistics for compressed KV cache.
pub struct MemoryUsage {
    /// Total bits for compressed keys.
    pub key_bits: usize,
    /// Total bits for compressed values.
    pub value_bits: usize,
    /// Total compressed bits.
    pub total_bits: usize,
    /// Equivalent fp16 bits (uncompressed).
    pub fp16_bits: usize,
    /// Compression ratio.
    pub compression_ratio: f64,
}

/// Compute memory usage for a compressed KV cache layer.
pub fn memory_usage(
    key_batch: &CompressedKeyBatch,
    value_batch: &CompressedValueBatch,
) -> MemoryUsage {
    let d = key_batch.shape.3;
    let n_keys = key_batch.keys.len();
    let n_values = value_batch.values.len();

    // Keys: f32 MSE reconstruction (32*d) + packed signs (d bits) + residual norm (16 bits)
    // In a production system, we'd store indices (mse_bits*d) instead of f32 reconstruction.
    // For now, count the packed representation:
    let key_bits_per_vec = d * 32 + d + 16; // k_mse(f32) + signs(1bit) + norm(f16)
    let key_bits = n_keys * key_bits_per_vec;

    // Values: indices (bits_per_coord * d) + vec_norm (16 bits)
    // We store u8 indices, so 8*d bits + 16 bits per vector
    let value_bits_per_vec = d * 8 + 16;
    let value_bits = n_values * value_bits_per_vec;

    let total_bits = key_bits + value_bits;
    let fp16_bits = (n_keys + n_values) * d * 16;

    MemoryUsage {
        key_bits,
        value_bits,
        total_bits,
        fp16_bits,
        compression_ratio: fp16_bits as f64 / total_bits as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_states(b: usize, h: usize, s: usize, d: usize, seed: u64) -> Vec<f64> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let normal = StandardNormal;
        (0..b * h * s * d).map(|_| normal.sample(&mut rng)).collect()
    }

    #[test]
    fn test_value_compress_decompress() {
        let d = 64;
        let bits = 3;
        let shape = (1, 2, 4, d); // B=1, H=2, S=4, D=64
        let states = random_states(1, 2, 4, d, 42);

        let compressor = ValueCompressor::new(d, bits, 42, false);
        let compressed = compressor.compress(&states, shape);
        assert_eq!(compressed.values.len(), 8); // 1*2*4

        let decompressed = compressor.decompress(&compressed);
        assert_eq!(decompressed.len(), states.len());

        // Check reconstruction quality (MSE should be bounded)
        let mse: f64 = states
            .iter()
            .zip(decompressed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / states.len() as f64;

        // MSE depends on vector magnitude and quantization bit-width.
        // For random N(0,1) vectors at d=64, 3-bit: MSE ~ 0.15-0.25.
        assert!(
            mse < 0.5,
            "Value decompression MSE too high: {}",
            mse
        );
    }

    #[test]
    fn test_key_compress() {
        let d = 64;
        let bits = 3;
        let shape = (1, 1, 8, d);
        let states = random_states(1, 1, 8, d, 99);

        let compressor = KeyCompressor::new(d, bits, 42, false);
        let compressed = compressor.compress(&states, shape);
        assert_eq!(compressed.keys.len(), 8);

        // Each key should have d MSE values and packed signs
        for key in &compressed.keys {
            assert_eq!(key.k_mse.len(), d);
            assert_eq!(key.signs.len(), d);
            assert!(key.residual_norm >= 0.0);
        }
    }

    #[test]
    fn test_attention_scores_shape() {
        let d = 32;
        let bits = 3;
        let shape = (1, 1, 4, d);
        let states = random_states(1, 1, 4, d, 42);
        let queries = random_states(1, 1, 2, d, 99);

        let compressor = KeyCompressor::new(d, bits, 42, false);
        let compressed = compressor.compress(&states, shape);
        let scores = compressor.attention_scores(&queries, 2, &compressed);

        assert_eq!(scores.len(), 2 * 4); // n_queries * n_keys
    }
}
