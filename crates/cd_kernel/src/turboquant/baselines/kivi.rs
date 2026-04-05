//! KIVI baseline: asymmetric 2-bit KV cache quantization.
//!
//! KIVI (ICML 2024, arXiv 2402.02750) uses different grouping strategies
//! for keys and values:
//!   - Keys: per-channel quantization (group across sequence dimension)
//!   - Values: per-token quantization (group across head_dim dimension)
//!
//! This asymmetry is motivated by the observation that key and value
//! distributions have fundamentally different outlier patterns:
//! keys have per-channel outliers, values have per-token outliers.

/// Per-channel key quantization (KIVI pattern).
///
/// For each channel c in [0, d), find min/max across all S tokens,
/// then quantize that channel uniformly for all tokens.
pub fn kivi_quantize_keys(
    keys: &[f64], // (S, d) flat row-major
    s: usize,     // sequence length
    d: usize,     // head dimension
    bits: u32,
) -> KiviCompressedKeys {
    let n_levels = (1u32 << bits) as f64;
    let mut scales = vec![0.0f32; d];
    let mut zero_points = vec![0.0f32; d];
    let mut indices = vec![0u8; s * d];

    // Per-channel: find min/max across all tokens
    for c in 0..d {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for t in 0..s {
            let v = keys[t * d + c];
            min = min.min(v);
            max = max.max(v);
        }
        let range = max - min;
        let scale = range / (n_levels - 1.0);
        scales[c] = scale.max(1e-15) as f32;
        zero_points[c] = min as f32;

        // Quantize all tokens for this channel
        for t in 0..s {
            let v = keys[t * d + c];
            let idx = ((v - min) / scale.max(1e-15))
                .round()
                .clamp(0.0, n_levels - 1.0) as u8;
            indices[t * d + c] = idx;
        }
    }

    KiviCompressedKeys {
        indices,
        scales,
        zero_points,
        s,
        d,
        bits,
    }
}

/// Per-token value quantization (KIVI pattern).
///
/// For each token t in [0, S), find min/max across all d dimensions,
/// then quantize that token uniformly for all dimensions.
pub fn kivi_quantize_values(
    values: &[f64], // (S, d) flat row-major
    s: usize,
    d: usize,
    bits: u32,
) -> KiviCompressedValues {
    let n_levels = (1u32 << bits) as f64;
    let mut scales = vec![0.0f32; s];
    let mut zero_points = vec![0.0f32; s];
    let mut indices = vec![0u8; s * d];

    // Per-token: find min/max across all dimensions
    for t in 0..s {
        let row = &values[t * d..(t + 1) * d];
        let min = row.iter().copied().fold(f64::MAX, f64::min);
        let max = row.iter().copied().fold(f64::MIN, f64::max);
        let range = max - min;
        let scale = range / (n_levels - 1.0);
        scales[t] = scale.max(1e-15) as f32;
        zero_points[t] = min as f32;

        for (c, &v) in row.iter().enumerate() {
            let idx = ((v - min) / scale.max(1e-15))
                .round()
                .clamp(0.0, n_levels - 1.0) as u8;
            indices[t * d + c] = idx;
        }
    }

    KiviCompressedValues {
        indices,
        scales,
        zero_points,
        s,
        d,
        bits,
    }
}

#[derive(Clone, Debug)]
pub struct KiviCompressedKeys {
    pub indices: Vec<u8>,
    pub scales: Vec<f32>,      // (d,) per-channel
    pub zero_points: Vec<f32>, // (d,) per-channel
    pub s: usize,
    pub d: usize,
    pub bits: u32,
}

#[derive(Clone, Debug)]
pub struct KiviCompressedValues {
    pub indices: Vec<u8>,
    pub scales: Vec<f32>,      // (S,) per-token
    pub zero_points: Vec<f32>, // (S,) per-token
    pub s: usize,
    pub d: usize,
    pub bits: u32,
}

impl KiviCompressedKeys {
    /// Dequantize keys back to f64.
    pub fn dequantize(&self) -> Vec<f64> {
        let mut out = vec![0.0f64; self.s * self.d];
        for t in 0..self.s {
            for c in 0..self.d {
                let idx = self.indices[t * self.d + c];
                out[t * self.d + c] =
                    idx as f64 * self.scales[c] as f64 + self.zero_points[c] as f64;
            }
        }
        out
    }

    /// Memory in bits.
    pub fn total_bits(&self) -> usize {
        self.s * self.d * self.bits as usize  // indices
        + self.d * 32                          // scales (f32 per channel)
        + self.d * 32 // zero_points (f32 per channel)
    }
}

impl KiviCompressedValues {
    /// Dequantize values back to f64.
    pub fn dequantize(&self) -> Vec<f64> {
        let mut out = vec![0.0f64; self.s * self.d];
        for t in 0..self.s {
            for c in 0..self.d {
                let idx = self.indices[t * self.d + c];
                out[t * self.d + c] =
                    idx as f64 * self.scales[t] as f64 + self.zero_points[t] as f64;
            }
        }
        out
    }

    /// Memory in bits.
    pub fn total_bits(&self) -> usize {
        self.s * self.d * self.bits as usize
        + self.s * 32                          // scales (f32 per token)
        + self.s * 32 // zero_points (f32 per token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kivi_keys_roundtrip() {
        let s = 64;
        let d = 128;
        let keys: Vec<f64> = (0..s * d).map(|i| (i as f64 * 0.01).sin()).collect();

        let compressed = kivi_quantize_keys(&keys, s, d, 2);
        assert_eq!(compressed.indices.len(), s * d);
        assert!(compressed.indices.iter().all(|&i| i <= 3)); // 2-bit: 0..3

        let decompressed = compressed.dequantize();
        let mse: f64 = keys
            .iter()
            .zip(decompressed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / (s * d) as f64;
        assert!(mse < 0.05, "KIVI keys MSE too high: {}", mse);
    }

    #[test]
    fn test_kivi_values_roundtrip() {
        let s = 64;
        let d = 128;
        let values: Vec<f64> = (0..s * d).map(|i| (i as f64 * 0.01).cos()).collect();

        let compressed = kivi_quantize_values(&values, s, d, 2);
        let decompressed = compressed.dequantize();
        let mse: f64 = values
            .iter()
            .zip(decompressed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / (s * d) as f64;
        assert!(mse < 0.05, "KIVI values MSE too high: {}", mse);
    }

    #[test]
    fn test_kivi_memory() {
        let s = 2048;
        let d = 128;
        let keys: Vec<f64> = vec![0.0; s * d];
        let compressed = kivi_quantize_keys(&keys, s, d, 2);
        let fp16_bits = s * d * 16;
        let kivi_bits = compressed.total_bits();
        let ratio = fp16_bits as f64 / kivi_bits as f64;
        // 2-bit should give ~8x compression (minus metadata)
        assert!(ratio > 5.0, "KIVI compression ratio too low: {:.1}x", ratio);
    }
}
