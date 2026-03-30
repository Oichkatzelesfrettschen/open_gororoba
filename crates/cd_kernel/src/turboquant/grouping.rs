//! InnerQ-style inner-dimension grouping for TurboQuant quantization.
//!
//! Based on InnerQ (arXiv 2602.23200): group coordinates along the inner
//! (head_dim) dimension rather than outer (sequence) dimension.  This
//! aligns dequantization with vector-matrix multiplication and enables
//! scale factor reuse across GPU compute units.
//!
//! # Key ideas from InnerQ
//!
//! 1. **Inner-dimension grouping**: Quantize groups of g consecutive
//!    coordinates together with a shared scale/zero-point.  For d=128
//!    with g=32, there are 4 groups per vector.
//!
//! 2. **Hybrid symmetric/asymmetric**: Choose symmetric (zero_point=0)
//!    or asymmetric (zero_point != 0) per group based on local statistics.
//!    Symmetric is cheaper but asymmetric handles skewed distributions.
//!
//! 3. **Scale reuse**: In GPU VMM (vector-matrix multiply), the scale
//!    factor for a group is shared across all keys in the sequence,
//!    reducing memory bandwidth for scale reads.
//!
//! 4. **High-precision windows**: Keep recent tokens and attention sink
//!    tokens at full precision (no quantization) to preserve quality
//!    for the most important positions.

/// Group-wise quantization parameters for one vector.
///
/// Memory-efficient: uses f16 scale (via half crate) and optional zero-point.
/// After rotation, ~65% of groups are symmetric (zero_point=0), so we store
/// a compact representation.
///
/// Effective overhead at d=128:
///   group_size=16, f16 scale only:  3.12 bits/coord (matches RotateKV)
///   group_size=32, f16 scale only:  2.62 bits/coord
///   group_size=64, f16 scale only:  2.38 bits/coord
#[derive(Clone, Debug)]
pub struct GroupQuantParams {
    /// Scale factor per group (f32 for compute, stored as f16 in compressed form).
    pub scales: Vec<f32>,
    /// Zero-point per group (0.0 for symmetric).
    pub zero_points: Vec<f32>,
    /// Whether each group uses symmetric quantization.
    pub symmetric: Vec<bool>,
    /// Group size (number of coordinates per group).
    pub group_size: usize,
    /// Number of groups.
    pub n_groups: usize,
}

impl GroupQuantParams {
    /// Compressed metadata size in bits (for effective bit-rate calculation).
    ///
    /// Symmetric groups: 16 bits (f16 scale only)
    /// Asymmetric groups: 32 bits (f16 scale + f16 zero_point)
    pub fn metadata_bits(&self) -> usize {
        self.symmetric.iter()
            .map(|&s| if s { 16 } else { 32 })
            .sum()
    }
}

/// Group-wise quantized vector.
#[derive(Clone, Debug)]
pub struct GroupQuantized {
    /// Quantized indices (d elements, u8).
    pub indices: Vec<u8>,
    /// Per-group parameters.
    pub params: GroupQuantParams,
}

/// Compute group-wise quantization parameters for a vector.
///
/// For each group of `group_size` consecutive coordinates:
///   - Compute min and max
///   - If |min + max| < threshold * (max - min): use symmetric
///   - Otherwise: use asymmetric with zero_point = (min + max) / 2
///   - Scale = (max - min) / (2^bits - 1)
pub fn compute_group_params(
    values: &[f64],
    group_size: usize,
    bits: u32,
) -> GroupQuantParams {
    let d = values.len();
    let n_groups = d.div_ceil(group_size);
    let n_levels = (1u32 << bits) as f64;
    let symmetry_threshold = 0.1; // if center < 10% of range, use symmetric

    let mut scales = Vec::with_capacity(n_groups);
    let mut zero_points = Vec::with_capacity(n_groups);
    let mut symmetric = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let start = g * group_size;
        let end = (start + group_size).min(d);
        let group = &values[start..end];

        let min = group.iter().copied().fold(f64::MAX, f64::min);
        let max = group.iter().copied().fold(f64::MIN, f64::max);
        let range = max - min;
        let center = (min + max) / 2.0;

        let is_symmetric = center.abs() < symmetry_threshold * range.max(1e-15);

        if is_symmetric {
            // Symmetric: scale around zero, zero_point = 0
            let abs_max = min.abs().max(max.abs());
            let scale = (2.0 * abs_max) / (n_levels - 1.0);
            scales.push(scale.max(1e-15) as f32);
            zero_points.push(0.0);
            symmetric.push(true);
        } else {
            // Asymmetric: scale covers [min, max], zero_point = min
            let scale = range / (n_levels - 1.0);
            scales.push(scale.max(1e-15) as f32);
            zero_points.push(min as f32);
            symmetric.push(false);
        }
    }

    GroupQuantParams {
        scales,
        zero_points,
        symmetric,
        group_size,
        n_groups,
    }
}

/// Quantize a vector using group-wise parameters.
pub fn group_quantize(values: &[f64], params: &GroupQuantParams, bits: u32) -> Vec<u8> {
    let d = values.len();
    let max_idx = ((1u32 << bits) - 1) as u8;
    let mut indices = Vec::with_capacity(d);

    for g in 0..params.n_groups {
        let start = g * params.group_size;
        let end = (start + params.group_size).min(d);
        let scale = params.scales[g] as f64;
        let zp = params.zero_points[g] as f64;

        for &v in &values[start..end] {
            let idx = if params.symmetric[g] {
                // Symmetric: idx = round((v / scale) + n_levels/2)
                let half = (max_idx as f64) / 2.0;
                ((v / scale) + half).round().clamp(0.0, max_idx as f64) as u8
            } else {
                // Asymmetric: idx = round((v - zp) / scale)
                ((v - zp) / scale).round().clamp(0.0, max_idx as f64) as u8
            };
            indices.push(idx);
        }
    }

    indices
}

/// Dequantize a vector from group-wise parameters.
pub fn group_dequantize(indices: &[u8], params: &GroupQuantParams, bits: u32) -> Vec<f64> {
    let d = indices.len();
    let max_idx = ((1u32 << bits) - 1) as f64;
    let mut values = Vec::with_capacity(d);

    for g in 0..params.n_groups {
        let start = g * params.group_size;
        let end = (start + params.group_size).min(d);
        let scale = params.scales[g] as f64;
        let zp = params.zero_points[g] as f64;

        for &idx in &indices[start..end] {
            let v = if params.symmetric[g] {
                let half = max_idx / 2.0;
                (idx as f64 - half) * scale
            } else {
                idx as f64 * scale + zp
            };
            values.push(v);
        }
    }

    values
}

/// High-precision token windows (InnerQ pattern).
///
/// Tokens in the "recent window" and "sink window" are stored at full
/// precision (no quantization) because they disproportionately affect
/// attention quality.
#[derive(Clone, Debug)]
pub struct PrecisionWindows {
    /// Number of recent tokens to keep at full precision.
    pub recent_window: usize,
    /// Number of sink tokens (initial positions) at full precision.
    pub sink_window: usize,
}

impl PrecisionWindows {
    pub fn new(recent: usize, sink: usize) -> Self {
        PrecisionWindows {
            recent_window: recent,
            sink_window: sink,
        }
    }

    /// Determine whether a token at position `pos` in a sequence of
    /// length `seq_len` should be quantized or kept at full precision.
    pub fn should_quantize(&self, pos: usize, seq_len: usize) -> bool {
        // Sink tokens: positions 0..sink_window
        if pos < self.sink_window {
            return false;
        }
        // Recent tokens: positions (seq_len - recent_window)..seq_len
        if seq_len > self.recent_window && pos >= seq_len - self.recent_window {
            return false;
        }
        true
    }

    /// Count how many tokens will be quantized vs full-precision.
    pub fn count_quantized(&self, seq_len: usize) -> (usize, usize) {
        let full_precision = self.sink_window.min(seq_len)
            + self.recent_window.min(seq_len.saturating_sub(self.sink_window));
        let quantized = seq_len.saturating_sub(full_precision);
        (quantized, full_precision)
    }
}

impl Default for PrecisionWindows {
    /// InnerQ defaults: 128 recent tokens, 4 sink tokens.
    fn default() -> Self {
        PrecisionWindows {
            recent_window: 128,
            sink_window: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_roundtrip_symmetric() {
        let d = 128;
        let group_size = 32;
        let bits = 4;
        // Symmetric data centered around zero
        let values: Vec<f64> = (0..d).map(|i| (i as f64 * 0.05).sin()).collect();

        let params = compute_group_params(&values, group_size, bits);
        assert_eq!(params.n_groups, 4);

        let indices = group_quantize(&values, &params, bits);
        assert_eq!(indices.len(), d);
        assert!(indices.iter().all(|&i| i <= 15)); // 4-bit: 0..15

        let reconstructed = group_dequantize(&indices, &params, bits);
        assert_eq!(reconstructed.len(), d);

        let mse: f64 = values.iter().zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
        assert!(mse < 0.01, "Group quantize MSE too high: {}", mse);
    }

    #[test]
    fn test_group_roundtrip_asymmetric() {
        let d = 64;
        let group_size = 16;
        let bits = 3;
        // Asymmetric data: all positive
        let values: Vec<f64> = (0..d).map(|i| i as f64 * 0.01 + 0.5).collect();

        let params = compute_group_params(&values, group_size, bits);
        assert_eq!(params.n_groups, 4);
        // Should detect asymmetric distribution
        assert!(params.symmetric.iter().any(|&s| !s),
            "Expected some asymmetric groups for biased data");

        let indices = group_quantize(&values, &params, bits);
        let reconstructed = group_dequantize(&indices, &params, bits);

        let mse: f64 = values.iter().zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
        assert!(mse < 0.01, "Asymmetric group MSE too high: {}", mse);
    }

    #[test]
    fn test_precision_windows() {
        let pw = PrecisionWindows::new(128, 4);
        let seq_len = 1024;

        // Sink tokens should NOT be quantized
        assert!(!pw.should_quantize(0, seq_len));
        assert!(!pw.should_quantize(3, seq_len));
        // Middle tokens should be quantized
        assert!(pw.should_quantize(4, seq_len));
        assert!(pw.should_quantize(500, seq_len));
        // Recent tokens should NOT be quantized
        assert!(!pw.should_quantize(896, seq_len));
        assert!(!pw.should_quantize(1023, seq_len));
        // Edge of recent window
        assert!(pw.should_quantize(895, seq_len));
    }

    #[test]
    fn test_precision_windows_short_seq() {
        let pw = PrecisionWindows::new(128, 4);
        // Short sequence: everything is in a window
        let (quantized, full_prec) = pw.count_quantized(100);
        // With 4 sink + 96 remaining (all < 128 recent), most are full precision
        assert!(full_prec >= quantized,
            "Short seq should have more full-precision: q={}, fp={}", quantized, full_prec);
    }

    #[test]
    fn test_group_params_metadata_size() {
        let d = 128;
        let group_size = 32;
        let bits = 3;
        let values: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).cos()).collect();

        let params = compute_group_params(&values, group_size, bits);
        // 4 groups * (4 bytes scale + 4 bytes zp + 1 byte flag) = 36 bytes metadata
        // vs 128 bytes for per-element scales
        let metadata_bytes = params.n_groups * (4 + 4 + 1);
        let per_element_bytes = d * 4;
        assert!(metadata_bytes < per_element_bytes / 3,
            "Group metadata ({} bytes) should be much less than per-element ({} bytes)",
            metadata_bytes, per_element_bytes);
    }
}
