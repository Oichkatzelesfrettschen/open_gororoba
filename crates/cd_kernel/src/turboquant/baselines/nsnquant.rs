//! NSNQuant baseline: calibration-free normalize-shift-normalize + Hadamard.
//!
//! NSNQuant (arXiv 2505.18231) achieves data-independent quantization via:
//!   1. Token-wise normalization (Normalize)
//!   2. Channel-wise centering (Shift)
//!   3. Second token-wise normalization (Normalize)
//!   + Hadamard transform for coordinate decorrelation
//!
//! After NSN preprocessing, coordinates are approximately standard normal,
//! enabling a single reusable codebook without calibration data.
//! This is similar to TurboQuant's rotation approach but uses explicit
//! normalization instead of random rotation.

/// Apply the NSN (Normalize-Shift-Normalize) preprocessing.
///
/// 1. Normalize: scale each vector to unit norm
/// 2. Shift: subtract per-channel mean across all vectors
/// 3. Normalize: re-normalize each vector to unit norm
///
/// Returns (preprocessed_data, channel_means, vector_norms_1, vector_norms_2)
/// for reconstruction.
pub fn nsn_preprocess(
    data: &[f64], // (n_vectors, d) flat row-major
    n: usize,
    d: usize,
) -> NsnState {
    let mut buf = data.to_vec();

    // Step 1: Token-wise normalization
    let mut norms_1 = vec![0.0f64; n];
    for t in 0..n {
        let row = &mut buf[t * d..(t + 1) * d];
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        norms_1[t] = norm;
        if norm > 1e-15 {
            let inv = 1.0 / norm;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }

    // Step 2: Channel-wise centering (subtract per-channel mean)
    let mut channel_means = vec![0.0f64; d];
    for c in 0..d {
        let mut sum = 0.0;
        for t in 0..n {
            sum += buf[t * d + c];
        }
        channel_means[c] = sum / n as f64;
    }
    for t in 0..n {
        for c in 0..d {
            buf[t * d + c] -= channel_means[c];
        }
    }

    // Step 3: Second token-wise normalization
    let mut norms_2 = vec![0.0f64; n];
    for t in 0..n {
        let row = &mut buf[t * d..(t + 1) * d];
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        norms_2[t] = norm;
        if norm > 1e-15 {
            let inv = 1.0 / norm;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }

    NsnState {
        preprocessed: buf,
        channel_means,
        norms_1,
        norms_2,
        n,
        d,
    }
}

/// Apply WHT to each preprocessed vector (decorrelation step).
pub fn nsn_apply_wht(state: &mut NsnState) {
    use crate::turboquant::rotation::wht_inplace;
    for t in 0..state.n {
        let row = &mut state.preprocessed[t * state.d..(t + 1) * state.d];
        wht_inplace(row);
    }
}

/// Quantize NSN-preprocessed data with a shared codebook.
///
/// After NSN + WHT, coordinates are approximately N(0, 1/d).
/// Use the same Lloyd-Max codebook as TurboQuant.
pub fn nsn_quantize(state: &NsnState, bits: u32) -> NsnCompressed {
    let d = state.d;
    let codebook = crate::lloyd_max::get_codebook(d, bits);
    let boundaries = &codebook.boundaries;

    let mut indices = Vec::with_capacity(state.n * d);
    for &v in &state.preprocessed {
        let mut idx = 0u8;
        for &b in boundaries.iter() {
            if v as f32 > b {
                idx += 1;
            } else {
                break;
            }
        }
        indices.push(idx);
    }

    NsnCompressed {
        indices,
        channel_means: state.channel_means.clone(),
        norms_1: state.norms_1.clone(),
        norms_2: state.norms_2.clone(),
        n: state.n,
        d: state.d,
        bits,
    }
}

/// Dequantize NSN-compressed data.
///
/// Reverse: codebook lookup -> inverse WHT -> add channel means ->
/// denormalize (norms_2) -> denormalize (norms_1)
pub fn nsn_dequantize(compressed: &NsnCompressed) -> Vec<f64> {
    use crate::turboquant::rotation::wht_inplace;
    let d = compressed.d;
    let n = compressed.n;
    let codebook = crate::lloyd_max::get_codebook(d, compressed.bits);

    // Step 1: codebook lookup
    let mut buf: Vec<f64> = compressed
        .indices
        .iter()
        .map(|&idx| codebook.centroids[idx as usize] as f64)
        .collect();

    // Step 2: inverse WHT (WHT is self-inverse)
    for t in 0..n {
        let row = &mut buf[t * d..(t + 1) * d];
        wht_inplace(row);
    }

    // Step 3: denormalize (norms_2) and add channel means
    for t in 0..n {
        let norm2 = compressed.norms_2[t];
        for c in 0..d {
            buf[t * d + c] = buf[t * d + c] * norm2 + compressed.channel_means[c];
        }
    }

    // Step 4: denormalize (norms_1)
    for t in 0..n {
        let norm1 = compressed.norms_1[t];
        for c in 0..d {
            buf[t * d + c] *= norm1;
        }
    }

    buf
}

#[derive(Clone, Debug)]
pub struct NsnState {
    pub preprocessed: Vec<f64>,
    pub channel_means: Vec<f64>,
    pub norms_1: Vec<f64>,
    pub norms_2: Vec<f64>,
    pub n: usize,
    pub d: usize,
}

#[derive(Clone, Debug)]
pub struct NsnCompressed {
    pub indices: Vec<u8>,
    pub channel_means: Vec<f64>,
    pub norms_1: Vec<f64>,
    pub norms_2: Vec<f64>,
    pub n: usize,
    pub d: usize,
    pub bits: u32,
}

impl NsnCompressed {
    /// Memory in bits (indices + metadata).
    pub fn total_bits(&self) -> usize {
        let index_bits = self.n * self.d * self.bits as usize;
        let meta_bits = (self.d + 2 * self.n) * 64; // channel_means(f64) + 2 norms(f64)
        index_bits + meta_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nsn_preprocess_unit_norm() {
        let n = 10;
        let d = 64;
        let data: Vec<f64> = (0..n * d).map(|i| (i as f64 * 0.1).sin()).collect();

        let state = nsn_preprocess(&data, n, d);

        // After NSN, each vector should have unit norm
        for t in 0..n {
            let row = &state.preprocessed[t * d..(t + 1) * d];
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "Vector {} has norm {} after NSN",
                t,
                norm
            );
        }
    }

    #[test]
    fn test_nsn_roundtrip() {
        let n = 20;
        let d = 128; // must be power of 2 for WHT
        let data: Vec<f64> = (0..n * d).map(|i| (i as f64 * 0.05).cos()).collect();

        let mut state = nsn_preprocess(&data, n, d);
        nsn_apply_wht(&mut state);
        let compressed = nsn_quantize(&state, 3);
        let decompressed = nsn_dequantize(&compressed);

        let mse: f64 = data
            .iter()
            .zip(decompressed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / (n * d) as f64;
        // NSNQuant at 3-bit should have reasonable MSE
        assert!(mse < 0.5, "NSNQuant roundtrip MSE too high: {}", mse);
    }

    #[test]
    fn test_nsn_zero_mean_after_shift() {
        let n = 50;
        let d = 64;
        let data: Vec<f64> = (0..n * d).map(|i| i as f64 * 0.001 + 1.0).collect();

        let state = nsn_preprocess(&data, n, d);

        // After NSN, per-channel mean should be near zero.
        // The second normalization (step 3) slightly perturbs the zero-mean
        // property established by step 2, but it should remain small.
        for c in 0..d {
            let mean: f64 = (0..n).map(|t| state.preprocessed[t * d + c]).sum::<f64>() / n as f64;
            assert!(
                mean.abs() < 0.1,
                "Channel {} has mean {} after NSN (expected near zero)",
                c,
                mean
            );
        }
    }
}
