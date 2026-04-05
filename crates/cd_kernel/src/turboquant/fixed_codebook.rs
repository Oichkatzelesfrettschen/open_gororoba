//! Fixed-point codebook for TurboQuant: Q16.16 and Q32.32 variants.
//!
//! Stores Lloyd-Max centroids and boundaries in fixed-point format for:
//! 1. Exact order-independent inner product accumulation on GPU (IADD3)
//! 2. Comparison with f32 codebook accuracy
//! 3. Basis for the Q16.16 CUDA kernel (turboquant_dequant_dot_q16)
//!
//! Based on steinmarder's precision-tier results:
//!   Q16.16: 1945 MLUPS, zero mass drift, resolution 1.53e-5
//!   Q32.32: 1035 MLUPS, zero mass drift, resolution 2.33e-10
//!   FP32:   1956 MLUPS, mass_drift 4.77e-7

use super::fixed_point::{Q16_16, Q32_32};
use crate::lloyd_max;

/// Q16.16 fixed-point codebook.
#[derive(Clone, Debug)]
pub struct FixedCodebook16 {
    /// Centroids in Q16.16.
    pub centroids: Vec<Q16_16>,
    /// Boundaries in Q16.16.
    pub boundaries: Vec<Q16_16>,
    /// Original f32 centroids for comparison.
    pub centroids_f32: Vec<f32>,
    /// Quantization error from f32->Q16.16 conversion.
    pub conversion_error: f64,
}

/// Q32.32 fixed-point codebook.
#[derive(Clone, Debug)]
pub struct FixedCodebook32 {
    pub centroids: Vec<Q32_32>,
    pub boundaries: Vec<Q32_32>,
    pub centroids_f32: Vec<f32>,
    pub conversion_error: f64,
}

impl FixedCodebook16 {
    /// Convert a Lloyd-Max codebook to Q16.16.
    pub fn from_lloyd_max(d: usize, bits: u32) -> Self {
        let cb = lloyd_max::get_codebook(d, bits);
        let centroids_q: Vec<Q16_16> = cb.centroids.iter().map(|&c| Q16_16::from_f32(c)).collect();
        let boundaries_q: Vec<Q16_16> =
            cb.boundaries.iter().map(|&b| Q16_16::from_f32(b)).collect();

        // Measure conversion error
        let error: f64 = cb
            .centroids
            .iter()
            .zip(centroids_q.iter())
            .map(|(&f, &q)| (f as f64 - q.to_f64()).powi(2))
            .sum::<f64>()
            / cb.centroids.len() as f64;

        FixedCodebook16 {
            centroids: centroids_q,
            boundaries: boundaries_q,
            centroids_f32: cb.centroids.clone(),
            conversion_error: error.sqrt(),
        }
    }

    /// Quantize using Q16.16 boundary search.
    pub fn quantize(&self, value: f64) -> u8 {
        let v = Q16_16::from_f64(value);
        let mut idx = 0u8;
        for &b in &self.boundaries {
            if v.0 > b.0 {
                // integer comparison -- exact, no floating-point issues
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Dequantize: return Q16.16 centroid.
    pub fn dequantize(&self, idx: u8) -> Q16_16 {
        self.centroids[idx as usize]
    }

    /// Exact dot product using Q16.16 accumulation.
    pub fn dot_product(&self, query: &[f64], indices: &[u8]) -> f64 {
        super::fixed_point::dot_product_q16(
            &query.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            indices,
            &self.centroids,
        )
    }
}

impl FixedCodebook32 {
    /// Convert a Lloyd-Max codebook to Q32.32.
    pub fn from_lloyd_max(d: usize, bits: u32) -> Self {
        let cb = lloyd_max::get_codebook(d, bits);
        let centroids_q: Vec<Q32_32> = cb
            .centroids
            .iter()
            .map(|&c| Q32_32::from_f64(c as f64))
            .collect();
        let boundaries_q: Vec<Q32_32> = cb
            .boundaries
            .iter()
            .map(|&b| Q32_32::from_f64(b as f64))
            .collect();

        let error: f64 = cb
            .centroids
            .iter()
            .zip(centroids_q.iter())
            .map(|(&f, &q)| (f as f64 - q.to_f64()).powi(2))
            .sum::<f64>()
            / cb.centroids.len() as f64;

        FixedCodebook32 {
            centroids: centroids_q,
            boundaries: boundaries_q,
            centroids_f32: cb.centroids.clone(),
            conversion_error: error.sqrt(),
        }
    }

    /// Quantize using Q32.32 boundary search.
    pub fn quantize(&self, value: f64) -> u8 {
        let v = Q32_32::from_f64(value);
        let mut idx = 0u8;
        for &b in &self.boundaries {
            if v.0 > b.0 {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Exact dot product using Q32.32 accumulation.
    pub fn dot_product(&self, query: &[f64], indices: &[u8]) -> f64 {
        super::fixed_point::dot_product_q32(query, indices, &self.centroids)
    }
}

/// Compare f32, Q16.16, and Q32.32 codebook accuracy.
///
/// Returns (f32_mse, q16_mse, q32_mse, q16_conversion_err, q32_conversion_err).
pub fn compare_codebook_precision(d: usize, bits: u32, n_test: usize) -> (f64, f64, f64, f64, f64) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::StandardNormal;

    let cb_f32 = lloyd_max::get_codebook(d, bits);
    let cb_q16 = FixedCodebook16::from_lloyd_max(d, bits);
    let cb_q32 = FixedCodebook32::from_lloyd_max(d, bits);

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let normal = StandardNormal;
    let sigma = 1.0 / (d as f64).sqrt();

    let mut f32_mse = 0.0f64;
    let mut q16_mse = 0.0f64;
    let mut q32_mse = 0.0f64;

    for _ in 0..n_test {
        let v: f64 = <rand_distr::StandardNormal as rand_distr::Distribution<f64>>::sample(
            &normal, &mut rng,
        ) * sigma;

        // f32 quantize/dequantize
        let f32_idx = {
            let mut idx = 0u8;
            for &b in cb_f32.boundaries.iter() {
                if v as f32 > b {
                    idx += 1;
                } else {
                    break;
                }
            }
            idx
        };
        let f32_recon = cb_f32.centroids[f32_idx as usize] as f64;
        f32_mse += (v - f32_recon).powi(2);

        // Q16.16 quantize/dequantize
        let q16_idx = cb_q16.quantize(v);
        let q16_recon = cb_q16.dequantize(q16_idx).to_f64();
        q16_mse += (v - q16_recon).powi(2);

        // Q32.32 quantize/dequantize
        let q32_idx = cb_q32.quantize(v);
        let q32_recon = cb_q32.centroids[q32_idx as usize].to_f64();
        q32_mse += (v - q32_recon).powi(2);
    }

    (
        f32_mse / n_test as f64,
        q16_mse / n_test as f64,
        q32_mse / n_test as f64,
        cb_q16.conversion_error,
        cb_q32.conversion_error,
    )
}

/// Map steinmarder precision tiers to TurboQuant codebook operations.
///
/// | Tier    | bytes/val | GPU pipeline     | TurboQuant use              | Expected perf  |
/// |---------|-----------|------------------|-----------------------------|----------------|
/// | FP8     | 1         | FFMA             | Ultra-compressed codebook   | Highest MLUPS  |
/// | INT8    | 1         | IADD3+FFMA       | Standard codebook indices   | Very high      |
/// | FP16    | 2         | HFMA2+FFMA       | Enhanced precision codebook | High           |
/// | Q16.16  | 4         | IADD3+FFMA       | Exact accumulation codebook | Medium (=FP32) |
/// | FP32    | 4         | FFMA             | Default codebook            | Medium         |
/// | Q32.32  | 8         | INT64+FFMA       | Ultra-precision codebook    | 1.88x FP64     |
/// | FP64    | 8         | FP64 ALU         | Validation reference        | Slow (gaming)  |
pub struct PrecisionTierMap;

impl PrecisionTierMap {
    /// Recommend the optimal precision tier for TurboQuant based on requirements.
    pub fn recommend(need_exact: bool, need_speed: bool, _gpu_has_fp64: bool) -> &'static str {
        match (need_exact, need_speed) {
            (true, true) => "Q16.16 (exact + FP32-speed on Ada via IADD3)",
            (true, false) => "Q32.32 (ultra-exact, 1.88x faster than FP64 on gaming)",
            (false, true) => "INT8 (Pareto-optimal: highest MLUPS at 1 byte/val)",
            (false, false) => "FP32 (default, no special hardware needed)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q16_codebook_matches_f32() {
        let cb_f32 = lloyd_max::get_codebook(128, 3);
        let cb_q16 = FixedCodebook16::from_lloyd_max(128, 3);

        println!("Q16.16 conversion error: {:.2e}", cb_q16.conversion_error);
        // Q16.16 resolution is 1/65536 = 1.53e-5
        assert!(
            cb_q16.conversion_error < 1e-4,
            "Q16.16 conversion error too high: {}",
            cb_q16.conversion_error
        );

        // Centroids should match to Q16.16 resolution
        for (i, (&f, &q)) in cb_f32
            .centroids
            .iter()
            .zip(cb_q16.centroids.iter())
            .enumerate()
        {
            let diff = (f as f64 - q.to_f64()).abs();
            assert!(
                diff < 2e-5,
                "Centroid {} mismatch: f32={}, q16={}, diff={:.2e}",
                i,
                f,
                q.to_f64(),
                diff
            );
        }
    }

    #[test]
    fn test_q32_codebook_matches_f32() {
        let cb_q32 = FixedCodebook32::from_lloyd_max(128, 3);
        println!("Q32.32 conversion error: {:.2e}", cb_q32.conversion_error);
        assert!(
            cb_q32.conversion_error < 1e-9,
            "Q32.32 conversion error too high: {}",
            cb_q32.conversion_error
        );
    }

    #[test]
    fn test_codebook_precision_comparison() {
        let (f32_mse, q16_mse, q32_mse, q16_err, q32_err) =
            compare_codebook_precision(128, 3, 100_000);

        println!("\nCodebook precision comparison (d=128, 3-bit, 100K samples):");
        println!("  f32 MSE:  {:.10e}", f32_mse);
        println!("  Q16 MSE:  {:.10e}", q16_mse);
        println!("  Q32 MSE:  {:.10e}", q32_mse);
        println!("  Q16 conv: {:.2e}", q16_err);
        println!("  Q32 conv: {:.2e}", q32_err);
        println!("  Q16/f32:  {:.6}", q16_mse / f32_mse);
        println!("  Q32/f32:  {:.6}", q32_mse / f32_mse);

        // Q32.32 should match f32 very closely
        assert!(
            (q32_mse / f32_mse - 1.0).abs() < 0.01,
            "Q32.32 should match f32 within 1%"
        );
        // Q16.16 may differ slightly due to resolution
        assert!(
            (q16_mse / f32_mse - 1.0).abs() < 0.1,
            "Q16.16 should match f32 within 10%"
        );
    }

    #[test]
    fn test_fixed_codebook_quantize() {
        let cb = FixedCodebook16::from_lloyd_max(128, 3);
        // Zero should map to middle index (3 or 4)
        let idx = cb.quantize(0.0);
        assert!(idx == 3 || idx == 4, "Zero maps to index {}", idx);
        // Large positive -> max index
        assert_eq!(cb.quantize(1.0), 7);
        // Large negative -> index 0
        assert_eq!(cb.quantize(-1.0), 0);
    }

    #[test]
    fn test_precision_tier_recommendations() {
        assert!(PrecisionTierMap::recommend(true, true, false).contains("Q16.16"));
        assert!(PrecisionTierMap::recommend(true, false, false).contains("Q32.32"));
        assert!(PrecisionTierMap::recommend(false, true, false).contains("INT8"));
        assert!(PrecisionTierMap::recommend(false, false, false).contains("FP32"));
    }
}
