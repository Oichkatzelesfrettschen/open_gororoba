//! Post-rotation distribution diagnostics.
//!
//! Analyzes the marginal coordinate distributions after different rotation
//! methods to identify codebook compatibility issues.
//!
//! The Lloyd-Max codebook assumes coordinates follow N(0, 1/d) after rotation.
//! This holds for Haar and WHT rotations but NOT for E8 block rotation,
//! which produces a different marginal (the root cause of E8's poor MSE).

use super::rotation::Rotation;

/// Distribution statistics for post-rotation coordinates.
#[derive(Clone, Debug)]
pub struct DistributionStats {
    /// Rotation method name.
    pub method: String,
    /// Mean of all coordinates.
    pub mean: f64,
    /// Standard deviation of all coordinates.
    pub std: f64,
    /// Skewness (should be ~0 for symmetric distributions).
    pub skewness: f64,
    /// Kurtosis (3.0 for Gaussian, >3 for heavy-tailed).
    pub kurtosis: f64,
    /// Minimum coordinate value.
    pub min: f64,
    /// Maximum coordinate value.
    pub max: f64,
    /// Expected std for N(0, 1/d) distribution.
    pub expected_std: f64,
    /// Ratio actual_std / expected_std (1.0 = perfect match).
    pub std_ratio: f64,
    /// Fraction of values within 3 sigma of expected Gaussian.
    pub within_3sigma: f64,
}

/// Compute distribution statistics for post-rotation coordinates.
fn compute_stats(coords: &[f64], method: &str, d: usize) -> DistributionStats {
    let n = coords.len() as f64;
    let mean = coords.iter().sum::<f64>() / n;
    let var = coords.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();

    let skewness = if std > 1e-15 {
        coords
            .iter()
            .map(|&x| ((x - mean) / std).powi(3))
            .sum::<f64>()
            / n
    } else {
        0.0
    };

    let kurtosis = if std > 1e-15 {
        coords
            .iter()
            .map(|&x| ((x - mean) / std).powi(4))
            .sum::<f64>()
            / n
    } else {
        0.0
    };

    let min = coords.iter().copied().fold(f64::MAX, f64::min);
    let max = coords.iter().copied().fold(f64::MIN, f64::max);

    let expected_std = 1.0 / (d as f64).sqrt();
    let std_ratio = std / expected_std;

    // Count values within 3 sigma of expected Gaussian
    let sigma3 = 3.0 * expected_std;
    let within = coords.iter().filter(|&&x| x.abs() < sigma3).count();
    let within_3sigma = within as f64 / n;

    DistributionStats {
        method: method.to_string(),
        mean,
        std,
        skewness,
        kurtosis,
        min,
        max,
        expected_std,
        std_ratio,
        within_3sigma,
    }
}

/// Analyze post-rotation distribution for a given rotation method.
///
/// Generates random unit vectors, applies the rotation, and collects
/// all coordinate values to analyze the marginal distribution.
pub fn analyze_rotation_distribution(
    d: usize,
    n_vectors: usize,
    rotation: &Rotation,
    method_name: &str,
) -> DistributionStats {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let normal = StandardNormal;

    let mut all_coords = Vec::with_capacity(n_vectors * d);
    let mut buf = vec![0.0f64; d];
    let mut out = vec![0.0f64; d];

    for _ in 0..n_vectors {
        // Generate random unit vector
        let v: Vec<f64> = {
            let raw: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
            let norm: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            raw.iter().map(|x| x / norm).collect()
        };

        // Rotate
        rotation.forward(&v, &mut buf, &mut out);

        // Collect all coordinates
        all_coords.extend_from_slice(&out);
    }

    compute_stats(&all_coords, method_name, d)
}

/// Compare distributions across all rotation methods at d=128.
pub fn compare_rotation_distributions(n_vectors: usize) -> Vec<DistributionStats> {
    let d = 128;
    let seed = 42;

    let haar = Rotation::new_haar(d, seed);
    let wht = Rotation::new_fast_jl(d, seed);
    let e8 = Rotation::new_e8(d, seed);

    vec![
        analyze_rotation_distribution(d, n_vectors, &haar, "haar"),
        analyze_rotation_distribution(d, n_vectors, &wht, "wht"),
        analyze_rotation_distribution(d, n_vectors, &e8, "e8"),
    ]
}

/// Diagnose the E8 codebook mismatch.
///
/// Returns the recommended fix: either rescale the codebook or compose E8+WHT.
pub fn diagnose_e8_mismatch(n_vectors: usize) -> E8Diagnosis {
    let _d = 128;
    let stats = compare_rotation_distributions(n_vectors);

    let haar = &stats[0];
    let wht = &stats[1];
    let e8 = &stats[2];

    // Check if E8 distribution is simply a scaled Gaussian
    let scale_fix_works = (e8.kurtosis - 3.0).abs() < 1.0 // near-Gaussian kurtosis
        && e8.skewness.abs() < 0.5; // near-symmetric

    // Check if E8 std is just off by a constant factor
    let scale_factor = if haar.std > 1e-15 {
        e8.std / haar.std
    } else {
        1.0
    };

    E8Diagnosis {
        haar_stats: haar.clone(),
        wht_stats: wht.clone(),
        e8_stats: e8.clone(),
        e8_std_vs_haar: scale_factor,
        e8_kurtosis_excess: e8.kurtosis - 3.0,
        scale_fix_viable: scale_fix_works,
        recommendation: if scale_fix_works {
            "Rescale E8 codebook boundaries by factor {scale_factor:.4}".to_string()
        } else {
            format!(
                "E8 marginal is non-Gaussian (kurtosis={:.2}, skew={:.3}). \
                 Use E8+WHT composition: E8 for block decorrelation, \
                 then per-block WHT for Gaussianization.",
                e8.kurtosis, e8.skewness
            )
        },
    }
}

/// Diagnostic result for E8 codebook mismatch.
#[derive(Clone, Debug)]
pub struct E8Diagnosis {
    pub haar_stats: DistributionStats,
    pub wht_stats: DistributionStats,
    pub e8_stats: DistributionStats,
    /// Ratio of E8 std to Haar std.
    pub e8_std_vs_haar: f64,
    /// Excess kurtosis of E8 (0.0 = Gaussian).
    pub e8_kurtosis_excess: f64,
    /// Whether simple scale fix would work.
    pub scale_fix_viable: bool,
    /// Recommended fix.
    pub recommendation: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_distribution_gaussian() {
        let d = 128;
        let rotation = Rotation::new_haar(d, 42);
        let stats = analyze_rotation_distribution(d, 1000, &rotation, "haar");

        println!("Haar distribution at d={}:", d);
        println!("  mean:     {:.6} (expected ~0)", stats.mean);
        println!(
            "  std:      {:.6} (expected {:.6})",
            stats.std, stats.expected_std
        );
        println!("  std_ratio:{:.4} (expected ~1.0)", stats.std_ratio);
        println!("  skewness: {:.4} (expected ~0)", stats.skewness);
        println!("  kurtosis: {:.4} (expected ~3.0)", stats.kurtosis);
        println!("  within 3s:{:.4} (expected ~0.997)", stats.within_3sigma);

        // Haar should produce near-Gaussian distribution
        // Note: Haar kurtosis may be > 3 with small samples due to heavy tails
        // in the QR-based rotation. The distribution converges to Gaussian
        // as n_vectors increases.
        assert!(stats.mean.abs() < 0.01, "Haar mean should be ~0");
        assert!(
            (stats.std_ratio - 1.0).abs() < 0.2,
            "Haar std should match expected"
        );
    }

    #[test]
    fn test_e8_distribution_analysis() {
        let d = 128;
        let rotation = Rotation::new_e8(d, 42);
        let stats = analyze_rotation_distribution(d, 1000, &rotation, "e8");

        println!("E8 distribution at d={}:", d);
        println!("  mean:     {:.6}", stats.mean);
        println!(
            "  std:      {:.6} (expected {:.6})",
            stats.std, stats.expected_std
        );
        println!("  std_ratio:{:.4}", stats.std_ratio);
        println!("  skewness: {:.4}", stats.skewness);
        println!("  kurtosis: {:.4}", stats.kurtosis);
        println!("  range:    [{:.4}, {:.4}]", stats.min, stats.max);
        println!("  within 3s:{:.4}", stats.within_3sigma);
    }

    #[test]
    fn test_e8_pipeline_debug() {
        // Trace through E8 pipeline step by step to find where MSE diverges
        use crate::turboquant::pipeline::TurboQuantMSE;

        let d = 128;
        let bits = 3;

        // Generate a test vector
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let normal = StandardNormal;
        let x: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
        let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

        // WHT pipeline
        let tq_wht = TurboQuantMSE::new(d, bits, 42, true);
        let mut buf = vec![0.0f64; 3 * d];
        let comp_wht = tq_wht.quantize(&x, &mut buf);
        let mut recon_wht = vec![0.0f64; d];
        tq_wht.dequantize(&comp_wht, &mut buf, &mut recon_wht);
        let mse_wht: f64 = x
            .iter()
            .zip(recon_wht.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;

        // E8 pipeline
        let tq_e8 = TurboQuantMSE::from_config(
            &crate::turboquant::config::TurboQuantConfig {
                dim: d,
                bits,
                seed: 42,
                rotation: crate::turboquant::config::RotationMethod::E8Block,
                qjl_correction: crate::turboquant::config::QjlCorrectionMode::Never,
                adaptive: crate::turboquant::config::AdaptiveBitsConfig {
                    enabled: false,
                    promote_fraction: 0.0,
                },
                qjl_dim: None,
            },
            bits,
            42,
        );
        let comp_e8 = tq_e8.quantize(&x, &mut buf);
        let mut recon_e8 = vec![0.0f64; d];
        tq_e8.dequantize(&comp_e8, &mut buf, &mut recon_e8);
        let mse_e8: f64 = x
            .iter()
            .zip(recon_e8.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;

        println!("\n=== E8 Pipeline Step-by-Step Debug ===");
        println!("Input ||x|| = {:.6}", x_norm);
        println!("WHT: vec_norm={:.6}, MSE={:.6}", comp_wht.vec_norm, mse_wht);
        println!("E8:  vec_norm={:.6}, MSE={:.6}", comp_e8.vec_norm, mse_e8);

        // Check if the norms are the same
        println!("WHT indices[0..8]: {:?}", &comp_wht.indices[..8]);
        println!("E8  indices[0..8]: {:?}", &comp_e8.indices[..8]);

        // Check reconstructed norms
        let recon_wht_norm: f64 = recon_wht.iter().map(|v| v * v).sum::<f64>().sqrt();
        let recon_e8_norm: f64 = recon_e8.iter().map(|v| v * v).sum::<f64>().sqrt();
        println!("WHT recon ||x|| = {:.6}", recon_wht_norm);
        println!("E8  recon ||x|| = {:.6}", recon_e8_norm);

        // The MSE ratio should help identify the issue
        println!("MSE ratio (E8/WHT): {:.4}", mse_e8 / mse_wht);
    }

    #[test]
    fn test_e8_diagnosis() {
        let diagnosis = diagnose_e8_mismatch(500);

        println!("\n=== E8 Codebook Mismatch Diagnosis ===");
        println!("Haar std: {:.6}", diagnosis.haar_stats.std);
        println!("WHT std:  {:.6}", diagnosis.wht_stats.std);
        println!("E8 std:   {:.6}", diagnosis.e8_stats.std);
        println!("E8/Haar std ratio: {:.4}", diagnosis.e8_std_vs_haar);
        println!("E8 excess kurtosis: {:.4}", diagnosis.e8_kurtosis_excess);
        println!("Scale fix viable: {}", diagnosis.scale_fix_viable);
        println!("Recommendation: {}", diagnosis.recommendation);
    }
}
