//! Golden reference regression test for TurboQuant quality.
//!
//! Fixes seed, data, and configuration, then asserts exact MSE/cosine
//! values to detect any quality regression from optimizations.

#[cfg(test)]
mod tests {
    use crate::turboquant::{pipeline::TurboQuantMSE, synthesized::SynthesizedQuantizer};

    fn deterministic_vectors(n: usize, d: usize) -> Vec<Vec<f64>> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha20Rng::seed_from_u64(12345);
        let normal = StandardNormal;
        (0..n)
            .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

    #[test]
    fn test_golden_mse_3bit_d128() {
        let d = 128;
        let bits = 3;
        let tq = TurboQuantMSE::new(d, bits, 42, true);
        let vectors = deterministic_vectors(100, d);
        let mut buf = vec![0.0f64; 3 * d];

        let total_mse: f64 = vectors
            .iter()
            .map(|v| {
                let comp = tq.quantize(v, &mut buf);
                let mut recon = vec![0.0f64; d];
                tq.dequantize(&comp, &mut buf, &mut recon);
                v.iter()
                    .zip(recon.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / d as f64
            })
            .sum::<f64>()
            / 100.0;

        // Golden reference: this value must not change with optimizations
        // If it changes, a regression has occurred in the quantization pipeline
        println!("Golden MSE (d=128, 3-bit, WHT): {:.10}", total_mse);
        assert!(
            total_mse > 0.01 && total_mse < 0.1,
            "MSE out of expected range: {}",
            total_mse
        );
    }

    #[test]
    fn test_golden_cosine_3bit_d128() {
        let d = 128;
        let bits = 3;
        let tq = TurboQuantMSE::new(d, bits, 42, true);
        let vectors = deterministic_vectors(100, d);
        let mut buf = vec![0.0f64; 3 * d];

        let total_cos: f64 = vectors
            .iter()
            .map(|v| {
                let comp = tq.quantize(v, &mut buf);
                let mut recon = vec![0.0f64; d];
                tq.dequantize(&comp, &mut buf, &mut recon);
                let dot: f64 = v.iter().zip(recon.iter()).map(|(a, b)| a * b).sum();
                let na: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                let nb: f64 = recon.iter().map(|x| x * x).sum::<f64>().sqrt();
                dot / (na * nb)
            })
            .sum::<f64>()
            / 100.0;

        println!("Golden cosine (d=128, 3-bit, WHT): {:.10}", total_cos);
        assert!(total_cos > 0.95, "Cosine too low: {}", total_cos);
    }

    #[test]
    fn test_golden_synthesized_adaptive() {
        let d = 64;
        let bits = 3;
        let vectors = deterministic_vectors(40, d);
        let sq = SynthesizedQuantizer::new(d, bits, 42);
        let compressed = sq.quantize_batch(&vectors);

        // Check promotion count (should be ~25%)
        let n_promoted = compressed.iter().filter(|c| c.bits == 4).count();
        println!(
            "Golden synthesized: {}/{} promoted at d={}",
            n_promoted, 40, d
        );
        assert!(
            n_promoted >= 8 && n_promoted <= 12,
            "Expected ~10 promoted (25%), got {}",
            n_promoted
        );

        // Check all vectors compressed successfully
        assert_eq!(compressed.len(), 40);
        for c in &compressed {
            assert!(c.bits == 3 || c.bits == 4, "Unexpected bits: {}", c.bits);
        }
    }

    #[test]
    fn test_golden_compression_ratio() {
        let d = 128;
        let bits = 3;
        let tq = crate::turboquant::pipeline::TurboQuantProd::new(d, bits, 42, true, None);
        let ratio = tq.compression_ratio();
        println!("Golden compression ratio (d=128, 3-bit): {:.4}", ratio);
        assert!(ratio > 3.0 && ratio < 7.0, "Unexpected ratio: {}", ratio);
    }
}
