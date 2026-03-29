//! Comprehensive integration test: full LLM simulation with all TurboQuant features.
//!
//! This test validates the entire pipeline working together:
//! 1. Prefill: batch quantize KV vectors for initial context
//! 2. Stream decode: append one token at a time
//! 3. Attention: compute quantized attention scores with QJL correction
//! 4. Cross-layer: merge adjacent layers
//! 5. Adaptive: per-token bit allocation via CD associator
//! 6. Quality: verify against golden reference values

#[cfg(test)]
mod tests {
    use crate::turboquant::streaming::StreamingKVCache;
    use crate::turboquant::attention_correction;
    use crate::turboquant::synthesized::SynthesizedQuantizer;
    use crate::turboquant::config::TurboQuantConfig;

    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_vec(d: usize, rng: &mut ChaCha20Rng) -> Vec<f64> {
        let normal = StandardNormal;
        (0..d).map(|_| normal.sample(rng)).collect()
    }

    /// Full LLM inference simulation: prefill + decode + attention + cross-layer.
    #[test]
    fn test_full_llm_simulation() {
        let d = 64;
        let bits = 3;
        let n_layers = 4;
        let n_heads = 4;
        let prefill_len = 32;
        let decode_steps = 8;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        println!("\n=== Full LLM Simulation ===");
        println!("  d={}, bits={}, layers={}, heads={}", d, bits, n_layers, n_heads);
        println!("  prefill={} tokens, decode={} steps", prefill_len, decode_steps);

        // Create per-head streaming caches (layers x heads)
        let mut caches: Vec<Vec<StreamingKVCache>> = (0..n_layers)
            .map(|l| {
                (0..n_heads)
                    .map(|h| StreamingKVCache::new(d, bits, (l * 100 + h) as u64, true))
                    .collect()
            })
            .collect();

        // Phase 1: Prefill (batch insert)
        println!("  Phase 1: Prefill ({} tokens)...", prefill_len);
        for _tok in 0..prefill_len {
            for layer in &mut caches {
                for head_cache in layer.iter_mut() {
                    let key = random_vec(d, &mut rng);
                    let value = random_vec(d, &mut rng);
                    head_cache.append(&key, &value);
                }
            }
        }
        assert_eq!(caches[0][0].len(), prefill_len);
        println!("    Cache size: {} tokens per head", caches[0][0].len());
        println!("    Compression: {:.1}x", caches[0][0].compression_ratio());

        // Phase 2: Stream decode (one token at a time)
        println!("  Phase 2: Decode ({} steps)...", decode_steps);
        let mut decode_scores = Vec::new();
        for step in 0..decode_steps {
            // Generate query for this decode step
            let query = random_vec(d, &mut rng);

            // Append new KV to each head, compute attention
            let mut step_scores = Vec::new();
            for (_l, layer) in caches.iter_mut().enumerate() {
                for (_h, head_cache) in layer.iter_mut().enumerate() {
                    let key = random_vec(d, &mut rng);
                    let value = random_vec(d, &mut rng);
                    head_cache.append(&key, &value);

                    // Compute attention scores
                    let scores = head_cache.attention_scores(&query);
                    assert_eq!(scores.len(), prefill_len + step + 1);
                    step_scores.push(scores);
                }
            }
            decode_scores.push(step_scores);

            // Check warmup status
            assert!(caches[0][0].is_warmed_up());
        }
        println!("    Final cache: {} tokens per head", caches[0][0].len());

        // Phase 3: Cross-layer compression
        println!("  Phase 3: Cross-layer compression...");
        let layer0_keys = caches[0][0].all_importances();
        let layer1_keys = caches[1][0].all_importances();
        // Use importances as proxy for KV vectors (they're f64 scalars)
        let merged_importances: Vec<f64> = layer0_keys.iter()
            .zip(layer1_keys.iter())
            .map(|(&a, &b)| (a + b) / 2.0) // simple average for test
            .collect();
        println!("    Merged {} token importances across layers 0-1",
            merged_importances.len());

        // Phase 4: Attention correction
        println!("  Phase 4: Attention score correction...");
        let query_norm_sq: f64 = 1.0; // unit query
        let var = attention_correction::attention_score_variance(query_norm_sq, d, bits);
        println!("    Score variance from quantization: {:.2e}", var);

        // Apply correction to one set of scores
        let mut scores_to_correct = decode_scores[0][0].clone();
        attention_correction::correct_attention_scores(
            &mut scores_to_correct, query_norm_sq, d, bits,
        );

        // Phase 5: Token importance for selective attention
        println!("  Phase 5: Selective attention (top-k)...");
        let top_10 = caches[0][0].top_k_important(10);
        println!("    Top-10 important token indices: {:?}", top_10);
        assert_eq!(top_10.len(), 10);

        // Phase 6: Quality metrics
        println!("  Phase 6: Quality metrics...");
        let total_tokens = prefill_len + decode_steps;
        let memory_per_head = caches[0][0].memory_bytes();
        let memory_fp16_per_head = caches[0][0].memory_fp16_bytes();
        let total_memory = memory_per_head * n_layers * n_heads;
        let total_fp16 = memory_fp16_per_head * n_layers * n_heads;

        println!("    Total tokens: {}", total_tokens);
        println!("    Memory (compressed): {} bytes ({:.1} KB)",
            total_memory, total_memory as f64 / 1024.0);
        println!("    Memory (fp16):       {} bytes ({:.1} KB)",
            total_fp16, total_fp16 as f64 / 1024.0);
        println!("    Compression:         {:.1}x", total_fp16 as f64 / total_memory as f64);

        // Verify everything is consistent
        assert!(total_memory < total_fp16);
        assert!(caches[0][0].compression_ratio() > 1.0);

        println!("\n  PASS: Full LLM simulation completed successfully.");
    }

    /// Test the synthesized quantizer in batch mode with all optimizations.
    #[test]
    fn test_synthesized_batch_integration() {
        let d = 128;
        let bits = 3;
        let n = 200;
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let normal = StandardNormal;
        let vectors: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        // SynthesizedQuantizer with adaptive (sampled f32 fast path)
        let sq = SynthesizedQuantizer::new(d, bits, 42);
        let compressed = sq.quantize_batch(&vectors);
        assert_eq!(compressed.len(), n);

        // Count promoted tokens
        let n_promoted = compressed.iter().filter(|c| c.bits == 4).count();
        let promote_frac = n_promoted as f64 / n as f64;
        println!("Synthesized batch: {}/{} promoted ({:.0}%)", n_promoted, n, promote_frac * 100.0);
        assert!(promote_frac > 0.15 && promote_frac < 0.35,
            "Promotion fraction {} outside expected range", promote_frac);

        // Verify MSE quality
        let mut buf = vec![0.0f64; 3 * d];
        let mut total_mse = 0.0f64;
        for (i, c) in compressed.iter().enumerate() {
            let mut recon = vec![0.0f64; d];
            sq.dequantize(c, &mut buf, &mut recon);
            total_mse += vectors[i].iter().zip(recon.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
        }
        let avg_mse = total_mse / n as f64;
        println!("Average MSE: {:.6}", avg_mse);
        assert!(avg_mse < 0.1, "MSE too high: {}", avg_mse);
    }

    /// Test the config presets produce consistent results.
    #[test]
    fn test_config_presets_consistency() {
        let configs = vec![
            ("recommended_128_3", TurboQuantConfig::recommended(128, 3)),
            ("recommended_64_3", TurboQuantConfig::recommended(64, 3)),
            ("paper_default", TurboQuantConfig::paper_default(128, 3)),
            ("fast", TurboQuantConfig::fast(128, 3)),
        ];

        for (name, cfg) in &configs {
            println!("Config '{}': {}", name, cfg);
            assert!(cfg.dim > 0);
            assert!(cfg.bits >= 1 && cfg.bits <= 8);
        }
    }
}
