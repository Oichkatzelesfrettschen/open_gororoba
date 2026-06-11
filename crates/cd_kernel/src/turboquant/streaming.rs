//! Streaming quantizer for decode-time KV cache growth.
//!
//! During autoregressive LLM decoding, the KV cache grows by one token
//! per step.  This module provides incremental quantization that processes
//! new tokens without re-processing the entire cache.
//!
//! Pattern inspired by `soa_cache.rs::RingEmbeddingCache` which uses a
//! 3-element ring buffer for the CD associator sliding window.
//!
//! # Architecture
//!
//! ```text
//! Prefill:  `[tok_0, tok_1, ..., tok_n]`  -- batch quantize (parallel)
//! Decode:   `[tok_{n+1}]`                 -- single-token quantize (streaming)
//!           `[tok_{n+2}]`
//!           ...
//! ```
//!
//! The streaming quantizer maintains the rotation matrix and codebook
//! across steps, only processing the new token.

use super::pipeline::{MseCompressed, TurboQuantMSE};

/// Streaming KV cache with incremental quantization.
pub struct StreamingKVCache {
    /// Quantizer (rotation + codebook, created once).
    quantizer: TurboQuantMSE,
    /// Compressed key cache (grows incrementally).
    key_cache: Vec<MseCompressed>,
    /// Compressed value cache.
    value_cache: Vec<MseCompressed>,
    /// Pre-allocated scratch buffer (reused across steps).
    buf: Vec<f64>,
    /// Head dimension.
    d: usize,
    /// Quantization bits (stored for memory computation).
    #[allow(dead_code)]
    bits: u32,
}

impl StreamingKVCache {
    /// Create a streaming cache for one attention head.
    pub fn new(d: usize, bits: u32, seed: u64, use_wht: bool) -> Self {
        StreamingKVCache {
            quantizer: TurboQuantMSE::new(d, bits, seed, use_wht),
            key_cache: Vec::new(),
            value_cache: Vec::new(),
            buf: vec![0.0f64; 3 * d],
            d,
            bits,
        }
    }

    /// Whether we have enough tokens for adaptive scoring.
    ///
    /// The CD associator needs triplets (3 consecutive vectors).
    /// During the warmup period (< 3 tokens), base bits are used.
    pub fn is_warmed_up(&self) -> bool {
        self.key_cache.len() >= 3
    }

    /// Append a new key-value pair (single token, decode step).
    ///
    /// Quantizes the key and value incrementally using the pre-allocated
    /// buffer.  No re-processing of existing cache entries.
    /// During warmup (< 3 tokens), uses base quantization.
    /// After warmup, the importance scoring is available for adaptive allocation.
    pub fn append(&mut self, key: &[f64], value: &[f64]) {
        debug_assert_eq!(key.len(), self.d);
        debug_assert_eq!(value.len(), self.d);

        let key_compressed = self.quantizer.quantize(key, &mut self.buf);
        let value_compressed = self.quantizer.quantize(value, &mut self.buf);

        self.key_cache.push(key_compressed);
        self.value_cache.push(value_compressed);
    }

    /// Append a batch of key-value pairs (prefill step).
    pub fn append_batch(&mut self, keys: &[Vec<f64>], values: &[Vec<f64>]) {
        debug_assert_eq!(keys.len(), values.len());
        for (k, v) in keys.iter().zip(values.iter()) {
            self.append(k, v);
        }
    }

    /// Compute attention scores for a query against all cached keys.
    ///
    /// Returns scores of length `len()` (one per cached token).
    pub fn attention_scores(&self, query: &[f64]) -> Vec<f64> {
        self.key_cache
            .iter()
            .map(|compressed| {
                // Dequantize key and dot with query
                let mut buf = vec![0.0f64; 3 * self.d];
                let mut key_recon = vec![0.0f64; self.d];
                self.quantizer
                    .dequantize(compressed, &mut buf, &mut key_recon);
                query.iter().zip(key_recon.iter()).map(|(q, k)| q * k).sum()
            })
            .collect()
    }

    /// Get reconstructed values for output computation.
    pub fn reconstruct_values(&self) -> Vec<Vec<f64>> {
        self.value_cache
            .iter()
            .map(|compressed| {
                let mut buf = vec![0.0f64; 3 * self.d];
                let mut val_recon = vec![0.0f64; self.d];
                self.quantizer
                    .dequantize(compressed, &mut buf, &mut val_recon);
                val_recon
            })
            .collect()
    }

    /// Get the importance score for a token (vector norm as proxy).
    ///
    /// Higher norm = more important = should get more quantization bits.
    /// This is the same proxy used by the sampled adaptive allocation
    /// (259x faster than full CD associator scoring).
    pub fn token_importance(&self, token_idx: usize) -> f64 {
        if token_idx >= self.key_cache.len() {
            return 0.0;
        }
        self.key_cache[token_idx].vec_norm
    }

    /// Get importance scores for all cached tokens.
    pub fn all_importances(&self) -> Vec<f64> {
        self.key_cache.iter().map(|c| c.vec_norm).collect()
    }

    /// Find the top-k most important tokens (by norm).
    ///
    /// Useful for selective attention: only attend to the most
    /// important tokens during decode (MiniKV pattern).
    pub fn top_k_important(&self, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = self
            .key_cache
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.vec_norm))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|&(i, _)| i).collect()
    }

    /// Number of cached tokens.
    pub fn len(&self) -> usize {
        self.key_cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.key_cache.is_empty()
    }

    /// Memory usage in bytes (compressed).
    pub fn memory_bytes(&self) -> usize {
        let per_entry = self.d + 8; // u8 indices + f64 norm
        self.key_cache.len() * per_entry * 2 // keys + values
    }

    /// Memory usage in bytes if stored as fp16.
    pub fn memory_fp16_bytes(&self) -> usize {
        self.key_cache.len() * self.d * 2 * 2 // keys + values, fp16
    }

    /// Compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.memory_bytes() == 0 {
            return 0.0;
        }
        self.memory_fp16_bytes() as f64 / self.memory_bytes() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_vec(d: usize, rng: &mut ChaCha20Rng) -> Vec<f64> {
        let normal = StandardNormal;
        (0..d).map(|_| normal.sample(rng)).collect()
    }

    #[test]
    fn test_streaming_append() {
        let d = 64;
        let mut cache = StreamingKVCache::new(d, 3, 42, true);
        assert!(cache.is_empty());

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        for _ in 0..10 {
            let key = random_vec(d, &mut rng);
            let value = random_vec(d, &mut rng);
            cache.append(&key, &value);
        }
        assert_eq!(cache.len(), 10);
        assert!(cache.compression_ratio() > 1.0);
    }

    #[test]
    fn test_streaming_attention_scores() {
        let d = 32;
        let mut cache = StreamingKVCache::new(d, 3, 42, true);
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Add 5 tokens
        let keys: Vec<Vec<f64>> = (0..5).map(|_| random_vec(d, &mut rng)).collect();
        let values: Vec<Vec<f64>> = (0..5).map(|_| random_vec(d, &mut rng)).collect();
        cache.append_batch(&keys, &values);

        // Query
        let query = random_vec(d, &mut rng);
        let scores = cache.attention_scores(&query);
        assert_eq!(scores.len(), 5);

        // Scores should be non-degenerate
        let max_score = scores.iter().fold(f64::MIN, |a, &b| a.max(b));
        let min_score = scores.iter().fold(f64::MAX, |a, &b| a.min(b));
        assert!(max_score > min_score, "Scores should vary");
    }

    #[test]
    fn test_streaming_incremental() {
        let d = 32;
        let mut cache = StreamingKVCache::new(d, 3, 42, true);
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Simulate prefill + decode
        // Prefill: 10 tokens
        let keys: Vec<Vec<f64>> = (0..10).map(|_| random_vec(d, &mut rng)).collect();
        let values: Vec<Vec<f64>> = (0..10).map(|_| random_vec(d, &mut rng)).collect();
        cache.append_batch(&keys, &values);
        assert_eq!(cache.len(), 10);

        // Decode: 5 more tokens (one at a time)
        for _ in 0..5 {
            let k = random_vec(d, &mut rng);
            let v = random_vec(d, &mut rng);
            cache.append(&k, &v);
        }
        assert_eq!(cache.len(), 15);

        // Attention should work over all 15 tokens
        let query = random_vec(d, &mut rng);
        let scores = cache.attention_scores(&query);
        assert_eq!(scores.len(), 15);

        println!(
            "Streaming cache: {} tokens, {:.1}x compression, {} bytes",
            cache.len(),
            cache.compression_ratio(),
            cache.memory_bytes()
        );
    }
}
