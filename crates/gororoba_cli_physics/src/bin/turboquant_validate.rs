//! TurboQuant quality validation against uncompressed attention scores.
//!
//! Generates synthetic attention data (Q, K, V), compresses KV cache,
//! and measures: cosine similarity, top-1 match, top-5 match, and
//! compression ratio.  Matches the metrics from turboquant-pytorch/validate.py.
//!
//! Usage:
//!   turboquant-validate --dim 128 --bits 3 --seq-lens 512,2048
//!   turboquant-validate --bits 2,3,4 --rotation wht

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf};

use cd_kernel::turboquant::compressor::KeyCompressor;

#[derive(Parser)]
#[command(name = "turboquant-validate")]
#[command(about = "TurboQuant quality validation: cosine, top-k, compression ratio")]
struct Cli {
    /// Head dimension.
    #[arg(long, default_value_t = 128)]
    dim: usize,

    /// Comma-separated bit-widths to test.
    #[arg(long, value_delimiter = ',', default_values_t = vec![2, 3, 4])]
    bits: Vec<u32>,

    /// Comma-separated sequence lengths to test.
    #[arg(long, value_delimiter = ',', default_values_t = vec![512, 2048])]
    seq_lens: Vec<usize>,

    /// Number of attention heads.
    #[arg(long, default_value_t = 4)]
    n_heads: usize,

    /// Use WHT rotation instead of Haar.
    #[arg(long, default_value_t = false)]
    wht: bool,

    /// Output JSON path.
    #[arg(long, default_value = "data/output/heliosphere/ablations/turboquant_validate.json")]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct ValidationResult {
    dim: usize,
    bits: u32,
    seq_len: usize,
    n_heads: usize,
    rotation: String,
    // Quality metrics
    cosine_similarity_mean: f64,
    cosine_similarity_min: f64,
    top1_match_rate: f64,
    top5_match_rate: f64,
    // Memory
    key_bytes_compressed: usize,
    value_bytes_compressed: usize,
    total_bytes_compressed: usize,
    total_bytes_fp16: usize,
    compression_ratio: f64,
}

#[derive(Debug, Serialize)]
struct ValidateOutput {
    configs_tested: usize,
    results: Vec<ValidationResult>,
}

fn generate_kv_data(
    n_heads: usize,
    seq_len: usize,
    d: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    let n = n_heads * seq_len * d;

    let keys: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let values: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    // Single query per head
    let q_n = n_heads * d;
    let queries: Vec<f64> = (0..q_n).map(|_| normal.sample(&mut rng)).collect();

    (keys, values, queries)
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 { 0.0 } else { dot / (na * nb) }
}

fn argmax(scores: &[f64]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn top_k_indices(scores: &[f64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

fn validate_config(
    d: usize,
    bits: u32,
    seq_len: usize,
    n_heads: usize,
    use_wht: bool,
) -> ValidationResult {
    let (keys, _values, queries) = generate_kv_data(n_heads, seq_len, d, 42);

    let key_comp = KeyCompressor::new(d, bits, 42, use_wht);
    let shape = (1, n_heads, seq_len, d);
    let compressed_keys = key_comp.compress(&keys, shape);

    // For each head, compute real and compressed attention scores
    let mut cosines = Vec::new();
    let mut top1_matches = 0usize;
    let mut top5_matches = 0usize;
    let total_heads = n_heads;

    for h in 0..n_heads {
        let q = &queries[h * d..(h + 1) * d];

        // Real attention scores: q @ K[h]^T
        let real_scores: Vec<f64> = (0..seq_len)
            .map(|s| {
                let k_offset = (h * seq_len + s) * d;
                let k = &keys[k_offset..k_offset + d];
                q.iter().zip(k.iter()).map(|(a, b)| a * b).sum()
            })
            .collect();

        // Compressed scores via KeyCompressor (full QJL correction)
        // Build a sub-batch for this head's keys
        let head_keys = compressed_keys.keys[h * seq_len..(h + 1) * seq_len].to_vec();
        let head_batch = cd_kernel::turboquant::compressor::CompressedKeyBatch {
            keys: head_keys,
            shape: (1, 1, seq_len, d),
        };
        let comp_scores = key_comp.attention_scores(q, 1, &head_batch);

        // Cosine similarity between real and compressed score vectors
        let cos = cosine_sim(&real_scores, &comp_scores);
        cosines.push(cos);

        // Top-1 match
        if argmax(&real_scores) == argmax(&comp_scores) {
            top1_matches += 1;
        }

        // Top-5 match: does real argmax appear in compressed top-5?
        let real_top1 = argmax(&real_scores);
        let comp_top5 = top_k_indices(&comp_scores, 5);
        if comp_top5.contains(&real_top1) {
            top5_matches += 1;
        }
    }

    let cos_mean = cosines.iter().sum::<f64>() / cosines.len().max(1) as f64;
    let cos_min = cosines.iter().copied().fold(f64::MAX, f64::min);

    // Memory computation
    let n_vecs = n_heads * seq_len;
    // Keys: f32 MSE (32*d bits) + signs (d bits) + norm (16 bits) per vector
    let key_bytes = n_vecs * (d * 4 + d / 8 + 2);
    // Values: u8 indices (8*d bits) + norm (16 bits) per vector
    let value_bytes = n_vecs * (d + 2);
    let total_compressed = key_bytes + value_bytes;
    let total_fp16 = 2 * n_vecs * d * 2; // keys + values, fp16

    ValidationResult {
        dim: d,
        bits,
        seq_len,
        n_heads,
        rotation: if use_wht { "wht".into() } else { "haar".into() },
        cosine_similarity_mean: cos_mean,
        cosine_similarity_min: cos_min,
        top1_match_rate: top1_matches as f64 / total_heads as f64 * 100.0,
        top5_match_rate: top5_matches as f64 / total_heads as f64 * 100.0,
        key_bytes_compressed: key_bytes,
        value_bytes_compressed: value_bytes,
        total_bytes_compressed: total_compressed,
        total_bytes_fp16: total_fp16,
        compression_ratio: total_fp16 as f64 / total_compressed as f64,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("=== TurboQuant Quality Validation ===");
    println!(
        "  dim={}, heads={}, rotation={}",
        cli.dim,
        cli.n_heads,
        if cli.wht { "wht" } else { "haar" }
    );

    let mut results = Vec::new();

    for &seq_len in &cli.seq_lens {
        for &bits in &cli.bits {
            print!(
                "  seq={:<5} bits={} ... ",
                seq_len, bits
            );

            let result = validate_config(cli.dim, bits, seq_len, cli.n_heads, cli.wht);

            println!(
                "cos={:.4}  top1={:.0}%  top5={:.0}%  ratio={:.1}x",
                result.cosine_similarity_mean,
                result.top1_match_rate,
                result.top5_match_rate,
                result.compression_ratio
            );

            results.push(result);
        }
    }

    let output = ValidateOutput {
        configs_tested: results.len(),
        results,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("\n  Wrote {}", cli.out_json.display());

    Ok(())
}
