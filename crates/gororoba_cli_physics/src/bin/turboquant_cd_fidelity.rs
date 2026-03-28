//! TurboQuant CD fidelity analyzer: measure phase-geometry distortion from
//! KV cache quantization using the 32D Cayley-Dickson associator.
//!
//! Reads pre-quantization and post-quantization attention key vectors (f64 binary),
//! embeds both in 32D pathion space via Takens delay, computes the CD associator
//! on each, and reports the fidelity ratio A_post/A_pre.

use anyhow::{bail, Result};
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "turboquant-cd-fidelity")]
struct Cli {
    /// Pre-quantization keys binary (f64, n_vectors * d elements).
    #[arg(long)]
    pre: PathBuf,

    /// Post-quantization keys binary (f64, n_vectors * d elements).
    #[arg(long)]
    post: PathBuf,

    /// Number of vectors.
    #[arg(long)]
    n_vectors: usize,

    /// Vector dimension.
    #[arg(long, default_value_t = 128)]
    dim: usize,

    /// Embedding dimension for CD associator.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/turboquant_cd_fidelity.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct FidelityResult {
    window_idx: usize,
    a_pre: f64,
    a_post: f64,
    fidelity_ratio: f64,
    cosine_sim_mean: f64,
}

#[derive(Debug, Serialize)]
struct TqCdOutput {
    n_vectors: usize,
    dim: usize,
    embedding_dim: usize,
    n_windows: usize,
    overall_a_pre: f64,
    overall_a_post: f64,
    overall_fidelity: f64,
    cosine_sim_mean: f64,
    results: Vec<FidelityResult>,
    interpretation: String,
}

fn read_f64_binary(path: &PathBuf, expected_count: usize) -> Result<Vec<f64>> {
    let bytes = fs::read(path)?;
    if bytes.len() != expected_count * 8 {
        bail!(
            "Expected {} f64 values ({} bytes), got {} bytes",
            expected_count,
            expected_count * 8,
            bytes.len()
        );
    }
    let vals: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();
    Ok(vals)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== TurboQuant CD Fidelity Analyzer ===");

    let total = cli.n_vectors * cli.dim;
    let pre = read_f64_binary(&cli.pre, total)?;
    let post = read_f64_binary(&cli.post, total)?;

    println!(
        "  Loaded {} vectors, d={}, embedding={}D",
        cli.n_vectors, cli.dim, cli.embedding_dim
    );

    // Compute cosine similarity for comparison
    let mut cos_sims = Vec::with_capacity(cli.n_vectors);
    for i in 0..cli.n_vectors {
        let base = i * cli.dim;
        let mut dot = 0.0f64;
        let mut norm_pre = 0.0f64;
        let mut norm_post = 0.0f64;
        for j in 0..cli.dim {
            let p = pre[base + j];
            let q = post[base + j];
            dot += p * q;
            norm_pre += p * p;
            norm_post += q * q;
        }
        let denom = (norm_pre * norm_post).sqrt();
        cos_sims.push(if denom > 1e-15 { dot / denom } else { 0.0 });
    }
    let cos_mean = cos_sims.iter().sum::<f64>() / cos_sims.len() as f64;
    println!("  Cosine similarity: mean={:.6}", cos_mean);

    // Build Takens embeddings for pre and post
    // Use the d-dimensional vectors as channels for the embedding
    // Embedding: take groups of vectors as windows, interleave their components
    let channels = cli.dim.min(cli.embedding_dim);
    let steps = cli.embedding_dim / channels;

    let window_size = steps;
    let n_windows = if cli.n_vectors > window_size + 2 {
        cli.n_vectors - window_size
    } else {
        bail!("Not enough vectors for windowed embedding");
    };

    // Process in blocks of ~100 windows
    let block_size = 100.min(n_windows);
    let n_blocks = (n_windows + block_size - 1) / block_size;

    let mut results = Vec::new();
    let mut total_a_pre = 0.0;
    let mut total_a_post = 0.0;
    let mut count = 0usize;

    for block in 0..n_blocks {
        let block_start = block * block_size;
        let block_end = (block_start + block_size).min(n_windows);

        // Build embeddings for this block
        let mut embedded_pre: Vec<Vec<f64>> = Vec::new();
        let mut embedded_post: Vec<Vec<f64>> = Vec::new();

        for w in block_start..block_end {
            let mut v_pre = Vec::with_capacity(cli.embedding_dim);
            let mut v_post = Vec::with_capacity(cli.embedding_dim);

            for s in 0..steps {
                let vec_idx = w + s;
                if vec_idx >= cli.n_vectors {
                    break;
                }
                let base = vec_idx * cli.dim;
                for c in 0..channels {
                    if v_pre.len() >= cli.embedding_dim {
                        break;
                    }
                    v_pre.push(pre[base + c]);
                    v_post.push(post[base + c]);
                }
            }

            // Pad if needed
            while v_pre.len() < cli.embedding_dim {
                v_pre.push(0.0);
                v_post.push(0.0);
            }
            v_pre.truncate(cli.embedding_dim);
            v_post.truncate(cli.embedding_dim);

            // Direction normalize
            let norm_pre: f64 = v_pre.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_post: f64 = v_post.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_pre > 1e-15 {
                for x in v_pre.iter_mut() {
                    *x /= norm_pre;
                }
            }
            if norm_post > 1e-15 {
                for x in v_post.iter_mut() {
                    *x /= norm_post;
                }
            }

            embedded_pre.push(v_pre);
            embedded_post.push(v_post);
        }

        if embedded_pre.len() < 3 {
            continue;
        }

        let norms_pre = cd_kernel::batch_sliding_associator_norms_parallel(
            &embedded_pre,
            cli.embedding_dim,
        );
        let norms_post = cd_kernel::batch_sliding_associator_norms_parallel(
            &embedded_post,
            cli.embedding_dim,
        );

        let a_pre = if norms_pre.is_empty() {
            0.0
        } else {
            norms_pre.iter().sum::<f64>() / norms_pre.len() as f64
        };
        let a_post = if norms_post.is_empty() {
            0.0
        } else {
            norms_post.iter().sum::<f64>() / norms_post.len() as f64
        };

        let fidelity = if a_pre > 1e-15 {
            a_post / a_pre
        } else {
            1.0
        };

        let block_cos = cos_sims[block_start..block_end.min(cos_sims.len())]
            .iter()
            .sum::<f64>()
            / (block_end - block_start) as f64;

        if block % 3 == 0 || block == n_blocks - 1 {
            println!(
                "  block {}/{}: A_pre={:.6}, A_post={:.6}, fidelity={:.4}, cos={:.4}",
                block + 1,
                n_blocks,
                a_pre,
                a_post,
                fidelity,
                block_cos
            );
        }

        results.push(FidelityResult {
            window_idx: block,
            a_pre,
            a_post,
            fidelity_ratio: fidelity,
            cosine_sim_mean: block_cos,
        });

        total_a_pre += a_pre;
        total_a_post += a_post;
        count += 1;
    }

    let overall_a_pre = if count > 0 { total_a_pre / count as f64 } else { 0.0 };
    let overall_a_post = if count > 0 { total_a_post / count as f64 } else { 0.0 };
    let overall_fidelity = if overall_a_pre > 1e-15 {
        overall_a_post / overall_a_pre
    } else {
        1.0
    };

    let interp = format!(
        "TurboQuant {}D -> {}D CD fidelity: A_pre={:.6}, A_post={:.6}, ratio={:.4}. Cosine sim={:.4}. {} blocks analyzed.",
        cli.dim, cli.embedding_dim, overall_a_pre, overall_a_post, overall_fidelity, cos_mean, count
    );
    println!("\n  {}", interp);

    let output = TqCdOutput {
        n_vectors: cli.n_vectors,
        dim: cli.dim,
        embedding_dim: cli.embedding_dim,
        n_windows,
        overall_a_pre,
        overall_a_post,
        overall_fidelity,
        cosine_sim_mean: cos_mean,
        results,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}
