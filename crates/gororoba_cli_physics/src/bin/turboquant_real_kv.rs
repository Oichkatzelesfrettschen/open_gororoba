//! TurboQuant evaluation on REAL LLM KV cache tensors.
//!
//! Reads binary f64 tensor files extracted from actual model forward passes
//! (via scripts/extract_real_kv_cache.py) and measures quantization quality
//! on real data -- not synthetic Gaussian vectors.
//!
//! Usage:
//!   python scripts/extract_real_kv_cache.py --model SmolLM2-135M
//!   cargo run --release -p gororoba_cli_physics --bin turboquant-real-kv

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf, time::Instant};

use cd_kernel::{
    lloyd_max::{self, DistributionFamily},
    turboquant::{
        cdaq::CdaqQuantizer, pipeline::TurboQuantMSE, simsimd_bridge::cosine_similarity_f64,
        streaming::StreamingKVCache,
    },
};

#[derive(Parser)]
#[command(name = "turboquant-real-kv")]
#[command(about = "Evaluate TurboQuant on real LLM KV cache tensors")]
struct Cli {
    /// Directory with extracted KV cache tensors.
    #[arg(long, default_value = "data/external/real_kv_cache")]
    data_dir: PathBuf,

    /// Comma-separated bit-widths to test.
    #[arg(long, value_delimiter = ',', default_values_t = vec![2, 3, 4])]
    bits: Vec<u32>,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/real_kv_quality.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Deserialize)]
struct KvMetadata {
    model: String,
    n_layers: usize,
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    n_params: u64,
}

#[derive(Debug, Serialize)]
struct LayerResult {
    layer: usize,
    bits: u32,
    key_mse: f64,
    key_cosine: f64,
    value_mse: f64,
    value_cosine: f64,
    n_vectors: usize,
}

#[derive(Debug, Serialize)]
struct AggregateResult {
    bits: u32,
    mean_key_mse: f64,
    mean_key_cosine: f64,
    mean_value_mse: f64,
    mean_value_cosine: f64,
    compression_ratio_vs_fp16: f64,
    streaming_compression_ratio: f64,
    total_vectors: usize,
    quantize_throughput_kvec_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct RealKvOutput {
    model: String,
    n_params: u64,
    n_layers: usize,
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    total_kv_vectors: usize,
    aggregates: Vec<AggregateResult>,
    per_layer: Vec<LayerResult>,
}

fn read_f64_vectors(path: &std::path::Path, d: usize) -> Result<Vec<Vec<f64>>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let n_f64 = bytes.len() / 8;
    let n_vectors = n_f64 / d;
    assert_eq!(n_f64 % d, 0, "file size not divisible by dim");

    let floats: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Ok(floats
        .chunks_exact(d)
        .map(|c| c.to_vec())
        .collect::<Vec<_>>()
        .into_iter()
        .take(n_vectors)
        .collect())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load metadata
    let meta_path = cli.data_dir.join("metadata.json");
    let meta_str = fs::read_to_string(&meta_path)
        .with_context(|| format!("reading {}", meta_path.display()))?;
    let meta: KvMetadata = serde_json::from_str(&meta_str)?;

    println!("=== TurboQuant Real KV Cache Evaluation ===");
    println!(
        "Model: {} ({:.1}M params)",
        meta.model,
        meta.n_params as f64 / 1e6
    );
    println!(
        "Architecture: {} layers, {} KV heads, head_dim={}",
        meta.n_layers, meta.n_heads, meta.head_dim
    );
    println!("Sequence: {} tokens", meta.seq_len);
    println!(
        "Total KV vectors: {}",
        meta.n_layers * meta.n_heads * meta.seq_len * 2
    );
    println!();

    // Analyze distribution of real KV cache values
    let key_path = cli.data_dir.join("keys_layer00.f64");
    let sample_keys = read_f64_vectors(&key_path, meta.head_dim)?;
    let all_vals: Vec<f64> = sample_keys.iter().flat_map(|v| v.iter().copied()).collect();
    let mean = all_vals.iter().sum::<f64>() / all_vals.len() as f64;
    let var = all_vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all_vals.len() as f64;
    let std = var.sqrt();
    let kurt = all_vals
        .iter()
        .map(|x| ((x - mean) / std).powi(4))
        .sum::<f64>()
        / all_vals.len() as f64;
    let min = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("Real KV Cache Distribution (layer 0 keys):");
    println!("  mean={:.6}, std={:.6}, kurtosis={:.2}", mean, std, kurt);
    println!("  min={:.4}, max={:.4}, range={:.4}", min, max, max - min);
    println!(
        "  Gaussian kurtosis=3.0; real={:.2} ({})",
        kurt,
        if (kurt - 3.0).abs() < 1.0 {
            "near-Gaussian"
        } else {
            "NON-Gaussian"
        }
    );
    println!();

    let mut all_layer_results = Vec::new();
    let mut all_aggregates = Vec::new();

    for &bits in &cli.bits {
        println!("--- {} bits ---", bits);
        let mut total_key_mse = 0.0;
        let mut total_key_cos = 0.0;
        let mut total_val_mse = 0.0;
        let mut total_val_cos = 0.0;
        let mut total_vectors = 0usize;
        let t_start = Instant::now();

        // Also test streaming cache on one layer
        let mut streaming_cache = StreamingKVCache::new(meta.head_dim, bits, 42, true);

        for layer_idx in 0..meta.n_layers {
            let key_path = cli.data_dir.join(format!("keys_layer{:02}.f64", layer_idx));
            let val_path = cli
                .data_dir
                .join(format!("values_layer{:02}.f64", layer_idx));

            let keys = read_f64_vectors(&key_path, meta.head_dim)?;
            let values = read_f64_vectors(&val_path, meta.head_dim)?;
            let n = keys.len();

            // Pipeline quantize + dequantize
            let tq = TurboQuantMSE::new(meta.head_dim, bits, 42, true);
            let mut buf = vec![0.0f64; 3 * meta.head_dim];

            let mut key_mse_sum = 0.0;
            let mut key_cos_sum = 0.0;
            let mut val_mse_sum = 0.0;
            let mut val_cos_sum = 0.0;

            for (k, v) in keys.iter().zip(values.iter()) {
                // Key
                let compressed_k = tq.quantize(k, &mut buf);
                let mut recon_k = vec![0.0f64; meta.head_dim];
                tq.dequantize(&compressed_k, &mut buf, &mut recon_k);
                let mse_k: f64 = k
                    .iter()
                    .zip(recon_k.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / meta.head_dim as f64;
                key_mse_sum += mse_k;
                key_cos_sum += cosine_similarity_f64(k, &recon_k);

                // Value
                let compressed_v = tq.quantize(v, &mut buf);
                let mut recon_v = vec![0.0f64; meta.head_dim];
                tq.dequantize(&compressed_v, &mut buf, &mut recon_v);
                let mse_v: f64 = v
                    .iter()
                    .zip(recon_v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / meta.head_dim as f64;
                val_mse_sum += mse_v;
                val_cos_sum += cosine_similarity_f64(v, &recon_v);

                // Feed to streaming cache (layer 0 only)
                if layer_idx == 0 {
                    streaming_cache.append(k, v);
                }
            }

            let key_mse = key_mse_sum / n as f64;
            let key_cos = key_cos_sum / n as f64;
            let val_mse = val_mse_sum / n as f64;
            let val_cos = val_cos_sum / n as f64;

            total_key_mse += key_mse;
            total_key_cos += key_cos;
            total_val_mse += val_mse;
            total_val_cos += val_cos;
            total_vectors += n * 2; // keys + values

            all_layer_results.push(LayerResult {
                layer: layer_idx,
                bits,
                key_mse,
                key_cosine: key_cos,
                value_mse: val_mse,
                value_cosine: val_cos,
                n_vectors: n,
            });

            if layer_idx < 3 || layer_idx == meta.n_layers - 1 {
                println!(
                    "  Layer {:2}: key MSE={:.6} cos={:.4} | val MSE={:.6} cos={:.4}",
                    layer_idx, key_mse, key_cos, val_mse, val_cos
                );
            } else if layer_idx == 3 {
                println!("  ...");
            }
        }

        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let n_layers_f = meta.n_layers as f64;
        let mean_key_mse = total_key_mse / n_layers_f;
        let mean_key_cos = total_key_cos / n_layers_f;
        let mean_val_mse = total_val_mse / n_layers_f;
        let mean_val_cos = total_val_cos / n_layers_f;

        // Compression: bits per coord / 16 (fp16)
        let compression = 16.0 / bits as f64;
        let throughput = total_vectors as f64 / (elapsed_ms / 1000.0) / 1000.0;

        println!();
        println!("  AGGREGATE ({}-bit):", bits);
        println!(
            "    Key  MSE={:.6}  cosine={:.4}",
            mean_key_mse, mean_key_cos
        );
        println!(
            "    Val  MSE={:.6}  cosine={:.4}",
            mean_val_mse, mean_val_cos
        );
        println!("    Compression: {:.1}x vs FP16", compression);
        println!(
            "    Throughput: {:.0} kvec/s ({:.1}ms total)",
            throughput, elapsed_ms
        );
        println!(
            "    Streaming cache: {} tokens, {:.1}x compression",
            streaming_cache.len(),
            streaming_cache.compression_ratio()
        );
        println!();

        all_aggregates.push(AggregateResult {
            bits,
            mean_key_mse,
            mean_key_cosine: mean_key_cos,
            mean_value_mse: mean_val_mse,
            mean_value_cosine: mean_val_cos,
            compression_ratio_vs_fp16: compression,
            streaming_compression_ratio: streaming_cache.compression_ratio(),
            total_vectors,
            quantize_throughput_kvec_per_sec: throughput,
        });
    }

    // === Codebook Distribution Comparison ===
    // Test how much better fitted codebooks are vs Gaussian on real data
    println!("=== Codebook Distribution Comparison (3-bit, all layers) ===");
    println!();
    let comparison_bits = 3u32;
    let families: Vec<(&str, DistributionFamily, f64)> = vec![
        ("Gaussian", DistributionFamily::Gaussian, 2.0),
        ("Laplace", DistributionFamily::Laplace, 1.0),
        (
            "GenGauss(0.9)",
            DistributionFamily::GeneralizedGaussian,
            0.9,
        ),
        (
            "GenGauss(0.8)",
            DistributionFamily::GeneralizedGaussian,
            0.8,
        ),
    ];

    for (name, family, beta) in &families {
        let codebook =
            lloyd_max::get_codebook_family(meta.head_dim, comparison_bits, *family, *beta);
        let tq = TurboQuantMSE::with_codebook(meta.head_dim, comparison_bits, 42, true, codebook);
        let mut buf = vec![0.0f64; 3 * meta.head_dim];

        let mut total_key_mse = 0.0;
        let mut total_val_mse = 0.0;
        let mut total_key_cos = 0.0;
        let mut total_val_cos = 0.0;
        let mut count = 0usize;

        for layer_idx in 0..meta.n_layers {
            let key_path = cli.data_dir.join(format!("keys_layer{:02}.f64", layer_idx));
            let val_path = cli
                .data_dir
                .join(format!("values_layer{:02}.f64", layer_idx));
            let keys = read_f64_vectors(&key_path, meta.head_dim)?;
            let values = read_f64_vectors(&val_path, meta.head_dim)?;

            for (k, v) in keys.iter().zip(values.iter()) {
                let ck = tq.quantize(k, &mut buf);
                let mut rk = vec![0.0f64; meta.head_dim];
                tq.dequantize(&ck, &mut buf, &mut rk);
                total_key_mse += k
                    .iter()
                    .zip(rk.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / meta.head_dim as f64;
                total_key_cos += cosine_similarity_f64(k, &rk);

                let cv = tq.quantize(v, &mut buf);
                let mut rv = vec![0.0f64; meta.head_dim];
                tq.dequantize(&cv, &mut buf, &mut rv);
                total_val_mse += v
                    .iter()
                    .zip(rv.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / meta.head_dim as f64;
                total_val_cos += cosine_similarity_f64(v, &rv);

                count += 1;
            }
        }

        let n = count as f64;
        println!(
            "  {:15} | key MSE={:.6} cos={:.4} | val MSE={:.6} cos={:.4}",
            name,
            total_key_mse / n,
            total_key_cos / n,
            total_val_mse / n,
            total_val_cos / n
        );
    }

    // Compute improvement and verify post-rotation distribution
    {
        let gauss_cb = lloyd_max::get_codebook_family(
            meta.head_dim,
            comparison_bits,
            DistributionFamily::Gaussian,
            2.0,
        );
        let gg_cb = lloyd_max::get_codebook_family(
            meta.head_dim,
            comparison_bits,
            DistributionFamily::GeneralizedGaussian,
            0.9,
        );
        println!();
        println!("  Gaussian codebook centroids: {:?}", gauss_cb.centroids);
        println!("  GenGauss(0.9) centroids:     {:?}", gg_cb.centroids);

        // Verify: rotation Gaussianizes the data (post-rotation kurtosis should be ~3)
        use cd_kernel::turboquant::rotation::Rotation;
        let rot = Rotation::new_fast_jl(meta.head_dim, 42);
        let key_path = cli.data_dir.join("keys_layer15.f64");
        let keys_mid = read_f64_vectors(&key_path, meta.head_dim)?;
        let mut buf_rot = vec![0.0f64; 3 * meta.head_dim];
        let mut post_rot_vals = Vec::new();
        for k in keys_mid.iter().take(200) {
            // Normalize
            let norm: f64 = k.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                continue;
            }
            let normalized: Vec<f64> = k.iter().map(|x| x / norm).collect();
            // Rotate (use split_at_mut to avoid aliasing)
            let (scratch, rest) = buf_rot.split_at_mut(meta.head_dim);
            let (_, rotated) = rest.split_at_mut(meta.head_dim);
            rot.forward(&normalized, scratch, rotated);
            post_rot_vals.extend_from_slice(rotated);
        }
        let pr_mean = post_rot_vals.iter().sum::<f64>() / post_rot_vals.len() as f64;
        let pr_var = post_rot_vals
            .iter()
            .map(|x| (x - pr_mean).powi(2))
            .sum::<f64>()
            / post_rot_vals.len() as f64;
        let pr_std = pr_var.sqrt();
        let pr_kurt = if pr_std > 1e-15 {
            post_rot_vals
                .iter()
                .map(|x| ((x - pr_mean) / pr_std).powi(4))
                .sum::<f64>()
                / post_rot_vals.len() as f64
        } else {
            0.0
        };
        println!();
        println!("  Post-rotation distribution (layer 15 keys, 200 vectors):");
        println!(
            "    mean={:.6}, std={:.6}, kurtosis={:.2}",
            pr_mean, pr_std, pr_kurt
        );
        println!(
            "    Gaussian kurtosis=3.0 -> rotation {}",
            if (pr_kurt - 3.0).abs() < 1.5 {
                "GAUSSIANIZES the data (codebook is optimal)"
            } else {
                "does NOT fully Gaussianize (fitted codebook may help)"
            }
        );
    }
    println!();

    // === Outlier Retention Comparison (2-bit) ===
    println!("=== Outlier Retention at 2-bit ===");
    println!();
    for keep_pct in [0.0, 0.5, 1.0, 2.0] {
        let tq =
            TurboQuantMSE::new(meta.head_dim, 2, 42, true).with_outlier_retention(keep_pct / 100.0);
        let mut buf = vec![0.0f64; 3 * meta.head_dim];
        let mut total_mse = 0.0;
        let mut total_cos = 0.0;
        let mut count = 0usize;
        let mut total_outlier_bytes = 0usize;

        for layer_idx in 0..meta.n_layers {
            let key_path = cli.data_dir.join(format!("keys_layer{:02}.f64", layer_idx));
            let keys = read_f64_vectors(&key_path, meta.head_dim)?;
            for k in &keys {
                let comp = tq.quantize(k, &mut buf);
                total_outlier_bytes += comp.outliers.len() * 4; // u16 + f32
                let mut recon = vec![0.0f64; meta.head_dim];
                tq.dequantize(&comp, &mut buf, &mut recon);
                total_mse += k
                    .iter()
                    .zip(recon.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / meta.head_dim as f64;
                total_cos += cosine_similarity_f64(k, &recon);
                count += 1;
            }
        }
        let mean_mse = total_mse / count as f64;
        let mean_cos = total_cos / count as f64;
        let eff_bits = if count > 0 {
            (count * meta.head_dim * 2 / 8 + total_outlier_bytes) as f64 * 8.0
                / (count * meta.head_dim) as f64
        } else {
            2.0
        };
        println!(
            "  Keep {:4.1}%: key MSE={:.6} cos={:.4} eff_bits={:.2}",
            keep_pct, mean_mse, mean_cos, eff_bits
        );
    }
    println!();

    // === CDAQ Pipeline Comparison (2-bit) ===
    println!("=== CDAQ vs Baseline at 2-bit (calibrated channel reorder + per-group scales) ===");
    println!();
    {
        // Collect calibration vectors from first 3 layers
        let mut cal_vecs: Vec<Vec<f64>> = Vec::new();
        for layer_idx in 0..3.min(meta.n_layers) {
            let key_path = cli.data_dir.join(format!("keys_layer{:02}.f64", layer_idx));
            let keys = read_f64_vectors(&key_path, meta.head_dim)?;
            cal_vecs.extend(keys);
        }
        let cal_refs: Vec<&[f64]> = cal_vecs.iter().map(|v| v.as_slice()).collect();

        // CDAQ with calibration
        let cdaq = CdaqQuantizer::new(meta.head_dim, 2, 42, &cal_refs, 16, 0.01);
        // Baseline TurboQuant
        let tq_base = TurboQuantMSE::new(meta.head_dim, 2, 42, true);

        let mut buf = vec![0.0f64; 3 * meta.head_dim];
        let mut cdaq_mse = 0.0;
        let mut cdaq_cos = 0.0;
        let mut base_mse = 0.0;
        let mut base_cos = 0.0;
        let mut count = 0usize;

        for layer_idx in 0..meta.n_layers {
            let key_path = cli.data_dir.join(format!("keys_layer{:02}.f64", layer_idx));
            let keys = read_f64_vectors(&key_path, meta.head_dim)?;
            for k in &keys {
                // CDAQ
                let comp_c = cdaq.quantize(k, &mut buf);
                let mut recon_c = vec![0.0f64; meta.head_dim];
                cdaq.dequantize(&comp_c, &mut buf, &mut recon_c);
                cdaq_mse += k
                    .iter()
                    .zip(recon_c.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / meta.head_dim as f64;
                cdaq_cos += cosine_similarity_f64(k, &recon_c);

                // Baseline
                let comp_b = tq_base.quantize(k, &mut buf);
                let mut recon_b = vec![0.0f64; meta.head_dim];
                tq_base.dequantize(&comp_b, &mut buf, &mut recon_b);
                base_mse += k
                    .iter()
                    .zip(recon_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / meta.head_dim as f64;
                base_cos += cosine_similarity_f64(k, &recon_b);

                count += 1;
            }
        }

        let n = count as f64;
        let improvement = (1.0 - cdaq_mse / base_mse) * 100.0;
        println!(
            "  TurboQuant baseline: key MSE={:.6} cos={:.4}",
            base_mse / n,
            base_cos / n
        );
        println!(
            "  CDAQ (calibrated):   key MSE={:.6} cos={:.4} ({:+.1}% MSE)",
            cdaq_mse / n,
            cdaq_cos / n,
            improvement
        );
        println!("  CDAQ eff bits: {:.2}", cdaq.effective_bits_per_coord());
    }
    println!();

    // Save JSON output
    let output = RealKvOutput {
        model: meta.model,
        n_params: meta.n_params,
        n_layers: meta.n_layers,
        n_heads: meta.n_heads,
        seq_len: meta.seq_len,
        head_dim: meta.head_dim,
        total_kv_vectors: meta.n_layers * meta.n_heads * meta.seq_len * 2,
        aggregates: all_aggregates,
        per_layer: all_layer_results,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&output)?;
    fs::write(&cli.out_json, &json)?;
    println!("Results saved to {}", cli.out_json.display());

    Ok(())
}
