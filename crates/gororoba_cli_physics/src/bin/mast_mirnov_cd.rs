//! MAST Mirnov CD analysis: read FAIR-MAST Zarr v2 data (blosc/lz4 compressed),
//! extract Mirnov coil time series, build 32D Takens embedding, compute CD associator.
//!
//! Data source: FAIR-MAST S3 bucket (public, no auth)
//! Zarr store: level1/shots/{shot}.zarr/ama/ (Centre Column Toroidal Bv Mirnovs)
//! Signals: n=2_signal, n=odd_signal, n=2_amplitude, n=2_frequency, time

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf, sync::Arc};
use zarrs::array::Array;
use zarrs_filesystem::FilesystemStore;

#[derive(Parser)]
#[command(name = "mast-mirnov-cd")]
struct Cli {
    /// Path to local Zarr store directory (e.g., data/external/mast/shot30420.zarr/ama).
    #[arg(long)]
    zarr_dir: PathBuf,

    /// Embedding dimension for CD associator.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Window stride (subsample factor for 650K-point data).
    #[arg(long, default_value_t = 100)]
    stride: usize,

    /// Normalization: "direction" or "raw".
    #[arg(long, default_value = "direction")]
    normalization: String,

    /// Precision: "f64" (default) or "f32" (2x faster for 256D+).
    #[arg(long, default_value = "f64")]
    precision: String,

    /// Use ring-buffer streaming mode (O(3*dim) memory, from steinmarder A-A pattern).
    #[arg(long, default_value_t = false)]
    stream: bool,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/mast_mirnov_cd.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct WindowResult {
    center_time: f64,
    mean_associator: f64,
    max_associator: f64,
    n_embeddings: usize,
}

#[derive(Debug, Serialize)]
struct MastCdOutput {
    zarr_dir: String,
    embedding_dim: usize,
    stride: usize,
    normalization: String,
    n_samples: usize,
    time_range: [f64; 2],
    signal_range: [f64; 2],
    n_windows: usize,
    results: Vec<WindowResult>,
    overall_mean_a: f64,
    overall_max_a: f64,
    interpretation: String,
}

fn read_zarr_f32(store: &Arc<FilesystemStore>, name: &str) -> Result<Vec<f32>> {
    let array = Array::open(store.clone(), &format!("/{name}"))
        .with_context(|| format!("Failed to open array '{name}'"))?;
    let shape = array.shape().to_vec();
    if shape.is_empty() {
        bail!("Array '{name}' has empty shape");
    }
    let subset = zarrs::array::ArraySubset::new_with_shape(shape.clone());
    let flat: Vec<f32> = array
        .retrieve_array_subset::<Vec<f32>>(&subset)
        .with_context(|| format!("Failed to read array '{name}'"))?;
    Ok(flat)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== MAST Mirnov CD Analysis (pure Rust zarrs) ===");
    println!("  Zarr dir: {}", cli.zarr_dir.display());

    let store = Arc::new(
        FilesystemStore::new(&cli.zarr_dir)
            .map_err(|e| anyhow::anyhow!("Failed to open store: {:?}", e))?,
    );

    // Read time and signals
    let time_f32 = read_zarr_f32(&store, "time")?;
    let sig_n2_f32 = read_zarr_f32(&store, "n=2_signal")?;
    let sig_nodd_f32 = read_zarr_f32(&store, "n=odd_signal")?;

    let n = time_f32.len();
    println!("  Loaded {} samples", n);
    println!(
        "  Time: {:.4} to {:.4} s",
        time_f32.first().unwrap_or(&0.0),
        time_f32.last().unwrap_or(&0.0)
    );

    // Convert to f64 for CD kernel
    let time: Vec<f64> = time_f32.iter().map(|&x| x as f64).collect();
    let sig_n2: Vec<f64> = sig_n2_f32.iter().map(|&x| x as f64).collect();
    let sig_nodd: Vec<f64> = sig_nodd_f32.iter().map(|&x| x as f64).collect();

    let sig_min = sig_n2.iter().cloned().fold(f64::INFINITY, f64::min);
    let sig_max = sig_n2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  n=2 signal range: {:.4e} to {:.4e}", sig_min, sig_max);

    // Build sliding windows of embedding vectors
    // 2 channels (n=2 + n=odd), embedding_dim steps per channel
    let channels = 2usize;
    let steps = cli.embedding_dim / channels;
    let remainder = cli.embedding_dim - steps * channels;
    let window_size = steps * cli.stride;

    if n < window_size + cli.stride {
        bail!(
            "Not enough data: {} samples, need {} for one window",
            n,
            window_size + cli.stride
        );
    }

    // Process in large windows (e.g., 10000 samples each) to get A(t)
    let mega_window = 50_000usize;
    let mega_stride = 25_000usize;
    let n_mega = if n > mega_window {
        (n - mega_window) / mega_stride + 1
    } else {
        1
    };

    // Streaming mode: use ring buffer for O(3*dim) memory
    if cli.stream {
        use cd_kernel::cayley_dickson::soa_cache::RingEmbeddingCache;
        println!("  Streaming mode (ring buffer, O(3*{}) memory)", cli.embedding_dim);

        let mut ring = RingEmbeddingCache::new(cli.embedding_dim);
        let n_embed = if n > steps * cli.stride {
            (n - steps * cli.stride) / cli.stride
        } else {
            0
        };

        let mut all_norms = Vec::new();
        for w in 0..n_embed {
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for s in 0..steps {
                let idx = w * cli.stride + s * cli.stride;
                if idx < n {
                    v.push(sig_n2[idx] as f32);
                    v.push(sig_nodd[idx] as f32);
                }
            }
            for _ in 0..remainder {
                v.push(0.0f32);
            }
            v.truncate(cli.embedding_dim);

            if cli.normalization == "direction" {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-7 {
                    for x in v.iter_mut() {
                        *x /= norm;
                    }
                }
            }

            if let Some(norm) = ring.push_and_compute(&v) {
                all_norms.push(norm);
            }
        }

        let mean_a = if all_norms.is_empty() {
            0.0
        } else {
            all_norms.iter().sum::<f32>() / all_norms.len() as f32
        };
        let max_a = all_norms.iter().cloned().fold(0.0f32, f32::max);

        println!(
            "  Stream: {} norms, A_mean={:.6}, A_max={:.6}",
            all_norms.len(), mean_a, max_a
        );

        let output = serde_json::json!({
            "mode": "stream",
            "n_samples": n,
            "n_norms": all_norms.len(),
            "overall_mean_a": mean_a,
            "overall_max_a": max_a,
            "embedding_dim": cli.embedding_dim,
            "stride": cli.stride,
        });
        if let Some(parent) = cli.out_json.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
        println!("  Wrote {}", cli.out_json.display());
        return Ok(());
    }

    println!(
        "  Processing {} mega-windows of {} samples (stride {})",
        n_mega, mega_window, mega_stride
    );

    let mut results = Vec::new();

    for mw in 0..n_mega {
        let start = mw * mega_stride;
        let end = (start + mega_window).min(n);
        let local_n = end - start;

        // Build embeddings within this mega-window
        let local_n_windows = if local_n > steps * cli.stride {
            (local_n - steps * cli.stride) / cli.stride
        } else {
            continue;
        };

        let mut embedded: Vec<Vec<f64>> = Vec::with_capacity(local_n_windows);
        for w in 0..local_n_windows {
            let base = start + w * cli.stride;
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for s in 0..steps {
                let idx = base + s * cli.stride;
                if idx < n {
                    v.push(sig_n2[idx]);
                    v.push(sig_nodd[idx]);
                }
            }
            for _ in 0..remainder {
                v.push(0.0);
            }
            if v.len() == cli.embedding_dim {
                embedded.push(v);
            }
        }

        if embedded.len() < 3 {
            continue;
        }

        // Direction-only normalization
        if cli.normalization == "direction" {
            for v in &mut embedded {
                let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    for x in v.iter_mut() {
                        *x /= norm;
                    }
                }
            }
        }

        let (mean_a, max_a) = if cli.precision == "f32" {
            // f32 quantized path: ~2x faster for 256D+
            let embedded_f32: Vec<Vec<f32>> = embedded
                .iter()
                .map(|v| v.iter().map(|&x| x as f32).collect())
                .collect();
            let norms = cd_kernel::cayley_dickson::simd::batch_sliding_associator_norms_f32(
                &embedded_f32,
                cli.embedding_dim,
            );
            if norms.is_empty() {
                continue;
            }
            let m = norms.iter().sum::<f32>() / norms.len() as f32;
            let mx = norms.iter().cloned().fold(0.0f32, f32::max);
            (m as f64, mx as f64)
        } else {
            let norms = cd_kernel::batch_sliding_associator_norms_parallel(
                &embedded,
                cli.embedding_dim,
            );
            if norms.is_empty() {
                continue;
            }
            let m = norms.iter().sum::<f64>() / norms.len() as f64;
            let mx = norms.iter().cloned().fold(0.0f64, f64::max);
            (m, mx)
        };
        let center_idx = start + local_n / 2;
        let center_time = if center_idx < n { time[center_idx] } else { 0.0 };

        if mw % 5 == 0 || mw == n_mega - 1 {
            println!(
                "  window {}/{}: t={:.4}s, A_mean={:.6}, A_max={:.6}, embeds={}",
                mw + 1,
                n_mega,
                center_time,
                mean_a,
                max_a,
                embedded.len()
            );
        }

        results.push(WindowResult {
            center_time,
            mean_associator: mean_a,
            max_associator: max_a,
            n_embeddings: embedded.len(),
        });
    }

    let overall_mean = if results.is_empty() {
        0.0
    } else {
        results.iter().map(|r| r.mean_associator).sum::<f64>() / results.len() as f64
    };
    let overall_max = results
        .iter()
        .map(|r| r.max_associator)
        .fold(0.0f64, f64::max);

    let interp = format!(
        "MAST shot 30420 ama Mirnov (n=2 + n=odd, {} channels). {} windows, overall A_mean={:.6}, A_max={:.6}. {}D embedding, stride={}, {}.",
        channels, results.len(), overall_mean, overall_max, cli.embedding_dim, cli.stride, cli.normalization
    );
    println!("\n  {}", interp);

    let output = MastCdOutput {
        zarr_dir: cli.zarr_dir.display().to_string(),
        embedding_dim: cli.embedding_dim,
        stride: cli.stride,
        normalization: cli.normalization,
        n_samples: n,
        time_range: [
            time.first().copied().unwrap_or(0.0),
            time.last().copied().unwrap_or(0.0),
        ],
        signal_range: [sig_min, sig_max],
        n_windows: results.len(),
        results,
        overall_mean_a: overall_mean,
        overall_max_a: overall_max,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}
