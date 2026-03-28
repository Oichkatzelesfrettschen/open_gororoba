//! TurboQuant SIMD codebook benchmark: compare scalar vs SIMD Lloyd-Max
//! quantization lookup. At 3-bit (8 centroids), one f32x8 SIMD register
//! holds the entire codebook -- enabling single-instruction distance computation.

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(name = "turboquant-simd-bench")]
struct Cli {
    /// Number of vectors to quantize.
    #[arg(long, default_value_t = 100_000)]
    n_vectors: usize,

    /// Vector dimension.
    #[arg(long, default_value_t = 128)]
    dim: usize,

    /// Quantization bits.
    #[arg(long, default_value_t = 3)]
    bits: u32,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/turboquant_simd_bench.json"
    )]
    out_json: PathBuf,
}

/// Scalar Lloyd-Max codebook lookup (baseline, mimics PyTorch argmin)
fn quantize_scalar(values: &[f32], centroids: &[f32]) -> Vec<u8> {
    let _n = centroids.len();
    values
        .iter()
        .map(|&v| {
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;
            for (i, &c) in centroids.iter().enumerate() {
                let d = (v - c).abs();
                if d < best_dist {
                    best_dist = d;
                    best_idx = i as u8;
                }
            }
            best_idx
        })
        .collect()
}

/// Boundary-based quantization (optimal for sorted centroids)
/// Uses binary search on precomputed boundaries instead of argmin
fn quantize_boundary(values: &[f32], boundaries: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|&v| {
            // boundaries.len() == n_centroids - 1
            // Binary search: find first boundary > v
            let mut idx = 0u8;
            for &b in boundaries {
                if v > b {
                    idx += 1;
                } else {
                    break;
                }
            }
            idx
        })
        .collect()
}

/// SIMD-style boundary quantization using wide operations
/// For 3-bit (7 boundaries), we can compare all 7 simultaneously
fn quantize_simd_style(values: &[f32], boundaries: &[f32]) -> Vec<u8> {
    // For 3-bit: 7 boundaries. Compare v > each boundary, count trues.
    // This is the SIMD pattern: broadcast v, compare against all boundaries at once.
    let n_bounds = boundaries.len();
    values
        .iter()
        .map(|&v| {
            let mut count = 0u8;
            // Unrolled comparison (compiler will auto-vectorize with -O2)
            for i in 0..n_bounds {
                if v > boundaries[i] {
                    count += 1;
                }
            }
            count
        })
        .collect()
}

/// Batched SIMD: process 8 values at once against boundaries
fn quantize_batched(values: &[f32], boundaries: &[f32]) -> Vec<u8> {
    let n = values.len();
    let mut result = vec![0u8; n];
    let n_bounds = boundaries.len();

    // Process in chunks of 8 for SIMD-friendly access
    let chunks = n / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        for bi in 0..n_bounds {
            let b = boundaries[bi];
            for j in 0..8 {
                if values[base + j] > b {
                    result[base + j] += 1;
                }
            }
        }
    }
    // Handle remainder
    for i in (chunks * 8)..n {
        for &b in boundaries {
            if values[i] > b {
                result[i] += 1;
            }
        }
    }
    result
}

#[derive(Debug, Serialize)]
struct BenchResult {
    method: String,
    n_values: usize,
    elapsed_ms: f64,
    throughput_mvalues_per_sec: f64,
    correct: bool,
}

#[derive(Debug, Serialize)]
struct BenchOutput {
    n_vectors: usize,
    dim: usize,
    bits: u32,
    total_values: usize,
    centroids: Vec<f32>,
    boundaries: Vec<f32>,
    results: Vec<BenchResult>,
    speedup_boundary_vs_scalar: f64,
    speedup_simd_vs_scalar: f64,
    speedup_batched_vs_scalar: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== TurboQuant SIMD Codebook Benchmark ===");

    let n_levels = 1u32 << cli.bits;
    let total_values = cli.n_vectors * cli.dim;

    // Generate Lloyd-Max codebook using cached pure-Rust solver
    // (replaces hardcoded centroids; TurboQuant pattern: cache per (d, bits))
    let codebook = cd_kernel::lloyd_max::get_codebook(cli.dim, cli.bits);
    let centroids = codebook.centroids.clone();
    let boundaries = codebook.boundaries.clone();

    println!(
        "  Codebook: {} centroids ({}-bit), {} boundaries",
        n_levels, cli.bits, boundaries.len()
    );
    println!("  Quantizing {} values ({} vectors x {} dim)", total_values, cli.n_vectors, cli.dim);

    // Generate random rotated values (mimic post-rotation distribution)
    let sigma = 1.0 / (cli.dim as f32).sqrt();
    let values: Vec<f32> = (0..total_values)
        .map(|i| {
            // Simple deterministic pseudo-random
            let x = ((i as f64 * 0.618033988749895) % 1.0 - 0.5) * 6.0 * sigma as f64;
            x as f32
        })
        .collect();

    // Benchmark 1: Scalar argmin (PyTorch-equivalent)
    let t0 = Instant::now();
    let ref_indices = quantize_scalar(&values, &centroids);
    let scalar_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Benchmark 2: Boundary search
    let t0 = Instant::now();
    let boundary_indices = quantize_boundary(&values, &boundaries);
    let boundary_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Benchmark 3: SIMD-style (auto-vectorizable)
    let t0 = Instant::now();
    let simd_indices = quantize_simd_style(&values, &boundaries);
    let simd_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Benchmark 4: Batched (SIMD-friendly layout)
    let t0 = Instant::now();
    let batched_indices = quantize_batched(&values, &boundaries);
    let batched_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Verify correctness
    let boundary_correct = ref_indices == boundary_indices;
    let simd_correct = ref_indices == simd_indices;
    let batched_correct = ref_indices == batched_indices;

    let throughput = |ms: f64| -> f64 { total_values as f64 / ms / 1000.0 };

    println!("\n  Results:");
    println!(
        "  Scalar argmin:     {:7.2} ms ({:.1} Mval/s)",
        scalar_ms,
        throughput(scalar_ms)
    );
    println!(
        "  Boundary search:   {:7.2} ms ({:.1} Mval/s) [{}x] correct={}",
        boundary_ms,
        throughput(boundary_ms),
        format!("{:.1}", scalar_ms / boundary_ms),
        boundary_correct
    );
    println!(
        "  SIMD-style:        {:7.2} ms ({:.1} Mval/s) [{}x] correct={}",
        simd_ms,
        throughput(simd_ms),
        format!("{:.1}", scalar_ms / simd_ms),
        simd_correct
    );
    println!(
        "  Batched:           {:7.2} ms ({:.1} Mval/s) [{}x] correct={}",
        batched_ms,
        throughput(batched_ms),
        format!("{:.1}", scalar_ms / batched_ms),
        batched_correct
    );

    let results = vec![
        BenchResult {
            method: "scalar_argmin".into(),
            n_values: total_values,
            elapsed_ms: scalar_ms,
            throughput_mvalues_per_sec: throughput(scalar_ms),
            correct: true,
        },
        BenchResult {
            method: "boundary_search".into(),
            n_values: total_values,
            elapsed_ms: boundary_ms,
            throughput_mvalues_per_sec: throughput(boundary_ms),
            correct: boundary_correct,
        },
        BenchResult {
            method: "simd_style".into(),
            n_values: total_values,
            elapsed_ms: simd_ms,
            throughput_mvalues_per_sec: throughput(simd_ms),
            correct: simd_correct,
        },
        BenchResult {
            method: "batched".into(),
            n_values: total_values,
            elapsed_ms: batched_ms,
            throughput_mvalues_per_sec: throughput(batched_ms),
            correct: batched_correct,
        },
    ];

    let output = BenchOutput {
        n_vectors: cli.n_vectors,
        dim: cli.dim,
        bits: cli.bits,
        total_values,
        centroids,
        boundaries,
        results,
        speedup_boundary_vs_scalar: scalar_ms / boundary_ms,
        speedup_simd_vs_scalar: scalar_ms / simd_ms,
        speedup_batched_vs_scalar: scalar_ms / batched_ms,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("\n  Wrote {}", cli.out_json.display());

    Ok(())
}
