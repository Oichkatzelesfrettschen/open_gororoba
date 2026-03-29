//! TurboQuant unified benchmark harness.
//!
//! Benchmarks the full TurboQuant pipeline (rotation, quantization, QJL,
//! inner product estimation) across dimensions, bit-widths, and rotation
//! methods.  Outputs machine-readable JSON with throughput, correctness,
//! and quality metrics.
//!
//! Usage:
//!   turboquant-bench --dims 64,128,256 --bits 2,3,4 --n-vectors 10000
//!   turboquant-bench --rotation wht --bits 3 --dim 128

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf, time::Instant};

use cd_kernel::turboquant::dispatch::{DispatchedQuantizer, detect_simd_level};
use cd_kernel::turboquant::pipeline::{TurboQuantMSE, TurboQuantProd};
use cd_kernel::turboquant::sign_pack::BitPackedSigns;
use cd_kernel::turboquant::simd_codebook::SimdBoundaries;

#[derive(Parser)]
#[command(name = "turboquant-bench")]
#[command(about = "TurboQuant pipeline benchmark: rotation + quantization + QJL")]
struct Cli {
    /// Comma-separated dimensions to benchmark.
    #[arg(long, value_delimiter = ',', default_values_t = vec![64, 128, 256])]
    dims: Vec<usize>,

    /// Comma-separated bit-widths to benchmark.
    #[arg(long, value_delimiter = ',', default_values_t = vec![2, 3, 4])]
    bits: Vec<u32>,

    /// Number of vectors to quantize per configuration.
    #[arg(long, default_value_t = 10_000)]
    n_vectors: usize,

    /// Rotation method: "haar", "wht", "e8", "all", or "both" (haar+wht).
    #[arg(long, default_value = "all")]
    rotation: String,

    /// Output JSON path.
    #[arg(long, default_value = "data/output/heliosphere/ablations/turboquant_bench.json")]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct PipelineResult {
    dim: usize,
    bits: u32,
    rotation: String,
    n_vectors: usize,
    // Timing
    quantize_ms: f64,
    dequantize_ms: f64,
    inner_product_ms: f64,
    // Throughput
    quantize_kvec_per_sec: f64,
    dequantize_kvec_per_sec: f64,
    inner_product_kvec_per_sec: f64,
    // Quality
    mse_per_coord: f64,
    cosine_similarity_mean: f64,
    ip_mean_abs_error: f64,
    ip_bias: f64,
    // Memory
    sign_pack_bytes: usize,
    sign_naive_bytes: usize,
    sign_compression_ratio: f64,
    bits_per_vector: usize,
    compression_ratio_vs_fp16: f64,
}

#[derive(Debug, Serialize)]
struct CodebookBenchResult {
    dim: usize,
    bits: u32,
    method: String,
    n_values: usize,
    elapsed_ms: f64,
    throughput_mvalues_per_sec: f64,
    correct: bool,
}

#[derive(Debug, Serialize)]
struct BenchOutput {
    n_vectors: usize,
    configs_tested: usize,
    simd_level: String,
    results: Vec<PipelineResult>,
    codebook_bench: Vec<CodebookBenchResult>,
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 {
        return 0.0;
    }
    dot / (na * nb)
}

fn generate_random_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    (0..n)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

fn bench_mse_pipeline(
    d: usize,
    bits: u32,
    rotation_name: &str,
    vectors: &[Vec<f64>],
    queries: &[Vec<f64>],
) -> PipelineResult {
    let n = vectors.len();

    // Select rotation method based on name
    let use_wht = rotation_name != "haar";
    let use_e8 = rotation_name == "e8" && d == 128;

    // MSE-only pipeline (must match the rotation used by prod)
    let tq_mse = if use_e8 {
        let mut cfg = cd_kernel::turboquant::config::TurboQuantConfig::recommended(d, bits);
        cfg.rotation = cd_kernel::turboquant::config::RotationMethod::E8Block;
        TurboQuantMSE::from_config(&cfg, bits, 42)
    } else {
        TurboQuantMSE::new(d, bits, 42, use_wht)
    };
    // Prod pipeline (MSE + QJL) -- must use same rotation as MSE
    let tq_prod = if use_e8 {
        // E8 prod not yet supported -- use MSE-only for E8 bench
        TurboQuantProd::new(d, bits, 42, true, None) // WHT fallback for QJL
    } else {
        TurboQuantProd::new(d, bits, 42, use_wht, None)
    };

    let mut buf = vec![0.0f64; 3 * d];

    // --- Quantize benchmark (MSE stage) ---
    let t0 = Instant::now();
    let mut mse_compressed: Vec<_> = Vec::with_capacity(n);
    for v in vectors {
        mse_compressed.push(tq_mse.quantize(v, &mut buf));
    }
    let quantize_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Also quantize with prod for QJL benchmarks (uses its own rotation)
    let mut compressed: Vec<_> = Vec::with_capacity(n);
    for v in vectors {
        compressed.push(tq_prod.quantize(v, &mut buf));
    }

    // --- Dequantize benchmark (using matched MSE rotation) ---
    let t0 = Instant::now();
    let mut reconstructed: Vec<Vec<f64>> = Vec::with_capacity(n);
    for c in &mse_compressed {
        let mut out = vec![0.0f64; d];
        tq_mse.dequantize(c, &mut buf, &mut out);
        reconstructed.push(out);
    }
    let dequantize_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // --- Inner product benchmark ---
    let n_ip = queries.len().min(n);
    let t0 = Instant::now();
    let mut ip_errors = Vec::with_capacity(n_ip);
    for (qi, q) in queries.iter().enumerate().take(n_ip) {
        let true_ip: f64 = q.iter().zip(vectors[qi].iter()).map(|(a, b)| a * b).sum();
        let est_ip = tq_prod.inner_product(q, &compressed[qi], &mut buf);
        ip_errors.push(est_ip - true_ip);
    }
    let inner_product_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // --- Quality metrics ---
    let total_coords = n * d;
    let mse_per_coord: f64 = vectors
        .iter()
        .zip(reconstructed.iter())
        .flat_map(|(orig, recon)| orig.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)))
        .sum::<f64>()
        / total_coords as f64;

    let cosine_mean: f64 = vectors
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| cosine_similarity(a, b))
        .sum::<f64>()
        / n as f64;

    let ip_mean_abs_error: f64 =
        ip_errors.iter().map(|e| e.abs()).sum::<f64>() / ip_errors.len().max(1) as f64;
    let ip_bias: f64 = ip_errors.iter().sum::<f64>() / ip_errors.len().max(1) as f64;

    // --- Sign packing stats ---
    let sign_pack_bytes: usize = compressed
        .iter()
        .map(|c| BitPackedSigns::pack(&c.qjl_signs).byte_size())
        .sum();
    let sign_naive_bytes = compressed.iter().map(|c| c.qjl_signs.len()).sum::<usize>();

    let bits_per_vector = tq_prod.bits_per_vector();
    let compression_ratio = tq_prod.compression_ratio();

    let kvec = |ms: f64| -> f64 { n as f64 / ms };

    PipelineResult {
        dim: d,
        bits,
        rotation: rotation_name.to_string(),
        n_vectors: n,
        quantize_ms,
        dequantize_ms,
        inner_product_ms,
        quantize_kvec_per_sec: kvec(quantize_ms),
        dequantize_kvec_per_sec: kvec(dequantize_ms),
        inner_product_kvec_per_sec: kvec(inner_product_ms),
        mse_per_coord,
        cosine_similarity_mean: cosine_mean,
        ip_mean_abs_error,
        ip_bias,
        sign_pack_bytes,
        sign_naive_bytes,
        sign_compression_ratio: sign_naive_bytes as f64 / sign_pack_bytes.max(1) as f64,
        bits_per_vector,
        compression_ratio_vs_fp16: compression_ratio,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("=== TurboQuant Pipeline Benchmark ===");
    println!(
        "  Configs: {} dims x {} bits x rotation={}",
        cli.dims.len(),
        cli.bits.len(),
        cli.rotation
    );
    println!("  Vectors per config: {}", cli.n_vectors);

    let rotation_names: Vec<&str> = match cli.rotation.as_str() {
        "haar" => vec!["haar"],
        "wht" => vec!["wht"],
        "e8" => vec!["e8"],
        "both" => vec!["haar", "wht"],
        "all" => vec!["haar", "wht", "e8"],
        _ => {
            eprintln!("Unknown rotation '{}', using all", cli.rotation);
            vec!["haar", "wht", "e8"]
        }
    };

    let mut results = Vec::new();

    for &d in &cli.dims {
        let vectors = generate_random_vectors(cli.n_vectors, d, 42);
        let queries = generate_random_vectors(cli.n_vectors.min(1000), d, 99);

        for &bits in &cli.bits {
            for &rot_name in &rotation_names {
                // E8 only works at d=128
                if rot_name == "e8" && d != 128 {
                    continue;
                }
                print!("  d={:<4} bits={} rot={:<4} ... ", d, bits, rot_name);

                let result = bench_mse_pipeline(d, bits, rot_name, &vectors, &queries);

                println!(
                    "quantize {:.0} kvec/s  MSE={:.6}  cos={:.4}  ratio={:.1}x",
                    result.quantize_kvec_per_sec,
                    result.mse_per_coord,
                    result.cosine_similarity_mean,
                    result.compression_ratio_vs_fp16
                );

                results.push(result);
            }
        }
    }

    // --- SIMD codebook micro-benchmark ---
    let simd_level = detect_simd_level();
    println!("\n  SIMD codebook benchmark (level: {})", simd_level);
    let mut codebook_bench = Vec::new();

    for &d in &cli.dims {
        for &bits in &cli.bits {
            let cb = cd_kernel::lloyd_max::get_codebook(d, bits);
            let sigma = 1.0 / (d as f32).sqrt();
            let total = cli.n_vectors * d;
            let values: Vec<f32> = (0..total)
                .map(|i| ((i as f32 * 0.618) % 1.0 - 0.5) * 7.0 * sigma)
                .collect();

            // Scalar boundary baseline
            let mut scalar_out = vec![0u8; total];
            let t0 = Instant::now();
            for (i, &v) in values.iter().enumerate() {
                scalar_out[i] = cd_kernel::turboquant::simd_codebook::quantize_scalar_boundary(v, &cb.boundaries);
            }
            let scalar_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // SIMD codebook
            let simd_cb = SimdBoundaries::from_boundaries(&cb.boundaries, bits);
            let mut simd_out = vec![0u8; total];
            let t0 = Instant::now();
            simd_cb.quantize_batch(&values, &mut simd_out);
            let simd_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let correct = scalar_out == simd_out;

            // Dispatched (auto-selects best)
            let disp = DispatchedQuantizer::new(&cb, bits);
            let mut disp_out = vec![0u8; total];
            let t0 = Instant::now();
            disp.quantize(&values, &mut disp_out);
            let disp_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let throughput = |ms: f64| total as f64 / ms / 1000.0;

            print!("  d={:<4} bits={} ", d, bits);
            println!(
                "scalar={:.1}Mv/s  simd={:.1}Mv/s ({:.1}x)  dispatch={:.1}Mv/s  ok={}",
                throughput(scalar_ms),
                throughput(simd_ms),
                scalar_ms / simd_ms,
                throughput(disp_ms),
                correct
            );

            codebook_bench.push(CodebookBenchResult {
                dim: d, bits,
                method: "scalar_boundary".into(),
                n_values: total,
                elapsed_ms: scalar_ms,
                throughput_mvalues_per_sec: throughput(scalar_ms),
                correct: true,
            });
            codebook_bench.push(CodebookBenchResult {
                dim: d, bits,
                method: format!("simd_f32x8_{}", simd_level),
                n_values: total,
                elapsed_ms: simd_ms,
                throughput_mvalues_per_sec: throughput(simd_ms),
                correct,
            });
            codebook_bench.push(CodebookBenchResult {
                dim: d, bits,
                method: format!("dispatched_{}", simd_level),
                n_values: total,
                elapsed_ms: disp_ms,
                throughput_mvalues_per_sec: throughput(disp_ms),
                correct: disp_out == scalar_out,
            });
        }
    }

    let output = BenchOutput {
        n_vectors: cli.n_vectors,
        configs_tested: results.len(),
        simd_level: format!("{}", simd_level),
        results,
        codebook_bench,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("\n  Wrote {}", cli.out_json.display());

    Ok(())
}
