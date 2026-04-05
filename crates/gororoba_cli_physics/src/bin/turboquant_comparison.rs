//! TurboQuant head-to-head comparison: TurboQuant vs KIVI vs NSNQuant.
//!
//! Benchmarks all quantization methods at the same bit-widths and dimensions,
//! reporting MSE, cosine similarity, throughput, and compression ratio in a
//! single JSON file for direct comparison.
//!
//! Usage:
//!   turboquant-comparison --dims 64,128 --bits 2,3,4 --n-vectors 5000

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf, time::Instant};

use cd_kernel::turboquant::{
    baselines::{kivi, nsnquant},
    config::TurboQuantConfig,
    grouping,
    pipeline::TurboQuantMSE,
};

#[derive(Parser)]
#[command(name = "turboquant-comparison")]
#[command(about = "Head-to-head comparison: TurboQuant vs KIVI vs NSNQuant")]
struct Cli {
    #[arg(long, value_delimiter = ',', default_values_t = vec![64, 128])]
    dims: Vec<usize>,

    #[arg(long, value_delimiter = ',', default_values_t = vec![2, 3, 4])]
    bits: Vec<u32>,

    #[arg(long, default_value_t = 2000)]
    n_vectors: usize,

    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/turboquant_comparison.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct MethodResult {
    method: String,
    dim: usize,
    bits: u32,
    n_vectors: usize,
    mse_per_coord: f64,
    cosine_mean: f64,
    quantize_ms: f64,
    throughput_kvec_per_sec: f64,
    total_bits: usize,
    fp16_bits: usize,
    compression_ratio: f64,
}

#[derive(Debug, Serialize)]
struct ComparisonOutput {
    configs_tested: usize,
    methods_per_config: usize,
    results: Vec<MethodResult>,
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn random_matrix(n: usize, d: usize, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    (0..n * d).map(|_| normal.sample(&mut rng)).collect()
}

fn bench_turboquant(data: &[f64], n: usize, d: usize, bits: u32) -> MethodResult {
    let cfg = TurboQuantConfig::recommended(d, bits);
    let tq = TurboQuantMSE::new(d, bits, 42, cfg.use_wht());
    let mut buf = vec![0.0f64; 2 * d];

    let t0 = Instant::now();
    let mut total_mse = 0.0f64;
    let mut total_cos = 0.0f64;

    for v in 0..n {
        let offset = v * d;
        let x = &data[offset..offset + d];
        let compressed = tq.quantize(x, &mut buf);
        let mut recon = vec![0.0f64; d];
        tq.dequantize(&compressed, &mut buf, &mut recon);

        let mse: f64 = x
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        total_mse += mse;
        total_cos += cosine_sim(x, &recon);
    }
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    let total_bits = n * d * bits as usize + n * 16; // indices + norms
    let fp16_bits = n * d * 16;

    MethodResult {
        method: format!(
            "turboquant-{}",
            if cfg.use_e8() {
                "e8"
            } else if cfg.use_wht() {
                "wht"
            } else {
                "haar"
            }
        ),
        dim: d,
        bits,
        n_vectors: n,
        mse_per_coord: total_mse / n as f64,
        cosine_mean: total_cos / n as f64,
        quantize_ms: elapsed,
        throughput_kvec_per_sec: n as f64 / elapsed,
        total_bits,
        fp16_bits,
        compression_ratio: fp16_bits as f64 / total_bits as f64,
    }
}

fn bench_kivi(data: &[f64], n: usize, d: usize, bits: u32) -> MethodResult {
    let t0 = Instant::now();
    let compressed = kivi::kivi_quantize_keys(data, n, d, bits);
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    let recon = compressed.dequantize();
    let mut total_mse = 0.0f64;
    let mut total_cos = 0.0f64;

    for v in 0..n {
        let offset = v * d;
        let orig = &data[offset..offset + d];
        let rec = &recon[offset..offset + d];
        total_mse += orig
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        total_cos += cosine_sim(orig, rec);
    }

    let total_bits = compressed.total_bits();
    let fp16_bits = n * d * 16;

    MethodResult {
        method: "kivi".into(),
        dim: d,
        bits,
        n_vectors: n,
        mse_per_coord: total_mse / n as f64,
        cosine_mean: total_cos / n as f64,
        quantize_ms: elapsed,
        throughput_kvec_per_sec: n as f64 / elapsed,
        total_bits,
        fp16_bits,
        compression_ratio: fp16_bits as f64 / total_bits as f64,
    }
}

fn bench_nsnquant(data: &[f64], n: usize, d: usize, bits: u32) -> MethodResult {
    // NSNQuant requires power-of-2 dimension for WHT
    if !d.is_power_of_two() {
        return MethodResult {
            method: "nsnquant".into(),
            dim: d,
            bits,
            n_vectors: n,
            mse_per_coord: f64::NAN,
            cosine_mean: f64::NAN,
            quantize_ms: 0.0,
            throughput_kvec_per_sec: 0.0,
            total_bits: 0,
            fp16_bits: 0,
            compression_ratio: 0.0,
        };
    }

    let t0 = Instant::now();
    let mut state = nsnquant::nsn_preprocess(data, n, d);
    nsnquant::nsn_apply_wht(&mut state);
    let compressed = nsnquant::nsn_quantize(&state, bits);
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    let recon = nsnquant::nsn_dequantize(&compressed);
    let mut total_mse = 0.0f64;
    let mut total_cos = 0.0f64;

    for v in 0..n {
        let offset = v * d;
        let orig = &data[offset..offset + d];
        let rec = &recon[offset..offset + d];
        total_mse += orig
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        total_cos += cosine_sim(orig, rec);
    }

    let total_bits = compressed.total_bits();
    let fp16_bits = n * d * 16;

    MethodResult {
        method: "nsnquant".into(),
        dim: d,
        bits,
        n_vectors: n,
        mse_per_coord: total_mse / n as f64,
        cosine_mean: total_cos / n as f64,
        quantize_ms: elapsed,
        throughput_kvec_per_sec: n as f64 / elapsed,
        total_bits,
        fp16_bits,
        compression_ratio: fp16_bits as f64 / total_bits as f64,
    }
}

fn bench_group_quant(data: &[f64], n: usize, d: usize, bits: u32) -> MethodResult {
    let group_size = 32;
    let t0 = Instant::now();

    let mut total_mse = 0.0f64;
    let mut total_cos = 0.0f64;
    let mut total_meta_bits = 0usize;

    for v in 0..n {
        let offset = v * d;
        let x = &data[offset..offset + d];
        let params = grouping::compute_group_params(x, group_size, bits);
        let indices = grouping::group_quantize(x, &params, bits);
        let recon = grouping::group_dequantize(&indices, &params, bits);

        total_mse += x
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        total_cos += cosine_sim(x, &recon);
        total_meta_bits += params.n_groups * (32 + 32 + 8); // scale + zp + flag per group
    }
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    let index_bits = n * d * bits as usize;
    let total_bits = index_bits + total_meta_bits;
    let fp16_bits = n * d * 16;

    MethodResult {
        method: format!("group-g{}", group_size),
        dim: d,
        bits,
        n_vectors: n,
        mse_per_coord: total_mse / n as f64,
        cosine_mean: total_cos / n as f64,
        quantize_ms: elapsed,
        throughput_kvec_per_sec: n as f64 / elapsed,
        total_bits,
        fp16_bits,
        compression_ratio: fp16_bits as f64 / total_bits as f64,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("=== TurboQuant Head-to-Head Comparison ===");
    println!("  Methods: TurboQuant(recommended), KIVI, NSNQuant, GroupQuant");
    println!("  Vectors: {}", cli.n_vectors);

    let mut results = Vec::new();

    for &d in &cli.dims {
        let data = random_matrix(cli.n_vectors, d, 42);

        for &bits in &cli.bits {
            println!("\n  d={}, bits={}:", d, bits);

            let tq = bench_turboquant(&data, cli.n_vectors, d, bits);
            println!(
                "    TurboQuant:  MSE={:.6}  cos={:.4}  {:.0} kvec/s  ratio={:.1}x",
                tq.mse_per_coord, tq.cosine_mean, tq.throughput_kvec_per_sec, tq.compression_ratio
            );
            results.push(tq);

            let kv = bench_kivi(&data, cli.n_vectors, d, bits);
            println!(
                "    KIVI:        MSE={:.6}  cos={:.4}  {:.0} kvec/s  ratio={:.1}x",
                kv.mse_per_coord, kv.cosine_mean, kv.throughput_kvec_per_sec, kv.compression_ratio
            );
            results.push(kv);

            let nsn = bench_nsnquant(&data, cli.n_vectors, d, bits);
            if !nsn.mse_per_coord.is_nan() {
                println!(
                    "    NSNQuant:    MSE={:.6}  cos={:.4}  {:.0} kvec/s  ratio={:.1}x",
                    nsn.mse_per_coord,
                    nsn.cosine_mean,
                    nsn.throughput_kvec_per_sec,
                    nsn.compression_ratio
                );
            }
            results.push(nsn);

            let gq = bench_group_quant(&data, cli.n_vectors, d, bits);
            println!(
                "    GroupQuant:  MSE={:.6}  cos={:.4}  {:.0} kvec/s  ratio={:.1}x",
                gq.mse_per_coord, gq.cosine_mean, gq.throughput_kvec_per_sec, gq.compression_ratio
            );
            results.push(gq);
        }
    }

    let output = ComparisonOutput {
        configs_tested: cli.dims.len() * cli.bits.len(),
        methods_per_config: 4,
        results,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("\n  Wrote {}", cli.out_json.display());

    Ok(())
}
