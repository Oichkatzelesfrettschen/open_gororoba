//! TurboQuant definitive sweep: all methods x all bit-widths x all dimensions.
//!
//! Produces a single machine-readable JSON with every quantization method
//! benchmarked under identical conditions.  This is the capstone benchmark.
//!
//! Methods:
//!   - TQ-WHT: TurboQuant with WHT rotation (MSE-only)
//!   - TQ-E8:  TurboQuant with E8 block rotation (d=128 only)
//!   - KIVI:   per-channel keys, per-token values (ICML 2024)
//!   - NSNQuant: normalize-shift-normalize + WHT (arXiv 2505.18231)
//!   - GroupQuant: InnerQ-style per-group scales (arXiv 2602.23200)
//!   - Hybrid: rotation + per-group scales
//!   - Synthesized: auto-selects best method per (bits, dim)
//!
//! Usage:
//!   turboquant-sweep --dims 64,128 --bits 2,3,4 --n-vectors 2000

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf, time::Instant};

use cd_kernel::turboquant::baselines::{kivi, nsnquant};
use cd_kernel::turboquant::grouping;
use cd_kernel::turboquant::hybrid::HybridQuantizer;
use cd_kernel::turboquant::pipeline::TurboQuantMSE;
use cd_kernel::turboquant::synthesized::SynthesizedQuantizer;

#[derive(Parser)]
#[command(name = "turboquant-sweep")]
#[command(about = "Definitive all-methods comparison sweep")]
struct Cli {
    #[arg(long, value_delimiter = ',', default_values_t = vec![64, 128])]
    dims: Vec<usize>,

    #[arg(long, value_delimiter = ',', default_values_t = vec![2, 3, 4])]
    bits: Vec<u32>,

    #[arg(long, default_value_t = 2000)]
    n_vectors: usize,

    #[arg(long, default_value = "data/output/heliosphere/ablations/turboquant_sweep.json")]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct SweepResult {
    method: String,
    dim: usize,
    bits: u32,
    mse: f64,
    cosine: f64,
    throughput_kvec_s: f64,
    rank: usize, // rank within this (dim, bits) group (1 = best MSE)
}

#[derive(Debug, Serialize)]
struct SweepOutput {
    n_vectors: usize,
    total_configs: usize,
    results: Vec<SweepResult>,
    summary: Vec<SweepSummary>,
}

#[derive(Debug, Serialize)]
struct SweepSummary {
    dim: usize,
    bits: u32,
    best_method: String,
    best_mse: f64,
    worst_method: String,
    worst_mse: f64,
    improvement_pct: f64, // best vs worst
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 { 0.0 } else { dot / (na * nb) }
}

fn random_matrix(n: usize, d: usize, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    (0..n * d).map(|_| normal.sample(&mut rng)).collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== TurboQuant Definitive Sweep ===");

    let mut all_results = Vec::new();
    let mut summaries = Vec::new();

    for &d in &cli.dims {
        let data = random_matrix(cli.n_vectors, d, 42);
        let n = cli.n_vectors;

        for &bits in &cli.bits {
            println!("\n  d={}, bits={}:", d, bits);
            let mut config_results: Vec<(String, f64, f64, f64)> = Vec::new();

            // 1. TQ-WHT
            {
                let tq = TurboQuantMSE::new(d, bits, 42, true);
                let mut buf = vec![0.0f64; 2 * d];
                let t0 = Instant::now();
                let mut mse_sum = 0.0f64;
                let mut cos_sum = 0.0f64;
                for v in 0..n {
                    let x = &data[v * d..(v + 1) * d];
                    let comp = tq.quantize(x, &mut buf);
                    let mut recon = vec![0.0f64; d];
                    tq.dequantize(&comp, &mut buf, &mut recon);
                    mse_sum += x.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
                    cos_sum += cosine_sim(x, &recon);
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                config_results.push(("TQ-WHT".into(), mse_sum / n as f64, cos_sum / n as f64, n as f64 / ms));
            }

            // 2. TQ-E8 (d=128 only)
            if d == 128 {
                let mut cfg = cd_kernel::turboquant::config::TurboQuantConfig::recommended(d, bits);
                cfg.rotation = cd_kernel::turboquant::config::RotationMethod::E8Block;
                let tq = TurboQuantMSE::from_config(&cfg, bits, 42);
                let mut buf = vec![0.0f64; 2 * d];
                let t0 = Instant::now();
                let mut mse_sum = 0.0f64;
                let mut cos_sum = 0.0f64;
                for v in 0..n {
                    let x = &data[v * d..(v + 1) * d];
                    let comp = tq.quantize(x, &mut buf);
                    let mut recon = vec![0.0f64; d];
                    tq.dequantize(&comp, &mut buf, &mut recon);
                    mse_sum += x.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
                    cos_sum += cosine_sim(x, &recon);
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                config_results.push(("TQ-E8".into(), mse_sum / n as f64, cos_sum / n as f64, n as f64 / ms));
            }

            // 3. KIVI
            {
                let t0 = Instant::now();
                let comp = kivi::kivi_quantize_keys(&data, n, d, bits);
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let recon = comp.dequantize();
                let mut mse_sum = 0.0f64;
                let mut cos_sum = 0.0f64;
                for v in 0..n {
                    let x = &data[v * d..(v + 1) * d];
                    let r = &recon[v * d..(v + 1) * d];
                    mse_sum += x.iter().zip(r.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
                    cos_sum += cosine_sim(x, r);
                }
                config_results.push(("KIVI".into(), mse_sum / n as f64, cos_sum / n as f64, n as f64 / ms));
            }

            // 4. NSNQuant (power-of-2 dims only)
            if d.is_power_of_two() {
                let t0 = Instant::now();
                let mut state = nsnquant::nsn_preprocess(&data, n, d);
                nsnquant::nsn_apply_wht(&mut state);
                let comp = nsnquant::nsn_quantize(&state, bits);
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let recon = nsnquant::nsn_dequantize(&comp);
                let mut mse_sum = 0.0f64;
                let mut cos_sum = 0.0f64;
                for v in 0..n {
                    let x = &data[v * d..(v + 1) * d];
                    let r = &recon[v * d..(v + 1) * d];
                    mse_sum += x.iter().zip(r.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
                    cos_sum += cosine_sim(x, r);
                }
                config_results.push(("NSNQuant".into(), mse_sum / n as f64, cos_sum / n as f64, n as f64 / ms));
            }

            // 5. GroupQuant
            {
                let group_size = 16;
                let t0 = Instant::now();
                let mut mse_sum = 0.0f64;
                let mut cos_sum = 0.0f64;
                for v in 0..n {
                    let x = &data[v * d..(v + 1) * d];
                    let params = grouping::compute_group_params(x, group_size, bits);
                    let indices = grouping::group_quantize(x, &params, bits);
                    let recon = grouping::group_dequantize(&indices, &params, bits);
                    mse_sum += x.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
                    cos_sum += cosine_sim(x, &recon);
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                config_results.push(("Group-16".into(), mse_sum / n as f64, cos_sum / n as f64, n as f64 / ms));
            }

            // 6. Hybrid
            {
                let hq = HybridQuantizer::new(d, bits, 42, true, 16);
                let mut buf = vec![0.0f64; 2 * d];
                let t0 = Instant::now();
                let mut mse_sum = 0.0f64;
                let mut cos_sum = 0.0f64;
                for v in 0..n {
                    let x = &data[v * d..(v + 1) * d];
                    let comp = hq.quantize(x, &mut buf);
                    let mut recon = vec![0.0f64; d];
                    hq.dequantize(&comp, &mut buf, &mut recon);
                    mse_sum += x.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
                    cos_sum += cosine_sim(x, &recon);
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                config_results.push(("Hybrid-16".into(), mse_sum / n as f64, cos_sum / n as f64, n as f64 / ms));
            }

            // 7. Synthesized
            {
                let sq = SynthesizedQuantizer::new(d, bits, 42);
                let mut buf = vec![0.0f64; 3 * d];
                let t0 = Instant::now();
                let mut mse_sum = 0.0f64;
                let mut cos_sum = 0.0f64;
                for v in 0..n {
                    let x = &data[v * d..(v + 1) * d];
                    let comp = sq.quantize(x, &mut buf);
                    let mut recon = vec![0.0f64; d];
                    sq.dequantize(&comp, &mut buf, &mut recon);
                    mse_sum += x.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
                    cos_sum += cosine_sim(x, &recon);
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                config_results.push(("Synthesized".into(), mse_sum / n as f64, cos_sum / n as f64, n as f64 / ms));
            }

            // Sort by MSE and assign ranks
            config_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for (rank, (method, mse, cosine, throughput)) in config_results.iter().enumerate() {
                let marker = if rank == 0 { " ***BEST***" } else { "" };
                println!("    {:>12}: MSE={:.6}  cos={:.4}  {:.0} kvec/s{}",
                    method, mse, cosine, throughput, marker);

                all_results.push(SweepResult {
                    method: method.clone(),
                    dim: d,
                    bits,
                    mse: *mse,
                    cosine: *cosine,
                    throughput_kvec_s: *throughput,
                    rank: rank + 1,
                });
            }

            // Summary
            let best = &config_results[0];
            let worst = config_results.last().unwrap();
            summaries.push(SweepSummary {
                dim: d,
                bits,
                best_method: best.0.clone(),
                best_mse: best.1,
                worst_method: worst.0.clone(),
                worst_mse: worst.1,
                improvement_pct: (worst.1 - best.1) / worst.1 * 100.0,
            });
        }
    }

    let output = SweepOutput {
        n_vectors: cli.n_vectors,
        total_configs: all_results.len(),
        results: all_results,
        summary: summaries,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("\n  Wrote {}", cli.out_json.display());

    Ok(())
}
