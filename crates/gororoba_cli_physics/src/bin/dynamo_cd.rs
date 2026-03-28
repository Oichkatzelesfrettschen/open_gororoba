//! Liquid metal dynamo CD analysis: synthetic VKS-like (Von Karman Sodium)
//! magnetic field time series through dynamo onset and reversal.
//!
//! The VKS experiment (Monchaux et al. 2007) observed:
//! 1. Sub-critical: no sustained field, small fluctuations
//! 2. Dynamo onset: exponential growth of B, Rm > Rm_crit
//! 3. Stationary dynamo: sustained dipolar field
//! 4. Reversal: polarity flip (like Earth's geomagnetic reversals)
//! 5. Chaotic: intermittent bursts between states
//!
//! The dynamo transition is controlled by the magnetic Reynolds number Rm.

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::{f64::consts::PI, fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "dynamo-cd")]
struct Cli {
    /// Samples per phase.
    #[arg(long, default_value_t = 50000)]
    samples_per_phase: usize,

    /// Embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/dynamo_cd.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct PhaseResult {
    phase: String,
    rm_value: f64,
    mean_a: f64,
    max_a: f64,
    mean_b_magnitude: f64,
}

#[derive(Debug, Serialize)]
struct DynamoOutput {
    samples_per_phase: usize,
    embedding_dim: usize,
    phases: Vec<PhaseResult>,
    interpretation: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== Liquid Metal Dynamo CD Analysis (synthetic VKS) ===");

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = cli.samples_per_phase;
    let dt = 0.001; // 1 kHz sampling

    // Generate 3-component B-field (Bx, By, Bz) for each dynamo phase
    let phases: Vec<(&str, f64, Vec<[f64; 3]>)> = vec![
        // Phase 1: Sub-critical (Rm < Rm_crit, no dynamo)
        ("subcritical", 20.0, {
            (0..n)
                .map(|_| {
                    [
                        0.01 * rng.gen_range(-1.0..1.0),
                        0.01 * rng.gen_range(-1.0..1.0),
                        0.01 * rng.gen_range(-1.0..1.0),
                    ]
                })
                .collect()
        }),
        // Phase 2: Dynamo onset (Rm ~ Rm_crit, exponential growth)
        ("onset", 32.0, {
            (0..n)
                .map(|i| {
                    let t = i as f64 * dt;
                    let growth = (0.5 * t).exp().min(10.0);
                    [
                        growth * (2.0 * PI * 1.5 * t).cos() + 0.1 * rng.gen_range(-1.0..1.0),
                        growth * (2.0 * PI * 1.5 * t + 0.5).sin()
                            + 0.1 * rng.gen_range(-1.0..1.0),
                        growth * 0.3 * (2.0 * PI * 0.8 * t).cos()
                            + 0.05 * rng.gen_range(-1.0..1.0),
                    ]
                })
                .collect()
        }),
        // Phase 3: Stationary dynamo (Rm >> Rm_crit, sustained dipole)
        ("stationary", 50.0, {
            (0..n)
                .map(|i| {
                    let t = i as f64 * dt;
                    // Steady dipole with small fluctuations
                    [
                        5.0 * (2.0 * PI * 0.1 * t).cos() + 0.2 * rng.gen_range(-1.0..1.0),
                        5.0 * (2.0 * PI * 0.1 * t + PI / 4.0).sin()
                            + 0.2 * rng.gen_range(-1.0..1.0),
                        8.0 + 0.3 * rng.gen_range(-1.0..1.0), // strong axial component
                    ]
                })
                .collect()
        }),
        // Phase 4: Reversal (polarity flip over ~10 diffusion times)
        ("reversal", 50.0, {
            (0..n)
                .map(|i| {
                    let t = i as f64 * dt;
                    let t_mid = n as f64 * dt / 2.0;
                    // Tanh transition for polarity flip
                    let polarity = -((t - t_mid) / 5.0).tanh();
                    [
                        polarity * 5.0 * (2.0 * PI * 0.1 * t).cos()
                            + 1.0 * rng.gen_range(-1.0..1.0),
                        polarity * 5.0 * (2.0 * PI * 0.1 * t + PI / 4.0).sin()
                            + 1.0 * rng.gen_range(-1.0..1.0),
                        polarity * 8.0 + 2.0 * rng.gen_range(-1.0..1.0),
                    ]
                })
                .collect()
        }),
        // Phase 5: Chaotic/intermittent (turbulent dynamo, bursts)
        ("chaotic", 45.0, {
            (0..n)
                .map(|i| {
                    let t = i as f64 * dt;
                    // Intermittent bursts with random phase jumps
                    let burst = if (5.0 * t).sin() > 0.7 { 3.0 } else { 0.5 };
                    [
                        burst * rng.gen_range(-1.0..1.0)
                            + 2.0 * (2.0 * PI * 2.3 * t + rng.gen_range(0.0..0.5)).sin(),
                        burst * rng.gen_range(-1.0..1.0)
                            + 2.0 * (2.0 * PI * 1.7 * t + rng.gen_range(0.0..0.5)).cos(),
                        burst * rng.gen_range(-1.0..1.0) + 1.0 * (2.0 * PI * 0.5 * t).sin(),
                    ]
                })
                .collect()
        }),
    ];

    let stride = 5;
    let channels = 3;
    let steps = cli.embedding_dim / channels;
    let mut results = Vec::new();

    for (name, rm, bfield) in &phases {
        let mean_bmag = bfield
            .iter()
            .map(|b| (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt())
            .sum::<f64>()
            / bfield.len() as f64;

        let n_windows = if bfield.len() > steps * stride {
            (bfield.len() - steps * stride) / stride
        } else {
            0
        };

        let mut embedded: Vec<Vec<f64>> = Vec::with_capacity(n_windows);
        for w in 0..n_windows {
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for s in 0..steps {
                let idx = w * stride + s * stride;
                if idx < bfield.len() {
                    v.push(bfield[idx][0]);
                    v.push(bfield[idx][1]);
                    v.push(bfield[idx][2]);
                }
            }
            // Pad remainder
            while v.len() < cli.embedding_dim {
                v.push(0.0);
            }
            v.truncate(cli.embedding_dim);

            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
            embedded.push(v);
        }

        if embedded.len() < 3 {
            continue;
        }

        let norms =
            cd_kernel::batch_sliding_associator_norms_dispatch(&embedded, cli.embedding_dim, if cli.embedding_dim >= 128 { "f32" } else { "f64" });

        let mean_a = norms.iter().sum::<f64>() / norms.len().max(1) as f64;
        let max_a = norms.iter().cloned().fold(0.0f64, f64::max);

        println!(
            "  {}: Rm={:.0}, |B|_mean={:.3}, A_mean={:.6}, A_max={:.6}",
            name, rm, mean_bmag, mean_a, max_a
        );

        results.push(PhaseResult {
            phase: name.to_string(),
            rm_value: *rm,
            mean_a,
            max_a,
            mean_b_magnitude: mean_bmag,
        });
    }

    let interp = format!(
        "VKS-like dynamo 5 phases. Sub A={:.4}, Onset A={:.4}, Stationary A={:.4}, Reversal A={:.4}, Chaotic A={:.4}. Chaotic/Stationary={:.1}x.",
        results.get(0).map(|r| r.mean_a).unwrap_or(0.0),
        results.get(1).map(|r| r.mean_a).unwrap_or(0.0),
        results.get(2).map(|r| r.mean_a).unwrap_or(0.0),
        results.get(3).map(|r| r.mean_a).unwrap_or(0.0),
        results.get(4).map(|r| r.mean_a).unwrap_or(0.0),
        if results.get(2).map(|r| r.mean_a).unwrap_or(0.0) > 1e-15 {
            results.get(4).map(|r| r.mean_a).unwrap_or(0.0) / results.get(2).map(|r| r.mean_a).unwrap_or(1.0)
        } else { 0.0 }
    );
    println!("\n  {}", interp);

    let output = DynamoOutput {
        samples_per_phase: n,
        embedding_dim: cli.embedding_dim,
        phases: results,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}
