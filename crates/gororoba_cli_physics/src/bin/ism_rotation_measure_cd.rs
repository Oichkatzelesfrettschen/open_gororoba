//! ISM Rotation Measure CD analysis: synthetic Galactic Faraday rotation
//! measure (RM) grid through different ISM phases.
//!
//! The rotation measure RM = 0.81 * integral(n_e * B_parallel * dl) traces
//! the line-of-sight magnetic field weighted by electron density.
//! Different ISM phases have distinct RM structure:
//! 1. Warm ionized medium (WIM): smooth, large-scale RM (~10-30 rad/m^2)
//! 2. HII regions: enhanced, structured RM (~50-200 rad/m^2)
//! 3. Supernova remnants: highly variable RM (~100-1000 rad/m^2)
//! 4. Turbulent cascade: Kolmogorov power spectrum in RM fluctuations

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::{f64::consts::PI, fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "ism-rotation-measure-cd")]
struct Cli {
    /// Grid size (n x n pulsars).
    #[arg(long, default_value_t = 64)]
    grid_size: usize,

    /// Embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/ism_rm_cd.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct PhaseResult {
    phase: String,
    mean_rm: f64,
    rm_std: f64,
    mean_a: f64,
    max_a: f64,
}

#[derive(Debug, Serialize)]
struct IsmOutput {
    grid_size: usize,
    embedding_dim: usize,
    phases: Vec<PhaseResult>,
    interpretation: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== ISM Rotation Measure CD Analysis ===");

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = cli.grid_size;

    // Generate RM grids for 4 ISM phases
    let phases: Vec<(&str, Vec<f64>)> = vec![
        // Phase 1: Warm ionized medium (smooth large-scale field)
        ("warm_ionized", {
            let mut rm = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let l = i as f64 / n as f64 * 360.0;
                    let b = (j as f64 / n as f64 - 0.5) * 180.0;
                    // Large-scale Galactic field: RM ~ 20 * cos(l) * cos(b)
                    rm[i * n + j] = 20.0 * (l.to_radians()).cos() * (b.to_radians()).cos()
                        + 3.0 * rng.gen_range(-1.0..1.0);
                }
            }
            rm
        }),
        // Phase 2: HII region (enhanced, bubble structure)
        ("hii_region", {
            let mut rm = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let x = i as f64 / n as f64 - 0.5;
                    let y = j as f64 / n as f64 - 0.5;
                    let r = (x * x + y * y).sqrt();
                    // Shell structure at r=0.2, enhanced RM
                    let shell = 150.0 * (-(r - 0.2).powi(2) / 0.01).exp();
                    rm[i * n + j] = 30.0 + shell + 10.0 * rng.gen_range(-1.0..1.0);
                }
            }
            rm
        }),
        // Phase 3: Supernova remnant (highly variable, filamentary)
        ("snr_filament", {
            let mut rm = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    // Multiple random filaments with high RM
                    let mut val = 50.0;
                    for k in 0..5 {
                        let cx = 0.1 + 0.2 * k as f64;
                        let cy = 0.2 + 0.15 * k as f64;
                        let dx = i as f64 / n as f64 - cx;
                        let dy = j as f64 / n as f64 - cy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        val += 200.0 * (-dist / 0.05).exp() * (1.0 + rng.gen_range(-0.5..0.5));
                    }
                    rm[i * n + j] = val + 30.0 * rng.gen_range(-1.0..1.0);
                }
            }
            rm
        }),
        // Phase 4: Turbulent cascade (Kolmogorov spectrum)
        ("turbulent_cascade", {
            let mut rm = vec![0.0; n * n];
            // Superpose modes with Kolmogorov scaling: P(k) ~ k^(-5/3)
            for mode in 1..20 {
                let k = mode as f64;
                let amp = 50.0 * k.powf(-5.0 / 6.0); // sqrt of P(k)
                let phase_x = rng.gen_range(0.0..2.0 * PI);
                let phase_y = rng.gen_range(0.0..2.0 * PI);
                for i in 0..n {
                    for j in 0..n {
                        let x = i as f64 / n as f64;
                        let y = j as f64 / n as f64;
                        rm[i * n + j] += amp * (2.0 * PI * k * x + phase_x).sin()
                            * (2.0 * PI * k * y + phase_y).cos();
                    }
                }
            }
            // Add noise
            for v in rm.iter_mut() {
                *v += 5.0 * rng.gen_range(-1.0..1.0);
            }
            rm
        }),
    ];

    let mut results = Vec::new();

    for (name, rm_grid) in &phases {
        let mean_rm = rm_grid.iter().sum::<f64>() / rm_grid.len() as f64;
        let var = rm_grid.iter().map(|v| (v - mean_rm).powi(2)).sum::<f64>() / rm_grid.len() as f64;
        let rm_std = var.sqrt();

        // Build Takens embedding from the RM grid
        // Sweep along rows, interleave with column neighbors
        let stride = 2;
        let steps = cli.embedding_dim;
        let total_cells = n * n;
        let n_windows = if total_cells > steps * stride {
            (total_cells - steps * stride) / stride
        } else {
            0
        };

        let mut embedded: Vec<Vec<f64>> = Vec::with_capacity(n_windows);
        for w in 0..n_windows {
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for s in 0..steps {
                let idx = w * stride + s * stride;
                if idx < total_cells {
                    v.push(rm_grid[idx]);
                }
            }
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
            cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);

        let mean_a = norms.iter().sum::<f64>() / norms.len().max(1) as f64;
        let max_a = norms.iter().cloned().fold(0.0f64, f64::max);

        println!(
            "  {}: RM_mean={:.1}, RM_std={:.1}, A_mean={:.6}, A_max={:.6}",
            name, mean_rm, rm_std, mean_a, max_a
        );

        results.push(PhaseResult {
            phase: name.to_string(),
            mean_rm,
            rm_std,
            mean_a,
            max_a,
        });
    }

    let interp = format!(
        "ISM RM 4 phases: WIM A={:.4}, HII A={:.4}, SNR A={:.4}, Turbulent A={:.4}. Turb/WIM={:.1}x.",
        results.get(0).map(|r| r.mean_a).unwrap_or(0.0),
        results.get(1).map(|r| r.mean_a).unwrap_or(0.0),
        results.get(2).map(|r| r.mean_a).unwrap_or(0.0),
        results.get(3).map(|r| r.mean_a).unwrap_or(0.0),
        if results.get(0).map(|r| r.mean_a).unwrap_or(0.0) > 1e-15 {
            results.get(3).map(|r| r.mean_a).unwrap_or(0.0) / results.get(0).map(|r| r.mean_a).unwrap_or(1.0)
        } else { 0.0 }
    );
    println!("\n  {}", interp);

    let output = IsmOutput {
        grid_size: n,
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
