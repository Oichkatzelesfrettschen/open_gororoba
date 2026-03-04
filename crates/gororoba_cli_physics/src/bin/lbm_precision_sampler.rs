// LBM Multi-Precision Sampler (Sprint 50B, E-069)
//
// Runs identical D3Q19 LBM with Kolmogorov forcing at four precision modes:
//   GPU BF16, GPU FP32, GPU FP64, CPU FP64
// and compares kinetic energy traces to determine whether ghost spectral
// artifacts are precision-dependent.
//
// WHY: Sprint 49B falsified C-792/C-793 (ghost peaks in BF16 = quantization
// noise). This experiment closes the loop by comparing across all available
// precisions on the same hardware.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[cfg(feature = "gpu")]
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};

use lbm_3d::solver::LbmSolver3D;

#[derive(Parser)]
#[command(name = "lbm-precision-sampler")]
#[command(about = "Multi-precision LBM comparison (BF16/FP32/FP64-GPU/FP64-CPU)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all precision modes and produce comparison CSV
    Sample {
        /// Grid resolution (cubic: NxNxN)
        #[arg(long, default_value_t = 32)]
        res: usize,

        /// Number of LBM timesteps
        #[arg(long, default_value_t = 500)]
        steps: usize,

        /// Relaxation time tau (>= 0.505 for BF16 stability)
        #[arg(long, default_value_t = 0.505)]
        tau: f64,

        /// Kolmogorov forcing amplitude
        #[arg(long, default_value_t = 0.01)]
        force_amp: f64,

        /// KE sampling interval (steps)
        #[arg(long, default_value_t = 20)]
        sample_every: usize,

        /// Output directory
        #[arg(long, default_value = "data/csv")]
        out_dir: PathBuf,

        /// Skip GPU modes (CPU-only run)
        #[arg(long)]
        cpu_only: bool,
    },

    /// Compare existing CSV files and print summary
    Compare {
        /// Input directory containing precision CSVs
        #[arg(long, default_value = "data/csv")]
        dir: PathBuf,
    },
}

/// Kinetic energy sample point
struct KeSample {
    step: usize,
    ke: f64,
    wall_ms: f64,
}

/// Build Kolmogorov sinusoidal forcing field: F_x = A * sin(2*pi*z/N)
fn kolmogorov_force(n: usize, amplitude: f64) -> Vec<[f64; 3]> {
    let mut force = vec![[0.0, 0.0, 0.0]; n * n * n];
    let pi2 = std::f64::consts::PI * 2.0;
    for z in 0..n {
        let fz = amplitude * (pi2 * z as f64 / n as f64).sin();
        for y in 0..n {
            for x in 0..n {
                let idx = z * n * n + y * n + x;
                force[idx] = [fz, 0.0, 0.0];
            }
        }
    }
    force
}

/// Compute kinetic energy from f32 host buffers (GPU path)
fn compute_ke_f32(rho: &[f32], u: &[[f32; 3]]) -> f64 {
    let mut ke = 0.0_f64;
    for (r, v) in rho.iter().zip(u.iter()) {
        let r64 = *r as f64;
        let u_sq = (v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2);
        ke += r64 * u_sq;
    }
    ke / rho.len() as f64
}

/// Compute kinetic energy from f64 host buffers (CPU path)
fn compute_ke_f64(rho: &[f64], u: &[[f64; 3]]) -> f64 {
    let mut ke = 0.0_f64;
    for (r, v) in rho.iter().zip(u.iter()) {
        let u_sq = v[0].powi(2) + v[1].powi(2) + v[2].powi(2);
        ke += r * u_sq;
    }
    ke / rho.len() as f64
}

/// Run GPU precision mode and collect KE trace
#[cfg(feature = "gpu")]
fn run_gpu(
    label: &str,
    precision: Precision,
    res: usize,
    steps: usize,
    tau: f64,
    force: &[[f64; 3]],
    sample_every: usize,
) -> Result<Vec<KeSample>> {
    eprintln!("[{label}] Initializing {}^3 grid...", res);
    let mut solver = LbmSolver3DCuda::new(res, res, res, tau, precision)
        .with_context(|| format!("Failed to create GPU solver for {label}"))?;

    solver
        .set_force_field(force)
        .context("set_force_field failed")?;

    let mut samples = Vec::new();
    let mut total_wall = 0.0_f64;

    // Initial KE
    solver.sync_to_host().context("initial sync failed")?;
    let ke0 = compute_ke_f32(&solver.rho, &solver.u);
    samples.push(KeSample {
        step: 0,
        ke: ke0,
        wall_ms: 0.0,
    });

    for s in 1..=steps {
        let t0 = Instant::now();
        solver.step().context("step failed")?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        total_wall += dt;

        if s % sample_every == 0 || s == steps {
            solver.sync_to_host().context("sync failed")?;
            let ke = compute_ke_f32(&solver.rho, &solver.u);
            samples.push(KeSample {
                step: s,
                ke,
                wall_ms: total_wall / s as f64,
            });
        }
    }

    eprintln!(
        "[{label}] Done. {} steps, {:.2} ms/step avg, final KE = {:.6e}",
        steps,
        total_wall / steps as f64,
        samples.last().unwrap().ke
    );
    Ok(samples)
}

/// Run CPU FP64 mode and collect KE trace
fn run_cpu_fp64(
    res: usize,
    steps: usize,
    tau: f64,
    force: &[[f64; 3]],
    sample_every: usize,
) -> Result<Vec<KeSample>> {
    eprintln!("[FP64 CPU] Initializing {}^3 grid...", res);
    let mut solver = LbmSolver3D::new(res, res, res, tau);

    // Set uniform tau field (required for spatial viscosity)
    let n_cells = res * res * res;
    let tau_field = vec![tau; n_cells];
    solver
        .set_viscosity_field(tau_field)
        .map_err(|e| anyhow::anyhow!(e))?;

    // Set force field
    solver
        .set_force_field(force.to_vec())
        .map_err(|e| anyhow::anyhow!(e))?;

    let mut samples = Vec::new();
    let mut total_wall = 0.0_f64;

    // Initial KE
    let ke0 = compute_ke_f64(&solver.rho, &solver.u);
    samples.push(KeSample {
        step: 0,
        ke: ke0,
        wall_ms: 0.0,
    });

    for s in 1..=steps {
        let t0 = Instant::now();
        solver.evolve_one_step();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        total_wall += dt;

        if s % sample_every == 0 || s == steps {
            let ke = compute_ke_f64(&solver.rho, &solver.u);
            samples.push(KeSample {
                step: s,
                ke,
                wall_ms: total_wall / s as f64,
            });
        }
    }

    eprintln!(
        "[FP64 CPU] Done. {} steps, {:.2} ms/step avg, final KE = {:.6e}",
        steps,
        total_wall / steps as f64,
        samples.last().unwrap().ke
    );
    Ok(samples)
}

/// Write KE trace to CSV
fn write_csv(
    path: &Path,
    label: &str,
    samples: &[KeSample],
    ref_final_ke: Option<f64>,
) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "precision,step,ke,wall_ms_per_step,ke_rel_error_vs_ref")?;
    for s in samples {
        let rel_err = match ref_final_ke {
            Some(ref_ke) if ref_ke.abs() > 1e-30 => (s.ke - ref_ke).abs() / ref_ke.abs(),
            _ => 0.0,
        };
        writeln!(
            f,
            "{},{},{:.12e},{:.4},{:.6e}",
            label, s.step, s.ke, s.wall_ms, rel_err
        )?;
    }
    Ok(())
}

fn run_sample(
    res: usize,
    steps: usize,
    tau: f64,
    force_amp: f64,
    sample_every: usize,
    out_dir: &Path,
    cpu_only: bool,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let force = kolmogorov_force(res, force_amp);

    // Results storage: (label, samples)
    let mut all_results: Vec<(String, Vec<KeSample>)> = Vec::new();

    // GPU modes (skip if --cpu-only)
    #[cfg(feature = "gpu")]
    if !cpu_only {
        let modes = [
            ("bf16_gpu", Precision::BF16),
            ("fp32_gpu", Precision::FP32),
            ("fp64_gpu", Precision::FP64),
        ];
        for (label, prec) in &modes {
            match run_gpu(label, *prec, res, steps, tau, &force, sample_every) {
                Ok(samples) => all_results.push((label.to_string(), samples)),
                Err(e) => eprintln!("[{label}] FAILED: {e:#}"),
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    if !cpu_only {
        eprintln!("GPU feature not enabled, skipping GPU modes");
    }

    // CPU FP64
    match run_cpu_fp64(res, steps, tau, &force, sample_every) {
        Ok(samples) => all_results.push(("fp64_cpu".to_string(), samples)),
        Err(e) => eprintln!("[FP64 CPU] FAILED: {e}"),
    }

    if all_results.is_empty() {
        anyhow::bail!("No precision modes completed successfully");
    }

    // Use FP64 GPU as reference if available, else FP64 CPU
    let ref_label = if all_results.iter().any(|(l, _)| l == "fp64_gpu") {
        "fp64_gpu"
    } else {
        "fp64_cpu"
    };
    let ref_final_ke = all_results
        .iter()
        .find(|(l, _)| l == ref_label)
        .and_then(|(_, s)| s.last())
        .map(|s| s.ke);

    // Write individual CSVs
    for (label, samples) in &all_results {
        let path = out_dir.join(format!("precision_{}_{}.csv", label, res));
        write_csv(&path, label, samples, ref_final_ke)?;
        eprintln!("Wrote {}", path.display());
    }

    // Write combined CSV
    let combined_path = out_dir.join(format!("precision_combined_{}.csv", res));
    {
        let mut f = std::fs::File::create(&combined_path)?;
        writeln!(f, "precision,step,ke,wall_ms_per_step,ke_rel_error_vs_ref")?;
        for (label, samples) in &all_results {
            for s in samples {
                let rel_err = match ref_final_ke {
                    Some(ref_ke) if ref_ke.abs() > 1e-30 => (s.ke - ref_ke).abs() / ref_ke.abs(),
                    _ => 0.0,
                };
                writeln!(
                    f,
                    "{},{},{:.12e},{:.4},{:.6e}",
                    label, s.step, s.ke, s.wall_ms, rel_err
                )?;
            }
        }
    }
    eprintln!("Wrote {}", combined_path.display());

    // Print summary table
    println!();
    println!(
        "{:<15} {:>14} {:>12} {:>20}",
        "Precision", "Final KE", "ms/step", "rel_error_vs_ref"
    );
    println!("{}", "-".repeat(65));
    for (label, samples) in &all_results {
        if let Some(last) = samples.last() {
            let rel_err = match ref_final_ke {
                Some(ref_ke) if ref_ke.abs() > 1e-30 => (last.ke - ref_ke).abs() / ref_ke.abs(),
                _ => 0.0,
            };
            let marker = if label == ref_label {
                " (reference)"
            } else {
                ""
            };
            println!(
                "{:<15} {:>14.6e} {:>12.2} {:>20.6e}{}",
                label, last.ke, last.wall_ms, rel_err, marker
            );
        }
    }
    println!();

    // Verdict
    if let Some(ref_ke) = ref_final_ke {
        let mut all_ok = true;
        for (label, samples) in &all_results {
            if let Some(last) = samples.last() {
                let rel_err = (last.ke - ref_ke).abs() / ref_ke.abs().max(1e-30);
                let threshold = match label.as_str() {
                    "bf16_gpu" => 1e-1,  // BF16: 8-bit mantissa, generous threshold
                    "fp32_gpu" => 1e-3,  // FP32: 23-bit mantissa
                    "fp64_cpu" => 1e-10, // FP64 CPU vs GPU: near-identical
                    _ => 1e-3,
                };
                if rel_err > threshold {
                    println!(
                        "WARNING: {} rel_error {:.2e} exceeds threshold {:.2e}",
                        label, rel_err, threshold
                    );
                    all_ok = false;
                }
            }
        }
        if all_ok {
            println!("PASS: All precisions converge within expected tolerances");
        }
    }

    Ok(())
}

fn run_compare(dir: &Path) -> Result<()> {
    // Find all precision_*_*.csv files and read final KE
    let pattern = format!("{}/**/precision_*_*.csv", dir.display());
    let paths: Vec<PathBuf> = glob::glob(&pattern)?
        .filter_map(|p| p.ok())
        .filter(|p| !p.to_string_lossy().contains("combined"))
        .collect();

    if paths.is_empty() {
        anyhow::bail!("No precision CSV files found in {}", dir.display());
    }

    println!("Found {} precision CSV files:", paths.len());
    for p in &paths {
        println!("  {}", p.display());
    }
    println!();

    // Read each file and extract last row
    println!("{:<40} {:>14} {:>12}", "File", "Final KE", "ms/step");
    println!("{}", "-".repeat(70));
    for p in &paths {
        let content = std::fs::read_to_string(p)?;
        if let Some(last_line) = content.lines().last() {
            let parts: Vec<&str> = last_line.split(',').collect();
            if parts.len() >= 4 {
                let ke = parts[2];
                let ms = parts[3];
                println!(
                    "{:<40} {:>14} {:>12}",
                    p.file_name().unwrap().to_string_lossy(),
                    ke,
                    ms
                );
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Sample {
            res,
            steps,
            tau,
            force_amp,
            sample_every,
            out_dir,
            cpu_only,
        } => run_sample(res, steps, tau, force_amp, sample_every, &out_dir, cpu_only),
        Commands::Compare { dir } => run_compare(&dir),
    }
}
