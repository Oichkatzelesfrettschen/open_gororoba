//! ZD Resonance Sweep: Falsification of Sedenion zero-divisor spectral locking.
//!
//! Tests whether the Sedenion ZD-modulated viscosity produces a spectral peak
//! at the ghost frequency (phi^{-1/2}) that is INDEPENDENT of the LBM relaxation
//! time tau (i.e., independent of Reynolds number).
//!
//! Three-dimensional falsification:
//! - tau sweep WITH ZD: Is the peak Re-independent?
//! - tau sweep WITHOUT ZD (control): Does the peak require ZD modulation?
//! - lambda sweep: Does the peak track seeding frequency?

use ash::vk;
use clap::{Parser, Subcommand};
use lbm_vulkan::compute::GororobaEngine;
use lbm_vulkan::VulkanContext;
use spectral_core::ghost_spectral::{
    check_ghost, compute_power_spectrum, find_peaks, peak_fwhm, GHOST_FREQ,
};
use std::io::Write;

#[derive(Parser)]
#[command(name = "zd-resonance-sweep", about = "ZD resonance falsification sweep")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Tau sweep WITH ZD modulation (tau_amp = 0.2).
    Sweep {
        /// Grid dimension (N^3).
        #[arg(long, default_value = "64")]
        grid: u32,
        /// Number of LBM steps per tau value.
        #[arg(long, default_value = "500")]
        steps: u32,
        /// Output CSV path.
        #[arg(long, default_value = "data/csv/zd_resonance_sweep.csv")]
        output: String,
    },
    /// Control sweep WITHOUT ZD modulation (tau_amp = 0.0).
    Control {
        #[arg(long, default_value = "64")]
        grid: u32,
        #[arg(long, default_value = "500")]
        steps: u32,
        #[arg(long, default_value = "data/csv/zd_resonance_control.csv")]
        output: String,
    },
    /// Fix tau, vary spatial seeding frequency via pc.lambda.
    SeedingSweep {
        #[arg(long, default_value = "64")]
        grid: u32,
        #[arg(long, default_value = "500")]
        steps: u32,
        #[arg(long, default_value = "0.55")]
        tau: f32,
        #[arg(long, default_value = "data/csv/zd_seeding_sweep.csv")]
        output: String,
    },
    /// Single-tau run, export full rho_mean time series for rho-ghost-fft.
    RhoTrace {
        #[arg(long, default_value = "64")]
        grid: u32,
        #[arg(long, default_value = "500")]
        steps: u32,
        #[arg(long, default_value = "0.55")]
        tau: f32,
        #[arg(long, default_value = "0.2")]
        tau_amp: f32,
        #[arg(long, default_value = "data/csv/zd_rho_trace.csv")]
        output: String,
    },
    /// Analyze sweep CSV and emit verdict.
    Analyze {
        /// Path to the sweep CSV file.
        #[arg(long, default_value = "data/csv/zd_resonance_sweep.csv")]
        sweep: String,
        /// Path to the control CSV file.
        #[arg(long, default_value = "data/csv/zd_resonance_control.csv")]
        control: String,
    },
}

/// Run LBM for `steps` steps and collect mean density per step.
fn run_sweep_point(
    ctx: &VulkanContext,
    grid: u32,
    steps: u32,
    tau_base: f32,
    tau_amp: f32,
    lambda: f32,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let grid_dim = (grid, grid, grid);
    let mut engine = GororobaEngine::new(ctx, grid_dim, (64, 64))?;

    // Initialize with uniform equilibrium + small central perturbation
    let n = (grid * grid * grid) as usize;
    let wf: [f32; 19] = [
        0.33333333, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556,
        0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
        0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
    ];
    let mut f_init = vec![0.0f32; n * 19];
    let mut force = vec![0.0f32; n * 3];
    let center = grid as f32 / 2.0;
    for z in 0..grid {
        for y in 0..grid {
            for x in 0..grid {
                let idx = (x + grid * (y + grid * z)) as usize;
                f_init[idx * 19..idx * 19 + 19].copy_from_slice(&wf);
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dz = z as f32 - center;
                let r_sq = dx * dx + dy * dy + dz * dz + 1.0;
                let mag = 50.0 / r_sq;
                force[idx * 3] = -mag * dx / r_sq.sqrt();
                force[idx * 3 + 1] = -mag * dy / r_sq.sqrt();
                force[idx * 3 + 2] = -mag * dz / r_sq.sqrt();
            }
        }
    }
    engine.upload_initial_state(ctx, &f_init, &force)?;

    // Command pool + buffer + fence
    let pool_info = vk::CommandPoolCreateInfo {
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };
    let pool = unsafe { ctx.device.create_command_pool(&pool_info, None) }?;
    let alloc_info = vk::CommandBufferAllocateInfo {
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    let cmd = unsafe { ctx.device.allocate_command_buffers(&alloc_info) }?[0];
    let fence_info = vk::FenceCreateInfo {
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };
    let fence = unsafe { ctx.device.create_fence(&fence_info, None) }?;

    let mut rho_means = Vec::with_capacity(steps as usize);
    for frame in 0..steps {
        unsafe {
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            ctx.device.reset_fences(&[fence])?;
            ctx.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            ctx.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
        }

        engine.step_with_params(cmd, frame, tau_base, tau_amp, lambda);

        unsafe {
            ctx.device.end_command_buffer(cmd)?;
            ctx.device.queue_submit(
                ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                fence,
            )?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        let rho = engine.read_rho_field();
        let mean = rho.iter().map(|&v| v as f64).sum::<f64>() / rho.len() as f64;
        rho_means.push(mean);
    }

    // Cleanup
    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }

    Ok(rho_means)
}

fn analyze_rho_series(rho_means: &[f64]) -> (f64, f64, f64, usize) {
    let (freqs, power) = compute_power_spectrum(rho_means);
    let peaks = find_peaks(&freqs, &power, 5);
    let ghost = check_ghost(&peaks);

    let (peak_freq, peak_power, ghost_rank) = if let Some(gp) = ghost {
        (gp.freq, gp.power, gp.rank)
    } else if let Some(top) = peaks.first() {
        (top.freq, top.power, 0)
    } else {
        (0.0, 0.0, 0)
    };

    let fwhm = peak_fwhm(&freqs, &power, peak_freq).unwrap_or(0.0);

    (peak_freq, peak_power, fwhm, ghost_rank)
}

fn run_tau_sweep(
    grid: u32,
    steps: u32,
    tau_amp: f32,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VulkanContext::new(true)?;
    let tau_values: Vec<f32> = vec![0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00];
    let lambda = 4.0f32; // Standard seeding frequency

    let mut file = std::fs::File::create(output)?;
    writeln!(
        file,
        "tau,tau_amp,re_eff,peak_freq,peak_power,fwhm,ghost_rank"
    )?;

    for &tau in &tau_values {
        let re_eff = (tau as f64 - 0.5) / 3.0;
        println!(
            "  tau={tau:.2}, tau_amp={tau_amp:.1}, Re_eff={re_eff:.4} ...",
        );
        let rho_means = run_sweep_point(&ctx, grid, steps, tau, tau_amp, lambda)?;
        let (peak_freq, peak_power, fwhm, ghost_rank) = analyze_rho_series(&rho_means);
        writeln!(
            file,
            "{tau},{tau_amp},{re_eff:.6},{peak_freq:.6},{peak_power:.6e},{fwhm:.6},{ghost_rank}"
        )?;
        println!(
            "    peak_freq={peak_freq:.4}, power={peak_power:.2e}, ghost_rank={ghost_rank}"
        );
    }

    println!("Wrote {output}");
    Ok(())
}

fn run_seeding_sweep(
    grid: u32,
    steps: u32,
    tau: f32,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VulkanContext::new(true)?;
    let lambda_values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0];

    let mut file = std::fs::File::create(output)?;
    writeln!(
        file,
        "tau,lambda,peak_freq,peak_power,fwhm,ghost_rank"
    )?;

    for &lambda in &lambda_values {
        println!("  tau={tau:.2}, lambda={lambda:.1} ...");
        let rho_means = run_sweep_point(&ctx, grid, steps, tau, 0.2, lambda)?;
        let (peak_freq, peak_power, fwhm, ghost_rank) = analyze_rho_series(&rho_means);
        writeln!(
            file,
            "{tau},{lambda},{peak_freq:.6},{peak_power:.6e},{fwhm:.6},{ghost_rank}"
        )?;
        println!(
            "    peak_freq={peak_freq:.4}, power={peak_power:.2e}, ghost_rank={ghost_rank}"
        );
    }

    println!("Wrote {output}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Sweep { grid, steps, output } => {
            println!("=== ZD Resonance Sweep (tau_amp=0.2) ===");
            println!("Grid: {grid}^3, Steps: {steps}");
            println!("Ghost freq target: {GHOST_FREQ:.6}");
            run_tau_sweep(grid, steps, 0.2, &output)?;
        }
        Cmd::Control { grid, steps, output } => {
            println!("=== Control Sweep (tau_amp=0.0, NO ZD modulation) ===");
            println!("Grid: {grid}^3, Steps: {steps}");
            run_tau_sweep(grid, steps, 0.0, &output)?;
        }
        Cmd::SeedingSweep {
            grid,
            steps,
            tau,
            output,
        } => {
            println!("=== Seeding Frequency Sweep (tau={tau:.2}) ===");
            run_seeding_sweep(grid, steps, tau, &output)?;
        }
        Cmd::RhoTrace {
            grid,
            steps,
            tau,
            tau_amp,
            output,
        } => {
            println!("=== Rho Trace (tau={tau:.2}, tau_amp={tau_amp:.1}) ===");
            let ctx = VulkanContext::new(true)?;
            let rho_means = run_sweep_point(&ctx, grid, steps, tau, tau_amp, 4.0)?;
            let mut file = std::fs::File::create(&output)?;
            writeln!(file, "step,rho_mean")?;
            for (i, &rho) in rho_means.iter().enumerate() {
                writeln!(file, "{i},{rho:.10}")?;
            }
            println!("Wrote {output} ({} steps)", rho_means.len());

            let (peak_freq, peak_power, fwhm, ghost_rank) = analyze_rho_series(&rho_means);
            println!("Peak: freq={peak_freq:.4}, power={peak_power:.2e}, FWHM={fwhm:.4}, ghost_rank={ghost_rank}");
        }
        Cmd::Analyze { sweep, control } => {
            println!("=== Analyze ZD Resonance Results ===");
            analyze_sweep_results(&sweep, &control)?;
        }
    }

    Ok(())
}

fn analyze_sweep_results(
    sweep_path: &str,
    control_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read sweep CSV
    let sweep_data = read_sweep_csv(sweep_path)?;
    let control_data = read_sweep_csv(control_path)?;

    // Check 1: Does the peak appear ONLY with ZD modulation?
    let sweep_ghost_count = sweep_data.iter().filter(|r| r.ghost_rank > 0).count();
    let control_ghost_count = control_data.iter().filter(|r| r.ghost_rank > 0).count();
    println!("Ghost detections: sweep={sweep_ghost_count}/{}, control={control_ghost_count}{}",
        sweep_data.len(), control_data.len());

    // Check 2: Is the peak frequency constant across tau?
    let sweep_freqs: Vec<f64> = sweep_data.iter().map(|r| r.peak_freq).collect();
    let freq_std = if sweep_freqs.len() > 1 {
        let mean = sweep_freqs.iter().sum::<f64>() / sweep_freqs.len() as f64;
        let var = sweep_freqs.iter().map(|f| (f - mean).powi(2)).sum::<f64>()
            / (sweep_freqs.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    // Linear regression: peak_freq vs tau (check for drift)
    let taus: Vec<f64> = sweep_data.iter().map(|r| r.tau).collect();
    let slope = linear_regression_slope(&taus, &sweep_freqs);

    println!("Frequency std: {freq_std:.6}");
    println!("Frequency vs tau slope: {slope:.6}");

    // Verdict
    let zd_only = sweep_ghost_count > 0 && control_ghost_count == 0;
    let freq_stable = slope.abs() < 0.01;

    if zd_only && freq_stable {
        println!("\n==> VERDICT: ZD RESONANCE CONFIRMED (C-774 PASS)");
        println!("    Peak requires ZD modulation and is Re-independent.");
    } else if !zd_only && control_ghost_count > 0 {
        println!("\n==> VERDICT: ZD RESONANCE FALSIFIED (C-774 FAIL)");
        println!("    Ghost peak appears WITHOUT ZD modulation (lattice artifact).");
    } else if slope.abs() >= 0.01 {
        println!("\n==> VERDICT: ZD RESONANCE FALSIFIED (C-774 FAIL)");
        println!("    Peak frequency drifts with tau (fluid artifact, slope={slope:.4}).");
    } else {
        println!("\n==> VERDICT: NULL RESULT (C-774 INCONCLUSIVE)");
        println!("    No ghost detected in sweep or control.");
    }

    Ok(())
}

struct SweepRow {
    tau: f64,
    peak_freq: f64,
    ghost_rank: usize,
}

fn read_sweep_csv(path: &str) -> Result<Vec<SweepRow>, Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let tau: f64 = record[0].parse()?;
        let peak_freq: f64 = record[3].parse()?;
        let ghost_rank: usize = record[6].parse()?;
        rows.push(SweepRow {
            tau,
            peak_freq,
            ghost_rank,
        });
    }
    Ok(rows)
}

fn linear_regression_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let x_mean = xs.iter().sum::<f64>() / n;
    let y_mean = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - x_mean) * (y - y_mean)).sum();
    let den: f64 = xs.iter().map(|x| (x - x_mean).powi(2)).sum();
    if den.abs() < 1e-15 {
        0.0
    } else {
        num / den
    }
}
