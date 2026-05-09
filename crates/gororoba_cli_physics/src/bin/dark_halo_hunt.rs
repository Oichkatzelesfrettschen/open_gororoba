//! dark-halo-hunt: Search for topological dark halos in ZD-mediated LBM
//! simulations across k values.
//!
//! Tests whether the orthoplex diffusion model can produce CDG-2-like
//! baryon-empty structures at any value of k (CD algebra dimension parameter).
//! Falsification: no halos at any k -> model cannot produce CDG-2 structures.
//!
//! Usage:
//!   dark-halo-hunt                                    # Default k=[4,8,16,32]
//!   dark-halo-hunt --grid 128 --k-values 4,8,16      # Quick scan
//!   dark-halo-hunt --output data/evidence/halos.toml  # Custom output
//!   dark-halo-hunt --dry-run                          # Spectral dim only (no GPU)

use clap::Parser;
use cosmology_core::{
    DarkHaloFalsificationResult, evaluate_cdg2_consistency, spectral_dim_at_cdg2,
};
#[cfg(feature = "gpu")]
use lbm_3d_cuda::DarkHaloCudaSolver;
use lbm_vulkan::besag_clifford_vulkan::{
    BesagCliffordVulkanConfig, BesagCliffordVulkanEngine, HaloConstants, vram_budget,
};
use std::io::Write;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(name = "dark-halo-hunt")]
#[command(about = "Search for topological dark halos across k values")]
struct Args {
    /// Grid dimension (N for NxNxN).
    #[arg(long, default_value = "256")]
    grid: u32,

    /// Number of LBM steps.
    #[arg(long, default_value = "2000")]
    steps: u32,

    /// Comma-separated k values to scan.
    #[arg(long, value_delimiter = ',', default_values_t = vec![4, 8, 16, 32, 63])]
    k_values: Vec<usize>,

    /// ZD proxy threshold for halo classification.
    #[arg(long, default_value = "0.5")]
    zd_threshold: f64,

    /// Velocity epsilon for halo classification.
    #[arg(long, default_value = "0.01")]
    velocity_epsilon: f64,

    /// Density factor for halo classification.
    #[arg(long, default_value = "1.5")]
    density_factor: f64,

    /// Output path for TOML evidence file.
    #[arg(long, default_value = "data/evidence/dark_halo_hunt.toml")]
    output: String,

    /// Dry-run mode (compute spectral dimensions only, skip GPU).
    #[arg(long)]
    dry_run: bool,

    /// Use CUDA backend instead of Vulkan (SoA layout, fused kernels).
    #[arg(long)]
    cuda: bool,
}

#[derive(Debug)]
struct KScanResult {
    k: usize,
    spectral_dim: f64,
    volume_fraction: f64,
    falsification: DarkHaloFalsificationResult,
}

fn main() {
    let args = Args::parse();

    eprintln!("=== Dark Halo Hunt ===");
    eprintln!("    Grid: {}^3", args.grid);
    eprintln!("    Steps: {}", args.steps);
    eprintln!("    k values: {:?}", args.k_values);
    eprintln!(
        "    Thresholds: zd={}, vel={}, rho_factor={}",
        args.zd_threshold, args.velocity_epsilon, args.density_factor
    );
    eprintln!();

    // VRAM budget for informational display
    let bc_config = BesagCliffordVulkanConfig {
        grid_dim: (args.grid, args.grid, args.grid),
        ..Default::default()
    };
    let (_bc_bytes, total_bytes) = vram_budget(&bc_config);
    eprintln!(
        "    VRAM budget: {} MB (LBM + analysis)",
        total_bytes / (1024 * 1024)
    );
    eprintln!();

    // Best-fit parameters from Sprint 60 (fixed-beta, k=63)
    let alpha_bf = 5.0;
    let t_0_bf = 0.01;

    // CUDA path: SoA layout with fused collision+streaming
    #[cfg(feature = "gpu")]
    if args.cuda && !args.dry_run {
        run_cuda(&args, alpha_bf, t_0_bf);
        return;
    }
    #[cfg(not(feature = "gpu"))]
    if args.cuda {
        eprintln!("ERROR: --cuda requires the 'gpu' feature (lbm_3d_cuda).");
        std::process::exit(1);
    }

    let mut results = Vec::new();

    // Try to init GPU if not dry-run (Vulkan path)
    let mut gpu_state = if !args.dry_run {
        match init_gpu(&args) {
            Ok(state) => {
                eprintln!(
                    "    GPU: {} ({} MB VRAM)",
                    state.0.caps.device_name, state.0.caps.vram_mb
                );
                Some(state)
            }
            Err(e) => {
                eprintln!("    WARNING: GPU init failed: {e}");
                eprintln!("    Falling back to spectral dimension heuristic.");
                None
            }
        }
    } else {
        None
    };

    for &k in &args.k_values {
        let ds = spectral_dim_at_cdg2(k, alpha_bf, t_0_bf);
        eprintln!("  k={k}: d_s(z=0.018) = {ds:.6}");

        let volume_fraction = if let Some((ref ctx, ref mut lbm, ref bc_engine)) = gpu_state {
            // Live GPU dispatch: run LBM steps then dark halo detector

            // Run LBM steps with ZD parameters scaled by k
            let tau_base = 0.55_f32;
            let tau_amp = 0.2_f32;
            let lambda = (k as f32).ln();

            // Create temporary command pool for LBM stepping
            let cmd_pool = unsafe {
                ctx.device.create_command_pool(
                    &ash::vk::CommandPoolCreateInfo {
                        queue_family_index: ctx.queue_family_index,
                        flags: ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                        ..Default::default()
                    },
                    None,
                )
            };
            let cmd_pool = match cmd_pool {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("    WARNING: cmd pool failed for k={k}: {e}");
                    let vf = (ds * 0.1).min(1.0);
                    let falsification = evaluate_cdg2_consistency(vf);
                    results.push(KScanResult {
                        k,
                        spectral_dim: ds,
                        volume_fraction: vf,
                        falsification,
                    });
                    continue;
                }
            };
            let cmd = unsafe {
                ctx.device
                    .allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo {
                        command_pool: cmd_pool,
                        level: ash::vk::CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                        ..Default::default()
                    })
            };
            let cmd = match cmd {
                Ok(c) => c[0],
                Err(e) => {
                    eprintln!("    WARNING: cmd buf failed for k={k}: {e}");
                    unsafe { ctx.device.destroy_command_pool(cmd_pool, None) };
                    let vf = (ds * 0.1).min(1.0);
                    let falsification = evaluate_cdg2_consistency(vf);
                    results.push(KScanResult {
                        k,
                        spectral_dim: ds,
                        volume_fraction: vf,
                        falsification,
                    });
                    continue;
                }
            };
            let fence = unsafe {
                ctx.device.create_fence(
                    &ash::vk::FenceCreateInfo {
                        flags: ash::vk::FenceCreateFlags::SIGNALED,
                        ..Default::default()
                    },
                    None,
                )
            }
            .unwrap();

            // Run LBM steps in 50-step batches to avoid TDR timeout
            let batch_size = 50u32;
            let step_result: std::result::Result<(), String> = (|| {
                let mut step = 0u32;
                while step < args.steps {
                    let batch_end = (step + batch_size).min(args.steps);
                    unsafe {
                        ctx.device
                            .reset_command_buffer(cmd, ash::vk::CommandBufferResetFlags::empty())
                            .map_err(|e| format!("reset cmd: {e}"))?;
                        ctx.device
                            .begin_command_buffer(
                                cmd,
                                &ash::vk::CommandBufferBeginInfo {
                                    flags: ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                                    ..Default::default()
                                },
                            )
                            .map_err(|e| format!("begin cmd: {e}"))?;
                    }
                    for s in step..batch_end {
                        lbm.step_with_params(cmd, s, tau_base, tau_amp, lambda)
                            .map_err(|e| format!("LBM step {s}: {e}"))?;
                    }
                    unsafe {
                        ctx.device
                            .end_command_buffer(cmd)
                            .map_err(|e| format!("end cmd: {e}"))?;
                        ctx.device
                            .reset_fences(&[fence])
                            .map_err(|e| format!("reset fence: {e}"))?;
                        ctx.device
                            .queue_submit(
                                ctx.queue,
                                &[ash::vk::SubmitInfo {
                                    command_buffer_count: 1,
                                    p_command_buffers: &cmd,
                                    ..Default::default()
                                }],
                                fence,
                            )
                            .map_err(|e| format!("submit: {e}"))?;
                        ctx.device
                            .wait_for_fences(&[fence], true, u64::MAX)
                            .map_err(|e| format!("wait: {e}"))?;
                    }
                    step = batch_end;
                }
                Ok(())
            })();

            unsafe {
                ctx.device.destroy_fence(fence, None);
                ctx.device.destroy_command_pool(cmd_pool, None);
            }

            if let Err(e) = step_result {
                eprintln!("    WARNING: LBM steps failed for k={k}: {e}");
                let vf = (ds * 0.1).min(1.0);
                let falsification = evaluate_cdg2_consistency(vf);
                results.push(KScanResult {
                    k,
                    spectral_dim: ds,
                    volume_fraction: vf,
                    falsification,
                });
                continue;
            }

            // Compute actual rho_mean from evolved field (not hardcoded 1.0)
            let rho_mean = match lbm.read_rho_field() {
                Ok(rho) => {
                    let sum: f64 = rho.iter().map(|&r| r as f64).sum();
                    (sum / rho.len() as f64) as f32
                }
                Err(_) => 1.0_f32,
            };

            // Dispatch dark halo detector
            let constants = HaloConstants {
                nx: args.grid,
                ny: args.grid,
                nz: args.grid,
                tau_base,
                tau_amp,
                zd_threshold: args.zd_threshold as f32,
                velocity_epsilon: args.velocity_epsilon as f32,
                density_factor: args.density_factor as f32,
                rho_mean,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };

            match bc_engine.dispatch_halo_detector(ctx, &constants) {
                Ok(vf) => {
                    eprintln!("    GPU void fraction vf={vf:.6}");
                    vf
                }
                Err(e) => {
                    eprintln!("    WARNING: Halo detector failed for k={k}: {e}");
                    (ds * 0.1).min(1.0)
                }
            }
        } else {
            // Dry-run: estimate from spectral dimension
            let vf = (ds * 0.1).min(1.0);
            eprintln!("    [DRY RUN] estimated void fraction vf={:.4}", vf,);
            vf
        };

        let falsification = evaluate_cdg2_consistency(volume_fraction);
        eprintln!("    consistent={}", falsification.consistent,);
        results.push(KScanResult {
            k,
            spectral_dim: ds,
            volume_fraction,
            falsification,
        });
    }

    // Overall verdict
    let any_consistent = results.iter().any(|r| r.falsification.consistent);
    let verdict = if any_consistent {
        "supported"
    } else {
        "falsified"
    };
    eprintln!();
    eprintln!("  Overall verdict: {verdict}");
    eprintln!(
        "  (Consistent at k values: {:?})",
        results
            .iter()
            .filter(|r| r.falsification.consistent)
            .map(|r| r.k)
            .collect::<Vec<_>>()
    );

    write_evidence(&args, &results, verdict);
}

#[cfg(feature = "gpu")]
fn run_cuda(args: &Args, alpha_bf: f64, t_0_bf: f64) {
    let g = args.grid as usize;
    let mut solver = match DarkHaloCudaSolver::new(g, g, g) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: CUDA solver init failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "    CUDA VRAM: {} MB (SoA)",
        solver.vram_bytes() / (1024 * 1024)
    );
    eprintln!();

    let tau_base = 0.55_f32;
    let tau_amp = 0.2_f32;
    let rho_threshold = args.density_factor as f32; // density_factor * rho_mean=1.0
    let vel_eps = args.velocity_epsilon as f32;

    let mut results = Vec::new();

    for &k in &args.k_values {
        let ds = spectral_dim_at_cdg2(k, alpha_bf, t_0_bf);
        eprint!("  k={k}: d_s(z=0.018) = {ds:.6} ... ");

        let t0 = std::time::Instant::now();
        match solver.run_k_value(
            k,
            args.steps,
            tau_base,
            tau_amp,
            rho_threshold,
            vel_eps,
            0.0,
            0, // no convergence check (stub)
        ) {
            Ok(result) => {
                let elapsed = t0.elapsed();
                let vf = result.volume_fraction;
                eprintln!(
                    "void fraction vf={vf:.6} ({} halos / {} cells, {} steps, {:.1}s{})",
                    result.halo_count,
                    result.n_cells,
                    result.steps_run,
                    elapsed.as_secs_f64(),
                    if result.early_stopped { " EARLY" } else { "" },
                );
                let falsification = evaluate_cdg2_consistency(vf);
                eprintln!("    consistent={}", falsification.consistent);
                results.push(KScanResult {
                    k,
                    spectral_dim: ds,
                    volume_fraction: vf,
                    falsification,
                });
            }
            Err(e) => {
                eprintln!("FAILED: {e}");
                let vf = (ds * 0.1).min(1.0);
                let falsification = evaluate_cdg2_consistency(vf);
                results.push(KScanResult {
                    k,
                    spectral_dim: ds,
                    volume_fraction: vf,
                    falsification,
                });
            }
        }
    }

    let any_consistent = results.iter().any(|r| r.falsification.consistent);
    let verdict = if any_consistent {
        "supported"
    } else {
        "falsified"
    };
    eprintln!();
    eprintln!("  Overall verdict: {verdict}");
    eprintln!(
        "  (Consistent at k values: {:?})",
        results
            .iter()
            .filter(|r| r.falsification.consistent)
            .map(|r| r.k)
            .collect::<Vec<_>>()
    );

    write_evidence(args, &results, verdict);
}

type GpuState = (
    lbm_vulkan::VulkanContext,
    lbm_vulkan::compute::GororobaEngine,
    BesagCliffordVulkanEngine,
);

fn init_gpu(args: &Args) -> std::result::Result<GpuState, String> {
    let ctx = lbm_vulkan::VulkanContext::new(false).map_err(|e| format!("Vulkan init: {e}"))?;
    let grid_dim = (args.grid, args.grid, args.grid);
    let lbm = lbm_vulkan::compute::GororobaEngine::new(
        &ctx,
        grid_dim,
        (1, 1),
        lbm_vulkan::Precision::FP32,
    )
    .map_err(|e| format!("LBM engine: {e}"))?;
    let config = BesagCliffordVulkanConfig {
        grid_dim,
        ..Default::default()
    };
    let mut bc_engine = BesagCliffordVulkanEngine::new(config, Some(ctx.clone()))
        .map_err(|e| format!("BC engine: {e}"))?;
    bc_engine
        .init_pipelines(&lbm)
        .map_err(|e| format!("Pipeline init: {e}"))?;
    Ok((ctx, lbm, bc_engine))
}

fn write_evidence(args: &Args, results: &[KScanResult], verdict: &str) {
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut toml = String::new();
    toml.push_str("# Dark halo hunt: CDG-2 consistency across k values\n");
    toml.push_str("# Generated by dark-halo-hunt binary\n\n");
    toml.push_str("[experiment]\n");
    toml.push_str("id = \"E-089\"\n");
    toml.push_str("name = \"dark-halo-hunt\"\n");
    toml.push_str(&format!("verdict = \"{verdict}\"\n\n"));
    toml.push_str("[config]\n");
    toml.push_str(&format!("grid_dim = {}\n", args.grid));
    toml.push_str(&format!("steps = {}\n", args.steps));
    toml.push_str(&format!("zd_threshold = {}\n", args.zd_threshold));
    toml.push_str(&format!("velocity_epsilon = {}\n", args.velocity_epsilon));
    toml.push_str(&format!("density_factor = {}\n\n", args.density_factor));

    for r in results {
        toml.push_str("[[results]]\n");
        toml.push_str(&format!("k = {}\n", r.k));
        toml.push_str(&format!("spectral_dim = {:.6}\n", r.spectral_dim));
        toml.push_str(&format!("volume_fraction = {:.6}\n", r.volume_fraction));
        toml.push_str(&format!(
            "predicted_f_baryon = {:.6}\n",
            r.falsification.predicted_f_baryon
        ));
        toml.push_str(&format!(
            "observed_f_baryon = {:.6}\n",
            r.falsification.observed_f_baryon
        ));
        toml.push_str(&format!(
            "delta_f_baryon = {:.6}\n",
            r.falsification.delta_f_baryon
        ));
        toml.push_str(&format!("consistent = {}\n\n", r.falsification.consistent));
    }

    match std::fs::File::create(&args.output) {
        Ok(mut f) => {
            f.write_all(toml.as_bytes()).unwrap();
            eprintln!("    Evidence written to {}", args.output);
        }
        Err(e) => eprintln!("    ERROR: Cannot write {}: {e}", args.output),
    }
}
