//! mrt-stability-audit: Systematic BGK vs MRT stability comparison.
//!
//! Runs 3D LBM at increasing density contrast and reports:
//! - Maximum Mach number reached before crash (NaN in density)
//! - Mass conservation error
//! - Crash threshold (density contrast where solver diverges)
//!
//! Expected result: MRT survives 3-5x higher density contrast than BGK
//! at matched tau, confirming ghost-moment damping advantage.

use clap::Parser;
use gororoba_cli_physics::lbm_dispatch::LbmBackend;
use lbm_3d::solver::CollisionMode;

#[derive(Parser)]
#[command(name = "mrt-stability-audit")]
#[command(about = "Compare BGK vs MRT stability at increasing density contrast")]
struct Args {
    /// Grid size (cubic: n x n x n)
    #[arg(long, default_value = "16")]
    n: usize,

    /// Relaxation time
    #[arg(long, default_value = "0.7")]
    tau: f64,

    /// Number of LBM steps per trial
    #[arg(long, default_value = "200")]
    steps: usize,

    /// Minimum density contrast (rho_peak / rho_background)
    #[arg(long, default_value = "2.0")]
    contrast_min: f64,

    /// Maximum density contrast
    #[arg(long, default_value = "100.0")]
    contrast_max: f64,

    /// Number of contrast levels to test
    #[arg(long, default_value = "10")]
    n_levels: usize,

    /// Output as CSV
    #[arg(long)]
    csv: bool,

    /// Use CUDA GPU solver (requires --features gpu)
    #[arg(long)]
    gpu: bool,

    /// Enable thread-coarsened GPU kernels (1 thread = 2 cells)
    #[arg(long)]
    coarsened: bool,

    /// Enable shared-memory tiled GPU kernels (8x8x4 tile + halo).
    /// Highest priority dispatch: overrides coarsening and pull-streaming.
    #[arg(long)]
    tiling: bool,

    /// Mach check interval: sample max Mach every N steps instead of
    /// every step. Higher values use graph-pair acceleration for bulk
    /// stepping between checks. Default 1 (check every step).
    #[arg(long, default_value = "1")]
    mach_interval: usize,
}

/// Result of one stability trial.
struct TrialResult {
    contrast: f64,
    mode: &'static str,
    survived: bool,
    max_mach: f64,
    mass_err: f64,
    final_step: usize,
}

/// Build Gaussian density perturbation centered on the grid.
fn gaussian_density(n: usize, contrast: f64) -> Vec<f64> {
    let center = n / 2;
    let sigma = (n as f64) / 6.0;
    let n_cells = n * n * n;
    let mut rho = Vec::with_capacity(n_cells);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let dx = ix as f64 - center as f64;
                let dy = iy as f64 - center as f64;
                let dz = iz as f64 - center as f64;
                let r2 = dx * dx + dy * dy + dz * dz;
                rho.push(1.0 + (contrast - 1.0) * (-r2 / (2.0 * sigma * sigma)).exp());
            }
        }
    }
    rho
}

/// Run one stability trial using the unified LbmBackend dispatcher.
fn run_trial(args: &Args, contrast: f64, mode: CollisionMode) -> TrialResult {
    let n = args.n;
    let mut backend = if args.gpu {
        #[cfg(feature = "gpu")]
        {
            LbmBackend::cuda(n, n, n, args.tau, mode).expect("GPU solver init failed")
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = mode;
            unreachable!("--gpu check prevents reaching here without feature");
        }
    } else {
        LbmBackend::cpu(n, n, n, args.tau, mode)
    };

    if args.coarsened {
        backend.set_coarsening(true);
    }
    if args.tiling {
        backend.set_tiling(true);
    }

    let rho = gaussian_density(n, contrast);
    let u_zero = vec![[0.0_f64; 3]; n * n * n];
    backend
        .initialize_custom(&rho, &u_zero)
        .expect("initialize_custom failed");

    let initial_mass = backend.total_mass().expect("total_mass failed");
    let mut max_mach = 0.0_f64;
    let mut final_step = 0;
    let interval = args.mach_interval.max(1);

    // Bulk-step with periodic Mach monitoring.
    // When interval > 1, step_n() uses graph-pair acceleration on GPU,
    // amortizing kernel launch overhead across the interval.
    let mut step = 0;
    while step < args.steps {
        let chunk = interval.min(args.steps - step);

        if chunk > 1 {
            if backend.step_n(chunk).is_err() {
                return TrialResult {
                    contrast,
                    mode: mode_name(mode),
                    survived: false,
                    max_mach,
                    mass_err: f64::NAN,
                    final_step: step + chunk,
                };
            }
        } else if backend.step().is_err() {
            return TrialResult {
                contrast,
                mode: mode_name(mode),
                survived: false,
                max_mach,
                mass_err: f64::NAN,
                final_step: step + 1,
            };
        }

        step += chunk;
        final_step = step;

        let ma = backend.max_mach_number().unwrap_or(f64::NAN);
        if !ma.is_finite() {
            return TrialResult {
                contrast,
                mode: mode_name(mode),
                survived: false,
                max_mach,
                mass_err: f64::NAN,
                final_step,
            };
        }
        if ma > max_mach {
            max_mach = ma;
        }
    }

    let final_mass = backend.total_mass().expect("total_mass failed");
    let mass_err = ((final_mass - initial_mass) / initial_mass).abs();

    TrialResult {
        contrast,
        mode: mode_name(mode),
        survived: true,
        max_mach,
        mass_err,
        final_step,
    }
}

fn mode_name(mode: CollisionMode) -> &'static str {
    match mode {
        CollisionMode::Bgk => "BGK",
        CollisionMode::Mrt => "MRT",
    }
}

fn print_result(result: &TrialResult, csv: bool) {
    if csv {
        println!(
            "{:.2},{},{},{:.6},{:.2e},{}",
            result.contrast,
            result.mode,
            result.survived,
            result.max_mach,
            result.mass_err,
            result.final_step
        );
    } else {
        println!(
            "{:>10.2} {:>5} {:>8} {:>10.6} {:>12.2e} {:>10}",
            result.contrast,
            result.mode,
            if result.survived { "OK" } else { "CRASH" },
            result.max_mach,
            result.mass_err,
            result.final_step
        );
    }
}

fn main() {
    let args = Args::parse();

    #[cfg(not(feature = "gpu"))]
    if args.gpu {
        eprintln!("ERROR: --gpu requires building with --features gpu");
        std::process::exit(1);
    }

    let contrasts: Vec<f64> = if args.n_levels <= 1 {
        vec![args.contrast_min]
    } else {
        (0..args.n_levels)
            .map(|i| {
                let t = i as f64 / (args.n_levels - 1) as f64;
                (args.contrast_min.ln() * (1.0 - t) + args.contrast_max.ln() * t).exp()
            })
            .collect()
    };

    let backend_name = if args.gpu { "GPU" } else { "CPU" };
    let kernel_tag = if args.tiling {
        " (tiled)"
    } else if args.coarsened {
        " (coarsened)"
    } else {
        ""
    };
    if args.csv {
        println!("contrast,mode,survived,max_mach,mass_err,final_step");
    } else {
        eprintln!(
            "MRT Stability Audit [{backend_name}{kernel_tag}]: {}^3 grid, tau={}, {} steps (Mach interval={}), {} levels [{:.1}..{:.1}]",
            args.n, args.tau, args.steps, args.mach_interval, args.n_levels, args.contrast_min, args.contrast_max
        );
        println!(
            "{:>10} {:>5} {:>8} {:>10} {:>12} {:>10}",
            "contrast", "mode", "survived", "max_mach", "mass_err", "step"
        );
        println!("{}", "-".repeat(60));
    }

    for &contrast in &contrasts {
        for mode in [CollisionMode::Bgk, CollisionMode::Mrt] {
            let result = run_trial(&args, contrast, mode);
            print_result(&result, args.csv);
        }
    }
}
