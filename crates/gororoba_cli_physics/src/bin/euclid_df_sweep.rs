//! euclid-df-sweep: Multi-galaxy fractal dimension sweep.
//!
//! Scales the single-galaxy D_f measurement from `euclid-dm-coupling` to
//! 100-1000+ galaxies using the Euclid Q1 catalog.  Supports both CUDA
//! (default) and CPU fallback paths.
//!
//! Usage:
//!   euclid-df-sweep --catalog data/external/euclid/.../useful_physical_measurements.parquet \
//!     --grid 64 --steps 200 --max-galaxies 100 --cuda
//!
//! The output CSV contains one row per galaxy with full provenance (Sersic params,
//! NFW halo, grid config, D_f before/after LBM, and elapsed time).

use clap::Parser;

#[cfg(any(test, feature = "euclid-catalog"))]
fn validate_null_design(alpha: f64, trials: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        alpha.is_finite() && alpha > 0.0,
        "null control requires positive alpha_zd to distinguish force conditions"
    );
    anyhow::ensure!(trials > 0, "null control requires at least one noise trial");
    Ok(())
}

#[cfg(test)]
mod null_design_tests {
    #[test]
    fn collapsed_controls_are_rejected() {
        assert!(super::validate_null_design(0.0, 10).is_err());
        assert!(super::validate_null_design(f64::NAN, 10).is_err());
        assert!(super::validate_null_design(0.1, 0).is_err());
        assert!(super::validate_null_design(0.1, 10).is_ok());
    }
}
#[cfg(feature = "euclid-catalog")]
use cosmology_core::galaxy_pipeline::{GalaxyDfRecord, GalaxyPipelineConfig, analyze_df_sweep};
#[cfg(feature = "euclid-catalog")]
use std::{io::Write, time::Instant};

#[cfg(feature = "euclid-catalog")]
use cosmology_core::{
    box_counting_fractal_dim, euclid_morphology::read_euclid_physical_measurements,
    galaxy_pipeline::prepare_galaxy,
};

#[derive(Parser)]
#[command(name = "euclid-df-sweep")]
#[command(about = "Multi-galaxy D_f sweep from Euclid Q1 catalog")]
struct Args {
    /// Path to Euclid useful_physical_measurements.parquet catalog.
    #[arg(long)]
    catalog: String,

    /// Grid dimension (cubic: N x N x N).
    #[arg(long, default_value = "64")]
    grid: usize,

    /// Number of LBM evolution steps per galaxy.
    #[arg(long, default_value = "200")]
    steps: usize,

    /// LBM relaxation time (minimum 0.501). Lower tau = less viscosity.
    /// With Plummer softening + MRT, tau=0.7 is the production sweet spot:
    /// low enough to preserve morphology, high enough for stability.
    #[arg(long, default_value = "0.7")]
    tau: f64,

    /// Cell size in kpc.
    #[arg(long, default_value = "1.0")]
    dx_kpc: f64,

    /// Maximum number of galaxies to process.
    #[arg(long, default_value = "100")]
    max_galaxies: usize,

    /// Output CSV path (default: stdout).
    #[arg(short, long)]
    output: Option<String>,

    /// Use morphological M/L (default: on).
    #[arg(long, default_value = "true")]
    morphological_ml: bool,

    /// Use CUDA GPU acceleration.
    #[arg(long, default_value = "true")]
    cuda: bool,

    /// Force CPU-only path (overrides --cuda).
    #[arg(long)]
    cpu: bool,

    /// Seed for bootstrap CI reproducibility.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Dry-run: measure D_f from initial density only (skip LBM evolution).
    #[arg(long)]
    skip_lbm: bool,

    /// Smagorinsky LES constant (0 = disabled, typical 0.1-0.2).
    /// Updates per-cell tau every 10 LBM steps to preserve density gradients.
    #[arg(long, default_value = "0.0")]
    smagorinsky_cs: f64,

    /// ZD algebraic forcing strength (0 = disabled, typical 0.1).
    /// Adds topological confinement from flat band localization.
    #[arg(long, default_value = "0.0")]
    alpha_zd: f64,

    /// Use MRT (Multiple-Relaxation-Time) collision instead of BGK.
    /// Stabilizes high-density-contrast galaxies where BGK diverges.
    #[arg(long)]
    mrt: bool,

    /// Plummer softening length in units of dx_kpc (default 0.5).
    /// Replaces singular NFW 1/r^2 cusp with 1/(r^2+eps^2).
    /// 0.5*dx eliminates the unphysical force spike while preserving macro D_f.
    #[arg(long, default_value = "0.5")]
    softening_eps: f64,

    /// Minimum density floor in LBM units (default 0.045).
    /// Controls max density contrast; 0.045 gives ~35:1, optimal for
    /// morphological signal with MRT stability at Ma < 1.5.
    #[arg(long, default_value = "0.045")]
    density_floor: f64,

    /// Run C0-C5 historical controls plus C6 matched Sersic NFW-only control.
    /// Evaluate six fixed numerical control contrasts around D_f=2.73.
    /// Physical interpretation requires separate convergence and model evidence.
    /// Requires --catalog for C4/C5/C6 and explicit mass/Mach budgets.
    #[arg(long)]
    null_hypothesis: bool,

    /// Noise realizations per stochastic condition (C2, C3).
    #[arg(long, default_value = "10")]
    null_n_trials: usize,

    /// Required null-control relative mass-error budget, measured from populations.
    #[arg(long, required_if_eq("null_hypothesis", "true"))]
    max_relative_mass_error: Option<f64>,

    /// Required null-control raw population-moment Mach budget.
    #[arg(long, required_if_eq("null_hypothesis", "true"))]
    max_mach: Option<f64>,

    /// New directory for exact null-control inputs and receipts; existing paths are refused.
    #[arg(long, required_if_eq("null_hypothesis", "true"))]
    null_evidence_dir: Option<std::path::PathBuf>,

    /// Enable shared-memory tiled GPU kernels (8x8x4 tile + halo).
    /// Highest priority dispatch: overrides coarsening and pull-streaming.
    /// Pull-scheme streaming from shared memory gives 15-25% speedup.
    #[arg(long)]
    tiling: bool,
}

fn main() {
    let args = Args::parse();

    // Physical floor: tau must exceed 0.5 for LBM stability (nu = (tau-0.5)/3 >= 0).
    // With Plummer softening, MRT is stable well below the old tau=1.5 floor.
    let tau = args.tau.max(0.501);
    if args.tau < 0.501 {
        eprintln!(
            "WARNING: tau={:.2} at/below LBM stability limit 0.5, clamped to 0.501",
            args.tau
        );
    }

    let use_gpu = args.cuda && !args.cpu;

    eprintln!("=== Euclid D_f Sweep ===");
    eprintln!(
        "Grid: {}^3, steps: {}, tau: {:.2}, max: {}",
        args.grid, args.steps, tau, args.max_galaxies
    );
    eprintln!(
        "Backend: {}, collision: {}, M/L: {}, Smagorinsky: {}",
        if use_gpu { "CUDA" } else { "CPU" },
        if args.mrt { "MRT" } else { "BGK" },
        if args.morphological_ml {
            "morphological"
        } else {
            "catalog"
        },
        if args.smagorinsky_cs > 0.0 {
            format!("C_s={:.2}", args.smagorinsky_cs)
        } else {
            "off".to_string()
        }
    );
    eprintln!(
        "Softening: eps={:.2}*dx, floor={:.4}",
        args.softening_eps, args.density_floor
    );

    #[cfg(feature = "euclid-catalog")]
    {
        let config = GalaxyPipelineConfig {
            grid_dim: args.grid,
            lbm_steps: args.steps,
            tau,
            dx_kpc: args.dx_kpc,
            max_galaxies: args.max_galaxies,
            morphological_ml: args.morphological_ml,
            alpha_zd: args.alpha_zd,
            softening_eps: args.softening_eps,
            density_floor: args.density_floor,
            halo_n_modes: 7,
        };

        run_sweep(&args, &config, use_gpu);
    }

    #[cfg(not(feature = "euclid-catalog"))]
    {
        let _ = (use_gpu, tau);
        eprintln!(
            "ERROR: --catalog requires the 'euclid-catalog' feature.\n\
             Rebuild with: cargo run --features euclid-catalog --bin euclid-df-sweep -- ..."
        );
        std::process::exit(1);
    }
}

#[cfg(feature = "euclid-catalog")]
fn run_sweep(args: &Args, config: &GalaxyPipelineConfig, use_gpu: bool) {
    // Null hypothesis control experiment bypasses catalog sweep entirely
    if args.null_hypothesis {
        validate_null_design(config.alpha_zd, args.null_n_trials)
            .expect("invalid null-control design");
        if let Err(error) = run_null_hypothesis(args, config, use_gpu) {
            eprintln!("null_control_failed: {error:#}");
            std::process::exit(1);
        }
        return;
    }

    // Read catalog
    eprintln!("Reading catalog: {}", args.catalog);
    let entries = read_euclid_physical_measurements(&args.catalog).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });
    eprintln!("Catalog: {} valid galaxies", entries.len());

    // Open output
    let mut out: Box<dyn Write> = if let Some(ref path) = args.output {
        Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("ERROR: cannot create output file {path}: {e}");
            std::process::exit(1);
        }))
    } else {
        Box::new(std::io::stdout().lock())
    };

    writeln!(out, "{}", GalaxyDfRecord::csv_header()).expect("write header");

    let mut records = Vec::new();
    let mut skipped = 0usize;
    let g = config.grid_dim;

    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            run_gpu_sweep(
                args,
                config,
                &entries,
                &mut out,
                &mut records,
                &mut skipped,
                g,
            );
        }
        #[cfg(not(feature = "gpu"))]
        {
            eprintln!("WARNING: GPU feature not enabled, falling back to CPU");
            run_cpu_sweep(
                args,
                config,
                &entries,
                &mut out,
                &mut records,
                &mut skipped,
                g,
            );
        }
    } else {
        run_cpu_sweep(
            args,
            config,
            &entries,
            &mut out,
            &mut records,
            &mut skipped,
            g,
        );
    }

    // Summary
    if records.is_empty() {
        eprintln!("\nNo valid galaxies processed (skipped={skipped}).");
        return;
    }

    let summary = analyze_df_sweep(&records, skipped, args.seed);
    eprintln!(
        "\n=== D_f Sweep Summary (N={}, skipped={}) ===",
        summary.n_galaxies, summary.n_skipped
    );
    eprintln!(
        "Overall:     D_f = {:.3} +/- {:.3} (CI_95: [{:.3}, {:.3}])",
        summary.overall_mean, summary.overall_std, summary.overall_ci_95.0, summary.overall_ci_95.1,
    );
    if let Some(ref d) = summary.disk_summary {
        eprintln!(
            "Disk (n<2):  D_f = {:.3} +/- {:.3} (CI_95: [{:.3}, {:.3}])  N={}",
            d.mean, d.std, d.ci_95.0, d.ci_95.1, d.n,
        );
    }
    if let Some(ref e) = summary.elliptical_summary {
        eprintln!(
            "Elliptical:  D_f = {:.3} +/- {:.3} (CI_95: [{:.3}, {:.3}])  N={}",
            e.mean, e.std, e.ci_95.0, e.ci_95.1, e.n,
        );
    }
}

#[cfg(all(feature = "euclid-catalog", feature = "gpu"))]
fn run_gpu_sweep(
    args: &Args,
    config: &GalaxyPipelineConfig,
    entries: &[cosmology_core::euclid_morphology::EuclidSersicParams],
    out: &mut Box<dyn Write>,
    records: &mut Vec<GalaxyDfRecord>,
    skipped: &mut usize,
    g: usize,
) {
    use lbm_3d_cuda::{LbmSolver3DCuda, Precision, box_counting_gpu::GpuBoxCounter};

    // 3-stage multi-stream pipeline (YSU-inspired):
    //   Stream A (solver): LBM evolution
    //   Stream B (box_counter_b): box-counting (concurrent with next galaxy's CPU prep)
    //
    // Timeline per galaxy K (pipelined, K >= 1):
    //   CPU:      prepare K+1          || (overlaps with GPU work on K)
    //   Stream A: LBM step K           || (10ms, records event_lbm)
    //   Stream B: wait(event_lbm), box-count K (2ms, records event_box)
    //   Sync:     wait(event_box) before initialize K+1 (protects d_rho)
    //
    // CPU prep for K+1 overlaps with GPU LBM+box-count for K.
    // Expected: max(10ms, 5ms) = ~10ms/galaxy vs 17ms sequential -> ~40% gain.

    // Create GPU solver (compile kernels once, reuse across galaxies)
    // MRT extends Ma stability from ~0.3 to ~1.5 for high-contrast galaxies.
    let mut solver = if args.mrt {
        LbmSolver3DCuda::new_mrt(g, g, g, config.tau, Precision::FP32)
    } else {
        LbmSolver3DCuda::new(g, g, g, config.tau, Precision::FP32)
    }
    .unwrap_or_else(|e| {
        eprintln!("ERROR: CUDA solver init failed: {e}");
        std::process::exit(1);
    });

    if args.tiling {
        solver.set_tiling(true);
    }

    // Pin rho in L2 cache for collision kernel locality (SM 8.0+).
    if let Err(e) = solver.set_l2_pinning(true) {
        eprintln!("WARNING: L2 pinning failed: {e} (non-fatal, continuing without)");
    }

    // Stream A: solver's default stream (LBM evolution + initial D_f)
    // Stream B: forked stream for final box-counting (overlaps with CPU prep)
    let stream_b = solver.fork_stream().unwrap_or_else(|e| {
        eprintln!("ERROR: fork_stream failed: {e}");
        std::process::exit(1);
    });

    // box_counter on stream A: used for initial D_f measurement (before LBM)
    let mut box_counter = GpuBoxCounter::new(solver.context()).unwrap_or_else(|e| {
        eprintln!("ERROR: GpuBoxCounter init failed: {e}");
        std::process::exit(1);
    });
    // box_counter_b on stream B: used for final D_f measurement (after LBM)
    let mut box_counter_b = GpuBoxCounter::new_with_stream(solver.context(), stream_b.clone())
        .unwrap_or_else(|e| {
            eprintln!("ERROR: GpuBoxCounter (stream B) init failed: {e}");
            std::process::exit(1);
        });

    let zeros_u = vec![[0.0_f64; 3]; g * g * g];
    let use_smag = args.smagorinsky_cs > 0.0;

    // Pipeline state: deferred final D_f from previous galaxy.
    // The box-count runs on stream B while CPU prepares the next galaxy.
    // We collect the result at the start of the next iteration (or after the loop).
    struct PendingResult {
        rec: GalaxyDfRecord,
        df_initial: f64,
        t0: Instant,
        idx: usize,
    }
    let mut pending: Option<PendingResult> = None;

    // Flush a pending result by reading D_f from stream B (syncs stream B).
    let flush_pending = |pending: &mut Option<PendingResult>,
                         box_counter_b: &mut GpuBoxCounter,
                         solver: &LbmSolver3DCuda,
                         out: &mut Box<dyn Write>,
                         records: &mut Vec<GalaxyDfRecord>,
                         skipped: &usize,
                         max_galaxies: usize,
                         g: usize,
                         skip_lbm: bool| {
        if let Some(mut p) = pending.take() {
            let df_final = if skip_lbm {
                p.df_initial
            } else {
                box_counter_b
                    .fractal_dimension_device_auto(solver.d_rho_bytes(), g, g, g)
                    .map(|r| r.d_f)
                    .unwrap_or_else(|e| {
                        eprintln!("  WARN: GPU D_f (stream B) failed: {e}");
                        3.0
                    })
            };
            let elapsed = p.t0.elapsed().as_millis() as u64;
            p.rec.df_initial = p.df_initial;
            p.rec.df_final = df_final;
            p.rec.elapsed_ms = elapsed;

            if p.rec.df_final < 1.0 || p.rec.df_final > 3.0 {
                eprintln!(
                    "  WARN: D_f={:.4} out of [1,3] for obj={}",
                    p.rec.df_final, p.rec.object_id
                );
            }
            writeln!(out, "{}", p.rec).expect("write record");
            eprintln!(
                "[{}/{}] obj={} n={:.2} type={} D_f={:.4} ({elapsed}ms)",
                p.idx + 1 - *skipped,
                max_galaxies,
                p.rec.object_id,
                p.rec.sersic_n,
                p.rec.morphological_type,
                p.rec.df_final,
            );
            records.push(p.rec);
        }
    };

    for (i, entry) in entries.iter().take(config.max_galaxies).enumerate() {
        // Flush previous galaxy's deferred D_f (syncs stream B if needed).
        // This blocks until box-counting for the previous galaxy completes,
        // but CPU prep below already ran in parallel with that box-count.
        flush_pending(
            &mut pending,
            &mut box_counter_b,
            &solver,
            out,
            records,
            skipped,
            config.max_galaxies,
            g,
            args.skip_lbm,
        );

        let setup = match prepare_galaxy(entry, config) {
            Some(s) => s,
            None => {
                *skipped += 1;
                continue;
            }
        };

        // Initialize LBM with combined density (safe: stream B is synced above)
        solver
            .initialize_custom(&setup.rho_total, &zeros_u)
            .unwrap_or_else(|e| {
                eprintln!("  WARN: init failed for obj={}: {e}", entry.object_id);
            });
        solver
            .set_force_field(&setup.force_field)
            .unwrap_or_else(|e| {
                eprintln!(
                    "  WARN: force field failed for obj={}: {e}",
                    entry.object_id
                );
            });

        // Measure initial D_f on stream A (before LBM evolution)
        let df_initial = box_counter
            .fractal_dimension_device_auto(solver.d_rho_bytes(), g, g, g)
            .map(|r| r.d_f)
            .unwrap_or_else(|e| {
                eprintln!("  WARN: GPU D_f failed for obj={}: {e}", entry.object_id);
                3.0
            });

        let t0 = Instant::now();

        // LBM evolution on stream A (async kernel launches)
        if !args.skip_lbm {
            if use_smag {
                for step in 0..config.lbm_steps {
                    if step % 10 == 0 {
                        solver
                            .update_smagorinsky_tau(args.smagorinsky_cs, config.dx_kpc, config.tau)
                            .unwrap_or_else(|e| {
                                eprintln!(
                                    "  WARN: Smagorinsky update failed for obj={}: {e}",
                                    entry.object_id
                                );
                            });
                    }
                    solver.step().expect("LBM step");
                }
            } else {
                solver.step_n(config.lbm_steps).expect("LBM step_n");
            }

            // Record event on stream A after LBM completes.
            // Stream B will wait for this before reading d_rho for box-counting.
            let event_lbm = solver
                .stream()
                .record_event(None)
                .expect("record LBM event");
            stream_b.wait(&event_lbm).expect("stream B wait for LBM");
        }

        // Defer the final D_f measurement to stream B.
        // CPU can immediately start preparing the next galaxy while stream B
        // runs box-counting in the background.
        pending = Some(PendingResult {
            rec: setup.record_template,
            df_initial,
            t0,
            idx: i,
        });
    }

    // Flush the last galaxy's deferred result.
    flush_pending(
        &mut pending,
        &mut box_counter_b,
        &solver,
        out,
        records,
        skipped,
        config.max_galaxies,
        g,
        args.skip_lbm,
    );
}

#[cfg(feature = "euclid-catalog")]
fn run_cpu_sweep(
    args: &Args,
    config: &GalaxyPipelineConfig,
    entries: &[cosmology_core::euclid_morphology::EuclidSersicParams],
    out: &mut Box<dyn Write>,
    records: &mut Vec<GalaxyDfRecord>,
    skipped: &mut usize,
    g: usize,
) {
    use lbm_3d::solver::{BgkCollision, LbmSolver3D};

    for (i, entry) in entries.iter().take(config.max_galaxies).enumerate() {
        let setup = match prepare_galaxy(entry, config) {
            Some(s) => s,
            None => {
                *skipped += 1;
                continue;
            }
        };

        // Initialize CPU LBM solver (MRT stabilizes high-contrast galaxies)
        let mut solver = if args.mrt {
            LbmSolver3D::new_mrt(g, g, g, config.tau)
        } else {
            LbmSolver3D::new(g, g, g, config.tau)
        };

        let lattice = &solver.collider.lattice;
        for iz in 0..g {
            for iy in 0..g {
                for ix in 0..g {
                    let idx = iz * g * g + iy * g + ix;
                    let rho_init = setup.rho_total[idx];
                    let f_eq = BgkCollision::initialize_rest(rho_init, lattice);
                    for (dir, &population) in f_eq.iter().enumerate() {
                        solver.f[lbm_3d::solver::aosoa_idx(idx, dir)] = population;
                    }
                    solver.rho[idx] = rho_init;
                    solver.u[idx] = [0.0, 0.0, 0.0];
                }
            }
        }
        solver
            .set_force_field(setup.force_field)
            .expect("force field size mismatch");

        let df_initial = box_counting_fractal_dim(&solver.rho, g, g, g);

        let t0 = Instant::now();

        if !args.skip_lbm {
            let use_smag = args.smagorinsky_cs > 0.0;
            for step in 0..config.lbm_steps {
                if use_smag && step % 10 == 0 {
                    solver
                        .update_smagorinsky_tau(args.smagorinsky_cs, config.dx_kpc, config.tau)
                        .unwrap_or_else(|e| {
                            eprintln!(
                                "  WARN: Smagorinsky update failed for obj={}: {e}",
                                entry.object_id
                            );
                        });
                }
                solver.evolve_one_step();

                // Early exit once fully diverged (>99% NaN)
                if step % 10 == 0 {
                    let nan_c = solver.rho.iter().filter(|r| !r.is_finite()).count();
                    if nan_c > solver.rho.len() * 99 / 100 {
                        break;
                    }
                }
            }
        }

        // Diagnose NaN/non-finite cells after LBM evolution
        let nan_count = solver.rho.iter().filter(|r| !r.is_finite()).count();
        if nan_count > 0 {
            let n = solver.rho.len();
            eprintln!(
                "  DIVERGED: obj={} has {nan_count}/{n} non-finite cells ({:.1}%)",
                entry.object_id,
                100.0 * nan_count as f64 / n as f64,
            );
        }

        let df_final = if args.skip_lbm {
            df_initial
        } else {
            box_counting_fractal_dim(&solver.rho, g, g, g)
        };

        let elapsed = t0.elapsed().as_millis() as u64;

        let mut rec = setup.record_template;
        rec.df_initial = df_initial;
        rec.df_final = df_final;
        rec.elapsed_ms = elapsed;

        if rec.df_final < 1.0 || rec.df_final > 3.0 {
            eprintln!(
                "  WARN: D_f={:.4} out of [1,3] for obj={}",
                rec.df_final, rec.object_id
            );
        }

        writeln!(out, "{rec}").expect("write record");
        eprintln!(
            "[{}/{}] obj={} n={:.2} type={} D_f={:.4} ({elapsed}ms)",
            i + 1 - *skipped,
            config.max_galaxies,
            rec.object_id,
            rec.sersic_n,
            rec.morphological_type,
            rec.df_final,
        );

        records.push(rec);
    }
}

// ---------------------------------------------------------------------------
// Null hypothesis control experiment
// ---------------------------------------------------------------------------

/// Result of one null-hypothesis trial.
#[cfg(feature = "euclid-catalog")]
#[derive(Clone, Debug)]
struct NullTrialResult {
    condition: String,
    trial: usize,
    df_initial: f64,
    df_final: f64,
    elapsed_ms: u64,
    seed: Option<u64>,
    observed_step: usize,
    attempted_step: usize,
    mass_error: f64,
    max_mach: f64,
    minimum_population: f64,
    minimum_density: f64,
    finite_state: bool,
    positive_density: bool,
    nonnegative_population: bool,
    mass_within_budget: bool,
    mach_within_budget: bool,
    failure: Option<String>,
}

#[cfg(feature = "euclid-catalog")]
fn retain_trial_inputs(
    directory: &std::path::Path,
    rho: &[f64],
    force: &[[f64; 3]],
    mut metadata: serde_json::Value,
) -> anyhow::Result<()> {
    use sha2::{Digest, Sha256};
    std::fs::create_dir(directory)?;
    for (name, values) in [
        ("rho.f64le", rho.to_vec()),
        ("force.xyz.f64le", force.iter().flatten().copied().collect()),
    ] {
        let bytes: Vec<_> = values.into_iter().flat_map(f64::to_le_bytes).collect();
        let digest = Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let mut file = std::fs::File::create_new(directory.join(name))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        metadata[name] = serde_json::json!({"sha256": digest, "bytes": bytes.len(), "encoding": "IEEE754_f64_little_endian"});
    }
    let mut receipt = std::fs::File::create_new(directory.join("input.json"))?;
    serde_json::to_writer_pretty(&mut receipt, &metadata)?;
    receipt.write_all(b"\n")?;
    receipt.sync_all()?;
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn record_null_observation(
    result: &mut NullTrialResult,
    observation: &gororoba_cli_physics::lbm_population_diagnostics::PopulationObservation,
    initial_mass: f64,
    budgets: (f64, f64),
    step: usize,
) {
    let mass_error = (observation.mass / initial_mass - 1.0).abs();
    let finite = observation.finite && mass_error.is_finite();
    let positive = observation.minimum_density.is_finite() && observation.minimum_density > 0.0;
    let nonnegative =
        observation.minimum_population.is_finite() && observation.minimum_population >= 0.0;
    let mass_ok = mass_error.is_finite() && mass_error <= budgets.0;
    let mach_ok = observation.mach.is_finite() && observation.mach <= budgets.1;
    if step == 0 {
        result.finite_state = finite;
        result.positive_density = positive;
        result.nonnegative_population = nonnegative;
        result.mass_within_budget = mass_ok;
        result.mach_within_budget = mach_ok;
    } else {
        result.finite_state &= finite;
        result.positive_density &= positive;
        result.nonnegative_population &= nonnegative;
        result.mass_within_budget &= mass_ok;
        result.mach_within_budget &= mach_ok;
    }
    result.observed_step = step;
    result.mass_error = mass_error;
    result.max_mach = result.max_mach.max(observation.mach);
    result.minimum_population = result
        .minimum_population
        .min(observation.minimum_population);
    result.minimum_density = result.minimum_density.min(observation.minimum_density);
}

/// Generate white-noise density field with uniform distribution [lo, hi).
/// Seeded PRNG for reproducibility across runs.
#[cfg(feature = "euclid-catalog")]
fn generate_white_noise(n_cells: usize, lo: f64, hi: f64, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Uniform};

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let dist = Uniform::new(lo, hi).expect("lo < hi for white-noise range");
    (0..n_cells).map(|_| dist.sample(&mut rng)).collect()
}

/// Compute ZD algebraic force field without galaxy-specific NFW gravity.
///
/// Mirrors the ZD forcing logic in `galaxy_pipeline::prepare_galaxy()` but
/// operates on an arbitrary grid without requiring a catalog entry.  The force
/// is radially inward (confining), exponentially decaying with scale length
/// `r_e_kpc`, and weighted by `alpha_zd * (1 - flat_band_fraction)`.
#[cfg(feature = "euclid-catalog")]
fn compute_zd_force_standalone(
    g: usize,
    dx_kpc: f64,
    r_e_kpc: f64,
    alpha_zd: f64,
) -> Vec<[f64; 3]> {
    let flat_band_fraction = 0.5; // proven constant for D >= 16
    let coupling = alpha_zd * (1.0 - flat_band_fraction);
    let force_target = 1e-4;
    let center = g as f64 / 2.0;
    let n_cells = g * g * g;
    let mut force = vec![[0.0_f64; 3]; n_cells];
    for iz in 0..g {
        for iy in 0..g {
            for ix in 0..g {
                let cell_dx = (ix as f64 + 0.5 - center) * dx_kpc;
                let cell_dy = (iy as f64 + 0.5 - center) * dx_kpc;
                let cell_dz = (iz as f64 + 0.5 - center) * dx_kpc;
                let r = (cell_dx * cell_dx + cell_dy * cell_dy + cell_dz * cell_dz)
                    .sqrt()
                    .max(0.1);
                let idx = iz * g * g + iy * g + ix;
                let weight = coupling * (-r / r_e_kpc).exp();
                if r > 0.1 {
                    force[idx][0] = -weight * force_target * cell_dx / r;
                    force[idx][1] = -weight * force_target * cell_dy / r;
                    force[idx][2] = -weight * force_target * cell_dz / r;
                }
            }
        }
    }
    force
}

#[cfg(feature = "euclid-catalog")]
const NULL_CONDITIONS: [&str; 7] = [
    "C0-uniform-zero",
    "C1-uniform-fzd",
    "C2-noise-zero",
    "C3-noise-fzd",
    "C4-sersic-zero",
    "C5-sersic-fzd",
    "C6-sersic-nfw",
];

#[cfg(feature = "euclid-catalog")]
fn trial_seed(seed: u64, trial: usize) -> u64 {
    seed.wrapping_add(trial as u64)
}

#[cfg(feature = "euclid-catalog")]
fn null_backend(
    args: &Args,
    config: &GalaxyPipelineConfig,
    use_gpu: bool,
) -> anyhow::Result<gororoba_cli_physics::lbm_dispatch::LbmBackend> {
    use gororoba_cli_physics::lbm_dispatch::LbmBackend;
    let dimension = config.grid_dim;
    let mode = if args.mrt {
        lbm_3d::solver::CollisionMode::Mrt
    } else {
        lbm_3d::solver::CollisionMode::Bgk
    };
    let mut backend = if use_gpu {
        #[cfg(feature = "gpu")]
        {
            LbmBackend::cuda(dimension, dimension, dimension, config.tau, mode)?
        }
        #[cfg(not(feature = "gpu"))]
        {
            anyhow::bail!(
                "requested CUDA null-control backend requires gpu feature; select --cpu explicitly"
            );
        }
    } else {
        LbmBackend::cpu(dimension, dimension, dimension, config.tau, mode)
    };
    anyhow::ensure!(
        !args.tiling || use_gpu,
        "tiling requires CUDA; CPU cannot execute this configuration"
    );
    backend.set_tiling(args.tiling);
    Ok(backend)
}

#[cfg(feature = "euclid-catalog")]
fn run_null_trial(
    args: &Args,
    config: &GalaxyPipelineConfig,
    use_gpu: bool,
    rho: &[f64],
    force: &[[f64; 3]],
    identity: (&str, usize, Option<u64>, &str),
) -> NullTrialResult {
    use gororoba_cli_physics::lbm_population_diagnostics::inspect_fields;
    use sha2::{Digest, Sha256};
    let (condition, trial, seed, object_id) = identity;
    let mut density_hash = Sha256::new();
    for value in rho {
        density_hash.update(value.to_le_bytes());
    }
    let mut force_hash = Sha256::new();
    for value in force.iter().flatten() {
        force_hash.update(value.to_le_bytes());
    }
    eprintln!(
        "trial_input condition={condition} trial={trial} seed={seed:?} rho_f64le_sha256={} force_xyz_f64le_sha256={}",
        density_hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        force_hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let start = Instant::now();
    let mut result = NullTrialResult {
        condition: condition.into(),
        trial,
        seed,
        df_initial: f64::NAN,
        df_final: f64::NAN,
        elapsed_ms: 0,
        observed_step: 0,
        attempted_step: 0,
        mass_error: f64::NAN,
        max_mach: 0.0,
        minimum_population: f64::INFINITY,
        minimum_density: f64::INFINITY,
        finite_state: false,
        positive_density: false,
        nonnegative_population: false,
        mass_within_budget: false,
        mach_within_budget: false,
        failure: None,
    };
    let execution = (|| -> anyhow::Result<()> {
        let evidence_root = args
            .null_evidence_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("null evidence directory required"))?;
        retain_trial_inputs(
            &evidence_root.join(format!("{condition}-{trial}")),
            rho,
            force,
            serde_json::json!({
                "condition":condition,"trial":trial,"seed":seed,"selected_object_id":object_id,
                "backend":if use_gpu {"CUDA_FP32"}else{"CPU_FP64"},"catalog":args.catalog,
                "grid":config.grid_dim,"steps":config.lbm_steps,"tau":config.tau,"dx_kpc":config.dx_kpc,
                "alpha_zd":config.alpha_zd,"density_floor":config.density_floor,"softening_eps":config.softening_eps,
                "smagorinsky_cs":args.smagorinsky_cs,"collision":if args.mrt {"MRT"}else{"BGK"},
                "max_relative_mass_error":args.max_relative_mass_error,"max_mach":args.max_mach,
            "observation":"direct_every_step_post_step_population_moments","zero_initial_velocity":true,"tiling":args.tiling
            }),
        )?;
        anyhow::ensure!(
            rho.len() == config.grid_dim.pow(3)
                && force.len() == rho.len()
                && force.iter().flatten().all(|value| value.is_finite()),
            "invalid trial input dimensions or force values"
        );
        let mut backend = null_backend(args, config, use_gpu)?;
        backend.initialize_custom(rho, &vec![[0.0; 3]; rho.len()])?;
        backend.set_force_field(force.to_vec())?;
        let initial = inspect_fields(&mut backend)?;
        let mass_budget = args
            .max_relative_mass_error
            .ok_or_else(|| anyhow::anyhow!("mass budget required"))?;
        let mach_budget = args
            .max_mach
            .ok_or_else(|| anyhow::anyhow!("Mach budget required"))?;
        record_null_observation(
            &mut result,
            &initial,
            initial.mass,
            (mass_budget, mach_budget),
            0,
        );
        initial.require_stable(initial.mass, mass_budget, mach_budget)?;
        result.df_initial = box_counting_fractal_dim(
            &initial.density,
            config.grid_dim,
            config.grid_dim,
            config.grid_dim,
        );
        anyhow::ensure!(
            result.df_initial.is_finite() && (0.0..=3.0).contains(&result.df_initial),
            "initial box-count dimension outside finite[0,3]"
        );
        let mut final_density = initial.density;
        for step in 0..config.lbm_steps {
            result.attempted_step = step + 1;
            if args.smagorinsky_cs > 0.0 && step % 10 == 0 {
                match &mut backend {
                    gororoba_cli_physics::lbm_dispatch::LbmBackend::Avx2(solver) => solver
                        .update_smagorinsky_tau(args.smagorinsky_cs, config.dx_kpc, config.tau)?,
                    #[cfg(feature = "gpu")]
                    gororoba_cli_physics::lbm_dispatch::LbmBackend::Cuda(solver) => solver
                        .update_smagorinsky_tau(args.smagorinsky_cs, config.dx_kpc, config.tau)?,
                }
            }
            backend.step()?;
            let observation = inspect_fields(&mut backend)?;
            record_null_observation(
                &mut result,
                &observation,
                initial.mass,
                (mass_budget, mach_budget),
                step + 1,
            );
            observation.require_stable(initial.mass, mass_budget, mach_budget)?;
            final_density = observation.density;
        }
        result.df_final = box_counting_fractal_dim(
            &final_density,
            config.grid_dim,
            config.grid_dim,
            config.grid_dim,
        );
        anyhow::ensure!(
            result.df_final.is_finite() && (0.0..=3.0).contains(&result.df_final),
            "final box-count dimension outside finite[0,3]"
        );
        Ok(())
    })();
    if let Err(error) = execution {
        result.failure = Some(error.to_string());
    }
    result.elapsed_ms = start.elapsed().as_millis() as u64;
    result
}

#[cfg(feature = "euclid-catalog")]
fn run_null_hypothesis(
    args: &Args,
    config: &GalaxyPipelineConfig,
    use_gpu: bool,
) -> anyhow::Result<()> {
    use cosmology_core::euclid_morphology::read_euclid_physical_measurements_audited;
    validate_null_design(config.alpha_zd, args.null_n_trials)?;
    anyhow::ensure!(
        args.tau.is_finite() && args.tau >= 0.501 && args.tau == config.tau,
        "null trials require an explicit finite tau>=0.501 without clamping"
    );
    anyhow::ensure!(
        config.dx_kpc.is_finite()
            && config.dx_kpc > 0.0
            && args.smagorinsky_cs.is_finite()
            && args.smagorinsky_cs >= 0.0,
        "positive finite spatial scale and nonnegative finite Smagorinsky coefficient required"
    );
    anyhow::ensure!(
        config.grid_dim.checked_pow(3).is_some(),
        "grid cell count overflow"
    );
    anyhow::ensure!(
        config.grid_dim >= 4 && config.lbm_steps > 0,
        "null trials require grid>=4 and steps>0"
    );
    anyhow::ensure!(
        !args.skip_lbm,
        "null trials require evolution; skip-lbm changes the declared experiment"
    );
    let mass_budget = args
        .max_relative_mass_error
        .ok_or_else(|| anyhow::anyhow!("mass budget required"))?;
    let mach_budget = args
        .max_mach
        .ok_or_else(|| anyhow::anyhow!("Mach budget required"))?;
    anyhow::ensure!(
        mass_budget.is_finite()
            && mass_budget >= 0.0
            && mach_budget.is_finite()
            && mach_budget > 0.0,
        "invalid mass/Mach budgets"
    );
    let catalog =
        read_euclid_physical_measurements_audited(&args.catalog).map_err(anyhow::Error::msg)?;
    let (entry, setup) = catalog
        .entries
        .iter()
        .find_map(|entry| prepare_galaxy(entry, config).map(|setup| (entry, setup)))
        .ok_or_else(|| anyhow::anyhow!("catalog lacks an admitted galaxy setup"))?;
    let mut nfw_config = config.clone();
    nfw_config.alpha_zd = 0.0;
    let nfw_setup = prepare_galaxy(entry, &nfw_config)
        .ok_or_else(|| anyhow::anyhow!("NFW-only paired setup failed"))?;
    anyhow::ensure!(
        setup.rho_total == nfw_setup.rho_total,
        "paired galaxy force controls require identical density"
    );
    anyhow::ensure!(
        setup.force_field != nfw_setup.force_field,
        "full and NFW-only forces coincide; ZD increment is unidentifiable"
    );
    eprintln!(
        "catalog_rows={} rejected_rows={} admitted_entries={} selected_object={} backend={} observation=post_step_population_moments df_oracle=host_box_counting observation_interval=1 mass_budget={} mach_budget={} readback_and_initialization_overhead=included",
        catalog.rows_read,
        catalog.rejected_rows.len(),
        catalog.entries.len(),
        entry.object_id,
        if use_gpu { "CUDA_FP32" } else { "CPU_FP64" },
        mass_budget,
        mach_budget
    );
    let cells = config.grid_dim.pow(3);
    let uniform = vec![1.0; cells];
    let zero = vec![[0.0; 3]; cells];
    let zd = compute_zd_force_standalone(
        config.grid_dim,
        config.dx_kpc,
        setup.record_template.r_e_kpc,
        config.alpha_zd,
    );
    let mut results = Vec::new();
    let evidence_root = args
        .null_evidence_dir
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("null evidence directory required"))?;
    std::fs::create_dir(evidence_root)?;
    let selected_object = entry.object_id.to_string();
    for (condition, force) in [(NULL_CONDITIONS[0], &zero), (NULL_CONDITIONS[1], &zd)] {
        results.push(run_null_trial(
            args,
            config,
            use_gpu,
            &uniform,
            force,
            (condition, 0, None, &selected_object),
        ));
    }
    for trial in 0..args.null_n_trials {
        let seed = trial_seed(args.seed, trial);
        let noise = generate_white_noise(cells, 0.5, 1.5, seed);
        for (condition, force) in [(NULL_CONDITIONS[2], &zero), (NULL_CONDITIONS[3], &zd)] {
            results.push(run_null_trial(
                args,
                config,
                use_gpu,
                &noise,
                force,
                (condition, trial, Some(seed), &selected_object),
            ));
        }
    }
    for (condition, force) in [
        (NULL_CONDITIONS[4], &zero),
        (NULL_CONDITIONS[5], &setup.force_field),
        (NULL_CONDITIONS[6], &nfw_setup.force_field),
    ] {
        results.push(run_null_trial(
            args,
            config,
            use_gpu,
            &setup.rho_total,
            force,
            (condition, 0, None, &selected_object),
        ));
    }
    let mut retained_rows = std::fs::File::create_new(evidence_root.join("trials.csv"))?;
    write_null_rows(&results, &mut retained_rows)?;
    retained_rows.sync_all()?;
    report_null_hypothesis(
        &results,
        args.null_n_trials,
        args.seed,
        config.lbm_steps,
        mass_budget,
        mach_budget,
    )
}

#[cfg(feature = "euclid-catalog")]
fn validate_trial_rows(
    results: &[NullTrialResult],
    trials: usize,
    seed: u64,
    steps: usize,
    mass_budget: f64,
    mach_budget: f64,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        mass_budget.is_finite()
            && mass_budget >= 0.0
            && mach_budget.is_finite()
            && mach_budget > 0.0,
        "invalid mass/Mach budgets"
    );
    let mut expected = std::collections::BTreeSet::new();
    for (index, condition) in NULL_CONDITIONS.iter().enumerate() {
        for trial in 0..if index == 2 || index == 3 { trials } else { 1 } {
            expected.insert((*condition, trial));
        }
    }
    let mut observed = std::collections::BTreeSet::new();
    for row in results {
        anyhow::ensure!(
            observed.insert((row.condition.as_str(), row.trial)),
            "duplicate trial row"
        );
        anyhow::ensure!(
            row.failure.is_none() && row.observed_step == steps && row.attempted_step == steps,
            "trial failed or incomplete: {}[{}] {:?}",
            row.condition,
            row.trial,
            row.failure
        );
        anyhow::ensure!(
            row.df_initial.is_finite()
                && row.df_final.is_finite()
                && (0.0..=3.0).contains(&row.df_final),
            "invalid dimension"
        );
        anyhow::ensure!(
            row.finite_state
                && row.positive_density
                && row.nonnegative_population
                && row.mass_within_budget
                && row.mach_within_budget
                && row.minimum_density.is_finite()
                && row.minimum_density > 0.0,
            "population assessment failed"
        );
        anyhow::ensure!(
            row.mass_error.is_finite()
                && row.mass_error >= 0.0
                && row.mass_error <= mass_budget
                && row.max_mach.is_finite()
                && row.max_mach >= 0.0
                && row.max_mach <= mach_budget
                && row.minimum_population.is_finite()
                && row.minimum_population >= 0.0,
            "invalid population diagnostics"
        );
        let expected_seed =
            if row.condition == NULL_CONDITIONS[2] || row.condition == NULL_CONDITIONS[3] {
                Some(trial_seed(seed, row.trial))
            } else {
                None
            };
        anyhow::ensure!(row.seed == expected_seed, "unpaired trial seed");
    }
    anyhow::ensure!(
        trials > 0 && observed == expected,
        "missing, extra or wrongly indexed trial rows"
    );
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn null_predicates(
    results: &[NullTrialResult],
    trials: usize,
    seed: u64,
    steps: usize,
    mass_budget: f64,
    mach_budget: f64,
) -> anyhow::Result<[bool; 6]> {
    validate_trial_rows(results, trials, seed, steps, mass_budget, mach_budget)?;
    let mean = |condition: &str| {
        let values: Vec<_> = results
            .iter()
            .filter(|row| row.condition == condition)
            .map(|row| row.df_final)
            .collect();
        values.iter().sum::<f64>() / values.len() as f64
    };
    Ok([
        mean(NULL_CONDITIONS[0]) > 2.95,
        (mean(NULL_CONDITIONS[1]) - 2.732).abs() > 0.10,
        mean(NULL_CONDITIONS[2]) > 2.90,
        (mean(NULL_CONDITIONS[3]) - 2.732).abs() > 0.10,
        (mean(NULL_CONDITIONS[4]) - 2.732).abs() > 0.10,
        (mean(NULL_CONDITIONS[5]) - 2.732).abs() < 0.07,
    ])
}

#[cfg(feature = "euclid-catalog")]
fn write_null_rows(results: &[NullTrialResult], output: &mut impl Write) -> anyhow::Result<()> {
    writeln!(
        output,
        "condition,trial,seed,df_initial,df_final,elapsed_ms,observed_step,attempted_step,mass_error,max_mach,minimum_population,minimum_density,finite_state,positive_density,nonnegative_population,mass_within_budget,mach_within_budget,failure"
    )?;
    for row in results {
        writeln!(
            output,
            "{},{},{},{:.17},{:.17},{},{},{},{:.17},{:.17},{:.17},{:.17},{},{},{},{},{},\"{}\"",
            row.condition,
            row.trial,
            row.seed.map(|value| value.to_string()).unwrap_or_default(),
            row.df_initial,
            row.df_final,
            row.elapsed_ms,
            row.observed_step,
            row.attempted_step,
            row.mass_error,
            row.max_mach,
            row.minimum_population,
            row.minimum_density,
            row.finite_state,
            row.positive_density,
            row.nonnegative_population,
            row.mass_within_budget,
            row.mach_within_budget,
            row.failure.as_deref().unwrap_or("").replace('"', "\"\"")
        )?;
    }
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn report_null_hypothesis(
    results: &[NullTrialResult],
    trials: usize,
    seed: u64,
    steps: usize,
    mass_budget: f64,
    mach_budget: f64,
) -> anyhow::Result<()> {
    write_null_rows(results, &mut std::io::stdout().lock())?;
    let predicates = null_predicates(results, trials, seed, steps, mass_budget, mach_budget)?;
    for (index, passed) in predicates.iter().enumerate() {
        eprintln!(
            "condition={} historical_predicate_pass={passed}",
            NULL_CONDITIONS[index]
        );
    }
    let paired: Vec<_> = (0..trials)
        .map(|trial| {
            let value = |condition: &str| {
                results
                    .iter()
                    .find(|row| row.condition == condition && row.trial == trial)
                    .unwrap()
                    .df_final
            };
            value(NULL_CONDITIONS[3]) - value(NULL_CONDITIONS[2])
        })
        .collect();
    let mean = paired.iter().sum::<f64>() / trials as f64;
    let sd = (trials > 1).then(|| {
        (paired
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (trials - 1) as f64)
            .sqrt()
    });
    let galaxy = |condition: &str| {
        results
            .iter()
            .find(|row| row.condition == condition)
            .unwrap()
            .df_final
    };
    eprintln!(
        "paired_noise_zd_increment_mean={mean:.17} paired_sample_sd={} confidence_interval=unassessed sersic_zd_increment={:.17} sersic_nfw_increment={:.17} physical_interpretation=unassessed convergence=unassessed",
        sd.map(|value| value.to_string())
            .unwrap_or_else(|| "unassessed_single_pair".into()),
        galaxy(NULL_CONDITIONS[5]) - galaxy(NULL_CONDITIONS[6]),
        galaxy(NULL_CONDITIONS[6]) - galaxy(NULL_CONDITIONS[4])
    );
    anyhow::ensure!(
        predicates.iter().all(|passed| *passed),
        "historical numerical control predicates failed"
    );
    Ok(())
}

#[cfg(all(test, feature = "euclid-catalog"))]
mod control_instrument_tests {
    use super::*;
    fn passing_rows() -> Vec<NullTrialResult> {
        NULL_CONDITIONS
            .iter()
            .enumerate()
            .map(|(index, condition)| NullTrialResult {
                condition: (*condition).into(),
                trial: 0,
                seed: [2, 3].contains(&index).then_some(42),
                df_initial: 3.0,
                df_final: if index == 5 { 2.732 } else { 3.0 },
                elapsed_ms: 0,
                observed_step: 24,
                attempted_step: 24,
                mass_error: 0.0,
                max_mach: 0.0,
                minimum_population: 0.01,
                minimum_density: 1.0,
                finite_state: true,
                positive_density: true,
                nonnegative_population: true,
                mass_within_budget: true,
                mach_within_budget: true,
                failure: None,
            })
            .collect()
    }
    fn outcome(rows: &[NullTrialResult]) -> anyhow::Result<[bool; 6]> {
        null_predicates(rows, 1, 42, 24, 1e-5, 0.3)
    }
    #[test]
    fn exact_rows_pairing_and_invalid_states_are_required() {
        let baseline = passing_rows();
        assert!(outcome(&baseline).unwrap().iter().all(|passed| *passed));
        for missing in 0..7 {
            let mut changed = baseline.clone();
            changed.remove(missing);
            assert!(outcome(&changed).is_err());
        }
        let mut changed = baseline.clone();
        changed.push(changed[0].clone());
        assert!(outcome(&changed).is_err());
        let mut changed = baseline.clone();
        changed[3].seed = Some(1042);
        assert!(outcome(&changed).is_err());
        for value in [f64::NAN, f64::INFINITY, -1.0, 3.1] {
            let mut changed = baseline.clone();
            changed[1].df_final = value;
            assert!(outcome(&changed).is_err());
        }
        let mut changed = baseline.clone();
        changed[0].mass_error = 1e113;
        assert!(outcome(&changed).is_err());
        let mut changed = baseline.clone();
        changed[0].minimum_population = -1e-9;
        assert!(outcome(&changed).is_err());
        let mut changed = baseline.clone();
        changed[0].max_mach = 0.31;
        assert!(outcome(&changed).is_err());
        let mut changed = baseline.clone();
        changed[0].observed_step = 23;
        assert!(outcome(&changed).is_err());
        let mut changed = baseline;
        changed[5].df_final = 3.0;
        assert!(!outcome(&changed).unwrap()[5]);
        assert!(report_null_hypothesis(&changed, 1, 42, 24, 1e-5, 0.3).is_err());
    }
    #[test]
    fn paired_seeds_recreate_exact_density_and_budgets_are_mandatory() {
        for trial in 0..10 {
            let seed = trial_seed(42, trial);
            assert_eq!(seed, 42 + trial as u64);
            assert_eq!(
                generate_white_noise(64, 0.5, 1.5, seed),
                generate_white_noise(64, 0.5, 1.5, seed)
            );
        }
        assert!(
            Args::try_parse_from(["sweep", "--catalog", "fixture", "--null-hypothesis"]).is_err()
        );
    }

    #[test]
    fn failed_attempt_still_serializes_every_planned_row() {
        let baseline = passing_rows();
        let mut rows: Vec<_> = baseline
            .iter()
            .filter(|row| row.seed.is_none())
            .cloned()
            .collect();
        for trial in 0..10 {
            for template in [&baseline[2], &baseline[3]] {
                let mut row = template.clone();
                row.trial = trial;
                row.seed = Some(trial_seed(42, trial));
                rows.push(row);
            }
        }
        rows[0].attempted_step = 1;
        rows[0].observed_step = 0;
        rows[0].failure = Some("post-step population readback failed".into());
        rows[0].df_final = f64::NAN;
        let mut bytes = Vec::new();
        write_null_rows(&rows, &mut bytes).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(text.lines().count(), 26);
        let columns: Vec<_> = text.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(columns[6], "0");
        assert_eq!(columns[7], "1");
        assert!(text.contains("post-step population readback failed"));
        assert!(null_predicates(&rows, 10, 42, 24, 1e-5, 0.3).is_err());
    }
}

#[cfg(all(test, feature = "euclid-catalog"))]
mod retained_input_tests {
    use super::*;
    #[test]
    fn exact_arrays_receipts_and_exclusive_creation_agree() {
        use sha2::{Digest, Sha256};
        let directory = std::env::temp_dir().join(format!(
            "null-input-receipt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rho = [1.0, -0.0, 0.5];
        let force = [[0.1, 0.2, -0.3]; 3];
        let metadata = serde_json::json!({"condition":"fixture","trial":0,"seed":42,"selected_object_id":"123"});
        retain_trial_inputs(&directory, &rho, &force, metadata.clone()).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(directory.join("input.json")).unwrap())
                .unwrap();
        assert_eq!(receipt["seed"], 42);
        assert_eq!(receipt["selected_object_id"], "123");
        for (name, expected) in [
            ("rho.f64le", rho.to_vec()),
            ("force.xyz.f64le", force.into_iter().flatten().collect()),
        ] {
            let bytes = std::fs::read(directory.join(name)).unwrap();
            let decoded: Vec<_> = bytes
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            assert_eq!(
                decoded
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            let digest = Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(receipt[name]["sha256"], digest);
            assert_eq!(receipt[name]["bytes"], bytes.len());
        }
        let before = std::fs::read(directory.join("rho.f64le")).unwrap();
        assert!(retain_trial_inputs(&directory, &[9.0; 3], &force, metadata).is_err());
        assert_eq!(std::fs::read(directory.join("rho.f64le")).unwrap(), before);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
