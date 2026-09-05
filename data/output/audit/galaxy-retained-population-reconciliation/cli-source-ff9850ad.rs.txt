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

    /// Run null-hypothesis control experiment (6 conditions, C0-C5).
    /// Tests whether D_f=2.73 is a genuine morphology+topology signal
    /// or a pipeline artifact.  Requires --catalog for C4/C5.
    #[arg(long)]
    null_hypothesis: bool,

    /// Noise realizations per stochastic condition (C2, C3).
    #[arg(long, default_value = "10")]
    null_n_trials: usize,

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
        if use_gpu {
            #[cfg(feature = "gpu")]
            run_null_hypothesis_gpu(args, config);
            #[cfg(not(feature = "gpu"))]
            {
                eprintln!("WARNING: GPU not enabled, using CPU for null hypothesis");
                run_null_hypothesis_cpu(args, config);
            }
        } else {
            run_null_hypothesis_cpu(args, config);
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
                    for dir in 0..19 {
                        solver.f[lbm_3d::solver::aosoa_idx(idx, dir)] = f_eq[dir];
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
// Null hypothesis control experiment (Phase 14)
// ---------------------------------------------------------------------------

/// Result of one null-hypothesis trial.
#[cfg(feature = "euclid-catalog")]
struct NullTrialResult {
    condition: String,
    trial: usize,
    df_initial: f64,
    df_final: f64,
    elapsed_ms: u64,
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

#[cfg(all(feature = "euclid-catalog", feature = "gpu"))]
fn run_null_hypothesis_gpu(args: &Args, config: &GalaxyPipelineConfig) {
    use lbm_3d_cuda::{LbmSolver3DCuda, Precision, box_counting_gpu::GpuBoxCounter};

    let g = config.grid_dim;
    let n_cells = g * g * g;
    let steps = config.lbm_steps;
    let smag = args.smagorinsky_cs;
    let use_smag = smag > 0.0;
    let tau = config.tau;
    let dx = config.dx_kpc;

    // Create GPU solver (reused across all conditions via initialize_custom)
    let mut solver = if args.mrt {
        LbmSolver3DCuda::new_mrt(g, g, g, tau, Precision::FP32)
    } else {
        LbmSolver3DCuda::new(g, g, g, tau, Precision::FP32)
    }
    .unwrap_or_else(|e| {
        eprintln!("ERROR: CUDA solver init failed: {e}");
        std::process::exit(1);
    });

    if args.tiling {
        solver.set_tiling(true);
    }

    if let Err(e) = solver.set_l2_pinning(true) {
        eprintln!("WARNING: L2 pinning failed: {e} (non-fatal, continuing without)");
    }

    let mut box_counter = GpuBoxCounter::new(solver.context()).unwrap_or_else(|e| {
        eprintln!("ERROR: GpuBoxCounter init failed: {e}");
        std::process::exit(1);
    });

    let zeros_u = vec![[0.0_f64; 3]; n_cells];
    let zeros_f: Vec<[f64; 3]> = vec![[0.0; 3]; n_cells];

    // Load first valid galaxy for C4/C5 and R_e reference
    let entries = read_euclid_physical_measurements(&args.catalog).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });
    let first_setup = entries
        .iter()
        .find_map(|e| prepare_galaxy(e, config))
        .unwrap_or_else(|| {
            eprintln!("ERROR: no valid galaxy in catalog for positive control");
            std::process::exit(1);
        });
    let r_e_kpc = first_setup.record_template.r_e_kpc;

    eprintln!(
        "=== Null Hypothesis Control (GPU) ===\n\
         Reference galaxy: obj={}, R_e={:.2} kpc, type={}",
        first_setup.record_template.object_id,
        r_e_kpc,
        first_setup.record_template.morphological_type,
    );

    // ZD force field (standalone, no NFW component)
    let zd_force = if config.alpha_zd > 0.0 {
        compute_zd_force_standalone(g, config.dx_kpc, r_e_kpc, config.alpha_zd)
    } else {
        eprintln!("WARNING: alpha_zd=0 -- C1/C3 degenerate to C0/C2 (no ZD forcing)");
        zeros_f.clone()
    };

    let rho_uniform: Vec<f64> = vec![1.0; n_cells];

    // Closure: run one trial and return result.
    // Captures &mut solver, &mut box_counter, &zeros_u, and Copy scalars.
    // Arguments (rho, force) are NOT captured -- passed per call.
    let mut run_trial =
        |rho: &[f64], force: &[[f64; 3]], label: String, trial: usize| -> NullTrialResult {
            solver
                .initialize_custom(rho, &zeros_u)
                .expect("init failed");
            solver.set_force_field(force).expect("force failed");

            let df_initial = box_counter
                .fractal_dimension_device_auto(solver.d_rho_bytes(), g, g, g)
                .map(|r| r.d_f)
                .unwrap_or(3.0);

            let t0 = Instant::now();
            if use_smag {
                for step in 0..steps {
                    if step % 10 == 0 {
                        let _ = solver.update_smagorinsky_tau(smag, dx, tau);
                    }
                    solver.step().expect("LBM step");
                }
            } else {
                solver.step_n(steps).expect("LBM step_n");
            }

            let df_final = box_counter
                .fractal_dimension_device_auto(solver.d_rho_bytes(), g, g, g)
                .map(|r| r.d_f)
                .unwrap_or(3.0);

            let elapsed = t0.elapsed().as_millis() as u64;
            eprintln!("  {label}[{trial}]: D_f = {df_initial:.4} -> {df_final:.4} ({elapsed}ms)");
            NullTrialResult {
                condition: label,
                trial,
                df_initial,
                df_final,
                elapsed_ms: elapsed,
            }
        };

    let mut results: Vec<NullTrialResult> = Vec::new();

    // C0: Uniform density + zero force
    results.push(run_trial(
        &rho_uniform,
        &zeros_f,
        "C0-uniform-zero".into(),
        0,
    ));

    // C1: Uniform density + ZD force only (no NFW gravity)
    results.push(run_trial(
        &rho_uniform,
        &zd_force,
        "C1-uniform-fzd".into(),
        0,
    ));

    // C2: White noise density + zero force (N trials)
    for trial in 0..args.null_n_trials {
        let rho_noise = generate_white_noise(n_cells, 0.5, 1.5, trial as u64);
        results.push(run_trial(
            &rho_noise,
            &zeros_f,
            "C2-noise-zero".into(),
            trial,
        ));
    }

    // C3: White noise density + ZD force (N trials)
    for trial in 0..args.null_n_trials {
        let rho_noise = generate_white_noise(n_cells, 0.5, 1.5, trial as u64 + 1000);
        results.push(run_trial(
            &rho_noise,
            &zd_force,
            "C3-noise-fzd".into(),
            trial,
        ));
    }

    // C4: Sersic galaxy density + zero force (morphology without dynamics)
    results.push(run_trial(
        &first_setup.rho_total,
        &zeros_f,
        "C4-sersic-zero".into(),
        0,
    ));

    // C5: Sersic galaxy + full force (positive control -- should match E-166)
    results.push(run_trial(
        &first_setup.rho_total,
        &first_setup.force_field,
        "C5-sersic-fzd".into(),
        0,
    ));

    drop(run_trial);
    report_null_hypothesis(&results);
}

#[cfg(feature = "euclid-catalog")]
fn run_null_hypothesis_cpu(args: &Args, config: &GalaxyPipelineConfig) {
    use lbm_3d::solver::{BgkCollision, LbmSolver3D, aosoa_idx};

    let g = config.grid_dim;
    let n_cells = g * g * g;
    let use_smag = args.smagorinsky_cs > 0.0;

    let entries = read_euclid_physical_measurements(&args.catalog).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });
    let first_setup = entries
        .iter()
        .find_map(|e| prepare_galaxy(e, config))
        .unwrap_or_else(|| {
            eprintln!("ERROR: no valid galaxy in catalog for positive control");
            std::process::exit(1);
        });
    let r_e_kpc = first_setup.record_template.r_e_kpc;

    eprintln!(
        "=== Null Hypothesis Control (CPU) ===\n\
         Reference galaxy: obj={}, R_e={:.2} kpc, type={}",
        first_setup.record_template.object_id,
        r_e_kpc,
        first_setup.record_template.morphological_type,
    );

    let zd_force = if config.alpha_zd > 0.0 {
        compute_zd_force_standalone(g, config.dx_kpc, r_e_kpc, config.alpha_zd)
    } else {
        eprintln!("WARNING: alpha_zd=0 -- C1/C3 degenerate to C0/C2 (no ZD forcing)");
        vec![[0.0; 3]; n_cells]
    };

    let zeros_f: Vec<[f64; 3]> = vec![[0.0; 3]; n_cells];
    let rho_uniform: Vec<f64> = vec![1.0; n_cells];

    let mut results: Vec<NullTrialResult> = Vec::new();

    // Helper: run one CPU trial with a fresh solver.
    // Takes force by reference and clones for set_force_field (which takes Vec).
    let run_cpu_trial = |rho: &[f64],
                         force: &[[f64; 3]],
                         label: &str,
                         trial: usize|
     -> NullTrialResult {
        let mut solver = if args.mrt {
            LbmSolver3D::new_mrt(g, g, g, config.tau)
        } else {
            LbmSolver3D::new(g, g, g, config.tau)
        };

        let lattice = &solver.collider.lattice;
        for idx in 0..n_cells {
            let f_eq = BgkCollision::initialize_rest(rho[idx], lattice);
            for (dir, &val) in f_eq.iter().enumerate() {
                solver.f[aosoa_idx(idx, dir)] = val;
            }
            solver.rho[idx] = rho[idx];
            solver.u[idx] = [0.0, 0.0, 0.0];
        }
        solver
            .set_force_field(force.to_vec())
            .expect("force size mismatch");

        let df_initial = box_counting_fractal_dim(&solver.rho, g, g, g);
        let t0 = Instant::now();

        for step in 0..config.lbm_steps {
            if use_smag && step % 10 == 0 {
                let _ =
                    solver.update_smagorinsky_tau(args.smagorinsky_cs, config.dx_kpc, config.tau);
            }
            solver.evolve_one_step();
        }

        let df_final = box_counting_fractal_dim(&solver.rho, g, g, g);
        let elapsed = t0.elapsed().as_millis() as u64;

        eprintln!("  {label}[{trial}]: D_f = {df_initial:.4} -> {df_final:.4} ({elapsed}ms)");
        NullTrialResult {
            condition: label.to_string(),
            trial,
            df_initial,
            df_final,
            elapsed_ms: elapsed,
        }
    };

    // C0: Uniform + zero force
    results.push(run_cpu_trial(&rho_uniform, &zeros_f, "C0-uniform-zero", 0));

    // C1: Uniform + ZD force only
    results.push(run_cpu_trial(&rho_uniform, &zd_force, "C1-uniform-fzd", 0));

    // C2: White noise + zero force (N trials)
    for trial in 0..args.null_n_trials {
        let rho_noise = generate_white_noise(n_cells, 0.5, 1.5, trial as u64);
        results.push(run_cpu_trial(&rho_noise, &zeros_f, "C2-noise-zero", trial));
    }

    // C3: White noise + ZD force (N trials)
    for trial in 0..args.null_n_trials {
        let rho_noise = generate_white_noise(n_cells, 0.5, 1.5, trial as u64 + 1000);
        results.push(run_cpu_trial(&rho_noise, &zd_force, "C3-noise-fzd", trial));
    }

    // C4: Sersic + zero force
    results.push(run_cpu_trial(
        &first_setup.rho_total,
        &zeros_f,
        "C4-sersic-zero",
        0,
    ));

    // C5: Sersic + full force (positive control)
    results.push(run_cpu_trial(
        &first_setup.rho_total,
        &first_setup.force_field,
        "C5-sersic-fzd",
        0,
    ));

    let _ = run_cpu_trial;
    report_null_hypothesis(&results);
}

/// Emit CSV + statistical verdict for the 6-condition null hypothesis experiment.
///
/// Pass criteria (ALL must hold for null hypothesis rejection):
/// - C0 D_f > 2.95 (uniform stays trivial)
/// - C1 |D_f - 2.73| > 0.10 (ZD alone does not produce the signal)
/// - C2 D_f > 2.90 (noise homogenizes under MRT diffusion)
/// - C3 |D_f - 2.73| > 0.10 (noise + ZD does not produce the signal)
/// - C4 |D_f - 2.73| > 0.10 (morphology alone diffuses away)
/// - C5 |D_f - 2.73| < 0.07 (positive control reproduces E-166)
#[cfg(feature = "euclid-catalog")]
fn report_null_hypothesis(results: &[NullTrialResult]) {
    // CSV output (stdout)
    println!("condition,trial,df_initial,df_final,elapsed_ms");
    for r in results {
        println!(
            "{},{},{:.4},{:.4},{}",
            r.condition, r.trial, r.df_initial, r.df_final, r.elapsed_ms
        );
    }

    // Per-condition statistics
    let e166_mean = 2.732;
    let sigma_3 = 0.10; // 3-sigma separation threshold

    let c0 = results.iter().find(|r| r.condition == "C0-uniform-zero");
    let c1 = results.iter().find(|r| r.condition == "C1-uniform-fzd");
    let c4 = results.iter().find(|r| r.condition == "C4-sersic-zero");
    let c5 = results.iter().find(|r| r.condition == "C5-sersic-fzd");

    let c2_vals: Vec<f64> = results
        .iter()
        .filter(|r| r.condition == "C2-noise-zero")
        .map(|r| r.df_final)
        .collect();
    let c3_vals: Vec<f64> = results
        .iter()
        .filter(|r| r.condition == "C3-noise-fzd")
        .map(|r| r.df_final)
        .collect();

    let mean_std = |vals: &[f64]| -> (f64, f64) {
        if vals.is_empty() {
            return (f64::NAN, 0.0);
        }
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        let v = if vals.len() < 2 {
            0.0
        } else {
            vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (vals.len() as f64 - 1.0)
        };
        (m, v.sqrt())
    };

    let (c2_mean, c2_std) = mean_std(&c2_vals);
    let (c3_mean, c3_std) = mean_std(&c3_vals);

    eprintln!("\n=== Null Hypothesis Verdict ===");
    let mut all_pass = true;

    // C0: D_f > 2.95 (uniform should stay near 3.0)
    if let Some(r) = c0 {
        let pass = r.df_final > 2.95;
        if !pass {
            all_pass = false;
        }
        eprintln!(
            "C0 uniform+zero:   D_f={:.4}  {}  (threshold: >2.95)",
            r.df_final,
            if pass { "PASS" } else { "**FAIL**" },
        );
    }

    // C1: |D_f - 2.73| > 0.10 (ZD alone must not produce 2.73)
    if let Some(r) = c1 {
        let delta = (r.df_final - e166_mean).abs();
        let pass = delta > sigma_3;
        if !pass {
            all_pass = false;
        }
        eprintln!(
            "C1 uniform+fzd:    D_f={:.4}  {}  (delta={:.3}, threshold: >{:.2})",
            r.df_final,
            if pass { "PASS" } else { "**FAIL**" },
            delta,
            sigma_3,
        );
    }

    // C2: mean D_f > 2.90 (noise homogenizes)
    {
        let pass = c2_mean > 2.90;
        if !pass {
            all_pass = false;
        }
        eprintln!(
            "C2 noise+zero:     D_f={:.4}+/-{:.4} (N={})  {}  (threshold: >2.90)",
            c2_mean,
            c2_std,
            c2_vals.len(),
            if pass { "PASS" } else { "**FAIL**" },
        );
    }

    // C3: |mean D_f - 2.73| > 0.10 (noise + ZD must not produce 2.73)
    {
        let delta = (c3_mean - e166_mean).abs();
        let pass = delta > sigma_3;
        if !pass {
            all_pass = false;
        }
        eprintln!(
            "C3 noise+fzd:      D_f={:.4}+/-{:.4} (N={})  {}  (delta={:.3}, threshold: >{:.2})",
            c3_mean,
            c3_std,
            c3_vals.len(),
            if pass { "PASS" } else { "**FAIL**" },
            delta,
            sigma_3,
        );
    }

    // C4: |D_f - 2.73| > 0.10 (morphology alone diffuses)
    if let Some(r) = c4 {
        let delta = (r.df_final - e166_mean).abs();
        let pass = delta > sigma_3;
        if !pass {
            all_pass = false;
        }
        eprintln!(
            "C4 sersic+zero:    D_f={:.4}  {}  (delta={:.3}, threshold: >{:.2})",
            r.df_final,
            if pass { "PASS" } else { "**FAIL**" },
            delta,
            sigma_3,
        );
    }

    // C5: |D_f - 2.73| < 0.07 (positive control, 2-sigma of E-166)
    if let Some(r) = c5 {
        let delta = (r.df_final - e166_mean).abs();
        let pass = delta < 0.07;
        if !pass {
            all_pass = false;
        }
        eprintln!(
            "C5 sersic+fzd:     D_f={:.4}  {}  (delta={:.3}, threshold: <0.07)",
            r.df_final,
            if pass { "PASS" } else { "**FAIL**" },
            delta,
        );
    }

    eprintln!("---");
    if all_pass {
        eprintln!("NULL HYPOTHESIS REJECTED: D_f=2.73 is genuine morphological signal.");
    } else {
        eprintln!("NULL HYPOTHESIS NOT REJECTED: Pipeline artifact detected.");
        eprintln!("INVESTIGATION REQUIRED before proceeding with 1000-galaxy sweep.");
    }
}
