//! lbm-rca-probe: Root Cause Analysis for LBM divergence; companion of the E-167 stability envelope.
//!
//! Instruments the LBM solver at the per-step level to identify the exact
//! mechanism driving instability at high density contrast.
//!
//! Two-phase diagnostic strategy:
//!
//! Phase 1 (positivity sentinel, cheap): at each step, compute min(f_i)
//! across all cells and max(Ma). When any f_i < 0, halt and record the
//! violation cell, direction index, local rho, and local Ma. This fires
//! 10-50 steps before the Mach-million blowup observed in E-167.
//!
//! Phase 2 (spatial/entropy analysis): re-run from step 0. At the
//! violation step, compute the normalized radial distance of the first
//! violation cell from the domain center (r_norm = dist / (n/2)). Values
//! near 0 indicate the high-rho gradient shell (compressibility breakdown);
//! values near 1 indicate the domain boundary (periodic wrap artifact).
//! Also compute relative entropy H_rel = sum_i f_i ln(f_i/f_i^eq) in the
//! neighborhood of the violation cell.
//!
//! Output:
//! - Step trajectory CSV (step, min_f, max_ma, stable) to --csv-out
//! - Violation summary to stdout
//! - Spatial analysis to stdout

use cd_kernel::batch_sliding_associator_norms_parallel;
use clap::{Parser, ValueEnum};
use lbm_3d::{
    lattice::D3Q19Lattice,
    solver::{CollisionMode, LbmSolver3D, aosoa_idx},
};

/// Collision mode flag for CLI.
#[derive(Clone, Copy, ValueEnum, Debug)]
enum Mode {
    Bgk,
    Mrt,
}

impl Mode {
    fn into_collision_mode(self) -> CollisionMode {
        match self {
            Mode::Bgk => CollisionMode::Bgk,
            Mode::Mrt => CollisionMode::Mrt,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Mode::Bgk => "BGK",
            Mode::Mrt => "MRT",
        }
    }
}

#[derive(Parser)]
#[command(name = "lbm-rca-probe")]
#[command(
    about = "Root Cause Analysis: per-step f_i positivity + entropy diagnostics for LBM divergence"
)]
struct Args {
    /// Grid size (cubic: n x n x n)
    #[arg(long, default_value = "16")]
    n: usize,

    /// Relaxation time
    #[arg(long, default_value = "0.7")]
    tau: f64,

    /// Density contrast (rho_peak / rho_background)
    #[arg(long, default_value = "23.0")]
    contrast: f64,

    /// Collision mode
    #[arg(long, value_enum, default_value = "bgk")]
    mode: Mode,

    /// Maximum number of steps (halts early on violation)
    #[arg(long, default_value = "400")]
    steps: usize,

    /// Write per-step trajectory CSV to this path
    #[arg(long)]
    csv_out: Option<String>,

    /// Neighborhood half-width for entropy analysis (cells)
    #[arg(long, default_value = "3")]
    entropy_radius: usize,

    /// Enable spatial CD associator probe at violation site (C-1591 test)
    #[arg(long)]
    cd_probe: bool,

    /// Embedding dimension for CD probe (must be power of 2, >= 32)
    #[arg(long, default_value_t = 32)]
    cd_dim: usize,
}

/// Per-step snapshot of global diagnostics.
struct StepSnapshot {
    step: usize,
    min_f: f64,
    max_ma: f64,
    stable: bool,
}

/// Violation record: first cell where f_i < 0.
struct ViolationRecord {
    step: usize,
    ix: usize,
    iy: usize,
    iz: usize,
    direction: usize,
    f_value: f64,
    local_rho: f64,
    local_ma: f64,
    /// Normalized radial distance from domain center: dist / (n/2).
    /// 0 = domain center (Gaussian peak), 1 = domain edge.
    r_norm: f64,
}

/// Build Gaussian density perturbation centered on the grid.
fn gaussian_density(n: usize, contrast: f64) -> Vec<f64> {
    let center = (n / 2) as f64;
    let sigma = (n as f64) / 6.0;
    let n_cells = n * n * n;
    let mut rho = Vec::with_capacity(n_cells);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let dx = ix as f64 - center;
                let dy = iy as f64 - center;
                let dz = iz as f64 - center;
                let r2 = dx * dx + dy * dy + dz * dz;
                rho.push(1.0 + (contrast - 1.0) * (-r2 / (2.0 * sigma * sigma)).exp());
            }
        }
    }
    rho
}

/// Build and initialize a fresh solver at the given contrast.
fn fresh_solver(n: usize, tau: f64, mode: CollisionMode, contrast: f64) -> LbmSolver3D {
    let mut solver = match mode {
        CollisionMode::Bgk => LbmSolver3D::new(n, n, n, tau),
        CollisionMode::Mrt => LbmSolver3D::new_mrt(n, n, n, tau),
    };
    let rho = gaussian_density(n, contrast);
    let u_zero = vec![[0.0_f64; 3]; n * n * n];
    // Set macroscopic fields then re-initialize f to equilibrium.
    solver.rho.copy_from_slice(&rho);
    solver.u.copy_from_slice(&u_zero);
    solver.reinitialize_from_macroscopic();
    solver
}

/// Extract per-cell distribution function as `[f64; 19]`.
fn get_f_cell(solver: &LbmSolver3D, cell: usize) -> [f64; 19] {
    let mut f = [0.0; 19];
    for (dir, slot) in f.iter_mut().enumerate() {
        *slot = solver.f[aosoa_idx(cell, dir)];
    }
    f
}

/// Compute min(f_i) across all domain cells.
fn global_min_f(solver: &LbmSolver3D) -> f64 {
    let n = solver.nx * solver.ny * solver.nz;
    let mut min_val = f64::INFINITY;
    for cell in 0..n {
        for dir in 0..19 {
            let v = solver.f[aosoa_idx(cell, dir)];
            if v < min_val {
                min_val = v;
            }
        }
    }
    min_val
}

/// Find the first (cell, direction) where f_i < 0.
fn find_first_violation(solver: &LbmSolver3D) -> Option<(usize, usize, f64)> {
    let n = solver.nx * solver.ny * solver.nz;
    for cell in 0..n {
        for dir in 0..19 {
            let v = solver.f[aosoa_idx(cell, dir)];
            if v < 0.0 {
                return Some((cell, dir, v));
            }
        }
    }
    None
}

/// Decode a linear cell index into (ix, iy, iz) for an n x n x n grid.
fn decode_cell(cell: usize, n: usize) -> (usize, usize, usize) {
    let iz = cell / (n * n);
    let iy = (cell % (n * n)) / n;
    let ix = cell % n;
    (ix, iy, iz)
}

/// Normalized radial distance from domain center.
fn r_norm(ix: usize, iy: usize, iz: usize, n: usize) -> f64 {
    let center = (n as f64) / 2.0;
    let dx = ix as f64 - center;
    let dy = iy as f64 - center;
    let dz = iz as f64 - center;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    dist / center
}

/// Relative entropy H_rel = sum_i f_i * ln(f_i / f_i^eq) for a single cell.
///
/// Returns NaN if any f_i <= 0 (distribution is already non-physical).
/// f_i^eq is computed from the cell's macroscopic (rho, u).
fn relative_entropy_cell(f: &[f64; 19], rho: f64, u: [f64; 3], lattice: &D3Q19Lattice) -> f64 {
    let mut h = 0.0;
    for (dir, &fi) in f.iter().enumerate() {
        let fi_eq = lattice.equilibrium(rho, u, dir);
        if fi <= 0.0 || fi_eq <= 0.0 {
            return f64::NAN;
        }
        h += fi * (fi / fi_eq).ln();
    }
    h
}

/// Compute relative entropy in the neighborhood of (cx, cy, cz).
///
/// Returns a Vec of (ix, iy, iz, r_norm, h_rel) sorted by r_norm.
fn entropy_neighborhood(
    solver: &LbmSolver3D,
    cx: usize,
    cy: usize,
    cz: usize,
    radius: usize,
    lattice: &D3Q19Lattice,
) -> Vec<(usize, usize, usize, f64, f64)> {
    let n = solver.nx;
    let mut entries = Vec::new();
    let lo = |c: usize| c.saturating_sub(radius);
    let hi = |c: usize| (c + radius + 1).min(n);

    for iz in lo(cz)..hi(cz) {
        for iy in lo(cy)..hi(cy) {
            for ix in lo(cx)..hi(cx) {
                let cell = iz * n * n + iy * n + ix;
                let f = get_f_cell(solver, cell);
                let rho = solver.rho[cell];
                let u = solver.u[cell];
                let h = relative_entropy_cell(&f, rho, u, lattice);
                let r = r_norm(ix, iy, iz, n);
                entries.push((ix, iy, iz, r, h));
            }
        }
    }
    entries.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    entries
}

fn main() {
    let args = Args::parse();
    let n = args.n;
    let tau = args.tau;
    let contrast = args.contrast;
    let mode = args.mode;
    let coll = mode.into_collision_mode();
    let cs = (1.0_f64 / 3.0_f64).sqrt();
    let lattice = D3Q19Lattice::new();

    eprintln!(
        "lbm-rca-probe: n={n}, tau={tau}, contrast={contrast:.1}, mode={}, steps={}",
        mode.as_str(),
        args.steps
    );
    eprintln!("Phase 1: per-step positivity sentinel");

    // ---- Phase 1: run and collect per-step snapshots ----
    let mut solver = fresh_solver(n, tau, coll, contrast);
    let mut snapshots: Vec<StepSnapshot> = Vec::with_capacity(args.steps);
    let mut violation: Option<ViolationRecord> = None;

    for step in 0..args.steps {
        // One LBM timestep: collision then streaming.
        let _ = solver.phase1_collision();
        let _ = solver.phase2_streaming();

        let min_f = global_min_f(&solver);
        let max_ma = solver.max_mach_number();
        let stable = min_f >= 0.0;

        snapshots.push(StepSnapshot {
            step,
            min_f,
            max_ma,
            stable,
        });

        if !stable && violation.is_none() {
            if let Some((cell, dir, f_val)) = find_first_violation(&solver) {
                let (ix, iy, iz) = decode_cell(cell, n);
                let rho = solver.rho[cell];
                let u = solver.u[cell];
                let local_ma = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt() / cs;
                let rn = r_norm(ix, iy, iz, n);
                violation = Some(ViolationRecord {
                    step: step + 1,
                    ix,
                    iy,
                    iz,
                    direction: dir,
                    f_value: f_val,
                    local_rho: rho,
                    local_ma,
                    r_norm: rn,
                });
            }
            // Continue for 5 more steps to see how quickly it deteriorates.
            if step + 1 >= violation.as_ref().map_or(0, |v| v.step) + 5 {
                break;
            }
        }

        if !max_ma.is_finite() {
            break;
        }
    }

    // ---- Write trajectory CSV ----
    if let Some(ref path) = args.csv_out {
        let mut lines = vec!["step,min_f,max_ma,stable".to_string()];
        for s in &snapshots {
            lines.push(format!(
                "{},{:.6e},{:.6},{}",
                s.step, s.min_f, s.max_ma, s.stable
            ));
        }
        std::fs::write(path, lines.join("\n") + "\n").expect("failed to write trajectory CSV");
        eprintln!("Trajectory CSV written: {path} ({} rows)", snapshots.len());
    }

    // ---- Print trajectory to stdout ----
    println!("step,min_f,max_ma,stable");
    for s in &snapshots {
        println!("{},{:.6e},{:.6},{}", s.step, s.min_f, s.max_ma, s.stable);
    }

    // ---- Phase 1 summary ----
    eprintln!("\n--- Phase 1 Summary ---");
    if let Some(ref v) = violation {
        eprintln!(
            "Positivity violation at step {}: cell=({},{},{}) dir={} f={:.4e}",
            v.step, v.ix, v.iy, v.iz, v.direction, v.f_value
        );
        eprintln!(
            "  local_rho={:.4}  local_Ma={:.4}  r_norm={:.3}",
            v.local_rho, v.local_ma, v.r_norm
        );
        let diagnosis = if v.r_norm < 0.5 {
            "HIGH-RHO GRADIENT SHELL (r_norm<0.5) -> compressibility breakdown"
        } else {
            "DOMAIN BOUNDARY REGION (r_norm>=0.5) -> periodic wrap artifact"
        };
        eprintln!("  Spatial diagnosis: {diagnosis}");

        // Check causality: was Ma > 0.3 before the violation?
        let ma_exceeded_before = snapshots[..v.step.saturating_sub(1)]
            .iter()
            .any(|s| s.max_ma > 0.3);
        eprintln!(
            "  Ma > 0.3 before violation: {}  (Ma causally precedes f_i<0: {})",
            ma_exceeded_before,
            if ma_exceeded_before {
                "YES -- compressibility truncation drove the violation"
            } else {
                "NO -- violation may have a different origin"
            }
        );
    } else {
        eprintln!(
            "No positivity violation in {} steps (min_f={:.4e}, max_Ma={:.4})",
            snapshots.len(),
            snapshots
                .iter()
                .map(|s| s.min_f)
                .fold(f64::INFINITY, f64::min),
            snapshots.iter().map(|s| s.max_ma).fold(0.0_f64, f64::max),
        );
    }

    // ---- Phase 2: entropy neighborhood ----
    if let Some(ref v) = violation {
        eprintln!(
            "\n--- Phase 2: Relative Entropy Neighborhood (radius={}) ---",
            args.entropy_radius
        );
        eprintln!(
            "Re-running to violation step {} for entropy snapshot...",
            v.step
        );

        let mut solver2 = fresh_solver(n, tau, coll, contrast);
        for _ in 0..v.step {
            let _ = solver2.phase1_collision();
            let _ = solver2.phase2_streaming();
        }

        let neighbors =
            entropy_neighborhood(&solver2, v.ix, v.iy, v.iz, args.entropy_radius, &lattice);

        eprintln!("  cell (ix,iy,iz)   r_norm   H_rel");
        for (ix, iy, iz, r, h) in &neighbors {
            let h_str = if h.is_nan() {
                "NaN (f_i<=0)".to_string()
            } else {
                format!("{h:.4e}")
            };
            eprintln!("  ({ix:2},{iy:2},{iz:2})          {r:.3}    {h_str}");
        }

        // Summarize: are the highest H_rel values near r_norm ~ 0 (center) or ~ 1 (edge)?
        let valid: Vec<_> = neighbors.iter().filter(|e| e.4.is_finite()).collect();
        if !valid.is_empty() {
            let max_h = valid.iter().map(|e| e.4).fold(f64::NEG_INFINITY, f64::max);
            let max_h_entry = valid
                .iter()
                .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
                .unwrap();
            eprintln!(
                "\n  Peak H_rel={:.4e} at ({},{},{}) r_norm={:.3}",
                max_h, max_h_entry.0, max_h_entry.1, max_h_entry.2, max_h_entry.3
            );
            let entropy_diagnosis = if max_h_entry.3 < 0.5 {
                "PEAK ENTROPY AT CENTER -> consistent with O(Ma^2) truncation error"
            } else {
                "PEAK ENTROPY AT BOUNDARY -> consistent with periodic wrap reflection"
            };
            eprintln!("  Entropy diagnosis: {entropy_diagnosis}");
        }
    }

    // ---- Phase 3 (optional): CD spatial associator probe (C-1591) ----
    if args.cd_probe {
        if let Some(ref v) = violation {
            let cd_dim = args.cd_dim;
            assert!(
                cd_dim.is_power_of_two() && cd_dim >= 19,
                "--cd-dim must be a power of 2 and >= 19 (got {cd_dim})"
            );

            eprintln!("\n--- Phase 3: CD Spatial Associator Probe (C-1591) ---");
            eprintln!(
                "cd_dim={cd_dim}, violation_step={}, site=({},{},{})",
                v.step, v.ix, v.iy, v.iz
            );

            // 6 face neighbors + center cell in D3Q19 cardinal directions.
            // Periodic wrapping via rem_euclid matches LBM boundary conditions.
            let face_offsets: [[i32; 3]; 7] = [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ];

            let mut solver3 = fresh_solver(n, tau, coll, contrast);

            // temporal_viol[step] = cd_dim-vector from violation cell at that step.
            let mut temporal_viol: Vec<Vec<f64>> = Vec::with_capacity(v.step + 1);
            // per_step_spatial_means[step] = mean associator norm across 5 spatial triples.
            let mut per_step_spatial_means: Vec<(usize, f64)> = Vec::with_capacity(v.step + 1);

            /// Pad 19 f_i values into a cd_dim-length unit-norm vector.
            /// Entries 19..cd_dim are zero (padding).
            fn pad_normalize(f: &[f64; 19], cd_dim: usize) -> Vec<f64> {
                let mut vec = vec![0.0f64; cd_dim];
                for (i, &fi) in f.iter().enumerate() {
                    vec[i] = fi;
                }
                let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    for x in vec.iter_mut() {
                        *x /= norm;
                    }
                }
                vec
            }

            // Collect initial state (step 0) before any stepping.
            {
                let viol_cell = v.iz * n * n + v.iy * n + v.ix;
                temporal_viol.push(pad_normalize(&get_f_cell(&solver3, viol_cell), cd_dim));

                let spatial_vecs: Vec<Vec<f64>> = face_offsets
                    .iter()
                    .map(|off| {
                        let ix2 = ((v.ix as i32 + off[0]).rem_euclid(n as i32)) as usize;
                        let iy2 = ((v.iy as i32 + off[1]).rem_euclid(n as i32)) as usize;
                        let iz2 = ((v.iz as i32 + off[2]).rem_euclid(n as i32)) as usize;
                        pad_normalize(&get_f_cell(&solver3, iz2 * n * n + iy2 * n + ix2), cd_dim)
                    })
                    .collect();
                let sn = batch_sliding_associator_norms_parallel(&spatial_vecs, cd_dim);
                let mean = sn.iter().sum::<f64>() / sn.len().max(1) as f64;
                per_step_spatial_means.push((0, mean));
            }

            // Step forward collecting state after each collision+streaming.
            for step in 1..=v.step {
                let _ = solver3.phase1_collision();
                let _ = solver3.phase2_streaming();

                let viol_cell = v.iz * n * n + v.iy * n + v.ix;
                temporal_viol.push(pad_normalize(&get_f_cell(&solver3, viol_cell), cd_dim));

                let spatial_vecs: Vec<Vec<f64>> = face_offsets
                    .iter()
                    .map(|off| {
                        let ix2 = ((v.ix as i32 + off[0]).rem_euclid(n as i32)) as usize;
                        let iy2 = ((v.iy as i32 + off[1]).rem_euclid(n as i32)) as usize;
                        let iz2 = ((v.iz as i32 + off[2]).rem_euclid(n as i32)) as usize;
                        pad_normalize(&get_f_cell(&solver3, iz2 * n * n + iy2 * n + ix2), cd_dim)
                    })
                    .collect();
                let sn = batch_sliding_associator_norms_parallel(&spatial_vecs, cd_dim);
                let mean = sn.iter().sum::<f64>() / sn.len().max(1) as f64;
                per_step_spatial_means.push((step, mean));
            }

            // Print per-step spatial means.
            eprintln!("  step  cd_spatial_mean_assoc");
            for (step, mean) in &per_step_spatial_means {
                eprintln!("  {step:4}  {mean:.6e}");
            }

            // Compute temporal norms from violation-cell sequence.
            let temporal_norms = batch_sliding_associator_norms_parallel(&temporal_viol, cd_dim);
            eprintln!(
                "\n  Temporal CD norms at violation cell ({} norms from {} steps):",
                temporal_norms.len(),
                temporal_viol.len()
            );
            for (i, &tn) in temporal_norms.iter().enumerate() {
                eprintln!("  triple [{}-{}]: {:.6e}", i, i + 2, tn);
            }

            // C-1591 verdict: does temporal associator increase toward violation?
            if temporal_norms.len() >= 2 {
                let half = temporal_norms.len() / 2;
                let early: f64 = temporal_norms[..half].iter().sum::<f64>() / half as f64;
                let late: f64 = temporal_norms[half..].iter().sum::<f64>()
                    / (temporal_norms.len() - half) as f64;
                eprintln!("\n  C-1591 VERDICT:");
                eprintln!(
                    "  Early-half mean={:.4e}  Late-half mean={:.4e}",
                    early, late
                );
                if late > early * 1.1 {
                    eprintln!(
                        "  SUPPORTED: temporal CD norm grows toward violation ({:.2}x increase)",
                        late / early.max(1e-20)
                    );
                } else {
                    eprintln!(
                        "  NOT SUPPORTED: no monotonic increase (ratio={:.3})",
                        late / early.max(1e-20)
                    );
                }
            }
        } else {
            eprintln!("\n--- Phase 3: CD probe skipped (no violation found) ---");
        }
    }
}
