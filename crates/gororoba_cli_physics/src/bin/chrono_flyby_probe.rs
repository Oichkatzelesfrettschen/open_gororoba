//! Chrono-Flyby Probe: temporal-fluid / flyby trajectory coupling with QEC syndrome detection.
//!
//! This binary is the grand unification pipeline connecting three subsystems:
//! 1. Flyby trajectory integration (Newtonian RK4 through Earth SOI)
//! 2. 3D LBM fluid with chrono-complex viscosity modulation
//! 3. Lattice QEC bridge: density anomalies -> CD algebraic syndromes
//!
//! The probe evolves a spacecraft trajectory and a co-moving fluid grid simultaneously.
//! At each snapshot interval, it extracts density/velocity anomalies from the fluid,
//! maps them to LatticeSiteState entries via Morton Z-order, feeds them through the
//! LatticeQecSnapshot syndrome detector, and emits a CSV row with topological friction
//! metrics (active syndrome count, mean syndrome strength, error mask size).
//!
//! See BIB-0307 (Moreno 1998) for CD zero-divisor structure,
//! BIB-0314 (Pastawski et al. 2015) for holographic code analogies,
//! and BIB-0297 (Anderson et al. 2008) for flyby anomaly data.

use algebra_experimental::moonshine_embedding::LEECH_DIM;
use clap::Parser;
use gororoba_cli_physics::flyby::{
    config::{GM_EARTH, R_EARTH, all_flybys},
    environment::{EarthOnlyNfwLike, State},
    geometry::{compute_soi_window, hyperbolic_initial_state},
    integrator::run_trajectory,
};
use lbm_3d::{boundary::GridIndex, dm_force::combine_forces, solver::LbmSolver3D};
use lbm_core::viscosity_with_chrono_complex;
use quantum_core::lattice_qec_bridge::{
    LatticeQecSnapshot, LatticeSiteState, VoidCodeAdapter, build_error_mask,
};
use std::{fs, path::PathBuf};

/// Chrono-flyby probe: temporal-fluid coupling with QEC syndrome detection.
///
/// Couples a spacecraft flyby trajectory with a 3D LBM fluid grid under
/// chrono-complex viscosity modulation. Extracts density anomalies and
/// maps them to Cayley-Dickson algebraic syndromes for topological
/// friction quantification.
#[derive(Parser)]
#[command(name = "chrono-flyby-probe")]
struct Cli {
    /// Spacecraft name (must match flyby database, e.g. "NEAR")
    #[arg(long, default_value = "NEAR")]
    spacecraft: String,

    /// LBM grid size in x
    #[arg(long, default_value_t = 32)]
    nx: usize,

    /// LBM grid size in y
    #[arg(long, default_value_t = 32)]
    ny: usize,

    /// LBM grid size in z
    #[arg(long, default_value_t = 32)]
    nz: usize,

    /// LBM relaxation time tau (>0.5 for stability)
    #[arg(long, default_value_t = 0.6)]
    tau: f64,

    /// Chrono coupling strength alpha (imaginary time -> viscosity modulation)
    #[arg(long, default_value_t = 0.01)]
    alpha_chrono: f64,

    /// Number of LBM timesteps per trajectory snapshot
    #[arg(long, default_value_t = 100)]
    lbm_steps_per_snap: usize,

    /// RK4 timestep for trajectory integration (seconds)
    #[arg(long, default_value_t = 10.0)]
    dt_trajectory: f64,

    /// Number of trajectory snapshots
    #[arg(long, default_value_t = 50)]
    n_snapshots: usize,

    /// Anomaly detection threshold (L2 deviation from equilibrium)
    #[arg(long, default_value_t = 0.01)]
    anomaly_threshold: f64,

    /// CD algebra dimension for syndrome mapping (16=Sedenion, 32=Pathion, 64=Chingon)
    #[arg(long, default_value_t = 32)]
    cd_dim: usize,

    /// Plummer softening length in lattice units (prevents 1/r^2 singularity).
    /// See BIB-0297 (Anderson et al. 2008) for flyby anomaly context.
    #[arg(long, default_value_t = 1.5)]
    softening: f64,

    /// Override GM in lattice units (auto-computed from grid scale if not set)
    #[arg(long)]
    gm_lattice: Option<f64>,

    /// Disable gravity wake injection (fluid stays at equilibrium)
    #[arg(long, default_value_t = false)]
    no_gravity_wake: bool,

    /// Output CSV path for topological friction report
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Keplerian acceleration (no anomalous force -- baseline Newtonian).
fn keplerian_accel(state: &State, _t: f64, _rho: f64, _tensor: &[[f64; 3]; 3]) -> [f64; 3] {
    let r2 = state.position[0] * state.position[0]
        + state.position[1] * state.position[1]
        + state.position[2] * state.position[2];
    let r = r2.sqrt();
    let r3 = r * r2;
    if r < 100.0 {
        return [0.0, 0.0, 0.0];
    }
    [
        -GM_EARTH * state.position[0] / r3,
        -GM_EARTH * state.position[1] / r3,
        -GM_EARTH * state.position[2] / r3,
    ]
}

/// Morton Z-order interleave for 3D coordinates.
///
/// Maps (x, y, z) to a single index preserving spatial locality.
/// Used to assign lattice sites to CD algebra basis indices.
fn morton_z_order(x: usize, y: usize, z: usize) -> usize {
    let mut result = 0usize;
    for bit in 0..10 {
        result |= ((x >> bit) & 1) << (3 * bit);
        result |= ((y >> bit) & 1) << (3 * bit + 1);
        result |= ((z >> bit) & 1) << (3 * bit + 2);
    }
    result
}

/// Extract density anomalies from the LBM solver.
///
/// Returns (site_index, deviation_norm, rho, ux, uy, uz) for sites where
/// |rho - 1.0| > threshold.
fn extract_anomalies(solver: &LbmSolver3D, threshold: f64) -> Vec<(usize, f64, f64, [f64; 3])> {
    let mut anomalies = Vec::new();
    for z in 0..solver.nz {
        for y in 0..solver.ny {
            for x in 0..solver.nx {
                let (rho, u) = solver.get_macroscopic(x, y, z);
                let dev = (rho - 1.0).abs();
                if dev > threshold {
                    let idx = z * (solver.nx * solver.ny) + y * solver.nx + x;
                    anomalies.push((idx, dev, rho, u));
                }
            }
        }
    }
    anomalies
}

/// Map anomalous fluid sites to LatticeSiteState entries via Morton Z-order.
fn anomalies_to_lattice_states(
    anomalies: &[(usize, f64, f64, [f64; 3])],
    nx: usize,
    ny: usize,
    cd_dim: usize,
) -> Vec<LatticeSiteState> {
    use algebra_experimental::higher_cd::SparseApeironState;

    anomalies
        .iter()
        .map(|&(idx, dev, rho, u)| {
            // Recover 3D coordinates from flat index
            let nxy = nx * ny;
            let z = idx / nxy;
            let rem = idx % nxy;
            let y = rem / nx;
            let x = rem % nx;

            // Morton Z-order maps spatial position to algebraic basis index
            let morton = morton_z_order(x, y, z) % cd_dim;

            // Build sparse CD state: velocity components mapped to neighboring axes
            let mut components = vec![(morton, rho - 1.0)];
            if u[0].abs() > 1e-12 {
                components.push(((morton + 1) % cd_dim, u[0]));
            }
            if u[1].abs() > 1e-12 {
                components.push(((morton + 2) % cd_dim, u[1]));
            }
            if u[2].abs() > 1e-12 {
                components.push(((morton + 3) % cd_dim, u[2]));
            }

            let state = SparseApeironState::from_pairs(cd_dim, components);
            LatticeSiteState::new(idx, state, dev)
        })
        .collect()
}

/// Compute gravitational acceleration field from a flyby mass on the LBM grid.
///
/// Maps the spacecraft's physical trajectory position (km) to lattice coordinates,
/// then computes Newtonian gravity with Plummer softening at every grid node.
/// The resulting Vec<[f64; 3]> is passed to `set_force_field()` for Guo forcing.
///
/// Coordinate convention: grid center (nx/2, ny/2, nz/2) = Earth center.
/// Physical-to-lattice scale: dx_km = 2 * soi_radius_km / max(nx, ny, nz).
///
/// The softening parameter prevents 1/r^2 singularity if the mass transits
/// through a lattice node. See BIB-0297 (Anderson et al. 2008) for the flyby
/// anomaly empirical formula that motivates this simulation.
fn gravity_wake_field(
    nx: usize,
    ny: usize,
    nz: usize,
    mass_lattice_pos: [f64; 3],
    gm_lattice: f64,
    softening: f64,
) -> Vec<[f64; 3]> {
    let n = nx * ny * nz;
    let mut force = vec![[0.0; 3]; n];
    let eps2 = softening * softening;

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = GridIndex::new(x, y, z).linearize(nx, ny);

                // Displacement from grid node to flyby mass (mass - node)
                // so force points FROM node TOWARD mass (attractive gravity).
                let dx = mass_lattice_pos[0] - x as f64;
                let dy = mass_lattice_pos[1] - y as f64;
                let dz = mass_lattice_pos[2] - z as f64;

                let r2 = dx * dx + dy * dy + dz * dz;
                // Plummer softening: a = -GM * r / (r^2 + eps^2)^(3/2)
                let denom = (r2 + eps2).powf(1.5);
                if denom < 1e-30 {
                    continue;
                }

                // Acceleration points FROM node TOWARD mass (attractive gravity)
                force[idx] = [
                    gm_lattice * dx / denom,
                    gm_lattice * dy / denom,
                    gm_lattice * dz / denom,
                ];
            }
        }
    }

    force
}

/// Convert physical trajectory position (km) to LBM lattice coordinates.
///
/// Earth center maps to grid center (nx/2, ny/2, nz/2).
/// Scale factor: dx_km = 2 * soi_radius / max(nx, ny, nz), so the SOI
/// window fits within the grid. The mass position in km is divided by
/// dx_km and offset to grid center.
fn physical_to_lattice(pos_km: [f64; 3], nx: usize, ny: usize, nz: usize, dx_km: f64) -> [f64; 3] {
    [
        pos_km[0] / dx_km + nx as f64 / 2.0,
        pos_km[1] / dx_km + ny as f64 / 2.0,
        pos_km[2] / dx_km + nz as f64 / 2.0,
    ]
}

/// Compute GM in lattice units by normalizing to a target peak force.
///
/// Standard LBM non-dimensionalization: instead of converting GM via raw dimensional
/// analysis (which yields ~10^-10 for Earth flyby scales due to enormous dx_km),
/// we compute the peak gravitational acceleration at the closest approach (perigee)
/// and scale GM_lattice so the peak lattice force equals `target_peak`.
///
/// This ensures the gravity wake produces O(0.001-0.005) perturbations in the LBM
/// fluid regardless of grid resolution or trajectory scale.
///
/// The peak lattice acceleration is: g_peak_lattice = GM_lattice / (r_perigee_lattice^2 + eps^2)
/// where r_perigee_lattice = r_perigee_km / dx_km (perigee distance in grid cells).
/// Setting g_peak_lattice = target_peak and solving for GM_lattice:
///   GM_lattice = target_peak * (r_perigee_lattice^2 + eps^2)
fn compute_gm_lattice(r_perigee_km: f64, dx_km: f64, softening: f64, target_peak: f64) -> f64 {
    let r_perigee_lattice = r_perigee_km / dx_km;
    let r2_eps2 = r_perigee_lattice * r_perigee_lattice + softening * softening;
    target_peak * r2_eps2
}

/// Apply chrono-complex viscosity modulation to the 3D LBM solver.
///
/// The imaginary component of the complex time step modulates viscosity
/// via the same formula used in 2D chrono_turbulence.rs:
///   nu_eff = viscosity_with_chrono_complex(nu_base, alpha, epsilon)
///   tau_eff = 3 * nu_eff + 0.5
///
/// Sets a uniform tau field across all grid points.
fn apply_chrono_viscosity(solver: &mut LbmSolver3D, alpha: f64, epsilon: f64) {
    // Read current tau from the first element of the tau field
    let tau_current = solver.collider.get_tau_field()[0];
    let nu_base = (tau_current - 0.5) / 3.0;
    let nu_eff = viscosity_with_chrono_complex(nu_base, alpha, epsilon);
    let tau_eff = 3.0 * nu_eff + 0.5;
    // Clamp tau to stability bound
    let tau_clamped = tau_eff.max(0.501);
    let n_nodes = solver.nx * solver.ny * solver.nz;
    let tau_field = vec![tau_clamped; n_nodes];
    // Infallible: length matches, values are finite and >= 0.501
    solver
        .set_viscosity_field(tau_field)
        .expect("tau field set");
}

/// Topological friction report row.
///
/// When cd_dim=32 (Pathion), syndrome axes are split into two sectors:
/// - Leech sector (axes 0..23): Standard Model / bosonic string subspace
/// - Dark sector (axes 24..31): orthogonal gauge space (octonion-dimensional)
///
/// The Leech-Pathion embedding (see moonshine_embedding.rs) predicts that
/// gravitational wake anomalies will be algebraically entangled between sectors
/// via the 71% cross-sector coupling fraction. Monitoring leech_syndromes vs
/// dark_syndromes reveals whether macroscopic fluid perturbations preferentially
/// excite topological defects in the visible or hidden sector.
struct FrictionRow {
    snap: usize,
    t_seconds: f64,
    r_km: f64,
    v_km_s: f64,
    n_anomalies: usize,
    n_syndromes_active: usize,
    n_syndromes_total: usize,
    mean_strength: f64,
    error_mask_size: usize,
    leech_syndromes: usize,
    dark_syndromes: usize,
    total_mass: f64,
    epsilon_chrono: f64,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Find spacecraft in database
    let flyby = all_flybys()
        .into_iter()
        .find(|f| f.name.contains(&cli.spacecraft))
        .unwrap_or_else(|| {
            let names: Vec<&str> = all_flybys().iter().map(|f| f.name).collect();
            panic!(
                "Spacecraft '{}' not found. Available: {:?}",
                cli.spacecraft, names
            );
        });

    eprintln!(
        "chrono-flyby-probe: {} | grid={}x{}x{} | tau={} | alpha={} | cd_dim={}",
        flyby.name, cli.nx, cli.ny, cli.nz, cli.tau, cli.alpha_chrono, cli.cd_dim
    );

    // Compute SOI window and initial trajectory state
    let t_window = compute_soi_window(&flyby);
    let (pos, vel) = hyperbolic_initial_state(&flyby, t_window);
    let mut traj_state = State {
        position: [pos[0], pos[1], pos[2]],
        velocity: [vel[0], vel[1], vel[2]],
    };

    eprintln!(
        "  SOI window: {t_window:.1} s | r_init: {:.1} km | v_init: {:.3} km/s",
        (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt(),
        (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt(),
    );

    // Initialize 3D LBM solver (uniform density, zero velocity)
    let mut solver = LbmSolver3D::new(cli.nx, cli.ny, cli.nz, cli.tau);
    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    // Environment model for trajectory integration
    let env = EarthOnlyNfwLike::default();

    // Gravity wake coordinate scaling:
    // SOI radius defines the physical extent of the grid.
    // GM_EARTH is in km^3/s^2 (3.986e5). dx_km maps grid spacing to physical km.
    let soi_radius_km = t_window * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
    let grid_max = cli.nx.max(cli.ny).max(cli.nz) as f64;
    let dx_km = 2.0 * soi_radius_km / grid_max;
    let r_perigee_km = R_EARTH + flyby.perigee_alt_km;
    let gm_lattice = cli.gm_lattice.unwrap_or_else(|| {
        // Auto-scale GM so peak lattice force at perigee is 0.005 (safe for LBM stability).
        // This normalization makes the gravity wake detectable at any grid resolution.
        compute_gm_lattice(r_perigee_km, dx_km, cli.softening, 0.005)
    });

    if !cli.no_gravity_wake {
        eprintln!(
            "  gravity wake: dx={dx_km:.1} km | GM_lattice={gm_lattice:.6e} | softening={:.1}",
            cli.softening
        );
    }

    // Capture background force field before time loop so each snapshot
    // combines gravity with the pristine background, not the accumulated field.
    let background = solver.force_field.clone();

    // Time bookkeeping
    let dt_snap = cli.dt_trajectory * cli.lbm_steps_per_snap as f64;
    let mut t_current = -t_window;

    // Output setup
    let mut csv_writer: Option<csv::Writer<fs::File>> = if let Some(ref path) = cli.out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let w = csv::Writer::from_path(path)?;
        Some(w)
    } else {
        None
    };

    // CSV header
    if let Some(ref mut w) = csv_writer {
        w.write_record([
            "snap",
            "t_s",
            "r_km",
            "v_km_s",
            "n_anomalies",
            "syndromes_active",
            "syndromes_total",
            "mean_strength",
            "error_mask_size",
            "leech_syndromes",
            "dark_syndromes",
            "total_mass",
            "epsilon_chrono",
        ])?;
    }

    // Header
    eprintln!(
        "{:>5} {:>10} {:>10} {:>8} {:>6} {:>5}/{:>5} {:>8} {:>5} {:>6} {:>5} {:>8}",
        "snap",
        "t_s",
        "r_km",
        "v_km/s",
        "anom",
        "act",
        "tot",
        "strength",
        "mask",
        "leech",
        "dark",
        "mass"
    );

    // Main loop: interleave trajectory steps with LBM evolution
    for snap in 0..cli.n_snapshots {
        // 1. Advance trajectory by one snapshot interval
        let t_end = (t_current + dt_snap).min(t_window);
        let (new_state, _steps) = run_trajectory(
            &traj_state,
            t_current,
            t_end,
            cli.dt_trajectory,
            &env,
            &keplerian_accel,
        );
        traj_state = new_state;
        t_current = t_end;

        // Compute trajectory metrics
        let r_km = (traj_state.position[0].powi(2)
            + traj_state.position[1].powi(2)
            + traj_state.position[2].powi(2))
        .sqrt();
        let v_km_s = (traj_state.velocity[0].powi(2)
            + traj_state.velocity[1].powi(2)
            + traj_state.velocity[2].powi(2))
        .sqrt();

        // 2. Compute chrono-complex epsilon from trajectory (proximity -> temporal flux)
        //    Closer to Earth => stronger imaginary time component
        let r_perigee = R_EARTH + flyby.perigee_alt_km;
        let epsilon = (r_perigee / r_km).powi(2).min(1.0);

        // 3. Apply chrono-complex viscosity modulation
        apply_chrono_viscosity(&mut solver, cli.alpha_chrono, epsilon);

        // 3.5. Inject gravitational wake from flyby mass into LBM force field.
        //      Maps physical trajectory position to lattice coordinates, computes
        //      Newtonian acceleration at every grid node, and sets via Guo forcing.
        //      See BIB-0297 (Anderson 2008) for flyby anomaly empirical context.
        if !cli.no_gravity_wake {
            let mass_lattice =
                physical_to_lattice(traj_state.position, cli.nx, cli.ny, cli.nz, dx_km);
            let grav_field = gravity_wake_field(
                cli.nx,
                cli.ny,
                cli.nz,
                mass_lattice,
                gm_lattice,
                cli.softening,
            );
            // Superimpose gravitational wake on the pristine background
            // (captured before the time loop) to avoid accumulating gravity
            // across successive snapshots.
            let combined = if let Some(bg) = background.as_ref() {
                combine_forces(bg, &grav_field)
            } else {
                grav_field
            };
            solver.set_force_field(combined).expect("force field set");
        }

        // 4. Evolve LBM fluid (Guo forcing applied in collision step)
        for _ in 0..cli.lbm_steps_per_snap {
            solver.evolve_one_step();
        }

        // 5. Extract density anomalies
        let anomalies = extract_anomalies(&solver, cli.anomaly_threshold);

        // 6. Map anomalies to LatticeSiteState via Morton Z-order
        let lattice_states = anomalies_to_lattice_states(&anomalies, cli.nx, cli.ny, cli.cd_dim);

        // 7. Build error mask and syndrome snapshot
        let error_mask = build_error_mask(&lattice_states, cli.anomaly_threshold);

        let void_codes: Vec<VoidCodeAdapter> = lattice_states
            .iter()
            .flat_map(|site| {
                let axes = site.active_axes(cli.anomaly_threshold * 0.1);
                if axes.len() >= 3 {
                    // Use first three axes as a (i,j,k) violation proxy
                    Some(VoidCodeAdapter::from_violation(
                        axes[0],
                        axes[1],
                        axes[2],
                        if axes.len() > 3 { axes[3] } else { 0 },
                        cli.cd_dim,
                    ))
                } else {
                    None
                }
            })
            .collect();

        let snapshot = LatticeQecSnapshot::from_violations(
            &void_codes
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let axes = lattice_states
                        .get(i)
                        .map(|s| s.active_axes(cli.anomaly_threshold * 0.1))
                        .unwrap_or_default();
                    let (a, b, c) = if axes.len() >= 3 {
                        (axes[0], axes[1], axes[2])
                    } else {
                        (0, 1, 2)
                    };
                    let m = if axes.len() > 3 { axes[3] } else { 0 };
                    (a, b, c, m, 1i32)
                })
                .collect::<Vec<_>>(),
            cli.cd_dim,
        );

        let (active, total, mean_str) = snapshot.detect_active_syndromes(&error_mask);

        // 8. Classify active syndrome axes into Leech vs Dark sector.
        //    Leech sector: axis < LEECH_DIM (24) -- Standard Model / bosonic string subspace.
        //    Dark sector: axis >= LEECH_DIM (24..31) -- orthogonal gauge space.
        //    Only meaningful when cd_dim=32 (Pathion); at other dims, all go to "leech".
        let (leech_syn, dark_syn) = if cli.cd_dim == 32 {
            let mut leech = 0usize;
            let mut dark = 0usize;
            for site in &lattice_states {
                for &axis in &site.active_axes(cli.anomaly_threshold * 0.1) {
                    if axis < LEECH_DIM {
                        leech += 1;
                    } else {
                        dark += 1;
                    }
                }
            }
            (leech, dark)
        } else {
            // Non-Pathion dims: all syndromes classified as "leech" (no dark sector)
            let total_axes: usize = lattice_states
                .iter()
                .map(|s| s.active_axes(cli.anomaly_threshold * 0.1).len())
                .sum();
            (total_axes, 0)
        };

        let row = FrictionRow {
            snap,
            t_seconds: t_current,
            r_km,
            v_km_s,
            n_anomalies: anomalies.len(),
            n_syndromes_active: active,
            n_syndromes_total: total,
            mean_strength: mean_str,
            error_mask_size: error_mask.len(),
            leech_syndromes: leech_syn,
            dark_syndromes: dark_syn,
            total_mass: solver.total_mass(),
            epsilon_chrono: epsilon,
        };

        // Print row
        eprintln!(
            "{:>5} {:>10.1} {:>10.1} {:>8.3} {:>6} {:>5}/{:>5} {:>8.4} {:>5} {:>6} {:>5} {:>8.4}",
            row.snap,
            row.t_seconds,
            row.r_km,
            row.v_km_s,
            row.n_anomalies,
            row.n_syndromes_active,
            row.n_syndromes_total,
            row.mean_strength,
            row.error_mask_size,
            row.leech_syndromes,
            row.dark_syndromes,
            row.total_mass,
        );

        // Write CSV
        if let Some(ref mut w) = csv_writer {
            w.write_record(&[
                row.snap.to_string(),
                format!("{:.2}", row.t_seconds),
                format!("{:.2}", row.r_km),
                format!("{:.4}", row.v_km_s),
                row.n_anomalies.to_string(),
                row.n_syndromes_active.to_string(),
                row.n_syndromes_total.to_string(),
                format!("{:.6}", row.mean_strength),
                row.error_mask_size.to_string(),
                row.leech_syndromes.to_string(),
                row.dark_syndromes.to_string(),
                format!("{:.6}", row.total_mass),
                format!("{:.6}", row.epsilon_chrono),
            ])?;
        }
    }

    if let Some(ref mut w) = csv_writer {
        w.flush()?;
    }

    eprintln!("done.");
    Ok(())
}
