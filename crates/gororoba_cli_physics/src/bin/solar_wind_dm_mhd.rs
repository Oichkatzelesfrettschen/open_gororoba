//! D3Q19 LBM + MHD + dark matter gravitational coupling (E-112).
//!
//! Extends the solar wind MHD simulation (E-111) with NFW dark matter
//! halo gravitational forcing. The DM force field is static (precomputed
//! once from NFW profile) and combined with the Lorentz force via the
//! existing Guo forcing scheme.
//!
//! The key result is a rigorous null measurement: the ratio
//! max|F_DM| / max|F_Lorentz| ~ O(10^-12), confirming that DM gravity
//! alone cannot produce observable solar wind perturbations at 1 AU.

use clap::Parser;
use lbm_3d::boundary::ZouHeBoundary;
use lbm_3d::dm_force::{combine_forces, DmForceConfig, DmForceField};
use lbm_3d::mhd::{MhdConfig, MhdField};
use lbm_3d::solver::{BgkCollision, LbmSolver3D};
use std::fs;
use std::io::{BufRead, Write};
use std::path::PathBuf;

/// D3Q19 LBM + MHD + NFW dark matter gravitational coupling.
///
/// Couples the magnetized solar wind simulation (Parker spiral B-field,
/// Zou-He inlet, Guo forcing) with a static NFW DM halo gravitational
/// force field. Quantifies the DM/Lorentz force ratio as a null test.
#[derive(Parser)]
#[command(name = "solar-wind-dm-mhd")]
struct Cli {
    /// Grid size in x (radial direction)
    #[arg(long, default_value_t = 128)]
    nx: usize,

    /// Grid size in y (transverse)
    #[arg(long, default_value_t = 32)]
    ny: usize,

    /// Grid size in z (transverse)
    #[arg(long, default_value_t = 32)]
    nz: usize,

    /// Number of LBM timesteps
    #[arg(long, default_value_t = 5000)]
    steps: usize,

    /// LBM relaxation time tau (>0.5 for stability)
    #[arg(long, default_value_t = 0.6)]
    tau: f64,

    /// Initial B-field magnitude (nT)
    #[arg(long, default_value_t = 5.0)]
    b0: f64,

    /// Solar wind bulk speed in LBM lattice units (Ma < 0.3 for stability)
    #[arg(long, default_value_t = 0.05)]
    v_sw: f64,

    /// Magnetic resistivity (0.0 = ideal MHD)
    #[arg(long, default_value_t = 0.0)]
    eta: f64,

    /// Snapshot interval (write output every N steps)
    #[arg(long, default_value_t = 500)]
    snap_interval: usize,

    /// Output directory for CSV snapshots
    #[arg(long)]
    out: Option<PathBuf>,

    // ---- DM parameters ----
    /// Local DM density (GeV/cm^3). Canonical value: 0.3
    #[arg(long, default_value_t = 0.3)]
    dm_density: f64,

    /// MW virial mass in solar masses
    #[arg(long, default_value_t = 1.0e12)]
    dm_m200: f64,

    /// NFW concentration parameter
    #[arg(long, default_value_t = 10.0)]
    dm_c200: f64,

    /// Gravitational focusing wake amplitude (0 = isotropic)
    #[arg(long, default_value_t = 0.0)]
    dm_wake: f64,

    /// DM wind x-component in lattice units
    #[arg(long, default_value_t = 0.0)]
    dm_wind_x: f64,

    /// DM wind y-component in lattice units
    #[arg(long, default_value_t = 0.0)]
    dm_wind_y: f64,

    /// DM wind z-component in lattice units
    #[arg(long, default_value_t = 0.0)]
    dm_wind_z: f64,

    /// DM-baryon scattering cross-section (cm^2). 0 = pure gravity.
    /// Nonzero values enable dynamic drag force recomputed each timestep.
    #[arg(long, default_value_t = 0.0)]
    dm_sigma: f64,

    /// Reference proton density (cm^-3) for drag unit conversion.
    /// Typically the median solar wind density (~5.9 cm^-3 from OMNI2).
    #[arg(long, default_value_t = 5.9)]
    dm_n_ref: f64,

    /// Reference bulk speed (km/s) for drag unit conversion.
    /// Typically the median solar wind speed (~393 km/s from OMNI2).
    #[arg(long, default_value_t = 393.0)]
    dm_v_ref: f64,

    /// Disable DM coupling (pure MHD baseline for A/B comparison)
    #[arg(long, default_value_t = false)]
    no_dm: bool,

    /// Path to initial condition CSV from solar-wind-ic.
    /// Format: x,y,z,rho,ux,uy,uz,bx,by,bz (header row skipped).
    /// When provided, uses real spacecraft data instead of uniform+Parker init.
    #[arg(long)]
    ic_file: Option<PathBuf>,
}

/// Write a CSV snapshot of the midplane (z=nz/2) with extended DM columns.
fn write_snapshot(
    path: &std::path::Path,
    solver: &LbmSolver3D,
    mhd: &MhdField,
    dm: Option<&DmForceField>,
    step: usize,
) -> std::io::Result<()> {
    let nx = solver.nx;
    let ny = solver.ny;
    let z_mid = solver.nz / 2;

    let filename = path.join(format!("snapshot_{step:06}.csv"));
    let mut file = fs::File::create(&filename)?;
    writeln!(
        file,
        "x,y,rho,ux,uy,uz,bx,by,bz,dm_fx,dm_fy,dm_fz,dm_rho"
    )?;

    for y in 0..ny {
        for x in 0..nx {
            let (rho, u) = solver.get_macroscopic(x, y, z_mid);
            let idx = z_mid * (nx * ny) + y * nx + x;

            let (dm_fx, dm_fy, dm_fz, dm_rho) = match dm {
                Some(d) => {
                    let f = d.force_at(idx);
                    (f[0], f[1], f[2], d.dm_density_at(idx))
                }
                None => (0.0, 0.0, 0.0, 0.0),
            };

            writeln!(
                file,
                "{x},{y},{rho:.8},{:.8},{:.8},{:.8},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}",
                u[0], u[1], u[2],
                mhd.bx[idx], mhd.by[idx], mhd.bz[idx],
                dm_fx, dm_fy, dm_fz, dm_rho,
            )?;
        }
    }
    Ok(())
}

/// Compute max magnitude of a force field.
fn max_force_mag(f: &[[f64; 3]]) -> f64 {
    f.iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0_f64, f64::max)
}

/// Load initial conditions from a CSV file produced by solar-wind-ic.
///
/// Format: x,y,z,rho,ux,uy,uz,bx,by,bz (header row skipped).
/// Populates solver (rho, u, f) and mhd (bx, by, bz) fields directly
/// from real spacecraft data.
fn load_ic_file(
    path: &std::path::Path,
    solver: &mut LbmSolver3D,
    mhd: &mut MhdField,
) -> anyhow::Result<usize> {
    let file = fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let lattice = &solver.collider.lattice;
    let nx = solver.nx;
    let ny = solver.ny;

    let mut loaded = 0usize;
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('x') || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 10 {
            continue;
        }
        let x: usize = fields[0].trim().parse()?;
        let y: usize = fields[1].trim().parse()?;
        let z: usize = fields[2].trim().parse()?;
        let rho: f64 = fields[3].trim().parse()?;
        let ux: f64 = fields[4].trim().parse()?;
        let uy: f64 = fields[5].trim().parse()?;
        let uz: f64 = fields[6].trim().parse()?;
        let bx: f64 = fields[7].trim().parse()?;
        let by: f64 = fields[8].trim().parse()?;
        let bz: f64 = fields[9].trim().parse()?;

        let idx = z * (nx * ny) + y * nx + x;
        if idx >= solver.rho.len() {
            continue;
        }

        solver.rho[idx] = rho;
        solver.u[idx] = [ux, uy, uz];

        // Re-initialize distribution function to equilibrium at (rho, u)
        let f_eq = BgkCollision::initialize_with_velocity(rho, [ux, uy, uz], lattice);
        for (i, &fi) in f_eq.iter().enumerate() {
            solver.f[idx * 19 + i] = fi;
        }

        mhd.bx[idx] = bx;
        mhd.by[idx] = by;
        mhd.bz[idx] = bz;

        loaded += 1;
    }
    Ok(loaded)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let dm_label = if cli.no_dm { "OFF" } else { "ON" };
    let ic_label = if cli.ic_file.is_some() {
        "real-data"
    } else {
        "synthetic"
    };
    eprintln!(
        "solar-wind-dm-mhd: {}x{}x{}, {} steps, tau={}, B0={} nT, v_sw={}, DM={}, IC={}",
        cli.nx, cli.ny, cli.nz, cli.steps, cli.tau, cli.b0, cli.v_sw, dm_label, ic_label,
    );

    // Initialize LBM solver
    let mut solver = LbmSolver3D::new(cli.nx, cli.ny, cli.nz, cli.tau);

    // Initialize MHD field
    let mhd_config = MhdConfig {
        b0_nt: cli.b0,
        omega: 2.662e-6,
        mu_0: 1.0,
        eta: cli.eta,
        dt_mhd: 1.0,
        cleaning_rate: 0.1,
    };
    let mut mhd = MhdField::new(cli.nx, cli.ny, cli.nz, mhd_config);

    // Initialize state: real data from IC file, or synthetic uniform+Parker
    let u_sw = if let Some(ref ic_path) = cli.ic_file {
        let loaded = load_ic_file(ic_path, &mut solver, &mut mhd)?;
        eprintln!("loaded {} cells from IC file: {}", loaded, ic_path.display());

        // Compute median u_x for inlet boundary (needed for Zou-He)
        let mut ux_vals: Vec<f64> = solver.u.iter().map(|u| u[0]).collect();
        ux_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ux = ux_vals[ux_vals.len() / 2];
        [median_ux, 0.0, 0.0]
    } else {
        let u_init = [cli.v_sw, 0.0, 0.0];
        solver.initialize_uniform(1.0, u_init);
        mhd.parker_spiral_init(cli.v_sw);
        u_init
    };

    // Helmholtz projection: remove magnetic monopoles from Cartesian discretization
    let (div_before, div_after) = mhd.project_divergence_free(5000, 1e-12);
    let initial_energy = mhd.magnetic_energy();
    eprintln!("div(B) projection: {div_before:.6e} -> {div_after:.6e}");
    eprintln!("initial magnetic energy (post-projection): {initial_energy:.6e}");

    // Initialize DM force field (if enabled)
    let dm_field = if !cli.no_dm {
        let dm_config = DmForceConfig {
            rho_dm_local_gev_cm3: cli.dm_density,
            m200_solar: cli.dm_m200,
            c200: cli.dm_c200,
            v_dm_wind: [cli.dm_wind_x, cli.dm_wind_y, cli.dm_wind_z],
            eta_wake: cli.dm_wake,
            sigma_chi_b: cli.dm_sigma,
            ..DmForceConfig::default()
        };
        let field = DmForceField::new(cli.nx, cli.ny, cli.nz, dm_config);
        eprintln!("DM max |F_grav|: {:.6e} (lattice units)", field.max_force_magnitude());
        if cli.dm_sigma > 0.0 {
            eprintln!(
                "DM drag: sigma={:.3e} cm^2, n_ref={:.1} cm^-3, v_ref={:.0} km/s",
                cli.dm_sigma, cli.dm_n_ref, cli.dm_v_ref,
            );
        }
        Some(field)
    } else {
        None
    };

    // Zou-He boundary for velocity inlet
    let zou_he = ZouHeBoundary::new();

    // Output directory setup
    if let Some(ref dir) = cli.out {
        fs::create_dir_all(dir)?;
    }

    // Time loop
    for step in 0..cli.steps {
        // 1. Apply Zou-He velocity inlet at x=0
        zou_he.apply_velocity_inlet_min_x(&mut solver.f, cli.nx, cli.ny, cli.nz, u_sw);

        // 2. Compute Lorentz force from current B-field
        let lorentz = mhd.lorentz_force();

        // 3. Combine with DM gravitational force (if enabled)
        let combined = match &dm_field {
            Some(dm) => {
                let grav_combined = combine_forces(&lorentz, &dm.force);
                // Add dynamic drag force when sigma_chi_b > 0
                if dm.config.sigma_chi_b > 0.0 {
                    let drag = dm.drag_force(
                        &solver.rho,
                        &solver.u,
                        cli.dm_n_ref,
                        cli.dm_v_ref,
                    );
                    combine_forces(&grav_combined, &drag)
                } else {
                    grav_combined
                }
            }
            None => lorentz,
        };

        // 4. Set combined force field for Guo scheme
        solver.set_force_field(combined).expect("force field set");

        // 5. LBM collision + streaming step
        solver.evolve_one_step();

        // 6. Evolve B-field using updated velocity
        mhd.evolve_b_field(&solver.u);

        // 7. Periodic output
        if (step + 1) % cli.snap_interval == 0 || step == 0 {
            let energy = mhd.magnetic_energy();
            let div = mhd.max_div_b();
            let mass = solver.total_mass();

            // Compute force ratio
            let lorentz_now = mhd.lorentz_force();
            let max_lorentz = max_force_mag(&lorentz_now);
            let max_dm = dm_field.as_ref().map_or(0.0, |d| d.max_force_magnitude());
            let ratio = if max_lorentz > 0.0 {
                max_dm / max_lorentz
            } else {
                0.0
            };

            // Report drag force magnitude when sigma > 0
            let drag_info = if let Some(dm) = dm_field.as_ref() {
                if dm.config.sigma_chi_b > 0.0 {
                    let drag = dm.drag_force(
                        &solver.rho,
                        &solver.u,
                        cli.dm_n_ref,
                        cli.dm_v_ref,
                    );
                    let max_drag = max_force_mag(&drag);
                    format!("  |F_drag|={max_drag:.3e}")
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            eprintln!(
                "step={:>6}  mass={mass:.6}  B_energy={energy:.6e}  max|divB|={div:.6e}  |F_DM|/|F_L|={ratio:.3e}{drag_info}",
                step + 1,
            );

            if let Some(ref dir) = cli.out {
                write_snapshot(dir, &solver, &mhd, dm_field.as_ref(), step + 1)?;
            }
        }
    }

    eprintln!("done.");
    Ok(())
}
