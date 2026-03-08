//! D3Q19 LBM + MHD simulation of magnetized solar wind.
//!
//! Couples the LBM fluid solver with magnetic field evolution to simulate
//! Parker spiral solar wind flow. Uses Zou-He velocity inlet at x=0 and
//! periodic boundaries in y/z.

use clap::Parser;
use lbm_3d::boundary::ZouHeBoundary;
use lbm_3d::mhd::{MhdConfig, MhdField};
use lbm_3d::solver::LbmSolver3D;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// D3Q19 LBM + MHD simulation of magnetized solar wind with Parker
/// spiral B-field and Zou-He velocity inlet.
#[derive(Parser)]
#[command(name = "solar-wind-mhd-sim")]
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

    /// Solar wind bulk speed in LBM lattice units (Ma < 0.3 for LBM stability)
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
}

/// Write a CSV snapshot of the midplane (z=nz/2) fields.
fn write_snapshot(
    path: &std::path::Path,
    solver: &LbmSolver3D,
    mhd: &MhdField,
    step: usize,
) -> std::io::Result<()> {
    let nx = solver.nx;
    let ny = solver.ny;
    let z_mid = solver.nz / 2;

    let filename = path.join(format!("snapshot_{step:06}.csv"));
    let mut file = fs::File::create(&filename)?;
    writeln!(file, "x,y,rho,ux,uy,uz,bx,by,bz")?;

    for y in 0..ny {
        for x in 0..nx {
            let (rho, u) = solver.get_macroscopic(x, y, z_mid);
            let idx = z_mid * (nx * ny) + y * nx + x;
            writeln!(
                file,
                "{x},{y},{rho:.8},{:.8},{:.8},{:.8},{:.8e},{:.8e},{:.8e}",
                u[0], u[1], u[2], mhd.bx[idx], mhd.by[idx], mhd.bz[idx],
            )?;
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!(
        "solar-wind-mhd-sim: {}x{}x{}, {} steps, tau={}, B0={} nT, v_sw={}",
        cli.nx, cli.ny, cli.nz, cli.steps, cli.tau, cli.b0, cli.v_sw,
    );

    // Initialize LBM solver
    let mut solver = LbmSolver3D::new(cli.nx, cli.ny, cli.nz, cli.tau);

    // Initialize with uniform solar wind velocity along x
    let u_sw = [cli.v_sw, 0.0, 0.0];
    solver.initialize_uniform(1.0, u_sw);

    // Initialize MHD field with Parker spiral
    let mhd_config = MhdConfig {
        b0_nt: cli.b0,
        omega: 2.662e-6,
        mu_0: 1.0,
        eta: cli.eta,
        dt_mhd: 1.0,
        cleaning_rate: 0.1,
    };
    let mut mhd = MhdField::new(cli.nx, cli.ny, cli.nz, mhd_config);
    mhd.parker_spiral_init(cli.v_sw);

    // Project B-field to divergence-free state. The Parker spiral initialization
    // on a Cartesian grid creates magnetic monopole artifacts (dBr/dx != 0 but
    // the Cartesian divergence operator doesn't account for the spherical area
    // element). This Helmholtz projection removes the irrotational component.
    let (div_before, div_after) = mhd.project_divergence_free(5000, 1e-12);
    let initial_energy = mhd.magnetic_energy();
    eprintln!("div(B) projection: {div_before:.6e} -> {div_after:.6e}");
    eprintln!("initial magnetic energy (post-projection): {initial_energy:.6e}");

    // Zou-He boundary for velocity inlet
    let zou_he = ZouHeBoundary::new();

    // Output directory setup
    if let Some(ref dir) = cli.out {
        fs::create_dir_all(dir)?;
    }

    // Time loop
    for step in 0..cli.steps {
        // 1. Apply Zou-He velocity inlet at x=0
        zou_he.apply_velocity_inlet_min_x(
            &mut solver.f,
            cli.nx,
            cli.ny,
            cli.nz,
            u_sw,
        );

        // 2. Compute Lorentz force from current B-field
        let lorentz = mhd.lorentz_force();
        solver.set_force_field(lorentz).expect("force field set");

        // 3. LBM collision + streaming step
        solver.evolve_one_step();

        // 4. Evolve B-field using updated velocity
        mhd.evolve_b_field(&solver.u);

        // 5. Periodic output
        if (step + 1) % cli.snap_interval == 0 || step == 0 {
            let energy = mhd.magnetic_energy();
            let div = mhd.max_div_b();
            let mass = solver.total_mass();
            eprintln!(
                "step={:>6}  mass={mass:.6}  B_energy={energy:.6e}  max|divB|={div:.6e}",
                step + 1,
            );

            if let Some(ref dir) = cli.out {
                write_snapshot(dir, &solver, &mhd, step + 1)?;
            }
        }
    }

    eprintln!("done.");
    Ok(())
}
