//! D3Q19 LBM + MHD simulation of magnetized solar wind.
//!
//! Couples the LBM fluid solver with magnetic field evolution to simulate
//! Parker spiral solar wind flow. Uses Zou-He velocity inlet at x=0 and
//! periodic boundaries in y/z.
//!
//! Supports loading initial conditions from solar wind-ic CSV files
//! containing real spacecraft data (NASA OMNI2 or ACE SWEPAM).

use clap::{Args, ValueEnum};
use lbm_3d::{
    boundary::ZouHeBoundary,
    mhd::{MhdConfig, MhdField, MhdIntegrator},
    solver::{BgkCollision, LbmSolver3D},
};
use std::{
    fs,
    io::{BufRead, Write},
    path::PathBuf,
};

/// D3Q19 LBM + MHD simulation of magnetized solar wind with Parker
/// spiral B-field and Zou-He velocity inlet.
#[derive(Args)]
pub struct Cli {
    /// Magnetic evolution method; finite-stage admission does not certify coupled stability.
    #[arg(long, value_enum, default_value = "legacy-euler")]
    mhd_integrator: MhdIntegratorChoice,

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

    /// Magnetic diffusivity in lattice length squared per LBM timestep (0 = ideal MHD).
    #[arg(
        long = "magnetic-diffusivity-lattice",
        alias = "eta",
        default_value_t = 0.0
    )]
    eta: f64,

    /// Snapshot interval (write output every N steps)
    #[arg(long, default_value_t = 500)]
    snap_interval: usize,

    /// Output directory for CSV snapshots
    #[arg(long)]
    out: Option<PathBuf>,

    /// Path to initial condition CSV from solar wind-ic.
    /// Format: x,y,z,rho,ux,uy,uz,bx,by,bz (header row skipped).
    /// When provided, uses real spacecraft data instead of uniform+Parker init.
    #[arg(long)]
    ic_file: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MhdIntegratorChoice {
    LegacyEuler,
    SspRk3,
}

impl MhdIntegratorChoice {
    fn method(self) -> MhdIntegrator {
        match self {
            Self::LegacyEuler => MhdIntegrator::LegacyEuler,
            Self::SspRk3 => MhdIntegrator::SspRk3,
        }
    }

    fn receipt(self) -> &'static str {
        match self {
            Self::LegacyEuler => {
                "mhd_integrator=legacy-euler transport_admission_scope=periodic_diffusion_coefficient_and_finite_state"
            }
            Self::SspRk3 => {
                "mhd_integrator=ssp-rk3 transport_admission_scope=prescribed_frozen_velocity_finite_stage_admission"
            }
        }
    }
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

/// Load initial conditions from a CSV file produced by solar wind-ic.
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

        if x >= nx || y >= ny || z >= solver.nz {
            continue;
        }
        let idx = z * (nx * ny) + y * nx + x;

        solver.rho[idx] = rho;
        solver.u[idx] = [ux, uy, uz];

        let f_eq = BgkCollision::initialize_with_velocity(rho, [ux, uy, uz], lattice);
        for (i, &fi) in f_eq.iter().enumerate() {
            solver.f[lbm_3d::solver::aosoa_idx(idx, i)] = fi;
        }

        mhd.bx[idx] = bx;
        mhd.by[idx] = by;
        mhd.bz[idx] = bz;

        loaded += 1;
    }
    Ok(loaded)
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    let ic_label = if cli.ic_file.is_some() {
        "real-data"
    } else {
        "synthetic"
    };
    eprintln!(
        "solar wind-mhd-sim: {}x{}x{}, {} steps, tau={}, B0={} nT, v_sw={}, IC={}",
        cli.nx, cli.ny, cli.nz, cli.steps, cli.tau, cli.b0, cli.v_sw, ic_label,
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
    let mut mhd = MhdField::try_new(cli.nx, cli.ny, cli.nz, mhd_config)?;
    eprintln!("{}", cli.mhd_integrator.receipt());
    eprintln!(
        "magnetic_diffusivity_lattice={:.17e} dt_mhd={:.17e} diffusion_gate_scope=periodic_diffusion_only",
        mhd.config.eta, mhd.config.dt_mhd,
    );

    // Initialize state: real data from IC file, or synthetic uniform+Parker
    let u_sw = if let Some(ref ic_path) = cli.ic_file {
        let loaded = load_ic_file(ic_path, &mut solver, &mut mhd)?;
        eprintln!(
            "loaded {} cells from IC file: {}",
            loaded,
            ic_path.display()
        );

        // Extract u_x from x=0 face only (the inflow boundary).
        // Using the global median would mix interior conditions that
        // may differ due to DM drag or magnetic pressure gradients.
        let mut ux_vals: Vec<f64> = Vec::new();
        for z in 0..cli.nz {
            for y in 0..cli.ny {
                let idx = z * (cli.nx * cli.ny) + y * cli.nx; // x=0
                ux_vals.push(solver.u[idx][0]);
            }
        }
        ux_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ux = ux_vals[ux_vals.len() / 2];
        [median_ux, 0.0, 0.0]
    } else {
        let u_init = [cli.v_sw, 0.0, 0.0];
        solver.initialize_uniform(1.0, u_init);
        mhd.parker_spiral_init(cli.v_sw);
        u_init
    };

    // Project B-field to divergence-free state
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
        zou_he.apply_velocity_inlet_min_x(&mut solver.f, cli.nx, cli.ny, cli.nz, u_sw);

        // 2. Compute Lorentz force from current B-field
        let lorentz = mhd.lorentz_force();
        solver.set_force_field(lorentz).expect("force field set");

        // 3. LBM collision + streaming step
        solver.evolve_one_step();

        // 4. Evolve B-field using updated velocity
        mhd.try_evolve_b_field_with_integrator(&solver.u, cli.mhd_integrator.method())?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestArgs {
        #[command(flatten)]
        cli: Cli,
    }

    #[test]
    fn mhd_integrator_cli_requires_explicit_rk3_selection() {
        assert!(matches!(
            TestArgs::try_parse_from(["solar"])
                .unwrap()
                .cli
                .mhd_integrator
                .method(),
            MhdIntegrator::LegacyEuler
        ));
        for (name, expected) in [
            ("legacy-euler", "mhd_integrator=legacy-euler"),
            ("ssp-rk3", "mhd_integrator=ssp-rk3"),
        ] {
            let parsed = TestArgs::try_parse_from(["solar", "--mhd-integrator", name])
                .unwrap()
                .cli;
            assert!(parsed.mhd_integrator.receipt().starts_with(expected));
            assert_eq!(
                matches!(parsed.mhd_integrator.method(), MhdIntegrator::SspRk3),
                name == "ssp-rk3"
            );
        }
        assert!(TestArgs::try_parse_from(["solar", "--mhd-integrator", "automatic"]).is_err());
    }

    #[test]
    fn magnetic_diffusivity_cli_preserves_lattice_alias() {
        assert_eq!(
            TestArgs::try_parse_from(["solar"])
                .unwrap()
                .cli
                .eta
                .to_bits(),
            0.0_f64.to_bits()
        );
        for flag in ["--eta", "--magnetic-diffusivity-lattice"] {
            assert_eq!(
                TestArgs::try_parse_from(["solar", flag, "0.125"])
                    .unwrap()
                    .cli
                    .eta
                    .to_bits(),
                0.125_f64.to_bits()
            );
        }
        assert!(TestArgs::try_parse_from(["solar", "--magnetic-diffusivity-m2-s", "1"]).is_err());
    }
}
