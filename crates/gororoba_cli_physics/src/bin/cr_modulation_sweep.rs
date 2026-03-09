//! Cosmic ray modulation sweep via the Parker Transport Equation (E-121..E-127).
//!
//! One-way-coupled pipeline:
//!   1. Initialize LBM solar wind on a radial grid (nx cells = radial bins)
//!   2. Run LBM N steps to evolve the solar wind velocity and B-field
//!   3. Feed u_sw + B into PteSolver.evolve_one_step() each window
//!   4. Compute Gleeson-Axford modulation potential phi(r) from radial profile
//!   5. Write snapshots: cr_snapshot_{step}.csv
//!   6. Optionally inject DM source term for annihilation signal search
//!
//! The radial x-axis maps to heliocentric distance [r_min_au, r_max_au].
//! The ISM local interstellar spectrum (LIS) is applied as a Bess-Pamela power
//! law at the outer boundary: J_LIS(R) = J_0 * (R/1 GV)^{-gamma}.

use clap::Parser;
use cr_transport::diffusion::DiffusionConfig;
use cr_transport::grid::RigidityGrid;
use cr_transport::modulation::ForceFieldProxy;
use cr_transport::snapshot::write_snapshot_csv;
use cr_transport::solver::PteSolver;
use cr_transport::source::{DmChannel, DmSource};
use lbm_3d::mhd::{MhdConfig, MhdField};
use lbm_3d::solver::LbmSolver3D;
use std::fs;
use std::io::Write as IoWrite;
use std::path::PathBuf;

/// Cosmic ray modulation sweep: LBM solar wind + Parker Transport Equation.
///
/// Couples heliospheric LBM/MHD fluid output to collisionless GCR transport.
/// Produces modulation potential phi(r), rigidity spectra at multiple distances,
/// and optional DM annihilation source injection diagnostics.
#[derive(Parser)]
#[command(name = "cr-modulation-sweep")]
struct Cli {
    /// Grid size in x (radial direction, maps to r_min_au..r_max_au)
    #[arg(long, default_value_t = 64)]
    nx: usize,

    /// Grid size in y (transverse)
    #[arg(long, default_value_t = 16)]
    ny: usize,

    /// Grid size in z (transverse)
    #[arg(long, default_value_t = 16)]
    nz: usize,

    /// LBM steps per PTE integration window
    #[arg(long, default_value_t = 1000)]
    steps: usize,

    /// LBM relaxation time tau (>0.5 for stability)
    #[arg(long, default_value_t = 0.6)]
    tau: f64,

    /// Initial B-field magnitude (nT)
    #[arg(long, default_value_t = 5.0)]
    b0: f64,

    /// Solar wind bulk speed in LBM lattice units
    #[arg(long, default_value_t = 0.05)]
    v_sw: f64,

    /// Minimum heliocentric radius (AU) for radial x-axis mapping
    #[arg(long, default_value_t = 1.0)]
    r_min_au: f64,

    /// Maximum heliocentric radius (AU) for radial x-axis mapping
    #[arg(long, default_value_t = 157.0)]
    r_max_au: f64,

    /// Number of logarithmic rigidity bins
    #[arg(long, default_value_t = 40)]
    n_p: usize,

    /// Minimum rigidity (GV)
    #[arg(long, default_value_t = 0.1)]
    r_min_gv: f64,

    /// Maximum rigidity (GV)
    #[arg(long, default_value_t = 1000.0)]
    r_max_gv: f64,

    /// DM particle mass (GeV). 0.0 = disable DM source injection.
    #[arg(long, default_value_t = 0.0)]
    dm_mass_gev: f64,

    /// DM annihilation channel (bb, ww, tau, monochromatic)
    #[arg(long, default_value = "bb")]
    dm_channel: String,

    /// DM annihilation cross section (cm^3/s)
    #[arg(long, default_value_t = 3e-26)]
    dm_sigma_v: f64,

    /// Reference diffusion coefficient (AU^2/s)
    #[arg(long, default_value_t = 4.5e-5)]
    kappa_0: f64,

    /// Rigidity power-law index for diffusion
    #[arg(long, default_value_t = 0.5)]
    alpha: f64,

    /// Perpendicular-to-parallel diffusion ratio
    #[arg(long, default_value_t = 0.02)]
    epsilon_perp: f64,

    /// Solar epoch sign (+1.0 = A>0, -1.0 = A<0)
    #[arg(long, default_value_t = 1.0)]
    solar_epoch_a: f64,

    /// Snapshot output interval (steps)
    #[arg(long, default_value_t = 200)]
    snapshot_interval: usize,

    /// Output directory for snapshots
    #[arg(long, default_value = "data/output/cr")]
    out_dir: PathBuf,
}

/// Local interstellar spectrum (LIS) power law.
/// J_LIS(R) = J_0 * (R / 1 GV)^{-gamma}
/// Bess-Pamela proton LIS reference: J_0 = 2.1e4 [m^-2 s^-1 sr^-1 GV^-1], gamma = 2.75.
fn lis_proton(r_gv: f64) -> f64 {
    let j0 = 2.1e4_f64;
    let gamma = 2.75_f64;
    j0 * r_gv.powf(-gamma)
}

/// Parse DM channel string into DmChannel enum.
fn parse_dm_channel(s: &str) -> DmChannel {
    match s.to_lowercase().as_str() {
        "ww" => DmChannel::WW,
        "tau" => DmChannel::Tau,
        "monochromatic" | "pos" => DmChannel::PositronAntiproton,
        _ => DmChannel::BbBar,
    }
}

/// Compute per-x-cell radial modulation potential phi(r_x) using ForceFieldProxy.
/// u_sw is the LBM velocity field; we take the x-component of the mid-plane cell.
fn compute_phi_map(
    solver: &LbmSolver3D,
    proxy: &ForceFieldProxy,
    diffusion: &DiffusionConfig,
    r_min_au: f64,
    r_max_au: f64,
) -> Vec<f64> {
    let nx = solver.nx;
    let ny = solver.ny;
    let nz = solver.nz;
    let y_mid = ny / 2;
    let z_mid = nz / 2;
    let dr = (r_max_au - r_min_au) / (nx as f64 - 1.0).max(1.0);

    // Build radial v_sw and kappa_rr profiles from LBM mid-plane
    let r_of_x = |x: usize| r_min_au + x as f64 * dr;
    let v_sw_of_r = {
        let mid_u: Vec<f64> = (0..nx)
            .map(|x| {
                let u = solver.u[z_mid * nx * ny + y_mid * nx + x];
                // convert LBM lattice speed to km/s using u_scale ~ 400 km/s / 0.05
                u[0].abs() * 400.0 / 0.05
            })
            .collect();
        move |r: f64| {
            let x_frac = ((r - r_min_au) / dr.max(1e-10)).clamp(0.0, (nx - 1) as f64);
            let x0 = x_frac.floor() as usize;
            let x1 = (x0 + 1).min(nx - 1);
            let t = x_frac - x0 as f64;
            mid_u[x0] * (1.0 - t) + mid_u[x1] * t
        }
    };

    // kappa_rr ~ kappa_0 * (R_ref/1 GV)^alpha * (B_0/B(r))
    // Approximate: B(r) ~ B_0 * (r_min/r)^1 (toroidal dominates at large r)
    let b0 = 5.0_f64; // nT reference
    let kappa_rr_of_r = move |r: f64| {
        let b_r = b0 * (r_min_au / r.max(r_min_au));
        diffusion.kappa_0_au2_per_s * (b0 / b_r.max(0.01))
    };

    (0..nx)
        .map(|x| {
            let r = r_of_x(x);
            if r >= proxy.r_boundary_au {
                0.0
            } else {
                proxy.modulation_potential(r, &v_sw_of_r, &kappa_rr_of_r)
            }
        })
        .collect()
}

fn main() {
    let cli = Cli::parse();

    // Validate parameters
    assert!(cli.nx >= 2, "nx must be >= 2");
    assert!(cli.tau > 0.5, "tau must be > 0.5 for LBM stability");
    assert!(cli.r_min_au < cli.r_max_au, "r_min_au must be < r_max_au");

    fs::create_dir_all(&cli.out_dir).expect("failed to create output directory");

    // --- LBM initialization ---
    let mut lbm = LbmSolver3D::new(cli.nx, cli.ny, cli.nz, cli.tau);
    lbm.initialize_uniform(1.0, [cli.v_sw, 0.0, 0.0]);

    let mhd_cfg = MhdConfig {
        b0_nt: cli.b0,
        ..Default::default()
    };
    let mut mhd = MhdField::new(cli.nx, cli.ny, cli.nz, mhd_cfg);
    mhd.parker_spiral_init(cli.v_sw);

    // --- PTE solver initialization ---
    let grid = RigidityGrid::new(cli.n_p, cli.r_min_gv, cli.r_max_gv);
    let diff_cfg = DiffusionConfig {
        kappa_0_au2_per_s: cli.kappa_0,
        r_ref_gv: 1.0,
        alpha: cli.alpha,
        epsilon_perp: cli.epsilon_perp,
        solar_epoch_a: cli.solar_epoch_a,
    };

    // Physical dt: LBM timestep in seconds
    // Assume LBM dx = 1 AU / nx, dt = dx / v_sw_physical (400 km/s)
    let dx_au = (cli.r_max_au - cli.r_min_au) / cli.nx as f64;
    let v_sw_ms = 400.0e3_f64; // m/s
    let au_m = 1.496e11_f64;
    let dt_s = dx_au * au_m / v_sw_ms;

    let mut pte = PteSolver::new(
        cli.nx,
        cli.ny,
        cli.nz,
        grid,
        diff_cfg.clone(),
        dt_s,
        dx_au,
    );
    pte.set_boundary_ism(&lis_proton);

    // --- Force-field proxy ---
    let proxy = ForceFieldProxy::new(cli.r_max_au, 0.938272);

    // --- DM source (optional) ---
    let dm_source = if cli.dm_mass_gev > 0.0 {
        let n_cells = cli.nx * cli.ny * cli.nz;
        // Uniform spatial profile (no NFW density from DmForceField here to keep binary standalone)
        let spatial_profile = vec![1.0_f64; n_cells];
        Some(DmSource {
            mass_gev: cli.dm_mass_gev,
            channel: parse_dm_channel(&cli.dm_channel),
            sigma_v_cm3_per_s: cli.dm_sigma_v,
            spatial_profile,
        })
    } else {
        None
    };

    let n_cells = cli.nx * cli.ny * cli.nz;
    let dm_density = vec![0.3_f64; n_cells]; // 0.3 GeV/cm^3 local DM density placeholder

    println!(
        "cr-modulation-sweep: nx={} ny={} nz={} steps={} n_p={} r=[{:.1},{:.1}] AU",
        cli.nx, cli.ny, cli.nz, cli.steps, cli.n_p, cli.r_min_au, cli.r_max_au
    );
    if cli.dm_mass_gev > 0.0 {
        println!(
            "  DM: mass={:.1} GeV channel={} sigma_v={:.2e} cm^3/s",
            cli.dm_mass_gev, cli.dm_channel, cli.dm_sigma_v
        );
    }

    // --- Main loop ---
    for step in 0..cli.steps {
        // LBM evolution
        if step > 0 {
            lbm.phase2_streaming().expect("phase2_streaming failed");
        }
        lbm.compute_macroscopic();
        let lorentz = mhd.lorentz_force();
        lbm.set_force_field(lorentz).expect("set_force_field failed");
        lbm.phase1_collision().expect("phase1_collision failed");
        mhd.evolve_b_field(&lbm.u);

        // PTE one step
        let bx = &mhd.bx;
        let by = &mhd.by;
        let bz = &mhd.bz;
        pte.evolve_one_step(&lbm.u, (bx, by, bz), dm_source.as_ref(), &dm_density);

        // Snapshot output
        if step % cli.snapshot_interval == 0 {
            let phi_map = compute_phi_map(&lbm, &proxy, &diff_cfg, cli.r_min_au, cli.r_max_au);
            let csv = write_snapshot_csv(&pte, &phi_map, step);
            let fname = cli.out_dir.join(format!("cr_snapshot_{step:06}.csv"));
            fs::write(&fname, csv).expect("failed to write snapshot");
            println!("  step {step}: wrote {}", fname.display());
        }
    }

    // Final snapshot
    let phi_map = compute_phi_map(&lbm, &proxy, &diff_cfg, cli.r_min_au, cli.r_max_au);
    let csv = write_snapshot_csv(&pte, &phi_map, cli.steps);
    let fname = cli.out_dir.join(format!("cr_snapshot_{:06}.csv", cli.steps));
    fs::write(&fname, csv).expect("failed to write final snapshot");

    // Summary phi at 1 AU (x=0)
    let phi_1au = phi_map.first().copied().unwrap_or(0.0);
    println!("\nDone. phi(1 AU) = {phi_1au:.4} GV");

    // Write summary file
    let summary_path = cli.out_dir.join("cr_summary.csv");
    let mut f = fs::File::create(&summary_path).expect("failed to create summary");
    writeln!(f, "r_au,phi_gv").unwrap();
    let dr = (cli.r_max_au - cli.r_min_au) / (cli.nx as f64 - 1.0).max(1.0);
    for (x, &phi) in phi_map.iter().enumerate() {
        let r = cli.r_min_au + x as f64 * dr;
        writeln!(f, "{r:.4},{phi:.6e}").unwrap();
    }
    println!("Summary phi(r) written to {}", summary_path.display());
}
