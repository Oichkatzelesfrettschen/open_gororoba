//! euclid-dm-coupling: Sersic baryonic profile + NFW dark matter -> LBM density.
//!
//! Pipeline:
//!   1. Construct Sersic surface brightness -> Abel deprojection -> 3D density
//!   2. Generate NFW dark matter halo for given M_200
//!   3. Combine rho_total = rho_baryon + rho_dm
//!   4. Initialize LBM solver with combined density
//!   5. Apply NFW gravitational force field and evolve
//!   6. Output density snapshots as CSV
//!   7. Measure fractal dimension D_f of evolved density isosurfaces
//!
//! Data provenance: parameters come from real Euclid Q1 catalog via --catalog,
//! or from explicit CLI args with traceable source. The --synthetic flag exists
//! for pipeline testing only and emits warnings.
//!
//! Usage (Euclid catalog):
//!   euclid-dm-coupling --catalog data/external/euclid/zenodo/15106473/useful_physical_measurements.parquet --index 0
//!   `euclid-dm-coupling --catalog <path> --object-id 12345`
//!
//! Usage (explicit params with provenance):
//!   euclid-dm-coupling --sersic-n 4.2 --r-e-kpc 6.1 --i-e 0.8 \
//!     --m200 8.5e11 --redshift 0.35 --ml-ratio 3.2 --ellipticity 0.15
//!
//! Usage (synthetic test -- no physical provenance):
//!   euclid-dm-coupling --synthetic --grid 32 --steps 100

use clap::Parser;
use cosmology_core::{
    box_counting_fractal_dim, nfw_density_from_params, nfw_params_from_mass,
    sersic::{SersicLbmConfig, sersic_to_lbm_density},
};
use lbm_3d::solver::LbmSolver3D;
use std::{fs, path::Path};

#[derive(Parser)]
#[command(name = "euclid-dm-coupling")]
#[command(about = "Sersic + NFW dark matter coupling via LBM")]
struct Args {
    /// Path to Euclid useful_physical_measurements.parquet catalog.
    /// When provided, reads real Sersic parameters from the catalog.
    #[arg(long)]
    catalog: Option<String>,

    /// Select galaxy by object_id from catalog (requires --catalog).
    #[arg(long)]
    object_id: Option<i64>,

    /// Select galaxy by row index from catalog (requires --catalog).
    /// Default: 0 (first valid galaxy).
    #[arg(long)]
    index: Option<usize>,

    /// Use synthetic placeholder parameters for pipeline testing.
    /// WARNING: results have no physical provenance.
    #[arg(long)]
    synthetic: bool,

    /// Grid size (cubic: N x N x N)
    #[arg(long, default_value = "32")]
    grid: usize,

    /// Number of LBM evolution steps
    #[arg(long, default_value = "100")]
    steps: usize,

    /// Sersic index (1=exponential, 4=de Vaucouleurs). REQUIRED unless --catalog/--synthetic.
    #[arg(long, required_unless_present_any = ["synthetic", "catalog"])]
    sersic_n: Option<f64>,

    /// Effective radius in kpc. REQUIRED unless --catalog/--synthetic.
    #[arg(long, required_unless_present_any = ["synthetic", "catalog"])]
    r_e_kpc: Option<f64>,

    /// Surface brightness at R_e. REQUIRED unless --catalog/--synthetic.
    #[arg(long, required_unless_present_any = ["synthetic", "catalog"])]
    i_e: Option<f64>,

    /// Virial mass M_200 in solar masses. REQUIRED unless --catalog/--synthetic.
    #[arg(long, required_unless_present_any = ["synthetic", "catalog"])]
    m200: Option<f64>,

    /// Redshift. REQUIRED unless --catalog/--synthetic.
    #[arg(long, required_unless_present_any = ["synthetic", "catalog"])]
    redshift: Option<f64>,

    /// Cell size in kpc
    #[arg(long, default_value = "1.0")]
    dx_kpc: f64,

    /// LBM relaxation time
    #[arg(long, default_value = "0.8")]
    tau: f64,

    /// Mass-to-light ratio. REQUIRED unless --catalog/--synthetic.
    #[arg(long, required_unless_present_any = ["synthetic", "catalog"])]
    ml_ratio: Option<f64>,

    /// Ellipticity (0 = circular). REQUIRED unless --catalog/--synthetic.
    #[arg(long, required_unless_present_any = ["synthetic", "catalog"])]
    ellipticity: Option<f64>,

    /// Output directory
    #[arg(short, long, default_value = "data/csv")]
    output_dir: String,

    /// Snapshot interval (write every N steps)
    #[arg(long, default_value = "50")]
    snapshot_interval: usize,
}

/// Resolved physics parameters with provenance tag.
struct PhysicsParams {
    sersic_n: f64,
    r_e_kpc: f64,
    i_e: f64,
    m200: f64,
    redshift: f64,
    ml_ratio: f64,
    ellipticity: f64,
    provenance: String,
}

impl PhysicsParams {
    fn resolve(args: &Args) -> Self {
        // Priority: --catalog > explicit CLI args > --synthetic
        if let Some(ref catalog_path) = args.catalog {
            return Self::from_catalog(catalog_path, args);
        }

        if args.synthetic {
            eprintln!("WARNING: --synthetic flag set. Using placeholder parameters.");
            eprintln!("WARNING: Results have NO physical provenance. For testing only.");
            return Self {
                sersic_n: args.sersic_n.unwrap_or(4.0),
                r_e_kpc: args.r_e_kpc.unwrap_or(5.0),
                i_e: args.i_e.unwrap_or(1.0),
                m200: args.m200.unwrap_or(1e12),
                redshift: args.redshift.unwrap_or(0.0),
                ml_ratio: args.ml_ratio.unwrap_or(3.0),
                ellipticity: args.ellipticity.unwrap_or(0.0),
                provenance: "SYNTHETIC (no provenance)".to_string(),
            };
        }

        // All fields guaranteed present by clap required_unless_present_any
        Self {
            sersic_n: args.sersic_n.unwrap(),
            r_e_kpc: args.r_e_kpc.unwrap(),
            i_e: args.i_e.unwrap(),
            m200: args.m200.unwrap(),
            redshift: args.redshift.unwrap(),
            ml_ratio: args.ml_ratio.unwrap(),
            ellipticity: args.ellipticity.unwrap(),
            provenance: "user-supplied CLI".to_string(),
        }
    }

    #[cfg(feature = "euclid-catalog")]
    fn from_catalog(catalog_path: &str, args: &Args) -> Self {
        use cosmology_core::euclid_morphology::read_euclid_physical_measurements;

        eprintln!("Reading Euclid catalog: {catalog_path}");
        let entries = read_euclid_physical_measurements(catalog_path).unwrap_or_else(|e| {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        });
        eprintln!("Catalog: {} valid galaxies", entries.len());

        let entry = if let Some(oid) = args.object_id {
            entries
                .iter()
                .find(|e| e.object_id == oid)
                .unwrap_or_else(|| {
                    eprintln!("ERROR: object_id={oid} not found in catalog");
                    std::process::exit(1);
                })
        } else {
            let idx = args.index.unwrap_or(0);
            entries.get(idx).unwrap_or_else(|| {
                eprintln!(
                    "ERROR: index={idx} out of range (catalog has {} entries)",
                    entries.len()
                );
                std::process::exit(1);
            })
        };

        eprintln!(
            "Selected: object_id={}, n={:.2}, r_e={:.3}\", z={:.3}, log(M*)={:.2}",
            entry.object_id, entry.n, entry.r_e_arcsec, entry.photo_z, entry.log_stellar_mass
        );

        let r_e_kpc = args.r_e_kpc.unwrap_or_else(|| entry.r_e_kpc());
        let m200 = args.m200.unwrap_or_else(|| entry.m200_solar());
        let i_e = args.i_e.unwrap_or_else(|| entry.i_e_solar_per_kpc2());
        let ml = args.ml_ratio.unwrap_or_else(|| entry.ml_ratio());

        Self {
            sersic_n: args.sersic_n.unwrap_or(entry.n),
            r_e_kpc,
            i_e,
            m200,
            redshift: args.redshift.unwrap_or(entry.photo_z),
            ml_ratio: ml,
            ellipticity: args.ellipticity.unwrap_or(entry.ellipticity),
            provenance: format!("Euclid Q1 object_id={} (Zenodo 15106473)", entry.object_id),
        }
    }

    #[cfg(not(feature = "euclid-catalog"))]
    fn from_catalog(catalog_path: &str, _args: &Args) -> Self {
        eprintln!(
            "ERROR: --catalog requires the 'euclid-catalog' feature.\n\
             Rebuild with: cargo run --features euclid-catalog --bin euclid-dm-coupling -- \
             --catalog {catalog_path}"
        );
        std::process::exit(1);
    }
}

fn write_density_csv(
    rho: &[f64],
    nx: usize,
    ny: usize,
    _nz: usize,
    step: usize,
    path: &Path,
) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["step", "x", "y", "z", "rho"])?;

    let cz = _nz / 2;
    for iy in 0..ny {
        for ix in 0..nx {
            let idx = cz * ny * nx + iy * nx + ix;
            wtr.write_record(&[
                step.to_string(),
                ix.to_string(),
                iy.to_string(),
                cz.to_string(),
                format!("{:.8e}", rho[idx]),
            ])?;
        }
    }
    wtr.flush()
}

fn main() {
    let args = Args::parse();
    let phys = PhysicsParams::resolve(&args);
    let n = args.grid;
    let total = n * n * n;

    let out_dir = Path::new(&args.output_dir);
    fs::create_dir_all(out_dir).expect("failed to create output directory");

    eprintln!("=== Euclid DM Coupling ===");
    eprintln!("Data provenance: {}", phys.provenance);
    eprintln!("Grid: {}^3, steps: {}, tau: {}", n, args.steps, args.tau);
    eprintln!(
        "Sersic: n={}, R_e={} kpc, I_e={}, M/L={}",
        phys.sersic_n, phys.r_e_kpc, phys.i_e, phys.ml_ratio
    );
    eprintln!("NFW: M200={:.2e} Msun, z={}", phys.m200, phys.redshift);

    // 1. Baryonic density from Sersic profile
    let cfg = SersicLbmConfig {
        nx: n,
        ny: n,
        nz: n,
        dx_kpc: args.dx_kpc,
        r_e_kpc: phys.r_e_kpc,
        n: phys.sersic_n,
        i_e: phys.i_e,
        ellipticity: phys.ellipticity,
        pa_rad: 0.0,
        ml_ratio: phys.ml_ratio,
    };
    let rho_baryon = sersic_to_lbm_density(&cfg);
    let baryon_total: f64 = rho_baryon.iter().sum();
    eprintln!("Baryonic density: total={:.6e}", baryon_total);

    // 2. NFW dark matter density
    let nfw = nfw_params_from_mass(phys.m200, phys.redshift);
    eprintln!(
        "NFW: c200={:.2}, r_s={:.2} kpc, r200={:.2} kpc",
        nfw.c200, nfw.r_s_kpc, nfw.r200_kpc
    );

    let mut rho_dm = vec![0.0_f64; total];
    let cx = n as f64 / 2.0;
    let cy = n as f64 / 2.0;
    let cz = n as f64 / 2.0;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let dx = (ix as f64 + 0.5 - cx) * args.dx_kpc;
                let dy = (iy as f64 + 0.5 - cy) * args.dx_kpc;
                let dz = (iz as f64 + 0.5 - cz) * args.dx_kpc;
                let r = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                let idx = iz * n * n + iy * n + ix;
                rho_dm[idx] = nfw_density_from_params(r, &nfw);
            }
        }
    }
    let dm_total: f64 = rho_dm.iter().sum();
    eprintln!("DM density: total={:.6e}", dm_total);

    // 3. Combine densities (normalize to LBM-safe range)
    let max_combined: f64 = rho_baryon
        .iter()
        .zip(rho_dm.iter())
        .map(|(b, d)| b + d)
        .fold(0.0_f64, f64::max);

    let scale = if max_combined > 0.0 {
        2.0 / max_combined
    } else {
        1.0
    };

    let mut rho_total = vec![0.0_f64; total];
    for i in 0..total {
        rho_total[i] = (rho_baryon[i] + rho_dm[i]) * scale;
    }
    eprintln!(
        "Combined: scale={:.6e}, max_rho={:.4}",
        scale,
        rho_total.iter().copied().fold(0.0_f64, f64::max)
    );

    // 4. NFW gravitational force field (radial, toward center)
    let mut raw_g = vec![[0.0_f64; 3]; total];
    let mut max_g = 0.0_f64;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let dx = (ix as f64 + 0.5 - cx) * args.dx_kpc;
                let dy = (iy as f64 + 0.5 - cy) * args.dx_kpc;
                let dz = (iz as f64 + 0.5 - cz) * args.dx_kpc;
                let r = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                let idx = iz * n * n + iy * n + ix;

                let x_rs = r / nfw.r_s_kpc;
                let m_enc = 4.0
                    * std::f64::consts::PI
                    * nfw.rho_s_solar_per_kpc3
                    * nfw.r_s_kpc.powi(3)
                    * ((1.0 + x_rs).ln() - x_rs / (1.0 + x_rs));
                let g_mag = m_enc / (r * r);
                max_g = max_g.max(g_mag);

                if r > 0.1 {
                    raw_g[idx] = [-g_mag * dx / r, -g_mag * dy / r, -g_mag * dz / r];
                }
            }
        }
    }

    let force_target = 1e-4;
    let force_norm = if max_g > 0.0 {
        force_target / max_g
    } else {
        0.0
    };
    let force_field: Vec<[f64; 3]> = raw_g
        .iter()
        .map(|g| [g[0] * force_norm, g[1] * force_norm, g[2] * force_norm])
        .collect();
    eprintln!(
        "Force normalization: max_raw={:.6e}, scale={:.6e}",
        max_g, force_norm
    );

    // 5. Initialize LBM and evolve
    let mut solver = LbmSolver3D::new(n, n, n, args.tau);

    let lattice = &solver.collider.lattice;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let idx = iz * n * n + iy * n + ix;
                let rho_init = rho_total[idx].max(0.01);
                let f_eq = lbm_3d::solver::BgkCollision::initialize_rest(rho_init, lattice);
                for (dir, &val) in f_eq.iter().enumerate() {
                    solver.f[lbm_3d::solver::aosoa_idx(idx, dir)] = val;
                }
                solver.rho[idx] = rho_init;
                solver.u[idx] = [0.0, 0.0, 0.0];
            }
        }
    }
    solver
        .set_force_field(force_field)
        .expect("force field size mismatch");

    // Initial snapshot
    let snap_path = out_dir.join("euclid_dm_density_step_0.csv");
    write_density_csv(&solver.rho, n, n, n, 0, &snap_path).expect("failed to write snapshot");
    eprintln!("Snapshot -> {}", snap_path.display());

    let df_initial = box_counting_fractal_dim(&solver.rho, n, n, n);
    eprintln!("D_f (initial): {:.4}", df_initial);

    // Evolve
    for step in 1..=args.steps {
        solver.evolve_one_step();

        if step % args.snapshot_interval == 0 || step == args.steps {
            let snap_path = out_dir.join(format!("euclid_dm_density_step_{}.csv", step));
            write_density_csv(&solver.rho, n, n, n, step, &snap_path)
                .expect("failed to write snapshot");
            let df = box_counting_fractal_dim(&solver.rho, n, n, n);
            let mass = solver.total_mass();
            eprintln!(
                "Step {}: mass={:.6e}, D_f={:.4} -> {}",
                step,
                mass,
                df,
                snap_path.display()
            );
        }
    }

    // Final D_f
    let df_final = box_counting_fractal_dim(&solver.rho, n, n, n);
    eprintln!("\n=== Results ===");
    eprintln!("Provenance: {}", phys.provenance);
    eprintln!("D_f initial: {:.4}", df_initial);
    eprintln!("D_f final:   {:.4}", df_final);
    eprintln!(
        "Delta D_f:   {:.4} (positive = clumping, negative = spreading)",
        df_initial - df_final
    );

    // Summary CSV
    let summary_path = out_dir.join("euclid_dm_coupling_summary.csv");
    let mut wtr = csv::Writer::from_path(&summary_path).expect("failed to create summary CSV");
    wtr.write_record([
        "grid",
        "steps",
        "sersic_n",
        "r_e_kpc",
        "m200",
        "tau",
        "df_initial",
        "df_final",
        "provenance",
    ])
    .unwrap();
    wtr.write_record(&[
        n.to_string(),
        args.steps.to_string(),
        format!("{:.2}", phys.sersic_n),
        format!("{:.2}", phys.r_e_kpc),
        format!("{:.2e}", phys.m200),
        format!("{:.2}", args.tau),
        format!("{:.4}", df_initial),
        format!("{:.4}", df_final),
        phys.provenance.clone(),
    ])
    .unwrap();
    wtr.flush().unwrap();
    eprintln!("Summary -> {}", summary_path.display());
}
