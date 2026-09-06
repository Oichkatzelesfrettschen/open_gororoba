//! D3Q19 LBM + MHD + density-weighted dark matter forcing.
//!
//! Guo coupling consumes force density. Diagnostics compare maxima of lattice
//! force-density fields, not a physical suppression bound: magnetic normalization,
//! mesh consistency, gravitational frame and numerical sensitivity remain unresolved.

use clap::{Args, ValueEnum};
use cosmology_core::concentration_mass_relation;
use lbm_3d::{
    dm_force::{DmForceConfig, DmForceField, combine_forces},
    mhd::{MagneticDiffusivity, MhdConfig, MhdField},
    open_x_boundary::{OpenXBoundary, XOutflow, population_mass},
    solver::{BgkCollision, LbmSolver3D},
    units::{LatticeUnits, UniformCartesianMesh},
};
use std::{
    fs,
    io::{BufRead, Write},
    path::PathBuf,
};

/// D3Q19 LBM + MHD + NFW dark matter gravitational coupling.
///
/// Couples the magnetized solar wind simulation (Parker spiral B-field,
/// Zou-He inlet, Guo forcing) with a static NFW DM halo gravitational
/// acceleration field. Reports lattice force-density diagnostics.
#[derive(Args)]
pub struct Cli {
    /// Explicit numerical max-x closure; physical suitability requires separate validation.
    #[arg(long, value_enum)]
    x_outflow: XOutflowChoice,

    /// Relative mass-ledger residual budget; collision has zero allowed mass source.
    #[arg(long)]
    max_relative_mass_ledger_error: f64,
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

    /// Magnetic diffusivity in lattice length squared per LBM timestep (0 = ideal MHD).
    #[arg(
        long = "magnetic-diffusivity-lattice",
        alias = "eta",
        default_value_t = 0.0
    )]
    eta: f64,

    /// Magnetic diffusivity in m^2/s, converted using admitted IC mesh and timestep metadata.
    #[arg(long, conflicts_with = "eta", requires = "ic_file")]
    magnetic_diffusivity_m2_s: Option<f64>,

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

    /// NFW concentration parameter. If not set, derived from --dm-m200
    /// via the Dutton & Maccio (2014) concentration-mass relation.
    #[arg(long)]
    dm_c200: Option<f64>,

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

    /// Minimum heliocentric distance (AU) for DM force grid mapping.
    /// x=0 maps to this distance. Default 0.5 (centered-on-1-AU slab).
    #[arg(long, default_value_t = 0.5)]
    dm_r_min: f64,

    /// Maximum heliocentric distance (AU) for DM force grid mapping.
    /// x=nx-1 maps to this distance. Default 1.5 (centered-on-1-AU slab).
    #[arg(long, default_value_t = 1.5)]
    dm_r_max: f64,

    /// Disable DM coupling (pure MHD baseline for A/B comparison)
    #[arg(long, default_value_t = false)]
    no_dm: bool,

    /// Initial-condition CSV with complete SI mesh and magnetic-normalization metadata.
    /// Every cell supplies x,y,z,rho,ux,uy,uz,bx,by,bz in declared lattice units.
    /// Legacy solar wind-ic exports require a separately justified metadata amendment.
    #[arg(long)]
    ic_file: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum XOutflowChoice {
    ZeroGradientPopulations,
}
impl From<XOutflowChoice> for XOutflow {
    fn from(value: XOutflowChoice) -> Self {
        match value {
            XOutflowChoice::ZeroGradientPopulations => Self::ZeroGradientPopulations,
        }
    }
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
        "x,y,rho,ux,uy,uz,bx,by,bz,dm_ax_lattice,dm_ay_lattice,dm_az_lattice,dm_density_kg_m3"
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
                u[0],
                u[1],
                u[2],
                mhd.bx[idx],
                mhd.by[idx],
                mhd.bz[idx],
                dm_fx,
                dm_fy,
                dm_fz,
                dm_rho,
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

fn gravitational_force_density(
    dm: &DmForceField,
    density: &[f64],
) -> anyhow::Result<Vec<[f64; 3]>> {
    anyhow::ensure!(
        dm.force.len() == density.len(),
        "gravity density length mismatch"
    );
    anyhow::ensure!(
        density
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "gravity density must be finite and nonnegative"
    );
    let force: Vec<_> = dm
        .force
        .iter()
        .zip(density)
        .map(|(acceleration, rho)| acceleration.map(|component| component * rho))
        .collect();
    anyhow::ensure!(
        force.iter().flatten().all(|value| value.is_finite()),
        "gravity force density must be finite"
    );
    Ok(force)
}

fn ratio_of_maxima(numerator: f64, denominator: f64) -> anyhow::Result<Option<f64>> {
    anyhow::ensure!(
        numerator.is_finite() && numerator >= 0.0 && denominator.is_finite() && denominator >= 0.0,
        "force maxima must be finite and nonnegative"
    );
    if denominator == 0.0 {
        return Ok(None);
    }
    let ratio = numerator / denominator;
    anyhow::ensure!(ratio.is_finite(), "force ratio overflow");
    Ok(Some(ratio))
}

/// Physical input declarations parsed from CSV comment headers.
/// Unit admission verifies consistency; provenance and observational independence
/// require separate evidence.
#[derive(Debug, Default)]
struct IcMetadata {
    n_ref_cm3: Option<f64>,
    v_ref_kms: Option<f64>,
    u_scale: Option<f64>,
    physical_units: Option<LatticeUnits>,
    physical_mesh: Option<UniformCartesianMesh>,
}

fn admit_ic_metadata(lines: &[String], dimensions: [usize; 3]) -> anyhow::Result<IcMetadata> {
    let mut fields = std::collections::BTreeMap::new();
    for line in lines {
        if let Some((key, value)) = line
            .trim()
            .strip_prefix('#')
            .and_then(|s| s.trim().split_once('='))
        {
            anyhow::ensure!(
                fields.insert(key.trim(), value.trim()).is_none(),
                "duplicate IC metadata key: {}",
                key.trim()
            );
        }
    }
    let required = |key: &str| -> anyhow::Result<&str> {
        fields
            .get(key)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("IC physical admission requires metadata {key}"))
    };
    anyhow::ensure!(
        required("mesh_kind")? == "uniform_cartesian",
        "IC physical admission requires uniform_cartesian mesh; logarithmic radial grids require a separate solver/discretization"
    );
    anyhow::ensure!(
        required("coordinate_frame")? == "heliocentric_cartesian",
        "IC coordinates require an explicit heliocentric_cartesian declaration; gravitational acceleration frame remains a separate model choice"
    );
    anyhow::ensure!(
        required("field_storage")? == "lattice",
        "IC fields must declare lattice storage"
    );
    anyhow::ensure!(
        required("magnetic_normalization")? == "sqrt_mu0_rho_ref_dx_over_dt",
        "IC magnetic normalization must use sqrt(mu0*rho_ref)*dx/dt"
    );
    let number = |key: &str| -> anyhow::Result<f64> { Ok(required(key)?.parse()?) };
    let mesh = UniformCartesianMesh::new(
        dimensions,
        [
            number("origin_x_m")?,
            number("origin_y_m")?,
            number("origin_z_m")?,
        ],
        number("spacing_m")?,
    )?;
    let units = LatticeUnits::new(&mesh, number("timestep_s")?, number("density_ref_kg_m3")?)?;
    let magnetic_unit = number("magnetic_unit_t")?;
    anyhow::ensure!(
        magnetic_unit.is_finite() && (magnetic_unit / units.magnetic_unit_t() - 1.0).abs() <= 1e-12,
        "IC magnetic unit disagrees with SI mesh and mass-density normalization"
    );
    let n_ref = number("n_ref_cm3")?;
    let v_ref = number("v_ref_kms")?;
    let u_scale = number("u_scale")?;
    anyhow::ensure!(
        [n_ref, v_ref, u_scale]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0),
        "IC reference values must be finite and positive"
    );
    anyhow::ensure!(
        (v_ref * 1000.0 / u_scale / units.velocity_unit_m_s() - 1.0).abs() <= 1e-12,
        "IC velocity reference disagrees with dx/dt"
    );
    // The drag interface assumes a proton plasma; a composition change needs
    // a declared mass-per-reference-particle contract.
    anyhow::ensure!(
        (n_ref * 1e6 * lbm_3d::dm_force::DRAG_PROTON_MASS_KG / units.density_ref_kg_m3() - 1.0)
            .abs()
            <= 1e-12,
        "IC proton number and mass density references disagree with declared drag model"
    );
    Ok(IcMetadata {
        n_ref_cm3: Some(n_ref),
        v_ref_kms: Some(v_ref),
        u_scale: Some(u_scale),
        physical_units: Some(units),
        physical_mesh: Some(mesh),
    })
}

fn load_ic_file(
    path: &std::path::Path,
    solver: &mut LbmSolver3D,
    mhd: &mut MhdField,
) -> anyhow::Result<(usize, IcMetadata)> {
    let file = fs::File::open(path)?;
    let lines: Vec<String> = std::io::BufReader::new(file)
        .lines()
        .collect::<Result<_, _>>()?;
    load_ic_lines(lines, solver, mhd)
}

fn load_ic_lines(
    lines: Vec<String>,
    solver: &mut LbmSolver3D,
    mhd: &mut MhdField,
) -> anyhow::Result<(usize, IcMetadata)> {
    let lattice = &solver.collider.lattice;
    let nx = solver.nx;
    let ny = solver.ny;

    let mut loaded = 0usize;
    let meta = admit_ic_metadata(&lines, [nx, ny, solver.nz])?;
    let mut cells = vec![None; nx * ny * solver.nz];

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line == "x,y,z,rho,ux,uy,uz,bx,by,bz" {
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        anyhow::ensure!(fields.len() == 10, "IC row requires exactly ten fields");
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

        anyhow::ensure!(
            x < nx && y < ny && z < solver.nz,
            "IC cell outside declared mesh"
        );
        let idx = z * (nx * ny) + y * nx + x;
        anyhow::ensure!(
            [rho, ux, uy, uz, bx, by, bz]
                .into_iter()
                .all(f64::is_finite)
                && rho > 0.0,
            "IC fields must be finite with positive density"
        );
        anyhow::ensure!(
            BgkCollision::initialize_with_velocity(rho, [ux, uy, uz], lattice)
                .into_iter()
                .all(f64::is_finite),
            "IC equilibrium populations overflow"
        );
        anyhow::ensure!(
            cells[idx].replace([rho, ux, uy, uz, bx, by, bz]).is_none(),
            "duplicate IC cell"
        );
        loaded += 1;
    }
    anyhow::ensure!(
        loaded == cells.len(),
        "IC requires every mesh cell exactly once: loaded {loaded} of {}",
        cells.len()
    );
    for (idx, cell) in cells.into_iter().enumerate() {
        let [rho, ux, uy, uz, bx, by, bz] =
            cell.ok_or_else(|| anyhow::anyhow!("IC missing cell {idx}"))?;
        solver.rho[idx] = rho;
        solver.u[idx] = [ux, uy, uz];

        // Re-initialize distribution function to equilibrium at (rho, u)
        let f_eq = BgkCollision::initialize_with_velocity(rho, [ux, uy, uz], lattice);
        for (i, &fi) in f_eq.iter().enumerate() {
            solver.f[lbm_3d::solver::aosoa_idx(idx, i)] = fi;
        }

        mhd.bx[idx] = bx;
        mhd.by[idx] = by;
        mhd.bz[idx] = bz;
    }
    Ok((loaded, meta))
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    anyhow::ensure!(
        cli.nx >= 2 && cli.ny > 0 && cli.nz > 0 && cli.snap_interval > 0,
        "positive grid/snapshot sizes and nx >= 2 required"
    );
    anyhow::ensure!(
        cli.max_relative_mass_ledger_error.is_finite() && cli.max_relative_mass_ledger_error >= 0.0,
        "finite nonnegative mass-ledger budget required"
    );
    eprintln!(
        "physical_comparison_admission=blocked: numerical open-x population closure is measured; its physical suitability, MHD magnetic boundary conditions and gravitational acceleration frame require separate specifications"
    );
    let dm_label = if cli.no_dm { "OFF" } else { "ON" };
    let ic_label = if cli.ic_file.is_some() {
        "declared-input"
    } else {
        "synthetic"
    };
    eprintln!(
        "solar wind-dm-mhd: {}x{}x{}, {} steps, tau={}, B0={} nT, v_sw={}, DM={}, IC={}",
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
    let mut mhd = MhdField::try_new(cli.nx, cli.ny, cli.nz, mhd_config)?;

    // Initialize state: real data from IC file, or synthetic uniform+Parker
    let (u_sw, ic_meta) = if let Some(ref ic_path) = cli.ic_file {
        let (loaded, ic_meta) = load_ic_file(ic_path, &mut solver, &mut mhd)?;
        eprintln!(
            "loaded {} cells from IC file: {}",
            loaded,
            ic_path.display()
        );
        // Report IC metadata when present
        if let Some(n) = ic_meta.n_ref_cm3 {
            eprintln!("  IC metadata: n_ref={n:.2} cm^-3");
        }
        if let Some(v) = ic_meta.v_ref_kms {
            eprintln!("  IC metadata: v_ref={v:.1} km/s");
        }
        if let Some(u) = ic_meta.u_scale {
            eprintln!("  IC metadata: u_scale={u:.4}");
        }

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
        ([median_ux, 0.0, 0.0], ic_meta)
    } else {
        let u_init = [cli.v_sw, 0.0, 0.0];
        solver.initialize_uniform(1.0, u_init);
        mhd.parker_spiral_init(cli.v_sw);
        (u_init, IcMetadata::default())
    };

    mhd.config.eta = admitted_magnetic_diffusivity(
        cli.eta,
        cli.magnetic_diffusivity_m2_s,
        ic_meta.physical_units.as_ref(),
    )?
    .lattice_value();
    mhd.validate_transport()?;
    eprintln!(
        "magnetic_diffusivity_lattice={:.17e} dt_mhd={:.17e} diffusion_gate_scope=periodic_diffusion_only",
        mhd.config.eta, mhd.config.dt_mhd,
    );
    if let Some(requested_m2_s) = cli.magnetic_diffusivity_m2_s {
        let units = ic_meta.physical_units.as_ref().ok_or_else(|| {
            anyhow::anyhow!("SI magnetic diffusivity receipt requires admitted IC physical units")
        })?;
        eprintln!(
            "magnetic_diffusivity_m2_s={requested_m2_s:.17e} magnetic_diffusivity_m2_s_per_lattice_unit={:.17e}",
            units.diffusivity_to_si(1.0),
        );
    }

    // Helmholtz projection: remove magnetic monopoles from Cartesian discretization
    let (div_before, div_after) = mhd.project_divergence_free(5000, 1e-12);
    let initial_energy = mhd.magnetic_energy();
    eprintln!("div(B) projection: {div_before:.6e} -> {div_after:.6e}");
    eprintln!("initial magnetic energy (post-projection): {initial_energy:.6e}");

    // Initialize DM force field (if enabled)
    let dm_field = if !cli.no_dm {
        if let Some(mesh) = &ic_meta.physical_mesh {
            let origin = mesh.origin_m();
            let upper = mesh.position_m([cli.nx - 1, 0, 0])?;
            let au_m = 1.496e11;
            let close = |left: f64, right: f64| {
                (left - right).abs() <= 1e-12 * left.abs().max(right.abs()).max(1.0)
            };
            anyhow::ensure!(
                cli.nx > 1
                    && close(origin[0], cli.dm_r_min * au_m)
                    && close(upper[0], cli.dm_r_max * au_m)
                    && close(origin[1], -(cli.ny as f64) * mesh.spacing_m() / 2.0)
                    && close(origin[2], -(cli.nz as f64) * mesh.spacing_m() / 2.0),
                "DM grid mapping disagrees with admitted SI mesh; declare matching radial endpoints and centered transverse coordinates"
            );
        }
        // Use IC metadata for unit conversion when available (overrides CLI defaults)
        let n_ref = ic_meta.n_ref_cm3.unwrap_or(cli.dm_n_ref);
        let v_ref = ic_meta.v_ref_kms.unwrap_or(cli.dm_v_ref);
        let u_sc = ic_meta.u_scale.unwrap_or(cli.v_sw);

        // Derive c200 from m200 via concentration-mass relation unless
        // the user explicitly set --dm-c200 on the command line.
        let c200 = cli
            .dm_c200
            .unwrap_or_else(|| concentration_mass_relation(cli.dm_m200, 0.0));

        // Derive force_scale from actual CLI nx and v_sw, not from the
        // DmForceConfig default (which assumes nx=128, v_sw=400 km/s).
        // delta_x = 1 AU / nx, delta_t = delta_x * (v_sw_lattice / v_sw_phys),
        // force_scale = delta_t^2 / delta_x.
        let au_m = 1.496e11;
        let delta_x = ic_meta
            .physical_units
            .as_ref()
            .map_or(au_m / cli.nx as f64, LatticeUnits::spacing_m);
        let v_sw_phys = v_ref * 1.0e3; // km/s -> m/s
        let delta_t = ic_meta
            .physical_units
            .as_ref()
            .map_or(delta_x * (u_sc / v_sw_phys), LatticeUnits::timestep_s);
        let force_scale = delta_t * delta_t / delta_x;

        let dm_config = DmForceConfig {
            rho_dm_local_gev_cm3: cli.dm_density,
            m200_solar: cli.dm_m200,
            c200,
            v_dm_wind: [cli.dm_wind_x, cli.dm_wind_y, cli.dm_wind_z],
            eta_wake: cli.dm_wake,
            force_scale,
            sigma_chi_b: cli.dm_sigma,
            n_ref_cm3: n_ref,
            v_ref_kms: v_ref,
            u_scale: u_sc,
            r_min_au: cli.dm_r_min,
            r_max_au: cli.dm_r_max,
            ..DmForceConfig::default()
        };
        let field = DmForceField::new(cli.nx, cli.ny, cli.nz, dm_config);
        eprintln!(
            "DM max |a_grav|: {:.6e} (lattice acceleration)",
            field.max_force_magnitude()
        );
        if cli.dm_sigma > 0.0 {
            eprintln!(
                "DM drag: sigma={:.3e} cm^2, kappa={:.6e}, n_ref={:.1} cm^-3, v_ref={:.0} km/s",
                cli.dm_sigma, field.kappa_drag, cli.dm_n_ref, cli.dm_v_ref,
            );
        }
        Some(field)
    } else {
        None
    };

    let mut open_boundary = OpenXBoundary::new([cli.nx, cli.ny, cli.nz])?;

    // Output directory setup
    if let Some(ref dir) = cli.out {
        fs::create_dir_all(dir)?;
    }
    let mut ledger_file = cli
        .out
        .as_ref()
        .map(|dir| fs::File::create(dir.join("mass_flux_ledger.csv")))
        .transpose()?;
    let ledger_header = "step,mass_before,min_x_outgoing,max_x_outgoing,min_x_incoming,max_x_incoming,mass_after_streaming,mass_after_boundary,mass_after_collision,streaming_residual,boundary_residual,collision_mass_delta,step_mass_residual,cumulative_net_inflow,cumulative_mass_residual";
    println!("{ledger_header}");
    if let Some(file) = &mut ledger_file {
        writeln!(file, "{ledger_header}")?;
    }
    let initial_population_mass = population_mass(&solver)?;
    let mut cumulative_net_inflow = 0.0;
    eprintln!(
        "mass_ledger_units=lattice_population_mass x_outflow=zero_gradient_populations transverse_boundaries=periodic budget={:.17e}",
        cli.max_relative_mass_ledger_error
    );

    // Time loop (stream-collide ordering)
    //
    // Correct LBM cycle for forced MHD:
    //   stream -> BC -> macroscopic -> force -> collision -> B-field
    //
    // This ensures forces are computed from the post-streaming velocity,
    // making the force field and collision velocity temporally consistent.
    // Each iteration transports initialized populations, reconstructs x faces,
    // and collides. The ledger uses populations rather than cached density.
    for step in 0..cli.steps {
        let ledger =
            open_boundary.stream_and_reconstruct(&mut solver, u_sw, cli.x_outflow.into())?;

        // 4. Compute Lorentz force from current B-field
        let lorentz = mhd.lorentz_force();

        // 5. Combine with DM gravitational force (if enabled)
        let combined = match &dm_field {
            Some(dm) => {
                let gravity = gravitational_force_density(dm, &solver.rho)?;
                let grav_combined = combine_forces(&lorentz, &gravity);
                // Add dynamic drag force when sigma_chi_b > 0 (kappa-based)
                if dm.config.sigma_chi_b > 0.0 {
                    let drag = dm.drag_force_density_lattice(&solver.rho, &solver.u);
                    combine_forces(&grav_combined, &drag)
                } else {
                    grav_combined
                }
            }
            None => lorentz,
        };

        // 6. Set combined force field for Guo scheme
        solver.set_force_field(combined)?;

        // 7. Collision (BGK + Phi_i source term with consistent u and F)
        solver.phase1_collision()?;
        let mass_after_collision = population_mass(&solver)?;
        let collision_mass_delta = mass_after_collision - ledger.mass_after_boundary;
        let step_mass_residual =
            mass_after_collision - ledger.mass_before - ledger.face.net_incoming();
        cumulative_net_inflow += ledger.face.net_incoming();
        let cumulative_mass_residual =
            mass_after_collision - initial_population_mass - cumulative_net_inflow;
        let row = format!(
            "{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            step + 1,
            ledger.mass_before,
            ledger.face.min_x_outgoing,
            ledger.face.max_x_outgoing,
            ledger.face.min_x_incoming,
            ledger.face.max_x_incoming,
            ledger.mass_after_streaming,
            ledger.mass_after_boundary,
            mass_after_collision,
            ledger.streaming_residual(),
            ledger.boundary_residual(),
            collision_mass_delta,
            step_mass_residual,
            cumulative_net_inflow,
            cumulative_mass_residual
        );
        println!("{row}");
        if let Some(file) = &mut ledger_file {
            writeln!(file, "{row}")?;
            file.flush()?;
        }
        let scale = initial_population_mass
            .abs()
            .max(ledger.mass_before.abs())
            .max(1.0);
        anyhow::ensure!(
            [
                ledger.streaming_residual(),
                ledger.boundary_residual(),
                collision_mass_delta,
                step_mass_residual,
                cumulative_mass_residual
            ]
            .into_iter()
            .all(|residual| residual.is_finite()
                && residual.abs() <= cli.max_relative_mass_ledger_error * scale),
            "mass-ledger budget exceeded at step {}; retained row includes collision and flux residuals",
            step + 1
        );

        // 8. Evolve B-field using force-corrected velocity u*
        mhd.try_evolve_b_field(&solver.u)?;

        // 9. Periodic output
        if (step + 1) % cli.snap_interval == 0 || step == 0 {
            let energy = mhd.magnetic_energy();
            let div = mhd.max_div_b();
            let mass = mass_after_collision;

            // Compute force ratio
            let lorentz_now = mhd.lorentz_force();
            let max_lorentz = max_force_mag(&lorentz_now);
            let max_dm = match &dm_field {
                Some(dm) => max_force_mag(&gravitational_force_density(dm, &solver.rho)?),
                None => 0.0,
            };
            let ratio = match ratio_of_maxima(max_dm, max_lorentz)? {
                Some(value) => format!("{value:.3e}"),
                None => "undefined_zero_lorentz".to_owned(),
            };

            // Report drag force magnitude when sigma > 0 (kappa-based)
            let drag_info = if let Some(dm) = dm_field.as_ref() {
                if dm.config.sigma_chi_b > 0.0 {
                    let drag = dm.drag_force_density_lattice(&solver.rho, &solver.u);
                    let max_drag = max_force_mag(&drag);
                    format!("  max_drag_force_density_lattice={max_drag:.3e}")
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            eprintln!(
                "step={:>6}  mass={mass:.6}  B_energy={energy:.6e}  max|divB|={div:.6e}  max_gravity_force_density_lattice={max_dm:.3e}  max_lorentz_force_density_lattice={max_lorentz:.3e}  gravity_to_lorentz_ratio_of_maxima={ratio}{drag_info}  physical_bound=unassessed",
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

fn admitted_magnetic_diffusivity(
    lattice_value: f64,
    physical_value: Option<f64>,
    units: Option<&LatticeUnits>,
) -> anyhow::Result<MagneticDiffusivity> {
    match physical_value {
        Some(value) => {
            let units = units.ok_or_else(|| {
                anyhow::anyhow!("SI magnetic diffusivity requires admitted IC physical units")
            })?;
            Ok(MagneticDiffusivity::from_si(value, units)?)
        }
        None => Ok(MagneticDiffusivity::from_lattice(lattice_value)?),
    }
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

    fn parse_diffusivity(extra: &[&str]) -> Result<TestArgs, clap::Error> {
        TestArgs::try_parse_from(
            [
                "solar",
                "--x-outflow",
                "zero-gradient-populations",
                "--max-relative-mass-ledger-error",
                "1e-10",
            ]
            .into_iter()
            .chain(extra.iter().copied()),
        )
    }

    #[test]
    fn magnetic_diffusivity_cli_preserves_lattice_alias_and_requires_si_admission() {
        assert_eq!(
            parse_diffusivity(&[]).unwrap().cli.eta.to_bits(),
            0.0_f64.to_bits()
        );
        for flag in ["--eta", "--magnetic-diffusivity-lattice"] {
            assert_eq!(
                parse_diffusivity(&[flag, "0.125"])
                    .unwrap()
                    .cli
                    .eta
                    .to_bits(),
                0.125_f64.to_bits()
            );
            assert!(
                parse_diffusivity(&[
                    flag,
                    "0",
                    "--magnetic-diffusivity-m2-s",
                    "4e8",
                    "--ic-file",
                    "input.csv"
                ])
                .is_err()
            );
        }
        assert!(parse_diffusivity(&["--magnetic-diffusivity-m2-s", "4e8"]).is_err());
        let parsed = parse_diffusivity(&[
            "--magnetic-diffusivity-m2-s",
            "4e8",
            "--ic-file",
            "input.csv",
        ])
        .unwrap()
        .cli;
        assert!(
            admitted_magnetic_diffusivity(parsed.eta, parsed.magnetic_diffusivity_m2_s, None)
                .is_err()
        );
    }

    #[test]
    fn magnetic_diffusivity_si_conversion_uses_admitted_mesh_and_timestep() {
        let metadata = admit_ic_metadata(&admitted_fixture(), [2; 3]).unwrap();
        let units = metadata.physical_units.as_ref();
        let converted = admitted_magnetic_diffusivity(0.0, Some(4e8), units).unwrap();
        assert!((converted.lattice_value() - 5e-5).abs() < 1e-18);
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(admitted_magnetic_diffusivity(0.0, Some(invalid), units).is_err());
        }
        assert_eq!(
            admitted_magnetic_diffusivity(0.125, None, None)
                .unwrap()
                .lattice_value()
                .to_bits(),
            0.125_f64.to_bits()
        );
    }

    fn admitted_fixture() -> Vec<String> {
        let density_ref = 5e6 * lbm_3d::dm_force::DRAG_PROTON_MASS_KG;
        let mesh = UniformCartesianMesh::new([2; 3], [0.0; 3], 1e6).unwrap();
        let units = LatticeUnits::new(&mesh, 0.125, density_ref).unwrap();
        let mut lines = format!("# mesh_kind=uniform_cartesian\n# coordinate_frame=heliocentric_cartesian\n# field_storage=lattice\n# magnetic_normalization=sqrt_mu0_rho_ref_dx_over_dt\n# origin_x_m=0\n# origin_y_m=0\n# origin_z_m=0\n# spacing_m=1000000\n# timestep_s=0.125\n# density_ref_kg_m3={density_ref:.17e}\n# magnetic_unit_t={:.17e}\n# n_ref_cm3=5\n# v_ref_kms=400\n# u_scale=0.05", units.magnetic_unit_t())
            .lines().map(str::to_owned).collect::<Vec<_>>();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    lines.push(format!("{x},{y},{z},1,0.05,0,0,0.001,0,0"));
                }
            }
        }
        lines
    }

    #[test]
    fn physical_ic_admission_rejects_missing_logarithmic_and_mismatched_units() {
        let valid = admitted_fixture();
        assert!(admit_ic_metadata(&valid, [2; 3]).is_ok());
        assert!(admit_ic_metadata(&[], [2; 3]).is_err());
        for (key, replacement) in [
            ("mesh_kind", "log_radial"),
            ("magnetic_unit_t", "1"),
            ("spacing_m", "0"),
            ("v_ref_kms", "300"),
            ("density_ref_kg_m3", "1"),
        ] {
            let prefix = format!("# {key}=");
            let changed = valid
                .iter()
                .map(|line| {
                    if line.starts_with(&prefix) {
                        format!("{prefix}{replacement}")
                    } else {
                        line.clone()
                    }
                })
                .collect::<Vec<_>>();
            assert!(
                admit_ic_metadata(&changed, [2; 3]).is_err(),
                "accepted {key}"
            );
        }
    }

    #[test]
    fn malformed_ic_preserves_all_target_fields() {
        let valid = admitted_fixture();
        for malformed in [
            "0,0,0,1,0,0,0,0,0,0",
            "2,0,0,1,0,0,0,0,0,0",
            "0,0,0,NaN,0,0,0,0,0,0",
            "0,0,0",
        ] {
            let mut lines = valid.clone();
            lines.push(malformed.to_owned());
            let mut solver = LbmSolver3D::new(2, 2, 2, 0.8);
            let mut mhd = MhdField::new(2, 2, 2, MhdConfig::default());
            let before = (
                solver.f.clone(),
                solver.rho.clone(),
                solver.u.clone(),
                mhd.bx.clone(),
                mhd.by.clone(),
                mhd.bz.clone(),
            );
            assert!(load_ic_lines(lines, &mut solver, &mut mhd).is_err());
            assert_eq!(
                before,
                (solver.f, solver.rho, solver.u, mhd.bx, mhd.by, mhd.bz)
            );
        }
        let mut solver = LbmSolver3D::new(2, 2, 2, 0.8);
        let mut mhd = MhdField::new(2, 2, 2, MhdConfig::default());
        let mut missing = valid.clone();
        missing.pop();
        assert!(load_ic_lines(missing, &mut solver, &mut mhd).is_err());
        assert_eq!(load_ic_lines(valid, &mut solver, &mut mhd).unwrap().0, 8);
        assert!(solver.rho.iter().all(|density| *density == 1.0));
    }

    #[test]
    fn density_weighting_and_undefined_ratio_are_explicit() {
        let mut field = DmForceField::new(2, 1, 1, DmForceConfig::default());
        field.force = vec![[0.01, 0.0, 0.0]; 2];
        assert_eq!(
            gravitational_force_density(&field, &[2.0, 3.0]).unwrap(),
            vec![[0.02, 0.0, 0.0], [0.03, 0.0, 0.0]]
        );
        assert!(gravitational_force_density(&field, &[1.0]).is_err());
        assert!(gravitational_force_density(&field, &[1.0, f64::NAN]).is_err());
        assert_eq!(ratio_of_maxima(2.0, 4.0).unwrap(), Some(0.5));
        assert_eq!(ratio_of_maxima(2.0, 0.0).unwrap(), None);
        assert_eq!(ratio_of_maxima(0.0, 0.0).unwrap(), None);
        assert!(ratio_of_maxima(f64::NAN, 1.0).is_err());
    }
}
