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
use cr_transport::{
    diffusion::DiffusionConfig,
    grid::RigidityGrid,
    modulation::ForceFieldProxy,
    snapshot::write_snapshot_csv,
    solver::PteSolver,
    source::{DmChannel, DmSource, DmSpecies, DmYieldSpectrum, M_PROTON_GEV},
};
use csv::Reader;
use lbm_3d::{
    mhd::{MhdConfig, MhdField},
    solver::LbmSolver3D,
};
use serde::Deserialize;
use std::{
    fmt::Write as FmtWrite,
    fs,
    io::{Read, Write as IoWrite},
    path::{Path, PathBuf},
};
use zip::ZipArchive;

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

    /// DM yield source kind: csv or pppc.
    #[arg(long, default_value = "csv")]
    dm_yield_source: String,

    /// Yield table file. CSV expects kinetic_energy_gev,dndlog10k.
    /// PPPC expects a .dat or .zip at-production table.
    #[arg(long)]
    dm_yield_file: Option<PathBuf>,

    /// Secondary-particle species represented by the yield table.
    #[arg(long, default_value = "proton")]
    dm_species: String,

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

    /// Optional CSV output for the modeled spectrum at the final step.
    #[arg(long)]
    model_flux_out: Option<PathBuf>,

    /// Optional radial profile CSV produced by solar-wind-ic.
    #[arg(long)]
    radial_profile: Option<PathBuf>,

    /// Profile mode: auto, static, or evolve.
    #[arg(long, default_value = "auto")]
    profile_mode: String,
}

#[derive(Clone, Debug, Deserialize)]
struct RadialProfileCsvRow {
    r_au: f64,
    density_cm3: f64,
    speed_kms: f64,
    temp_k: f64,
    br_nt: f64,
    bt_nt: f64,
    bn_nt: f64,
    b_mag_nt: f64,
}

/// Local interstellar spectrum (LIS) power law.
/// J_LIS(R) = J_0 * (R / 1 GV)^{-gamma}
/// Bess-Pamela proton LIS reference: J_0 = 2.1e4 [m^-2 s^-1 sr^-1 GV^-1], gamma = 2.75.
fn lis_proton(r_gv: f64) -> f64 {
    let j0 = 2.1e4_f64;
    let gamma = 2.75_f64;
    j0 * r_gv.powf(-gamma)
}

fn load_radial_profile_csv(path: &PathBuf) -> Vec<RadialProfileCsvRow> {
    let mut reader = Reader::from_path(path).expect("failed to open radial profile CSV");
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        let row: RadialProfileCsvRow = row.expect("failed to parse radial profile row");
        rows.push(row);
    }
    rows.sort_by(|a, b| {
        a.r_au
            .partial_cmp(&b.r_au)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    assert!(!rows.is_empty(), "radial profile CSV must not be empty");
    rows
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

fn interp_profile(rows: &[RadialProfileCsvRow], r_au: f64) -> RadialProfileCsvRow {
    if rows.len() == 1 {
        return rows[0].clone();
    }
    if r_au <= rows[0].r_au {
        return rows[0].clone();
    }
    if r_au >= rows[rows.len() - 1].r_au {
        return rows[rows.len() - 1].clone();
    }
    let idx = rows
        .iter()
        .position(|row| row.r_au > r_au)
        .unwrap_or(rows.len() - 1)
        .saturating_sub(1);
    let left = &rows[idx];
    let right = &rows[idx + 1];
    let span = (right.r_au - left.r_au).max(1e-12);
    let t = (r_au - left.r_au) / span;
    RadialProfileCsvRow {
        r_au,
        density_cm3: lerp(left.density_cm3, right.density_cm3, t),
        speed_kms: lerp(left.speed_kms, right.speed_kms, t),
        temp_k: lerp(left.temp_k, right.temp_k, t),
        br_nt: lerp(left.br_nt, right.br_nt, t),
        bt_nt: lerp(left.bt_nt, right.bt_nt, t),
        bn_nt: lerp(left.bn_nt, right.bn_nt, t),
        b_mag_nt: lerp(left.b_mag_nt, right.b_mag_nt, t),
    }
}

fn static_fields_from_profile(
    rows: &[RadialProfileCsvRow],
    nx: usize,
    ny: usize,
    nz: usize,
    r_min_au: f64,
    r_max_au: f64,
) -> (Vec<[f64; 3]>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut u_sw = vec![[0.0_f64; 3]; nx * ny * nz];
    let mut bx = vec![0.0_f64; nx * ny * nz];
    let mut by = vec![0.0_f64; nx * ny * nz];
    let mut bz = vec![0.0_f64; nx * ny * nz];
    let dr = (r_max_au - r_min_au) / (nx as f64 - 1.0).max(1.0);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = z * nx * ny + y * nx + x;
                let r_au = r_min_au + x as f64 * dr;
                let row = interp_profile(rows, r_au);
                u_sw[idx] = [row.speed_kms * 0.05 / 400.0, 0.0, 0.0];
                bx[idx] = row.br_nt;
                by[idx] = row.bt_nt;
                bz[idx] = row.bn_nt;
            }
        }
    }
    (u_sw, bx, by, bz)
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

fn parse_dm_species(s: &str) -> DmSpecies {
    match s.to_lowercase().as_str() {
        "antiproton" | "pbar" => DmSpecies::Antiproton,
        "positron" | "e+" | "epos" => DmSpecies::Positron,
        _ => DmSpecies::Proton,
    }
}

fn load_text_maybe_zipped(path: &Path) -> String {
    if path.extension().is_some_and(|ext| ext == "zip") {
        let file = fs::File::open(path)
            .unwrap_or_else(|err| panic!("failed to open {}: {err}", path.display()));
        let mut archive = ZipArchive::new(file)
            .unwrap_or_else(|err| panic!("failed to read zip {}: {err}", path.display()));
        assert!(
            !archive.is_empty(),
            "zip archive {} must contain at least one file",
            path.display()
        );
        let mut zipped = archive
            .by_index(0)
            .unwrap_or_else(|err| panic!("failed to read zip member in {}: {err}", path.display()));
        let mut content = String::new();
        zipped
            .read_to_string(&mut content)
            .unwrap_or_else(|err| panic!("failed to read text from {}: {err}", path.display()));
        content
    } else {
        fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()))
    }
}

fn load_dm_yield_csv(path: &Path, species: DmSpecies) -> DmYieldSpectrum {
    #[derive(Deserialize)]
    struct YieldRow {
        kinetic_energy_gev: f64,
        dndlog10k: f64,
    }

    let mut reader = Reader::from_path(path)
        .unwrap_or_else(|err| panic!("failed to open DM yield CSV {}: {err}", path.display()));
    let mut energies = Vec::new();
    let mut values = Vec::new();
    for row in reader.deserialize() {
        let row: YieldRow = row.unwrap_or_else(|err| {
            panic!(
                "failed to parse DM yield row from {}: {err}",
                path.display()
            )
        });
        energies.push(row.kinetic_energy_gev);
        values.push(row.dndlog10k);
    }
    DmYieldSpectrum::new(species, energies, values)
        .unwrap_or_else(|err| panic!("invalid DM yield CSV spectrum in {}: {err}", path.display()))
}

fn pppc_channel_name(channel: DmChannel) -> &'static str {
    match channel {
        DmChannel::BbBar => "b",
        DmChannel::WW => "W",
        DmChannel::Tau => "\\[Tau]",
        DmChannel::PositronAntiproton => "V->e",
    }
}

fn load_dm_yield_pppc(
    path: &Path,
    species: DmSpecies,
    channel: DmChannel,
    mass_gev: f64,
) -> DmYieldSpectrum {
    assert!(
        species != DmSpecies::Proton,
        "PPPC at-production tables do not provide proton yields; supply --dm-yield-file with a proton table instead"
    );
    let content = load_text_maybe_zipped(path);
    let mut lines = content.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("PPPC table {} is empty", path.display()));
    let header_cols: Vec<&str> = header.split_whitespace().collect();
    let column_name = pppc_channel_name(channel);
    let value_idx = header_cols
        .iter()
        .position(|name| *name == column_name)
        .unwrap_or_else(|| {
            panic!(
                "PPPC table {} does not contain channel column {}",
                path.display(),
                column_name
            )
        });

    let mut rows = Vec::new();
    for line in lines {
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() <= value_idx {
            continue;
        }
        let mass = match cols[0].parse::<f64>() {
            Ok(value) => value,
            Err(_) => continue,
        };
        let logx = match cols[1].parse::<f64>() {
            Ok(value) => value,
            Err(_) => continue,
        };
        let value = match cols[value_idx].parse::<f64>() {
            Ok(value) => value,
            Err(_) => continue,
        };
        rows.push((mass, logx, value));
    }
    assert!(
        !rows.is_empty(),
        "PPPC table {} did not yield any usable rows",
        path.display()
    );

    let mut masses: Vec<f64> = rows.iter().map(|(mass, _, _)| *mass).collect();
    masses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    masses.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let lower_mass = masses
        .iter()
        .copied()
        .filter(|mass| *mass <= mass_gev)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(masses[0]);
    let upper_mass = masses
        .iter()
        .copied()
        .filter(|mass| *mass >= mass_gev)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(masses[masses.len() - 1]);

    let collect_mass_rows = |target_mass: f64| -> Vec<(f64, f64)> {
        let mut selected: Vec<(f64, f64)> = rows
            .iter()
            .filter(|(mass, _, _)| (*mass - target_mass).abs() < 1e-9)
            .map(|(_, logx, value)| (*logx, *value))
            .collect();
        selected.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        selected
    };

    let lower_rows = collect_mass_rows(lower_mass);
    let (logx_values, dndlog10k_values) = if (upper_mass - lower_mass).abs() < 1e-9 {
        (
            lower_rows.iter().map(|(logx, _)| *logx).collect::<Vec<_>>(),
            lower_rows
                .iter()
                .map(|(_, value)| *value)
                .collect::<Vec<_>>(),
        )
    } else {
        let upper_rows = collect_mass_rows(upper_mass);
        assert_eq!(
            lower_rows.len(),
            upper_rows.len(),
            "PPPC interpolation mass grids differ in {}",
            path.display()
        );
        let t = (mass_gev.ln() - lower_mass.ln()) / (upper_mass.ln() - lower_mass.ln());
        let mut logx_values = Vec::with_capacity(lower_rows.len());
        let mut values = Vec::with_capacity(lower_rows.len());
        for ((logx_l, value_l), (logx_u, value_u)) in lower_rows.iter().zip(&upper_rows) {
            assert!(
                (logx_l - logx_u).abs() < 1e-9,
                "PPPC interpolation grids differ in {}",
                path.display()
            );
            logx_values.push(*logx_l);
            values.push(value_l * (1.0 - t) + value_u * t);
        }
        (logx_values, values)
    };

    let energies = logx_values
        .iter()
        .map(|logx| 10_f64.powf(*logx) * mass_gev)
        .collect::<Vec<_>>();
    DmYieldSpectrum::new(species, energies, dndlog10k_values).unwrap_or_else(|err| {
        panic!(
            "invalid PPPC-derived spectrum from {}: {err}",
            path.display()
        )
    })
}

fn load_dm_yield_spectrum(cli: &Cli) -> DmYieldSpectrum {
    let species = parse_dm_species(&cli.dm_species);
    let path = cli.dm_yield_file.as_deref().unwrap_or_else(|| {
        panic!("DM source injection now requires --dm-yield-file with a tabulated spectrum")
    });
    match cli.dm_yield_source.to_lowercase().as_str() {
        "pppc" => load_dm_yield_pppc(
            path,
            species,
            parse_dm_channel(&cli.dm_channel),
            cli.dm_mass_gev,
        ),
        _ => load_dm_yield_csv(path, species),
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
                // Convert LBM lattice speed to AU/s:
                // u_phys_ms = u_lattice * (400 km/s / 0.05)
                // u_phys_au_s = u_phys_ms * 1e3 / 1.496e11
                u[0].abs() * 400.0 / 0.05 * 1e3 / 1.496e11
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

fn write_model_flux_csv(
    solver: &PteSolver,
    phi_map: &[f64],
    step: usize,
    r_min_au: f64,
    r_max_au: f64,
) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "step,r_au,x,y,z,rigidity_gv,kinetic_energy_gev,flux_j,modulation_phi"
    )
    .unwrap();
    let y_mid = solver.ny / 2;
    let z_mid = solver.nz / 2;
    let dr = (r_max_au - r_min_au) / (solver.nx as f64 - 1.0).max(1.0);

    for x in 0..solver.nx {
        let phi = phi_map.get(x).copied().unwrap_or(0.0);
        let r_au = r_min_au + x as f64 * dr;
        let flux = solver.flux_at(x, y_mid, z_mid);
        for (p_idx, &j) in flux.iter().enumerate() {
            let rigidity_gv = solver.grid.rigidity(p_idx);
            let kinetic_energy_gev = solver.grid.kinetic_energy_gev(p_idx, M_PROTON_GEV);
            writeln!(
                out,
                "{step},{r_au:.6e},{x},{y_mid},{z_mid},{rigidity_gv:.6e},{kinetic_energy_gev:.6e},{j:.6e},{phi:.6e}"
            )
            .unwrap();
        }
    }

    out
}

fn compute_phi_map_from_profile(
    rows: &[RadialProfileCsvRow],
    proxy: &ForceFieldProxy,
    diffusion: &DiffusionConfig,
    nx: usize,
    r_min_au: f64,
    r_max_au: f64,
) -> Vec<f64> {
    let dr = (r_max_au - r_min_au) / (nx as f64 - 1.0).max(1.0);
    let v_sw_of_r = |r: f64| interp_profile(rows, r).speed_kms;
    let b0 = rows
        .iter()
        .find(|row| row.b_mag_nt.is_finite() && row.b_mag_nt > 0.0)
        .map(|row| row.b_mag_nt)
        .unwrap_or(5.0);
    let b_of_r = |r: f64| interp_profile(rows, r).b_mag_nt.abs().max(0.01);
    let kappa_rr_of_r = move |r: f64| diffusion.kappa_0_au2_per_s * (b0 / b_of_r(r));
    (0..nx)
        .map(|x| {
            let r = r_min_au + x as f64 * dr;
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

    let profile_mode = cli.profile_mode.to_lowercase();
    let use_static_profile = match profile_mode.as_str() {
        "static" => true,
        "evolve" => false,
        "auto" => cli.radial_profile.is_some(),
        other => panic!("unknown profile mode '{other}': use auto, static, or evolve"),
    };
    assert!(
        !(use_static_profile && cli.radial_profile.is_none()),
        "static profile mode requires --radial-profile"
    );

    let radial_profile_rows = cli.radial_profile.as_ref().map(load_radial_profile_csv);
    let (mut lbm, mut mhd) = if use_static_profile {
        let mut lbm = LbmSolver3D::new(cli.nx, cli.ny, cli.nz, cli.tau);
        lbm.initialize_uniform(1.0, [0.0, 0.0, 0.0]);
        let mhd_cfg = MhdConfig {
            b0_nt: cli.b0,
            ..Default::default()
        };
        let mut mhd = MhdField::new(cli.nx, cli.ny, cli.nz, mhd_cfg);
        if let Some(ref rows) = radial_profile_rows {
            let (u_sw, bx, by, bz) = static_fields_from_profile(
                rows,
                cli.nx,
                cli.ny,
                cli.nz,
                cli.r_min_au,
                cli.r_max_au,
            );
            lbm.u = u_sw;
            mhd.bx = bx;
            mhd.by = by;
            mhd.bz = bz;
        }
        (lbm, mhd)
    } else {
        let mut lbm = LbmSolver3D::new(cli.nx, cli.ny, cli.nz, cli.tau);
        lbm.initialize_uniform(1.0, [cli.v_sw, 0.0, 0.0]);

        let mhd_cfg = MhdConfig {
            b0_nt: cli.b0,
            ..Default::default()
        };
        let mut mhd = MhdField::new(cli.nx, cli.ny, cli.nz, mhd_cfg);
        mhd.parker_spiral_init(cli.v_sw);
        (lbm, mhd)
    };

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
    let dx_au = (cli.r_max_au - cli.r_min_au) / (cli.nx as f64 - 1.0).max(1.0);
    let v_sw_ms = 400.0e3_f64; // m/s
    let au_m = 1.496e11_f64;
    let dt_s = dx_au * au_m / v_sw_ms;

    let mut pte = PteSolver::new(cli.nx, cli.ny, cli.nz, grid, diff_cfg.clone(), dt_s, dx_au);
    pte.r_min_au = cli.r_min_au;
    pte.set_boundary_ism(&lis_proton);

    // --- Force-field proxy ---
    let proxy = ForceFieldProxy::new(cli.r_max_au, 0.938272);

    // --- DM source (optional) ---
    let dm_source = if cli.dm_mass_gev > 0.0 {
        let n_cells = cli.nx * cli.ny * cli.nz;
        let spatial_profile = vec![1.0_f64; n_cells];
        Some(DmSource {
            mass_gev: cli.dm_mass_gev,
            channel: parse_dm_channel(&cli.dm_channel),
            sigma_v_cm3_per_s: cli.dm_sigma_v,
            spatial_profile,
            yield_spectrum: load_dm_yield_spectrum(&cli),
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
    println!(
        "  profile mode: {}",
        if use_static_profile {
            "static"
        } else {
            "evolve"
        }
    );
    if cli.dm_mass_gev > 0.0 {
        println!(
            "  DM: mass={:.1} GeV channel={} sigma_v={:.2e} cm^3/s source={} species={}",
            cli.dm_mass_gev, cli.dm_channel, cli.dm_sigma_v, cli.dm_yield_source, cli.dm_species
        );
    }

    // --- Main loop ---
    for step in 0..cli.steps {
        if !use_static_profile {
            if step > 0 {
                lbm.phase2_streaming().expect("phase2_streaming failed");
            }
            lbm.compute_macroscopic();
            let lorentz = mhd.lorentz_force();
            lbm.set_force_field(lorentz)
                .expect("set_force_field failed");
            lbm.phase1_collision().expect("phase1_collision failed");
            mhd.evolve_b_field(&lbm.u);
        }

        // Convert LBM lattice velocities to physical AU/s for PTE.
        // u_phys = u_lattice * (v_sw_ms / u_lbm_ref) / AU_m
        let u_phys: Vec<[f64; 3]> = lbm
            .u
            .iter()
            .map(|u| {
                let scale = v_sw_ms / (cli.v_sw * au_m);
                [u[0] * scale, u[1] * scale, u[2] * scale]
            })
            .collect();

        // PTE one step
        let bx = &mhd.bx;
        let by = &mhd.by;
        let bz = &mhd.bz;
        pte.evolve_one_step(
            &u_phys,
            (bx, by, bz),
            dm_source.as_ref(),
            &dm_density,
            Some(&lis_proton),
        );

        // Snapshot output
        if step % cli.snapshot_interval == 0 {
            let phi_map = if let Some(ref rows) = radial_profile_rows {
                if use_static_profile {
                    compute_phi_map_from_profile(
                        rows,
                        &proxy,
                        &diff_cfg,
                        cli.nx,
                        cli.r_min_au,
                        cli.r_max_au,
                    )
                } else {
                    compute_phi_map(&lbm, &proxy, &diff_cfg, cli.r_min_au, cli.r_max_au)
                }
            } else {
                compute_phi_map(&lbm, &proxy, &diff_cfg, cli.r_min_au, cli.r_max_au)
            };
            let csv = write_snapshot_csv(&pte, &phi_map, step);
            let fname = cli.out_dir.join(format!("cr_snapshot_{step:06}.csv"));
            fs::write(&fname, csv).expect("failed to write snapshot");
            println!("  step {step}: wrote {}", fname.display());
        }
    }

    // Final snapshot
    let phi_map = if let Some(ref rows) = radial_profile_rows {
        if use_static_profile {
            compute_phi_map_from_profile(
                rows,
                &proxy,
                &diff_cfg,
                cli.nx,
                cli.r_min_au,
                cli.r_max_au,
            )
        } else {
            compute_phi_map(&lbm, &proxy, &diff_cfg, cli.r_min_au, cli.r_max_au)
        }
    } else {
        compute_phi_map(&lbm, &proxy, &diff_cfg, cli.r_min_au, cli.r_max_au)
    };
    let csv = write_snapshot_csv(&pte, &phi_map, cli.steps);
    let fname = cli
        .out_dir
        .join(format!("cr_snapshot_{:06}.csv", cli.steps));
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

    let model_flux_path = cli
        .model_flux_out
        .clone()
        .unwrap_or_else(|| cli.out_dir.join("cr_model_flux.csv"));
    let model_flux_csv =
        write_model_flux_csv(&pte, &phi_map, cli.steps, cli.r_min_au, cli.r_max_au);
    fs::write(&model_flux_path, model_flux_csv).expect("failed to write modeled flux CSV");
    println!("Modeled flux written to {}", model_flux_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_rows() -> Vec<RadialProfileCsvRow> {
        vec![
            RadialProfileCsvRow {
                r_au: 1.0,
                density_cm3: 5.0,
                speed_kms: 400.0,
                temp_k: 1.0e5,
                br_nt: 5.0,
                bt_nt: -3.0,
                bn_nt: 0.0,
                b_mag_nt: 5.83,
            },
            RadialProfileCsvRow {
                r_au: 50.0,
                density_cm3: 0.002,
                speed_kms: 410.0,
                temp_k: 8.0e3,
                br_nt: 0.002,
                bt_nt: -0.06,
                bn_nt: 0.0,
                b_mag_nt: 0.06003,
            },
            RadialProfileCsvRow {
                r_au: 100.0,
                density_cm3: 0.0005,
                speed_kms: 420.0,
                temp_k: 4.0e3,
                br_nt: 0.0005,
                bt_nt: -0.03,
                bn_nt: 0.0,
                b_mag_nt: 0.03000,
            },
        ]
    }

    #[test]
    fn test_interp_profile_monotone_speed() {
        let row = interp_profile(&sample_rows(), 75.0);
        assert!(row.speed_kms > 410.0 && row.speed_kms < 420.0);
        assert!((row.r_au - 75.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_phi_map_from_profile_decreases_with_radius() {
        let proxy = ForceFieldProxy::new(157.0, 0.938272);
        let diffusion = DiffusionConfig {
            kappa_0_au2_per_s: 4.5e-5,
            r_ref_gv: 1.0,
            alpha: 0.5,
            epsilon_perp: 0.02,
            solar_epoch_a: 1.0,
        };
        let phi = compute_phi_map_from_profile(&sample_rows(), &proxy, &diffusion, 8, 1.0, 100.0);
        assert!(phi.first().unwrap() > phi.last().unwrap());
    }

    fn unique_temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!("gororoba_{name}_{nanos}"))
    }

    #[test]
    fn test_load_dm_yield_csv() {
        let path = unique_temp_path("dm_yield.csv");
        fs::write(
            &path,
            "kinetic_energy_gev,dndlog10k\n0.1,0.2\n1.0,1.1\n10.0,0.3\n",
        )
        .expect("write csv fixture");
        let spectrum = load_dm_yield_csv(&path, DmSpecies::Proton);
        assert_eq!(spectrum.kinetic_energy_gev.len(), 3);
        assert!((spectrum.dndlog10k[1] - 1.1).abs() < 1e-12);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_load_dm_yield_pppc_interpolates_mass() {
        let path = unique_temp_path("pppc.dat");
        fs::write(
            &path,
            "mDM Log[10,x] b W \\\\[Tau] V->e\n\
10 -2.0 1.0 2.0 3.0 4.0\n\
10 -1.0 2.0 4.0 6.0 8.0\n\
100 -2.0 3.0 6.0 9.0 12.0\n\
100 -1.0 5.0 10.0 15.0 20.0\n",
        )
        .expect("write pppc fixture");
        let spectrum =
            load_dm_yield_pppc(&path, DmSpecies::Antiproton, DmChannel::BbBar, 31.6227766);
        assert_eq!(spectrum.kinetic_energy_gev.len(), 2);
        assert!((spectrum.kinetic_energy_gev[0] - 0.316227766).abs() < 1e-6);
        assert!((spectrum.dndlog10k[0] - 2.0).abs() < 1e-6);
        assert!((spectrum.dndlog10k[1] - 3.5).abs() < 1e-6);
        let _ = fs::remove_file(path);
    }
}
