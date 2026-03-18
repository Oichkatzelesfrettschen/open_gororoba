//! Galaxy-ensemble Rayleigh phase coherence test for MaNGA rotation curves.
//!
//! Computes per-galaxy DFT phases at CD-ZD wavenumbers across the N=6992
//! galaxy ensemble and applies the Rayleigh R statistic to test for coherent
//! phases. Fixes the N=19 statistical limitation of the x-bin jackknife (E-192).
//!
//! Under H0 (incoherent phases): R ~ 1/sqrt(N_gal) ~ 0.012, p > 0.9.
//! The executed MaNGA run yields modest but nonzero coherence
//! R = 0.033..0.141 across the 7 modes, rising monotonically with k.
//! This is inconsistent with pure random phases, but also not a selective
//! low-mode detection signature.
//!
//! Reference: C-1406, E-194

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, galaxy_ensemble_phase_coherence,
        predicted_wavenumbers_cd,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use std::path::PathBuf;

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "harmonic-halo-galaxy-phase")]
#[command(about = "Galaxy-ensemble Rayleigh phase coherence test at CD-ZD wavenumbers")]
struct Cli {
    /// Path to MaNGA rotation curves CSV.
    #[arg(long, default_value = "data/external/manga/manga_rotcurves_all.csv")]
    rotcurves: PathBuf,

    /// Path to DAPall selection CSV.
    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    dapall: PathBuf,

    /// CD dimension for wavenumber prediction.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Minimum rotation curve points per galaxy.
    #[arg(long, default_value_t = 5)]
    min_points: usize,

    /// Output CSV path (k, rayleigh_r, rayleigh_p, n_gal).
    #[arg(long, default_value = "galaxy_phase_coherence.csv")]
    rayleigh_csv: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading MaNGA rotation curves from {:?}...", cli.rotcurves);
    let rotcurves = parse_manga_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("Loaded {} galaxy rotation curves", rotcurves.len());

    eprintln!("Loading DAPall metadata from {:?}...", cli.dapall);
    let dapall = parse_manga_dapall_csv(&cli.dapall).map_err(|e| anyhow::anyhow!(e))?;

    // Build lookup map
    let dapall_map: std::collections::HashMap<String, _> = dapall
        .into_iter()
        .map(|e| (e.plateifu.clone(), e))
        .collect();

    // Normalize each galaxy to x = r/r_s and fractional residuals
    let mut normalized: Vec<NormalizedResiduals> = Vec::new();
    let mut skipped = 0usize;

    for galaxy in &rotcurves {
        let meta = match dapall_map.get(&galaxy.plateifu) {
            Some(m) => m,
            None => {
                skipped += 1;
                continue;
            }
        };

        let m200 = 10.0_f64.powf(meta.estimated_log_m200());
        let z = meta.z;
        let nfw = nfw_params_from_mass(m200, z);
        let r_s = nfw.r_s_kpc;

        let points: Vec<NormalizedPoint> = galaxy
            .points
            .iter()
            .filter_map(|pt| {
                if pt.r_kpc <= 0.0 || r_s <= 0.0 {
                    return None;
                }
                let x = pt.r_kpc / r_s;
                let m_enc = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw);
                let v_sq = G_KPC_KMS2 * m_enc / pt.r_kpc;
                let v_nfw = v_sq.max(0.0).sqrt();
                if v_nfw < 1.0 {
                    return None;
                }
                let delta_v = (pt.v_obs_km_s - v_nfw) / v_nfw;
                let delta_v_err = pt.v_err_km_s / v_nfw;
                if !delta_v.is_finite() || !delta_v_err.is_finite() {
                    return None;
                }
                Some(NormalizedPoint {
                    x,
                    delta_v,
                    delta_v_err,
                })
            })
            .collect();

        if points.len() >= cli.min_points {
            normalized.push(NormalizedResiduals {
                name: galaxy.plateifu.clone(),
                r_s_kpc: r_s,
                points,
            });
        } else {
            skipped += 1;
        }
    }

    eprintln!(
        "Normalized {} galaxies ({} skipped: <{} points or missing metadata)",
        normalized.len(),
        skipped,
        cli.min_points
    );

    // Compute ensemble Rayleigh phase coherence
    let wavenumbers = predicted_wavenumbers_cd(cli.cd_dim);
    let n_modes = CdDimensionParams::new(cli.cd_dim).n_modes;
    eprintln!(
        "CD dimension D={}: {} modes, wavenumbers k_1..k_{}",
        cli.cd_dim, n_modes, n_modes
    );

    let result = galaxy_ensemble_phase_coherence(&normalized, &wavenumbers, cli.min_points);

    eprintln!(
        "Galaxy-ensemble Rayleigh test: N_gal = {}",
        result.n_jackknife
    );
    eprintln!(
        "  Expected R under H0: {:.4}",
        1.0 / (result.n_jackknife as f64).sqrt()
    );

    // Report results
    for (i, (&k, (&r, &p))) in wavenumbers
        .iter()
        .zip(result.rayleigh_r.iter().zip(result.rayleigh_p.iter()))
        .enumerate()
    {
        eprintln!("  mode {}: k={:.4}, R={:.4}, p={:.4}", i + 1, k, r, p);
    }

    // Write CSV
    let mut wtr = csv::Writer::from_path(&cli.rayleigh_csv)?;
    wtr.write_record(["mode", "k", "rayleigh_r", "rayleigh_p", "n_gal"])?;
    for (i, (&k, (&r, &p))) in wavenumbers
        .iter()
        .zip(result.rayleigh_r.iter().zip(result.rayleigh_p.iter()))
        .enumerate()
    {
        wtr.write_record(&[
            (i + 1).to_string(),
            format!("{k:.6}"),
            format!("{r:.6}"),
            format!("{p:.6}"),
            result.n_jackknife.to_string(),
        ])?;
    }
    wtr.flush()?;
    eprintln!("Rayleigh CSV written to {:?}", cli.rayleigh_csv);

    Ok(())
}
