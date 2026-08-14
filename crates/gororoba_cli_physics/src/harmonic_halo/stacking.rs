//! Harmonic halo stacking analysis binary.
//!
//! Loads SPARC rotation curves + NFW fits, normalizes to x = r/r_s,
//! computes fractional residuals, stacks, and Fourier-analyzes for
//! the 7 predicted box-kite wavenumbers.
//!
//! Usage:
//!   harmonic-halo-stacking --sparc-dir data/external/sparc/Rotmod_LTG \
//!                          --nfw-fits data/external/sparc_nfw/table2.dat \
//!                          --csv output.csv
//!
//! Reference: C-1332 (stacking falsification thesis), E-180

use clap::Args;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        detection_threshold, stack_residuals,
    },
    nfw_utils::nfw_enclosed_mass_from_params,
};
use data_core::catalogs::sparc::{
    load_sparc_rotmod_dir, parse_sparc_nfw_fits, parse_sparc_vizier_table2,
};
use std::path::PathBuf;

/// Gravitational constant in kpc^3 Msun^{-1} (km/s)^2 units.
const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Args)]
pub struct Cli {
    /// Path to SPARC rotation curves (directory of .dat files OR single VizieR table2.dat).
    #[arg(long, default_value = "data/external/sparc/Rotmod_LTG")]
    sparc: PathBuf,

    /// Path to Li et al. (2020) NFW fit table (table1.dat from VizieR J/ApJS/247/31).
    #[arg(long, default_value = "data/external/sparc_nfw/table1.dat")]
    nfw_fits: PathBuf,

    /// Output CSV path for stacked residual profile.
    #[arg(long, default_value = "harmonic_halo_stacking.csv")]
    csv: PathBuf,

    /// Minimum x = r/r_s for analysis grid.
    #[arg(long, default_value_t = 0.1)]
    x_min: f64,

    /// Maximum x = r/r_s for analysis grid.
    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Number of grid points.
    #[arg(long, default_value_t = 200)]
    n_grid: usize,

    /// Minimum galaxies per x-bin.
    #[arg(long, default_value_t = 5)]
    min_per_bin: usize,

    /// Maximum reduced chi-squared for NFW fit quality cut.
    #[arg(long, default_value_t = 5.0)]
    max_chi2: f64,

    /// Minimum NFW concentration c200 (filters degenerate flat fits).
    #[arg(long, default_value_t = 2.0)]
    min_c200: f64,

    /// Cayley-Dickson dimension (16=sedenion, 32=pathion, 64=chingon, 128=routon, 256=voudion).
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {

    // Load SPARC rotation curves: auto-detect directory vs single VizieR table2 file
    eprintln!(
        "Loading SPARC rotation curves from {}...",
        cli.sparc.display()
    );
    let galaxies = if cli.sparc.is_dir() {
        load_sparc_rotmod_dir(&cli.sparc).map_err(|e| anyhow::anyhow!(e))?
    } else {
        parse_sparc_vizier_table2(&cli.sparc).map_err(|e| anyhow::anyhow!(e))?
    };
    eprintln!("  Loaded {} galaxies", galaxies.len());

    // Load NFW fits
    eprintln!("Loading NFW fits from {}...", cli.nfw_fits.display());
    let nfw_fits = parse_sparc_nfw_fits(&cli.nfw_fits).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("  Loaded {} NFW fits", nfw_fits.len());

    // Build lookup: galaxy name -> NFW fit
    let nfw_map: std::collections::HashMap<&str, &data_core::catalogs::sparc::SparcNfwFit> =
        nfw_fits.iter().map(|f| (f.name.as_str(), f)).collect();

    // Normalize each galaxy
    let mut normalized = Vec::new();
    let mut skipped = 0_usize;

    for galaxy in &galaxies {
        let Some(nfw) = nfw_map.get(galaxy.name.as_str()) else {
            skipped += 1;
            continue;
        };
        if nfw.r_s_kpc <= 0.0 || nfw.c200 <= 0.0 {
            skipped += 1;
            continue;
        }
        // Quality cuts: chi2 and concentration
        if nfw.chi2_red > cli.max_chi2 || nfw.c200 < cli.min_c200 {
            skipped += 1;
            continue;
        }

        // Build NfwParams from Li et al. published fit parameters
        let nfw_params = cosmology_core::nfw_utils::NfwParams {
            r_s_kpc: nfw.r_s_kpc,
            rho_s_solar_per_kpc3: nfw.rho_s_solar_per_kpc3(),
            c200: nfw.c200,
            r200_kpc: nfw.r_s_kpc * nfw.c200,
            m200_solar: nfw.m200_solar(),
            inner_gradient_warning: false,
        };

        let mut points = Vec::new();
        for pt in &galaxy.points {
            let x = pt.r_kpc / nfw.r_s_kpc;
            if !(0.01..=20.0).contains(&x) {
                continue;
            }
            let m_enc = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw_params);
            let v_nfw = (G_KPC_KMS2 * m_enc / pt.r_kpc).max(0.0).sqrt();
            if v_nfw < 1.0 {
                continue; // Avoid division by near-zero
            }
            let delta_v = (pt.v_obs_km_s - v_nfw) / v_nfw;
            let delta_v_err = pt.v_err_km_s / v_nfw;
            points.push(NormalizedPoint {
                x,
                delta_v,
                delta_v_err,
            });
        }

        if points.len() >= 5 {
            normalized.push(NormalizedResiduals {
                name: galaxy.name.clone(),
                r_s_kpc: nfw.r_s_kpc,
                points,
            });
        } else {
            skipped += 1;
        }
    }

    eprintln!(
        "Normalized {} galaxies ({} skipped: no NFW fit or too few points)",
        normalized.len(),
        skipped
    );

    if normalized.is_empty() {
        anyhow::bail!("No galaxies available for stacking");
    }

    // Stack with CD dimension parameters
    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let config = StackingConfig {
        x_min: cli.x_min,
        x_max: cli.x_max,
        n_grid: cli.n_grid,
        min_galaxies_per_bin: cli.min_per_bin,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: false,
    };

    eprintln!(
        "CD dimension: D={} ({} modes, af={:.3})",
        cd_params.cd_dim, cd_params.n_modes, cd_params.assessor_fraction
    );
    eprintln!(
        "Stacking {} galaxies on {}-point grid [{}, {}]...",
        normalized.len(),
        cli.n_grid,
        cli.x_min,
        cli.x_max
    );
    let result = stack_residuals(&normalized, &config);

    // Print summary
    let k = cd_params.predicted_wavenumbers();
    eprintln!();
    eprintln!("=== Stacking Result (D={}) ===", result.cd_dim);
    eprintln!("Galaxies stacked: {}", result.n_galaxies);
    eprintln!("Modes analyzed: {}", result.n_modes);
    eprintln!("RMS residual: {:.6}", result.rms_residual);
    eprintln!("Detection SNR: {:.2}", result.detection_snr);
    eprintln!("alpha_zd estimate: {:.6}", result.alpha_zd_estimate);
    eprintln!();
    eprintln!("Fourier power at predicted wavenumbers:");
    for (n, (&k_n, (&pwr, &ph))) in k
        .iter()
        .zip(result.fourier_power.iter().zip(result.fourier_phase.iter()))
        .enumerate()
    {
        eprintln!(
            "  mode {} (k={:.4}): power={:.2e}, phase={:.3} rad",
            n + 1,
            k_n,
            pwr,
            ph
        );
    }

    // Detection thresholds
    let thr = detection_threshold(normalized.len(), 0.05);
    eprintln!();
    eprintln!("Detection threshold (5% v_err): alpha_zd >= {:.4}", thr);

    // Write CSV
    let mut wtr = csv::Writer::from_path(&cli.csv)?;
    wtr.write_record(["x", "delta_stack", "delta_stack_err", "n_contributing"])?;
    for i in 0..result.x_grid.len() {
        wtr.write_record(&[
            format!("{:.6}", result.x_grid[i]),
            format!("{:.8e}", result.delta_stack[i]),
            format!("{:.8e}", result.delta_stack_err[i]),
            format!("{}", result.n_contributing[i]),
        ])?;
    }
    wtr.flush()?;
    eprintln!();
    eprintln!("Stacked profile written to {}", cli.csv.display());

    Ok(())
}
