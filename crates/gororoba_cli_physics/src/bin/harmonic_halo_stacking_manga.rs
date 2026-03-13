//! Harmonic halo stacking for MaNGA rotation curves.
//!
//! Loads MaNGA pseudo-slit rotation curves + DAPall metadata, uses
//! Moster+2013 SMHM relation for NFW priors, normalizes to x = r/r_s,
//! and stacks for harmonic halo detection.
//!
//! This is the N~2500 successor to the N=93 SPARC stacking (E-180).
//! With sqrt(N) ~ 50, the noise floor drops from ~1.6% to ~0.3%,
//! bringing sub-percent alpha_zd into reach.
//!
//! Usage:
//!   harmonic-halo-stacking-manga --rotcurves data/external/manga/rotcurves/manga_rotcurves_all.csv \
//!       --dapall data/external/manga/dapall_selection.csv --cd-dim 16 --csv manga_stacking.csv
//!
//! Reference: E-183 (MaNGA harmonic stacking)

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        detection_threshold, stack_residuals,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use std::path::PathBuf;

/// Gravitational constant in kpc^3 Msun^{-1} (km/s)^2 units.
const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "harmonic-halo-stacking-manga")]
#[command(about = "Stack MaNGA rotation curve residuals for harmonic halo detection")]
struct Cli {
    /// Path to MaNGA rotation curves CSV (from manga_maps_to_rotcurves.py).
    #[arg(
        long,
        default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv"
    )]
    rotcurves: PathBuf,

    /// Path to DAPall selection CSV (from manga_dapall_to_csv.py).
    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    dapall: PathBuf,

    /// Output CSV path.
    #[arg(long, default_value = "manga_harmonic_stacking.csv")]
    csv: PathBuf,

    /// Minimum x = r/r_s for analysis grid.
    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    /// Maximum x = r/r_s for analysis grid.
    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Number of grid points.
    #[arg(long, default_value_t = 200)]
    n_grid: usize,

    /// Minimum galaxies per x-bin.
    #[arg(long, default_value_t = 10)]
    min_per_bin: usize,

    /// Cayley-Dickson dimension.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Minimum number of rotation curve points per galaxy.
    #[arg(long, default_value_t = 8)]
    min_points: usize,

    /// Optional: path to Euclid-MaNGA cross-match CSV from euclid_manga_crossmatch.py.
    ///
    /// When provided, only galaxies present in this CSV (which already has
    /// morphology cuts applied) are included in the stack. Produces a
    /// morphologically-confirmed sub-sample. With Euclid Q1 this will be
    /// very few galaxies; use with Euclid DR2+ for statistical power.
    #[arg(long)]
    euclid_morphology_csv: Option<PathBuf>,

    /// Exclude beam-smearing-affected inner bins (psf_flag = true) from stacking.
    ///
    /// Requires rotation curves produced by manga-maps-extractor, which writes
    /// a psf_flag column. The flag is set for bins within 1 MaNGA PSF FWHM (~2.5 arcsec)
    /// of the galaxy center. Reduces the +29% inner-halo projection spike (E-196).
    #[arg(long, default_value_t = false)]
    exclude_psf_flagged: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    // Load rotation curves
    eprintln!(
        "Loading MaNGA rotation curves from {}...",
        cli.rotcurves.display()
    );
    let galaxies = parse_manga_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("  Loaded {} galaxies", galaxies.len());

    // Load DAPall metadata
    eprintln!("Loading DAPall metadata from {}...", cli.dapall.display());
    let dapall = parse_manga_dapall_csv(&cli.dapall).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("  Loaded {} entries", dapall.len());

    // Build lookup: plateifu -> DAPall entry
    let dapall_map: std::collections::HashMap<&str, &data_core::catalogs::manga::MangaDapallEntry> =
        dapall.iter().map(|e| (e.plateifu.as_str(), e)).collect();

    // Optional Euclid morphology filter
    let morphology_allowed: Option<std::collections::HashSet<String>> =
        if let Some(ref csv_path) = cli.euclid_morphology_csv {
            let mut allowed = std::collections::HashSet::new();
            let mut rdr = csv::Reader::from_path(csv_path)
                .map_err(|e| anyhow::anyhow!("Cannot open {}: {}", csv_path.display(), e))?;
            let headers = rdr.headers()?.clone();
            let plateifu_idx = headers
                .iter()
                .position(|h| h == "plateifu")
                .ok_or_else(|| anyhow::anyhow!("No 'plateifu' column in {}", csv_path.display()))?;
            for result in rdr.records() {
                let record = result?;
                if let Some(id) = record.get(plateifu_idx) {
                    allowed.insert(id.to_string());
                }
            }
            eprintln!(
                "Euclid morphology filter: {} confirmed galaxies from {}",
                allowed.len(),
                csv_path.display()
            );
            Some(allowed)
        } else {
            None
        };

    // Normalize each galaxy using SMHM-derived NFW parameters
    let mut normalized = Vec::new();
    let mut skipped_no_meta = 0_usize;
    let mut skipped_smhm = 0_usize;
    let mut skipped_few_pts = 0_usize;
    let mut skipped_no_morph = 0_usize;

    for galaxy in &galaxies {
        // Euclid morphology filter (skip if not in confirmed set)
        if let Some(ref allowed) = morphology_allowed
            && !allowed.contains(galaxy.plateifu.as_str())
        {
            skipped_no_morph += 1;
            continue;
        }

        let Some(meta) = dapall_map.get(galaxy.plateifu.as_str()) else {
            skipped_no_meta += 1;
            continue;
        };

        // SMHM halo mass estimate
        let log_m200 = meta.estimated_log_m200();
        if !log_m200.is_finite() || log_m200 < 10.0 {
            skipped_smhm += 1;
            continue;
        }
        let m200 = 10.0_f64.powf(log_m200);

        // NFW parameters from halo mass at galaxy redshift
        let nfw = nfw_params_from_mass(m200, meta.z);
        if nfw.r_s_kpc <= 0.0 || nfw.c200 <= 0.0 {
            skipped_smhm += 1;
            continue;
        }

        // Normalize rotation curve
        let mut points = Vec::new();
        for pt in &galaxy.points {
            // Skip beam-smearing-affected inner bins when requested (E-196).
            if cli.exclude_psf_flagged && pt.psf_flag {
                continue;
            }
            let x = pt.r_kpc / nfw.r_s_kpc;
            if !(0.01..=20.0).contains(&x) {
                continue;
            }
            let m_enc = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw);
            let v_nfw = (G_KPC_KMS2 * m_enc / pt.r_kpc).max(0.0).sqrt();
            if v_nfw < 1.0 {
                continue;
            }
            let delta_v = (pt.v_obs_km_s - v_nfw) / v_nfw;
            let delta_v_err = pt.v_err_km_s / v_nfw;
            points.push(NormalizedPoint {
                x,
                delta_v,
                delta_v_err,
            });
        }

        if points.len() >= cli.min_points {
            normalized.push(NormalizedResiduals {
                name: galaxy.plateifu.clone(),
                r_s_kpc: nfw.r_s_kpc,
                points,
            });
        } else {
            skipped_few_pts += 1;
        }
    }

    eprintln!(
        "Normalized {} galaxies (skipped: {} no metadata, {} bad SMHM, {} too few points, {} no morphology)",
        normalized.len(),
        skipped_no_meta,
        skipped_smhm,
        skipped_few_pts,
        skipped_no_morph,
    );

    if normalized.is_empty() {
        anyhow::bail!("No galaxies available for stacking");
    }

    // Stack
    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let config = StackingConfig {
        x_min: cli.x_min,
        x_max: cli.x_max,
        n_grid: cli.n_grid,
        min_galaxies_per_bin: cli.min_per_bin,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: cli.exclude_psf_flagged,
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
    eprintln!("=== MaNGA Stacking Result (D={}) ===", result.cd_dim);
    eprintln!("Galaxies stacked: {}", result.n_galaxies);
    eprintln!("Modes analyzed: {}", result.n_modes);
    eprintln!("RMS residual: {:.6}", result.rms_residual);
    eprintln!("Detection SNR: {:.2}", result.detection_snr);
    eprintln!("alpha_zd estimate: {:.6}", result.alpha_zd_estimate);
    eprintln!();

    let n_print = k.len().min(15);
    eprintln!("Fourier power at first {} predicted wavenumbers:", n_print);
    for (n, (&k_n, (&pwr, &ph))) in k
        .iter()
        .zip(result.fourier_power.iter().zip(result.fourier_phase.iter()))
        .enumerate()
        .take(n_print)
    {
        eprintln!(
            "  mode {} (k={:.4}): power={:.2e}, phase={:.3} rad",
            n + 1,
            k_n,
            pwr,
            ph
        );
    }

    let thr = detection_threshold(normalized.len(), 0.05);
    eprintln!();
    eprintln!("Detection threshold (5% v_err): alpha_zd >= {:.6}", thr);
    eprintln!(
        "Improvement over SPARC (N=93): {:.1}x",
        (93.0_f64 / normalized.len() as f64).sqrt()
    );

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
