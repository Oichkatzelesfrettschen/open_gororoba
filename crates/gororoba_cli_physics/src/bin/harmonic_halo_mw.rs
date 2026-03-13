//! Harmonic halo analysis for the Milky Way rotation curve.
//!
//! Loads the Gaia DR3 MW rotation curve (Jiao et al. 2023), fits an NFW
//! model, computes residuals, and searches for the 7-mode harmonic
//! signature in the single-galaxy residual profile.
//!
//! Usage:
//!   harmonic-halo-mw --gaia data/external/gaia_mw_rotation/jiao2023_table1.dat \
//!                    --m200 1e12 --c200 12.0 --csv mw_halos.csv
//!
//! Reference: C-1332, C-1313, E-180

use clap::Parser;
use cosmology_core::{
    harmonic_halos::{HarmonicHaloConfig, v_circ_nfw, v_circ_with_halos},
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig, stack_residuals,
    },
    nfw_utils::nfw_params_from_mass,
};
use data_core::catalogs::gaia_mw_rotation::parse_mw_rotation_curve;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "harmonic-halo-mw")]
#[command(about = "Analyze Milky Way rotation curve for harmonic halo modulation")]
struct Cli {
    /// Path to Gaia MW rotation curve table.
    #[arg(
        long,
        default_value = "data/external/gaia_mw_rotation/jiao2023_table1.dat"
    )]
    gaia: PathBuf,

    /// MW halo virial mass in solar masses.
    #[arg(long, default_value_t = 1.0e12)]
    m200: f64,

    /// MW halo concentration parameter.
    #[arg(long, default_value_t = 12.0)]
    c200: f64,

    /// Redshift (0 for MW).
    #[arg(long, default_value_t = 0.0)]
    z: f64,

    /// alpha_zd values to test (comma-separated).
    #[arg(long, default_value = "0.0,0.01,0.02,0.05,0.1")]
    alpha_zd_values: String,

    /// Output CSV path.
    #[arg(long, default_value = "harmonic_halo_mw.csv")]
    csv: PathBuf,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    // Load MW rotation curve
    eprintln!(
        "Loading Gaia MW rotation curve from {}...",
        cli.gaia.display()
    );
    let curve = parse_mw_rotation_curve(&cli.gaia, "Jiao2023").map_err(|e| anyhow::anyhow!(e))?;
    eprintln!(
        "  {} data points, R = {:.1} -- {:.1} kpc",
        curve.points.len(),
        curve.points.first().map_or(0.0, |p| p.r_kpc),
        curve.points.last().map_or(0.0, |p| p.r_kpc)
    );

    // NFW parameters
    let nfw = nfw_params_from_mass(cli.m200, cli.z);
    let r_s = nfw.r200_kpc / cli.c200;
    eprintln!(
        "NFW: M200={:.2e} Msun, c200={:.1}, r_s={:.2} kpc, r200={:.2} kpc",
        cli.m200, cli.c200, r_s, nfw.r200_kpc
    );

    // Parse alpha_zd values
    let alpha_values: Vec<f64> = cli
        .alpha_zd_values
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Write CSV with columns for each alpha_zd
    let mut wtr = csv::Writer::from_path(&cli.csv)?;

    // Header
    let mut header = vec![
        "r_kpc".to_string(),
        "v_obs_km_s".to_string(),
        "v_err_km_s".to_string(),
        "v_nfw_km_s".to_string(),
        "residual_frac".to_string(),
    ];
    for &alpha in &alpha_values {
        if alpha > 0.0 {
            header.push(format!("v_halo_a{alpha:.3}"));
            header.push(format!("residual_halo_a{alpha:.3}"));
        }
    }
    wtr.write_record(&header)?;

    // Compute and write data rows
    for pt in &curve.points {
        let v_nfw_val = v_circ_nfw(pt.r_kpc, cli.m200, cli.z);
        let residual = if v_nfw_val > 1.0 {
            (pt.v_c_km_s - v_nfw_val) / v_nfw_val
        } else {
            0.0
        };

        let mut row = vec![
            format!("{:.3}", pt.r_kpc),
            format!("{:.2}", pt.v_c_km_s),
            format!("{:.2}", pt.v_err_sym_km_s()),
            format!("{:.2}", v_nfw_val),
            format!("{:.6}", residual),
        ];

        for &alpha in &alpha_values {
            if alpha > 0.0 {
                let config = HarmonicHaloConfig::new(alpha, 7, r_s);
                let v_halo = v_circ_with_halos(pt.r_kpc, cli.m200, cli.c200, cli.z, &config);
                let res_halo = if v_halo > 1.0 {
                    (pt.v_c_km_s - v_halo) / v_halo
                } else {
                    0.0
                };
                row.push(format!("{:.2}", v_halo));
                row.push(format!("{:.6}", res_halo));
            }
        }

        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    eprintln!("Rotation curve analysis written to {}", cli.csv.display());

    // Single-galaxy Fourier analysis of residuals
    eprintln!();
    eprintln!("=== Fourier Analysis of MW Residuals ===");

    let norm_pts: Vec<NormalizedPoint> = curve
        .points
        .iter()
        .filter_map(|pt| {
            let v_nfw_val = v_circ_nfw(pt.r_kpc, cli.m200, cli.z);
            if v_nfw_val < 1.0 {
                return None;
            }
            let x = pt.r_kpc / r_s;
            Some(NormalizedPoint {
                x,
                delta_v: (pt.v_c_km_s - v_nfw_val) / v_nfw_val,
                delta_v_err: pt.v_err_sym_km_s() / v_nfw_val,
            })
        })
        .collect();

    let norm = NormalizedResiduals {
        name: "MilkyWay".to_string(),
        r_s_kpc: r_s,
        points: norm_pts,
    };

    let cd_params = CdDimensionParams::sedenion();
    let config = StackingConfig {
        x_min: 0.1,
        x_max: 5.0, // MW data is limited to ~25 kpc
        n_grid: 100,
        min_galaxies_per_bin: 1, // Single galaxy
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
    };

    let result = stack_residuals(&[norm], &config);
    let k = cd_params.predicted_wavenumbers();

    eprintln!("RMS residual: {:.4}", result.rms_residual);
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
    eprintln!();
    eprintln!("Detection SNR: {:.2}", result.detection_snr);
    eprintln!("alpha_zd estimate: {:.6}", result.alpha_zd_estimate);

    Ok(())
}
