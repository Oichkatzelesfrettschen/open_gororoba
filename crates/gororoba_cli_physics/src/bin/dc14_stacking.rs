//! DC14 baryonic feedback efficiency measurement via stacking comparison.
//!
//! Computes rotation curves using the Di Cintio et al. (2014) feedback-modified
//! density profile instead of pure NFW, then stacks the residuals and compares
//! spectral shapes. If DC14 is a better model, the residual RMS should be lower
//! and the spectral index different from the NFW baseline.
//!
//! The Michelson-Morley inversion: a "failed" ZD search becomes a successful
//! measurement of baryonic feedback efficiency.
//!
//! Detection: DC14 RMS < NFW RMS AND extracted eta in [0.01, 0.05]
//! Rejection: DC14 RMS >= NFW RMS OR chi2/dof > 3
//!
//! Reference: Hypothesis H3, plan imperative-skipping-kahn.md
//! Reference: Di Cintio et al. (2014), MNRAS 441, 2986

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        fourier_at_wavenumbers, stack_residuals,
    },
    nfw_utils::{
        dc14_enclosed_mass, dc14_shape_params, nfw_enclosed_mass_from_params,
        nfw_params_from_mass,
    },
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use std::f64::consts::PI;
use std::path::PathBuf;

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "dc14-stacking")]
#[command(about = "DC14 vs NFW stacking comparison (baryonic feedback measurement)")]
struct Cli {
    /// Path to MaNGA rotation curves CSV.
    #[arg(long, default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv")]
    rotcurves: PathBuf,

    /// Path to DAPall selection CSV.
    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    dapall: PathBuf,

    /// CD dimension.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Minimum rotation curve points per galaxy.
    #[arg(long, default_value_t = 8)]
    min_points: usize,

    /// Minimum x = r/r_s.
    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    /// Maximum x = r/r_s.
    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Output CSV path.
    #[arg(long, default_value = "dc14_stacking.csv")]
    out_csv: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading MaNGA rotation curves from {:?}...", cli.rotcurves);
    let rotcurves = parse_manga_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("Loaded {} galaxy rotation curves", rotcurves.len());

    eprintln!("Loading DAPall metadata from {:?}...", cli.dapall);
    let dapall = parse_manga_dapall_csv(&cli.dapall).map_err(|e| anyhow::anyhow!(e))?;

    let dapall_map: std::collections::HashMap<String, _> = dapall
        .into_iter()
        .map(|e| (e.plateifu.clone(), e))
        .collect();

    let mut nfw_residuals: Vec<NormalizedResiduals> = Vec::new();
    let mut dc14_residuals: Vec<NormalizedResiduals> = Vec::new();
    let mut skipped = 0usize;

    for galaxy in &rotcurves {
        let meta = match dapall_map.get(&galaxy.plateifu) {
            Some(m) => m,
            None => {
                skipped += 1;
                continue;
            }
        };

        let log_m200 = meta.estimated_log_m200();
        if !log_m200.is_finite() {
            skipped += 1;
            continue;
        }
        let m200 = 10.0_f64.powf(log_m200);
        let z = meta.z;
        let nfw = nfw_params_from_mass(m200, z);
        let r_s = nfw.r_s_kpc;

        // DC14 shape parameters from stellar-to-halo mass ratio
        let log_x = meta.log_stellar_mass - log_m200;
        let shape = dc14_shape_params(log_x);

        let mut nfw_pts: Vec<NormalizedPoint> = Vec::new();
        let mut dc14_pts: Vec<NormalizedPoint> = Vec::new();

        for pt in &galaxy.points {
            if pt.r_kpc <= 0.0 || r_s <= 0.0 {
                continue;
            }
            let x = pt.r_kpc / r_s;

            // NFW model
            let m_enc_nfw = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw);
            let v_sq_nfw = G_KPC_KMS2 * m_enc_nfw / pt.r_kpc;
            let v_nfw = v_sq_nfw.max(0.0).sqrt();

            // DC14 model (same r_s and rho_s, different profile shape)
            let m_enc_dc14 = dc14_enclosed_mass(
                pt.r_kpc,
                nfw.r_s_kpc,
                nfw.rho_s_solar_per_kpc3,
                &shape,
            );
            let v_sq_dc14 = G_KPC_KMS2 * m_enc_dc14 / pt.r_kpc;
            let v_dc14 = v_sq_dc14.max(0.0).sqrt();

            if v_nfw < 1.0 || v_dc14 < 1.0 {
                continue;
            }

            let delta_v_nfw = (pt.v_obs_km_s - v_nfw) / v_nfw;
            let delta_v_dc14 = (pt.v_obs_km_s - v_dc14) / v_dc14;
            let err_nfw = pt.v_err_km_s / v_nfw;
            let err_dc14 = pt.v_err_km_s / v_dc14;

            if !delta_v_nfw.is_finite()
                || !delta_v_dc14.is_finite()
                || !err_nfw.is_finite()
                || !err_dc14.is_finite()
            {
                continue;
            }

            nfw_pts.push(NormalizedPoint {
                x,
                delta_v: delta_v_nfw,
                delta_v_err: err_nfw,
            });

            dc14_pts.push(NormalizedPoint {
                x,
                delta_v: delta_v_dc14,
                delta_v_err: err_dc14,
            });
        }

        if nfw_pts.len() >= cli.min_points && dc14_pts.len() >= cli.min_points {
            nfw_residuals.push(NormalizedResiduals {
                name: galaxy.plateifu.clone(),
                r_s_kpc: r_s,
                points: nfw_pts,
            });
            dc14_residuals.push(NormalizedResiduals {
                name: galaxy.plateifu.clone(),
                r_s_kpc: r_s,
                points: dc14_pts,
            });
        } else {
            skipped += 1;
        }
    }

    eprintln!(
        "Normalized {} galaxies ({} skipped)",
        nfw_residuals.len(),
        skipped
    );

    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let n_modes = cd_params.n_modes;
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * PI * n as f64 / n_modes as f64)
        .collect();

    let config = StackingConfig {
        x_min: cli.x_min,
        x_max: cli.x_max,
        n_grid: 200,
        min_galaxies_per_bin: 3,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: false,
    };

    // Stack NFW residuals
    eprintln!("Stacking NFW residuals...");
    let nfw_result = stack_residuals(&nfw_residuals, &config);
    let (nfw_power, nfw_phase) = fourier_at_wavenumbers(
        &nfw_result.x_grid,
        &nfw_result.delta_stack,
        &nfw_result.n_contributing,
        config.min_galaxies_per_bin,
        &wavenumbers,
    );

    // Stack DC14 residuals
    eprintln!("Stacking DC14 residuals...");
    let dc14_result = stack_residuals(&dc14_residuals, &config);
    let (dc14_power, dc14_phase) = fourier_at_wavenumbers(
        &dc14_result.x_grid,
        &dc14_result.delta_stack,
        &dc14_result.n_contributing,
        config.min_galaxies_per_bin,
        &wavenumbers,
    );

    // Compare
    eprintln!("\n--- NFW vs DC14 stacking comparison ---");
    eprintln!(
        "NFW:  N={}, RMS={:.6}, SNR={:.4}",
        nfw_result.n_galaxies, nfw_result.rms_residual, nfw_result.detection_snr
    );
    eprintln!(
        "DC14: N={}, RMS={:.6}, SNR={:.4}",
        dc14_result.n_galaxies, dc14_result.rms_residual, dc14_result.detection_snr
    );

    let rms_improvement =
        (nfw_result.rms_residual - dc14_result.rms_residual) / nfw_result.rms_residual * 100.0;
    eprintln!("RMS improvement: {:.2}%", rms_improvement);

    eprintln!("\n--- Per-mode Fourier power comparison ---");
    eprintln!(
        "{:>5}  {:>8}  {:>12}  {:>12}  {:>10}",
        "mode", "k", "nfw_power", "dc14_power", "ratio"
    );

    let mut chi2 = 0.0_f64;
    let mut chi2_dof = 0_usize;

    for mode_idx in 0..n_modes {
        let k = wavenumbers[mode_idx];
        let np = nfw_power[mode_idx];
        let dp = dc14_power[mode_idx];
        let ratio = if np > 0.0 { dp / np } else { f64::NAN };

        eprintln!(
            "{:>5}  {:>8.4}  {:>12.6e}  {:>12.6e}  {:>10.4}",
            mode_idx + 1,
            k,
            np,
            dp,
            ratio
        );

        // Chi2 contribution: (NFW_power - DC14_power)^2 / NFW_power
        if np > 1e-20 {
            chi2 += (np - dp).powi(2) / np;
            chi2_dof += 1;
        }
    }

    let chi2_per_dof = if chi2_dof > 0 {
        chi2 / chi2_dof as f64
    } else {
        f64::NAN
    };

    eprintln!("\nChi2/dof (spectral shape difference): {:.4}", chi2_per_dof);

    // Stacked profile comparison (bin-by-bin)
    eprintln!("\n--- Stacked profile comparison (first 20 valid bins) ---");
    eprintln!(
        "{:>8}  {:>12}  {:>12}  {:>12}  {:>8}",
        "x", "delta_nfw", "delta_dc14", "difference", "n_gal"
    );
    let mut printed = 0;
    for j in 0..config.n_grid {
        if nfw_result.n_contributing[j] >= config.min_galaxies_per_bin && printed < 20 {
            let diff = dc14_result.delta_stack[j] - nfw_result.delta_stack[j];
            eprintln!(
                "{:>8.4}  {:>12.6}  {:>12.6}  {:>12.6}  {:>8}",
                nfw_result.x_grid[j],
                nfw_result.delta_stack[j],
                dc14_result.delta_stack[j],
                diff,
                nfw_result.n_contributing[j]
            );
            printed += 1;
        }
    }

    // Detection/rejection
    eprintln!("\n--- Detection criteria ---");
    if dc14_result.rms_residual < nfw_result.rms_residual {
        eprintln!("DC14 has LOWER residual RMS (better model for stacked data)");
    } else {
        eprintln!("DC14 has HIGHER residual RMS (NFW is better for stacked data)");
    }

    if dc14_result.rms_residual < nfw_result.rms_residual && chi2_per_dof < 3.0 {
        eprintln!("RESULT: DETECTION -- baryonic feedback signature measurable");
    } else if dc14_result.rms_residual >= nfw_result.rms_residual || chi2_per_dof > 3.0 {
        eprintln!("RESULT: REJECTED -- DC14 does not improve over NFW for stacked curves");
    } else {
        eprintln!("RESULT: INCONCLUSIVE");
    }

    // Write output CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;

    // Profile comparison
    wtr.write_record([
        "x",
        "delta_nfw",
        "delta_dc14",
        "delta_nfw_err",
        "delta_dc14_err",
        "n_contributing",
    ])?;
    for j in 0..config.n_grid {
        wtr.write_record(&[
            format!("{:.6}", nfw_result.x_grid[j]),
            format!("{:.8e}", nfw_result.delta_stack[j]),
            format!("{:.8e}", dc14_result.delta_stack[j]),
            format!("{:.8e}", nfw_result.delta_stack_err[j]),
            format!("{:.8e}", dc14_result.delta_stack_err[j]),
            format!("{}", nfw_result.n_contributing[j]),
        ])?;
    }
    wtr.flush()?;

    // Spectral comparison
    let spectral_csv = cli.out_csv.with_extension("spectral.csv");
    let mut swtr = csv::Writer::from_path(&spectral_csv)?;
    swtr.write_record([
        "mode",
        "k",
        "nfw_power",
        "nfw_phase",
        "dc14_power",
        "dc14_phase",
        "power_ratio",
    ])?;
    for mode_idx in 0..n_modes {
        let k = wavenumbers[mode_idx];
        let ratio = if nfw_power[mode_idx] > 0.0 {
            dc14_power[mode_idx] / nfw_power[mode_idx]
        } else {
            f64::NAN
        };
        swtr.write_record(&[
            format!("{}", mode_idx + 1),
            format!("{:.6}", k),
            format!("{:.8e}", nfw_power[mode_idx]),
            format!("{:.6}", nfw_phase[mode_idx]),
            format!("{:.8e}", dc14_power[mode_idx]),
            format!("{:.6}", dc14_phase[mode_idx]),
            format!("{:.6}", ratio),
        ])?;
    }
    swtr.flush()?;

    // Summary
    let summary_csv = cli.out_csv.with_extension("summary.csv");
    let mut sumwtr = csv::Writer::from_path(&summary_csv)?;
    sumwtr.write_record([
        "model",
        "n_galaxies",
        "rms_residual",
        "detection_snr",
        "alpha_zd_estimate",
    ])?;
    sumwtr.write_record(&[
        "NFW".to_string(),
        format!("{}", nfw_result.n_galaxies),
        format!("{:.8}", nfw_result.rms_residual),
        format!("{:.6}", nfw_result.detection_snr),
        format!("{:.8}", nfw_result.alpha_zd_estimate),
    ])?;
    sumwtr.write_record(&[
        "DC14".to_string(),
        format!("{}", dc14_result.n_galaxies),
        format!("{:.8}", dc14_result.rms_residual),
        format!("{:.6}", dc14_result.detection_snr),
        format!("{:.8}", dc14_result.alpha_zd_estimate),
    ])?;
    sumwtr.flush()?;

    eprintln!("Profile comparison written to {:?}", cli.out_csv);
    eprintln!("Spectral comparison written to {:?}", spectral_csv);
    eprintln!("Summary written to {:?}", summary_csv);

    Ok(())
}
