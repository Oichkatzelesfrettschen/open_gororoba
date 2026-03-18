//! Signal injection-recovery sweep for MaNGA harmonic stacking pipeline.
//!
//! Injects the predicted CD-ZD harmonic signal at a range of alpha_zd values
//! and measures the recovery ratio.
//!
//! The current executed sweep is a calibration failure:
//! the recovered SNR stays flat near 0.43 and the recovered alpha_zd stays
//! near 0.295 for injected alpha_zd in [0, 0.016].
//! This means the lane is not yet valid evidence for pipeline sensitivity.
//!
//! Reference: C-1405, E-197

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        injection_recovery_sweep,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use std::path::PathBuf;

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "harmonic-halo-injection")]
#[command(about = "ZD signal injection-recovery sweep for pipeline validation")]
struct Cli {
    /// Path to MaNGA rotation curves CSV.
    #[arg(long, default_value = "data/external/manga/manga_rotcurves_all.csv")]
    rotcurves: PathBuf,

    /// Path to DAPall selection CSV.
    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    dapall: PathBuf,

    /// CD dimension for signal injection.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Single injection amplitude (only used without --sweep).
    #[arg(long, default_value_t = 0.004)]
    alpha_zd: f64,

    /// Sweep alpha_zd over {0, 0.001, 0.002, 0.004, 0.008, 0.016}.
    #[arg(long, default_value_t = false)]
    sweep: bool,

    /// Minimum rotation curve points per galaxy.
    #[arg(long, default_value_t = 5)]
    min_points: usize,

    /// Minimum x = r/r_s for stacking grid.
    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    /// Maximum x = r/r_s for stacking grid.
    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Output CSV path.
    #[arg(long, default_value = "injection_recovery.csv")]
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
        "Normalized {} galaxies ({} skipped)",
        normalized.len(),
        skipped
    );

    let alpha_zd_values: Vec<f64> = if cli.sweep {
        vec![0.0, 0.001, 0.002, 0.004, 0.008, 0.016]
    } else {
        vec![0.0, cli.alpha_zd]
    };

    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let config = StackingConfig {
        x_min: cli.x_min,
        x_max: cli.x_max,
        n_grid: 200,
        min_galaxies_per_bin: 3,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: false,
    };

    eprintln!(
        "Running injection-recovery sweep over {} alpha_zd values...",
        alpha_zd_values.len()
    );
    let results = injection_recovery_sweep(&normalized, &alpha_zd_values, cli.cd_dim, &config);

    eprintln!("Injection-recovery results:");
    eprintln!(
        "  {:>8}  {:>10}  {:>12}  {:>14}",
        "alpha_inj", "snr", "alpha_rec", "recovery_ratio"
    );
    for r in &results {
        eprintln!(
            "  {:>8.4}  {:>10.4}  {:>12.4}  {:>14}",
            r.alpha_zd_injected,
            r.detected_snr,
            r.alpha_zd_recovered,
            if r.recovery_ratio.is_nan() {
                "NaN".to_string()
            } else {
                format!("{:.4}", r.recovery_ratio)
            }
        );
    }

    // Write output CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "alpha_zd_injected",
        "detected_snr",
        "alpha_zd_recovered",
        "recovery_ratio",
    ])?;
    for r in &results {
        wtr.write_record(&[
            format!("{}", r.alpha_zd_injected),
            format!("{}", r.detected_snr),
            format!("{}", r.alpha_zd_recovered),
            if r.recovery_ratio.is_nan() {
                "NaN".to_string()
            } else {
                format!("{}", r.recovery_ratio)
            },
        ])?;
    }
    wtr.flush()?;
    eprintln!("Results written to {:?}", cli.out_csv);

    Ok(())
}
