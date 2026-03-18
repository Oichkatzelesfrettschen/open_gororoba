//! Contrarian D2: Mass-quartile independent Fourier stacking.
//!
//! Challenges the null result by testing whether mass-averaging destroys a
//! mass-localized signal. The full-sample stack at x=0.5 is dominated by
//! high-mass galaxies (r_s ~ 25 kpc), while x=1.0 is dominated by low-mass
//! galaxies (r_s ~ 10 kpc). If the baryonic systematic has a mass-dependent
//! shape, a universal stacking could cancel genuine harmonic structure.
//!
//! Method:
//! 1. Sort galaxies by log(M200), split into 4 quartiles (~1750 each)
//! 2. Stack each quartile independently on its own x-grid
//! 3. Compute 7-mode Fourier power and SNR per quartile
//! 4. Compare: does any quartile show SNR > 3.0 while full sample shows 0.43?
//!
//! Detection: Any quartile SNR > 3.0 (mass averaging destroys signal)
//! Failure: All quartiles SNR < 1.5 and consistent (null is mass-universal)
//!
//! Key difference from H2 (mass_binned_phase): H2 tested phase TRENDS across bins.
//! D2 tests absolute Fourier POWER per quartile -- looking for mass-localized
//! concentrations, not monotonic trends.

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        fourier_at_wavenumbers, stack_residuals,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use std::{f64::consts::PI, path::PathBuf};

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "mass-quartile-fourier")]
#[command(about = "D2: Mass-quartile independent Fourier stacking")]
struct Cli {
    #[arg(
        long,
        default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv"
    )]
    rotcurves: PathBuf,

    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    dapall: PathBuf,

    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    #[arg(long, default_value_t = 8)]
    min_points: usize,

    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Number of mass quartiles (default 4).
    #[arg(long, default_value_t = 4)]
    n_quartiles: usize,

    #[arg(long, default_value = "data/results/e183/mass_quartile_fourier.csv")]
    out_csv: PathBuf,
}

struct GalaxyWithMass {
    log_m200: f64,
    r_s_kpc: f64,
    residuals: NormalizedResiduals,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading MaNGA rotation curves...");
    let rotcurves = parse_manga_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    let dapall = parse_manga_dapall_csv(&cli.dapall).map_err(|e| anyhow::anyhow!(e))?;
    let dapall_map: std::collections::HashMap<String, _> = dapall
        .into_iter()
        .map(|e| (e.plateifu.clone(), e))
        .collect();

    let mut galaxies: Vec<GalaxyWithMass> = Vec::new();
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
        let nfw = nfw_params_from_mass(m200, meta.z);
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
            galaxies.push(GalaxyWithMass {
                log_m200,
                r_s_kpc: r_s,
                residuals: NormalizedResiduals {
                    name: galaxy.plateifu.clone(),
                    r_s_kpc: r_s,
                    points,
                },
            });
        } else {
            skipped += 1;
        }
    }

    let n_total = galaxies.len();
    eprintln!("Normalized {} galaxies ({} skipped)", n_total, skipped);

    // Sort by log(M200)
    galaxies.sort_by(|a, b| a.log_m200.partial_cmp(&b.log_m200).unwrap());

    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let n_modes = cd_params.n_modes;
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * PI * n as f64 / n_modes as f64)
        .collect();

    // Full-sample baseline
    eprintln!("\n--- Full-sample baseline ---");
    let full_residuals: Vec<NormalizedResiduals> =
        galaxies.iter().map(|g| g.residuals.clone()).collect();
    let full_config = StackingConfig {
        x_min: cli.x_min,
        x_max: cli.x_max,
        n_grid: 200,
        min_galaxies_per_bin: 3,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: false,
    };
    let full_result = stack_residuals(&full_residuals, &full_config);
    let (full_power, _) = fourier_at_wavenumbers(
        &full_result.x_grid,
        &full_result.delta_stack,
        &full_result.n_contributing,
        full_config.min_galaxies_per_bin,
        &wavenumbers,
    );
    let full_max_power = full_power.iter().copied().fold(0.0_f64, f64::max);
    let full_snr = if full_result.rms_residual > 0.0 {
        full_max_power.sqrt() / full_result.rms_residual
    } else {
        0.0
    };

    eprintln!(
        "Full sample: N={}, RMS={:.6}, SNR={:.4}",
        n_total, full_result.rms_residual, full_snr
    );

    // Quartile analysis
    let q_size = n_total / cli.n_quartiles;
    let remainder = n_total % cli.n_quartiles;

    struct QuartileResult {
        quartile: usize,
        n_galaxies: usize,
        log_m200_min: f64,
        log_m200_max: f64,
        log_m200_median: f64,
        r_s_median: f64,
        n_valid_bins: usize,
        max_x_populated: f64,
        rms_residual: f64,
        snr: f64,
        power: Vec<f64>,
    }

    let mut quartile_results: Vec<QuartileResult> = Vec::new();
    let mut offset = 0usize;

    eprintln!("\n--- Mass quartile analysis ---");
    for q in 0..cli.n_quartiles {
        let this_size = q_size + if q < remainder { 1 } else { 0 };
        let q_galaxies = &galaxies[offset..offset + this_size];

        let log_m200_min = q_galaxies[0].log_m200;
        let log_m200_max = q_galaxies[this_size - 1].log_m200;
        let log_m200_median = q_galaxies[this_size / 2].log_m200;

        let mut r_s_vals: Vec<f64> = q_galaxies.iter().map(|g| g.r_s_kpc).collect();
        r_s_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let r_s_median = r_s_vals[this_size / 2];

        let q_residuals: Vec<NormalizedResiduals> =
            q_galaxies.iter().map(|g| g.residuals.clone()).collect();

        let q_result = stack_residuals(&q_residuals, &full_config);
        let (q_power, _) = fourier_at_wavenumbers(
            &q_result.x_grid,
            &q_result.delta_stack,
            &q_result.n_contributing,
            full_config.min_galaxies_per_bin,
            &wavenumbers,
        );

        let q_max_power = q_power.iter().copied().fold(0.0_f64, f64::max);
        let q_snr = if q_result.rms_residual > 0.0 {
            q_max_power.sqrt() / q_result.rms_residual
        } else {
            0.0
        };

        // Count valid bins and max x populated
        let n_valid = q_result
            .n_contributing
            .iter()
            .filter(|&&n| n >= full_config.min_galaxies_per_bin)
            .count();
        let max_x = q_result
            .x_grid
            .iter()
            .zip(q_result.n_contributing.iter())
            .filter(|(_, n)| **n >= full_config.min_galaxies_per_bin)
            .map(|(x, _)| *x)
            .fold(0.0_f64, f64::max);

        eprintln!(
            "  Q{}: N={}, log(M200)=[{:.2}, {:.2}], r_s_med={:.1} kpc, valid_bins={}, max_x={:.2}, RMS={:.6}, SNR={:.4}",
            q + 1,
            this_size,
            log_m200_min,
            log_m200_max,
            r_s_median,
            n_valid,
            max_x,
            q_result.rms_residual,
            q_snr
        );

        // Per-mode power
        for (i, (&k, &p)) in wavenumbers.iter().zip(q_power.iter()).enumerate() {
            let mode_snr = if q_result.rms_residual > 0.0 {
                p.sqrt() / q_result.rms_residual
            } else {
                0.0
            };
            eprintln!(
                "    mode {}: k={:.4}, power={:.6e}, snr={:.4}",
                i + 1,
                k,
                p,
                mode_snr
            );
        }

        quartile_results.push(QuartileResult {
            quartile: q + 1,
            n_galaxies: this_size,
            log_m200_min,
            log_m200_max,
            log_m200_median,
            r_s_median,
            n_valid_bins: n_valid,
            max_x_populated: max_x,
            rms_residual: q_result.rms_residual,
            snr: q_snr,
            power: q_power,
        });

        offset += this_size;
    }

    // Verdict
    let max_q_snr = quartile_results
        .iter()
        .map(|q| q.snr)
        .fold(0.0_f64, f64::max);
    let any_detection = quartile_results.iter().any(|q| q.snr > 3.0);

    eprintln!("\n--- Contrarian D2 Verdict ---");
    eprintln!("Full-sample SNR: {:.4}", full_snr);
    eprintln!("Max quartile SNR: {:.4}", max_q_snr);

    if any_detection {
        let det_q: Vec<usize> = quartile_results
            .iter()
            .filter(|q| q.snr > 3.0)
            .map(|q| q.quartile)
            .collect();
        eprintln!("DETECTION: Quartile(s) {:?} show SNR > 3.0", det_q);
        eprintln!("  Mass averaging is destroying a mass-localized signal!");
    } else if max_q_snr < 1.5 {
        eprintln!("FAILURE: All quartiles show SNR < 1.5");
        eprintln!("  The null result is mass-universal -- no mass-localized signal exists");

        // Bonus: check if baryonic shape is consistent across quartiles
        let r_s_range = quartile_results.last().unwrap().r_s_median
            / quartile_results.first().unwrap().r_s_median;
        eprintln!(
            "  r_s range: {:.1}x (Q1 median {:.1} kpc, Q4 median {:.1} kpc)",
            r_s_range,
            quartile_results.first().unwrap().r_s_median,
            quartile_results.last().unwrap().r_s_median,
        );
        eprintln!(
            "  NFW normalization produces consistent null across {:.1}x variation in r_s",
            r_s_range
        );
    } else {
        eprintln!(
            "INCONCLUSIVE: Max quartile SNR={:.4} (between 1.5 and 3.0)",
            max_q_snr
        );
    }

    // Write CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "quartile",
        "n_galaxies",
        "log_m200_min",
        "log_m200_max",
        "log_m200_median",
        "r_s_median_kpc",
        "n_valid_bins",
        "max_x_populated",
        "rms_residual",
        "snr",
        "power_mode1",
        "power_mode2",
        "power_mode3",
        "power_mode4",
        "power_mode5",
        "power_mode6",
        "power_mode7",
    ])?;

    // Full sample row
    wtr.write_record(&[
        "FULL".to_string(),
        format!("{}", n_total),
        format!("{:.4}", galaxies[0].log_m200),
        format!("{:.4}", galaxies[n_total - 1].log_m200),
        format!("{:.4}", galaxies[n_total / 2].log_m200),
        "".to_string(),
        "".to_string(),
        "".to_string(),
        format!("{:.8}", full_result.rms_residual),
        format!("{:.6}", full_snr),
        format!("{:.8e}", full_power[0]),
        format!("{:.8e}", full_power[1]),
        format!("{:.8e}", full_power[2]),
        format!("{:.8e}", full_power[3]),
        format!("{:.8e}", full_power[4]),
        format!("{:.8e}", full_power[5]),
        format!("{:.8e}", full_power[6]),
    ])?;

    for q in &quartile_results {
        wtr.write_record(&[
            format!("Q{}", q.quartile),
            format!("{}", q.n_galaxies),
            format!("{:.4}", q.log_m200_min),
            format!("{:.4}", q.log_m200_max),
            format!("{:.4}", q.log_m200_median),
            format!("{:.2}", q.r_s_median),
            format!("{}", q.n_valid_bins),
            format!("{:.4}", q.max_x_populated),
            format!("{:.8}", q.rms_residual),
            format!("{:.6}", q.snr),
            format!("{:.8e}", q.power[0]),
            format!("{:.8e}", q.power[1]),
            format!("{:.8e}", q.power[2]),
            format!("{:.8e}", q.power[3]),
            format!("{:.8e}", q.power[4]),
            format!("{:.8e}", q.power[5]),
            format!("{:.8e}", q.power[6]),
        ])?;
    }

    wtr.flush()?;
    eprintln!("\nResults written to {:?}", cli.out_csv);

    Ok(())
}
