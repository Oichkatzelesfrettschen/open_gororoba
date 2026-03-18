//! Hypothesis B: Inclination-stratified injection recovery calibration.
//!
//! Tests whether the injection recovery non-linearity (ratios 18-300:1 at
//! alpha_zd 0.001-0.016) is driven by inclination-dependent PSF smearing.
//!
//! Runs injection recovery at 6 alpha_zd levels for 3 inclination sub-samples:
//! - Low-i:  30-45 deg (least projection effects)
//! - Mid-i:  45-60 deg
//! - High-i: 60-70 deg (most projection effects, IFU edge spike)
//!
//! Detection: Recovery ratio at alpha_zd=0.004 differs by >2x between low-i and high-i
//! Failure: Recovery ratios agree within 20% across inclination bins

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        fourier_at_wavenumbers, stack_residuals,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use std::f64::consts::PI;
use std::path::PathBuf;

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "inclination-injection-calibration")]
#[command(about = "Inclination-stratified injection recovery for PSF calibration")]
struct Cli {
    #[arg(long, default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv")]
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

    #[arg(long, default_value = "data/results/e183/inclination_injection.csv")]
    out_csv: PathBuf,
}

struct GalaxyWithInclination {
    inclination_deg: f64,
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

    let mut galaxies: Vec<GalaxyWithInclination> = Vec::new();
    let mut skipped = 0usize;

    for galaxy in &rotcurves {
        let meta = match dapall_map.get(&galaxy.plateifu) {
            Some(m) => m,
            None => { skipped += 1; continue; }
        };

        let log_m200 = meta.estimated_log_m200();
        if !log_m200.is_finite() { skipped += 1; continue; }
        let incl = meta.inclination_deg();
        if !incl.is_finite() { skipped += 1; continue; }

        let m200 = 10.0_f64.powf(log_m200);
        let nfw = nfw_params_from_mass(m200, meta.z);
        let r_s = nfw.r_s_kpc;

        let points: Vec<NormalizedPoint> = galaxy.points.iter().filter_map(|pt| {
            if pt.r_kpc <= 0.0 || r_s <= 0.0 { return None; }
            let x = pt.r_kpc / r_s;
            let m_enc = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw);
            let v_sq = G_KPC_KMS2 * m_enc / pt.r_kpc;
            let v_nfw = v_sq.max(0.0).sqrt();
            if v_nfw < 1.0 { return None; }
            let delta_v = (pt.v_obs_km_s - v_nfw) / v_nfw;
            let delta_v_err = pt.v_err_km_s / v_nfw;
            if !delta_v.is_finite() || !delta_v_err.is_finite() { return None; }
            Some(NormalizedPoint { x, delta_v, delta_v_err })
        }).collect();

        if points.len() >= cli.min_points {
            galaxies.push(GalaxyWithInclination {
                inclination_deg: incl,
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

    eprintln!("Normalized {} galaxies ({} skipped)", galaxies.len(), skipped);

    // Define inclination bins
    let incl_bins: [(f64, f64, &str); 3] = [
        (30.0, 45.0, "low-i"),
        (45.0, 60.0, "mid-i"),
        (60.0, 70.0, "high-i"),
    ];

    // Injection levels
    let alpha_levels: [f64; 6] = [0.0, 0.001, 0.002, 0.004, 0.008, 0.016];

    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let n_modes = cd_params.n_modes;
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * PI * n as f64 / n_modes as f64)
        .collect();

    let mut all_results: Vec<(String, f64, f64, f64, f64, usize)> = Vec::new();

    for &(incl_min, incl_max, bin_name) in &incl_bins {
        let subset: Vec<&GalaxyWithInclination> = galaxies.iter()
            .filter(|g| g.inclination_deg >= incl_min && g.inclination_deg < incl_max)
            .collect();

        let n_sub = subset.len();
        eprintln!("\n{}: {} galaxies (incl {:.0}-{:.0} deg)", bin_name, n_sub, incl_min, incl_max);

        if n_sub < 50 {
            eprintln!("  Too few galaxies, skipping");
            continue;
        }

        for &alpha_zd in &alpha_levels {
            // Inject signal: delta_v += alpha_zd * sum_n (0.5/n) * cos(k_n * x) * exp(-x)
            let injected: Vec<NormalizedResiduals> = subset.iter().map(|g| {
                let mut points = g.residuals.points.clone();
                if alpha_zd > 0.0 {
                    for pt in &mut points {
                        let mut signal = 0.0_f64;
                        for (mode_idx, &k) in wavenumbers.iter().enumerate() {
                            let n = (mode_idx + 1) as f64;
                            let w = 0.5 / n;
                            signal += w * (k * pt.x).cos() * (-pt.x).exp();
                        }
                        pt.delta_v += alpha_zd * signal;
                    }
                }
                NormalizedResiduals {
                    name: g.residuals.name.clone(),
                    r_s_kpc: g.residuals.r_s_kpc,
                    points,
                }
            }).collect();

            let config = StackingConfig {
                x_min: cli.x_min,
                x_max: cli.x_max,
                n_grid: 200,
                min_galaxies_per_bin: 3,
                inverse_variance_weighting: true,
                cd_params: cd_params.clone(),
                exclude_psf_flagged: false,
            };

            let result = stack_residuals(&injected, &config);

            // Compute Fourier at CD-ZD wavenumbers
            let (power, _phase) = fourier_at_wavenumbers(
                &result.x_grid, &result.delta_stack, &result.n_contributing,
                config.min_galaxies_per_bin, &wavenumbers,
            );

            let max_power = power.iter().copied().fold(0.0_f64, f64::max);
            let detected_snr = if result.rms_residual > 0.0 {
                max_power.sqrt() / result.rms_residual
            } else {
                0.0
            };

            // Recovery: alpha_recovered = 4 * sqrt(max_power) / sqrt(n_sub)
            let alpha_recovered = 4.0 * max_power.sqrt();
            let recovery_ratio = if alpha_zd > 0.0 {
                alpha_recovered / alpha_zd
            } else {
                f64::NAN
            };

            eprintln!(
                "  alpha_zd={:.4}: SNR={:.4}, alpha_rec={:.6}, ratio={:.2}",
                alpha_zd, detected_snr, alpha_recovered, recovery_ratio
            );

            all_results.push((
                bin_name.to_string(),
                alpha_zd,
                detected_snr,
                alpha_recovered,
                recovery_ratio,
                n_sub,
            ));
        }
    }

    // Analysis: compare recovery ratios at alpha_zd=0.004 across inclination bins
    eprintln!("\n--- Recovery Ratio Comparison at alpha_zd=0.004 ---");
    let mut ratios_at_004: Vec<(String, f64)> = Vec::new();
    for r in &all_results {
        if (r.1 - 0.004).abs() < 1e-6 {
            eprintln!("  {}: recovery_ratio={:.2}, N={}", r.0, r.4, r.5);
            ratios_at_004.push((r.0.clone(), r.4));
        }
    }

    if ratios_at_004.len() >= 2 {
        let max_ratio = ratios_at_004.iter().map(|r| r.1).fold(0.0_f64, f64::max);
        let min_ratio = ratios_at_004.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        let spread = if min_ratio > 0.0 { max_ratio / min_ratio } else { f64::INFINITY };

        eprintln!("\n--- Hypothesis B Verdict ---");
        if spread > 2.0 {
            eprintln!("DETECTION: Recovery ratio differs by {:.1}x between inclination bins", spread);
            eprintln!("  PSF smearing is the dominant source of injection non-linearity");
        } else if spread < 1.2 {
            eprintln!("FAILURE: Recovery ratios agree within {:.0}%", (spread - 1.0) * 100.0);
            eprintln!("  Non-linearity is NOT PSF-driven");
        } else {
            eprintln!("INCONCLUSIVE: Recovery ratio spread = {:.2}x", spread);
        }
    }

    // Write CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "inclination_bin", "alpha_zd_injected", "detected_snr",
        "alpha_zd_recovered", "recovery_ratio", "n_galaxies",
    ])?;
    for r in &all_results {
        wtr.write_record(&[
            r.0.clone(),
            format!("{:.6}", r.1),
            format!("{:.6}", r.2),
            format!("{:.6}", r.3),
            format!("{:.4}", r.4),
            format!("{}", r.5),
        ])?;
    }
    wtr.flush()?;

    eprintln!("\nResults written to {:?}", cli.out_csv);
    Ok(())
}
