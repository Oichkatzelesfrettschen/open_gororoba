//! F2: Q3 mass-quartile injection recovery validation.
//!
//! The Q3 mass quartile (log M200 = 11.54--12.07, N=1749) shows anomalous
//! SNR=3.60 with only 6 valid x-bins at x=0.50--0.74.  This binary tests
//! whether the stacking-Fourier pipeline can actually *detect* an injected
//! signal with so few bins, or whether the Q3 "detection" is a sparse-bin
//! artifact.
//!
//! Method:
//! 1. Sort galaxies by log(M200), extract Q3 (50th--75th percentile)
//! 2. Stack Q3 baseline and compute SNR (should reproduce 3.60)
//! 3. Inject signal at alpha_zd = {0.001, 0.004, 0.01} into Q3
//! 4. Measure delta_SNR = SNR(injected) - SNR(baseline)
//! 5. Bootstrap 200x to get 95% CI on delta_SNR
//! 6. Compare with Q1 control (19 valid bins) and synthetic-6-bin Q1
//!
//! Detection (genuine sensitivity): delta_SNR > 2.0 at alpha_zd=0.004
//! Failure (artifact): delta_SNR < 0.5 (pipeline blind at 6 bins)
//!
//! Reference: E-202

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        fourier_at_wavenumbers, inject_zd_signal, stack_residuals,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;
use std::path::PathBuf;

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "q3-injection-recovery")]
#[command(about = "F2: Q3 mass-quartile injection recovery validation (E-202)")]
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

    /// Number of bootstrap resamples.
    #[arg(long, default_value_t = 200)]
    n_bootstrap: usize,

    /// Random seed for bootstrap.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value = "data/results/e202/q3_injection_recovery.csv")]
    out_csv: PathBuf,
}

struct GalaxyWithMass {
    log_m200: f64,
    residuals: NormalizedResiduals,
}

/// Compute SNR from a stacking result: sqrt(max_power) / rms.
fn compute_snr(
    residuals: &[NormalizedResiduals],
    config: &StackingConfig,
    wavenumbers: &[f64],
) -> (f64, f64, usize) {
    let result = stack_residuals(residuals, config);
    let (power, _) = fourier_at_wavenumbers(
        &result.x_grid,
        &result.delta_stack,
        &result.n_contributing,
        config.min_galaxies_per_bin,
        wavenumbers,
    );
    let max_power = power.iter().copied().fold(0.0_f64, f64::max);
    let snr = if result.rms_residual > 0.0 {
        max_power.sqrt() / result.rms_residual
    } else {
        0.0
    };
    let n_valid = result
        .n_contributing
        .iter()
        .filter(|&&n| n >= config.min_galaxies_per_bin)
        .count();
    (snr, result.rms_residual, n_valid)
}

/// Compute per-mode SNR from stacking result.
fn compute_per_mode_snr(
    residuals: &[NormalizedResiduals],
    config: &StackingConfig,
    wavenumbers: &[f64],
) -> Vec<f64> {
    let result = stack_residuals(residuals, config);
    let (power, _) = fourier_at_wavenumbers(
        &result.x_grid,
        &result.delta_stack,
        &result.n_contributing,
        config.min_galaxies_per_bin,
        wavenumbers,
    );
    power
        .iter()
        .map(|&p| {
            if result.rms_residual > 0.0 {
                p.sqrt() / result.rms_residual
            } else {
                0.0
            }
        })
        .collect()
}

/// Bootstrap resample: draw n galaxies with replacement.
fn bootstrap_sample(
    galaxies: &[NormalizedResiduals],
    rng: &mut ChaCha8Rng,
) -> Vec<NormalizedResiduals> {
    let n = galaxies.len();
    (0..n)
        .map(|_| {
            let idx = rng.gen_range(0..n);
            galaxies[idx].clone()
        })
        .collect()
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

    // Sort by log(M200) and extract quartiles
    galaxies.sort_by(|a, b| a.log_m200.partial_cmp(&b.log_m200).unwrap());
    let q_size = n_total / 4;

    let q1_galaxies: Vec<NormalizedResiduals> =
        galaxies[..q_size].iter().map(|g| g.residuals.clone()).collect();
    let q3_galaxies: Vec<NormalizedResiduals> = galaxies[2 * q_size..3 * q_size]
        .iter()
        .map(|g| g.residuals.clone())
        .collect();

    let q3_log_m200_min = galaxies[2 * q_size].log_m200;
    let q3_log_m200_max = galaxies[3 * q_size - 1].log_m200;
    eprintln!(
        "Q3: N={}, log(M200)=[{:.2}, {:.2}]",
        q3_galaxies.len(),
        q3_log_m200_min,
        q3_log_m200_max
    );
    eprintln!("Q1 control: N={}", q1_galaxies.len());

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

    // Restricted config for synthetic-6-bin ablation: x_max=0.74
    let config_6bin = StackingConfig {
        x_max: 0.74,
        ..config.clone()
    };

    // ---- Q3 baseline ----
    let (q3_baseline_snr, q3_rms, q3_valid_bins) =
        compute_snr(&q3_galaxies, &config, &wavenumbers);
    eprintln!(
        "\nQ3 baseline: SNR={:.4}, RMS={:.6}, valid_bins={}",
        q3_baseline_snr, q3_rms, q3_valid_bins
    );

    // ---- Q1 baseline ----
    let (q1_baseline_snr, q1_rms, q1_valid_bins) =
        compute_snr(&q1_galaxies, &config, &wavenumbers);
    eprintln!(
        "Q1 baseline: SNR={:.4}, RMS={:.6}, valid_bins={}",
        q1_baseline_snr, q1_rms, q1_valid_bins
    );

    // ---- Injection sweep ----
    let alpha_zd_levels = [0.001, 0.004, 0.01];

    struct InjectionResult {
        alpha_zd: f64,
        quartile: &'static str,
        config_label: &'static str,
        baseline_snr: f64,
        injected_snr: f64,
        delta_snr: f64,
        n_valid_bins: usize,
        per_mode_snr_baseline: Vec<f64>,
        per_mode_snr_injected: Vec<f64>,
        bootstrap_delta_snr_median: f64,
        bootstrap_delta_snr_lo: f64,
        bootstrap_delta_snr_hi: f64,
    }

    let mut results: Vec<InjectionResult> = Vec::new();

    for &alpha_zd in &alpha_zd_levels {
        // Q3 injection
        let q3_injected = inject_zd_signal(&q3_galaxies, alpha_zd, cli.cd_dim);
        let (q3_inj_snr, _, q3_inj_bins) = compute_snr(&q3_injected, &config, &wavenumbers);
        let q3_delta = q3_inj_snr - q3_baseline_snr;

        let q3_mode_base = compute_per_mode_snr(&q3_galaxies, &config, &wavenumbers);
        let q3_mode_inj = compute_per_mode_snr(&q3_injected, &config, &wavenumbers);

        eprintln!(
            "\nQ3 alpha_zd={:.3}: SNR={:.4}, delta_SNR={:.4}, bins={}",
            alpha_zd, q3_inj_snr, q3_delta, q3_inj_bins
        );

        // Bootstrap Q3
        let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);
        let mut boot_deltas: Vec<f64> = Vec::with_capacity(cli.n_bootstrap);
        let mut boot_success = 0usize;

        for _ in 0..cli.n_bootstrap {
            let sample = bootstrap_sample(&q3_galaxies, &mut rng);
            let (base_snr, base_rms, base_bins) = compute_snr(&sample, &config, &wavenumbers);
            if base_rms < 1e-10 || base_bins < 3 {
                continue;
            }
            let injected = inject_zd_signal(&sample, alpha_zd, cli.cd_dim);
            let (inj_snr, _, _) = compute_snr(&injected, &config, &wavenumbers);
            boot_deltas.push(inj_snr - base_snr);
            boot_success += 1;
        }

        boot_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (median, lo, hi) = if boot_deltas.len() >= 10 {
            let lo_idx = (boot_deltas.len() as f64 * 0.025) as usize;
            let hi_idx = (boot_deltas.len() as f64 * 0.975).min(boot_deltas.len() as f64 - 1.0)
                as usize;
            let med_idx = boot_deltas.len() / 2;
            (boot_deltas[med_idx], boot_deltas[lo_idx], boot_deltas[hi_idx])
        } else {
            (q3_delta, q3_delta, q3_delta)
        };

        eprintln!(
            "  Bootstrap ({}/{}): median={:.4}, 95% CI=[{:.4}, {:.4}]",
            boot_success, cli.n_bootstrap, median, lo, hi
        );

        results.push(InjectionResult {
            alpha_zd,
            quartile: "Q3",
            config_label: "standard",
            baseline_snr: q3_baseline_snr,
            injected_snr: q3_inj_snr,
            delta_snr: q3_delta,
            n_valid_bins: q3_inj_bins,
            per_mode_snr_baseline: q3_mode_base.clone(),
            per_mode_snr_injected: q3_mode_inj,
            bootstrap_delta_snr_median: median,
            bootstrap_delta_snr_lo: lo,
            bootstrap_delta_snr_hi: hi,
        });

        // Q1 control injection
        let q1_injected = inject_zd_signal(&q1_galaxies, alpha_zd, cli.cd_dim);
        let (q1_inj_snr, _, q1_inj_bins) = compute_snr(&q1_injected, &config, &wavenumbers);
        let q1_delta = q1_inj_snr - q1_baseline_snr;

        let q1_mode_base = compute_per_mode_snr(&q1_galaxies, &config, &wavenumbers);
        let q1_mode_inj = compute_per_mode_snr(&q1_injected, &config, &wavenumbers);

        eprintln!(
            "Q1 alpha_zd={:.3}: SNR={:.4}, delta_SNR={:.4}, bins={}",
            alpha_zd, q1_inj_snr, q1_delta, q1_inj_bins
        );

        // Bootstrap Q1
        let mut rng_q1 = ChaCha8Rng::seed_from_u64(cli.seed + 1000);
        let mut boot_deltas_q1: Vec<f64> = Vec::with_capacity(cli.n_bootstrap);

        for _ in 0..cli.n_bootstrap {
            let sample = bootstrap_sample(&q1_galaxies, &mut rng_q1);
            let (base_snr, base_rms, base_bins) = compute_snr(&sample, &config, &wavenumbers);
            if base_rms < 1e-10 || base_bins < 3 {
                continue;
            }
            let injected = inject_zd_signal(&sample, alpha_zd, cli.cd_dim);
            let (inj_snr, _, _) = compute_snr(&injected, &config, &wavenumbers);
            boot_deltas_q1.push(inj_snr - base_snr);
        }

        boot_deltas_q1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (med_q1, lo_q1, hi_q1) = if boot_deltas_q1.len() >= 10 {
            let lo_idx = (boot_deltas_q1.len() as f64 * 0.025) as usize;
            let hi_idx = (boot_deltas_q1.len() as f64 * 0.975)
                .min(boot_deltas_q1.len() as f64 - 1.0) as usize;
            let med_idx = boot_deltas_q1.len() / 2;
            (
                boot_deltas_q1[med_idx],
                boot_deltas_q1[lo_idx],
                boot_deltas_q1[hi_idx],
            )
        } else {
            (q1_delta, q1_delta, q1_delta)
        };

        results.push(InjectionResult {
            alpha_zd,
            quartile: "Q1",
            config_label: "standard",
            baseline_snr: q1_baseline_snr,
            injected_snr: q1_inj_snr,
            delta_snr: q1_delta,
            n_valid_bins: q1_inj_bins,
            per_mode_snr_baseline: q1_mode_base,
            per_mode_snr_injected: q1_mode_inj,
            bootstrap_delta_snr_median: med_q1,
            bootstrap_delta_snr_lo: lo_q1,
            bootstrap_delta_snr_hi: hi_q1,
        });

        // Synthetic-6-bin ablation: Q1 galaxies with x_max=0.74
        let (q1_6bin_baseline_snr, _, q1_6bin_valid) =
            compute_snr(&q1_galaxies, &config_6bin, &wavenumbers);
        let q1_6bin_injected = inject_zd_signal(&q1_galaxies, alpha_zd, cli.cd_dim);
        let (q1_6bin_inj_snr, _, _) = compute_snr(&q1_6bin_injected, &config_6bin, &wavenumbers);
        let q1_6bin_delta = q1_6bin_inj_snr - q1_6bin_baseline_snr;

        let q1_6bin_mode_base = compute_per_mode_snr(&q1_galaxies, &config_6bin, &wavenumbers);
        let q1_6bin_mode_inj =
            compute_per_mode_snr(&q1_6bin_injected, &config_6bin, &wavenumbers);

        eprintln!(
            "Q1-6bin alpha_zd={:.3}: SNR={:.4}, delta_SNR={:.4}, bins={}",
            alpha_zd, q1_6bin_inj_snr, q1_6bin_delta, q1_6bin_valid
        );

        results.push(InjectionResult {
            alpha_zd,
            quartile: "Q1",
            config_label: "synthetic_6bin",
            baseline_snr: q1_6bin_baseline_snr,
            injected_snr: q1_6bin_inj_snr,
            delta_snr: q1_6bin_delta,
            n_valid_bins: q1_6bin_valid,
            per_mode_snr_baseline: q1_6bin_mode_base,
            per_mode_snr_injected: q1_6bin_mode_inj,
            bootstrap_delta_snr_median: q1_6bin_delta,
            bootstrap_delta_snr_lo: q1_6bin_delta,
            bootstrap_delta_snr_hi: q1_6bin_delta,
        });
    }

    // ---- Verdict ----
    eprintln!("\n--- F2 Verdict ---");
    let q3_004 = results
        .iter()
        .find(|r| r.quartile == "Q3" && (r.alpha_zd - 0.004).abs() < 1e-6);
    if let Some(r) = q3_004 {
        if r.delta_snr > 2.0 {
            eprintln!(
                "DETECTION: Q3 injection at 0.004 produces delta_SNR={:.4} > 2.0",
                r.delta_snr
            );
            eprintln!("  Pipeline CAN detect signals at 6 bins; Q3 SNR=3.60 may be genuine");
        } else if r.delta_snr < 0.5 {
            eprintln!(
                "ARTIFACT: Q3 injection at 0.004 produces delta_SNR={:.4} < 0.5",
                r.delta_snr
            );
            eprintln!("  Pipeline is BLIND at 6 bins; Q3 SNR=3.60 is a sparse-bin artifact");
        } else {
            eprintln!(
                "INCONCLUSIVE: Q3 injection at 0.004 produces delta_SNR={:.4} (0.5--2.0)",
                r.delta_snr
            );
        }
    }

    // ---- Write CSV ----
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "quartile",
        "config",
        "alpha_zd",
        "baseline_snr",
        "injected_snr",
        "delta_snr",
        "n_valid_bins",
        "boot_median",
        "boot_lo_95",
        "boot_hi_95",
        "mode1_base",
        "mode2_base",
        "mode3_base",
        "mode4_base",
        "mode5_base",
        "mode6_base",
        "mode7_base",
        "mode1_inj",
        "mode2_inj",
        "mode3_inj",
        "mode4_inj",
        "mode5_inj",
        "mode6_inj",
        "mode7_inj",
    ])?;

    for r in &results {
        let mut record: Vec<String> = vec![
            r.quartile.to_string(),
            r.config_label.to_string(),
            format!("{:.6}", r.alpha_zd),
            format!("{:.6}", r.baseline_snr),
            format!("{:.6}", r.injected_snr),
            format!("{:.6}", r.delta_snr),
            format!("{}", r.n_valid_bins),
            format!("{:.6}", r.bootstrap_delta_snr_median),
            format!("{:.6}", r.bootstrap_delta_snr_lo),
            format!("{:.6}", r.bootstrap_delta_snr_hi),
        ];
        for i in 0..7 {
            record.push(format!(
                "{:.6e}",
                r.per_mode_snr_baseline.get(i).copied().unwrap_or(0.0)
            ));
        }
        for i in 0..7 {
            record.push(format!(
                "{:.6e}",
                r.per_mode_snr_injected.get(i).copied().unwrap_or(0.0)
            ));
        }
        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    eprintln!("\nResults written to {:?}", cli.out_csv);

    Ok(())
}
