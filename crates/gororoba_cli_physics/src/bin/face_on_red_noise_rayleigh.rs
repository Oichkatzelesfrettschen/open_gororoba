//! F1: Face-on galaxy reanalysis with red-noise baseline subtraction.
//!
//! Restricts to face-on galaxies (i < 45 deg, N=3139) to minimize projection
//! artifacts, then subtracts the R(k) ~ k^gamma red-noise envelope from the
//! Rayleigh R test before computing a corrected alpha_zd upper limit.
//!
//! The red-noise spectral index gamma=0.808 was measured in E-183 via
//! multifeature-shape-regression (C-1410).  Previous analyses reported raw
//! Rayleigh R values (0.97--0.99) that appear highly coherent but are entirely
//! explained by the k-dependent baryonic power spectrum.
//!
//! Method:
//! 1. Stack face-on galaxies, compute Rayleigh R at 7 CD-ZD modes
//! 2. Fit red-noise envelope R(k) = A * k^gamma via log-log OLS
//! 3. Compute residual R_resid(k) = R_obs(k) - R_pred(k)
//! 4. Detection: any |R_resid| > 3*sigma_resid
//! 5. Bootstrap 200x for alpha_zd upper limit CI
//!
//! Detection: Any mode shows residual R > 3*sigma above envelope
//! Failure: All modes consistent with envelope; corrected alpha_zd < 0.0015
//!
//! Reference: E-201

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        jackknife_phase_coherence, stack_residuals,
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
#[command(name = "face-on-red-noise-rayleigh")]
#[command(about = "F1: Face-on Rayleigh test with red-noise baseline subtraction (E-201)")]
struct Cli {
    #[arg(long, default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv")]
    rotcurves: PathBuf,

    #[arg(long, default_value = "data/external/manga/subsamples/dapall_lowi.csv")]
    dapall: PathBuf,

    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    #[arg(long, default_value_t = 8)]
    min_points: usize,

    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    #[arg(long, default_value_t = 10)]
    min_per_bin: usize,

    /// Number of bootstrap resamples.
    #[arg(long, default_value_t = 200)]
    n_bootstrap: usize,

    /// Random seed for bootstrap.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Fix gamma to this value (0 = free fit).
    #[arg(long, default_value_t = 0.808)]
    gamma_prior: f64,

    #[arg(long, default_value = "data/results/e201/face_on_red_noise_rayleigh.csv")]
    out_csv: PathBuf,
}

/// Log-log linear regression: fit log(y) = a + b * log(x).
/// Returns (slope, intercept).
fn log_log_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let mut lx = Vec::new();
    let mut ly = Vec::new();

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if xi > 0.0 && yi > 0.0 {
            lx.push(xi.ln());
            ly.push(yi.ln());
        }
    }

    let n = lx.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0);
    }

    let mean_lx = lx.iter().sum::<f64>() / n;
    let mean_ly = ly.iter().sum::<f64>() / n;

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (xi, yi) in lx.iter().zip(ly.iter()) {
        let dx = xi - mean_lx;
        let dy = yi - mean_ly;
        num += dx * dy;
        den += dx * dx;
    }

    if den.abs() < 1e-30 {
        return (0.0, mean_ly);
    }

    let slope = num / den;
    let intercept = mean_ly - slope * mean_lx;
    (slope, intercept)
}

/// Fit red-noise envelope and compute residuals + corrected alpha_zd.
struct RedNoiseResult {
    gamma: f64,
    amplitude: f64,
    rayleigh_r_obs: Vec<f64>,
    rayleigh_r_pred: Vec<f64>,
    residuals: Vec<f64>,
    sigma_resid: f64,
    max_residual_sigma: f64,
    corrected_alpha_zd: f64,
}

fn red_noise_analysis(
    normalized: &[NormalizedResiduals],
    config: &StackingConfig,
    wavenumbers: &[f64],
    gamma_prior: f64,
) -> RedNoiseResult {
    let result = stack_residuals(normalized, config);
    let rayleigh = jackknife_phase_coherence(
        &result.x_grid,
        &result.delta_stack,
        &result.n_contributing,
        config.min_galaxies_per_bin,
        wavenumbers,
    );

    let rayleigh_r_obs = rayleigh.rayleigh_r.clone();

    // Fit or use prior for gamma
    let (gamma, log_a) = if gamma_prior > 0.0 {
        // Fixed gamma: fit only amplitude via log(R) = log(A) + gamma*log(k)
        let mut sum_log_r = 0.0_f64;
        let mut sum_log_k = 0.0_f64;
        let mut count = 0usize;
        for (&k, &r) in wavenumbers.iter().zip(rayleigh_r_obs.iter()) {
            if k > 0.0 && r > 0.0 {
                sum_log_r += r.ln();
                sum_log_k += k.ln();
                count += 1;
            }
        }
        let log_a = if count > 0 {
            (sum_log_r - gamma_prior * sum_log_k) / count as f64
        } else {
            0.0
        };
        (gamma_prior, log_a)
    } else {
        // Free fit
        let (slope, intercept) = log_log_fit(wavenumbers, &rayleigh_r_obs);
        (slope, intercept)
    };

    let amplitude = log_a.exp();

    // Predicted R and residuals
    let rayleigh_r_pred: Vec<f64> = wavenumbers
        .iter()
        .map(|&k| amplitude * k.powf(gamma))
        .collect();

    let residuals: Vec<f64> = rayleigh_r_obs
        .iter()
        .zip(rayleigh_r_pred.iter())
        .map(|(&obs, &pred)| obs - pred)
        .collect();

    let mean_resid = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let sigma_resid = if residuals.len() > 1 {
        let var = residuals
            .iter()
            .map(|r| (r - mean_resid).powi(2))
            .sum::<f64>()
            / (residuals.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    let max_residual_sigma = if sigma_resid > 0.0 {
        residuals
            .iter()
            .map(|r| r.abs() / sigma_resid)
            .fold(0.0_f64, f64::max)
    } else {
        0.0
    };

    // Corrected alpha_zd from residual Fourier power
    let af = config.cd_params.assessor_fraction;
    let max_resid_power = residuals
        .iter()
        .map(|r| r.powi(2))
        .fold(0.0_f64, f64::max);
    // alpha_zd ~ sqrt(max_resid_power) * rms / (af * exp(-x_peak))
    // x_peak ~ 0.6 (where most galaxies contribute)
    let corrected_alpha_zd =
        max_resid_power.sqrt() * result.rms_residual / (af * (-0.6_f64).exp());

    RedNoiseResult {
        gamma,
        amplitude,
        rayleigh_r_obs,
        rayleigh_r_pred,
        residuals,
        sigma_resid,
        max_residual_sigma,
        corrected_alpha_zd,
    }
}

/// Bootstrap resample galaxies with replacement.
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

        let log_m200 = meta.estimated_log_m200();
        if !log_m200.is_finite() || log_m200 < 10.0 {
            skipped += 1;
            continue;
        }
        let m200 = 10.0_f64.powf(log_m200);
        let nfw = nfw_params_from_mass(m200, meta.z);
        if nfw.r_s_kpc <= 0.0 || nfw.c200 <= 0.0 {
            skipped += 1;
            continue;
        }

        let points: Vec<NormalizedPoint> = galaxy
            .points
            .iter()
            .filter_map(|pt| {
                if pt.r_kpc <= 0.0 || nfw.r_s_kpc <= 0.0 {
                    return None;
                }
                let x = pt.r_kpc / nfw.r_s_kpc;
                if !(0.01..=20.0).contains(&x) {
                    return None;
                }
                let m_enc = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw);
                let v_nfw = (G_KPC_KMS2 * m_enc / pt.r_kpc).max(0.0).sqrt();
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
                r_s_kpc: nfw.r_s_kpc,
                points,
            });
        } else {
            skipped += 1;
        }
    }

    eprintln!(
        "Normalized {} face-on galaxies ({} skipped)",
        normalized.len(),
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
        min_galaxies_per_bin: cli.min_per_bin,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: false,
    };

    // ---- Main analysis (fixed gamma) ----
    eprintln!("\n--- Red-noise analysis (gamma={:.3} prior) ---", cli.gamma_prior);
    let main_result = red_noise_analysis(&normalized, &config, &wavenumbers, cli.gamma_prior);

    eprintln!("Fitted: gamma={:.4}, A={:.6}", main_result.gamma, main_result.amplitude);
    for (i, ((&obs, &pred), &resid)) in main_result
        .rayleigh_r_obs
        .iter()
        .zip(main_result.rayleigh_r_pred.iter())
        .zip(main_result.residuals.iter())
        .enumerate()
    {
        eprintln!(
            "  mode {}: R_obs={:.6}, R_pred={:.6}, resid={:.6} ({:.2} sigma)",
            i + 1,
            obs,
            pred,
            resid,
            if main_result.sigma_resid > 0.0 {
                resid.abs() / main_result.sigma_resid
            } else {
                0.0
            }
        );
    }
    eprintln!(
        "sigma_resid={:.6}, max_residual={:.2} sigma",
        main_result.sigma_resid, main_result.max_residual_sigma
    );
    eprintln!(
        "Corrected alpha_zd={:.6}",
        main_result.corrected_alpha_zd
    );

    // ---- Free-fit ablation ----
    eprintln!("\n--- Ablation: free gamma fit ---");
    let free_result = red_noise_analysis(&normalized, &config, &wavenumbers, 0.0);
    eprintln!(
        "Free fit: gamma={:.4}, A={:.6}, corrected alpha_zd={:.6}",
        free_result.gamma, free_result.amplitude, free_result.corrected_alpha_zd
    );

    // ---- Bootstrap ----
    eprintln!("\n--- Bootstrap ({} resamples) ---", cli.n_bootstrap);
    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);
    let mut boot_alpha_zd: Vec<f64> = Vec::with_capacity(cli.n_bootstrap);
    let mut boot_max_sigma: Vec<f64> = Vec::with_capacity(cli.n_bootstrap);
    let mut boot_success = 0usize;

    for _ in 0..cli.n_bootstrap {
        let sample = bootstrap_sample(&normalized, &mut rng);
        let result = red_noise_analysis(&sample, &config, &wavenumbers, cli.gamma_prior);
        if result.sigma_resid > 0.0 && result.corrected_alpha_zd.is_finite() {
            boot_alpha_zd.push(result.corrected_alpha_zd);
            boot_max_sigma.push(result.max_residual_sigma);
            boot_success += 1;
        }
    }

    boot_alpha_zd.sort_by(|a, b| a.partial_cmp(b).unwrap());
    boot_max_sigma.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (alpha_median, alpha_lo, alpha_hi) = if boot_alpha_zd.len() >= 10 {
        let lo = (boot_alpha_zd.len() as f64 * 0.025) as usize;
        let hi =
            (boot_alpha_zd.len() as f64 * 0.975).min(boot_alpha_zd.len() as f64 - 1.0) as usize;
        let med = boot_alpha_zd.len() / 2;
        (boot_alpha_zd[med], boot_alpha_zd[lo], boot_alpha_zd[hi])
    } else {
        let a = main_result.corrected_alpha_zd;
        (a, a, a)
    };

    eprintln!(
        "Bootstrap ({}/{}): alpha_zd median={:.6}, 95% CI=[{:.6}, {:.6}]",
        boot_success, cli.n_bootstrap, alpha_median, alpha_lo, alpha_hi
    );

    let detection_frac = boot_max_sigma
        .iter()
        .filter(|&&s| s > 3.0)
        .count() as f64
        / boot_max_sigma.len().max(1) as f64;
    eprintln!(
        "Detection fraction (max_sigma > 3.0): {:.1}%",
        detection_frac * 100.0
    );

    // ---- Verdict ----
    eprintln!("\n--- F1 Verdict ---");
    if main_result.max_residual_sigma > 3.0 {
        eprintln!(
            "DETECTION: Mode with residual at {:.2} sigma above red-noise envelope",
            main_result.max_residual_sigma
        );
    } else if alpha_hi < 0.0015 {
        eprintln!(
            "TIGHTER BOUND: Corrected alpha_zd < {:.6} (95% upper), improving E-183 limit of 0.002392",
            alpha_hi
        );
    } else {
        eprintln!(
            "CONSISTENT: All modes within {:.2} sigma of envelope; alpha_zd 95% upper = {:.6}",
            main_result.max_residual_sigma, alpha_hi
        );
    }

    // ---- Write CSV ----
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "analysis",
        "gamma",
        "amplitude",
        "sigma_resid",
        "max_residual_sigma",
        "corrected_alpha_zd",
        "boot_alpha_median",
        "boot_alpha_lo_95",
        "boot_alpha_hi_95",
        "detection_frac",
        "n_galaxies",
        "n_bootstrap_success",
        "r_obs_mode1",
        "r_obs_mode2",
        "r_obs_mode3",
        "r_obs_mode4",
        "r_obs_mode5",
        "r_obs_mode6",
        "r_obs_mode7",
        "r_pred_mode1",
        "r_pred_mode2",
        "r_pred_mode3",
        "r_pred_mode4",
        "r_pred_mode5",
        "r_pred_mode6",
        "r_pred_mode7",
        "resid_mode1",
        "resid_mode2",
        "resid_mode3",
        "resid_mode4",
        "resid_mode5",
        "resid_mode6",
        "resid_mode7",
    ])?;

    // Fixed-gamma row
    let mut record: Vec<String> = vec![
        "fixed_gamma".to_string(),
        format!("{:.6}", main_result.gamma),
        format!("{:.8}", main_result.amplitude),
        format!("{:.8}", main_result.sigma_resid),
        format!("{:.4}", main_result.max_residual_sigma),
        format!("{:.8}", main_result.corrected_alpha_zd),
        format!("{:.8}", alpha_median),
        format!("{:.8}", alpha_lo),
        format!("{:.8}", alpha_hi),
        format!("{:.4}", detection_frac),
        format!("{}", normalized.len()),
        format!("{}", boot_success),
    ];
    for r in &main_result.rayleigh_r_obs {
        record.push(format!("{:.8}", r));
    }
    for r in &main_result.rayleigh_r_pred {
        record.push(format!("{:.8}", r));
    }
    for r in &main_result.residuals {
        record.push(format!("{:.8}", r));
    }
    wtr.write_record(&record)?;

    // Free-fit row
    let mut record2: Vec<String> = vec![
        "free_gamma".to_string(),
        format!("{:.6}", free_result.gamma),
        format!("{:.8}", free_result.amplitude),
        format!("{:.8}", free_result.sigma_resid),
        format!("{:.4}", free_result.max_residual_sigma),
        format!("{:.8}", free_result.corrected_alpha_zd),
        "".to_string(),
        "".to_string(),
        "".to_string(),
        "".to_string(),
        format!("{}", normalized.len()),
        "".to_string(),
    ];
    for r in &free_result.rayleigh_r_obs {
        record2.push(format!("{:.8}", r));
    }
    for r in &free_result.rayleigh_r_pred {
        record2.push(format!("{:.8}", r));
    }
    for r in &free_result.residuals {
        record2.push(format!("{:.8}", r));
    }
    wtr.write_record(&record2)?;

    wtr.flush()?;
    eprintln!("\nResults written to {:?}", cli.out_csv);

    Ok(())
}
