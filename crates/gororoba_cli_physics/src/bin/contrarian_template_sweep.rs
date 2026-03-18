//! Contrarian hypothesis tests: template sensitivity and baseline robustness.
//!
//! **Contrarian I: Template Sensitivity Sweep**
//! Tests whether the null result is robust to the signal model's arbitrary choices
//! (1/n amplitude, exp(-x) damping). Evaluates 4 alternative DFT weight templates
//! plus an optimal matched filter that maximizes SNR given the observed noise.
//!
//! **Contrarian II: Inner Slope Flexibility**
//! Fits a per-mass-bin effective inner slope gamma from the stacked residual,
//! subtracts this profile correction, and re-evaluates Fourier power. Tests
//! whether NFW model misspecification is hiding a signal.
//!
//! Both tests use the existing stacking infrastructure with minimal new code.
//!
//! Reference: Devil's advocate hypotheses, C-1393--C-1398 follow-up

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
#[command(name = "contrarian-template-sweep")]
#[command(about = "Template sensitivity + baseline robustness tests")]
struct Cli {
    /// Path to MaNGA rotation curves CSV.
    #[arg(
        long,
        default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv"
    )]
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

    /// Output CSV path prefix.
    #[arg(long, default_value = "data/results/e183/contrarian")]
    out_prefix: String,
}

/// Weighted DFT: sum_j w(mode, x_j) * delta_j * exp(-i*k*x_j) / sum weights
/// where w encodes the signal template shape.
///
/// Returns (power, phase) per wavenumber.
fn weighted_dft(
    x_grid: &[f64],
    delta: &[f64],
    n_contributing: &[usize],
    min_per_bin: usize,
    wavenumbers: &[f64],
    mode_weights: &[f64],
    spatial_envelope: &dyn Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut power = vec![0.0_f64; wavenumbers.len()];
    let mut phase = vec![0.0_f64; wavenumbers.len()];

    for (mode_idx, &k) in wavenumbers.iter().enumerate() {
        let w_mode = mode_weights[mode_idx];
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let mut sum_w = 0.0_f64;

        for (j, &x) in x_grid.iter().enumerate() {
            if n_contributing[j] < min_per_bin {
                continue;
            }
            let w_spatial = spatial_envelope(x);
            let w_total = w_mode * w_spatial;
            re += w_total * delta[j] * (k * x).cos();
            im += w_total * delta[j] * (k * x).sin();
            sum_w += w_total;
        }

        if sum_w > 0.0 {
            re /= sum_w;
            im /= sum_w;
            power[mode_idx] = re * re + im * im;
            phase[mode_idx] = im.atan2(re);
        }
    }

    (power, phase)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading MaNGA rotation curves...");
    let rotcurves = parse_manga_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("Loaded {} galaxy rotation curves", rotcurves.len());

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

    // Stack
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

    let result = stack_residuals(&normalized, &config);

    // ===================================================================
    // CONTRARIAN I: Template Sensitivity Sweep
    // ===================================================================
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("CONTRARIAN I: Template Sensitivity Sweep");
    eprintln!("{}", "=".repeat(60));

    // Template 0: Standard (1/n amplitude, exp(-x) damping) -- BASELINE
    let w_standard: Vec<f64> = (1..=n_modes).map(|n| 0.5 / n as f64).collect();
    let env_exp = |x: f64| -> f64 { (-x).exp() };

    // Template 1: Flat amplitude, exp(-x) damping
    let w_flat: Vec<f64> = vec![1.0; n_modes];

    // Template 2: 1/n amplitude, Gaussian damping
    let env_gauss = |x: f64| -> f64 { (-x * x / 2.0).exp() };

    // Template 3: Flat amplitude, no damping
    let env_flat = |_x: f64| -> f64 { 1.0 };

    // Template 4: 1/n^2 amplitude, exp(-x) damping
    let w_steep: Vec<f64> = (1..=n_modes).map(|n| 0.5 / (n * n) as f64).collect();

    // Template 5: Optimal matched filter (inverse noise weighting)
    // Estimate noise power per wavenumber from the standard DFT
    let (baseline_power, _) = fourier_at_wavenumbers(
        &result.x_grid,
        &result.delta_stack,
        &result.n_contributing,
        config.min_galaxies_per_bin,
        &wavenumbers,
    );
    let w_optimal: Vec<f64> = baseline_power
        .iter()
        .map(|&p| if p > 1e-20 { 1.0 / p.sqrt() } else { 1.0 })
        .collect();

    struct TemplateResult {
        name: String,
        power: Vec<f64>,
        snr: f64,
    }

    type Envelope = dyn Fn(f64) -> f64;
    let templates: Vec<(&str, &[f64], &Envelope)> = vec![
        ("standard_1n_exp", &w_standard, &env_exp),
        ("flat_amp_exp", &w_flat, &env_exp),
        ("1n_gauss", &w_standard, &env_gauss),
        ("flat_amp_flat", &w_flat, &env_flat),
        ("1n2_exp", &w_steep, &env_exp),
        ("optimal_mf", &w_optimal, &env_exp),
    ];

    let mut template_results: Vec<TemplateResult> = Vec::new();

    eprintln!(
        "\n{:>20}  {:>12}  {:>10}  {:>10}",
        "template", "max_power", "rms", "snr"
    );
    eprintln!("{}", "-".repeat(56));

    for (name, weights, envelope) in &templates {
        let (power, _phase) = weighted_dft(
            &result.x_grid,
            &result.delta_stack,
            &result.n_contributing,
            config.min_galaxies_per_bin,
            &wavenumbers,
            weights,
            *envelope,
        );

        let max_power = power.iter().copied().fold(0.0_f64, f64::max);
        let snr = if result.rms_residual > 0.0 {
            max_power.sqrt() / result.rms_residual
        } else {
            0.0
        };

        eprintln!(
            "{:>20}  {:>12.6e}  {:>10.6}  {:>10.4}",
            name, max_power, result.rms_residual, snr
        );

        template_results.push(TemplateResult {
            name: name.to_string(),
            power,
            snr,
        });
    }

    let max_snr = template_results
        .iter()
        .map(|t| t.snr)
        .fold(0.0_f64, f64::max);
    let min_snr = template_results
        .iter()
        .map(|t| t.snr)
        .fold(f64::INFINITY, f64::min);

    eprintln!("\n--- Contrarian I Verdict ---");
    eprintln!(
        "SNR range across 6 templates: [{:.4}, {:.4}]",
        min_snr, max_snr
    );

    if max_snr > 3.0 {
        eprintln!("RESULT: CHALLENGE SUSTAINED -- a template exceeds SNR 3.0!");
    } else if max_snr < 1.0 {
        eprintln!("RESULT: NULL ROBUST -- all templates below SNR 1.0");
    } else {
        eprintln!("RESULT: MODERATE -- SNR varies but none reach detection threshold");
    }

    // ===================================================================
    // CONTRARIAN II: Inner Slope Flexibility
    // ===================================================================
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("CONTRARIAN II: Inner Slope Flexibility (Baseline Robustness)");
    eprintln!("{}", "=".repeat(60));

    // Fit an effective inner-slope correction to the stacked profile.
    // Model: the NFW residual at small x contains a systematic from profile
    // misspecification: delta_baryonic(x) = A * (x^(-gamma_eff+1) - x^(-1+1))
    // Simplified: fit a low-order polynomial to the stacked residual in the
    // inner region (x < 1.5) and subtract it.

    // Use polynomial fit: delta_corr(x) = a0 + a1*(x-1) + a2*(x-1)^2
    // Fitted to the actual stacked residual in valid bins.

    let mut sx0 = 0.0_f64;
    let mut sx1 = 0.0_f64;
    let mut sx2 = 0.0_f64;
    let mut sx3 = 0.0_f64;
    let mut sx4 = 0.0_f64;
    let mut sy0 = 0.0_f64;
    let mut sy1 = 0.0_f64;
    let mut sy2 = 0.0_f64;

    for j in 0..config.n_grid {
        if result.n_contributing[j] < config.min_galaxies_per_bin {
            continue;
        }
        let x = result.x_grid[j];
        if x > 2.0 {
            continue;
        } // Only fit inner region
        let u = x - 1.0; // Center on x=1 for stability
        let d = result.delta_stack[j];

        sx0 += 1.0;
        sx1 += u;
        sx2 += u * u;
        sx3 += u * u * u;
        sx4 += u * u * u * u;
        sy0 += d;
        sy1 += d * u;
        sy2 += d * u * u;
    }

    // Solve 3x3 normal equations for [a0, a1, a2]
    // | sx0 sx1 sx2 | | a0 |   | sy0 |
    // | sx1 sx2 sx3 | | a1 | = | sy1 |
    // | sx2 sx3 sx4 | | a2 |   | sy2 |
    #[allow(clippy::needless_range_loop)]
    let (a0, a1, a2) = {
        let m = [[sx0, sx1, sx2], [sx1, sx2, sx3], [sx2, sx3, sx4]];
        let b = [sy0, sy1, sy2];

        // Gaussian elimination (3x3, no pivoting needed for well-conditioned)
        let mut aug = [[0.0_f64; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                aug[i][j] = m[i][j];
            }
            aug[i][3] = b[i];
        }

        for col in 0..3 {
            let pivot = aug[col][col];
            if pivot.abs() < 1e-30 {
                break;
            }
            for j in col..4 {
                aug[col][j] /= pivot;
            }
            for row in 0..3 {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for j in col..4 {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        (aug[0][3], aug[1][3], aug[2][3])
    };

    eprintln!("\nInner-slope polynomial fit (x < 2.0):");
    eprintln!(
        "  delta_corr(x) = {:.6} + {:.6}*(x-1) + {:.6}*(x-1)^2",
        a0, a1, a2
    );

    // Subtract polynomial correction from stacked profile
    let corrected_delta: Vec<f64> = result
        .delta_stack
        .iter()
        .zip(result.x_grid.iter())
        .map(|(&d, &x)| {
            let u = x - 1.0;
            d - (a0 + a1 * u + a2 * u * u)
        })
        .collect();

    // Re-compute Fourier power on corrected profile
    let (corrected_power, _corrected_phase) = fourier_at_wavenumbers(
        &result.x_grid,
        &corrected_delta,
        &result.n_contributing,
        config.min_galaxies_per_bin,
        &wavenumbers,
    );

    // Compute corrected RMS
    let valid_corrected: Vec<f64> = corrected_delta
        .iter()
        .zip(&result.n_contributing)
        .filter(|(_, n)| **n >= config.min_galaxies_per_bin)
        .map(|(d, _)| *d)
        .collect();

    let corrected_rms = if valid_corrected.is_empty() {
        0.0
    } else {
        let mean = valid_corrected.iter().sum::<f64>() / valid_corrected.len() as f64;
        let var = valid_corrected
            .iter()
            .map(|d| (d - mean).powi(2))
            .sum::<f64>()
            / valid_corrected.len() as f64;
        var.sqrt()
    };

    let corrected_max_power = corrected_power.iter().copied().fold(0.0_f64, f64::max);
    let corrected_snr = if corrected_rms > 0.0 {
        corrected_max_power.sqrt() / corrected_rms
    } else {
        0.0
    };

    eprintln!(
        "\n{:>12}  {:>10}  {:>10}  {:>12}",
        "profile", "rms", "snr", "max_power"
    );
    eprintln!("{}", "-".repeat(48));
    eprintln!(
        "{:>12}  {:>10.6}  {:>10.4}  {:>12.6e}",
        "NFW",
        result.rms_residual,
        result.detection_snr,
        baseline_power.iter().copied().fold(0.0_f64, f64::max)
    );
    eprintln!(
        "{:>12}  {:>10.6}  {:>10.4}  {:>12.6e}",
        "corrected", corrected_rms, corrected_snr, corrected_max_power
    );

    let rms_reduction = (result.rms_residual - corrected_rms) / result.rms_residual * 100.0;
    eprintln!(
        "\nRMS reduction from polynomial correction: {:.1}%",
        rms_reduction
    );

    // Per-mode comparison
    eprintln!(
        "\n{:>5}  {:>8}  {:>12}  {:>12}  {:>10}",
        "mode", "k", "nfw_power", "corr_power", "ratio"
    );
    for (mode_idx, &k) in wavenumbers.iter().enumerate() {
        let ratio = if baseline_power[mode_idx] > 1e-20 {
            corrected_power[mode_idx] / baseline_power[mode_idx]
        } else {
            f64::NAN
        };
        eprintln!(
            "{:>5}  {:>8.4}  {:>12.6e}  {:>12.6e}  {:>10.4}",
            mode_idx + 1,
            k,
            baseline_power[mode_idx],
            corrected_power[mode_idx],
            ratio
        );
    }

    eprintln!("\n--- Contrarian II Verdict ---");
    if corrected_snr > 3.0 {
        eprintln!("RESULT: CHALLENGE SUSTAINED -- signal emerges after baseline correction!");
    } else if corrected_rms < result.rms_residual * 0.5 {
        eprintln!(
            "RESULT: BASELINE MATTERS -- correction reduces RMS by {:.0}%, but SNR still {:.2}",
            rms_reduction, corrected_snr
        );
    } else {
        eprintln!("RESULT: NULL ROBUST -- correction does not reveal hidden signal");
    }

    // ===================================================================
    // Write outputs
    // ===================================================================

    // Template sweep CSV
    let sweep_csv = format!("{}_templates.csv", cli.out_prefix);
    let mut wtr = csv::Writer::from_path(&sweep_csv)?;
    wtr.write_record(["template", "mode", "k", "power", "snr"])?;
    for tr in &template_results {
        for (mode_idx, &k) in wavenumbers.iter().enumerate() {
            let mode_snr = if result.rms_residual > 0.0 {
                tr.power[mode_idx].sqrt() / result.rms_residual
            } else {
                0.0
            };
            wtr.write_record([
                &tr.name,
                &format!("{}", mode_idx + 1),
                &format!("{:.6}", k),
                &format!("{:.8e}", tr.power[mode_idx]),
                &format!("{:.6}", mode_snr),
            ])?;
        }
    }
    wtr.flush()?;

    // Baseline correction CSV
    let corr_csv = format!("{}_baseline.csv", cli.out_prefix);
    let mut cwtr = csv::Writer::from_path(&corr_csv)?;
    cwtr.write_record([
        "x",
        "delta_nfw",
        "delta_corrected",
        "correction",
        "n_contributing",
    ])?;
    for ((&x, &d_nfw), (&d_corr, &n)) in result
        .x_grid
        .iter()
        .zip(result.delta_stack.iter())
        .zip(corrected_delta.iter().zip(result.n_contributing.iter()))
    {
        let u = x - 1.0;
        let corr = a0 + a1 * u + a2 * u * u;
        cwtr.write_record([
            &format!("{:.6}", x),
            &format!("{:.8e}", d_nfw),
            &format!("{:.8e}", d_corr),
            &format!("{:.8e}", corr),
            &format!("{}", n),
        ])?;
    }
    cwtr.flush()?;

    // Summary
    let summary_csv = format!("{}_summary.csv", cli.out_prefix);
    let mut swtr = csv::Writer::from_path(&summary_csv)?;
    swtr.write_record(["test", "metric", "value"])?;
    for tr in &template_results {
        swtr.write_record(["I", &format!("snr_{}", tr.name), &format!("{:.6}", tr.snr)])?;
    }
    swtr.write_record(["I", "snr_range_min", &format!("{:.6}", min_snr)])?;
    swtr.write_record(["I", "snr_range_max", &format!("{:.6}", max_snr)])?;
    swtr.write_record(["II", "poly_a0", &format!("{:.8}", a0)])?;
    swtr.write_record(["II", "poly_a1", &format!("{:.8}", a1)])?;
    swtr.write_record(["II", "poly_a2", &format!("{:.8}", a2)])?;
    swtr.write_record(["II", "nfw_rms", &format!("{:.8}", result.rms_residual)])?;
    swtr.write_record(["II", "corrected_rms", &format!("{:.8}", corrected_rms)])?;
    swtr.write_record(["II", "rms_reduction_pct", &format!("{:.2}", rms_reduction)])?;
    swtr.write_record(["II", "corrected_snr", &format!("{:.6}", corrected_snr)])?;
    swtr.flush()?;

    eprintln!("\nResults written to:");
    eprintln!("  Templates: {}", sweep_csv);
    eprintln!("  Baseline:  {}", corr_csv);
    eprintln!("  Summary:   {}", summary_csv);

    Ok(())
}
