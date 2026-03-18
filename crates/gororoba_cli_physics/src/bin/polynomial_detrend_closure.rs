//! Polynomial detrending closure test for MaNGA harmonic stacking null result.
//!
//! Tests whether a low-order polynomial absorbs all Fourier power at CD-ZD
//! wavenumbers because the dominant residual shape is a smooth baryonic trend,
//! not a harmonic signal. Extends C-1400 by scanning ALL wavenumbers k in
//! [k_min, k_max] (not just the 7 CD-ZD modes) to close the escape route
//! that a non-ZD peaked signal might be hidden.
//!
//! After detrending, computes a tightened alpha_zd upper limit with bootstrap
//! 95% CI from galaxy-level resampling.
//!
//! Detection: any k with SNR > 3.0 after detrending
//! Rejection: all scanned frequencies have SNR < 1.0
//!
//! Reference: Hypothesis H1, E-205

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        fourier_at_wavenumbers, stack_residuals,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::{f64::consts::PI, path::PathBuf};

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "polynomial-detrend-closure")]
#[command(about = "Broadband Fourier scan after polynomial detrending")]
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

    /// Minimum wavenumber for broadband scan.
    #[arg(long, default_value_t = 0.5)]
    k_min: f64,

    /// Maximum wavenumber for broadband scan.
    #[arg(long, default_value_t = 20.0)]
    k_max: f64,

    /// Wavenumber step for broadband scan.
    #[arg(long, default_value_t = 0.1)]
    k_step: f64,

    /// Polynomial order for detrending (2 = quadratic).
    #[arg(long, default_value_t = 2)]
    poly_order: usize,

    /// Number of bootstrap resamples for alpha_zd CI.
    #[arg(long, default_value_t = 1000)]
    n_bootstrap: usize,

    /// Output CSV path prefix.
    #[arg(long, default_value = "data/results/e205/detrend")]
    out_prefix: String,
}

/// Fit a polynomial of given order to (x, y) data using normal equations.
/// Returns coefficients [a0, a1, ..., a_order] where poly(x) = sum a_i * x^i.
fn fit_polynomial(x: &[f64], y: &[f64], order: usize) -> Vec<f64> {
    let n_coeffs = order + 1;

    // Build normal equations: A^T A c = A^T y
    // where A_ij = x_i^j
    let mut ata = vec![0.0_f64; n_coeffs * n_coeffs];
    let mut aty = vec![0.0_f64; n_coeffs];

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let mut xi_pow = vec![1.0_f64; n_coeffs];
        for j in 1..n_coeffs {
            xi_pow[j] = xi_pow[j - 1] * xi;
        }
        for j in 0..n_coeffs {
            aty[j] += xi_pow[j] * yi;
            for k in 0..n_coeffs {
                ata[j * n_coeffs + k] += xi_pow[j] * xi_pow[k];
            }
        }
    }

    // Gaussian elimination with partial pivoting
    let mut aug = vec![0.0_f64; n_coeffs * (n_coeffs + 1)];
    for i in 0..n_coeffs {
        for j in 0..n_coeffs {
            aug[i * (n_coeffs + 1) + j] = ata[i * n_coeffs + j];
        }
        aug[i * (n_coeffs + 1) + n_coeffs] = aty[i];
    }

    for col in 0..n_coeffs {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * (n_coeffs + 1) + col].abs();
        for row in (col + 1)..n_coeffs {
            let val = aug[row * (n_coeffs + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..=n_coeffs {
                aug.swap(col * (n_coeffs + 1) + j, max_row * (n_coeffs + 1) + j);
            }
        }

        let pivot = aug[col * (n_coeffs + 1) + col];
        if pivot.abs() < 1e-30 {
            continue;
        }
        for j in col..=n_coeffs {
            aug[col * (n_coeffs + 1) + j] /= pivot;
        }
        for row in 0..n_coeffs {
            if row == col {
                continue;
            }
            let factor = aug[row * (n_coeffs + 1) + col];
            for j in col..=n_coeffs {
                aug[row * (n_coeffs + 1) + j] -= factor * aug[col * (n_coeffs + 1) + j];
            }
        }
    }

    (0..n_coeffs)
        .map(|i| aug[i * (n_coeffs + 1) + n_coeffs])
        .collect()
}

/// Evaluate polynomial at a point.
fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0_f64;
    let mut x_pow = 1.0_f64;
    for &c in coeffs {
        result += c * x_pow;
        x_pow *= x;
    }
    result
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

    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let n_modes = cd_params.n_modes;

    let config = StackingConfig {
        x_min: cli.x_min,
        x_max: cli.x_max,
        n_grid: 200,
        min_galaxies_per_bin: 3,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: false,
    };

    // Stack full sample
    let result = stack_residuals(&normalized, &config);

    // CD-ZD wavenumbers for reference
    let cd_wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * PI * n as f64 / n_modes as f64)
        .collect();

    // Broadband scan wavenumbers
    let mut scan_wavenumbers: Vec<f64> = Vec::new();
    let mut k = cli.k_min;
    while k <= cli.k_max {
        scan_wavenumbers.push(k);
        k += cli.k_step;
    }

    eprintln!(
        "Broadband scan: {} wavenumbers in [{:.1}, {:.1}]",
        scan_wavenumbers.len(),
        cli.k_min,
        cli.k_max
    );

    // Fit polynomial to stacked profile (valid bins only)
    let mut fit_x: Vec<f64> = Vec::new();
    let mut fit_y: Vec<f64> = Vec::new();
    for j in 0..config.n_grid {
        if result.n_contributing[j] >= config.min_galaxies_per_bin {
            fit_x.push(result.x_grid[j]);
            fit_y.push(result.delta_stack[j]);
        }
    }

    let coeffs = fit_polynomial(&fit_x, &fit_y, cli.poly_order);

    eprintln!(
        "\nPolynomial order {} fit ({} valid bins):",
        cli.poly_order,
        fit_x.len()
    );
    for (i, &c) in coeffs.iter().enumerate() {
        eprintln!("  a{} = {:.8}", i, c);
    }

    // Subtract polynomial from stacked profile
    let corrected_delta: Vec<f64> = result
        .x_grid
        .iter()
        .map(|&x| {
            let d_idx = result
                .x_grid
                .iter()
                .position(|&xg| (xg - x).abs() < 1e-12)
                .unwrap();
            result.delta_stack[d_idx] - eval_poly(&coeffs, x)
        })
        .collect();

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

    // NFW (pre-correction) Fourier at CD-ZD and broadband
    let (nfw_cd_power, _) = fourier_at_wavenumbers(
        &result.x_grid,
        &result.delta_stack,
        &result.n_contributing,
        config.min_galaxies_per_bin,
        &cd_wavenumbers,
    );

    // Corrected Fourier at CD-ZD wavenumbers
    let (corr_cd_power, corr_cd_phase) = fourier_at_wavenumbers(
        &result.x_grid,
        &corrected_delta,
        &result.n_contributing,
        config.min_galaxies_per_bin,
        &cd_wavenumbers,
    );

    // Broadband scan on corrected profile
    let (scan_power, scan_phase) = fourier_at_wavenumbers(
        &result.x_grid,
        &corrected_delta,
        &result.n_contributing,
        config.min_galaxies_per_bin,
        &scan_wavenumbers,
    );

    // Compute SNR at each scanned wavenumber
    let scan_snr: Vec<f64> = scan_power
        .iter()
        .map(|&p| {
            if corrected_rms > 0.0 {
                p.sqrt() / corrected_rms
            } else {
                0.0
            }
        })
        .collect();

    let max_snr = scan_snr.iter().copied().fold(0.0_f64, f64::max);
    let max_snr_idx = scan_snr
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let k_at_max = scan_wavenumbers[max_snr_idx];

    eprintln!("\n--- Broadband scan results ---");
    eprintln!("NFW RMS: {:.8}", result.rms_residual);
    eprintln!("Corrected RMS: {:.8}", corrected_rms);
    eprintln!(
        "RMS reduction: {:.2}%",
        (result.rms_residual - corrected_rms) / result.rms_residual * 100.0
    );
    eprintln!("Max broadband SNR: {:.6} at k={:.4}", max_snr, k_at_max);

    // CD-ZD mode comparison
    eprintln!("\n--- CD-ZD wavenumber comparison ---");
    eprintln!(
        "{:>5}  {:>8}  {:>12}  {:>12}  {:>10}",
        "mode", "k", "nfw_power", "corr_power", "ratio"
    );
    for (mode_idx, &k) in cd_wavenumbers.iter().enumerate() {
        let ratio = if nfw_cd_power[mode_idx] > 1e-20 {
            corr_cd_power[mode_idx] / nfw_cd_power[mode_idx]
        } else {
            f64::NAN
        };
        eprintln!(
            "{:>5}  {:>8.4}  {:>12.6e}  {:>12.6e}  {:>10.6}",
            mode_idx + 1,
            k,
            nfw_cd_power[mode_idx],
            corr_cd_power[mode_idx],
            ratio
        );
    }

    // Tightened alpha_zd from corrected amplitude at mode 1
    let corr_max_cd_power = corr_cd_power.iter().copied().fold(0.0_f64, f64::max);
    let corr_cd_snr = if corrected_rms > 0.0 {
        corr_max_cd_power.sqrt() / corrected_rms
    } else {
        0.0
    };

    // alpha_zd estimate: 4 * amplitude / sqrt(N_galaxies)
    let n_gal = normalized.len() as f64;
    let alpha_zd = 4.0 * corr_max_cd_power.sqrt() / n_gal.sqrt();

    eprintln!("\nCorrected CD-ZD SNR: {:.6}", corr_cd_snr);
    eprintln!("Tightened alpha_zd: {:.8}", alpha_zd);

    // Bootstrap CI on alpha_zd
    eprintln!(
        "\nBootstrap ({} resamples) for alpha_zd 95% CI...",
        cli.n_bootstrap
    );
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut bootstrap_alpha: Vec<f64> = Vec::with_capacity(cli.n_bootstrap);

    for b in 0..cli.n_bootstrap {
        // Resample galaxy indices with replacement
        let resampled: Vec<NormalizedResiduals> = (0..normalized.len())
            .map(|_| {
                let idx = rand::Rng::gen_range(&mut rng, 0..normalized.len());
                normalized[idx].clone()
            })
            .collect();

        let boot_result = stack_residuals(&resampled, &config);

        // Fit polynomial to bootstrapped stack
        let mut bfit_x: Vec<f64> = Vec::new();
        let mut bfit_y: Vec<f64> = Vec::new();
        for j in 0..config.n_grid {
            if boot_result.n_contributing[j] >= config.min_galaxies_per_bin {
                bfit_x.push(boot_result.x_grid[j]);
                bfit_y.push(boot_result.delta_stack[j]);
            }
        }

        let bcoeffs = fit_polynomial(&bfit_x, &bfit_y, cli.poly_order);

        let bcorrected: Vec<f64> = boot_result
            .x_grid
            .iter()
            .map(|&x| {
                let d_idx = boot_result
                    .x_grid
                    .iter()
                    .position(|&xg| (xg - x).abs() < 1e-12)
                    .unwrap();
                boot_result.delta_stack[d_idx] - eval_poly(&bcoeffs, x)
            })
            .collect();

        let (bpower, _) = fourier_at_wavenumbers(
            &boot_result.x_grid,
            &bcorrected,
            &boot_result.n_contributing,
            config.min_galaxies_per_bin,
            &cd_wavenumbers,
        );

        let bmax_power = bpower.iter().copied().fold(0.0_f64, f64::max);
        let balpha = 4.0 * bmax_power.sqrt() / n_gal.sqrt();
        bootstrap_alpha.push(balpha);

        if (b + 1) % 200 == 0 {
            eprintln!("  bootstrap {}/{}", b + 1, cli.n_bootstrap);
        }
    }

    bootstrap_alpha.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lo = bootstrap_alpha[(cli.n_bootstrap as f64 * 0.025) as usize];
    let ci_hi = bootstrap_alpha[(cli.n_bootstrap as f64 * 0.975) as usize];
    let boot_median = bootstrap_alpha[cli.n_bootstrap / 2];

    eprintln!(
        "alpha_zd bootstrap: median={:.8}, 95% CI=[{:.8}, {:.8}]",
        boot_median, ci_lo, ci_hi
    );

    // Verdict
    eprintln!("\n--- Detection criteria ---");
    if max_snr > 3.0 {
        eprintln!(
            "RESULT: ESCAPE ROUTE OPEN -- broadband SNR {:.2} > 3.0 at k={:.4}",
            max_snr, k_at_max
        );
    } else if max_snr < 1.0 {
        eprintln!(
            "RESULT: NULL CONFIRMED -- all {} frequencies below SNR 1.0",
            scan_wavenumbers.len()
        );
        eprintln!(
            "Tightened alpha_zd upper limit: {:.6} (95% CI: [{:.6}, {:.6}])",
            alpha_zd, ci_lo, ci_hi
        );
    } else {
        eprintln!(
            "RESULT: MODERATE -- max broadband SNR {:.2}, no detection but above noise floor",
            max_snr
        );
    }

    // Write broadband scan CSV
    let scan_csv = format!("{}_broadband_scan.csv", cli.out_prefix);
    let mut wtr = csv::Writer::from_path(&scan_csv)?;
    wtr.write_record(["k", "power", "phase", "snr"])?;
    for (i, &k) in scan_wavenumbers.iter().enumerate() {
        wtr.write_record([
            &format!("{:.4}", k),
            &format!("{:.8e}", scan_power[i]),
            &format!("{:.6}", scan_phase[i]),
            &format!("{:.6}", scan_snr[i]),
        ])?;
    }
    wtr.flush()?;

    // Write summary CSV
    let summary_csv = format!("{}_summary.csv", cli.out_prefix);
    let mut swtr = csv::Writer::from_path(&summary_csv)?;
    swtr.write_record(["metric", "value"])?;
    swtr.write_record(["n_galaxies", &format!("{}", normalized.len())])?;
    swtr.write_record(["poly_order", &format!("{}", cli.poly_order)])?;
    for (i, &c) in coeffs.iter().enumerate() {
        swtr.write_record([&format!("poly_a{}", i), &format!("{:.8}", c)])?;
    }
    swtr.write_record(["nfw_rms", &format!("{:.8}", result.rms_residual)])?;
    swtr.write_record(["corrected_rms", &format!("{:.8}", corrected_rms)])?;
    swtr.write_record([
        "rms_reduction_pct",
        &format!(
            "{:.2}",
            (result.rms_residual - corrected_rms) / result.rms_residual * 100.0
        ),
    ])?;
    swtr.write_record(["corrected_cd_snr", &format!("{:.6}", corr_cd_snr)])?;
    swtr.write_record(["broadband_max_snr", &format!("{:.6}", max_snr)])?;
    swtr.write_record(["broadband_k_at_max", &format!("{:.4}", k_at_max)])?;
    swtr.write_record(["n_scan_wavenumbers", &format!("{}", scan_wavenumbers.len())])?;
    swtr.write_record(["alpha_zd_point", &format!("{:.8}", alpha_zd)])?;
    swtr.write_record(["alpha_zd_ci_lo", &format!("{:.8}", ci_lo)])?;
    swtr.write_record(["alpha_zd_ci_hi", &format!("{:.8}", ci_hi)])?;
    swtr.write_record(["alpha_zd_median", &format!("{:.8}", boot_median)])?;
    swtr.write_record(["n_bootstrap", &format!("{}", cli.n_bootstrap)])?;
    swtr.flush()?;

    // Write CD-ZD mode detail CSV
    let cd_csv = format!("{}_cd_modes.csv", cli.out_prefix);
    let mut cwtr = csv::Writer::from_path(&cd_csv)?;
    cwtr.write_record([
        "mode",
        "k",
        "nfw_power",
        "corr_power",
        "corr_phase",
        "ratio",
    ])?;
    for (mode_idx, &k) in cd_wavenumbers.iter().enumerate() {
        let ratio = if nfw_cd_power[mode_idx] > 1e-20 {
            corr_cd_power[mode_idx] / nfw_cd_power[mode_idx]
        } else {
            f64::NAN
        };
        cwtr.write_record([
            &format!("{}", mode_idx + 1),
            &format!("{:.6}", k),
            &format!("{:.8e}", nfw_cd_power[mode_idx]),
            &format!("{:.8e}", corr_cd_power[mode_idx]),
            &format!("{:.6}", corr_cd_phase[mode_idx]),
            &format!("{:.8}", ratio),
        ])?;
    }
    cwtr.flush()?;

    eprintln!("\nResults written to:");
    eprintln!("  Broadband: {}", scan_csv);
    eprintln!("  CD modes:  {}", cd_csv);
    eprintln!("  Summary:   {}", summary_csv);

    Ok(())
}
