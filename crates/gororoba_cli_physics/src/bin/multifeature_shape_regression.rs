//! Hypothesis A: Multi-feature baryonic shape regression.
//!
//! Extends the single-feature mean regression (ml_baryonic_denoise) to:
//! 1. Project each galaxy's residual onto 2 SVD modes (capturing 98.4% of variance)
//! 2. Regress both SVD coefficients on 4 metadata features via least-squares
//! 3. Subtract the predicted SHAPE (not just mean) before restacking
//! 4. Measure corrected RMS and SNR
//!
//! Also embeds Hypothesis C: Red-noise spectral index diagnostic.
//! Fits R(k) ~ k^gamma to the 7 Rayleigh R values from galaxy_phase_coherence.csv
//! and checks consistency with DC14 spectral slope.
//!
//! Detection A: R^2 > 0.45 AND corrected RMS < 0.126 (original)
//! Failure A: R^2 < 0.35 OR corrected RMS >= 0.126
//! Detection C: gamma in [0.5, 1.5] AND predicted/observed R ratio in [0.5, 2.0] for >= 5/7 modes
//! Failure C: gamma outside [0.2, 2.0]

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig, stack_residuals,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use nalgebra::{DMatrix, DVector};
use std::{f64::consts::PI, path::PathBuf};

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "multifeature-shape-regression")]
#[command(about = "Multi-feature baryonic shape regression + red-noise spectral index")]
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

    #[arg(long, default_value = "data/results/e183/multifeature_shape.csv")]
    out_csv: PathBuf,
}

struct GalaxyRecord {
    residuals: NormalizedResiduals,
    // Metadata features (standardized)
    features: [f64; 4], // ha_ew, sersic_n, inclination, sigma
    // SVD projection coefficients (filled after basis construction)
    svd_coeffs: [f64; 2],
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

    // Build per-galaxy residuals with metadata
    let mut galaxies: Vec<GalaxyRecord> = Vec::new();
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
            galaxies.push(GalaxyRecord {
                residuals: NormalizedResiduals {
                    name: galaxy.plateifu.clone(),
                    r_s_kpc: r_s,
                    points,
                },
                features: [
                    meta.ha_ew,
                    meta.sersic_n,
                    meta.inclination_deg(),
                    meta.stellar_sigma_km_s,
                ],
                svd_coeffs: [0.0, 0.0],
            });
        } else {
            skipped += 1;
        }
    }

    let n_gal = galaxies.len();
    eprintln!("Normalized {} galaxies ({} skipped)", n_gal, skipped);

    // Step 1: Stack to get the reference profile (200 bins)
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

    let owned_residuals: Vec<NormalizedResiduals> =
        galaxies.iter().map(|g| g.residuals.clone()).collect();
    let baseline_result = stack_residuals(&owned_residuals, &config);
    let original_rms = baseline_result.rms_residual;
    let original_snr = baseline_result.detection_snr;

    eprintln!(
        "Original stacked RMS: {:.6}, SNR: {:.4}",
        original_rms, original_snr
    );

    // Step 2: Build SVD basis from the stacked profile
    // Use only populated bins (n_contributing >= 3)
    let valid_indices: Vec<usize> = (0..config.n_grid)
        .filter(|&j| baseline_result.n_contributing[j] >= config.min_galaxies_per_bin)
        .collect();
    let n_valid = valid_indices.len();
    eprintln!("Valid stacking bins: {}", n_valid);

    // Build per-galaxy interpolated profiles on the valid grid
    let x_grid: Vec<f64> = valid_indices
        .iter()
        .map(|&j| baseline_result.x_grid[j])
        .collect();

    // Interpolate each galaxy onto the common grid
    let mut galaxy_profiles: Vec<Vec<f64>> = Vec::with_capacity(n_gal);
    for g in &galaxies {
        let mut profile = vec![0.0_f64; n_valid];
        for (gi, &x_target) in x_grid.iter().enumerate() {
            // Linear interpolation from galaxy's sparse points
            let pts = &g.residuals.points;
            let mut val = 0.0_f64;
            let mut found = false;

            for w in pts.windows(2) {
                if w[0].x <= x_target && w[1].x >= x_target {
                    let frac = if (w[1].x - w[0].x).abs() > 1e-10 {
                        (x_target - w[0].x) / (w[1].x - w[0].x)
                    } else {
                        0.5
                    };
                    val = w[0].delta_v + frac * (w[1].delta_v - w[0].delta_v);
                    found = true;
                    break;
                }
            }

            if !found {
                // Nearest neighbor extrapolation
                val = pts
                    .iter()
                    .min_by(|a, b| {
                        (a.x - x_target)
                            .abs()
                            .partial_cmp(&(b.x - x_target).abs())
                            .unwrap()
                    })
                    .map(|p| p.delta_v)
                    .unwrap_or(0.0);
            }
            profile[gi] = val;
        }
        galaxy_profiles.push(profile);
    }

    // Build data matrix (n_gal x n_valid) and compute SVD
    eprintln!(
        "Computing SVD of {} x {} galaxy profile matrix...",
        n_gal, n_valid
    );
    let data_matrix = DMatrix::from_fn(n_gal, n_valid, |i, j| galaxy_profiles[i][j]);

    // Compute mean profile and center
    let mean_profile: Vec<f64> = (0..n_valid)
        .map(|j| data_matrix.column(j).iter().sum::<f64>() / n_gal as f64)
        .collect();

    let centered = DMatrix::from_fn(n_gal, n_valid, |i, j| data_matrix[(i, j)] - mean_profile[j]);

    // SVD on the centered matrix (truncated to first 2 components)
    let svd = centered.clone().svd(true, true);
    let u = svd.u.as_ref().unwrap();
    let sigma = &svd.singular_values;
    let vt = svd.v_t.as_ref().unwrap();

    let total_var: f64 = sigma.iter().map(|s| s * s).sum();
    let var_2modes = sigma[0] * sigma[0] + sigma[1] * sigma[1];
    eprintln!("SVD: sigma_1={:.4e}, sigma_2={:.4e}", sigma[0], sigma[1]);
    eprintln!(
        "2-mode variance explained: {:.1}%",
        var_2modes / total_var * 100.0
    );

    // Extract per-galaxy 2-component SVD coefficients
    // c_i = U[i, :2] * diag(sigma[:2])
    for (i, g) in galaxies.iter_mut().enumerate() {
        g.svd_coeffs[0] = u[(i, 0)] * sigma[0];
        g.svd_coeffs[1] = u[(i, 1)] * sigma[1];
    }

    // Step 3: Standardize features
    let mut feat_means = [0.0_f64; 4];
    let mut feat_stds = [0.0_f64; 4];
    for f in 0..4 {
        let vals: Vec<f64> = galaxies
            .iter()
            .map(|g| g.features[f])
            .filter(|v| v.is_finite())
            .collect();
        let n = vals.len() as f64;
        feat_means[f] = vals.iter().sum::<f64>() / n;
        feat_stds[f] = (vals
            .iter()
            .map(|v| (v - feat_means[f]).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        if feat_stds[f] < 1e-10 {
            feat_stds[f] = 1.0;
        }
    }

    // Step 4: Multi-feature least-squares regression
    // For each SVD coefficient (2 targets), regress on 4 standardized features + intercept
    // Design matrix X: (n_gal x 5) = [1, f1, f2, f3, f4]
    let n_features = 5; // intercept + 4 features
    let mut x_mat = DMatrix::zeros(n_gal, n_features);
    let mut y_svd0 = DVector::zeros(n_gal);
    let mut y_svd1 = DVector::zeros(n_gal);
    let mut valid_count = 0usize;

    for g in &galaxies {
        let all_finite = g.features.iter().all(|f| f.is_finite());
        if !all_finite {
            continue;
        }

        x_mat[(valid_count, 0)] = 1.0; // intercept
        for f in 0..4 {
            x_mat[(valid_count, f + 1)] = (g.features[f] - feat_means[f]) / feat_stds[f];
        }
        y_svd0[valid_count] = g.svd_coeffs[0];
        y_svd1[valid_count] = g.svd_coeffs[1];
        valid_count += 1;
    }

    // Truncate to valid rows
    let x_valid = x_mat.rows(0, valid_count).clone_owned();
    let y0_valid = y_svd0.rows(0, valid_count).clone_owned();
    let y1_valid = y_svd1.rows(0, valid_count).clone_owned();

    eprintln!("Valid galaxies with all features: {}", valid_count);

    // Solve X^T X beta = X^T y via normal equations
    let xtx = x_valid.transpose() * &x_valid;
    let xty0 = x_valid.transpose() * &y0_valid;
    let xty1 = x_valid.transpose() * &y1_valid;

    let xtx_svd = xtx.clone().svd(true, true);
    let beta0 = xtx_svd
        .solve(&xty0, 1e-12)
        .unwrap_or_else(|_| DVector::zeros(n_features));
    let beta1 = xtx_svd
        .solve(&xty1, 1e-12)
        .unwrap_or_else(|_| DVector::zeros(n_features));

    // Compute R^2 for each SVD coefficient
    let y0_pred = &x_valid * &beta0;
    let y1_pred = &x_valid * &beta1;

    let r2_svd0 = r_squared(&y0_valid, &y0_pred);
    let r2_svd1 = r_squared(&y1_valid, &y1_pred);

    // Combined R^2: weighted by variance explained
    let w0 = sigma[0] * sigma[0] / var_2modes;
    let w1 = sigma[1] * sigma[1] / var_2modes;
    let r2_combined = w0 * r2_svd0 + w1 * r2_svd1;

    eprintln!("\n--- Multi-Feature Shape Regression ---");
    eprintln!("Features: ha_ew, sersic_n, inclination, sigma (standardized)");
    eprintln!("R^2 (SVD mode 1): {:.6}", r2_svd0);
    eprintln!("R^2 (SVD mode 2): {:.6}", r2_svd1);
    eprintln!("R^2 (combined, variance-weighted): {:.6}", r2_combined);

    let feat_names = ["intercept", "ha_ew", "sersic_n", "inclination", "sigma"];
    eprintln!("\nRegression coefficients (mode 1):");
    for (i, name) in feat_names.iter().enumerate() {
        eprintln!("  {:>12}: {:.6}", name, beta0[i]);
    }
    eprintln!("Regression coefficients (mode 2):");
    for (i, name) in feat_names.iter().enumerate() {
        eprintln!("  {:>12}: {:.6}", name, beta1[i]);
    }

    // Step 5: Subtract predicted shape from each galaxy and restack
    eprintln!("\nSubtracting predicted baryonic shape from each galaxy...");

    // Reconstruct predicted profile for each galaxy:
    // predicted_profile[i] = mean + c0_pred * v0 + c1_pred * v1
    // where v0, v1 are the first 2 right singular vectors
    let v0: Vec<f64> = (0..n_valid).map(|j| vt[(0, j)]).collect();
    let v1: Vec<f64> = (0..n_valid).map(|j| vt[(1, j)]).collect();

    let mut corrected_galaxies: Vec<NormalizedResiduals> = Vec::new();

    for (i, g) in galaxies.iter().enumerate() {
        let all_finite = g.features.iter().all(|f| f.is_finite());

        // Predict SVD coefficients from metadata
        let (c0_pred, c1_pred) = if all_finite {
            let mut x_row = DVector::zeros(n_features);
            x_row[0] = 1.0;
            for f in 0..4 {
                x_row[f + 1] = (g.features[f] - feat_means[f]) / feat_stds[f];
            }
            (x_row.dot(&beta0), x_row.dot(&beta1))
        } else {
            (0.0, 0.0) // No correction for galaxies with missing features
        };

        // Subtract predicted shape from galaxy's rotation curve points
        let corrected_points: Vec<NormalizedPoint> = g
            .residuals
            .points
            .iter()
            .map(|pt| {
                // Find predicted correction at this x value
                let correction = interpolate_correction(
                    pt.x,
                    &x_grid,
                    &mean_profile,
                    &v0,
                    &v1,
                    c0_pred,
                    c1_pred,
                );
                NormalizedPoint {
                    x: pt.x,
                    delta_v: pt.delta_v - correction,
                    delta_v_err: pt.delta_v_err,
                }
            })
            .collect();

        corrected_galaxies.push(NormalizedResiduals {
            name: g.residuals.name.clone(),
            r_s_kpc: g.residuals.r_s_kpc,
            points: corrected_points,
        });

        let _ = i;
    }

    // Restack corrected galaxies
    let corrected_result = stack_residuals(&corrected_galaxies, &config);
    let corrected_rms = corrected_result.rms_residual;
    let corrected_snr = corrected_result.detection_snr;
    let rms_change = (corrected_rms - original_rms) / original_rms * 100.0;

    eprintln!("\n--- Corrected Stacking Results ---");
    eprintln!(
        "Original  RMS: {:.6}, SNR: {:.4}",
        original_rms, original_snr
    );
    eprintln!(
        "Corrected RMS: {:.6}, SNR: {:.4}",
        corrected_rms, corrected_snr
    );
    eprintln!("RMS change: {:+.2}%", rms_change);

    // Hypothesis A verdict
    eprintln!("\n--- Hypothesis A Verdict ---");
    if r2_combined > 0.45 && corrected_rms < original_rms {
        eprintln!("DETECTION: Multi-feature shape model reduces RMS");
        eprintln!(
            "  R^2={:.4} > 0.45, corrected RMS {:.6} < original {:.6}",
            r2_combined, corrected_rms, original_rms
        );
    } else if r2_combined < 0.35 || corrected_rms >= original_rms {
        eprintln!("FAILURE: Multi-feature shape model does not improve stacking");
        eprintln!(
            "  R^2={:.4}, corrected RMS {:.6} vs original {:.6}",
            r2_combined, corrected_rms, original_rms
        );
    } else {
        eprintln!(
            "INCONCLUSIVE: R^2={:.4}, RMS change {:+.2}%",
            r2_combined, rms_change
        );
    }

    // =================================================================
    // Hypothesis C: Red-noise spectral index
    // =================================================================
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("HYPOTHESIS C: Red-Noise Spectral Index Diagnostic");
    eprintln!("{}", "=".repeat(60));

    // Galaxy phase coherence Rayleigh R values (from galaxy_phase_coherence.csv)
    let n_modes = cd_params.n_modes;
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * PI * n as f64 / n_modes as f64)
        .collect();

    // Compute per-galaxy Rayleigh R at each mode
    let mut mode_rayleigh_r = vec![0.0_f64; n_modes];
    for mode_idx in 0..n_modes {
        let k = wavenumbers[mode_idx];
        let mut sum_cos = 0.0_f64;
        let mut sum_sin = 0.0_f64;

        for g in &galaxies {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            let mut count = 0_usize;
            for pt in &g.residuals.points {
                re += pt.delta_v * (k * pt.x).cos();
                im += pt.delta_v * (k * pt.x).sin();
                count += 1;
            }
            if count > 0 {
                re /= count as f64;
                im /= count as f64;
            }
            let phase = im.atan2(re);
            sum_cos += phase.cos();
            sum_sin += phase.sin();
        }

        let n = n_gal as f64;
        mode_rayleigh_r[mode_idx] = ((sum_cos * sum_cos + sum_sin * sum_sin) / (n * n)).sqrt();
    }

    eprintln!("\nRayleigh R per mode:");
    for (i, (&k, &r)) in wavenumbers.iter().zip(mode_rayleigh_r.iter()).enumerate() {
        eprintln!("  mode {}: k={:.4}, R={:.6}", i + 1, k, r);
    }

    // Fit log(R) = gamma * log(k) + const via least-squares
    let log_k: Vec<f64> = wavenumbers.iter().map(|k| k.ln()).collect();
    let log_r: Vec<f64> = mode_rayleigh_r
        .iter()
        .map(|r| if *r > 0.0 { r.ln() } else { f64::NEG_INFINITY })
        .collect();

    // Filter valid points
    let valid_pairs: Vec<(f64, f64)> = log_k
        .iter()
        .zip(log_r.iter())
        .filter(|(_, lr)| lr.is_finite())
        .map(|(&lk, &lr)| (lk, lr))
        .collect();

    let gamma = if valid_pairs.len() >= 2 {
        let n = valid_pairs.len() as f64;
        let mean_lk = valid_pairs.iter().map(|p| p.0).sum::<f64>() / n;
        let mean_lr = valid_pairs.iter().map(|p| p.1).sum::<f64>() / n;
        let num: f64 = valid_pairs
            .iter()
            .map(|p| (p.0 - mean_lk) * (p.1 - mean_lr))
            .sum();
        let den: f64 = valid_pairs.iter().map(|p| (p.0 - mean_lk).powi(2)).sum();
        if den.abs() > 1e-30 { num / den } else { 0.0 }
    } else {
        0.0
    };

    let implied_beta = -2.0 * gamma; // P(k) ~ k^{-beta}, R ~ k^{gamma} => gamma = -beta/2
    eprintln!("\nSpectral index fit: R(k) ~ k^gamma");
    eprintln!("  gamma = {:.4}", gamma);
    eprintln!("  Implied PSD slope beta = -2*gamma = {:.4}", implied_beta);
    eprintln!("  DC14 measured beta = -2.205 (from H3)");
    eprintln!(
        "  Consistency: {:.1}%",
        (1.0 - (implied_beta - (-2.205)).abs() / 2.205) * 100.0
    );

    // Check predicted/observed R ratio
    let const_fit = if !valid_pairs.is_empty() {
        let n = valid_pairs.len() as f64;
        let mean_lr = valid_pairs.iter().map(|p| p.1).sum::<f64>() / n;
        let mean_lk = valid_pairs.iter().map(|p| p.0).sum::<f64>() / n;
        mean_lr - gamma * mean_lk
    } else {
        0.0
    };

    let mut n_good_ratio = 0usize;
    eprintln!("\nPredicted vs observed Rayleigh R:");
    for (i, (&k, &r_obs)) in wavenumbers.iter().zip(mode_rayleigh_r.iter()).enumerate() {
        let r_pred = (const_fit + gamma * k.ln()).exp();
        let ratio = if r_obs > 0.0 {
            r_pred / r_obs
        } else {
            f64::NAN
        };
        let ok = ratio.is_finite() && ratio > 0.5 && ratio < 2.0;
        if ok {
            n_good_ratio += 1;
        }
        eprintln!(
            "  mode {}: R_obs={:.6}, R_pred={:.6}, ratio={:.4} {}",
            i + 1,
            r_obs,
            r_pred,
            ratio,
            if ok { "OK" } else { "MISS" }
        );
    }

    eprintln!("\n--- Hypothesis C Verdict ---");
    if (0.5..=1.5).contains(&gamma) && n_good_ratio >= 5 {
        eprintln!(
            "DETECTION: Red-noise scaling R(k)~k^{:.2} confirmed ({}/7 modes match)",
            gamma, n_good_ratio
        );
    } else if !(0.2..=2.0).contains(&gamma) {
        eprintln!("FAILURE: gamma={:.4} outside [0.2, 2.0]", gamma);
    } else {
        eprintln!(
            "INCONCLUSIVE: gamma={:.4}, {}/7 modes within [0.5, 2.0] ratio",
            gamma, n_good_ratio
        );
    }

    // Write output CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record(["test", "metric", "value"])?;

    // Hypothesis A results
    wtr.write_record(["A", "r2_svd0", &format!("{:.6}", r2_svd0)])?;
    wtr.write_record(["A", "r2_svd1", &format!("{:.6}", r2_svd1)])?;
    wtr.write_record(["A", "r2_combined", &format!("{:.6}", r2_combined)])?;
    wtr.write_record(["A", "original_rms", &format!("{:.6}", original_rms)])?;
    wtr.write_record(["A", "corrected_rms", &format!("{:.6}", corrected_rms)])?;
    wtr.write_record(["A", "rms_change_pct", &format!("{:+.4}", rms_change)])?;
    wtr.write_record(["A", "original_snr", &format!("{:.6}", original_snr)])?;
    wtr.write_record(["A", "corrected_snr", &format!("{:.6}", corrected_snr)])?;
    wtr.write_record(["A", "n_galaxies", &format!("{}", n_gal)])?;
    wtr.write_record(["A", "n_valid_features", &format!("{}", valid_count)])?;
    wtr.write_record([
        "A",
        "svd_var_2modes_pct",
        &format!("{:.2}", var_2modes / total_var * 100.0),
    ])?;

    // Feature importance (absolute beta coefficients)
    for (i, name) in feat_names.iter().enumerate() {
        wtr.write_record(["A", &format!("beta0_{}", name), &format!("{:.6}", beta0[i])])?;
        wtr.write_record(["A", &format!("beta1_{}", name), &format!("{:.6}", beta1[i])])?;
    }

    // Hypothesis C results
    wtr.write_record(["C", "gamma", &format!("{:.6}", gamma)])?;
    wtr.write_record(["C", "implied_beta", &format!("{:.6}", implied_beta)])?;
    wtr.write_record(["C", "n_modes_within_ratio", &format!("{}", n_good_ratio)])?;
    for (i, (&r, &k)) in mode_rayleigh_r.iter().zip(wavenumbers.iter()).enumerate() {
        wtr.write_record([
            "C",
            &format!("rayleigh_r_mode{}", i + 1),
            &format!("{:.6}", r),
        ])?;
        let r_pred = (const_fit + gamma * k.ln()).exp();
        wtr.write_record([
            "C",
            &format!("rayleigh_r_pred_mode{}", i + 1),
            &format!("{:.6}", r_pred),
        ])?;
    }

    wtr.flush()?;
    eprintln!("\nResults written to {:?}", cli.out_csv);

    Ok(())
}

fn r_squared(y_true: &DVector<f64>, y_pred: &DVector<f64>) -> f64 {
    let mean = y_true.mean();
    let ss_tot: f64 = y_true.iter().map(|yi| (yi - mean).powi(2)).sum();
    let ss_res: f64 = (y_true - y_pred).iter().map(|r| r.powi(2)).sum();
    if ss_tot < 1e-30 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

fn interpolate_correction(
    x: f64,
    x_grid: &[f64],
    mean_profile: &[f64],
    v0: &[f64],
    v1: &[f64],
    c0: f64,
    c1: f64,
) -> f64 {
    // Find bracketing grid indices
    if x_grid.is_empty() {
        return 0.0;
    }
    if x <= x_grid[0] {
        return mean_profile[0] + c0 * v0[0] + c1 * v1[0];
    }
    if x >= x_grid[x_grid.len() - 1] {
        let last = x_grid.len() - 1;
        return mean_profile[last] + c0 * v0[last] + c1 * v1[last];
    }

    for i in 0..x_grid.len() - 1 {
        if x_grid[i] <= x && x_grid[i + 1] >= x {
            let frac = if (x_grid[i + 1] - x_grid[i]).abs() > 1e-10 {
                (x - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
            } else {
                0.5
            };
            let v0_interp = v0[i] + frac * (v0[i + 1] - v0[i]);
            let v1_interp = v1[i] + frac * (v1[i + 1] - v1[i]);
            let mean_interp = mean_profile[i] + frac * (mean_profile[i + 1] - mean_profile[i]);
            return mean_interp + c0 * v0_interp + c1 * v1_interp;
        }
    }
    0.0
}
