//! ML-grounded baryonic denoising and anomaly detection for MaNGA stacking.
//!
//! Two practical ML hypotheses tested in a single binary:
//!
//! **Hypothesis A: Metadata-Conditioned Baryonic Denoising**
//! Train a simple model (median-binned regression) on galaxy metadata to predict
//! per-galaxy mean NFW residual. Subtract prediction before stacking. If the
//! denoised stack has lower RMS and maintains SNR < 1.0, the null result is
//! strengthened. If SNR > 3.0 appears, something was hiding under baryonic noise.
//!
//! **Hypothesis B: Fourier Anomaly Detection**
//! Compute per-galaxy Fourier power at 7 CD-ZD wavenumbers. Rank galaxies by
//! total harmonic power. Test if the top 1% (70 galaxies) show phase coherence
//! at any mode (Rayleigh R > 0.5). If so, a sub-population signal exists.
//!
//! Both hypotheses use no external ML crates -- the regression is a binned
//! conditional mean (nonparametric, zero dependencies), and the anomaly
//! detection is rank-based sorting on Fourier power.
//!
//! Reference: Practical ML hypotheses, C-1393/C-1394 follow-up

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
#[command(name = "ml-baryonic-denoise")]
#[command(about = "ML baryonic denoising + Fourier anomaly detection")]
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

    /// Top fraction of galaxies to flag as anomalous.
    #[arg(long, default_value_t = 0.01)]
    anomaly_fraction: f64,

    /// Number of bins for metadata regression.
    #[arg(long, default_value_t = 20)]
    n_regression_bins: usize,

    /// Output CSV path prefix.
    #[arg(long, default_value = "data/results/e183/ml_denoise")]
    out_prefix: String,
}

/// Galaxy with metadata and residuals.
struct GalaxyRecord {
    name: String,
    residuals: NormalizedResiduals,
    // Metadata features
    log_m200: f64,
    inclination_deg: f64,
    sersic_n: f64,
    log_stellar_mass: f64,
    stellar_sigma: f64,
    ha_ew: f64,
    z: f64,
    r_eff: f64,
    abs_mag_r: f64,
    // Computed from residuals
    mean_delta_v: f64,
}

/// Rayleigh R statistic for phase clustering.
fn rayleigh_r(phases: &[f64]) -> f64 {
    let n = phases.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
    let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();
    ((sum_cos * sum_cos + sum_sin * sum_sin) / (n * n)).sqrt()
}

/// Binned conditional mean regression.
///
/// Sorts galaxies by a single feature, splits into n_bins, and computes
/// the mean target (mean_delta_v) in each bin. Prediction for a new galaxy:
/// find its bin, return the bin mean. This is a nonparametric, zero-dependency
/// regression that captures the dominant feature's effect.
///
/// Returns: (bin_edges, bin_means) for the best single feature.
fn best_single_feature_regression(
    galaxies: &[GalaxyRecord],
    n_bins: usize,
) -> (usize, Vec<f64>, Vec<f64>, f64) {
    type FeatureExtractor = Box<dyn Fn(&GalaxyRecord) -> f64>;
    let features: Vec<(&str, FeatureExtractor)> = vec![
        ("log_m200", Box::new(|g: &GalaxyRecord| g.log_m200)),
        (
            "inclination",
            Box::new(|g: &GalaxyRecord| g.inclination_deg),
        ),
        ("sersic_n", Box::new(|g: &GalaxyRecord| g.sersic_n)),
        ("log_mstar", Box::new(|g: &GalaxyRecord| g.log_stellar_mass)),
        ("sigma", Box::new(|g: &GalaxyRecord| g.stellar_sigma)),
        ("ha_ew", Box::new(|g: &GalaxyRecord| g.ha_ew)),
        ("z", Box::new(|g: &GalaxyRecord| g.z)),
        ("r_eff", Box::new(|g: &GalaxyRecord| g.r_eff)),
        ("abs_mag_r", Box::new(|g: &GalaxyRecord| g.abs_mag_r)),
    ];

    let target_mean: f64 =
        galaxies.iter().map(|g| g.mean_delta_v).sum::<f64>() / galaxies.len() as f64;
    let ss_tot: f64 = galaxies
        .iter()
        .map(|g| (g.mean_delta_v - target_mean).powi(2))
        .sum();

    let mut best_feat_idx = 0;
    let mut best_r2 = f64::NEG_INFINITY;
    let mut best_edges: Vec<f64> = Vec::new();
    let mut best_means: Vec<f64> = Vec::new();

    for (feat_idx, (feat_name, feat_fn)) in features.iter().enumerate() {
        // Sort indices by feature value
        let mut indexed: Vec<(usize, f64)> = galaxies
            .iter()
            .enumerate()
            .map(|(i, g)| (i, feat_fn(g)))
            .filter(|(_, v)| v.is_finite())
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if indexed.len() < n_bins * 2 {
            continue;
        }

        let bin_size = indexed.len() / n_bins;
        let mut edges: Vec<f64> = Vec::new();
        let mut means: Vec<f64> = Vec::new();

        let mut ss_res = 0.0_f64;

        for bin in 0..n_bins {
            let start = bin * bin_size;
            let end = if bin == n_bins - 1 {
                indexed.len()
            } else {
                (bin + 1) * bin_size
            };

            let edge = indexed[start].1;
            edges.push(edge);

            let bin_mean: f64 = indexed[start..end]
                .iter()
                .map(|(i, _)| galaxies[*i].mean_delta_v)
                .sum::<f64>()
                / (end - start) as f64;
            means.push(bin_mean);

            for &(i, _) in &indexed[start..end] {
                ss_res += (galaxies[i].mean_delta_v - bin_mean).powi(2);
            }
        }

        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        eprintln!("  Feature '{}': R^2 = {:.4}", feat_name, r2);

        if r2 > best_r2 {
            best_r2 = r2;
            best_feat_idx = feat_idx;
            best_edges = edges;
            best_means = means;
        }
    }

    (best_feat_idx, best_edges, best_means, best_r2)
}

/// Predict mean_delta_v from the binned regression model.
fn predict_from_bins(value: f64, edges: &[f64], means: &[f64]) -> f64 {
    if !value.is_finite() || edges.is_empty() {
        return 0.0;
    }
    // Find the bin this value falls into
    let mut bin = 0;
    for (i, &edge) in edges.iter().enumerate().skip(1) {
        if value >= edge {
            bin = i;
        }
    }
    means[bin]
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

    // Build galaxy records with metadata
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
            let mean_dv = points.iter().map(|p| p.delta_v).sum::<f64>() / points.len() as f64;

            galaxies.push(GalaxyRecord {
                name: galaxy.plateifu.clone(),
                residuals: NormalizedResiduals {
                    name: galaxy.plateifu.clone(),
                    r_s_kpc: r_s,
                    points,
                },
                log_m200,
                inclination_deg: meta.inclination_deg(),
                sersic_n: meta.sersic_n,
                log_stellar_mass: meta.log_stellar_mass,
                stellar_sigma: meta.stellar_sigma_km_s,
                ha_ew: meta.ha_ew,
                z: meta.z,
                r_eff: meta.r_eff_arcsec,
                abs_mag_r: meta.abs_mag_r,
                mean_delta_v: mean_dv,
            });
        } else {
            skipped += 1;
        }
    }

    eprintln!(
        "Built {} galaxy records ({} skipped)",
        galaxies.len(),
        skipped
    );

    // ===================================================================
    // HYPOTHESIS A: Metadata-Conditioned Baryonic Denoising
    // ===================================================================
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("HYPOTHESIS A: Metadata-Conditioned Baryonic Denoising");
    eprintln!("{}", "=".repeat(60));

    // Find best single-feature binned regression
    eprintln!("\nEvaluating 9 metadata features for residual prediction:");
    let (best_idx, edges, means, best_r2) =
        best_single_feature_regression(&galaxies, cli.n_regression_bins);

    let feat_names = [
        "log_m200",
        "inclination",
        "sersic_n",
        "log_mstar",
        "sigma",
        "ha_ew",
        "z",
        "r_eff",
        "abs_mag_r",
    ];
    type FeatFn = Box<dyn Fn(&GalaxyRecord) -> f64>;
    let feat_fns: Vec<FeatFn> = vec![
        Box::new(|g: &GalaxyRecord| g.log_m200),
        Box::new(|g: &GalaxyRecord| g.inclination_deg),
        Box::new(|g: &GalaxyRecord| g.sersic_n),
        Box::new(|g: &GalaxyRecord| g.log_stellar_mass),
        Box::new(|g: &GalaxyRecord| g.stellar_sigma),
        Box::new(|g: &GalaxyRecord| g.ha_ew),
        Box::new(|g: &GalaxyRecord| g.z),
        Box::new(|g: &GalaxyRecord| g.r_eff),
        Box::new(|g: &GalaxyRecord| g.abs_mag_r),
    ];

    eprintln!(
        "\nBest feature: '{}' (R^2 = {:.4})",
        feat_names[best_idx], best_r2
    );

    // Create denoised residuals: subtract predicted mean from each galaxy
    let mut denoised_residuals: Vec<NormalizedResiduals> = Vec::new();
    let mut original_residuals: Vec<NormalizedResiduals> = Vec::new();

    for g in &galaxies {
        let feat_val = feat_fns[best_idx](g);
        let predicted_mean = predict_from_bins(feat_val, &edges, &means);
        let correction = predicted_mean;

        let denoised_pts: Vec<NormalizedPoint> = g
            .residuals
            .points
            .iter()
            .map(|p| NormalizedPoint {
                x: p.x,
                delta_v: p.delta_v - correction,
                delta_v_err: p.delta_v_err,
            })
            .collect();

        denoised_residuals.push(NormalizedResiduals {
            name: g.name.clone(),
            r_s_kpc: g.residuals.r_s_kpc,
            points: denoised_pts,
        });
        original_residuals.push(g.residuals.clone());
    }

    // Stack original and denoised
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

    eprintln!("\nStacking original residuals...");
    let orig_result = stack_residuals(&original_residuals, &config);

    eprintln!("Stacking denoised residuals...");
    let denoised_result = stack_residuals(&denoised_residuals, &config);

    let rms_improvement = (orig_result.rms_residual - denoised_result.rms_residual)
        / orig_result.rms_residual
        * 100.0;

    eprintln!("\n--- Hypothesis A Results ---");
    eprintln!(
        "Original:  RMS={:.6}, SNR={:.4}",
        orig_result.rms_residual, orig_result.detection_snr
    );
    eprintln!(
        "Denoised:  RMS={:.6}, SNR={:.4}",
        denoised_result.rms_residual, denoised_result.detection_snr
    );
    eprintln!("RMS improvement: {:.2}%", rms_improvement);
    eprintln!("Best feature R^2: {:.4}", best_r2);

    if best_r2 > 0.3 && rms_improvement > 15.0 && denoised_result.detection_snr < 1.0 {
        eprintln!("VERDICT: CONFIRMED -- baryonic denoising works, null result strengthened");
    } else if denoised_result.detection_snr > 3.0 {
        eprintln!("VERDICT: DISCOVERY -- hidden signal revealed after denoising!");
    } else if best_r2 < 0.1 {
        eprintln!("VERDICT: REJECTED -- metadata does not predict residuals (R^2 < 0.1)");
    } else {
        eprintln!(
            "VERDICT: PARTIAL -- some metadata prediction (R^2={:.3}) but insufficient denoising ({:.1}%)",
            best_r2, rms_improvement
        );
    }

    // ===================================================================
    // HYPOTHESIS B: Fourier Anomaly Detection
    // ===================================================================
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("HYPOTHESIS B: Fourier Anomaly Detection for Outlier Galaxies");
    eprintln!("{}", "=".repeat(60));

    // Compute per-galaxy Fourier power at CD-ZD wavenumbers
    struct GalaxyFourier {
        name: String,
        total_power: f64,
        mode_power: Vec<f64>,
        mode_phase: Vec<f64>,
    }

    let mut galaxy_fourier: Vec<GalaxyFourier> = Vec::new();

    for g in &galaxies {
        let x_vals: Vec<f64> = g.residuals.points.iter().map(|p| p.x).collect();
        let dv_vals: Vec<f64> = g.residuals.points.iter().map(|p| p.delta_v).collect();
        let n_contrib: Vec<usize> = vec![1; x_vals.len()]; // Each point contributes 1

        let (power, phase) = fourier_at_wavenumbers(&x_vals, &dv_vals, &n_contrib, 1, &wavenumbers);
        let total: f64 = power.iter().sum();

        galaxy_fourier.push(GalaxyFourier {
            name: g.name.clone(),
            total_power: total,
            mode_power: power,
            mode_phase: phase,
        });
    }

    // Sort by total Fourier power (descending)
    galaxy_fourier.sort_by(|a, b| b.total_power.partial_cmp(&a.total_power).unwrap());

    let n_anomalous = (galaxies.len() as f64 * cli.anomaly_fraction).ceil() as usize;
    let n_anomalous = n_anomalous.max(10); // At least 10

    eprintln!("\nTop {} galaxies by total Fourier power:", n_anomalous);
    for (i, gf) in galaxy_fourier.iter().take(5).enumerate() {
        eprintln!(
            "  #{}: {} total_power={:.6e}",
            i + 1,
            gf.name,
            gf.total_power
        );
    }

    // Rayleigh test on anomalous subset phases
    eprintln!(
        "\nPhase coherence (Rayleigh R) in top {} galaxies:",
        n_anomalous
    );
    eprintln!(
        "{:>5}  {:>8}  {:>10}  {:>12}",
        "mode", "k", "rayleigh_r", "mean_power"
    );

    let mut any_coherent = false;
    let mut anomaly_mode_results: Vec<(usize, f64, f64, f64)> = Vec::new();

    for (mode_idx, &k) in wavenumbers.iter().enumerate() {
        let phases: Vec<f64> = galaxy_fourier
            .iter()
            .take(n_anomalous)
            .map(|gf| gf.mode_phase[mode_idx])
            .collect();
        let powers: Vec<f64> = galaxy_fourier
            .iter()
            .take(n_anomalous)
            .map(|gf| gf.mode_power[mode_idx])
            .collect();

        let r = rayleigh_r(&phases);
        let mean_pow = powers.iter().sum::<f64>() / powers.len() as f64;

        eprintln!(
            "{:>5}  {:>8.4}  {:>10.4}  {:>12.6e}",
            mode_idx + 1,
            k,
            r,
            mean_pow
        );

        if r > 0.5 {
            any_coherent = true;
        }
        anomaly_mode_results.push((mode_idx + 1, k, r, mean_pow));
    }

    // Stack only the anomalous galaxies
    let anomalous_names: std::collections::HashSet<String> = galaxy_fourier
        .iter()
        .take(n_anomalous)
        .map(|gf| gf.name.clone())
        .collect();

    let anomalous_residuals: Vec<NormalizedResiduals> = galaxies
        .iter()
        .filter(|g| anomalous_names.contains(&g.name))
        .map(|g| g.residuals.clone())
        .collect();

    let anomaly_result = stack_residuals(&anomalous_residuals, &config);

    eprintln!("\n--- Anomalous subset stacking ---");
    eprintln!(
        "N={}, RMS={:.6}, SNR={:.4}",
        anomaly_result.n_galaxies, anomaly_result.rms_residual, anomaly_result.detection_snr
    );

    // Compare with bottom 70 galaxies (lowest Fourier power)
    let quiet_names: std::collections::HashSet<String> = galaxy_fourier
        .iter()
        .rev()
        .take(n_anomalous)
        .map(|gf| gf.name.clone())
        .collect();

    let quiet_residuals: Vec<NormalizedResiduals> = galaxies
        .iter()
        .filter(|g| quiet_names.contains(&g.name))
        .map(|g| g.residuals.clone())
        .collect();

    let quiet_result = stack_residuals(&quiet_residuals, &config);

    eprintln!(
        "Quiet subset (bottom {}): N={}, RMS={:.6}, SNR={:.4}",
        n_anomalous, quiet_result.n_galaxies, quiet_result.rms_residual, quiet_result.detection_snr
    );

    let snr_ratio = if quiet_result.detection_snr > 0.0 {
        anomaly_result.detection_snr / quiet_result.detection_snr
    } else {
        f64::NAN
    };
    eprintln!("Anomalous/Quiet SNR ratio: {:.2}", snr_ratio);

    if any_coherent && anomaly_result.detection_snr > 2.0 {
        eprintln!("VERDICT: DETECTION -- coherent outlier sub-population found");
    } else if anomaly_result.detection_snr < 2.0 {
        let max_r = anomaly_mode_results
            .iter()
            .map(|m| m.2)
            .fold(0.0_f64, f64::max);
        if max_r < 0.3 {
            eprintln!("VERDICT: REJECTED -- no outlier sub-population (R < 0.3, SNR < 2.0)");
        } else {
            eprintln!(
                "VERDICT: INCONCLUSIVE -- some phase clustering (R={:.2}) but low SNR ({:.2})",
                max_r, anomaly_result.detection_snr
            );
        }
    } else {
        eprintln!("VERDICT: INCONCLUSIVE");
    }

    // ===================================================================
    // Write output CSVs
    // ===================================================================

    // Hypothesis A: denoised stacking comparison
    let denoise_csv = format!("{}_ha.csv", cli.out_prefix);
    let mut wtr = csv::Writer::from_path(&denoise_csv)?;
    wtr.write_record([
        "x",
        "delta_orig",
        "delta_denoised",
        "err_orig",
        "err_denoised",
        "n_contributing",
    ])?;
    for j in 0..config.n_grid {
        wtr.write_record(&[
            format!("{:.6}", orig_result.x_grid[j]),
            format!("{:.8e}", orig_result.delta_stack[j]),
            format!("{:.8e}", denoised_result.delta_stack[j]),
            format!("{:.8e}", orig_result.delta_stack_err[j]),
            format!("{:.8e}", denoised_result.delta_stack_err[j]),
            format!("{}", orig_result.n_contributing[j]),
        ])?;
    }
    wtr.flush()?;

    // Hypothesis B: per-galaxy Fourier power
    let anomaly_csv = format!("{}_hb.csv", cli.out_prefix);
    let mut wtr2 = csv::Writer::from_path(&anomaly_csv)?;
    wtr2.write_record([
        "rank",
        "plateifu",
        "total_power",
        "is_anomalous",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "ph1",
        "ph2",
        "ph3",
        "ph4",
        "ph5",
        "ph6",
        "ph7",
    ])?;
    for (rank, gf) in galaxy_fourier.iter().enumerate() {
        let is_anom = if rank < n_anomalous { "1" } else { "0" };
        let mut row = vec![
            format!("{}", rank + 1),
            gf.name.clone(),
            format!("{:.8e}", gf.total_power),
            is_anom.to_string(),
        ];
        for p in &gf.mode_power {
            row.push(format!("{:.8e}", p));
        }
        for ph in &gf.mode_phase {
            row.push(format!("{:.6}", ph));
        }
        wtr2.write_record(&row)?;
    }
    wtr2.flush()?;

    // Summary
    let summary_csv = format!("{}_summary.csv", cli.out_prefix);
    let mut swtr = csv::Writer::from_path(&summary_csv)?;
    swtr.write_record(["hypothesis", "metric", "value"])?;
    swtr.write_record(["A", "best_feature", feat_names[best_idx]])?;
    swtr.write_record(["A", "r_squared", &format!("{:.6}", best_r2)])?;
    swtr.write_record(["A", "orig_rms", &format!("{:.8}", orig_result.rms_residual)])?;
    swtr.write_record([
        "A",
        "denoised_rms",
        &format!("{:.8}", denoised_result.rms_residual),
    ])?;
    swtr.write_record([
        "A",
        "rms_improvement_pct",
        &format!("{:.2}", rms_improvement),
    ])?;
    swtr.write_record([
        "A",
        "orig_snr",
        &format!("{:.6}", orig_result.detection_snr),
    ])?;
    swtr.write_record([
        "A",
        "denoised_snr",
        &format!("{:.6}", denoised_result.detection_snr),
    ])?;
    swtr.write_record(["B", "n_anomalous", &format!("{}", n_anomalous)])?;
    swtr.write_record([
        "B",
        "anomaly_snr",
        &format!("{:.6}", anomaly_result.detection_snr),
    ])?;
    swtr.write_record([
        "B",
        "quiet_snr",
        &format!("{:.6}", quiet_result.detection_snr),
    ])?;
    swtr.write_record(["B", "snr_ratio", &format!("{:.4}", snr_ratio)])?;
    for (mode, _k, r, _) in &anomaly_mode_results {
        swtr.write_record([
            "B",
            &format!("rayleigh_r_mode{}", mode),
            &format!("{:.6}", r),
        ])?;
    }
    swtr.flush()?;

    eprintln!("\nResults written to:");
    eprintln!("  Hypothesis A: {}", denoise_csv);
    eprintln!("  Hypothesis B: {}", anomaly_csv);
    eprintln!("  Summary:      {}", summary_csv);

    Ok(())
}
