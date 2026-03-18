//! F3: Cross-algebra excess decomposition into baryonic + geometric + residual.
//!
//! The cross-algebra correlation excess (rho_obs - rho_null) for 6 algebra pairs
//! was previously modeled with DC14 colored noise alone (chi2/dof=22.55).  This
//! binary adds a second component: the geometric k-band overlap weighted by the
//! measured k^{-1.62} noise spectrum.  The unexplained residual after subtracting
//! both components is the quantity of interest.
//!
//! Three-component decomposition:
//!   rho_excess = rho_dc14_baryonic + rho_geometric_noise + residual
//!
//! where:
//!   rho_dc14_baryonic = predict_colored_correlation(DC14 PSD)
//!   rho_geometric_noise = overlap(k_a, k_b) * noise_power(gamma=1.62)
//!   residual = rho_obs - rho_null - rho_dc14 - rho_geometric
//!
//! Detection: residual < sigma_rho (0.012) for >= 5/6 pairs
//! Failure: residual > 3*sigma_rho for >= 3/6 pairs
//!
//! Reference: E-203

use std::{f64::consts::PI, path::PathBuf};

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        albert_peirce_wavenumbers, g2_angular_wavenumbers, sl2_partner_graph_wavenumbers,
        stack_residuals,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "cross-algebra-decomposition")]
#[command(about = "F3: Three-component cross-algebra excess decomposition (E-203)")]
struct Cli {
    /// Path to cross_algebra_correlation.csv (precomputed).
    #[arg(
        long,
        default_value = "data/results/e183/cross_algebra_correlation.csv"
    )]
    correlation_csv: PathBuf,

    /// Path to DC14 stacking CSV (precomputed).
    #[arg(long, default_value = "data/results/e183/dc14_stacking.csv")]
    dc14_csv: PathBuf,

    /// Path to MaNGA rotation curves (for bootstrap).
    #[arg(
        long,
        default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv"
    )]
    rotcurves: PathBuf,

    /// Path to DAPall metadata (for bootstrap).
    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    dapall: PathBuf,

    /// CD dimension.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Noise spectral index (gamma for P(k) ~ k^{-gamma}).
    #[arg(long, default_value_t = 1.62)]
    noise_gamma: f64,

    /// Gaussian bandwidth for k-band overlap (fraction of fundamental).
    #[arg(long, default_value_t = 0.269)]
    k_bandwidth: f64,

    /// Number of bootstrap resamples.
    #[arg(long, default_value_t = 200)]
    n_bootstrap: usize,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Minimum rotation curve points per galaxy.
    #[arg(long, default_value_t = 8)]
    min_points: usize,

    #[arg(
        long,
        default_value = "data/results/e203/cross_algebra_decomposition.csv"
    )]
    out_csv: PathBuf,
}

struct ObservedCorrelation {
    pair_name: String,
    rho_avg: f64,
    rho_null: f64,
    excess: f64,
}

struct Dc14Bin {
    x: f64,
    _delta_nfw: f64,
    delta_dc14: f64,
    n_contributing: usize,
}

fn parse_correlation_csv(path: &std::path::Path) -> anyhow::Result<Vec<ObservedCorrelation>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result?;
        rows.push(ObservedCorrelation {
            pair_name: record[0].to_string(),
            rho_avg: record[3].parse()?,
            rho_null: record[4].parse()?,
            excess: record[5].parse()?,
        });
    }
    Ok(rows)
}

fn parse_dc14_csv(path: &std::path::Path) -> anyhow::Result<Vec<Dc14Bin>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut bins = Vec::new();
    for result in rdr.records() {
        let record = result?;
        bins.push(Dc14Bin {
            x: record[0].parse()?,
            _delta_nfw: record[1].parse()?,
            delta_dc14: record[2].parse()?,
            n_contributing: record[5].parse()?,
        });
    }
    Ok(bins)
}

/// Compute PSD S(dk) of a residual profile at difference wavenumber dk.
fn spectral_density_at_dk(x_values: &[f64], delta: &[f64], dk: f64) -> f64 {
    let n = x_values.len();
    if n < 2 {
        return 0.0;
    }

    let mut re = 0.0_f64;
    let mut im = 0.0_f64;

    for i in 0..n {
        let x = x_values[i];
        let d = delta[i];
        let dx = if i == 0 {
            x_values[1] - x_values[0]
        } else if i == n - 1 {
            x_values[n - 1] - x_values[n - 2]
        } else {
            (x_values[i + 1] - x_values[i - 1]) / 2.0
        };
        re += d * (dk * x).cos() * dx;
        im += d * (dk * x).sin() * dx;
    }

    let l = x_values[n - 1] - x_values[0];
    if l <= 0.0 {
        return 0.0;
    }
    (re * re + im * im) / l
}

/// Predict cross-correlation under colored noise with PSD S(k).
fn predict_colored_correlation(
    x_values: &[f64],
    delta: &[f64],
    wavenumbers_a: &[f64],
    weights_a: &[f64],
    wavenumbers_b: &[f64],
    weights_b: &[f64],
) -> f64 {
    let mut inner = 0.0_f64;
    for (&ka, &wa) in wavenumbers_a.iter().zip(weights_a.iter()) {
        for (&kb, &wb) in wavenumbers_b.iter().zip(weights_b.iter()) {
            inner += wa * wb * spectral_density_at_dk(x_values, delta, ka - kb);
        }
    }

    let mut norm_a = 0.0_f64;
    for (&ka, &wa) in wavenumbers_a.iter().zip(weights_a.iter()) {
        for (&ka2, &wa2) in wavenumbers_a.iter().zip(weights_a.iter()) {
            norm_a += wa * wa2 * spectral_density_at_dk(x_values, delta, ka - ka2);
        }
    }

    let mut norm_b = 0.0_f64;
    for (&kb, &wb) in wavenumbers_b.iter().zip(weights_b.iter()) {
        for (&kb2, &wb2) in wavenumbers_b.iter().zip(weights_b.iter()) {
            norm_b += wb * wb2 * spectral_density_at_dk(x_values, delta, kb - kb2);
        }
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    inner / (norm_a * norm_b).sqrt()
}

/// Compute the geometric k-band overlap fraction between two wavenumber sets.
///
/// Each wavenumber is modeled as a Gaussian lobe of width `bandwidth`.
/// The overlap is the sum of pairwise Gaussian products, weighted by the
/// noise spectral shape k^{-gamma}.
fn geometric_noise_correlation(
    wavenumbers_a: &[f64],
    weights_a: &[f64],
    wavenumbers_b: &[f64],
    weights_b: &[f64],
    gamma: f64,
    bandwidth: f64,
) -> f64 {
    let sigma2 = bandwidth * bandwidth;

    let mut inner = 0.0_f64;
    for (&ka, &wa) in wavenumbers_a.iter().zip(weights_a.iter()) {
        for (&kb, &wb) in wavenumbers_b.iter().zip(weights_b.iter()) {
            // Gaussian overlap at dk = ka - kb
            let dk = ka - kb;
            let overlap = (-dk * dk / (4.0 * sigma2)).exp();
            // Noise weighting at the mean wavenumber
            let k_mean = (ka + kb) / 2.0;
            let noise_weight = if k_mean > 0.0 {
                k_mean.powf(-gamma)
            } else {
                1.0
            };
            inner += wa * wb * overlap * noise_weight;
        }
    }

    // Self-overlap normalization
    let mut norm_a = 0.0_f64;
    for (&ka, &wa) in wavenumbers_a.iter().zip(weights_a.iter()) {
        for (&ka2, &wa2) in wavenumbers_a.iter().zip(weights_a.iter()) {
            let dk = ka - ka2;
            let overlap = (-dk * dk / (4.0 * sigma2)).exp();
            let k_mean = (ka + ka2) / 2.0;
            let noise_weight = if k_mean > 0.0 {
                k_mean.powf(-gamma)
            } else {
                1.0
            };
            norm_a += wa * wa2 * overlap * noise_weight;
        }
    }

    let mut norm_b = 0.0_f64;
    for (&kb, &wb) in wavenumbers_b.iter().zip(weights_b.iter()) {
        for (&kb2, &wb2) in wavenumbers_b.iter().zip(weights_b.iter()) {
            let dk = kb - kb2;
            let overlap = (-dk * dk / (4.0 * sigma2)).exp();
            let k_mean = (kb + kb2) / 2.0;
            let noise_weight = if k_mean > 0.0 {
                k_mean.powf(-gamma)
            } else {
                1.0
            };
            norm_b += wb * wb2 * overlap * noise_weight;
        }
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    inner / (norm_a * norm_b).sqrt()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // ---- Load precomputed data ----
    eprintln!("Loading observed cross-algebra correlations...");
    let observed = parse_correlation_csv(&cli.correlation_csv)?;
    eprintln!("Loaded {} algebra pairs", observed.len());

    eprintln!("Loading DC14 stacking data...");
    let dc14_data = parse_dc14_csv(&cli.dc14_csv)?;
    let valid_dc14: Vec<&Dc14Bin> = dc14_data.iter().filter(|b| b.n_contributing >= 3).collect();
    eprintln!("Valid DC14 bins: {}", valid_dc14.len());

    let x_values: Vec<f64> = valid_dc14.iter().map(|b| b.x).collect();
    let delta_dc14: Vec<f64> = valid_dc14.iter().map(|b| b.delta_dc14).collect();

    // ---- Define algebra wavenumbers and weights ----
    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let cd_k: Vec<f64> = (1..=cd_params.n_modes)
        .map(|n| 2.0 * PI * n as f64 / cd_params.n_modes as f64)
        .collect();
    let cd_w: Vec<f64> = (1..=cd_params.n_modes)
        .map(|n| cd_params.assessor_fraction / n as f64)
        .collect();

    let g2_k = g2_angular_wavenumbers();
    let g2_w: Vec<f64> = (1..=g2_k.len()).map(|n| 1.0 / (n as f64).sqrt()).collect();

    let j3o_k = albert_peirce_wavenumbers();
    let j3o_w: Vec<f64> = vec![1.0; j3o_k.len()];

    let sl2_k = sl2_partner_graph_wavenumbers();
    let sl2_w: Vec<f64> = vec![14.0 / 84.0, 7.0 / 84.0];

    let _algebra_names = ["CD-ZD", "G2", "J3(O)", "sl(2)"];
    let all_k: [&[f64]; 4] = [&cd_k, &g2_k, &j3o_k, &sl2_k];
    let all_w: [&[f64]; 4] = [&cd_w, &g2_w, &j3o_w, &sl2_w];
    let pairs: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    let n_gal = 6992_usize;
    let sigma_rho = 1.0 / ((n_gal - 3) as f64).sqrt();

    // ---- Three-component decomposition ----
    eprintln!("\n--- Three-component decomposition ---");
    eprintln!(
        "{:>12}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}  {:>8}",
        "pair", "excess", "dc14", "geometric", "residual", "|r|/sigma", "verdict"
    );

    struct DecompositionResult {
        pair_name: String,
        rho_obs: f64,
        rho_null: f64,
        excess: f64,
        dc14_component: f64,
        geometric_component: f64,
        residual: f64,
        residual_sigma: f64,
    }

    let mut decomp_results: Vec<DecompositionResult> = Vec::new();

    for (pi, &(a, b)) in pairs.iter().enumerate() {
        let obs = &observed[pi];

        // Component 1: DC14 baryonic absorption
        let rho_dc14 = predict_colored_correlation(
            &x_values,
            &delta_dc14,
            all_k[a],
            all_w[a],
            all_k[b],
            all_w[b],
        );
        // DC14 excess = dc14_pred - null (the amount DC14 colored noise adds above white)
        let dc14_component = rho_dc14 - obs.rho_null;

        // Component 2: Geometric k-band overlap with noise spectral shape
        let geometric_component = geometric_noise_correlation(
            all_k[a],
            all_w[a],
            all_k[b],
            all_w[b],
            cli.noise_gamma,
            cli.k_bandwidth,
        );
        // Scale geometric contribution: use only the part not already captured by DC14
        // The geometric term predicts correlation from pure noise spectral shape;
        // DC14 captures correlation from baryonic profile.
        // To avoid double-counting, the geometric term is the residual after DC14.
        let geometric_scaled = geometric_component * (1.0 - dc14_component.abs().min(1.0));

        // Residual
        let residual = obs.excess - dc14_component - geometric_scaled;
        let residual_sigma = residual.abs() / sigma_rho;

        let verdict = if residual.abs() < sigma_rho {
            "OK"
        } else if residual.abs() < 3.0 * sigma_rho {
            "marginal"
        } else {
            "EXCESS"
        };

        eprintln!(
            "{:>12}  {:>8.4}  {:>8.4}  {:>10.4}  {:>10.6}  {:>10.2}  {:>8}",
            obs.pair_name,
            obs.excess,
            dc14_component,
            geometric_scaled,
            residual,
            residual_sigma,
            verdict
        );

        decomp_results.push(DecompositionResult {
            pair_name: obs.pair_name.clone(),
            rho_obs: obs.rho_avg,
            rho_null: obs.rho_null,
            excess: obs.excess,
            dc14_component,
            geometric_component: geometric_scaled,
            residual,
            residual_sigma,
        });
    }

    // ---- Summary statistics ----
    let n_below_sigma = decomp_results
        .iter()
        .filter(|r| r.residual.abs() < sigma_rho)
        .count();
    let n_above_3sigma = decomp_results
        .iter()
        .filter(|r| r.residual.abs() > 3.0 * sigma_rho)
        .count();
    let max_residual_sigma = decomp_results
        .iter()
        .map(|r| r.residual_sigma)
        .fold(0.0_f64, f64::max);

    eprintln!("\n--- Summary ---");
    eprintln!("sigma_rho = {:.6} (N={})", sigma_rho, n_gal);
    eprintln!("Pairs with |residual| < sigma_rho: {}/6", n_below_sigma);
    eprintln!("Pairs with |residual| > 3*sigma_rho: {}/6", n_above_3sigma);
    eprintln!("Max residual: {:.2} sigma", max_residual_sigma);

    // ---- Bootstrap (resample galaxies, recompute cross-correlation + decomposition) ----
    eprintln!("\n--- Bootstrap ({} resamples) ---", cli.n_bootstrap);

    // Load galaxy data for bootstrap
    let rotcurves = parse_manga_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    let dapall = parse_manga_dapall_csv(&cli.dapall).map_err(|e| anyhow::anyhow!(e))?;
    let dapall_map: std::collections::HashMap<String, _> = dapall
        .into_iter()
        .map(|e| (e.plateifu.clone(), e))
        .collect();

    let mut all_galaxies: Vec<NormalizedResiduals> = Vec::new();
    for galaxy in &rotcurves {
        let meta = match dapall_map.get(&galaxy.plateifu) {
            Some(m) => m,
            None => continue,
        };
        let log_m200 = meta.estimated_log_m200();
        if !log_m200.is_finite() || log_m200 < 10.0 {
            continue;
        }
        let m200 = 10.0_f64.powf(log_m200);
        let nfw = nfw_params_from_mass(m200, meta.z);
        if nfw.r_s_kpc <= 0.0 || nfw.c200 <= 0.0 {
            continue;
        }
        let points: Vec<NormalizedPoint> = galaxy
            .points
            .iter()
            .filter_map(|pt| {
                if pt.r_kpc <= 0.0 {
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
            all_galaxies.push(NormalizedResiduals {
                name: galaxy.plateifu.clone(),
                r_s_kpc: nfw.r_s_kpc,
                points,
            });
        }
    }
    eprintln!("Loaded {} galaxies for bootstrap", all_galaxies.len());

    let config = StackingConfig {
        x_min: 0.5,
        x_max: 10.0,
        n_grid: 200,
        min_galaxies_per_bin: 10,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: false,
    };

    // Bootstrap residuals per pair
    let mut boot_residuals: Vec<Vec<f64>> = vec![Vec::new(); 6];
    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);
    let n_gal_boot = all_galaxies.len();

    for b in 0..cli.n_bootstrap {
        // Resample galaxies
        let sample: Vec<NormalizedResiduals> = (0..n_gal_boot)
            .map(|_| {
                let idx = rng.gen_range(0..n_gal_boot);
                all_galaxies[idx].clone()
            })
            .collect();

        // Stack resampled galaxies
        let result = stack_residuals(&sample, &config);
        let valid_mask: Vec<bool> = result
            .n_contributing
            .iter()
            .map(|&n| n >= config.min_galaxies_per_bin)
            .collect();
        let x_valid: Vec<f64> = result
            .x_grid
            .iter()
            .zip(valid_mask.iter())
            .filter(|(_, m)| **m)
            .map(|(x, _)| *x)
            .collect();
        let delta_valid: Vec<f64> = result
            .delta_stack
            .iter()
            .zip(valid_mask.iter())
            .filter(|(_, m)| **m)
            .map(|(d, _)| *d)
            .collect();

        if x_valid.len() < 5 {
            continue;
        }

        // For each pair, compute cross-correlation from this bootstrap sample's PSD
        for (pi, &(a, b_idx)) in pairs.iter().enumerate() {
            let rho_boot = predict_colored_correlation(
                &x_valid,
                &delta_valid,
                all_k[a],
                all_w[a],
                all_k[b_idx],
                all_w[b_idx],
            );

            // Use the original null (shuffled) as reference
            let obs = &observed[pi];
            let boot_excess = rho_boot - obs.rho_null;

            // Apply same decomposition
            let dc14_comp = predict_colored_correlation(
                &x_values,
                &delta_dc14,
                all_k[a],
                all_w[a],
                all_k[b_idx],
                all_w[b_idx],
            ) - obs.rho_null;
            let geo_comp = geometric_noise_correlation(
                all_k[a],
                all_w[a],
                all_k[b_idx],
                all_w[b_idx],
                cli.noise_gamma,
                cli.k_bandwidth,
            ) * (1.0 - dc14_comp.abs().min(1.0));

            let boot_resid = boot_excess - dc14_comp - geo_comp;
            boot_residuals[pi].push(boot_resid);
        }

        if (b + 1) % 50 == 0 {
            eprintln!("  Bootstrap {}/{}", b + 1, cli.n_bootstrap);
        }
    }

    // ---- Verdict ----
    eprintln!("\n--- F3 Verdict ---");
    if n_below_sigma >= 5 {
        eprintln!(
            "EXPLAINED: {}/6 pairs have |residual| < sigma_rho ({:.4})",
            n_below_sigma, sigma_rho
        );
        eprintln!("  Cross-algebra excess is FULLY explained by DC14 + geometric noise");
    } else if n_above_3sigma >= 3 {
        eprintln!(
            "UNEXPLAINED: {}/6 pairs have |residual| > 3*sigma_rho",
            n_above_3sigma
        );
        eprintln!("  Non-baryonic, non-geometric structure exists in cross-algebra correlation");
    } else {
        eprintln!(
            "PARTIAL: {}/6 below sigma_rho, {}/6 above 3*sigma_rho",
            n_below_sigma, n_above_3sigma
        );
    }

    // ---- Write CSV ----
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "pair",
        "rho_obs",
        "rho_null",
        "excess",
        "dc14_component",
        "geometric_component",
        "residual",
        "residual_sigma",
        "boot_resid_median",
        "boot_resid_lo_95",
        "boot_resid_hi_95",
    ])?;

    for (pi, r) in decomp_results.iter().enumerate() {
        let mut br = boot_residuals[pi].clone();
        br.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (med, lo, hi) = if br.len() >= 10 {
            let lo_idx = (br.len() as f64 * 0.025) as usize;
            let hi_idx = (br.len() as f64 * 0.975).min(br.len() as f64 - 1.0) as usize;
            let med_idx = br.len() / 2;
            (br[med_idx], br[lo_idx], br[hi_idx])
        } else {
            (r.residual, r.residual, r.residual)
        };

        wtr.write_record(&[
            r.pair_name.clone(),
            format!("{:.8}", r.rho_obs),
            format!("{:.8}", r.rho_null),
            format!("{:.8}", r.excess),
            format!("{:.8}", r.dc14_component),
            format!("{:.8}", r.geometric_component),
            format!("{:.8}", r.residual),
            format!("{:.4}", r.residual_sigma),
            format!("{:.8}", med),
            format!("{:.8}", lo),
            format!("{:.8}", hi),
        ])?;
    }

    // Summary row
    wtr.write_record(&[
        "SUMMARY".to_string(),
        format!("sigma_rho={:.6}", sigma_rho),
        format!("n_below_sigma={}", n_below_sigma),
        format!("n_above_3sigma={}", n_above_3sigma),
        format!("max_resid_sigma={:.4}", max_residual_sigma),
        format!("noise_gamma={:.4}", cli.noise_gamma),
        format!("k_bandwidth={:.4}", cli.k_bandwidth),
        format!("n_bootstrap={}", cli.n_bootstrap),
        format!("boot_success={}", boot_residuals[0].len()),
        "".to_string(),
        "".to_string(),
    ])?;

    wtr.flush()?;
    eprintln!("\nResults written to {:?}", cli.out_csv);

    Ok(())
}
