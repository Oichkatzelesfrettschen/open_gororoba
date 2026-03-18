//! H3: Cross-algebra excess correlation as a cusp-core discriminant.
//!
//! The cross-algebra correlation excess (rho_obs - rho_null) was interpreted as
//! "shared noise." This test inverts that interpretation: the excess IS the signal
//! -- a quantitative measurement of the NFW cusp-core deviation.
//!
//! Method:
//! 1. Read observed cross-algebra correlations from cross_algebra_correlation.csv
//! 2. Read DC14 baryonic residual from dc14_stacking.csv
//! 3. Compute power spectral density of DC14 residual
//! 4. Predict cross-algebra correlation under colored (DC14) noise model
//! 5. Chi-squared comparison: does DC14 spectral shape explain the excess?
//!
//! Detection: chi^2/dof < 2.0 across all 6 algebra pairs
//! Failure: chi^2/dof > 5.0 (implies non-baryonic residual structure)
//!
//! Cross-domain analogy: CMB lensing convergence -- the cross-correlation between
//! different frequency channels is the lensing signal, not noise.

use std::{f64::consts::PI, path::PathBuf};

use clap::Parser;
use cosmology_core::harmonic_stacking::{
    CdDimensionParams, albert_peirce_wavenumbers, g2_angular_wavenumbers,
    sl2_partner_graph_wavenumbers,
};

#[derive(Parser)]
#[command(name = "cross-algebra-cusp-core")]
#[command(about = "H3: Cross-algebra excess as cusp-core discriminant")]
struct Cli {
    /// Path to cross_algebra_correlation.csv.
    #[arg(
        long,
        default_value = "data/results/e183/cross_algebra_correlation.csv"
    )]
    correlation_csv: PathBuf,

    /// Path to dc14_stacking.csv.
    #[arg(long, default_value = "data/results/e183/dc14_stacking.csv")]
    dc14_csv: PathBuf,

    /// Path to contrarian_baseline.csv (NFW residual for spectral comparison).
    #[arg(long, default_value = "data/results/e183/contrarian_baseline.csv")]
    baseline_csv: PathBuf,

    /// CD dimension.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Minimum x for spectral analysis.
    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    /// Maximum x for spectral analysis.
    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Output CSV path.
    #[arg(long, default_value = "data/results/e183/cusp_core_discriminant.csv")]
    out_csv: PathBuf,
}

struct ObservedCorrelation {
    rho_avg: f64,
    rho_null: f64,
    excess: f64,
}

struct Dc14Bin {
    x: f64,
    delta_nfw: f64,
    delta_dc14: f64,
    n_contributing: usize,
}

fn parse_correlation_csv(path: &std::path::Path) -> anyhow::Result<Vec<ObservedCorrelation>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result?;
        rows.push(ObservedCorrelation {
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
            delta_nfw: record[1].parse()?,
            delta_dc14: record[2].parse()?,
            n_contributing: record[5].parse()?,
        });
    }
    Ok(bins)
}

/// Compute the power spectral density S(dk) of a residual profile at a given
/// difference wavenumber dk, using numerical integration over the populated domain.
///
/// S(dk) = |F{delta}(dk)|^2 / L
/// where F is the Fourier transform and L = x_max - x_min.
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
        // Trapezoidal weight
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

/// Predict cross-correlation between two algebra templates under colored noise
/// with power spectral density S(k).
///
/// Under colored noise with PSD S(k):
/// rho_colored = sum_{ia,ib} w_a(ia) * w_b(ib) * S(k_a(ia) - k_b(ib))
///             / sqrt(sum_{ia,ia'} w_a(ia)*w_a(ia') * S(k_a(ia)-k_a(ia')))
///             / sqrt(sum_{ib,ib'} w_b(ib)*w_b(ib') * S(k_b(ib)-k_b(ib')))
fn predict_colored_correlation(
    x_values: &[f64],
    delta: &[f64],
    wavenumbers_a: &[f64],
    weights_a: &[f64],
    wavenumbers_b: &[f64],
    weights_b: &[f64],
) -> f64 {
    let mut inner = 0.0_f64;
    for (ia, (&ka, &wa)) in wavenumbers_a.iter().zip(weights_a.iter()).enumerate() {
        for (ib, (&kb, &wb)) in wavenumbers_b.iter().zip(weights_b.iter()).enumerate() {
            let s = spectral_density_at_dk(x_values, delta, ka - kb);
            inner += wa * wb * s;
            let _ = (ia, ib);
        }
    }

    let mut norm_a = 0.0_f64;
    for (ia, (&ka, &wa)) in wavenumbers_a.iter().zip(weights_a.iter()).enumerate() {
        for (ia2, (&ka2, &wa2)) in wavenumbers_a.iter().zip(weights_a.iter()).enumerate() {
            let s = spectral_density_at_dk(x_values, delta, ka - ka2);
            norm_a += wa * wa2 * s;
            let _ = (ia, ia2);
        }
    }

    let mut norm_b = 0.0_f64;
    for (ib, (&kb, &wb)) in wavenumbers_b.iter().zip(weights_b.iter()).enumerate() {
        for (ib2, (&kb2, &wb2)) in wavenumbers_b.iter().zip(weights_b.iter()).enumerate() {
            let s = spectral_density_at_dk(x_values, delta, kb - kb2);
            norm_b += wb * wb2 * s;
            let _ = (ib, ib2);
        }
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    inner / (norm_a * norm_b).sqrt()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading observed cross-algebra correlations...");
    let observed = parse_correlation_csv(&cli.correlation_csv)?;
    eprintln!("Loaded {} algebra pairs", observed.len());

    eprintln!("Loading DC14 stacking data...");
    let dc14_data = parse_dc14_csv(&cli.dc14_csv)?;

    // Filter to populated bins
    let valid: Vec<&Dc14Bin> = dc14_data.iter().filter(|b| b.n_contributing >= 3).collect();
    eprintln!("Valid DC14 bins: {}", valid.len());

    let x_values: Vec<f64> = valid.iter().map(|b| b.x).collect();
    let delta_nfw: Vec<f64> = valid.iter().map(|b| b.delta_nfw).collect();
    let delta_dc14: Vec<f64> = valid.iter().map(|b| b.delta_dc14).collect();

    // Define algebra wavenumbers and weights (matching cross_algebra_correlation.rs)
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

    let algebra_names = ["CD-ZD", "G2", "J3(O)", "sl(2)"];
    let all_k = [
        cd_k.as_slice(),
        g2_k.as_slice(),
        j3o_k.as_slice(),
        sl2_k.as_slice(),
    ];
    let all_w = [
        cd_w.as_slice(),
        g2_w.as_slice(),
        j3o_w.as_slice(),
        sl2_w.as_slice(),
    ];

    // Pair indices (matching cross_algebra_correlation.rs order)
    let pairs: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    // Predict correlations under three noise models:
    // 1. White noise (flat PSD) -> should match rho_null from original analysis
    // 2. NFW-colored noise (PSD from NFW residual)
    // 3. DC14-colored noise (PSD from DC14 residual)

    eprintln!("\n--- Predicted correlations under colored noise models ---");
    eprintln!(
        "{:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "pair", "observed", "rho_null", "nfw_pred", "dc14_pred", "obs-dc14"
    );

    let mut results: Vec<(String, f64, f64, f64, f64, f64, f64)> = Vec::new();

    for (pi, &(a, b)) in pairs.iter().enumerate() {
        let obs = &observed[pi];

        // NFW-colored prediction
        let rho_nfw = predict_colored_correlation(
            &x_values, &delta_nfw, all_k[a], all_w[a], all_k[b], all_w[b],
        );

        // DC14-colored prediction
        let rho_dc14 = predict_colored_correlation(
            &x_values,
            &delta_dc14,
            all_k[a],
            all_w[a],
            all_k[b],
            all_w[b],
        );

        let pair_name = format!("{}-{}", algebra_names[a], algebra_names[b]);
        let residual_dc14 = obs.rho_avg - rho_dc14;

        eprintln!(
            "{:>12}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
            pair_name, obs.rho_avg, obs.rho_null, rho_nfw, rho_dc14, residual_dc14
        );

        results.push((
            pair_name,
            obs.rho_avg,
            obs.rho_null,
            rho_nfw,
            rho_dc14,
            residual_dc14,
            obs.excess,
        ));
    }

    // Chi-squared for DC14 model (null hypothesis: excess is fully explained by DC14 PSD)
    // Assume uncertainty on rho ~ 1/sqrt(N-3) for Fisher z-transform
    let n_gal = 6992_usize;
    let sigma_rho = 1.0 / ((n_gal - 3) as f64).sqrt(); // ~0.012

    let mut chi2_dc14 = 0.0_f64;
    let mut chi2_nfw = 0.0_f64;
    let dof = results.len();

    for r in &results {
        let residual_dc14 = r.1 - r.4; // observed - dc14_predicted
        let residual_nfw = r.1 - r.3; // observed - nfw_predicted
        chi2_dc14 += (residual_dc14 / sigma_rho).powi(2);
        chi2_nfw += (residual_nfw / sigma_rho).powi(2);
    }

    let chi2_dof_dc14 = chi2_dc14 / dof as f64;
    let chi2_dof_nfw = chi2_nfw / dof as f64;

    eprintln!("\n--- Chi-squared Analysis ---");
    eprintln!(
        "NFW-colored model:  chi^2/dof = {:.2} ({:.1} / {})",
        chi2_dof_nfw, chi2_nfw, dof
    );
    eprintln!(
        "DC14-colored model: chi^2/dof = {:.2} ({:.1} / {})",
        chi2_dof_dc14, chi2_dc14, dof
    );
    eprintln!(
        "sigma_rho = {:.6} (Fisher z uncertainty, N={})",
        sigma_rho, n_gal
    );

    // Spectral slope analysis: fit P(k) ~ k^{-beta}
    eprintln!("\n--- Spectral slope analysis ---");
    let test_ks: Vec<f64> = (1..=20).map(|n| n as f64 * 0.5).collect();
    let psd_nfw: Vec<f64> = test_ks
        .iter()
        .map(|&k| spectral_density_at_dk(&x_values, &delta_nfw, k))
        .collect();
    let psd_dc14: Vec<f64> = test_ks
        .iter()
        .map(|&k| spectral_density_at_dk(&x_values, &delta_dc14, k))
        .collect();

    // Log-log linear fit for spectral slope
    let (beta_nfw, _) = log_log_fit(&test_ks, &psd_nfw);
    let (beta_dc14, _) = log_log_fit(&test_ks, &psd_dc14);

    eprintln!("NFW residual spectral slope:  beta = {:.3}", beta_nfw);
    eprintln!("DC14 residual spectral slope: beta = {:.3}", beta_dc14);

    // Verdict
    eprintln!("\n--- H3 Verdict ---");
    if chi2_dof_dc14 < 2.0 {
        eprintln!("DETECTION: DC14 spectral shape explains the cross-algebra excess");
        eprintln!(
            "  The excess IS a cusp-core measurement (chi^2/dof={:.2})",
            chi2_dof_dc14
        );
    } else if chi2_dof_dc14 > 5.0 {
        eprintln!("FAILURE: DC14 spectral shape does NOT explain the excess");
        eprintln!(
            "  Implies additional non-baryonic residual structure (chi^2/dof={:.2})",
            chi2_dof_dc14
        );
    } else {
        eprintln!(
            "INCONCLUSIVE: chi^2/dof={:.2} (between 2.0 and 5.0)",
            chi2_dof_dc14
        );
    }

    if chi2_dof_nfw < chi2_dof_dc14 {
        eprintln!(
            "NOTE: NFW-colored model fits BETTER than DC14 (chi^2/dof={:.2} vs {:.2})",
            chi2_dof_nfw, chi2_dof_dc14
        );
    } else {
        eprintln!(
            "NOTE: DC14-colored model fits better than NFW (chi^2/dof={:.2} vs {:.2})",
            chi2_dof_dc14, chi2_dof_nfw
        );
    }

    // Write output CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "pair",
        "rho_observed",
        "rho_null_white",
        "rho_pred_nfw",
        "rho_pred_dc14",
        "excess_observed",
        "excess_dc14_explained",
        "residual_dc14",
    ])?;

    for r in &results {
        let dc14_excess = r.4 - r.2; // dc14_pred - white_null
        wtr.write_record(&[
            r.0.clone(),
            format!("{:.8}", r.1),
            format!("{:.8}", r.2),
            format!("{:.8}", r.3),
            format!("{:.8}", r.4),
            format!("{:.8}", r.6),
            format!("{:.8}", dc14_excess),
            format!("{:.8}", r.5),
        ])?;
    }

    // Summary row
    wtr.write_record(&[
        "SUMMARY".to_string(),
        format!("chi2_dof_dc14={:.4}", chi2_dof_dc14),
        format!("chi2_dof_nfw={:.4}", chi2_dof_nfw),
        format!("beta_nfw={:.4}", beta_nfw),
        format!("beta_dc14={:.4}", beta_dc14),
        format!("sigma_rho={:.6}", sigma_rho),
        format!("n_valid_bins={}", valid.len()),
        format!("n_gal={}", n_gal),
    ])?;

    wtr.flush()?;
    eprintln!("\nResults written to {:?}", cli.out_csv);

    Ok(())
}

/// Log-log linear regression: fit log(y) = a + b * log(x).
/// Returns (slope b, intercept a).
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
