//! H1: Baryonic Eigenspectrum -- ZD modes as a natural basis for feedback tomography.
//!
//! Tests whether the CD-ZD wavenumbers form a well-conditioned basis for decomposing
//! the baryonic inner-halo systematic (cusp trough + bulge excess + IFU edge spike).
//!
//! Method:
//! 1. Build design matrix A (N_valid_bins x 7) from cos(k_n * x) * w_n * exp(-x)
//! 2. SVD decomposition -> singular values, condition number
//! 3. Project delta_nfw and delta_dc14 onto ZD mode basis -> R^2
//! 4. Compare to random wavenumber control sets -> is the ZD basis privileged?
//!
//! Detection: kappa < 50 AND R^2(NFW) > 0.85
//! Failure: kappa > 500 OR R^2(NFW) < 0.3
//!
//! Cross-domain analogy: NMR spectroscopy -- Larmor frequencies set by physics,
//! but signal content at those frequencies encodes tissue contrast (biology).

use std::{f64::consts::PI, path::PathBuf};

use clap::Parser;
use stats_core::helpers::{singular_values, svd_projection_r_squared};

#[derive(Parser)]
#[command(name = "baryonic-eigenspectrum")]
#[command(about = "H1: ZD modes as baryonic feedback basis (SVD analysis)")]
struct Cli {
    /// Path to contrarian_baseline.csv (stacked NFW residual).
    #[arg(long, default_value = "data/results/e183/contrarian_baseline.csv")]
    baseline_csv: PathBuf,

    /// Path to dc14_stacking.csv (DC14 feedback residual).
    #[arg(long, default_value = "data/results/e183/dc14_stacking.csv")]
    dc14_csv: PathBuf,

    /// CD dimension.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Minimum contributing galaxies per bin.
    #[arg(long, default_value_t = 3)]
    min_contributing: usize,

    /// Number of random wavenumber control trials.
    #[arg(long, default_value_t = 200)]
    n_random_trials: usize,

    /// Output CSV path.
    #[arg(long, default_value = "data/results/e183/baryonic_eigenspectrum.csv")]
    out_csv: PathBuf,
}

struct BaselineBin {
    x: f64,
    delta_nfw: f64,
    n_contributing: usize,
}

struct Dc14Bin {
    x: f64,
    delta_dc14: f64,
}

fn parse_baseline_csv(path: &std::path::Path) -> anyhow::Result<Vec<BaselineBin>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut bins = Vec::new();
    for result in rdr.records() {
        let record = result?;
        bins.push(BaselineBin {
            x: record[0].parse()?,
            delta_nfw: record[1].parse()?,
            n_contributing: record[4].parse()?,
        });
    }
    Ok(bins)
}

fn parse_dc14_csv(path: &std::path::Path) -> anyhow::Result<Vec<Dc14Bin>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut bins = Vec::new();
    for result in rdr.records() {
        let record = result?;
        bins.push(Dc14Bin {
            x: record[0].parse()?,
            delta_dc14: record[2].parse()?,
        });
    }
    Ok(bins)
}

/// Build design matrix A for given wavenumbers and mode weights.
/// A[i][j] = w_j * cos(k_j * x_i) * envelope(x_i)
fn build_design_matrix(
    x_values: &[f64],
    wavenumbers: &[f64],
    mode_weights: &[f64],
) -> Vec<Vec<f64>> {
    let n_cols = wavenumbers.len();
    x_values
        .iter()
        .map(|&x| {
            (0..n_cols)
                .map(|j| mode_weights[j] * (wavenumbers[j] * x).cos() * (-x).exp())
                .collect()
        })
        .collect()
}


fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading baseline data...");
    let baseline = parse_baseline_csv(&cli.baseline_csv)?;
    let dc14_data = parse_dc14_csv(&cli.dc14_csv)?;

    // Filter to valid bins
    let valid_baseline: Vec<&BaselineBin> = baseline
        .iter()
        .filter(|b| b.n_contributing >= cli.min_contributing)
        .collect();

    eprintln!(
        "Valid bins (n_contributing >= {}): {} / {}",
        cli.min_contributing,
        valid_baseline.len(),
        baseline.len()
    );

    let x_values: Vec<f64> = valid_baseline.iter().map(|b| b.x).collect();
    let delta_nfw: Vec<f64> = valid_baseline.iter().map(|b| b.delta_nfw).collect();

    // Match DC14 residuals to the same x values
    let delta_dc14: Vec<f64> = x_values
        .iter()
        .map(|&x| {
            dc14_data
                .iter()
                .find(|d| (d.x - x).abs() < 1e-4)
                .map(|d| d.delta_dc14)
                .unwrap_or(0.0)
        })
        .collect();

    let n_valid = x_values.len();
    if n_valid < 3 {
        anyhow::bail!("Too few valid bins ({}) for SVD analysis", n_valid);
    }

    // CD-ZD wavenumbers and weights
    let n_modes = cli.cd_dim / 2 - 1;
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * PI * n as f64 / n_modes as f64)
        .collect();
    let mode_weights: Vec<f64> = (1..=n_modes).map(|n| 0.5 / n as f64).collect();

    eprintln!(
        "CD-ZD modes: {} wavenumbers, k in [{:.3}, {:.3}]",
        n_modes,
        wavenumbers[0],
        wavenumbers[n_modes - 1]
    );

    // Build design matrix
    let a = build_design_matrix(&x_values, &wavenumbers, &mode_weights);

    // SVD
    let svals = singular_values(&a);
    let kappa = if svals[svals.len() - 1] > 1e-15 {
        svals[0] / svals[svals.len() - 1]
    } else {
        f64::INFINITY
    };

    eprintln!("\n--- SVD Results ---");
    eprintln!("Design matrix: {} x {}", n_valid, n_modes);
    eprintln!("Singular values:");
    let mut cumulative_var = 0.0_f64;
    let total_var: f64 = svals.iter().map(|s| s * s).sum();
    for (i, &s) in svals.iter().enumerate() {
        cumulative_var += s * s;
        let frac = if total_var > 0.0 {
            cumulative_var / total_var
        } else {
            0.0
        };
        eprintln!(
            "  sigma_{}: {:.6e}  (cumulative variance: {:.1}%)",
            i + 1,
            s,
            frac * 100.0
        );
    }
    eprintln!("Condition number kappa: {:.2}", kappa);

    // Project delta_nfw onto ZD mode basis
    let r2_nfw = svd_projection_r_squared(&a, &delta_nfw);
    eprintln!("\nR^2 (NFW residual -> ZD modes): {:.6}", r2_nfw);

    // Project delta_dc14 onto ZD mode basis
    let r2_dc14 = svd_projection_r_squared(&a, &delta_dc14);
    eprintln!("R^2 (DC14 residual -> ZD modes): {:.6}", r2_dc14);

    // Random wavenumber control: how does the ZD basis compare to random k-sets?
    eprintln!(
        "\n--- Random Wavenumber Control ({} trials) ---",
        cli.n_random_trials
    );

    let mut rng_state: u64 = 42;
    let mut random_kappas: Vec<f64> = Vec::with_capacity(cli.n_random_trials);
    let mut random_r2_nfw: Vec<f64> = Vec::with_capacity(cli.n_random_trials);
    let mut random_r2_dc14: Vec<f64> = Vec::with_capacity(cli.n_random_trials);

    for _ in 0..cli.n_random_trials {
        // Generate random wavenumbers in same range as ZD modes
        let rand_k: Vec<f64> = (0..n_modes)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                wavenumbers[0] + u * (wavenumbers[n_modes - 1] - wavenumbers[0])
            })
            .collect();

        let a_rand = build_design_matrix(&x_values, &rand_k, &mode_weights);
        let sig_rand = singular_values(&a_rand);
        let k_rand = if sig_rand[sig_rand.len() - 1] > 1e-15 {
            sig_rand[0] / sig_rand[sig_rand.len() - 1]
        } else {
            f64::INFINITY
        };

        random_kappas.push(k_rand);
        random_r2_nfw.push(svd_projection_r_squared(&a_rand, &delta_nfw));
        random_r2_dc14.push(svd_projection_r_squared(&a_rand, &delta_dc14));
    }

    // Sort for percentile computation
    random_kappas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    random_r2_nfw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    random_r2_dc14.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_trials = cli.n_random_trials;
    let median_kappa = random_kappas[n_trials / 2];
    let p25_kappa = random_kappas[n_trials / 4];
    let p75_kappa = random_kappas[3 * n_trials / 4];
    let median_r2_nfw = random_r2_nfw[n_trials / 2];
    let median_r2_dc14 = random_r2_dc14[n_trials / 2];

    // Percentile rank of ZD kappa among random kappas
    let kappa_percentile =
        random_kappas.iter().filter(|&&k| k <= kappa).count() as f64 / n_trials as f64 * 100.0;
    let r2_nfw_percentile =
        random_r2_nfw.iter().filter(|&&r| r <= r2_nfw).count() as f64 / n_trials as f64 * 100.0;

    eprintln!(
        "Random kappa: median={:.2}, IQR=[{:.2}, {:.2}]",
        median_kappa, p25_kappa, p75_kappa
    );
    eprintln!(
        "ZD kappa={:.2} is at percentile {:.1}% of random",
        kappa, kappa_percentile
    );
    eprintln!("Random R^2(NFW): median={:.6}", median_r2_nfw);
    eprintln!(
        "ZD R^2(NFW)={:.6} is at percentile {:.1}%",
        r2_nfw, r2_nfw_percentile
    );
    eprintln!("Random R^2(DC14): median={:.6}", median_r2_dc14);

    // Detection verdict
    eprintln!("\n--- H1 Verdict ---");
    if kappa < 50.0 && r2_nfw > 0.85 {
        eprintln!("DETECTION: ZD modes form a well-conditioned baryonic basis");
        eprintln!("  kappa={:.2} < 50, R^2={:.4} > 0.85", kappa, r2_nfw);
    } else if kappa > 500.0 || r2_nfw < 0.3 {
        eprintln!("FAILURE: ZD modes do not span baryonic space");
        eprintln!("  kappa={:.2}, R^2={:.4}", kappa, r2_nfw);
    } else {
        eprintln!("INCONCLUSIVE: kappa={:.2}, R^2={:.4}", kappa, r2_nfw);
        eprintln!("  Between detection (kappa<50, R^2>0.85) and failure (kappa>500, R^2<0.3)");
    }

    // Is ZD privileged over random?
    if kappa_percentile < 25.0 && r2_nfw_percentile > 75.0 {
        eprintln!("ZD BASIS IS PRIVILEGED: lower kappa and higher R^2 than most random sets");
    } else if kappa_percentile > 75.0 || r2_nfw_percentile < 25.0 {
        eprintln!("ZD BASIS IS NOT PRIVILEGED: no advantage over random wavenumber sets");
    } else {
        eprintln!("ZD BASIS PRIVILEGE INCONCLUSIVE");
    }

    // Write output CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "metric",
        "zd_value",
        "random_median",
        "random_p25",
        "random_p75",
        "percentile",
    ])?;
    wtr.write_record(&[
        "kappa".to_string(),
        format!("{:.6}", kappa),
        format!("{:.6}", median_kappa),
        format!("{:.6}", p25_kappa),
        format!("{:.6}", p75_kappa),
        format!("{:.2}", kappa_percentile),
    ])?;
    wtr.write_record(&[
        "r2_nfw".to_string(),
        format!("{:.6}", r2_nfw),
        format!("{:.6}", median_r2_nfw),
        format!("{:.6}", random_r2_nfw[n_trials / 4]),
        format!("{:.6}", random_r2_nfw[3 * n_trials / 4]),
        format!("{:.2}", r2_nfw_percentile),
    ])?;
    wtr.write_record(&[
        "r2_dc14".to_string(),
        format!("{:.6}", r2_dc14),
        format!("{:.6}", median_r2_dc14),
        format!("{:.6}", random_r2_dc14[n_trials / 4]),
        format!("{:.6}", random_r2_dc14[3 * n_trials / 4]),
        format!(
            "{:.2}",
            random_r2_dc14.iter().filter(|&&r| r <= r2_dc14).count() as f64 / n_trials as f64
                * 100.0
        ),
    ])?;

    // Write singular values
    wtr.write_record(&[
        "n_valid_bins".to_string(),
        format!("{}", n_valid),
        "".to_string(),
        "".to_string(),
        "".to_string(),
        "".to_string(),
    ])?;
    for (i, &s) in svals.iter().enumerate() {
        wtr.write_record(&[
            format!("sigma_{}", i + 1),
            format!("{:.8e}", s),
            "".to_string(),
            "".to_string(),
            "".to_string(),
            "".to_string(),
        ])?;
    }
    wtr.flush()?;

    eprintln!("\nResults written to {:?}", cli.out_csv);

    Ok(())
}
