//! Synthesis audit: resolves the two unresolved disagreements from the
//! nine-hypothesis analysis.
//!
//! S1: Robust-estimator ZD bound comparison.
//!     Converts D4's trimmed DFT power (from per-galaxy CSV) into alpha_zd
//!     bounds at mean, 5%-trimmed, 10%-trimmed, 20%-trimmed, and median levels.
//!
//! S2: Spectral leakage decomposition.
//!     Simulates white noise on the actual data grid, computes the expected
//!     leakage spectrum, and decomposes the observed red-noise into leakage
//!     vs intrinsic components.
//!
//! Reference: Synthesis hypotheses S1 and S2 from hypothesis_synthesis_2026.md.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "synthesis-audit")]
#[command(about = "S1 robust bounds + S2 spectral leakage decomposition")]
struct Cli {
    /// Input stacked CSV.
    #[arg(long)]
    stacked_csv: PathBuf,

    /// Per-galaxy DFT CSV.
    #[arg(long)]
    galaxies_csv: PathBuf,

    /// Minimum galaxies per x-bin.
    #[arg(long, default_value_t = 10)]
    min_per_bin: usize,

    /// Number of white-noise simulations for leakage estimate.
    #[arg(long, default_value_t = 10000)]
    n_sim: usize,

    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn load_valid_bins(
    path: &std::path::Path,
    min_per_bin: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();

    let x_idx = headers.iter().position(|h| h == "x").unwrap();
    let d_idx = headers.iter().position(|h| h == "delta_stack").unwrap();
    let e_idx = headers.iter().position(|h| h == "delta_stack_err").unwrap();
    let n_idx = headers.iter().position(|h| h == "n_contributing").unwrap();

    let mut x = Vec::new();
    let mut delta = Vec::new();
    let mut err = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        let nc: usize = rec.get(n_idx).unwrap_or("0").parse().unwrap_or(0);
        if nc >= min_per_bin {
            x.push(rec.get(x_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
            delta.push(rec.get(d_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
            err.push(rec.get(e_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
        }
    }

    Ok((x, delta, err))
}

fn trimmed_mean_complex(values: &[(f64, f64)], trim_frac: f64) -> (f64, f64) {
    let n = values.len();
    let trim_count = (n as f64 * trim_frac) as usize;

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| {
        let pa = a.0 * a.0 + a.1 * a.1;
        let pb = b.0 * b.0 + b.1 * b.1;
        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let trimmed = &sorted[trim_count..(n - trim_count)];
    let n_t = trimmed.len() as f64;
    if n_t == 0.0 {
        return (0.0, 0.0);
    }
    let re = trimmed.iter().map(|(r, _)| r).sum::<f64>() / n_t;
    let im = trimmed.iter().map(|(_, i)| i).sum::<f64>() / n_t;
    (re, im)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== SYNTHESIS AUDIT: S1 + S2 ===");
    eprintln!();

    // ==================== S1: ROBUST-ESTIMATOR ZD BOUNDS ====================
    eprintln!("=== S1: Robust-Estimator ZD Bound Comparison ===");
    eprintln!();

    // Load per-galaxy DFT coefficients.
    let mut rdr = csv::Reader::from_path(&cli.galaxies_csv)?;
    let mut cd_coeffs: Vec<(f64, f64)> = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        let re: f64 = rec.get(1).unwrap_or("0").parse().unwrap_or(0.0);
        let im: f64 = rec.get(2).unwrap_or("0").parse().unwrap_or(0.0);
        cd_coeffs.push((re, im));
    }

    let n_gal = cd_coeffs.len();
    eprintln!("  {} galaxies loaded", n_gal);

    // Compute mean, trimmed means, and median.
    let trim_levels = [0.0_f64, 0.01, 0.05, 0.10, 0.20];
    let trim_labels = [
        "mean (full)",
        "1%-trimmed",
        "5%-trimmed",
        "10%-trimmed",
        "20%-trimmed",
    ];

    // Alpha_zd conversion: alpha_zd = sqrt(power) / af * exp(1)
    // where af = 0.5 (assessor fraction for D=16 sedenion).
    let af = 0.5_f64;
    let exp1 = 1.0_f64.exp();

    println!("estimator,n_used,mean_re,mean_im,power,amplitude,alpha_zd_est,bound_ratio");

    let full_re = cd_coeffs.iter().map(|(r, _)| r).sum::<f64>() / n_gal as f64;
    let full_im = cd_coeffs.iter().map(|(_, i)| i).sum::<f64>() / n_gal as f64;
    let full_power = full_re * full_re + full_im * full_im;
    let full_alpha = full_power.sqrt() / af * exp1;

    for (idx, &trim_frac) in trim_levels.iter().enumerate() {
        let (re, im) = if trim_frac == 0.0 {
            (full_re, full_im)
        } else {
            trimmed_mean_complex(&cd_coeffs, trim_frac)
        };

        let power = re * re + im * im;
        let amplitude = power.sqrt();
        let alpha_zd = amplitude / af * exp1;
        let n_used = n_gal - 2 * ((n_gal as f64 * trim_frac) as usize);
        let bound_ratio = if full_alpha > 1e-30 {
            alpha_zd / full_alpha
        } else {
            0.0
        };

        println!(
            "{},{},{:.8e},{:.8e},{:.8e},{:.8e},{:.6},{:.4}",
            trim_labels[idx], n_used, re, im, power, amplitude, alpha_zd, bound_ratio
        );

        eprintln!(
            "  {}: alpha_zd = {:.6}, ratio vs full = {:.4}",
            trim_labels[idx], alpha_zd, bound_ratio
        );
    }

    // Median.
    let mut sorted_re: Vec<f64> = cd_coeffs.iter().map(|(r, _)| *r).collect();
    sorted_re.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_im: Vec<f64> = cd_coeffs.iter().map(|(_, i)| *i).collect();
    sorted_im.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = n_gal / 2;
    let med_re = sorted_re[mid];
    let med_im = sorted_im[mid];
    let med_power = med_re * med_re + med_im * med_im;
    let med_alpha = med_power.sqrt() / af * exp1;
    let med_ratio = if full_alpha > 1e-30 {
        med_alpha / full_alpha
    } else {
        0.0
    };

    println!(
        "median,{},{:.8e},{:.8e},{:.8e},{:.8e},{:.6},{:.4}",
        n_gal,
        med_re,
        med_im,
        med_power,
        med_power.sqrt(),
        med_alpha,
        med_ratio
    );
    eprintln!(
        "  median: alpha_zd = {:.6}, ratio vs full = {:.4}",
        med_alpha, med_ratio
    );

    eprintln!();
    eprintln!("  S1 Verdict:");
    if med_ratio < 0.33 {
        eprintln!(
            "  DIVERGENCE: median bound is {:.0}% of mean bound.",
            med_ratio * 100.0
        );
        eprintln!("  The mean-based bound is UNRELIABLE (outlier-dominated).");
        eprintln!("  Paper MUST report both mean and median bounds.");
    } else {
        eprintln!("  CONVERGENCE: median bound is within 3x of mean bound.");
        eprintln!("  Outlier domination does not critically affect the sensitivity claim.");
    }

    // ==================== S2: SPECTRAL LEAKAGE DECOMPOSITION ====================
    eprintln!();
    eprintln!("=== S2: Spectral Leakage Decomposition ===");
    eprintln!();

    let (x, delta, err) = load_valid_bins(&cli.stacked_csv, cli.min_per_bin)?;
    let n_bins = x.len();

    let n_modes = 7;
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * std::f64::consts::PI * n as f64 / n_modes as f64)
        .collect();

    // Compute observed whitened DFT power at each mode.
    let d_white: Vec<f64> = delta
        .iter()
        .zip(err.iter())
        .map(|(d, e)| if *e > 1e-30 { d / e } else { 0.0 })
        .collect();

    let mut observed_power: Vec<f64> = Vec::new();
    for &k in &wavenumbers {
        let re: f64 = d_white
            .iter()
            .zip(x.iter())
            .map(|(d, xj)| d * (k * xj).cos())
            .sum();
        let im: f64 = d_white
            .iter()
            .zip(x.iter())
            .map(|(d, xj)| d * (k * xj).sin())
            .sum();
        observed_power.push(re * re + im * im);
    }

    // Simulate white noise: N_sim realizations of unit-variance white noise
    // on the same x-grid, compute DFT power at each mode, average.
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);
    let normal = StandardNormal;

    let mut leakage_power = vec![0.0_f64; n_modes];

    for _ in 0..cli.n_sim {
        // Generate white noise on the data grid.
        let noise: Vec<f64> = (0..n_bins).map(|_| normal.sample(&mut rng)).collect();

        for (mode_idx, &k) in wavenumbers.iter().enumerate() {
            let re: f64 = noise
                .iter()
                .zip(x.iter())
                .map(|(n, xj)| n * (k * xj).cos())
                .sum();
            let im: f64 = noise
                .iter()
                .zip(x.iter())
                .map(|(n, xj)| n * (k * xj).sin())
                .sum();
            leakage_power[mode_idx] += re * re + im * im;
        }
    }

    // Average leakage power.
    for p in &mut leakage_power {
        *p /= cli.n_sim as f64;
    }

    // Decompose: intrinsic = observed - leakage.
    eprintln!("  {} white-noise simulations on {} bins", cli.n_sim, n_bins);
    eprintln!();

    println!(
        "mode,k,observed_power,leakage_power,intrinsic_power,leakage_fraction,intrinsic_normalized"
    );

    let mut total_leakage_frac = 0.0_f64;
    let mut intrinsic_powers: Vec<f64> = Vec::new();

    for (i, &k) in wavenumbers.iter().enumerate() {
        let obs = observed_power[i];
        let leak = leakage_power[i];
        let intrinsic = (obs - leak).max(0.0);
        let leak_frac = if obs > 1e-30 { leak / obs } else { 0.0 };
        // Normalize intrinsic by expected chi^2(2) = 2*sigma^2.
        let intrinsic_norm = intrinsic / leak;

        intrinsic_powers.push(intrinsic);
        total_leakage_frac += leak_frac;

        println!(
            "{},{:.6},{:.4e},{:.4e},{:.4e},{:.6},{:.6}",
            i + 1,
            k,
            obs,
            leak,
            intrinsic,
            leak_frac,
            intrinsic_norm
        );

        eprintln!(
            "  Mode {}: observed={:.2e}, leakage={:.2e} ({:.1}%), intrinsic={:.2e}, norm={:.2}",
            i + 1,
            obs,
            leak,
            leak_frac * 100.0,
            intrinsic,
            intrinsic_norm
        );
    }

    let mean_leakage_frac = total_leakage_frac / n_modes as f64;

    // Fit red-noise to intrinsic spectrum.
    let log_k: Vec<f64> = wavenumbers.iter().map(|k| k.ln()).collect();
    let log_intr: Vec<f64> = intrinsic_powers.iter().map(|p| (p + 1e-30).ln()).collect();

    let n = log_k.len() as f64;
    let mean_lk = log_k.iter().sum::<f64>() / n;
    let mean_li = log_intr.iter().sum::<f64>() / n;
    let cov = log_k
        .iter()
        .zip(log_intr.iter())
        .map(|(lk, li)| (lk - mean_lk) * (li - mean_li))
        .sum::<f64>();
    let var_lk: f64 = log_k.iter().map(|lk| (lk - mean_lk).powi(2)).sum();
    let gamma_intrinsic = if var_lk > 1e-30 { -cov / var_lk } else { 0.0 };

    eprintln!();
    eprintln!("--- Spectral slopes ---");
    eprintln!("  Observed red-noise gamma: ~0.78 (from P1/C-1423)");
    eprintln!("  Mean leakage fraction: {:.1}%", mean_leakage_frac * 100.0);
    eprintln!(
        "  Intrinsic red-noise gamma (after leakage removal): {:.4}",
        gamma_intrinsic
    );

    eprintln!();
    eprintln!("  S2 Verdict:");
    if mean_leakage_frac > 0.80 {
        eprintln!(
            "  Leakage accounts for >{:.0}% of observed power.",
            mean_leakage_frac * 100.0
        );
        eprintln!("  The red-noise spectrum is PREDOMINANTLY a leakage artifact.");
        eprintln!("  The 'baryonic floor' spectral shape is mostly window-induced.");
    } else if gamma_intrinsic > 0.3 {
        eprintln!(
            "  Intrinsic gamma = {:.4} > 0.3: genuine red-noise survives leakage removal.",
            gamma_intrinsic
        );
        eprintln!("  The baryonic floor has REAL spectral structure beyond leakage.");
        eprintln!(
            "  P1's characterization (k^{{-0.78}}) is partially intrinsic, partially leakage."
        );
    } else {
        eprintln!(
            "  Intrinsic gamma = {:.4} <= 0.3: red-noise is largely explained by leakage.",
            gamma_intrinsic
        );
        eprintln!("  Residual spectral structure is weak.");
    }

    Ok(())
}
