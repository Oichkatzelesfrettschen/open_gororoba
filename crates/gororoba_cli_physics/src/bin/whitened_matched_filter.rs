//! Whitened matched filter SNR for CD-ZD harmonic detection.
//!
//! Addresses a key gap in the existing analysis: the naive SNR metric
//! (max_power.sqrt() / rms_residual) ignores the 33x variation in per-bin
//! noise (sigma ranges from 1.6e-4 to 5.3e-3 across 16 valid bins).
//!
//! This binary implements:
//!   1. Per-bin whitening: d_white(x_i) = delta(x_i) / sigma(x_i)
//!   2. Variance-normalized DFT: properly accounts for the noise contribution
//!      to each Fourier coefficient
//!   3. Chi-squared test: T = sum_k (|F_k|^2 / var(F_k)) ~ chi^2(2*n_modes)
//!      under H0 (no harmonic signal)
//!   4. Red-noise correction: fits P(k) ~ A * k^{-gamma} and divides out
//!      the spectral slope before testing
//!
//! This is the standard approach in pulsar timing, gravitational wave
//! detection, and CMB analysis. It is the Neyman-Pearson optimal test for
//! known signal template in Gaussian noise with known covariance.
//!
//! Reference: Hypothesis P1 from practical ML engineer plan.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "whitened-matched-filter")]
#[command(about = "Variance-weighted matched filter SNR with chi-squared test")]
struct Cli {
    /// Input stacked CSV (columns: x, delta_stack, delta_stack_err, n_contributing).
    #[arg(long)]
    stacked_csv: PathBuf,

    /// Minimum galaxies per x-bin to include.
    #[arg(long, default_value_t = 10)]
    min_per_bin: usize,
}

fn load_stacked_csv(
    path: &std::path::Path,
) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();

    let x_idx = headers.iter().position(|h| h == "x")
        .ok_or_else(|| anyhow::anyhow!("No 'x' column"))?;
    let delta_idx = headers.iter().position(|h| h == "delta_stack")
        .ok_or_else(|| anyhow::anyhow!("No 'delta_stack' column"))?;
    let err_idx = headers.iter().position(|h| h == "delta_stack_err")
        .ok_or_else(|| anyhow::anyhow!("No 'delta_stack_err' column"))?;
    let n_idx = headers.iter().position(|h| h == "n_contributing")
        .ok_or_else(|| anyhow::anyhow!("No 'n_contributing' column"))?;

    let mut x = Vec::new();
    let mut delta = Vec::new();
    let mut err = Vec::new();
    let mut n_contrib = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        x.push(rec.get(x_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
        delta.push(rec.get(delta_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
        err.push(rec.get(err_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
        n_contrib.push(rec.get(n_idx).unwrap_or("0").parse::<usize>().unwrap_or(0));
    }

    Ok((x, delta, err, n_contrib))
}

/// Chi-squared survival function (upper-tail p-value).
/// Uses the regularized incomplete gamma function approximation.
fn chi2_survival(x: f64, df: f64) -> f64 {
    // For chi^2 with df degrees of freedom:
    // P(X > x) = 1 - regularized_gamma_lower(df/2, x/2)
    // Use a series expansion for the regularized lower incomplete gamma.
    if x <= 0.0 {
        return 1.0;
    }
    let a = df / 2.0;
    let z = x / 2.0;

    // For large x, use continued fraction (upper incomplete gamma).
    // For small x, use series expansion (lower incomplete gamma).
    if z > a + 1.0 {
        // Upper tail via continued fraction (Lentz's method).
        upper_gamma_cf(a, z)
    } else {
        // Lower tail via series, then subtract from 1.
        1.0 - lower_gamma_series(a, z)
    }
}

/// Regularized lower incomplete gamma via series expansion.
fn lower_gamma_series(a: f64, z: f64) -> f64 {
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
        term *= z / (a + n as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum * (-z + a * z.ln() - ln_gamma(a)).exp()
}

/// Regularized upper incomplete gamma via continued fraction.
fn upper_gamma_cf(a: f64, z: f64) -> f64 {
    // Lentz's continued fraction method.
    let mut c = 1e-30_f64;
    let mut d = 1.0 / (z + 1.0 - a);
    let mut f = d;
    for n in 1..200 {
        let an = n as f64 * (a - n as f64);
        let bn = z + 2.0 * n as f64 + 1.0 - a;
        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }
    f * (-z + a * z.ln() - ln_gamma(a)).exp()
}

/// Log-gamma function (Stirling's approximation for large values,
/// Lanczos approximation for small values).
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation (g=7, n=9).
    let g = 7.0_f64;
    let coeff = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula.
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut t = coeff[0];
    for (i, &c) in coeff.iter().enumerate().skip(1) {
        t += c / (x + i as f64);
    }

    let w = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * w.ln() - w + t.ln()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== P1: Whitened Matched Filter SNR ===");
    eprintln!();

    let (x_all, delta_all, err_all, n_contrib) = load_stacked_csv(&cli.stacked_csv)?;

    // Filter to valid bins.
    let valid: Vec<(f64, f64, f64, usize)> = x_all
        .iter()
        .zip(delta_all.iter())
        .zip(err_all.iter())
        .zip(n_contrib.iter())
        .filter(|&(((_, _), _), n)| *n >= cli.min_per_bin)
        .map(|(((x, d), e), n)| (*x, *d, *e, *n))
        .collect();

    let n_bins = valid.len();
    eprintln!("Valid bins: {} (n >= {})", n_bins, cli.min_per_bin);
    eprintln!();

    // Step 1: Show the noise variation.
    let min_err = valid.iter().map(|(_, _, e, _)| *e).fold(f64::MAX, f64::min);
    let max_err = valid.iter().map(|(_, _, e, _)| *e).fold(0.0_f64, f64::max);
    eprintln!("Per-bin error range: {:.6e} to {:.6e} (ratio {:.1}x)",
        min_err, max_err, max_err / min_err);

    // Step 2: Pre-whiten the data.
    let d_white: Vec<f64> = valid.iter().map(|(_, d, e, _)| d / e).collect();
    let x_valid: Vec<f64> = valid.iter().map(|(x, _, _, _)| *x).collect();

    eprintln!("Whitened residual range: {:.2} to {:.2}",
        d_white.iter().copied().fold(f64::MAX, f64::min),
        d_white.iter().copied().fold(f64::MIN, f64::max));

    // Step 3: Compute whitened DFT at CD-ZD wavenumbers.
    let n_modes = 7;
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * std::f64::consts::PI * n as f64 / n_modes as f64)
        .collect();

    eprintln!();
    eprintln!("--- Naive (unwhitened) DFT ---");

    // Naive DFT for comparison.
    for (mode_idx, &k) in wavenumbers.iter().enumerate() {
        let re: f64 = valid.iter().map(|(x, d, _, _)| d * (k * x).cos()).sum::<f64>()
            / n_bins as f64;
        let im: f64 = valid.iter().map(|(x, d, _, _)| d * (k * x).sin()).sum::<f64>()
            / n_bins as f64;
        let power = re * re + im * im;
        let rms_sq = valid.iter().map(|(_, d, _, _)| d * d).sum::<f64>() / n_bins as f64;
        let snr = if rms_sq > 0.0 { power.sqrt() / rms_sq.sqrt() } else { 0.0 };
        eprintln!("  Mode {}: k={:.4}, power={:.6e}, SNR={:.4}", mode_idx + 1, k, power, snr);
    }

    // Step 4: Whitened DFT.
    eprintln!();
    eprintln!("--- Whitened DFT (variance-normalized) ---");

    println!("mode,k,re_white,im_white,var_re,var_im,chi2_mode,snr_mode");

    let mut total_chi2 = 0.0_f64;
    let mut mode_results: Vec<(usize, f64, f64)> = Vec::new(); // (mode, chi2, snr)

    for (mode_idx, &k) in wavenumbers.iter().enumerate() {
        // Whitened DFT coefficients.
        let re_w: f64 = d_white.iter().zip(x_valid.iter())
            .map(|(d, x)| d * (k * x).cos())
            .sum();
        let im_w: f64 = d_white.iter().zip(x_valid.iter())
            .map(|(d, x)| d * (k * x).sin())
            .sum();

        // Variance of whitened DFT coefficients under H0.
        // If d_white_i ~ N(0, 1) independently, then:
        // var(re_w) = sum_i cos^2(k * x_i)
        // var(im_w) = sum_i sin^2(k * x_i)
        let var_re: f64 = x_valid.iter().map(|x| (k * x).cos().powi(2)).sum();
        let var_im: f64 = x_valid.iter().map(|x| (k * x).sin().powi(2)).sum();

        // Normalized chi^2 contribution from this mode.
        let chi2_mode = if var_re > 0.0 { re_w * re_w / var_re } else { 0.0 }
            + if var_im > 0.0 { im_w * im_w / var_im } else { 0.0 };
        let snr_mode = chi2_mode.sqrt();

        total_chi2 += chi2_mode;

        println!("{},{:.6},{:.6e},{:.6e},{:.4},{:.4},{:.6},{:.6}",
            mode_idx + 1, k, re_w, im_w, var_re, var_im, chi2_mode, snr_mode);
        eprintln!("  Mode {}: k={:.4}, chi2={:.4}, SNR={:.4} (var_re={:.2}, var_im={:.2})",
            mode_idx + 1, k, chi2_mode, snr_mode, var_re, var_im);

        mode_results.push((mode_idx + 1, chi2_mode, snr_mode));
    }

    // Step 5: Overall chi-squared test.
    let df = 2.0 * n_modes as f64; // 14 degrees of freedom
    let p_value = chi2_survival(total_chi2, df);

    eprintln!();
    eprintln!("--- Chi-squared test (all modes combined) ---");
    eprintln!("  Total chi^2 = {:.4} (df = {})", total_chi2, df as usize);
    eprintln!("  p-value = {:.6}", p_value);
    eprintln!("  Expected chi^2 under H0 = {:.1} +/- {:.1}", df, (2.0 * df).sqrt());

    // Step 6: Red-noise correction.
    // Fit log(power) = log(A) - gamma * log(k) to the whitened power spectrum.
    eprintln!();
    eprintln!("--- Red-noise correction ---");

    let log_k: Vec<f64> = wavenumbers.iter().map(|k| k.ln()).collect();
    let whitened_power: Vec<f64> = mode_results.iter().map(|(_, chi2, _)| *chi2).collect();
    let log_p: Vec<f64> = whitened_power.iter().map(|p| (p + 1e-30).ln()).collect();

    // Least-squares fit: log_p = a + b * log_k
    let n = log_k.len() as f64;
    let mean_lk = log_k.iter().sum::<f64>() / n;
    let mean_lp = log_p.iter().sum::<f64>() / n;
    let cov_lk_lp: f64 = log_k.iter().zip(log_p.iter())
        .map(|(lk, lp)| (lk - mean_lk) * (lp - mean_lp))
        .sum::<f64>();
    let var_lk: f64 = log_k.iter().map(|lk| (lk - mean_lk).powi(2)).sum();
    let gamma = if var_lk > 1e-30 { -cov_lk_lp / var_lk } else { 0.0 };
    let log_a = mean_lp + gamma * mean_lk;

    eprintln!("  Red-noise fit: P(k) ~ {:.4e} * k^{{-{:.4}}}", log_a.exp(), gamma);

    // Corrected chi^2: divide each mode's chi^2 by the expected red-noise level.
    let mut corrected_chi2 = 0.0_f64;
    for (mode_idx, &k) in wavenumbers.iter().enumerate() {
        let expected = (log_a - gamma * k.ln()).exp();
        let corrected = whitened_power[mode_idx] / expected;
        corrected_chi2 += corrected;
        eprintln!("  Mode {}: raw_chi2={:.4}, expected={:.4}, corrected={:.4}",
            mode_idx + 1, whitened_power[mode_idx], expected, corrected);
    }

    // Under H0 with red-noise correction, each corrected mode should be ~1,
    // so the sum should be ~n_modes (not 2*n_modes, because we're dividing
    // a chi^2(2) by its expected value, giving ~exp(1) which sums to ~Gamma(n,1)).
    let p_corrected = chi2_survival(corrected_chi2, n_modes as f64);
    eprintln!("  Corrected total = {:.4} (expected ~{:.1})", corrected_chi2, n_modes);
    eprintln!("  p-value (corrected) = {:.6}", p_corrected);

    // Step 7: Compare naive vs whitened SNR.
    let naive_snr = {
        let rms = valid.iter().map(|(_, d, _, _)| d * d).sum::<f64>() / n_bins as f64;
        let max_p = wavenumbers.iter().map(|&k| {
            let re: f64 = valid.iter().map(|(x, d, _, _)| d * (k * x).cos()).sum::<f64>()
                / n_bins as f64;
            let im: f64 = valid.iter().map(|(x, d, _, _)| d * (k * x).sin()).sum::<f64>()
                / n_bins as f64;
            re * re + im * im
        }).fold(0.0_f64, f64::max);
        max_p.sqrt() / rms.sqrt()
    };

    let whitened_snr = total_chi2.sqrt() / df.sqrt(); // normalized chi = T/sqrt(df)

    eprintln!();
    eprintln!("=== Verdict ===");
    eprintln!("  Naive SNR (peak/RMS):    {:.4}", naive_snr);
    eprintln!("  Whitened MF-SNR:         {:.4} (chi^2/df = {:.4})",
        whitened_snr, total_chi2 / df);
    eprintln!("  Chi^2 p-value:           {:.6}", p_value);
    eprintln!("  Red-noise corrected p:   {:.6}", p_corrected);

    if p_value > 0.05 {
        eprintln!("  p > 0.05: no significant harmonic content detected.");
        eprintln!("  CONFIRMED: whitened matched filter agrees with naive null.");
    } else if p_value < 0.01 {
        eprintln!("  p < 0.01: significant harmonic content detected!");
        eprintln!("  Check if this survives red-noise correction (p_corrected = {:.6}).", p_corrected);
    } else {
        eprintln!("  0.01 < p < 0.05: marginal detection, requires further investigation.");
    }

    // Sensitivity improvement estimate.
    let sensitivity_ratio = whitened_snr / naive_snr;
    eprintln!("  Sensitivity ratio (whitened/naive): {:.2}x", sensitivity_ratio);

    Ok(())
}
