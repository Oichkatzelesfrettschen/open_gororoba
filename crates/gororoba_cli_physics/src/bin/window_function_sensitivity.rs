//! Window function sensitivity analysis for CD-ZD harmonic detection.
//!
//! Quantifies a fundamental blind spot: the data window (0.50-1.22 r/r_s,
//! L=0.72) covers only 10-72% of one wavelength for modes 1-7. This makes
//! the DFT ill-conditioned for long-wavelength modes.
//!
//! For each mode, this binary:
//!   1. Injects a unit-amplitude sinusoid on the actual data grid
//!   2. Computes the DFT recovery fraction (how much power is recovered)
//!   3. Weights the chi-squared test by recovery fractions
//!   4. Reports effective DOF and window-corrected detection threshold
//!
//! The contrarian claim: the existing chi-squared test (df=14) is diluted
//! by insensitive modes. Window-corrected DOF may be much less than 14,
//! changing the detection threshold.
//!
//! Reference: Contrarian hypothesis D3.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "window-function-sensitivity")]
#[command(about = "Window function response at CD-ZD wavenumbers")]
struct Cli {
    /// Input stacked CSV.
    #[arg(long)]
    stacked_csv: PathBuf,

    /// Minimum galaxies per x-bin.
    #[arg(long, default_value_t = 10)]
    min_per_bin: usize,
}

fn load_valid_bins(
    path: &std::path::Path,
    min_per_bin: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();

    let x_idx = headers.iter().position(|h| h == "x").unwrap();
    let d_idx = headers.iter().position(|h| h == "delta_stack").unwrap();
    let e_idx = headers.iter().position(|h| h == "delta_stack_err").unwrap();
    let n_idx = headers.iter().position(|h| h == "n_contributing").unwrap();

    let mut x = Vec::new();
    let mut delta = Vec::new();
    let mut err = Vec::new();
    let mut n_contrib = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        let nc: usize = rec.get(n_idx).unwrap_or("0").parse().unwrap_or(0);
        if nc >= min_per_bin {
            x.push(rec.get(x_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
            delta.push(rec.get(d_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
            err.push(rec.get(e_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
            n_contrib.push(nc);
        }
    }

    Ok((x, delta, err, n_contrib))
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== D3: Window Function Sensitivity Analysis ===");
    eprintln!();

    let (x, delta, err, _n_contrib) = load_valid_bins(&cli.stacked_csv, cli.min_per_bin)?;
    let n_bins = x.len();
    let x_min = x[0];
    let x_max = x[n_bins - 1];
    let window_length = x_max - x_min;

    eprintln!("Data window: x = [{:.4}, {:.4}], L = {:.4} r/r_s, {} bins",
        x_min, x_max, window_length, n_bins);
    eprintln!();

    let n_modes = 7;
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * std::f64::consts::PI * n as f64 / n_modes as f64)
        .collect();

    // Part 1: Wavelength coverage analysis.
    eprintln!("--- Part 1: Wavelength coverage ---");
    println!("mode,k,wavelength,cycles_in_window,window_fraction");

    for (i, &k) in wavenumbers.iter().enumerate() {
        let wavelength = 2.0 * std::f64::consts::PI / k;
        let cycles = window_length / wavelength;
        println!("{},{:.6},{:.4},{:.4},{:.4}", i + 1, k, wavelength, cycles, cycles);
        eprintln!("  Mode {}: lambda={:.3}, {:.1}% of one cycle in window",
            i + 1, wavelength, cycles * 100.0);
    }

    // Part 2: DFT injection response.
    // For each mode, inject a unit sinusoid and measure recovery.
    eprintln!();
    eprintln!("--- Part 2: DFT injection recovery on actual grid ---");
    println!("mode,k,injected_power,recovered_power,recovery_fraction");

    let mut recovery_fractions: Vec<f64> = Vec::new();

    for (i, &k) in wavenumbers.iter().enumerate() {
        // Inject: s(x_j) = cos(k * x_j) for all valid bins.
        // The "true" power for a unit sinusoid over infinite range is 0.5.
        // The DFT on a finite window recovers a fraction of this.
        let n = n_bins as f64;

        // DFT of injected signal.
        let re_inj: f64 = x.iter().map(|&xj| (k * xj).cos() * (k * xj).cos()).sum::<f64>() / n;
        let im_inj: f64 = x.iter().map(|&xj| (k * xj).cos() * (k * xj).sin()).sum::<f64>() / n;
        let recovered_power = re_inj * re_inj + im_inj * im_inj;

        // Also inject sin(k*x) to get the full complex response.
        let re_sin: f64 = x.iter().map(|&xj| (k * xj).sin() * (k * xj).cos()).sum::<f64>() / n;
        let im_sin: f64 = x.iter().map(|&xj| (k * xj).sin() * (k * xj).sin()).sum::<f64>() / n;
        let recovered_power_sin = re_sin * re_sin + im_sin * im_sin;

        // Average response (phase-independent).
        let avg_recovery = (recovered_power + recovered_power_sin) / 2.0;

        // The theoretical max for a full-window sinusoid is 0.25 (half-power in each quadrature).
        // Normalize by this.
        let recovery_frac = avg_recovery / 0.25;

        recovery_fractions.push(recovery_frac);

        println!("{},{:.6},{:.6e},{:.6e},{:.6}",
            i + 1, k, 0.25_f64, avg_recovery, recovery_frac);
        eprintln!("  Mode {}: recovery fraction = {:.4} ({:.1}%)",
            i + 1, recovery_frac, recovery_frac * 100.0);
    }

    // Part 3: Window-corrected chi-squared test.
    // Reweight the P1 whitened DFT by recovery fractions.
    eprintln!();
    eprintln!("--- Part 3: Window-corrected matched filter ---");

    // Compute whitened DFT (same as P1).
    let d_white: Vec<f64> = delta.iter().zip(err.iter())
        .map(|(d, e)| if *e > 1e-30 { d / e } else { 0.0 })
        .collect();

    let mut total_chi2_raw = 0.0_f64;
    let mut total_chi2_corrected = 0.0_f64;
    let mut effective_dof = 0.0_f64;

    println!("mode,k,chi2_raw,recovery_frac,chi2_corrected,weight");

    for (i, &k) in wavenumbers.iter().enumerate() {
        let re_w: f64 = d_white.iter().zip(x.iter())
            .map(|(d, xj)| d * (k * xj).cos())
            .sum();
        let im_w: f64 = d_white.iter().zip(x.iter())
            .map(|(d, xj)| d * (k * xj).sin())
            .sum();

        let var_re: f64 = x.iter().map(|xj| (k * xj).cos().powi(2)).sum();
        let var_im: f64 = x.iter().map(|xj| (k * xj).sin().powi(2)).sum();

        let chi2_mode = if var_re > 0.0 { re_w * re_w / var_re } else { 0.0 }
            + if var_im > 0.0 { im_w * im_w / var_im } else { 0.0 };

        total_chi2_raw += chi2_mode;

        // Window-corrected: weight by recovery fraction.
        let rf = recovery_fractions[i];
        let weight = rf.min(1.0); // cap at 1.0
        let chi2_corrected = chi2_mode * weight;
        total_chi2_corrected += chi2_corrected;
        effective_dof += 2.0 * weight; // 2 DOF per mode, weighted

        println!("{},{:.6},{:.4},{:.4},{:.4},{:.4}",
            i + 1, k, chi2_mode, rf, chi2_corrected, weight);
    }

    eprintln!();
    eprintln!("--- Summary ---");
    eprintln!("  Raw chi^2 = {:.2} (df = 14)", total_chi2_raw);
    eprintln!("  Window-corrected chi^2 = {:.2} (effective df = {:.2})",
        total_chi2_corrected, effective_dof);
    eprintln!("  Chi^2/df ratio: raw = {:.2}, corrected = {:.2}",
        total_chi2_raw / 14.0, total_chi2_corrected / effective_dof);

    // Part 4: Which modes are actually detectable?
    eprintln!();
    eprintln!("--- Part 4: Mode detectability classification ---");
    let mut n_detectable = 0;
    let mut n_marginal = 0;
    let mut n_undetectable = 0;

    for (i, rf) in recovery_fractions.iter().enumerate() {
        let classification = if *rf > 0.5 {
            n_detectable += 1;
            "DETECTABLE (>50% recovery)"
        } else if *rf > 0.2 {
            n_marginal += 1;
            "MARGINAL (20-50% recovery)"
        } else {
            n_undetectable += 1;
            "UNDETECTABLE (<20% recovery)"
        };
        eprintln!("  Mode {}: recovery={:.1}% -> {}", i + 1, rf * 100.0, classification);
    }

    eprintln!();
    eprintln!("=== Verdict ===");
    eprintln!("  Detectable modes: {}/7", n_detectable);
    eprintln!("  Marginal modes: {}/7", n_marginal);
    eprintln!("  Undetectable modes: {}/7", n_undetectable);
    eprintln!("  Effective DOF: {:.1} (vs nominal 14)", effective_dof);

    if n_undetectable > 0 {
        eprintln!("  CHALLENGE: {} modes are undetectable in the data window.", n_undetectable);
        eprintln!("  The chi^2 test is diluted by insensitive modes.");
        eprintln!("  Restricting to detectable modes would change the detection threshold.");
    }
    if effective_dof < 10.0 {
        eprintln!("  CRITICAL: effective DOF ({:.1}) is {:.0}% below nominal (14).",
            effective_dof, (1.0 - effective_dof / 14.0) * 100.0);
        eprintln!("  Detection thresholds derived from chi^2(14) are INVALID.");
    } else {
        eprintln!("  Window correction is moderate ({:.0}% DOF reduction).",
            (1.0 - effective_dof / 14.0) * 100.0);
        eprintln!("  Null result remains valid but with reduced statistical power.");
    }

    Ok(())
}
