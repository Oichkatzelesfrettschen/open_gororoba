//! Non-static signal analysis of stacked rotation curve profiles.
//!
//! Applies three complementary non-static diagnostics to a pre-computed
//! stacked delta_v(x) CSV (from harmonic-halo-stacking-manga or
//! harmonic-halo-stacking):
//!
//!   1. STFT spectrogram: sliding Gaussian-windowed DFT revealing WHERE
//!      in x = r/r_s space harmonic power is concentrated. Baryonic
//!      features (bulge excess, NFW cusp trough) are localized at x < 2.
//!      A genuine ZD forcing signal would be distributed across x > 2.
//!
//!   2. Derivative stacking: d(delta_stack)/dx via central finite differences.
//!      Annihilates the DC baryonic baseline (-9% constant offset), exposing
//!      oscillatory content. If harmonic mode k is present, the derivative
//!      power at k scales as k^2 times the original power.
//!
//!   3. Jackknife phase coherence: drop each x-bin in turn, recompute the
//!      complex Fourier coefficient at each CD-ZD wavenumber. The Rayleigh R
//!      statistic tests whether phases cluster (coherent signal, R -> 1)
//!      or scatter randomly (noise, R -> 0). Under H0 (uniform phases):
//!      p ~= exp(-n * R^2).
//!
//! # Interpretation guide
//!
//! If ZD forcing is physical:
//!   - STFT power should be broadly distributed over x > 1 (halo-scale),
//!     not concentrated only at x < 2 (bulge).
//!   - Derivative DFT should show excess at CD-ZD wavenumbers k_n = 2*pi*n/7.
//!   - Rayleigh R > 0.5 with p < 0.05 at the predicted modes.
//!
//! If the signal is baryonic noise:
//!   - STFT power concentrated at x < 2 (bulge and cusp region).
//!   - Derivative DFT red-noise spectrum with monotonic 1/k falloff.
//!   - Rayleigh R < 0.3 with p > 0.3 at all modes.
//!
//! Reference: E-192, C-1373, C-1369..C-1372

use clap::Parser;
use cosmology_core::harmonic_stacking::{
    derivative_stack, jackknife_phase_coherence, predicted_wavenumbers, stft_spectrogram,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "harmonic-halo-signal-analysis")]
#[command(about = "Non-static STFT + derivative + Rayleigh phase coherence analysis")]
struct Cli {
    /// Input stacked CSV (columns: x, delta_stack, delta_stack_err, n_contributing).
    #[arg(long)]
    stacked_csv: PathBuf,

    /// Minimum galaxies per x-bin to include in analysis.
    #[arg(long, default_value_t = 10)]
    min_per_bin: usize,

    /// Gaussian window half-width for STFT in r/r_s units (default 1.5).
    #[arg(long, default_value_t = 1.5)]
    sigma_x: f64,

    /// Number of STFT window centers spanning the x range (default 32).
    #[arg(long, default_value_t = 32)]
    n_windows: usize,

    /// Output CSV for STFT spectrogram (x_center, k, power, phase, n_eff).
    #[arg(long)]
    stft_csv: Option<PathBuf>,

    /// Output CSV for derivative stacking (x_mid, d_delta_dx).
    #[arg(long)]
    deriv_csv: Option<PathBuf>,

    /// Output CSV for Rayleigh phase coherence (k, rayleigh_r, rayleigh_p).
    #[arg(long)]
    rayleigh_csv: Option<PathBuf>,
}

fn load_stacked_csv(path: &std::path::Path) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();

    let x_idx = headers
        .iter()
        .position(|h| h == "x")
        .ok_or_else(|| anyhow::anyhow!("No 'x' column in {}", path.display()))?;
    let delta_idx = headers
        .iter()
        .position(|h| h == "delta_stack")
        .ok_or_else(|| anyhow::anyhow!("No 'delta_stack' column in {}", path.display()))?;
    let n_idx = headers
        .iter()
        .position(|h| h == "n_contributing")
        .ok_or_else(|| anyhow::anyhow!("No 'n_contributing' column in {}", path.display()))?;

    let mut x_grid = Vec::new();
    let mut delta = Vec::new();
    let mut n_contrib = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        let x: f64 = rec.get(x_idx).unwrap_or("0").parse().unwrap_or(0.0);
        let d: f64 = rec.get(delta_idx).unwrap_or("0").parse().unwrap_or(0.0);
        let n: usize = rec.get(n_idx).unwrap_or("0").parse().unwrap_or(0);
        x_grid.push(x);
        delta.push(d);
        n_contrib.push(n);
    }

    Ok((x_grid, delta, n_contrib))
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!(
        "Loading stacked profile from {}...",
        cli.stacked_csv.display()
    );
    let (x_grid, delta, n_contrib) = load_stacked_csv(&cli.stacked_csv)?;

    let valid_bins = n_contrib.iter().filter(|&&n| n >= cli.min_per_bin).count();
    eprintln!(
        "  {} total bins, {} with >= {} galaxies",
        x_grid.len(),
        valid_bins,
        cli.min_per_bin
    );

    let wavenumbers = predicted_wavenumbers();
    eprintln!(
        "  Using {} CD-ZD wavenumbers (sedenion box-kite modes)",
        wavenumbers.len()
    );

    // ----- 1. STFT spectrogram -----
    eprintln!();
    eprintln!(
        "=== STFT Spectrogram (sigma_x={:.2}, n_windows={}) ===",
        cli.sigma_x, cli.n_windows
    );

    let spec = stft_spectrogram(
        &x_grid,
        &delta,
        &n_contrib,
        cli.min_per_bin,
        &wavenumbers,
        cli.sigma_x,
        cli.n_windows,
    );

    // Find peak power location for each wavenumber.
    eprintln!(
        "{:<6} {:>8} {:>10} {:>10} {:>10}",
        "mode", "k", "x_peak", "peak_pwr", "pwr_ratio"
    );
    eprintln!("{}", "-".repeat(50));
    for (ki, &k) in wavenumbers.iter().enumerate() {
        // Power at each window center for this mode.
        let pw: Vec<f64> = spec.power.iter().map(|row| row[ki]).collect();
        let total_pwr: f64 = pw.iter().sum();
        if total_pwr < 1e-30 {
            continue;
        }
        let (peak_ci, &peak_pwr) = pw
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let x_peak = spec.x_centers[peak_ci];
        // Fraction of power at x < 2 (baryonic region).
        let baryonic_pwr: f64 = spec
            .x_centers
            .iter()
            .zip(&pw)
            .filter(|(xc, _)| **xc < 2.0)
            .map(|(_, p)| *p)
            .sum();
        let baryonic_frac = baryonic_pwr / total_pwr;
        eprintln!(
            "{:<6} {:>8.4} {:>10.4} {:>10.4e} {:>10.3} (baryonic_frac={:.3})",
            ki + 1,
            k,
            x_peak,
            peak_pwr,
            peak_pwr / total_pwr,
            baryonic_frac
        );
    }

    if let Some(path) = &cli.stft_csv {
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record(["x_center", "mode", "k", "power", "phase", "n_eff"])?;
        for (ci, &x_c) in spec.x_centers.iter().enumerate() {
            for (ki, &k) in spec.wavenumbers.iter().enumerate() {
                wtr.write_record([
                    &format!("{:.6}", x_c),
                    &format!("{}", ki + 1),
                    &format!("{:.6}", k),
                    &format!("{:.6e}", spec.power[ci][ki]),
                    &format!("{:.6}", spec.phase[ci][ki]),
                    &format!("{:.4}", spec.n_eff[ci]),
                ])?;
            }
        }
        wtr.flush()?;
        eprintln!("STFT spectrogram written to {}", path.display());
    }

    // ----- 2. Derivative stacking -----
    eprintln!();
    eprintln!("=== Derivative Stacking (d(delta_stack)/dx) ===");

    let (x_deriv, dy) = derivative_stack(&x_grid, &delta, &n_contrib, cli.min_per_bin);

    if x_deriv.is_empty() {
        eprintln!("  Not enough valid bins for derivative computation.");
    } else {
        let dy_rms = {
            let mean = dy.iter().sum::<f64>() / dy.len() as f64;
            let var = dy.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / dy.len() as f64;
            var.sqrt()
        };
        let dy_max = dy.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let dy_min = dy.iter().copied().fold(f64::INFINITY, f64::min);
        eprintln!(
            "  {} derivative points, RMS={:.4e}, range=[{:.4e}, {:.4e}]",
            dy.len(),
            dy_rms,
            dy_min,
            dy_max
        );

        // DFT of derivative at CD-ZD wavenumbers.
        use cosmology_core::harmonic_stacking::fourier_at_wavenumbers;
        let ones: Vec<usize> = vec![1_usize; x_deriv.len()];
        let (pwr_deriv, _) = fourier_at_wavenumbers(&x_deriv, &dy, &ones, 1, &wavenumbers);
        let max_deriv_pwr = pwr_deriv.iter().copied().fold(0.0_f64, f64::max);
        eprintln!(
            "  Derivative DFT at CD-ZD modes (max power = {:.4e}):",
            max_deriv_pwr
        );
        eprintln!(
            "  {:<6} {:>8} {:>12} {:>8}",
            "mode", "k", "deriv_pwr", "k^2 ratio"
        );
        for (ki, (&k, &p)) in wavenumbers.iter().zip(&pwr_deriv).enumerate() {
            // Expected scaling: power_deriv ~ k^2 * power_orig
            // If signal: ratio should be ~1. If noise: ratio is unpredictable.
            let k2_factor = k * k;
            eprintln!(
                "  {:<6} {:>8.4} {:>12.4e} {:>8.4}",
                ki + 1,
                k,
                p,
                p / k2_factor
            );
        }

        if let Some(path) = &cli.deriv_csv {
            let mut wtr = csv::Writer::from_path(path)?;
            wtr.write_record(["x_mid", "d_delta_dx"])?;
            for (&x, &d) in x_deriv.iter().zip(&dy) {
                wtr.write_record([&format!("{:.6}", x), &format!("{:.8e}", d)])?;
            }
            wtr.flush()?;
            eprintln!("Derivative profile written to {}", path.display());
        }
    }

    // ----- 3. Jackknife phase coherence -----
    eprintln!();
    eprintln!("=== Jackknife Phase Coherence (Rayleigh R test) ===");
    eprintln!("  H0: DFT phases are uniformly distributed across jackknife samples");
    eprintln!("  H1: Phases cluster (R > 0.5, p < 0.05) => coherent signal");

    let coherence =
        jackknife_phase_coherence(&x_grid, &delta, &n_contrib, cli.min_per_bin, &wavenumbers);

    eprintln!(
        "  n_jackknife = {} (one drop per valid bin)",
        coherence.n_jackknife
    );
    eprintln!(
        "  {:<6} {:>8} {:>10} {:>12} {:>10}",
        "mode", "k", "Rayleigh_R", "p-value", "verdict"
    );
    eprintln!("{}", "-".repeat(55));
    for (ki, &k) in wavenumbers.iter().enumerate() {
        let r = coherence.rayleigh_r[ki];
        let p = coherence.rayleigh_p[ki];
        let verdict = if p < 0.01 {
            "COHERENT **"
        } else if p < 0.05 {
            "marginal *"
        } else {
            "noise"
        };
        eprintln!(
            "  {:<6} {:>8.4} {:>10.4} {:>12.4e} {:>10}",
            ki + 1,
            k,
            r,
            p,
            verdict
        );
    }

    if let Some(path) = &cli.rayleigh_csv {
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record(["mode", "k", "rayleigh_r", "rayleigh_p", "n_jackknife"])?;
        for (ki, &k) in wavenumbers.iter().enumerate() {
            wtr.write_record([
                &format!("{}", ki + 1),
                &format!("{:.6}", k),
                &format!("{:.6}", coherence.rayleigh_r[ki]),
                &format!("{:.6e}", coherence.rayleigh_p[ki]),
                &format!("{}", coherence.n_jackknife),
            ])?;
        }
        wtr.flush()?;
        eprintln!("Rayleigh coherence results written to {}", path.display());
    }

    // ----- Summary verdict -----
    eprintln!();
    eprintln!("=== Non-Static Analysis Verdict ===");
    let max_r = coherence.rayleigh_r.iter().cloned().fold(0.0_f64, f64::max);
    let min_p = coherence.rayleigh_p.iter().cloned().fold(1.0_f64, f64::min);
    eprintln!("  Max Rayleigh R across all modes: {:.4}", max_r);
    eprintln!("  Min p-value across all modes:    {:.4e}", min_p);
    if min_p < 0.05 {
        eprintln!("  => Marginal or significant phase coherence detected.");
        eprintln!("     Warrants investigation: check STFT for x-localization.");
    } else {
        eprintln!("  => No phase coherence (p > 0.05 for all modes).");
        eprintln!("     Consistent with baryonic noise-dominated null result.");
    }

    Ok(())
}
