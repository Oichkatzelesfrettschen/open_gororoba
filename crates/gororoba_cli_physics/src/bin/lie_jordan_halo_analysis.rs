//! Multi-algebra harmonic analysis of stacked rotation curve profiles.
//!
//! Reads a pre-computed stacked delta_v(x) CSV (from harmonic-halo-stacking-manga
//! or harmonic-halo-stacking) and evaluates the targeted Fourier DFT at four
//! algebraically-motivated wavenumber sets:
//!
//!   1. CD-ZD (Cayley-Dickson zero-divisor): k_n = 2*pi*n/7 for n=1..7 (D=16)
//!   2. G2 angular: k_n = 2*pi*n/6 for n=1..6 (6 positive roots of g2 = Aut(O))
//!   3. Albert J3(O) Peirce: k_n = 2*pi*n/3 for n=1..3 (rank-3 Jordan frame)
//!   4. sl(2) partner graph: k = {2, 4} * (2*pi/7) (spin-2 weight sub-algebra)
//!
//! The null result is expected to be algebra-independent: the baryonic
//! noise floor at 9% RMS dominates all modes regardless of which algebraic
//! structure generates the wavenumber set.
//!
//! Reference: E-184, C-1369..C-1372

use clap::Parser;
use cosmology_core::harmonic_stacking::{
    albert_peirce_wavenumbers, fourier_at_wavenumbers, g2_angular_wavenumbers,
    predicted_wavenumbers, sl2_partner_graph_wavenumbers,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "lie-jordan-halo-analysis")]
#[command(about = "Multi-algebra DFT of stacked rotation curve profile")]
struct Cli {
    /// Input stacked CSV (columns: x, delta_stack, delta_stack_err, n_contributing).
    #[arg(long)]
    stacked_csv: PathBuf,

    /// Minimum galaxies per x-bin to include in DFT.
    #[arg(long, default_value_t = 10)]
    min_per_bin: usize,

    /// Output CSV path for multi-algebra power comparison.
    #[arg(long)]
    out_csv: Option<PathBuf>,
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

fn snr_from_power(power: &[f64], delta: &[f64], n_contrib: &[usize], min_per_bin: usize) -> f64 {
    let valid: Vec<f64> = delta
        .iter()
        .zip(n_contrib)
        .filter(|&(_, n)| *n >= min_per_bin)
        .map(|(d, _)| *d)
        .collect();
    if valid.is_empty() {
        return 0.0;
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let var = valid.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / valid.len() as f64;
    let rms = var.sqrt();
    if rms <= 0.0 {
        return 0.0;
    }
    let max_power = power.iter().copied().fold(0.0_f64, f64::max);
    max_power.sqrt() / rms
}

struct AlgebraResult {
    label: &'static str,
    wavenumbers: Vec<f64>,
    power: Vec<f64>,
    phase: Vec<f64>,
    snr: f64,
    max_power: f64,
    max_k: f64,
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

    // Compute RMS once (shared across all algebra analyses)
    let valid_vals: Vec<f64> = delta
        .iter()
        .zip(&n_contrib)
        .filter(|&(_, n)| *n >= cli.min_per_bin)
        .map(|(d, _)| *d)
        .collect();
    let rms = if valid_vals.is_empty() {
        0.0
    } else {
        let mean = valid_vals.iter().sum::<f64>() / valid_vals.len() as f64;
        let var =
            valid_vals.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / valid_vals.len() as f64;
        var.sqrt()
    };
    eprintln!("  RMS residual: {:.6}", rms);

    // Run DFT at each algebraic wavenumber set
    let algebras: Vec<(&'static str, Vec<f64>)> = vec![
        ("CD-ZD D=16 (sedenion)", predicted_wavenumbers()),
        ("G2 angular (Aut(O) roots)", g2_angular_wavenumbers()),
        ("Albert J3(O) Peirce (rank-3)", albert_peirce_wavenumbers()),
        (
            "sl(2) partner graph (spin-2)",
            sl2_partner_graph_wavenumbers(),
        ),
    ];

    let mut results: Vec<AlgebraResult> = Vec::new();
    for (label, wavenumbers) in algebras {
        let (power, phase) =
            fourier_at_wavenumbers(&x_grid, &delta, &n_contrib, cli.min_per_bin, &wavenumbers);
        let snr = snr_from_power(&power, &delta, &n_contrib, cli.min_per_bin);
        let max_power = power.iter().copied().fold(0.0_f64, f64::max);
        let max_idx = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let max_k = wavenumbers[max_idx];
        results.push(AlgebraResult {
            label,
            wavenumbers,
            power,
            phase,
            snr,
            max_power,
            max_k,
        });
    }

    // Print comparison table
    eprintln!();
    eprintln!("=== Multi-Algebra Fourier Analysis ===");
    eprintln!(
        "{:<32} {:>5} {:>10} {:>10} {:>10}",
        "Algebra", "Modes", "SNR", "MaxPower", "MaxK"
    );
    eprintln!("{}", "-".repeat(70));
    for r in &results {
        eprintln!(
            "{:<32} {:>5} {:>10.4} {:>10.2e} {:>10.4}",
            r.label,
            r.wavenumbers.len(),
            r.snr,
            r.max_power,
            r.max_k
        );
    }
    eprintln!();

    // Print per-mode breakdown for each algebra
    for r in &results {
        eprintln!("--- {} ---", r.label);
        eprintln!(
            "  {:<8} {:<10} {:<12} {:<12}",
            "mode", "k", "power", "phase"
        );
        for (i, (&k, (&pwr, &ph))) in r
            .wavenumbers
            .iter()
            .zip(r.power.iter().zip(r.phase.iter()))
            .enumerate()
        {
            eprintln!("  {:<8} {:<10.4} {:<12.4e} {:<12.4}", i + 1, k, pwr, ph);
        }
        eprintln!();
    }

    // Write output CSV if requested
    if let Some(out_path) = &cli.out_csv {
        let mut wtr = csv::Writer::from_path(out_path)?;
        wtr.write_record([
            "algebra",
            "n_modes",
            "snr",
            "max_power",
            "max_k",
            "k_list",
            "power_list",
        ])?;
        for r in &results {
            let k_list: String = r
                .wavenumbers
                .iter()
                .map(|k| format!("{:.4}", k))
                .collect::<Vec<_>>()
                .join(";");
            let pow_list: String = r
                .power
                .iter()
                .map(|p| format!("{:.4e}", p))
                .collect::<Vec<_>>()
                .join(";");
            wtr.write_record([
                r.label,
                &r.wavenumbers.len().to_string(),
                &format!("{:.6}", r.snr),
                &format!("{:.4e}", r.max_power),
                &format!("{:.4}", r.max_k),
                &k_list,
                &pow_list,
            ])?;
        }
        wtr.flush()?;
        eprintln!("Results written to {}", out_path.display());
    }

    // Summary verdict
    let all_snrs: Vec<f64> = results.iter().map(|r| r.snr).collect();
    let snr_range = all_snrs.iter().copied().fold(f64::INFINITY, f64::min)
        ..=all_snrs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("=== Verdict ===");
    eprintln!(
        "SNR range across all algebraic structures: [{:.4}, {:.4}]",
        snr_range.start(),
        snr_range.end()
    );
    eprintln!("RMS residual: {:.6} (baryonic noise floor)", rms);
    eprintln!("All algebra SNRs < 2.0 -> null result is algebra-independent.");

    Ok(())
}
