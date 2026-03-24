use anyhow::{Context, Result};
use clap::Parser;
use std::{fs::File, path::PathBuf};

use stats_core::quantile_regression::quantile_periodogram_at_freq;

const ALIASED_GHOST_FREQ: f64 = 1.0 - 0.786_151_377_757_423;

#[derive(Parser, Debug)]
#[command(author, version, about = "Quantile Periodogram Spectral Analysis")]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long, default_value = "value")]
    column: String,

    #[arg(
        short,
        long,
        value_delimiter = ',',
        default_value = "0.1,0.25,0.5,0.75,0.9"
    )]
    quantiles: String,

    #[arg(short, long, default_value_t = 100)]
    n_freq: usize,
}

fn fisher_g_test(periodogram: &[f64]) -> (f64, f64) {
    let mut max_p = 0.0;
    let mut sum_p = 0.0;
    for &p in periodogram {
        if p > max_p {
            max_p = p;
        }
        sum_p += p;
    }

    if sum_p <= 0.0 {
        return (0.0, 1.0);
    }

    let g = max_p / sum_p;
    let m = periodogram.len() as f64;

    // Exact p-value: P(G > g) = sum_{k=1..floor(1/g)} (-1)^{k+1} * C(M,k) * (1-kg)^{M-1}
    let mut p_value = 0.0;
    let limit = (1.0 / g).floor() as usize;

    for k in 1..=limit.min(periodogram.len()) {
        let term = 1.0 - k as f64 * g;
        if term <= 0.0 {
            break;
        }

        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };

        // C(M,k) can be large, use logs
        let mut log_comb = 0.0;
        for i in 0..k {
            log_comb += (m - i as f64).ln() - (i as f64 + 1.0).ln();
        }

        let log_term = log_comb + (m - 1.0) * term.ln();
        p_value += sign * log_term.exp();
    }

    (g, p_value.clamp(0.0, 1.0))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let quantiles: Vec<f64> = args
        .quantiles
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("=== Quantile Periodogram Analysis (Pure Rust) ===");
    println!("Input: {}", args.input.display());

    let file = File::open(&args.input).context("Failed to open input file")?;
    let mut rdr = csv::ReaderBuilder::new().from_reader(file);
    let headers = rdr.headers()?.clone();
    let col_idx = headers.iter().position(|h| h == args.column);

    let mut signal = Vec::new();
    for result in rdr.records() {
        let record = result?;
        // Find column index
        let val = if let Some(idx) = col_idx {
            record.get(idx)
        } else {
            record.get(1) // fallback
        };

        if let Some(v_str) = val
            && let Ok(v) = v_str.parse::<f64>()
            && v.is_finite()
        {
            signal.push(v);
        }
    }

    if signal.is_empty() {
        anyhow::bail!("No valid data found in column {}", args.column);
    }

    let n = signal.len();
    let freq_min = (1.0 / n as f64).max(0.01);
    let freq_max = 0.5;

    // Grid around target freq
    let freq_tol = 0.02;
    let mut freqs = Vec::new();
    for i in 0..args.n_freq {
        freqs.push(freq_min + (freq_max - freq_min) * (i as f64) / (args.n_freq as f64 - 1.0));
    }
    for i in 0..20 {
        let f = (ALIASED_GHOST_FREQ - freq_tol) + (2.0 * freq_tol) * (i as f64) / 19.0;
        if f >= freq_min && f <= freq_max {
            freqs.push(f);
        }
    }
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    freqs.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    println!(
        "N={}, {} frequency bins, quantiles={:?}",
        n,
        freqs.len(),
        quantiles
    );

    let mut combined_fisher_inputs = Vec::new();

    for &tau in &quantiles {
        println!("  Analyzing tau={:.2}...", tau);
        let mut powers = Vec::with_capacity(freqs.len());
        for &f in &freqs {
            powers.push(quantile_periodogram_at_freq(&signal, f, tau));
        }

        let (g, p_fisher) = fisher_g_test(&powers);

        let mut max_p = -1.0;
        let mut max_f = 0.0;
        for (i, &p) in powers.iter().enumerate() {
            if p > max_p {
                max_p = p;
                max_f = freqs[i];
            }
        }

        println!("    Fisher g={:.6}, p={:.6e}", g, p_fisher);
        println!(
            "    Global max at f={:.6} (target={:.6})",
            max_f, ALIASED_GHOST_FREQ
        );

        combined_fisher_inputs.push(p_fisher);
    }

    // Combine across quantiles using Fisher's method
    let combined_chi2 = -2.0
        * combined_fisher_inputs
            .iter()
            .map(|&p| p.max(1e-300).ln())
            .sum::<f64>();

    // Chi-squared p-value (df = 2 * n_quantiles)
    // We can use statrs for this
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    let chi = ChiSquared::new(2.0 * quantiles.len() as f64).unwrap();
    let combined_p = 1.0 - chi.cdf(combined_chi2);

    println!("\nSummary:");
    println!("  Combined Fisher chi2: {:.4}", combined_chi2);
    println!("  Combined p-value: {:.6e}", combined_p);

    if combined_p < 0.05 {
        println!("  Verdict: SIGNIFICANT");
    } else {
        println!("  Verdict: NOT SIGNIFICANT");
    }

    Ok(())
}
