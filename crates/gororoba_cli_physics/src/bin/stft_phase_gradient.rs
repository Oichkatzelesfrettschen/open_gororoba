//! STFT phase-gradient analysis (chirp detection, Hypothesis H4).
//!
//! Tests whether the STFT phases show a linear gradient
//! `phi(x) = a + b*x` consistent with coherent harmonic forcing
//! (ZD prediction: slope `b = k`). Baryonic contamination produces
//! phase jumps or flat profiles.
//!
//! Detection: `R^2 > 0.7` AND fitted slope within 20% of `k`
//! for at least 3 modes.
//! Rejection: `R^2 < 0.3` for all 7 modes.

use anyhow::{Context, Result};
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use std::{
    collections::BTreeMap,
    f64::consts::PI,
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(name = "stft-phase-gradient")]
#[command(about = "STFT phase-gradient chirp test for MaNGA harmonic stacking")]
struct Cli {
    /// Input STFT CSV with columns x_center,mode,k,power,phase,n_eff.
    #[arg(long, default_value = "data/results/e183/stft_full.csv")]
    csv: PathBuf,

    /// Minimum effective weight required to keep a row.
    #[arg(long, default_value_t = 5.0)]
    min_n_eff: f64,

    /// Output CSV path. Defaults beside the input file.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Default)]
struct ModeSeries {
    x: Vec<f64>,
    phase: Vec<f64>,
    k: f64,
}

#[derive(Debug)]
struct PhaseGradientRow {
    mode: u32,
    k: f64,
    wavelength: f64,
    n_pts: usize,
    slope: f64,
    intercept: f64,
    r_squared: f64,
    slope_over_k: f64,
    n_cycles: f64,
    verdict: &'static str,
}

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len();
    if n < 3 {
        return (0.0, 0.0, 0.0);
    }

    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|xi| xi * xi).sum();
    let sxy: f64 = x.iter().zip(y).map(|(xi, yi)| xi * yi).sum();

    let denom = n as f64 * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return (0.0, 0.0, 0.0);
    }

    let slope = (n as f64 * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n as f64;

    let y_mean = sy / n as f64;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
        .sum();
    let r_squared = if ss_tot > 1e-30 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (slope, intercept, r_squared)
}

fn unwrap_phases(phases: &[f64]) -> Vec<f64> {
    if phases.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(phases.len());
    out.push(phases[0]);
    for pair in phases.windows(2) {
        let mut diff = pair[1] - pair[0];
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
        }
        let prev = *out.last().unwrap_or(&phases[0]);
        out.push(prev + diff);
    }
    out
}

fn default_out_path(input: &Path) -> PathBuf {
    if let Some(name) = input.file_name().and_then(|s| s.to_str())
        && name == "stft_full.csv"
    {
        return input.with_file_name("stft_phase_gradient.csv");
    }
    input.with_extension("phase_gradient.csv")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let out_path = cli
        .out
        .clone()
        .unwrap_or_else(|| default_out_path(&cli.csv));

    let file =
        File::open(&cli.csv).with_context(|| format!("opening STFT CSV {}", cli.csv.display()))?;
    let mut reader = ReaderBuilder::new().from_reader(file);

    let mut modes: BTreeMap<u32, ModeSeries> = BTreeMap::new();
    for row in reader.deserialize::<BTreeMap<String, String>>() {
        let row = row.with_context(|| format!("reading {}", cli.csv.display()))?;
        let mode: u32 = row["mode"].parse().context("parsing mode")?;
        let x: f64 = row["x_center"].parse().context("parsing x_center")?;
        let phase: f64 = row["phase"].parse().context("parsing phase")?;
        let k: f64 = row["k"].parse().context("parsing k")?;
        let n_eff: f64 = row["n_eff"].parse().context("parsing n_eff")?;
        if n_eff < cli.min_n_eff {
            continue;
        }
        let entry = modes.entry(mode).or_default();
        entry.x.push(x);
        entry.phase.push(phase);
        entry.k = k;
    }

    println!("{}", "=".repeat(72));
    println!("STFT Phase Gradient Analysis (Hypothesis H4: Chirp Detection)");
    println!("{}", "=".repeat(72));
    println!();
    println!(
        "{:>5}  {:>8}  {:>10}  {:>5}  {:>10}  {:>8}  {:>8}  {:>8}  {:>12}",
        "mode", "k", "wavelength", "n_pts", "slope", "k_pred", "slope/k", "R^2", "verdict"
    );
    println!("{}", "-".repeat(95));

    let mut rows = Vec::new();
    for (mode, series) in modes {
        let phases = unwrap_phases(&series.phase);
        let (slope, intercept, r_squared) = linear_regression(&series.x, &phases);
        let wavelength = if series.k > 0.0 {
            2.0 * PI / series.k
        } else {
            f64::INFINITY
        };
        let x_range = match (series.x.first(), series.x.last()) {
            (Some(lo), Some(hi)) => hi - lo,
            _ => 0.0,
        };
        let n_cycles = if wavelength.is_finite() && wavelength > 0.0 {
            x_range / wavelength
        } else {
            0.0
        };
        let slope_over_k = if series.k.abs() > 1e-10 {
            slope / series.k
        } else {
            f64::NAN
        };
        let verdict = if r_squared > 0.7 && (0.8..1.2).contains(&slope_over_k.abs()) {
            "COHERENT"
        } else if r_squared < 0.3 {
            "NOISE"
        } else if n_cycles < 0.5 {
            "DEGENERATE"
        } else {
            "AMBIGUOUS"
        };

        println!(
            "{:>5}  {:>8.4}  {:>10.4}  {:>5}  {:>10.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>12}",
            mode,
            series.k,
            wavelength,
            series.x.len(),
            slope,
            series.k,
            slope_over_k,
            r_squared,
            verdict
        );

        rows.push(PhaseGradientRow {
            mode,
            k: series.k,
            wavelength,
            n_pts: series.x.len(),
            slope,
            intercept,
            r_squared,
            slope_over_k,
            n_cycles,
            verdict,
        });
    }

    println!();
    println!("{}", "=".repeat(72));
    println!("Summary");
    println!("{}", "=".repeat(72));

    let coherent_modes = rows.iter().filter(|r| r.verdict == "COHERENT").count();
    let noise_modes = rows.iter().filter(|r| r.verdict == "NOISE").count();
    let degenerate_modes = rows.iter().filter(|r| r.verdict == "DEGENERATE").count();
    let max_r2 = rows.iter().map(|r| r.r_squared).fold(0.0_f64, f64::max);

    println!(
        "  Coherent modes (R^2 > 0.7, slope/k in [0.8, 1.2]): {}",
        coherent_modes
    );
    println!("  Noise modes (R^2 < 0.3): {}", noise_modes);
    println!(
        "  Degenerate modes (< 0.5 cycles in window): {}",
        degenerate_modes
    );
    println!("  Max R^2: {:.4}", max_r2);
    println!();

    if coherent_modes >= 3 {
        println!("RESULT: DETECTION -- coherent phase gradient found in >= 3 modes");
    } else if noise_modes == rows.len() {
        println!("RESULT: REJECTED -- all modes show noise-like phase");
    } else if max_r2 < 0.3 {
        println!("RESULT: REJECTED -- no mode exceeds R^2 = 0.3");
    } else {
        println!("RESULT: INCONCLUSIVE -- some modes show structure but not enough for detection");
    }

    let out_file = File::create(&out_path)
        .with_context(|| format!("creating phase-gradient CSV {}", out_path.display()))?;
    let mut writer = WriterBuilder::new().from_writer(out_file);
    writer.write_record([
        "mode",
        "k",
        "wavelength",
        "n_pts",
        "slope",
        "intercept",
        "r_squared",
        "slope_over_k",
        "n_cycles",
        "verdict",
    ])?;
    for row in &rows {
        writer.write_record([
            row.mode.to_string(),
            row.k.to_string(),
            row.wavelength.to_string(),
            row.n_pts.to_string(),
            row.slope.to_string(),
            row.intercept.to_string(),
            row.r_squared.to_string(),
            row.slope_over_k.to_string(),
            row.n_cycles.to_string(),
            row.verdict.to_string(),
        ])?;
    }
    writer.flush()?;

    println!("\nResults written to {}", out_path.display());
    Ok(())
}
