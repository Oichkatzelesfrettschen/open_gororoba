//! Wow! + Sedenion Synthesis: Integrated pipeline combining all Sprint 49 tools.
//!
//! End-to-end analysis pipeline:
//! 1. Ingest BL filterbank data or synthetic test signal
//! 2. Run ghost frequency detection (spectral analysis)
//! 3. Apply Moonshine filterbank (Leech Lattice projection)
//! 4. Compute ultrametric structure on candidate feature vectors
//! 5. Consolidated report with pass/fail for each thesis

use algebra_core::experimental::leech_lattice::{
    HoleStatistics, LeechBasis, project_signal_chunk,
};
use clap::{Parser, Subcommand};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use spectral_core::ghost_spectral::{
    check_ghost, compute_power_spectrum, find_peaks, noise_floor, peak_fwhm, peak_snr, GHOST_FREQ,
};
use stats_core::ultrametric::ultrametric_fraction_test;
use std::io::Write;

#[derive(Parser)]
#[command(
    name = "wow-sedenion-synthesis",
    about = "Integrated Wow! + Sedenion analysis pipeline"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run full pipeline on synthetic test data.
    Synth {
        /// Number of samples in synthetic signal.
        #[arg(long, default_value = "4096")]
        n_samples: usize,
        /// Inject ghost frequency signal.
        #[arg(long, default_value = "true")]
        inject_ghost: bool,
        /// Output report path.
        #[arg(long, default_value = "data/csv/wow_sedenion_report.csv")]
        output: String,
    },
    /// Run pipeline on a rho trace CSV from zd-resonance-sweep.
    FromTrace {
        /// Input CSV with columns: step, rho_mean.
        #[arg(long)]
        input: String,
        /// Output report path.
        #[arg(long, default_value = "data/csv/wow_sedenion_trace_report.csv")]
        output: String,
    },
    /// Summary of all thesis results (reads existing CSVs).
    Summary {
        /// Directory containing result CSVs.
        #[arg(long, default_value = "data/csv")]
        dir: String,
    },
}

/// Results from a single pipeline run.
struct PipelineResult {
    // Spectral analysis
    ghost_detected: bool,
    ghost_freq: f64,
    ghost_snr: f64,
    ghost_fwhm: f64,
    top_peak_freq: f64,
    // Moonshine filterbank
    lattice_frac: f64,
    shallow_frac: f64,
    deep_frac: f64,
    // Ultrametric
    ultrametric_frac: f64,
    ultrametric_p_value: f64,
}

impl PipelineResult {
    fn header() -> &'static str {
        "ghost_detected,ghost_freq,ghost_snr,ghost_fwhm,top_peak_freq,lattice_frac,shallow_frac,deep_frac,ultrametric_frac,ultrametric_p_value"
    }

    fn to_csv_row(&self) -> String {
        format!(
            "{},{:.6},{:.4},{:.6},{:.6},{:.4},{:.4},{:.4},{:.6},{:.6}",
            self.ghost_detected,
            self.ghost_freq,
            self.ghost_snr,
            self.ghost_fwhm,
            self.top_peak_freq,
            self.lattice_frac,
            self.shallow_frac,
            self.deep_frac,
            self.ultrametric_frac,
            self.ultrametric_p_value,
        )
    }
}

/// Run the full analysis pipeline on a time series.
fn run_pipeline(signal: &[f64]) -> PipelineResult {
    // 1. Spectral analysis
    let (freqs, power) = compute_power_spectrum(signal);
    let peaks = find_peaks(&freqs, &power, 10);
    let nf = noise_floor(&power);
    let ghost = check_ghost(&peaks);

    let (ghost_detected, ghost_freq, ghost_snr, ghost_fwhm) = if let Some(gp) = ghost {
        let snr = peak_snr(gp, nf);
        let fwhm = peak_fwhm(&freqs, &power, gp.freq).unwrap_or(0.0);
        (true, gp.freq, snr, fwhm)
    } else {
        (false, 0.0, 0.0, 0.0)
    };

    let top_peak_freq = peaks.first().map_or(0.0, |p| p.freq);

    // 2. Moonshine filterbank (Leech Lattice projection)
    let basis = LeechBasis::new();
    let chunk_size = 24;
    let step = chunk_size / 2;
    let mut cvp_results = Vec::new();
    let mut i = 0;
    while i + chunk_size <= signal.len() {
        let chunk = &signal[i..i + chunk_size];
        let point = project_signal_chunk(chunk);
        let cvp = basis.cvp(&point);
        cvp_results.push(cvp);
        i += step;
    }
    let hole_stats = HoleStatistics::from_results(&cvp_results);

    // 3. Ultrametric analysis on spectral peak frequencies
    // Use 1D peak frequencies for ultrametric triple test
    let peak_freqs: Vec<f64> = peaks.iter().take(10).map(|p| p.freq).collect();

    let (ultra_frac, ultra_p) = if peak_freqs.len() >= 4 {
        let result = ultrametric_fraction_test(&peak_freqs, 1000, 100, 42);
        (result.ultrametric_fraction, result.p_value)
    } else {
        (0.0, 1.0)
    };

    PipelineResult {
        ghost_detected,
        ghost_freq,
        ghost_snr,
        ghost_fwhm,
        top_peak_freq,
        lattice_frac: hole_stats.lattice_fraction(),
        shallow_frac: hole_stats.shallow_fraction(),
        deep_frac: hole_stats.deep_fraction(),
        ultrametric_frac: ultra_frac,
        ultrametric_p_value: ultra_p,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Synth {
            n_samples,
            inject_ghost,
            output,
        } => {
            println!("=== Wow! + Sedenion Synthesis: Synthetic Pipeline ===");
            println!("Samples: {n_samples}, Inject ghost: {inject_ghost}");

            let mut rng = ChaCha8Rng::seed_from_u64(49);
            let mut signal = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let noise: f64 = rng.r#gen::<f64>() * 0.1 - 0.05;
                let base = 1.0 + noise;
                let ghost_component = if inject_ghost {
                    0.01 * (2.0 * std::f64::consts::PI * GHOST_FREQ * i as f64).sin()
                } else {
                    0.0
                };
                // Add some structure at various frequencies
                let structure = 0.005 * (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin()
                    + 0.003 * (2.0 * std::f64::consts::PI * 0.25 * i as f64).sin();
                signal.push(base + ghost_component + structure);
            }

            let result = run_pipeline(&signal);

            println!("\n--- Pipeline Results ---");
            println!(
                "Ghost: {} (freq={:.4}, SNR={:.2}, FWHM={:.4})",
                if result.ghost_detected { "DETECTED" } else { "NOT DETECTED" },
                result.ghost_freq,
                result.ghost_snr,
                result.ghost_fwhm
            );
            println!("Top peak freq: {:.4}", result.top_peak_freq);
            println!(
                "Moonshine: lattice={:.4}, shallow={:.4}, deep={:.4}",
                result.lattice_frac, result.shallow_frac, result.deep_frac
            );
            println!(
                "Ultrametric: fraction={:.4}, p-value={:.6}",
                result.ultrametric_frac, result.ultrametric_p_value
            );

            let mut file = std::fs::File::create(&output)?;
            writeln!(file, "{}", PipelineResult::header())?;
            writeln!(file, "{}", result.to_csv_row())?;
            println!("\nWrote {output}");
        }
        Cmd::FromTrace { input, output } => {
            println!("=== Wow! + Sedenion Synthesis: From Rho Trace ===");
            println!("Input: {input}");

            // Read rho trace CSV
            let mut rdr = csv::Reader::from_path(&input)?;
            let mut signal = Vec::new();
            for record in rdr.records() {
                let rec = record?;
                if let Ok(v) = rec[1].parse::<f64>() {
                    signal.push(v);
                }
            }
            println!("Read {} samples", signal.len());

            if signal.len() < 48 {
                eprintln!("ERROR: Need at least 48 samples for pipeline (got {})", signal.len());
                std::process::exit(1);
            }

            let result = run_pipeline(&signal);

            println!("\n--- Pipeline Results ---");
            println!(
                "Ghost: {} (freq={:.4}, SNR={:.2})",
                if result.ghost_detected { "DETECTED" } else { "NOT DETECTED" },
                result.ghost_freq,
                result.ghost_snr
            );
            println!(
                "Moonshine: lattice={:.4}, deep={:.4}",
                result.lattice_frac, result.deep_frac
            );
            println!(
                "Ultrametric: fraction={:.4}, p={:.6}",
                result.ultrametric_frac, result.ultrametric_p_value
            );

            let mut file = std::fs::File::create(&output)?;
            writeln!(file, "{}", PipelineResult::header())?;
            writeln!(file, "{}", result.to_csv_row())?;
            println!("\nWrote {output}");
        }
        Cmd::Summary { dir } => {
            println!("=== Sprint 49: Thesis Summary ===\n");

            // Try to read each result file and report status
            let files = [
                ("C-774 ZD Resonance", "zd_resonance_sweep.csv"),
                ("C-774 Control", "zd_resonance_control.csv"),
                ("C-776 Pathion Sink", "pathion_sink_compare.csv"),
                ("C-777 Chirality Lock", "chirality_matrix.csv"),
                ("C-778 Moonshine", "moonshine_synth.csv"),
                ("Pipeline", "wow_sedenion_report.csv"),
            ];

            for (label, filename) in &files {
                let path = format!("{dir}/{filename}");
                if std::path::Path::new(&path).exists() {
                    println!("  [{:^8}] {label}: {path}", "FOUND");
                } else {
                    println!("  [{:^8}] {label}: {path}", "MISSING");
                }
            }

            println!("\nRun individual binaries to generate missing results:");
            println!("  cargo run --release --bin zd-resonance-sweep -- sweep");
            println!("  cargo run --release --bin zd-resonance-sweep -- control");
            println!("  cargo run --release --bin pathion-sink-test -- compare");
            println!("  cargo run --release --bin chirality-lock-test -- matrix");
            println!("  cargo run --release --bin moonshine-filterbank -- synth");
            println!("  cargo run --release --bin wow-sedenion-synthesis -- synth");
        }
    }

    Ok(())
}
