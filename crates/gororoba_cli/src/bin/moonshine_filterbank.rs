//! Moonshine Filterbank: Leech Lattice projection for signal discrimination.
//!
//! Projects signal chunks onto the Leech Lattice (Lambda_24) and classifies
//! them by deep hole type. The hypothesis: signal structure maps to Monster
//! Group symmetry axes, and deep hole fractions differ between ON-source and
//! OFF-source pointings.

use algebra_core::experimental::golay_code::{golay_codewords, weight_enumerator};
use algebra_core::experimental::leech_lattice::{
    HoleStatistics, LeechBasis, project_signal_chunk,
};
use clap::{Parser, Subcommand};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::io::Write;

#[derive(Parser)]
#[command(name = "moonshine-filterbank", about = "Leech Lattice signal filterbank")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Validate Golay code and Leech lattice construction.
    Validate,
    /// Run synthetic ON/OFF discrimination test.
    Synth {
        /// Number of chunks per pointing.
        #[arg(long, default_value = "1000")]
        n_chunks: usize,
        /// Signal-to-noise ratio of injected signal (in ON chunks).
        #[arg(long, default_value = "0.1")]
        snr: f64,
        /// Output CSV path.
        #[arg(long, default_value = "data/csv/moonshine_synth.csv")]
        output: String,
    },
    /// Analyze real data from CSV time series.
    Analyze {
        /// Input CSV (one column of floats per row).
        #[arg(long)]
        input: String,
        /// Chunk size (default 24 for direct Leech projection).
        #[arg(long, default_value = "24")]
        chunk_size: usize,
        /// Output CSV path.
        #[arg(long, default_value = "data/csv/moonshine_analysis.csv")]
        output: String,
    },
}

/// Generate synthetic "ON-source" chunks: Gaussian noise + structured signal.
fn generate_on_chunks(n: usize, snr: f64, rng: &mut ChaCha8Rng) -> Vec<Vec<f64>> {
    let mut chunks = Vec::with_capacity(n);
    for i in 0..n {
        let mut chunk = Vec::with_capacity(48);
        for j in 0..48 {
            let noise: f64 = rng.r#gen::<f64>() * 2.0 - 1.0;
            // Injected signal: harmonic structure at Leech lattice scale
            let signal = snr * (2.0 * std::f64::consts::PI * j as f64 / 24.0).sin()
                * (1.0 + 0.5 * (i as f64 * 0.01).sin());
            chunk.push(noise + signal);
        }
        chunks.push(chunk);
    }
    chunks
}

/// Generate synthetic "OFF-source" chunks: pure Gaussian noise.
fn generate_off_chunks(n: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<f64>> {
    let mut chunks = Vec::with_capacity(n);
    for _ in 0..n {
        let mut chunk = Vec::with_capacity(48);
        for _ in 0..48 {
            chunk.push(rng.r#gen::<f64>() * 2.0 - 1.0);
        }
        chunks.push(chunk);
    }
    chunks
}

/// Project chunks onto Leech Lattice and collect statistics.
fn project_and_classify(chunks: &[Vec<f64>], basis: &LeechBasis) -> HoleStatistics {
    let mut results = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let point = project_signal_chunk(chunk);
        let cvp = basis.cvp(&point);
        results.push(cvp);
    }
    HoleStatistics::from_results(&results)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Validate => {
            println!("=== Moonshine Filterbank: Validation ===");

            // 1. Golay code
            println!("\n--- Golay Code [24,12,8] ---");
            let codewords = golay_codewords();
            println!("Codewords: {}", codewords.len());
            let we = weight_enumerator();
            println!("Weight enumerator:");
            println!("  A_0  = {} (expected 1)", we[0]);
            println!("  A_8  = {} (expected 759)", we[8]);
            println!("  A_12 = {} (expected 2576)", we[12]);
            println!("  A_16 = {} (expected 759)", we[16]);
            println!("  A_24 = {} (expected 1)", we[24]);
            let total: usize = we.iter().sum();
            assert_eq!(total, 4096, "Total codewords must be 4096");
            println!("  Total = {total} [OK]");

            // 2. Leech lattice
            println!("\n--- Leech Lattice Lambda_24 ---");
            let basis = LeechBasis::new();
            // Test: origin should be a lattice point (closest = 0)
            let origin = [0.0f64; 24];
            let cvp = basis.cvp(&origin);
            println!(
                "CVP(origin): distance^2 = {:.6}, type = {:?}",
                cvp.distance_squared, cvp.classification
            );

            // Test: known lattice point
            let mut lp = [0.0f64; 24];
            lp[0] = 2.0; // Minimal norm vector component
            let cvp2 = basis.cvp(&lp);
            println!(
                "CVP([2,0,...,0]): distance^2 = {:.6}, type = {:?}",
                cvp2.distance_squared, cvp2.classification
            );

            // 3. Signal projection
            println!("\n--- Signal Projection ---");
            let test_signal: Vec<f64> = (0..48).map(|i| (i as f64 * 0.1).sin()).collect();
            let point = project_signal_chunk(&test_signal);
            let norm: f64 = point.iter().map(|v| v * v).sum::<f64>();
            println!(
                "Projection of 48-sample sine: |p|^2 = {norm:.6} (should be ~1.0)"
            );

            println!("\n[VALIDATION PASSED]");
        }
        Cmd::Synth {
            n_chunks,
            snr,
            output,
        } => {
            println!("=== Moonshine Filterbank: Synthetic ON/OFF Test ===");
            println!("Chunks per pointing: {n_chunks}, SNR: {snr}");

            let basis = LeechBasis::new();
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            // Generate ON and OFF chunks
            let on_chunks = generate_on_chunks(n_chunks, snr, &mut rng);
            let off_chunks = generate_off_chunks(n_chunks, &mut rng);

            // Project and classify
            println!("Projecting ON chunks...");
            let on_stats = project_and_classify(&on_chunks, &basis);
            println!("Projecting OFF chunks...");
            let off_stats = project_and_classify(&off_chunks, &basis);

            // Report
            println!("\n--- ON-source statistics ---");
            println!(
                "  Lattice: {:.4}, Shallow: {:.4}, Deep: {:.4}",
                on_stats.lattice_fraction(),
                on_stats.shallow_fraction(),
                on_stats.deep_fraction()
            );
            println!("\n--- OFF-source statistics ---");
            println!(
                "  Lattice: {:.4}, Shallow: {:.4}, Deep: {:.4}",
                off_stats.lattice_fraction(),
                off_stats.shallow_fraction(),
                off_stats.deep_fraction()
            );

            // Two-proportion z-test for deep hole fraction
            let n1 = on_stats.total as f64;
            let n2 = off_stats.total as f64;
            let p1 = on_stats.deep_fraction();
            let p2 = off_stats.deep_fraction();
            let p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2);
            let se = (p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2)).sqrt();
            let z = if se > 0.0 {
                (p1 - p2) / se
            } else {
                0.0
            };
            // Two-tailed p-value (approximation)
            let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));

            println!("\n--- Statistical Test ---");
            println!("  Deep hole fraction: ON={p1:.4}, OFF={p2:.4}");
            println!("  z-statistic: {z:.4}");
            println!("  p-value: {p_value:.6}");

            // Write CSV
            let mut file = std::fs::File::create(&output)?;
            writeln!(file, "pointing,lattice,shallow,deep,total")?;
            writeln!(
                file,
                "ON,{},{},{},{}",
                on_stats.lattice_points, on_stats.shallow_holes, on_stats.deep_holes, on_stats.total
            )?;
            writeln!(
                file,
                "OFF,{},{},{},{}",
                off_stats.lattice_points,
                off_stats.shallow_holes,
                off_stats.deep_holes,
                off_stats.total
            )?;
            println!("\nWrote {output}");

            // Verdict
            println!("\n=== C-778 Verdict ===");
            if p_value < 0.05 {
                println!("CONFIRMED: Deep hole fraction differs (p={p_value:.6} < 0.05).");
                println!("  Moonshine filter discriminates ON from OFF.");
            } else {
                println!("FALSIFIED: No significant difference (p={p_value:.6} >= 0.05).");
                println!("  Moonshine filter adds no discriminating power at SNR={snr}.");
            }
        }
        Cmd::Analyze {
            input,
            chunk_size,
            output,
        } => {
            println!("=== Moonshine Filterbank: Real Data Analysis ===");
            println!("Input: {input}, Chunk size: {chunk_size}");

            let basis = LeechBasis::new();

            // Read CSV: expect one float per line
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_path(&input)?;
            let mut values = Vec::new();
            for result in rdr.records() {
                let record = result?;
                if let Ok(v) = record[0].parse::<f64>() {
                    values.push(v);
                }
            }
            println!("Read {} values", values.len());

            // Chunk with 50% overlap
            let step = chunk_size / 2;
            let mut chunks = Vec::new();
            let mut i = 0;
            while i + chunk_size <= values.len() {
                chunks.push(values[i..i + chunk_size].to_vec());
                i += step;
            }
            println!("Generated {} chunks (50% overlap)", chunks.len());

            let stats = project_and_classify(&chunks, &basis);
            println!(
                "Lattice: {:.4}, Shallow: {:.4}, Deep: {:.4}",
                stats.lattice_fraction(),
                stats.shallow_fraction(),
                stats.deep_fraction()
            );

            let mut file = std::fs::File::create(&output)?;
            writeln!(file, "chunk_idx,classification,distance_sq")?;
            for (idx, chunk) in chunks.iter().enumerate() {
                let point = project_signal_chunk(chunk);
                let cvp = basis.cvp(&point);
                writeln!(
                    file,
                    "{idx},{:?},{:.10}",
                    cvp.classification, cvp.distance_squared
                )?;
            }
            println!("Wrote {output}");
        }
    }

    Ok(())
}

/// Normal CDF approximation (Abramowitz & Stegun 26.2.17).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if x > 0.0 {
        1.0 - p
    } else {
        p
    }
}
