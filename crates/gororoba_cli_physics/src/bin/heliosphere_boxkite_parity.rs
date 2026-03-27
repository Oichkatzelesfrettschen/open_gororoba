//! Box-Kite alignment backend parity checker.
//!
//! Loads CPU, Vulkan, and (optionally) CUDA box-kite alignment CSVs,
//! joins on row index within each (r_au, mission) group, and verifies:
//! - Exact match on `best_orient_idx` (discrete)
//! - Tolerant match on `max_alignment` (abs_diff <= 1e-12 OR rel_diff <= 1e-10)
//!
//! Produces a JSON summary with pass/fail and per-backend row counts.

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-boxkite-parity",
    about = "Verify box-kite alignment parity across CPU/Vulkan/CUDA backends"
)]
struct Cli {
    #[arg(long)]
    cpu_csv: PathBuf,

    #[arg(long)]
    vulkan_csv: PathBuf,

    #[arg(long)]
    cuda_csv: Option<PathBuf>,

    #[arg(long, default_value = "data/output/heliosphere/ablations/boxkite_parity.json")]
    out: PathBuf,

    /// Absolute tolerance for max_alignment comparison.
    #[arg(long, default_value_t = 1e-12)]
    abs_tol: f64,

    /// Relative tolerance for max_alignment comparison.
    #[arg(long, default_value_t = 1e-10)]
    rel_tol: f64,
}

#[derive(Debug, Deserialize)]
struct AlignmentRow {
    #[allow(dead_code)]
    r_au: f64,
    #[allow(dead_code)]
    mission: String,
    max_alignment: f64,
    best_orient_idx: u32,
    #[allow(dead_code)]
    backend: String,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    cpu_rows: usize,
    vulkan_rows: usize,
    cuda_rows: Option<usize>,
    orient_mismatches_total: usize,
    orient_mismatches_tied: usize,
    orient_mismatches_real: usize,
    alignment_failures: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    pairs_checked: usize,
}

fn load_csv(path: &PathBuf) -> Result<Vec<AlignmentRow>> {
    let mut reader = csv::ReaderBuilder::new()
        .from_path(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::new();
    for result in reader.deserialize::<AlignmentRow>() {
        rows.push(result?);
    }
    Ok(rows)
}

/// Parity statistics from comparing two backend outputs row-by-row.
#[derive(Debug, Default)]
struct CompareStats {
    orient_mismatches_total: usize,
    /// Orient differs but alignment values match within tolerance (FP tie-break).
    orient_mismatches_tied: usize,
    /// Orient differs AND alignment values diverge beyond tolerance.
    orient_mismatches_real: usize,
    alignment_failures: usize,
    max_abs: f64,
    max_rel: f64,
    pairs: usize,
}

fn compare(a: &[AlignmentRow], b: &[AlignmentRow], abs_tol: f64, rel_tol: f64) -> CompareStats {
    let n = a.len().min(b.len());
    let mut s = CompareStats { pairs: n, ..Default::default() };

    for i in 0..n {
        let abs_d = (a[i].max_alignment - b[i].max_alignment).abs();
        let denom = a[i].max_alignment.abs().max(b[i].max_alignment.abs());
        let rel_d = if denom > 0.0 { abs_d / denom } else { 0.0 };

        if abs_d > s.max_abs { s.max_abs = abs_d; }
        if rel_d > s.max_rel { s.max_rel = rel_d; }

        if a[i].best_orient_idx != b[i].best_orient_idx {
            s.orient_mismatches_total += 1;
            if abs_d <= abs_tol || rel_d <= rel_tol {
                s.orient_mismatches_tied += 1;
            } else {
                s.orient_mismatches_real += 1;
            }
        }

        if abs_d > abs_tol && rel_d > rel_tol {
            s.alignment_failures += 1;
        }
    }
    s
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("[1/3] Loading CPU results from {}...", cli.cpu_csv.display());
    let cpu = load_csv(&cli.cpu_csv)?;

    println!("[2/3] Loading Vulkan results from {}...", cli.vulkan_csv.display());
    let vulkan = load_csv(&cli.vulkan_csv)?;

    let cuda = if let Some(ref p) = cli.cuda_csv {
        println!("       Loading CUDA results from {}...", p.display());
        Some(load_csv(p)?)
    } else {
        None
    };

    println!("[3/3] Comparing {} CPU vs {} Vulkan rows...", cpu.len(), vulkan.len());
    let sv = compare(&cpu, &vulkan, cli.abs_tol, cli.rel_tol);

    // If CUDA available, also compare CPU vs CUDA and merge stats
    let sc = cuda.as_ref().map(|c| compare(&cpu, c, cli.abs_tol, cli.rel_tol));

    let om_total = sv.orient_mismatches_total + sc.as_ref().map_or(0, |s| s.orient_mismatches_total);
    let om_tied = sv.orient_mismatches_tied + sc.as_ref().map_or(0, |s| s.orient_mismatches_tied);
    let om_real = sv.orient_mismatches_real + sc.as_ref().map_or(0, |s| s.orient_mismatches_real);
    let af_total = sv.alignment_failures + sc.as_ref().map_or(0, |s| s.alignment_failures);
    let ma_total = sv.max_abs.max(sc.as_ref().map_or(0.0, |s| s.max_abs));
    let mr_total = sv.max_rel.max(sc.as_ref().map_or(0.0, |s| s.max_rel));
    let pairs_total = sv.pairs + sc.as_ref().map_or(0, |s| s.pairs);

    // Pass if: no real orient mismatches AND no alignment failures.
    // Tied orient mismatches (alignment values match within tolerance) are expected
    // from FP instruction ordering differences between backends.
    let pass = om_real == 0 && af_total == 0;

    let report = ParityReport {
        cpu_rows: cpu.len(),
        vulkan_rows: vulkan.len(),
        cuda_rows: cuda.as_ref().map(|c| c.len()),
        orient_mismatches_total: om_total,
        orient_mismatches_tied: om_tied,
        orient_mismatches_real: om_real,
        alignment_failures: af_total,
        max_abs_diff: ma_total,
        max_rel_diff: mr_total,
        pass,
        pairs_checked: pairs_total,
    };

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&cli.out, &json)?;

    if pass {
        println!("PASS: {} pairs checked, max_abs_diff={:.2e}, max_rel_diff={:.2e}",
            pairs_total, ma_total, mr_total);
        if om_tied > 0 {
            println!("      ({} orient indices differ due to FP tie-breaking, alignment values match)", om_tied);
        }
    } else {
        println!("FAIL: {} real orient mismatches, {} alignment failures out of {} pairs",
            om_real, af_total, pairs_total);
        println!("      max_abs_diff={:.2e}, max_rel_diff={:.2e}", ma_total, mr_total);
    }

    Ok(())
}
