//! Bispectral Fano selection rule test on stacked rotation curve residuals.
//!
//! Tests whether three-wave phase coupling (bicoherence) at CD-ZD wavenumber
//! triads obeys the Fano plane selection rules of the octonion multiplication
//! table. The Fano plane PG(2,2) has 7 lines:
//!   {1,2,4}, {2,3,5}, {3,4,6}, {4,5,7}, {1,5,6}, {2,6,7}, {1,3,7}
//!
//! Each line represents a valid multiplication triple e_a * e_b = +/- e_c.
//! If ZD forcing creates harmonic modes, three-wave coupling should follow
//! this structure: Fano-line triples may have different bicoherence from
//! non-Fano triples.
//!
//! Two analysis modes:
//!   1. Stacked profile: bispectrum of the single stacked delta(x).
//!   2. Ensemble: per-galaxy bispectral coefficients averaged over N galaxies
//!      (loaded from pre-computed per-galaxy DFT CSV).
//!
//! Reference: Hypothesis H1 from novel hypothesis plan.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "bispectral-selection-rule")]
#[command(about = "Fano plane bispectral selection rule test on rotation curve residuals")]
struct Cli {
    /// Input stacked CSV (columns: x, delta_stack, delta_stack_err, n_contributing).
    #[arg(long)]
    stacked_csv: PathBuf,

    /// Minimum galaxies per x-bin to include.
    #[arg(long, default_value_t = 10)]
    min_per_bin: usize,

    /// Optional per-galaxy DFT CSV for ensemble bicoherence.
    /// Columns: plateifu, cd_re, cd_im, g2_re, g2_im, j3o_re, j3o_im, sl2_re, sl2_im.
    #[arg(long)]
    galaxies_csv: Option<PathBuf>,
}

/// The 7 lines of the Fano plane PG(2,2), using 1-indexed octonion units.
/// Each triple {a,b,c} satisfies e_a * e_b = +/- e_c in the octonion algebra.
const FANO_LINES: [[usize; 3]; 7] = [
    [1, 2, 4],
    [2, 3, 5],
    [3, 4, 6],
    [4, 5, 7],
    [1, 5, 6],
    [2, 6, 7],
    [1, 3, 7],
];

fn load_stacked_csv(path: &std::path::Path) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();

    let x_idx = headers.iter().position(|h| h == "x")
        .ok_or_else(|| anyhow::anyhow!("No 'x' column"))?;
    let delta_idx = headers.iter().position(|h| h == "delta_stack")
        .ok_or_else(|| anyhow::anyhow!("No 'delta_stack' column"))?;
    let n_idx = headers.iter().position(|h| h == "n_contributing")
        .ok_or_else(|| anyhow::anyhow!("No 'n_contributing' column"))?;

    let mut x_grid = Vec::new();
    let mut delta = Vec::new();
    let mut n_contrib = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        x_grid.push(rec.get(x_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
        delta.push(rec.get(delta_idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0));
        n_contrib.push(rec.get(n_idx).unwrap_or("0").parse::<usize>().unwrap_or(0));
    }

    Ok((x_grid, delta, n_contrib))
}

/// Compute complex DFT coefficients at given wavenumbers.
/// Returns Vec<(re, im)> for each wavenumber.
fn complex_dft(
    x_grid: &[f64],
    delta: &[f64],
    n_contrib: &[usize],
    min_per_bin: usize,
    wavenumbers: &[f64],
) -> Vec<(f64, f64)> {
    wavenumbers
        .iter()
        .map(|&k| {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            let mut count = 0_usize;

            for (j, &x) in x_grid.iter().enumerate() {
                if n_contrib[j] < min_per_bin {
                    continue;
                }
                re += delta[j] * (k * x).cos();
                im += delta[j] * (k * x).sin();
                count += 1;
            }

            if count > 0 {
                let norm = count as f64;
                (re / norm, im / norm)
            } else {
                (0.0, 0.0)
            }
        })
        .collect()
}

/// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i.
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Complex conjugate.
fn conj(z: (f64, f64)) -> (f64, f64) {
    (z.0, -z.1)
}

/// Complex magnitude squared.
fn mag2(z: (f64, f64)) -> f64 {
    z.0 * z.0 + z.1 * z.1
}

/// Check if a triple of mode indices (1-indexed) forms a Fano line.
fn is_fano_triple(a: usize, b: usize, c: usize) -> bool {
    let mut triple = [a, b, c];
    triple.sort();
    FANO_LINES.iter().any(|line| {
        let mut l = *line;
        l.sort();
        l == triple
    })
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== H1: Bispectral Fano Selection Rule Test ===");
    eprintln!();

    // Load stacked profile.
    let (x_grid, delta, n_contrib) = load_stacked_csv(&cli.stacked_csv)?;
    let valid_bins = n_contrib.iter().filter(|&&n| n >= cli.min_per_bin).count();
    eprintln!("Loaded {} bins, {} valid (n >= {})", x_grid.len(), valid_bins, cli.min_per_bin);

    // CD-ZD wavenumbers: k_n = 2*pi*n/7 for n=1..7.
    let n_modes = 7;
    let cd_wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * std::f64::consts::PI * n as f64 / n_modes as f64)
        .collect();

    // All unique sum wavenumbers k_i + k_j for i <= j.
    let mut all_wavenumbers = cd_wavenumbers.clone();
    let mut sum_map: Vec<(usize, usize, usize)> = Vec::new(); // (i, j, index in all_wavenumbers)

    for i in 0..n_modes {
        for j in i..n_modes {
            let k_sum = cd_wavenumbers[i] + cd_wavenumbers[j];
            // Check if this sum is already in our wavenumber set.
            let idx = all_wavenumbers
                .iter()
                .position(|&k| (k - k_sum).abs() < 1e-10);
            let sum_idx = match idx {
                Some(existing) => existing,
                None => {
                    all_wavenumbers.push(k_sum);
                    all_wavenumbers.len() - 1
                }
            };
            sum_map.push((i, j, sum_idx));
        }
    }

    eprintln!("Evaluating DFT at {} wavenumbers ({} CD + {} sums)",
        all_wavenumbers.len(), n_modes, all_wavenumbers.len() - n_modes);

    // Compute complex DFT at all needed wavenumbers.
    let coeffs = complex_dft(&x_grid, &delta, &n_contrib, cli.min_per_bin, &all_wavenumbers);

    // Compute bispectrum for each pair.
    eprintln!();
    eprintln!("--- Bispectral analysis (stacked profile) ---");
    println!("mode_i,mode_j,mode_sum,k_i,k_j,k_sum,bispec_re,bispec_im,bispec_mag,bicoherence,is_fano");

    let mut fano_bicoh: Vec<f64> = Vec::new();
    let mut nonfano_bicoh: Vec<f64> = Vec::new();

    for &(i, j, sum_idx) in &sum_map {
        let f_i = coeffs[i];
        let f_j = coeffs[j];
        let f_sum = coeffs[sum_idx];

        // B(k_i, k_j) = F(k_i) * F(k_j) * conj(F(k_i + k_j))
        let bispec = cmul(cmul(f_i, f_j), conj(f_sum));
        let bispec_mag = mag2(bispec).sqrt();

        // Bicoherence: b^2 = |B|^2 / (|F_i * F_j|^2 * |F_sum|^2)
        let denom_sq = mag2(f_i) * mag2(f_j) * mag2(f_sum);
        let bicoherence = if denom_sq > 1e-30 {
            mag2(bispec) / denom_sq
        } else {
            0.0
        };

        // Check Fano membership: modes are 1-indexed.
        // The bispectral triple is (mode i+1, mode j+1, mode corresponding to k_i+k_j).
        // The sum mode index: if k_i + k_j matches a CD mode, it's sum_idx+1.
        let sum_mode = if sum_idx < n_modes { sum_idx + 1 } else { 0 };
        let is_fano = if sum_mode > 0 {
            is_fano_triple(i + 1, j + 1, sum_mode)
        } else {
            false
        };

        println!(
            "{},{},{},{:.6},{:.6},{:.6},{:.10e},{:.10e},{:.10e},{:.10e},{}",
            i + 1,
            j + 1,
            sum_mode,
            cd_wavenumbers[i],
            cd_wavenumbers[j],
            all_wavenumbers[sum_idx],
            bispec.0,
            bispec.1,
            bispec_mag,
            bicoherence,
            if is_fano { 1 } else { 0 }
        );

        if is_fano {
            fano_bicoh.push(bicoherence);
        } else {
            nonfano_bicoh.push(bicoherence);
        }
    }

    eprintln!();
    eprintln!("--- Fano vs non-Fano bicoherence ---");

    let fano_mean = if fano_bicoh.is_empty() {
        0.0
    } else {
        fano_bicoh.iter().sum::<f64>() / fano_bicoh.len() as f64
    };
    let nonfano_mean = if nonfano_bicoh.is_empty() {
        0.0
    } else {
        nonfano_bicoh.iter().sum::<f64>() / nonfano_bicoh.len() as f64
    };
    let ratio = if nonfano_mean > 1e-30 {
        fano_mean / nonfano_mean
    } else {
        f64::NAN
    };

    eprintln!("  Fano triples ({}): mean bicoherence = {:.8e}", fano_bicoh.len(), fano_mean);
    eprintln!("  Non-Fano pairs ({}): mean bicoherence = {:.8e}", nonfano_bicoh.len(), nonfano_mean);
    eprintln!("  Fano/non-Fano ratio = {:.4}", ratio);

    eprintln!();
    eprintln!("=== Verdict ===");
    if ratio.is_nan() {
        eprintln!("  Cannot compute ratio (zero denominator).");
    } else if ratio > 0.7 && ratio < 1.3 {
        eprintln!("  Ratio in [0.7, 1.3]: no Fano selection rule detected.");
        eprintln!("  FALSIFICATION: bicoherence ratio consistent with null -> REJECT.");
    } else if ratio < 0.5 {
        eprintln!("  Fano suppression detected (ratio < 0.5)!");
        eprintln!("  ZD algebraic structure may impose bispectral selection rules.");
    } else {
        eprintln!("  Ratio outside [0.7, 1.3] but above 0.5: weak/ambiguous signal.");
    }

    // Per-galaxy ensemble bicoherence (if galaxies CSV provided).
    if let Some(galaxies_path) = &cli.galaxies_csv {
        eprintln!();
        eprintln!("--- Ensemble bicoherence from per-galaxy DFT ---");
        let mut rdr = csv::Reader::from_path(galaxies_path)?;
        let mut cd_coeffs: Vec<(f64, f64)> = Vec::new();

        for result in rdr.records() {
            let rec = result?;
            let re: f64 = rec.get(1).unwrap_or("0").parse().unwrap_or(0.0);
            let im: f64 = rec.get(2).unwrap_or("0").parse().unwrap_or(0.0);
            cd_coeffs.push((re, im));
        }

        let n_gal = cd_coeffs.len();
        eprintln!("  Loaded {} galaxy DFT coefficients", n_gal);

        // Per-galaxy aggregate coefficient: this is a single (re,im) per galaxy.
        // Compute per-galaxy bispectral statistic: B = F * F * conj(F) = |F|^2 * conj(F).
        let mut bispec_sum = (0.0_f64, 0.0_f64);
        let mut denom_sum = 0.0_f64;

        for &f in &cd_coeffs {
            let b = cmul(cmul(f, f), conj(f)); // F^2 * F* = |F|^2 * F
            bispec_sum.0 += b.0;
            bispec_sum.1 += b.1;
            let power = mag2(f);
            denom_sum += power * power.sqrt(); // |F|^2 * |F| = |F|^3
        }

        bispec_sum.0 /= n_gal as f64;
        bispec_sum.1 /= n_gal as f64;
        denom_sum /= n_gal as f64;

        let ensemble_bicoh = if denom_sum > 1e-30 {
            mag2(bispec_sum).sqrt() / denom_sum
        } else {
            0.0
        };

        eprintln!("  Ensemble bicoherence (aggregate CD mode) = {:.8e}", ensemble_bicoh);
        eprintln!("  (For phase-incoherent noise: expected ~ 1/sqrt(N) = {:.6e})",
            1.0 / (n_gal as f64).sqrt());
    }

    Ok(())
}
