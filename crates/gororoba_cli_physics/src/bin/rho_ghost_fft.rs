//! Ghost spectral analysis -- statistically hardened (Sprint 50).
//!
//! Hypothesis: Sorted catalog data or LBM rho_mean contains a spectral peak
//! at phi^{-1/2} ~ 0.214 (Nyquist-folded). This binary tests that hypothesis
//! with rigorous statistics: MAD noise floor, BH-FDR, Bonferroni correction,
//! permutation testing, and sorted-distribution bootstrap null.
//!
//! Subcommands:
//! - `analyze`:    Read rho_mean from HDF5 and perform FFT spectral analysis
//! - `csv`:        Read from a CSV column with full statistical pipeline
//! - `synth`:      Generate synthetic test signals to validate the FFT pipeline
//! - `multi-rate`: Subsample at varying strides and track alias migration
//! - `batch`:      Analyze multiple CSVs with joint FDR and Stouffer combination

use clap::{Parser, Subcommand};
use spectral_core::ghost_spectral::{
    ALIASED_GHOST_FREQ, FREQ_TOL, GHOST_FREQ, GhostVerdict, PHI, check_ghost,
    check_ghost_at_stride, check_ghost_rigorous, combine_p_values, compute_power_spectrum,
    find_peaks, ghost_aliases, is_ghost_freq, is_monotonically_sorted, noise_floor, peak_fwhm,
    peak_fwhm_clamped, peak_snr, permutation_test, robust_noise_floor, sorted_distribution_null,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "rho-ghost-fft",
    about = "Ghost spectral analysis with rigorous statistical testing"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read rho_mean from HDF5 artifact and perform FFT analysis.
    Analyze {
        /// Path to the warp ring HDF5 artifact.
        #[arg(long)]
        hdf5: PathBuf,

        /// Number of top spectral peaks to report.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
    },
    /// Read a time series from a CSV column and perform FFT analysis.
    Csv {
        /// Path to CSV file.
        #[arg(long)]
        path: PathBuf,

        /// Column name containing the time series.
        #[arg(long, default_value = "rho_mean")]
        column: String,

        /// Number of top spectral peaks to report.
        #[arg(long, default_value_t = 10)]
        top_k: usize,

        /// Enable full rigorous statistical pipeline.
        #[arg(long, default_value_t = true)]
        rigorous: bool,

        /// Number of permutation test iterations (0 to skip).
        #[arg(long, default_value_t = 1000)]
        permutations: usize,

        /// Number of sorted-distribution bootstrap iterations (0 to skip).
        #[arg(long, default_value_t = 10000)]
        bootstrap: usize,

        /// Number of datasets for cross-catalog Bonferroni correction.
        #[arg(long, default_value_t = 1)]
        bonferroni_datasets: usize,

        /// Random seed for permutation/bootstrap tests.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Generate synthetic signals to validate the FFT pipeline.
    Synth {
        /// Number of samples.
        #[arg(long, default_value_t = 1024)]
        n: usize,

        /// Inject a signal at the ghost frequency.
        #[arg(long, default_value_t = true)]
        inject_ghost: bool,

        /// Amplitude of injected signal (relative to noise).
        #[arg(long, default_value_t = 0.1)]
        amplitude: f64,

        /// Noise standard deviation.
        #[arg(long, default_value_t = 1.0)]
        noise_std: f64,

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Multi-rate sweep: subsample at varying strides and track how the
    /// aliased ghost peak migrates.
    MultiRate {
        /// Path to CSV file containing the time series.
        #[arg(long)]
        path: PathBuf,

        /// Column name containing the time series.
        #[arg(long, default_value = "rho_mean")]
        column: String,

        /// Minimum subsampling stride (every Nth sample).
        #[arg(long, default_value_t = 1)]
        stride_min: usize,

        /// Maximum subsampling stride.
        #[arg(long, default_value_t = 15)]
        stride_max: usize,
    },
    /// Batch analysis: process multiple CSVs, apply joint FDR correction,
    /// and compute Stouffer combined p-value across datasets.
    Batch {
        /// Directory containing CSV files to analyze.
        #[arg(long)]
        datasets: PathBuf,

        /// Column name in each CSV.
        #[arg(long, default_value = "value")]
        column: String,

        /// Number of permutation test iterations per dataset.
        #[arg(long, default_value_t = 1000)]
        permutations: usize,

        /// Number of bootstrap iterations per dataset.
        #[arg(long, default_value_t = 10000)]
        bootstrap: usize,

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

/// Print legacy spectral analysis (kept for backward compatibility).
fn print_analysis_legacy(signal: &[f64], top_k: usize, source: &str) {
    println!("=== Ghost Spectral Analysis (legacy mode) ===");
    println!();
    println!("Source: {}", source);
    println!("Samples: {}", signal.len());
    println!("Target ghost frequency: {:.6} (phi^{{-1/2}})", GHOST_FREQ);
    println!(
        "Aliased ghost frequency: {:.6} (1 - phi^{{-1/2}}, Nyquist fold)",
        ALIASED_GHOST_FREQ
    );
    println!("phi = {:.15}", PHI);
    println!();

    let n = signal.len() as f64;
    let mean = signal.iter().sum::<f64>() / n;
    let var = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = var.sqrt();
    let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Signal statistics:");
    println!("  mean = {:.6e}", mean);
    println!("  std  = {:.6e}", std_dev);
    println!("  min  = {:.6e}", min);
    println!("  max  = {:.6e}", max);
    println!("  CV   = {:.6}", std_dev / mean.abs().max(1e-30));
    println!();

    let (freqs, power) = compute_power_spectrum(signal);
    let peaks = find_peaks(&freqs, &power, top_k);
    let nf = noise_floor(&power);

    println!("Power spectrum:");
    println!(
        "  Total power (excl DC): {:.6e}",
        power[1..].iter().sum::<f64>()
    );
    println!("  Noise floor (mean PSD): {:.6e}", nf);
    println!(
        "  Frequency resolution: {:.6} cycles/sample",
        1.0 / signal.len() as f64
    );
    println!();

    println!("Top {} spectral peaks:", peaks.len());
    println!(
        "  {:>4}  {:>12}  {:>12}  {:>8}",
        "Rank", "Frequency", "Power", "SNR"
    );
    for peak in &peaks {
        let snr = peak_snr(peak, nf);
        let ghost_marker = if is_ghost_freq(peak.freq) {
            if (peak.freq - ALIASED_GHOST_FREQ).abs() < FREQ_TOL {
                " <-- GHOST (aliased)"
            } else {
                " <-- GHOST?"
            }
        } else {
            ""
        };
        println!(
            "  {:>4}  {:>12.6}  {:>12.4e}  {:>8.2}{}",
            peak.rank, peak.freq, peak.power, snr, ghost_marker
        );
    }
    println!();

    let alias_target = ghost_aliases(1)[0];
    if let Some(fwhm) = peak_fwhm(&freqs, &power, alias_target) {
        let freq_res = 1.0 / signal.len() as f64;
        let fwhm_bins = fwhm / freq_res;
        println!("Spectral width at ghost alias ({:.4}):", alias_target);
        println!("  FWHM = {:.6} cycles/sample ({:.1} bins)", fwhm, fwhm_bins);
        if fwhm_bins < 3.0 {
            println!("  -> SHARP: algebraic origin (resolution-limited)");
        } else if fwhm_bins < 10.0 {
            println!("  -> MODERATE: possible chirp or weak coupling");
        } else {
            println!("  -> BROAD: physical/chirping origin (spectral blur)");
        }
        println!();
    }

    match check_ghost(&peaks) {
        Some(ghost_peak) => {
            let snr = peak_snr(ghost_peak, nf);
            println!(
                "GHOST DETECTED at freq={:.6} (rank {}, SNR={:.2})",
                ghost_peak.freq, ghost_peak.rank, snr
            );
            if ghost_peak.rank <= 3 && snr > 5.0 {
                println!(
                    "VERDICT: The ghost is REAL. Dominant spectral peak near phi^{{-1/2}}={:.6}.",
                    GHOST_FREQ
                );
            } else {
                println!(
                    "VERDICT: Weak ghost signal (rank {}, SNR={:.2}). Marginal evidence.",
                    ghost_peak.rank, snr
                );
            }
        }
        None => {
            println!(
                "NO GHOST: No spectral peak found near phi^{{-1/2}}={:.6}.",
                GHOST_FREQ
            );
            if peaks.is_empty() {
                println!("VERDICT: White noise or Brownian drift. The ghost is NOT real.");
            } else {
                println!(
                    "VERDICT: Dominant frequency at {:.6} (not phi^{{-1/2}}). The ghost is NOT real.",
                    peaks[0].freq
                );
            }
        }
    }
}

/// Print rigorous analysis with full statistical pipeline (Sprint 50).
fn print_analysis_rigorous(
    signal: &[f64],
    top_k: usize,
    source: &str,
    n_permutations: usize,
    n_bootstrap: usize,
    bonferroni_datasets: usize,
    seed: u64,
) {
    println!("=== Ghost Spectral Analysis (RIGOROUS, Sprint 50) ===");
    println!();
    println!("Source: {}", source);
    println!("Samples: {}", signal.len());
    println!("Target ghost frequency: {:.6} (phi^{{-1/2}})", GHOST_FREQ);
    println!(
        "Aliased ghost frequency: {:.6} (1 - phi^{{-1/2}})",
        ALIASED_GHOST_FREQ
    );
    println!();

    // Basic statistics
    let n = signal.len() as f64;
    let mean = signal.iter().sum::<f64>() / n;
    let var = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = var.sqrt();
    println!("Signal statistics:");
    println!(
        "  mean={:.6e}  std={:.6e}  N={}",
        mean,
        std_dev,
        signal.len()
    );
    println!();

    // Rigorous hypothesis test
    let result = check_ghost_rigorous(signal, ALIASED_GHOST_FREQ, FREQ_TOL, 0.05);

    // Show both legacy and corrected noise floor for comparison
    let (freqs, power) = compute_power_spectrum(signal);
    let legacy_nf = noise_floor(&power);
    let peaks = find_peaks(&freqs, &power, top_k);
    let target_idx = freqs
        .iter()
        .enumerate()
        .skip(1)
        .min_by(|(_, a), (_, b)| {
            ((**a - ALIASED_GHOST_FREQ).abs())
                .partial_cmp(&((**b - ALIASED_GHOST_FREQ).abs()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(1);
    let robust_nf = robust_noise_floor(&power, &[target_idx], 3);

    println!("Power spectrum:");
    println!(
        "  Frequency resolution: {:.6} cycles/sample",
        1.0 / signal.len() as f64
    );
    println!("  Noise floor (legacy mean):  {:.6e}", legacy_nf);
    println!("  Noise floor (robust MAD):   {:.6e}", robust_nf);
    println!(
        "  Correction factor:          {:.2}x",
        legacy_nf / robust_nf.max(1e-300)
    );
    println!();

    println!("Top {} spectral peaks:", peaks.len().min(top_k));
    println!(
        "  {:>4}  {:>12}  {:>12}  {:>8}  {:>10}",
        "Rank", "Frequency", "Power", "SNR_rob", "p_raw"
    );
    for peak in peaks.iter().take(top_k) {
        let snr = peak_snr(peak, robust_nf);
        let p_raw =
            spectral_core::ghost_spectral::false_alarm_probability_single(peak.power, robust_nf);
        let ghost_marker = if is_ghost_freq(peak.freq) {
            " <-- TARGET"
        } else {
            ""
        };
        println!(
            "  {:>4}  {:>12.6}  {:>12.4e}  {:>8.2}  {:>10.4e}{}",
            peak.rank, peak.freq, peak.power, snr, p_raw, ghost_marker
        );
    }
    println!();

    // FWHM with clamping
    let alias_target = ghost_aliases(1)[0];
    if let Some((fwhm_cps, fwhm_bins, clamped)) = peak_fwhm_clamped(&freqs, &power, alias_target) {
        println!("FWHM at ghost alias ({:.4}):", alias_target);
        println!("  {:.6} cycles/sample ({:.1} bins)", fwhm_cps, fwhm_bins);
        if clamped {
            println!(
                "  WARNING: FWHM clamped to resolution limit (1 bin = 1/N). \
                 Sub-bin width is an interpolation artifact, not physics."
            );
        }
        println!();
    }

    // Rigorous verdict
    println!("--- Rigorous Hypothesis Test ---");
    println!("  Raw SNR (legacy):     {:.2}", result.raw_snr);
    println!("  Corrected SNR (MAD):  {:.2}", result.corrected_snr);
    println!("  Frequency delta:      {:.6}", result.freq_delta);
    println!("  Trials factor:        {} bins searched", result.n_trials);
    println!("  p_raw (single bin):   {:.4e}", result.p_raw);
    println!("  p_bonferroni:         {:.4e}", result.p_bonferroni);
    println!("  p_fdr (BH):           {:.4e}", result.p_fdr);
    println!("  VERDICT:              {}", result.verdict);
    if !result.methodology_note.is_empty() {
        println!("  Note: {}", result.methodology_note);
    }

    // Cross-catalog Bonferroni
    if bonferroni_datasets > 1 {
        let p_cross = 1.0 - (1.0 - result.p_bonferroni).powi(bonferroni_datasets as i32);
        println!(
            "  p_cross_catalog ({} datasets): {:.4e}",
            bonferroni_datasets, p_cross
        );
    }
    println!();

    // Permutation test
    if n_permutations > 0 {
        println!("--- Permutation Test ({} iterations) ---", n_permutations);
        let perm = permutation_test(signal, ALIASED_GHOST_FREQ, n_permutations, seed);
        println!("  Observed power:    {:.6e}", perm.observed_power);
        println!(
            "  Null mean +/- std: {:.6e} +/- {:.6e}",
            perm.null_mean, perm.null_std
        );
        println!("  Empirical p-value: {:.4e}", perm.p_empirical);
        let perm_verdict = if perm.p_empirical < 0.05 {
            "SIGNIFICANT (p < 0.05)"
        } else {
            "NOT SIGNIFICANT"
        };
        println!("  Verdict:           {}", perm_verdict);
        println!();
    }

    // Bootstrap null (only for sorted data)
    if n_bootstrap > 0 {
        println!(
            "--- Sorted-Distribution Bootstrap Null ({} iterations) ---",
            n_bootstrap
        );
        let boot = sorted_distribution_null(signal, ALIASED_GHOST_FREQ, n_bootstrap, seed);
        println!("  Observed power:    {:.6e}", boot.observed_power);
        println!(
            "  Null mean +/- std: {:.6e} +/- {:.6e}",
            boot.null_mean, boot.null_std
        );
        println!("  Empirical p-value: {:.4e}", boot.p_empirical);
        let boot_verdict = if boot.p_empirical < 0.05 {
            "Peak EXCEEDS bootstrap null (sorted-distribution does NOT explain it)"
        } else {
            "Peak CONSISTENT with bootstrap null (sorting artifact)"
        };
        println!("  Verdict:           {}", boot_verdict);
        println!();
    }

    // Final combined verdict
    println!("=== FINAL VERDICT ===");
    match result.verdict {
        GhostVerdict::Detected => {
            println!(
                "DETECTED: Ghost peak survives all corrections (p_fdr < 0.05 AND p_bonf < 0.05)."
            );
            println!("Proceed to Tier 2 tests (Lomb-Scargle, multitaper, quantile periodogram).");
        }
        GhostVerdict::Marginal => {
            println!("MARGINAL: Ghost peak survives some but not all corrections.");
            println!("Interpretation uncertain -- may be noise fluctuation or weak signal.");
        }
        GhostVerdict::Null => {
            println!("NULL: Ghost peak does NOT survive corrected statistical testing.");
            println!("Consistent with noise. No evidence for phi^{{-1/2}} spectral line.");
        }
    }
}

/// Read rho_mean from a CSV file column.
fn read_csv_column(path: &std::path::Path, column: &str) -> Vec<f64> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: Failed to read {}: {}", path.display(), e);
            std::process::exit(1);
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        eprintln!("ERROR: Empty CSV file");
        std::process::exit(1);
    }

    // Find header row (skip comment lines)
    let mut header_idx = 0;
    for (i, line) in lines.iter().enumerate() {
        if !line.starts_with('#') && !line.is_empty() {
            header_idx = i;
            break;
        }
    }

    let headers: Vec<&str> = lines[header_idx].split(',').map(|s| s.trim()).collect();
    let col_idx = match headers.iter().position(|&h| h == column) {
        Some(idx) => idx,
        None => {
            eprintln!(
                "ERROR: Column '{}' not found. Available: {:?}",
                column, headers
            );
            std::process::exit(1);
        }
    };

    let mut values = Vec::new();
    for line in &lines[header_idx + 1..] {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if col_idx < fields.len()
            && let Ok(v) = fields[col_idx].trim().parse::<f64>()
            && v.is_finite()
        {
            values.push(v);
        }
    }

    values
}

#[cfg(feature = "hdf5-export")]
fn run_analyze(hdf5_path: &std::path::Path, top_k: usize) {
    use data_artifacts_core::hdf5_export::read_rho_mean_trace;

    let signal = match read_rho_mean_trace(hdf5_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "ERROR: Failed to read rho_mean from {}: {}",
                hdf5_path.display(),
                e
            );
            std::process::exit(1);
        }
    };

    if signal.is_empty() {
        eprintln!("ERROR: rho_mean trace is empty");
        std::process::exit(1);
    }

    print_analysis_legacy(&signal, top_k, &format!("HDF5: {}", hdf5_path.display()));
}

#[cfg(not(feature = "hdf5-export"))]
fn run_analyze(_hdf5_path: &std::path::Path, _top_k: usize) {
    eprintln!("ERROR: HDF5 support not enabled.");
    eprintln!("Rebuild with: cargo build --bin rho-ghost-fft --features hdf5-export");
    std::process::exit(1);
}

/// Configuration for `run_csv`: signal location (path + column), peak
/// extraction (top_k), and the four permutation/bootstrap statistics
/// parameters that define the rigorous-mode test.
struct CsvRunConfig<'a> {
    path: &'a std::path::Path,
    column: &'a str,
    top_k: usize,
    rigorous: bool,
    n_permutations: usize,
    n_bootstrap: usize,
    bonferroni_datasets: usize,
    seed: u64,
}

fn run_csv(cfg: CsvRunConfig<'_>) {
    let path = cfg.path;
    let column = cfg.column;
    let top_k = cfg.top_k;
    let rigorous = cfg.rigorous;
    let n_permutations = cfg.n_permutations;
    let n_bootstrap = cfg.n_bootstrap;
    let bonferroni_datasets = cfg.bonferroni_datasets;
    let seed = cfg.seed;
    let signal = read_csv_column(path, column);

    if signal.is_empty() {
        eprintln!("ERROR: No valid data in column '{}'", column);
        std::process::exit(1);
    }

    let source = format!("CSV: {} [{}]", path.display(), column);

    if rigorous {
        print_analysis_rigorous(
            &signal,
            top_k,
            &source,
            n_permutations,
            n_bootstrap,
            bonferroni_datasets,
            seed,
        );
    } else {
        print_analysis_legacy(&signal, top_k, &source);
    }
}

fn run_synth(n: usize, inject_ghost: bool, amplitude: f64, noise_std: f64, seed: u64) {
    use rand::{SeedableRng, rngs::StdRng};
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise_std).expect("Invalid noise std");

    let mut signal: Vec<f64> = (0..n).map(|_| 1.0 + normal.sample(&mut rng)).collect();

    if inject_ghost {
        for (i, val) in signal.iter_mut().enumerate() {
            *val += amplitude * (2.0 * std::f64::consts::PI * GHOST_FREQ * i as f64).sin();
        }
        println!(
            "Injected ghost signal: A={}, f={:.6} (phi^{{-1/2}}), N={}",
            amplitude, GHOST_FREQ, n
        );
    } else {
        println!("Pure noise: std={}, N={}", noise_std, n);
    }
    println!();

    print_analysis_rigorous(
        &signal,
        10,
        &format!(
            "Synthetic (N={}, ghost={}, A={}, seed={})",
            n, inject_ghost, amplitude, seed
        ),
        1000,
        0, // no bootstrap for non-sorted synth data
        1,
        seed,
    );
}

fn run_multi_rate(path: &std::path::Path, column: &str, stride_min: usize, stride_max: usize) {
    let signal = read_csv_column(path, column);
    if signal.is_empty() {
        eprintln!("ERROR: No valid data in column '{}'", column);
        std::process::exit(1);
    }

    println!("=== Multi-Rate Ghost Aliasing Sweep ===");
    println!();
    println!("Source: {} [{}]", path.display(), column);
    println!("Original samples: {}", signal.len());
    println!(
        "Ghost frequency: {:.6} (phi^{{-1/2}}), alias: {:.6}",
        GHOST_FREQ, ALIASED_GHOST_FREQ
    );
    println!(
        "Stride range: {} .. {} (subsampling factor)",
        stride_min, stride_max
    );
    println!();
    println!(
        "{:>6}  {:>8}  {:>12}  {:>12}  {:>12}  {:>8}  {:>6}  {:>6}",
        "Stride", "N_sub", "Expect_alias", "Peak1_freq", "Peak1_power", "Ghost?", "Rank", "FWHM"
    );

    for stride in stride_min..=stride_max {
        if stride == 0 {
            continue;
        }
        let subsampled: Vec<f64> = signal.iter().step_by(stride).copied().collect();
        if subsampled.len() < 16 {
            println!(
                "{:>6}  {:>8}  (too few samples after subsampling)",
                stride,
                subsampled.len()
            );
            continue;
        }

        let (freqs, power) = compute_power_spectrum(&subsampled);
        let peaks = find_peaks(&freqs, &power, 10);

        if peaks.is_empty() {
            let expected_alias = ghost_aliases(stride)[0];
            println!(
                "{:>6}  {:>8}  {:>12.6}  {:>12}  {:>12}  {:>8}  {:>6}  {:>6}",
                stride,
                subsampled.len(),
                expected_alias,
                "--",
                "--",
                "NO",
                "--",
                "--"
            );
            continue;
        }

        let expected_alias = ghost_aliases(stride)[0];
        let ghost = check_ghost_at_stride(&peaks, stride);
        let ghost_marker = if ghost.is_some() { "YES" } else { "NO" };
        let ghost_rank = ghost.map_or_else(|| "--".to_string(), |g| format!("{}", g.rank));

        let fwhm_info = peak_fwhm(&freqs, &power, expected_alias)
            .map(|f| {
                let bins = f * subsampled.len() as f64;
                format!("{:.1}", bins)
            })
            .unwrap_or_else(|| "--".to_string());

        println!(
            "{:>6}  {:>8}  {:>12.6}  {:>12.6}  {:>12.4e}  {:>8}  {:>6}  {:>6}",
            stride,
            subsampled.len(),
            expected_alias,
            peaks[0].freq,
            peaks[0].power,
            ghost_marker,
            ghost_rank,
            fwhm_info
        );
    }

    println!();
    println!("Interpretation:");
    println!("  If Ghost?=YES and Peak1_freq tracks Expect_alias -> possible periodic signal.");
    println!("  If Ghost?=YES appears sporadically with random Peak1_freq -> noise artifact.");
    println!("  Use 'csv --rigorous' for proper statistical testing before drawing conclusions.");
}

fn run_batch(
    datasets_dir: &std::path::Path,
    column: &str,
    n_permutations: usize,
    n_bootstrap: usize,
    seed: u64,
) {
    // Find all CSV files in the directory
    let mut csv_files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(datasets_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "csv") {
                csv_files.push(path);
            }
        }
    } else {
        eprintln!("ERROR: Cannot read directory {}", datasets_dir.display());
        std::process::exit(1);
    }

    csv_files.sort();
    let n_datasets = csv_files.len();

    if n_datasets == 0 {
        eprintln!("ERROR: No CSV files found in {}", datasets_dir.display());
        std::process::exit(1);
    }

    println!("=== Batch Ghost Analysis ({} datasets) ===", n_datasets);
    println!("Directory: {}", datasets_dir.display());
    println!("Column: {}", column);
    println!(
        "Permutations: {}, Bootstrap: {}",
        n_permutations, n_bootstrap
    );
    println!();

    let mut all_p_rigorous: Vec<f64> = Vec::new();
    let mut all_p_permutation: Vec<f64> = Vec::new();
    let mut all_p_bootstrap: Vec<f64> = Vec::new();
    let mut boot_sample_sizes: Vec<f64> = Vec::new();
    let mut sample_sizes: Vec<f64> = Vec::new();
    let mut dataset_names: Vec<String> = Vec::new();

    // Per-dataset analysis
    println!(
        "{:<30}  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
        "Dataset", "N", "p_fdr", "p_bonf", "p_perm", "p_boot", "Verdict"
    );

    for csv_path in &csv_files {
        let name = csv_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let signal = read_csv_column(csv_path, column);
        if signal.is_empty() {
            println!("{:<30}  {:>6}  (no valid data)", name, 0);
            continue;
        }

        let n_samples = signal.len();
        let result = check_ghost_rigorous(&signal, ALIASED_GHOST_FREQ, FREQ_TOL, 0.05);

        let p_perm = if n_permutations > 0 {
            let perm = permutation_test(&signal, ALIASED_GHOST_FREQ, n_permutations, seed);
            perm.p_empirical
        } else {
            1.0
        };

        // Bootstrap null is only meaningful for sorted catalog data.
        // For unsorted data, it compares apples to oranges (unsorted vs sorted spectra).
        let data_is_sorted = is_monotonically_sorted(&signal);
        let p_boot = if n_bootstrap > 0 && data_is_sorted {
            let boot = sorted_distribution_null(&signal, ALIASED_GHOST_FREQ, n_bootstrap, seed);
            boot.p_empirical
        } else {
            f64::NAN
        };

        let p_boot_str = if p_boot.is_nan() {
            "       N/A".to_string()
        } else {
            format!("{:>10.4e}", p_boot)
        };
        println!(
            "{:<30}  {:>6}  {:>10.4e}  {:>10.4e}  {:>10.4e}  {}  {:>8}",
            name, n_samples, result.p_fdr, result.p_bonferroni, p_perm, p_boot_str, result.verdict
        );

        all_p_rigorous.push(result.p_fdr);
        all_p_permutation.push(p_perm);
        // Only include bootstrap p-values for sorted data in combination
        if !p_boot.is_nan() {
            all_p_bootstrap.push(p_boot);
            boot_sample_sizes.push(n_samples as f64);
        }
        sample_sizes.push(n_samples as f64);
        dataset_names.push(name);
    }

    println!();

    // Combined analysis
    if all_p_rigorous.len() >= 2 {
        println!("--- Combined Analysis (Fisher + Stouffer) ---");

        let combined_rig = combine_p_values(&all_p_rigorous, &sample_sizes);
        println!(
            "  FDR p-values:   Fisher chi2={:.2}, p={:.4e}  |  Stouffer Z={:.4}, p={:.4e}",
            combined_rig.fisher_chi2,
            combined_rig.fisher_p,
            combined_rig.stouffer_z,
            combined_rig.stouffer_p
        );

        if !all_p_permutation.iter().all(|&p| p == 1.0) {
            let combined_perm = combine_p_values(&all_p_permutation, &sample_sizes);
            println!(
                "  Permutation:    Fisher chi2={:.2}, p={:.4e}  |  Stouffer Z={:.4}, p={:.4e}",
                combined_perm.fisher_chi2,
                combined_perm.fisher_p,
                combined_perm.stouffer_z,
                combined_perm.stouffer_p
            );
        }

        if !all_p_bootstrap.is_empty() && !all_p_bootstrap.iter().all(|&p| p == 1.0) {
            let combined_boot = combine_p_values(&all_p_bootstrap, &boot_sample_sizes);
            println!(
                "  Bootstrap null: Fisher chi2={:.2}, p={:.4e}  |  Stouffer Z={:.4}, p={:.4e}",
                combined_boot.fisher_chi2,
                combined_boot.fisher_p,
                combined_boot.stouffer_z,
                combined_boot.stouffer_p
            );
        }

        println!();

        // Overall verdict
        let all_null = all_p_rigorous.iter().all(|&p| p >= 0.05);
        let any_detected = all_p_rigorous.iter().any(|&p| p < 0.01);
        let combined_sig = combined_rig.stouffer_p < 0.05;

        println!("=== BATCH VERDICT ===");
        if all_null && !combined_sig {
            println!("NULL across all datasets. No evidence for ghost frequency.");
            println!("The phi^{{-1/2}} spectral line does NOT survive corrected statistics.");
        } else if any_detected && combined_sig {
            println!("DETECTED in at least one dataset with combined significance.");
            println!("Proceed to Tier 2 testing (Lomb-Scargle, multitaper, quantile).");
        } else {
            println!("MIXED results. Some datasets marginal, some null.");
            println!("Combined Stouffer p = {:.4e}", combined_rig.stouffer_p);
        }
    }
}

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Analyze { hdf5, top_k } => run_analyze(&hdf5, top_k),
        Command::Csv {
            path,
            column,
            top_k,
            rigorous,
            permutations,
            bootstrap,
            bonferroni_datasets,
            seed,
        } => run_csv(CsvRunConfig {
            path: &path,
            column: &column,
            top_k,
            rigorous,
            n_permutations: permutations,
            n_bootstrap: bootstrap,
            bonferroni_datasets,
            seed,
        }),
        Command::Synth {
            n,
            inject_ghost,
            amplitude,
            noise_std,
            seed,
        } => run_synth(n, inject_ghost, amplitude, noise_std, seed),
        Command::MultiRate {
            path,
            column,
            stride_min,
            stride_max,
        } => run_multi_rate(&path, &column, stride_min, stride_max),
        Command::Batch {
            datasets,
            column,
            permutations,
            bootstrap,
            seed,
        } => run_batch(&datasets, &column, permutations, bootstrap, seed),
    }
}

#[cfg(test)]
mod tests {
    use spectral_core::ghost_spectral::*;

    #[test]
    fn test_no_ghost_in_noise() {
        use rand::{SeedableRng, rngs::StdRng};
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(12345);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let signal: Vec<f64> = (0..512).map(|_| normal.sample(&mut rng)).collect();

        let result = check_ghost_rigorous(&signal, ALIASED_GHOST_FREQ, FREQ_TOL, 0.05);
        assert_eq!(
            result.verdict,
            GhostVerdict::Null,
            "Pure noise should give NULL verdict, got {}",
            result.verdict
        );
    }

    #[test]
    fn test_rigorous_replaces_legacy_verdict() {
        // The old verdict was rank<=3 && SNR>5.0. The new verdict uses
        // p_fdr < alpha AND p_bonferroni < alpha. These are NOT equivalent:
        // many peaks with SNR > 5 fail FDR correction.
        use rand::{SeedableRng, rngs::StdRng};
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(99999);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let signal: Vec<f64> = (0..256).map(|_| normal.sample(&mut rng)).collect();

        let (freqs, power) = compute_power_spectrum(&signal);
        let peaks = find_peaks(&freqs, &power, 5);
        let nf = noise_floor(&power);

        // Legacy might report a "ghost" in noise (any peak near 0.214)
        let legacy_ghost = check_ghost(&peaks);

        // Rigorous should always return NULL for pure noise
        let rigorous = check_ghost_rigorous(&signal, ALIASED_GHOST_FREQ, FREQ_TOL, 0.05);

        if let Some(gp) = legacy_ghost {
            let snr = peak_snr(gp, nf);
            if gp.rank <= 3 && snr > 5.0 {
                // Legacy would claim "REAL" -- rigorous should disagree
                assert_ne!(
                    rigorous.verdict,
                    GhostVerdict::Detected,
                    "Rigorous should NOT confirm legacy false positive (SNR={:.2})",
                    snr
                );
            }
        }
        // In any case, rigorous on pure noise should be NULL
        assert_eq!(rigorous.verdict, GhostVerdict::Null);
    }
}
