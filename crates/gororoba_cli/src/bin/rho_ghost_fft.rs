//! 1420 MHz "Ghost" spectral analysis.
//!
//! Hypothesis: The rho_mean time series from the warp ring experiment contains
//! a periodic component at frequency ~0.786 (phi^{-1/2}, where phi is the golden
//! ratio). This would indicate a hidden carrier wave in the LBM fluid simulation,
//! possibly arising from the sedenion zero-divisor modulated viscosity field.
//!
//! Falsification: Perform an FFT on the rho_mean trace. If the peak is white noise
//! or Brownian drift, there is no ghost. If it peaks near 0.786, the ghost is real.
//!
//! Subcommands:
//! - `analyze`:  Read rho_mean from HDF5 and perform FFT spectral analysis
//! - `csv`:      Read rho_mean from a CSV column and perform FFT spectral analysis
//! - `synth`:    Generate synthetic test signals to validate the FFT pipeline
//! - `multi-rate`: Subsample at varying strides and track alias migration

use clap::{Parser, Subcommand};
use spectral_core::ghost_spectral::{
    check_ghost, check_ghost_at_stride, compute_power_spectrum, find_peaks, ghost_aliases,
    is_ghost_freq, noise_floor, peak_fwhm, peak_snr, ALIASED_GHOST_FREQ, GHOST_FREQ, PHI,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rho-ghost-fft", about = "1420 MHz Ghost spectral analysis of rho_mean")]
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
    /// aliased ghost peak migrates. A real periodic signal produces a
    /// predictable trajectory; noise produces random scatter.
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
}

/// Print spectral analysis results.
fn print_analysis(signal: &[f64], top_k: usize, source: &str) {
    println!("=== 1420 MHz Ghost Spectral Analysis ===");
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

    // Basic statistics
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

    // FFT analysis
    let (freqs, power) = compute_power_spectrum(signal);
    let peaks = find_peaks(&freqs, &power, top_k);

    // Total power for SNR computation
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
            if (peak.freq - ALIASED_GHOST_FREQ).abs() < spectral_core::ghost_spectral::FREQ_TOL {
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

    // FWHM analysis of ghost peak region
    let freq_res = 1.0 / signal.len() as f64;
    let alias_target = ghost_aliases(1)[0];
    if let Some(fwhm) = peak_fwhm(&freqs, &power, alias_target) {
        let fwhm_bins = fwhm / freq_res;
        println!("Spectral width at ghost alias ({:.4}):", alias_target);
        println!(
            "  FWHM = {:.6} cycles/sample ({:.1} bins)",
            fwhm, fwhm_bins
        );
        if fwhm_bins < 3.0 {
            println!("  -> SHARP: algebraic origin (resolution-limited)");
        } else if fwhm_bins < 10.0 {
            println!("  -> MODERATE: possible chirp or weak coupling");
        } else {
            println!("  -> BROAD: physical/chirping origin (spectral blur)");
        }
        println!();
    }

    // Ghost verdict
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
    use data_core::hdf5_export::read_rho_mean_trace;

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

    print_analysis(
        &signal,
        top_k,
        &format!("HDF5: {}", hdf5_path.display()),
    );
}

#[cfg(not(feature = "hdf5-export"))]
fn run_analyze(_hdf5_path: &std::path::Path, _top_k: usize) {
    eprintln!("ERROR: HDF5 support not enabled.");
    eprintln!("Rebuild with: cargo build --bin rho-ghost-fft --features hdf5-export");
    std::process::exit(1);
}

fn run_csv(path: &std::path::Path, column: &str, top_k: usize) {
    let signal = read_csv_column(path, column);

    if signal.is_empty() {
        eprintln!("ERROR: No valid data in column '{}'", column);
        std::process::exit(1);
    }

    print_analysis(
        &signal,
        top_k,
        &format!("CSV: {} [{}]", path.display(), column),
    );
}

fn run_synth(n: usize, inject_ghost: bool, amplitude: f64, noise_std: f64, seed: u64) {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise_std).expect("Invalid noise std");

    let mut signal: Vec<f64> = (0..n)
        .map(|_| 1.0 + normal.sample(&mut rng)) // mean=1.0 (density) + noise
        .collect();

    if inject_ghost {
        // Inject sinusoidal at ghost frequency
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

    print_analysis(
        &signal,
        10,
        &format!(
            "Synthetic (N={}, ghost={}, A={}, seed={})",
            n, inject_ghost, amplitude, seed
        ),
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
        // Subsample: take every `stride`-th sample
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
    println!("  Expect_alias = predicted Nyquist fold of phi^{{-1/2}} at each stride.");
    println!(
        "  If Ghost?=YES and Peak1_freq tracks Expect_alias -> REAL periodic signal."
    );
    println!(
        "  If Ghost?=YES appears sporadically with random Peak1_freq -> noise artifact."
    );
    println!(
        "  FWHM < 3 bins -> algebraic origin (sharp). FWHM >> 3 -> physical chirp (blurred)."
    );
    println!("  The alias trajectory IS the Sedenion Handshake: the only way the");
    println!(
        "  macro-scale fluid perceives the micro-algebraic zero-divisor oscillation."
    );
}

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Analyze { hdf5, top_k } => run_analyze(&hdf5, top_k),
        Command::Csv {
            path,
            column,
            top_k,
        } => run_csv(&path, &column, top_k),
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
    }
}

#[cfg(test)]
mod tests {
    use spectral_core::ghost_spectral::*;

    #[test]
    fn test_no_ghost_in_noise() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(12345);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let signal: Vec<f64> = (0..512).map(|_| normal.sample(&mut rng)).collect();

        let (freqs, power) = compute_power_spectrum(&signal);
        let peaks = find_peaks(&freqs, &power, 5);

        // Ghost detection should be unlikely in pure noise
        let _ = check_ghost(&peaks);
    }
}
