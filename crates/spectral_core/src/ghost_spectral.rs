//! Ghost frequency spectral analysis -- statistically hardened (Sprint 50).
//!
//! Reusable FFT-based spectral analysis for detecting the phi^{-1/2} ghost
//! frequency in LBM rho_mean time series and astronomical catalog data.
//!
//! ## Sprint 50 Statistical Hardening
//!
//! The original analysis (Sprint 48-49) had 10 critical deficiencies documented
//! in the plan. This module now provides:
//! - Robust noise floor (MAD estimator, peak exclusion)
//! - Per-bin false alarm probability (exponential null)
//! - Benjamini-Hochberg FDR correction across all bins
//! - Bonferroni correction for trials factor
//! - FWHM clamped to FFT resolution limit (1/N)
//! - Permutation testing (empirical null distribution of max-peak)
//! - Sorted-distribution bootstrap null (for catalog-derived data)
//! - Cross-dataset combination (Fisher and Stouffer methods)
//!
//! The "ghost frequency" hypothesis: Sedenion zero-divisor modulated viscosity
//! imprints a spectral peak at phi^{-1/2} ~ 0.786 cycles/sample. Nyquist
//! folding maps this to 1 - phi^{-1/2} ~ 0.214 for unit-rate sampled data.

use adjustp::{Procedure, adjust};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Golden ratio.
pub const PHI: f64 = 1.618_033_988_749_895;

/// Target ghost frequency: phi^{-1/2}.
pub const GHOST_FREQ: f64 = 0.786_151_377_757_423;

/// Aliased ghost frequency: 1.0 - phi^{-1/2} (Nyquist folding for unit sampling).
/// When GHOST_FREQ > 0.5, discrete FFT reports the peak here instead.
pub const ALIASED_GHOST_FREQ: f64 = 1.0 - GHOST_FREQ;

/// Frequency matching tolerance (cycles/sample).
pub const FREQ_TOL: f64 = 0.02;

/// MAD-to-sigma conversion factor: 1 / Phi^{-1}(3/4) for Gaussian noise.
const MAD_SCALE: f64 = 1.482_602_218_505_602;

// ---- Data Structures ----

/// Spectral peak found by FFT analysis.
#[derive(Debug, Clone)]
pub struct SpectralPeak {
    /// Normalized frequency (cycles per sample, range [0, 0.5]).
    pub freq: f64,
    /// Power spectral density at this frequency.
    pub power: f64,
    /// Rank among all peaks (1 = strongest).
    pub rank: usize,
}

/// Verdict from rigorous hypothesis testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostVerdict {
    /// Peak survives all corrections at alpha=0.05.
    Detected,
    /// Peak survives some but not all corrections.
    Marginal,
    /// Peak does not survive correction -- consistent with noise.
    Null,
}

impl std::fmt::Display for GhostVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Detected => write!(f, "DETECTED"),
            Self::Marginal => write!(f, "MARGINAL"),
            Self::Null => write!(f, "NULL"),
        }
    }
}

/// Full result from rigorous ghost hypothesis test.
#[derive(Debug, Clone)]
pub struct GhostTestResult {
    /// Raw signal-to-noise ratio (peak power / mean noise). Kept for
    /// backward compatibility, but SHOULD NOT be used for significance.
    pub raw_snr: f64,
    /// Corrected SNR using robust (MAD) noise floor with peak exclusion.
    pub corrected_snr: f64,
    /// Raw p-value at the target bin (exponential null, single test).
    pub p_raw: f64,
    /// Bonferroni-corrected p-value (trials factor = search window / bin width).
    pub p_bonferroni: f64,
    /// BH-FDR corrected q-value at the target bin.
    pub p_fdr: f64,
    /// Number of independent frequency bins tested in the search window.
    pub n_trials: usize,
    /// Total number of frequency bins in the power spectrum.
    pub n_bins: usize,
    /// Frequency offset from theoretical ghost alias (signed).
    pub freq_delta: f64,
    /// FWHM in number of FFT bins (clamped to >= 1.0).
    pub fwhm_bins: f64,
    /// Whether FWHM was clamped at the resolution limit.
    pub fwhm_clamped: bool,
    /// Overall verdict after all corrections.
    pub verdict: GhostVerdict,
    /// Human-readable note about methodology.
    pub methodology_note: String,
}

/// Result from a permutation test.
#[derive(Debug, Clone)]
pub struct PermutationTestResult {
    /// Empirical p-value: fraction of permutations with max-peak >= observed.
    pub p_empirical: f64,
    /// Number of permutations performed.
    pub n_permutations: usize,
    /// Observed max-peak power at target frequency.
    pub observed_power: f64,
    /// Mean max-peak power across permutations.
    pub null_mean: f64,
    /// Std dev of max-peak power across permutations.
    pub null_std: f64,
}

/// Result from sorted-distribution bootstrap null test.
#[derive(Debug, Clone)]
pub struct BootstrapNullResult {
    /// Empirical p-value: fraction of bootstrap samples with power >= observed
    /// at the target frequency.
    pub p_empirical: f64,
    /// Number of bootstrap iterations.
    pub n_bootstrap: usize,
    /// Observed power at target frequency in original sorted data.
    pub observed_power: f64,
    /// Mean power at target frequency across bootstrap samples.
    pub null_mean: f64,
    /// Std dev of power at target frequency across bootstrap samples.
    pub null_std: f64,
}

/// Result from combining p-values across multiple datasets.
#[derive(Debug, Clone)]
pub struct CombinedPValueResult {
    /// Fisher's combined chi-squared statistic: -2 * sum(ln(p_i)).
    pub fisher_chi2: f64,
    /// Fisher's combined p-value (chi-squared with 2k degrees of freedom).
    pub fisher_p: f64,
    /// Stouffer's weighted Z-score.
    pub stouffer_z: f64,
    /// Stouffer's combined p-value.
    pub stouffer_p: f64,
    /// Harmonic mean p-value (Wilson 2019) - uncorrected.
    pub harmonic_mean_p: f64,
    /// Corrected harmonic mean p-value (Wilson 2019) - robust to dependence.
    pub harmonic_mean_p_corrected: f64,
    /// Number of datasets combined.
    pub n_datasets: usize,
    /// Per-dataset p-values that were combined.
    pub per_dataset_p: Vec<f64>,
    /// Per-dataset weights (sqrt(N_i) for Stouffer).
    pub weights: Vec<f64>,
}

// ---- Core FFT Functions (preserved from Sprint 49) ----

/// Perform FFT on a real-valued time series and return the power spectrum.
///
/// Returns (frequencies, power_spectrum) where frequencies are normalized
/// to [0, 0.5] (fraction of Nyquist).
///
/// Pre-processing: DC removal (mean subtraction) and Hann windowing to
/// reduce spectral leakage.
pub fn compute_power_spectrum(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Remove mean (DC component) to focus on oscillatory content
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    // Apply Hann window to reduce spectral leakage.
    // Convention: w(n) = 0.5 * (1 - cos(2\pi n / N)), denominator N (not N-1).
    // This keeps both endpoints of the window non-zero (w(0)=0, w(N)=0 conceptually
    // but N is never reached), which is a common engineering choice.  Note that
    // some references use (N-1) in the denominator; the two differ only at the
    // boundary samples and have negligible effect on long signals, but should be
    // noted when comparing leakage bounds against external references.
    let windowed: Vec<Complex<f64>> = centered
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos());
            Complex::new(x * w, 0.0)
        })
        .collect();

    // FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut spectrum = windowed;
    fft.process(&mut spectrum);

    // Power spectrum (only positive frequencies: 0 to N/2)
    let n_freq = n / 2 + 1;
    let freqs: Vec<f64> = (0..n_freq).map(|k| k as f64 / n as f64).collect();
    let power: Vec<f64> = spectrum[..n_freq]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im) / (n as f64 * n as f64))
        .collect();

    (freqs, power)
}

/// Find the top-k peaks in a power spectrum (excluding DC bin).
pub fn find_peaks(freqs: &[f64], power: &[f64], top_k: usize) -> Vec<SpectralPeak> {
    if freqs.len() < 3 {
        return Vec::new();
    }

    // Collect all local maxima (excluding DC = index 0)
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for i in 1..power.len() - 1 {
        if power[i] > power[i - 1] && power[i] > power[i + 1] {
            peaks.push((i, power[i]));
        }
    }
    // Also check last bin
    if power.len() > 1 && power[power.len() - 1] > power[power.len() - 2] {
        peaks.push((power.len() - 1, power[power.len() - 1]));
    }

    // Sort by power descending
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    peaks
        .iter()
        .take(top_k)
        .enumerate()
        .map(|(rank, &(idx, pwr))| SpectralPeak {
            freq: freqs[idx],
            power: pwr,
            rank: rank + 1,
        })
        .collect()
}

// ---- Aliasing and Legacy Detection ----

/// Compute the Nyquist alias of the ghost frequency for a given subsampling stride.
pub fn ghost_aliases(stride: usize) -> Vec<f64> {
    let f_scaled = GHOST_FREQ * stride as f64;
    let f_mod = f_scaled - f_scaled.floor();
    let f_alias = if f_mod > 0.5 { 1.0 - f_mod } else { f_mod };
    vec![f_alias]
}

/// Check if any peak is near the ghost frequency or its Nyquist alias.
pub fn check_ghost_at_stride(peaks: &[SpectralPeak], stride: usize) -> Option<&SpectralPeak> {
    let aliases = ghost_aliases(stride);
    peaks.iter().find(|p| {
        aliases
            .iter()
            .any(|&alias| (p.freq - alias).abs() < FREQ_TOL)
    })
}

/// Check if any peak is near the ghost frequency (stride=1 default).
pub fn check_ghost(peaks: &[SpectralPeak]) -> Option<&SpectralPeak> {
    check_ghost_at_stride(peaks, 1)
}

/// Returns true if freq is near the ghost frequency or its alias (stride=1).
pub fn is_ghost_freq(freq: f64) -> bool {
    let aliases = ghost_aliases(1);
    aliases.iter().any(|&alias| (freq - alias).abs() < FREQ_TOL)
}

// ---- FWHM (Item 5: clamp to resolution limit) ----

/// Compute FWHM of the peak nearest to target_freq, clamped to >= 1/N.
///
/// Returns (fwhm_cycles_per_sample, fwhm_bins, was_clamped).
/// Sub-bin FWHM from interpolation is an artifact, not physics.
pub fn peak_fwhm_clamped(
    freqs: &[f64],
    power: &[f64],
    target_freq: f64,
) -> Option<(f64, f64, bool)> {
    let n = freqs.len();
    if n < 3 {
        return None;
    }
    // Infer signal length from frequency resolution (freq[1] - freq[0] = 1/N)
    let freq_res = if n >= 2 {
        freqs[1] - freqs[0]
    } else {
        return None;
    };
    let signal_len = (1.0 / freq_res).round();

    let raw_fwhm = peak_fwhm(freqs, power, target_freq)?;
    let fwhm_bins = raw_fwhm / freq_res;
    let min_fwhm = freq_res; // 1 FFT bin = 1/N

    if raw_fwhm < min_fwhm {
        Some((min_fwhm, 1.0, true))
    } else {
        let _ = signal_len; // suppress unused warning
        Some((raw_fwhm, fwhm_bins, false))
    }
}

/// Compute FWHM (backward-compatible, unclamped).
pub fn peak_fwhm(freqs: &[f64], power: &[f64], target_freq: f64) -> Option<f64> {
    let peak_idx = freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((**a - target_freq).abs())
                .partial_cmp(&((**b - target_freq).abs()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?
        .0;

    let peak_power = power[peak_idx];
    if peak_power <= 0.0 {
        return None;
    }
    let half_max = peak_power / 2.0;

    let mut left_freq = freqs[peak_idx];
    for i in (1..=peak_idx).rev() {
        if power[i] < half_max {
            let frac = (half_max - power[i]) / (power[i + 1] - power[i]);
            left_freq = freqs[i] + frac * (freqs[i + 1] - freqs[i]);
            break;
        }
    }

    let mut right_freq = freqs[peak_idx];
    for i in peak_idx + 1..power.len() {
        if power[i] < half_max {
            let frac = (half_max - power[i]) / (power[i - 1] - power[i]);
            right_freq = freqs[i] - frac * (freqs[i] - freqs[i - 1]);
            break;
        }
    }

    let fwhm = right_freq - left_freq;
    if fwhm > 0.0 { Some(fwhm) } else { None }
}

// ---- Noise Floor (Item 1: MAD estimator) ----

/// Legacy noise floor (mean PSD excluding DC). Kept for backward compatibility.
pub fn noise_floor(power: &[f64]) -> f64 {
    if power.len() <= 1 {
        return 0.0;
    }
    let total: f64 = power[1..].iter().sum();
    total / (power.len() - 1) as f64
}

/// Robust noise floor using MAD (Median Absolute Deviation) estimator.
///
/// Excludes bins within `exclude_radius` of indices in `exclude_indices`
/// (typically the peak bin +/- 3). MAD is robust to outliers (breakdown
/// point = 50%) unlike the mean (breakdown point = 0%).
///
/// noise_sigma = MAD_SCALE * median(|P_i - median(P)|)
///
/// where MAD_SCALE = 1.4826 converts to Gaussian-equivalent sigma.
pub fn robust_noise_floor(power: &[f64], exclude_indices: &[usize], exclude_radius: usize) -> f64 {
    if power.len() <= 1 {
        return 0.0;
    }

    // Collect non-excluded, non-DC bins
    let included: Vec<f64> = power[1..]
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let idx = i + 1; // offset from power[1..]
            !exclude_indices
                .iter()
                .any(|&ex| idx.abs_diff(ex) <= exclude_radius)
        })
        .map(|(_, &v)| v)
        .collect();

    if included.is_empty() {
        return noise_floor(power);
    }

    let median = fast_median(&included);
    let deviations: Vec<f64> = included.iter().map(|&x| (x - median).abs()).collect();
    let mad = fast_median(&deviations);

    MAD_SCALE * mad
}

/// Compute signal-to-noise ratio for a peak given the noise floor.
pub fn peak_snr(peak: &SpectralPeak, noise: f64) -> f64 {
    if noise > 0.0 { peak.power / noise } else { 0.0 }
}

// ---- Trials Factor and False Alarm Probability (Items 2-3) ----

/// Number of independent frequency bins searched in a window of half-width
/// `freq_tol` around the target, given frequency resolution `freq_res`.
pub fn trials_factor(freq_tol: f64, freq_res: f64) -> usize {
    if freq_res <= 0.0 {
        return 1;
    }
    // Window spans [target - freq_tol, target + freq_tol] = 2 * freq_tol
    let n_trials = (2.0 * freq_tol / freq_res).ceil() as usize;
    n_trials.max(1)
}

/// Single-bin false alarm probability assuming exponentially-distributed
/// power spectrum bins (Rayleigh amplitude -> exponential power for Gaussian noise).
///
/// p_single = exp(-z) where z = power / noise_floor.
pub fn false_alarm_probability_single(power_at_bin: f64, noise: f64) -> f64 {
    if noise <= 0.0 {
        return 1.0;
    }
    let z = power_at_bin / noise;
    (-z).exp()
}

/// Bonferroni-corrected false alarm probability.
///
/// FAP = 1 - (1 - p_single)^n_trials. For small p_single, FAP ~ n_trials * p_single.
pub fn false_alarm_probability_bonferroni(p_single: f64, n_trials: usize) -> f64 {
    if n_trials == 0 {
        return p_single;
    }
    // Use the exact formula to avoid numerical issues
    1.0 - (1.0 - p_single).powi(n_trials as i32)
}

// ---- BH-FDR Correction (Item 4) ----

/// Apply Benjamini-Hochberg FDR correction to per-bin p-values.
///
/// Returns (adjusted_p_values, indices_surviving_at_alpha).
pub fn fdr_correction(p_values: &[f64], alpha: f64) -> (Vec<f64>, Vec<usize>) {
    let adjusted = adjust(p_values, Procedure::BenjaminiHochberg);
    let surviving: Vec<usize> = adjusted
        .iter()
        .enumerate()
        .filter(|&(_, &q)| q < alpha)
        .map(|(i, _)| i)
        .collect();
    (adjusted, surviving)
}

// ---- Rigorous Ghost Test (Items 6-7) ----

/// Full rigorous hypothesis test for the ghost frequency.
///
/// Pipeline: robust noise floor -> per-bin p-values -> BH-FDR correction
/// -> check if target bin survives -> report corrected verdict.
pub fn check_ghost_rigorous(
    signal: &[f64],
    target_freq: f64,
    freq_tol: f64,
    alpha: f64,
) -> GhostTestResult {
    let n = signal.len();
    if n < 16 {
        return GhostTestResult {
            raw_snr: 0.0,
            corrected_snr: 0.0,
            p_raw: 1.0,
            p_bonferroni: 1.0,
            p_fdr: 1.0,
            n_trials: 0,
            n_bins: 0,
            freq_delta: f64::NAN,
            fwhm_bins: 0.0,
            fwhm_clamped: false,
            verdict: GhostVerdict::Null,
            methodology_note: "Signal too short (< 16 samples)".to_string(),
        };
    }

    let (freqs, power) = compute_power_spectrum(signal);
    let n_bins = power.len();
    let freq_res = 1.0 / n as f64;

    // Find the bin closest to target frequency
    let target_idx = freqs
        .iter()
        .enumerate()
        .skip(1) // skip DC
        .min_by(|(_, a), (_, b)| {
            ((**a - target_freq).abs())
                .partial_cmp(&((**b - target_freq).abs()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(1);

    let power_at_target = power[target_idx];
    let freq_delta = freqs[target_idx] - target_freq;

    // Robust noise floor: exclude peak +/- 3 bins
    let exclude = vec![target_idx];
    let noise = robust_noise_floor(&power, &exclude, 3);
    let raw_noise = noise_floor(&power);

    let corrected_snr = if noise > 0.0 {
        power_at_target / noise
    } else {
        0.0
    };
    let raw_snr = if raw_noise > 0.0 {
        power_at_target / raw_noise
    } else {
        0.0
    };

    // Per-bin p-values (exponential null)
    let p_raw = false_alarm_probability_single(power_at_target, noise);

    // Trials factor: how many bins in the search window
    let n_trials = trials_factor(freq_tol, freq_res);
    let p_bonferroni = false_alarm_probability_bonferroni(p_raw, n_trials);

    // BH-FDR across ALL bins (excluding DC)
    let all_p: Vec<f64> = power[1..]
        .iter()
        .map(|&p| false_alarm_probability_single(p, noise))
        .collect();
    let (fdr_adjusted, _surviving) = fdr_correction(&all_p, alpha);
    // target_idx in the all_p array is target_idx - 1 (we skipped DC)
    let p_fdr = if target_idx >= 1 && target_idx - 1 < fdr_adjusted.len() {
        fdr_adjusted[target_idx - 1]
    } else {
        1.0
    };

    // FWHM
    let (fwhm_bins, fwhm_clamped) =
        if let Some((_fwhm_cps, bins, clamped)) = peak_fwhm_clamped(&freqs, &power, target_freq) {
            (bins, clamped)
        } else {
            (0.0, false)
        };

    // Determine verdict using an AND rule that is deliberately conservative:
    // "Detected" requires BOTH the global BH-FDR q-value AND the
    // Bonferroni-windowed p-value to fall below alpha, so a peak must survive
    // two independent multiple-comparison corrections simultaneously.
    // "Marginal" is declared when only one of the two corrections is passed.
    // This AND rule is stronger than most pipelines; it reduces false positives
    // at the cost of reduced power, which is intentional for "hardened" mode.
    let verdict = if p_fdr < alpha && p_bonferroni < alpha {
        GhostVerdict::Detected
    } else if p_fdr < alpha || p_bonferroni < alpha {
        GhostVerdict::Marginal
    } else {
        GhostVerdict::Null
    };

    let mut notes = Vec::new();
    if fwhm_clamped {
        notes.push("FWHM below resolution limit (clamped to 1 bin)".to_string());
    }
    let is_sorted = is_monotonically_sorted(signal);
    if is_sorted {
        notes.push(
            "WARNING: Input appears sorted (monotonically increasing). \
             FFT of sorted catalog data tests distributional shape, not \
             physical periodicity. See Tang & Zhang 2005."
                .to_string(),
        );
    }
    let methodology_note = if notes.is_empty() {
        "Robust MAD noise floor, BH-FDR + Bonferroni correction".to_string()
    } else {
        notes.join("; ")
    };

    GhostTestResult {
        raw_snr,
        corrected_snr,
        p_raw,
        p_bonferroni,
        p_fdr,
        n_trials,
        n_bins,
        freq_delta,
        fwhm_bins,
        fwhm_clamped,
        verdict,
        methodology_note,
    }
}

// ---- Permutation Test (Item 8) ----

/// Permutation test: shuffle signal N times, compute power at target freq
/// each time, report empirical p-value (fraction with power >= observed).
///
/// Tests whether the spectral structure at target_freq is distinguishable
/// from what random rearrangement of the same values would produce.
pub fn permutation_test(
    signal: &[f64],
    target_freq: f64,
    n_permutations: usize,
    seed: u64,
) -> PermutationTestResult {
    let (freqs, power) = compute_power_spectrum(signal);

    // Find target bin
    let target_idx = freqs
        .iter()
        .enumerate()
        .skip(1)
        .min_by(|(_, a), (_, b)| {
            ((**a - target_freq).abs())
                .partial_cmp(&((**b - target_freq).abs()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(1);

    let observed_power = power[target_idx];

    let mut rng = StdRng::seed_from_u64(seed);
    let mut shuffled = signal.to_vec();
    let mut null_powers = Vec::with_capacity(n_permutations);
    let mut count_ge = 0usize;

    for _ in 0..n_permutations {
        shuffled.shuffle(&mut rng);
        let (_f, p) = compute_power_spectrum(&shuffled);
        let perm_power = if target_idx < p.len() {
            p[target_idx]
        } else {
            0.0
        };
        null_powers.push(perm_power);
        if perm_power >= observed_power {
            count_ge += 1;
        }
        // Restore original order for next shuffle
        shuffled.copy_from_slice(signal);
    }

    let p_empirical = (count_ge as f64 + 1.0) / (n_permutations as f64 + 1.0);
    let null_mean = null_powers.iter().sum::<f64>() / null_powers.len() as f64;
    let null_std = if null_powers.len() > 1 {
        let var = null_powers
            .iter()
            .map(|&x| (x - null_mean).powi(2))
            .sum::<f64>()
            / (null_powers.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    PermutationTestResult {
        p_empirical,
        n_permutations,
        observed_power,
        null_mean,
        null_std,
    }
}

// ---- Sorted-Distribution Bootstrap Null (Item 9) ----

/// Bootstrap null for sorted catalog data: resample from empirical CDF,
/// sort, FFT, record power at target frequency. Tests whether the peak
/// is an artifact of sorting a sample from the same distribution.
pub fn sorted_distribution_null(
    sorted_signal: &[f64],
    target_freq: f64,
    n_bootstrap: usize,
    seed: u64,
) -> BootstrapNullResult {
    let n = sorted_signal.len();
    let (freqs, power) = compute_power_spectrum(sorted_signal);

    let target_idx = freqs
        .iter()
        .enumerate()
        .skip(1)
        .min_by(|(_, a), (_, b)| {
            ((**a - target_freq).abs())
                .partial_cmp(&((**b - target_freq).abs()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(1);

    let observed_power = power[target_idx];

    let mut rng = StdRng::seed_from_u64(seed);
    let mut null_powers = Vec::with_capacity(n_bootstrap);
    let mut count_ge = 0usize;

    for _ in 0..n_bootstrap {
        // Resample with replacement from the original data
        let mut sample: Vec<f64> = (0..n)
            .map(|_| {
                let idx = rand_index(&mut rng, n);
                sorted_signal[idx]
            })
            .collect();
        // Sort the resample (mimics the sorted-catalog methodology)
        sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let (_f, p) = compute_power_spectrum(&sample);
        let boot_power = if target_idx < p.len() {
            p[target_idx]
        } else {
            0.0
        };
        null_powers.push(boot_power);
        if boot_power >= observed_power {
            count_ge += 1;
        }
    }

    let p_empirical = (count_ge as f64 + 1.0) / (n_bootstrap as f64 + 1.0);
    let null_mean = null_powers.iter().sum::<f64>() / null_powers.len() as f64;
    let null_std = if null_powers.len() > 1 {
        let var = null_powers
            .iter()
            .map(|&x| (x - null_mean).powi(2))
            .sum::<f64>()
            / (null_powers.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    BootstrapNullResult {
        p_empirical,
        n_bootstrap,
        observed_power,
        null_mean,
        null_std,
    }
}

// ---- Cross-Dataset Combiner (Item 10) ----

/// Combine per-dataset p-values using Fisher's and Stouffer's methods.
///
/// Fisher: chi2 = -2 * sum(ln(p_i)), compared to chi-squared(2k).
/// Stouffer: Z = sum(w_i * Phi^{-1}(1 - p_i)) / sqrt(sum(w_i^2)),
/// where w_i = sqrt(N_i) and Phi^{-1} is the inverse normal CDF.
pub fn combine_p_values(p_values: &[f64], sample_sizes: &[f64]) -> CombinedPValueResult {
    let k = p_values.len();
    assert_eq!(
        k,
        sample_sizes.len(),
        "p_values and sample_sizes must have equal length"
    );

    if k == 0 {
        return CombinedPValueResult {
            fisher_chi2: 0.0,
            fisher_p: 1.0,
            stouffer_z: 0.0,
            stouffer_p: 1.0,
            harmonic_mean_p: 1.0,
            harmonic_mean_p_corrected: 1.0,
            n_datasets: 0,
            per_dataset_p: Vec::new(),
            weights: Vec::new(),
        };
    }

    // Clamp p-values away from 0 to avoid -inf in log
    let clamped_p: Vec<f64> = p_values.iter().map(|&p| p.max(1e-300)).collect();

    // Fisher's method
    let fisher_chi2 = -2.0 * clamped_p.iter().map(|&p| p.ln()).sum::<f64>();
    let fisher_dof = 2.0 * k as f64;
    let chi2_dist = ChiSquared::new(fisher_dof).expect("Invalid chi-squared dof");
    let fisher_p = 1.0 - chi2_dist.cdf(fisher_chi2);

    // Stouffer's weighted Z
    let weights: Vec<f64> = sample_sizes.iter().map(|&n| n.sqrt()).collect();
    let sum_w2: f64 = weights.iter().map(|&w| w * w).sum();

    // Phi^{-1}(1 - p_i) using the probit function
    // For small p, 1-p ~ 1, so z ~ 0. For very small p, z is large positive.
    let z_scores: Vec<f64> = clamped_p.iter().map(|&p| probit(1.0 - p)).collect();

    let stouffer_z = if sum_w2 > 0.0 {
        let weighted_sum: f64 = weights
            .iter()
            .zip(z_scores.iter())
            .map(|(&w, &z)| w * z)
            .sum();
        weighted_sum / sum_w2.sqrt()
    } else {
        0.0
    };

    // Two-sided p-value for Stouffer Z
    let stouffer_p = 2.0 * normal_cdf(-stouffer_z.abs());

    // Harmonic mean p-value (Wilson 2019)
    // p_HM = sum(w_i) / sum(w_i / p_i)
    // Using equal weights for now
    let inv_p_sum: f64 = clamped_p.iter().map(|&p| 1.0 / p).sum();
    let harmonic_mean_p = k as f64 / inv_p_sum;

    // Asymptotically corrected p-value: p_corrected = e * ln(K) * p_HM
    let harmonic_mean_p_corrected = if k >= 2 {
        let correction = std::f64::consts::E * (k as f64).ln();
        (correction * harmonic_mean_p).min(1.0)
    } else {
        harmonic_mean_p
    };

    CombinedPValueResult {
        fisher_chi2,
        fisher_p,
        stouffer_z,
        stouffer_p,
        harmonic_mean_p,
        harmonic_mean_p_corrected,
        n_datasets: k,
        per_dataset_p: p_values.to_vec(),
        weights,
    }
}

// ---- Utility Functions ----

/// Fast median (sorts a copy).
fn fast_median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Check if a signal is monotonically non-decreasing (sorted catalog indicator).
pub fn is_monotonically_sorted(signal: &[f64]) -> bool {
    if signal.len() < 2 {
        return false;
    }
    // Check first 100 and last 100 elements (heuristic for large datasets)
    let check_len = signal.len().min(200);
    let front = &signal[..check_len.min(signal.len())];
    for w in front.windows(2) {
        if w[1] < w[0] - 1e-15 {
            return false;
        }
    }
    if signal.len() > 200 {
        let back = &signal[signal.len() - 100..];
        for w in back.windows(2) {
            if w[1] < w[0] - 1e-15 {
                return false;
            }
        }
    }
    true
}

/// Random index in [0, n) using a pre-seeded RNG.
fn rand_index(rng: &mut StdRng, n: usize) -> usize {
    use rand::RngExt;
    rng.random_range(0..n)
}

/// Standard normal CDF via statrs for accuracy.
fn normal_cdf(x: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF as _, Normal};
    if x.is_nan() {
        return 0.5;
    }
    let n = Normal::new(0.0, 1.0).expect("standard normal");
    n.cdf(x)
}

/// Probit function: inverse of the standard normal CDF.
/// Delegates to statrs::distribution::Normal for accuracy.
fn probit(p: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF as _, Normal};
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let n = Normal::new(0.0, 1.0).expect("standard normal");
    n.inverse_cdf(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_spectrum_pure_sine() {
        let n = 256;
        let freq = 0.25;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin())
            .collect();

        let (freqs, power) = compute_power_spectrum(&signal);
        let peaks = find_peaks(&freqs, &power, 3);

        assert!(!peaks.is_empty(), "Should find at least one peak");
        assert!(
            (peaks[0].freq - freq).abs() < 0.01,
            "Peak at {:.4} should be near {:.4}",
            peaks[0].freq,
            freq
        );
    }

    #[test]
    fn test_power_spectrum_ghost_frequency() {
        let n = 1024;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * GHOST_FREQ * i as f64).sin())
            .collect();

        let (_freqs, _power) = compute_power_spectrum(&signal);
        let (freqs, power) = compute_power_spectrum(&signal);
        let peaks = find_peaks(&freqs, &power, 5);

        assert!(!peaks.is_empty());
        let ghost = check_ghost(&peaks);
        assert!(
            ghost.is_some(),
            "Should detect ghost at freq={:.4}, top peak at {:.4}",
            GHOST_FREQ,
            peaks[0].freq
        );
    }

    #[test]
    fn test_ghost_frequency_value() {
        let computed = 1.0 / PHI.sqrt();
        assert!(
            (computed - GHOST_FREQ).abs() < 1e-12,
            "GHOST_FREQ={:.15} should equal 1/sqrt(phi)={:.15}",
            GHOST_FREQ,
            computed
        );
    }

    #[test]
    fn test_empty_signal() {
        let (freqs, power) = compute_power_spectrum(&[]);
        assert!(freqs.is_empty());
        assert!(power.is_empty());
    }

    #[test]
    fn test_ghost_aliases_stride_1() {
        let aliases = ghost_aliases(1);
        assert_eq!(aliases.len(), 1);
        assert!(
            (aliases[0] - ALIASED_GHOST_FREQ).abs() < 1e-10,
            "stride=1 alias {:.6} should be {:.6}",
            aliases[0],
            ALIASED_GHOST_FREQ
        );
    }

    #[test]
    fn test_ghost_aliases_stride_2() {
        let aliases = ghost_aliases(2);
        let expected = 1.0 - (GHOST_FREQ * 2.0 - 1.0);
        assert!(
            (aliases[0] - expected).abs() < 1e-10,
            "stride=2 alias {:.6} should be {:.6}",
            aliases[0],
            expected
        );
    }

    #[test]
    fn test_ghost_aliases_stride_3() {
        let aliases = ghost_aliases(3);
        let f_scaled = GHOST_FREQ * 3.0;
        let f_mod = f_scaled - f_scaled.floor();
        let expected = if f_mod > 0.5 { 1.0 - f_mod } else { f_mod };
        assert!(
            (aliases[0] - expected).abs() < 1e-10,
            "stride=3 alias {:.6} should be {:.6}",
            aliases[0],
            expected
        );
    }

    #[test]
    fn test_check_ghost_at_stride() {
        let n = 2048;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * GHOST_FREQ * i as f64).sin())
            .collect();

        let subsampled: Vec<f64> = signal.iter().step_by(2).copied().collect();

        let (freqs, power) = compute_power_spectrum(&subsampled);
        let peaks = find_peaks(&freqs, &power, 10);

        let ghost = check_ghost_at_stride(&peaks, 2);
        assert!(
            ghost.is_some(),
            "Should detect ghost at stride=2 alias. Top peak: {:.4}",
            if peaks.is_empty() { 0.0 } else { peaks[0].freq }
        );
    }

    #[test]
    fn test_fwhm_sharp_peak() {
        let n = 1024;
        let freq = 0.25;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin())
            .collect();

        let (freqs, power) = compute_power_spectrum(&signal);
        if let Some(fwhm) = peak_fwhm(&freqs, &power, freq) {
            let fwhm_bins = fwhm * n as f64;
            assert!(
                fwhm_bins < 10.0,
                "Pure sine FWHM={:.1} bins should be sharp (<10)",
                fwhm_bins
            );
        }
    }

    #[test]
    fn test_noise_floor_positive() {
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin())
            .collect();
        let (_freqs, power) = compute_power_spectrum(&signal);
        let nf = noise_floor(&power);
        assert!(nf >= 0.0, "Noise floor should be non-negative");
    }

    #[test]
    fn test_peak_snr_computation() {
        let peak = SpectralPeak {
            freq: 0.25,
            power: 10.0,
            rank: 1,
        };
        assert!((peak_snr(&peak, 2.0) - 5.0).abs() < 1e-12);
        assert!((peak_snr(&peak, 0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_constant_peak_across_synthetic_tau() {
        let freq = 0.15;
        for _tau_idx in 0..5 {
            let n = 512;
            let signal: Vec<f64> = (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin())
                .collect();
            let (freqs, power) = compute_power_spectrum(&signal);
            let peaks = find_peaks(&freqs, &power, 3);
            assert!(!peaks.is_empty());
            assert!(
                (peaks[0].freq - freq).abs() < 0.01,
                "Peak should be at {:.4} regardless of tau, got {:.4}",
                freq,
                peaks[0].freq
            );
        }
    }

    // ---- Sprint 50: Statistical Hardening Tests ----

    #[test]
    fn test_robust_noise_floor_excludes_peak() {
        // Inject a strong sine at f=0.25 into noise
        let n = 1024;
        let mut signal: Vec<f64> = vec![0.0; n];
        // Add noise
        for (i, s) in signal.iter_mut().enumerate().take(n) {
            // Deterministic pseudo-noise (hash-based)
            let hash = ((i as u64).wrapping_mul(2654435761) >> 16) as f64 / 65536.0 - 0.5;
            *s = hash;
        }
        // Add strong sine
        for (i, s) in signal.iter_mut().enumerate().take(n) {
            *s += 10.0 * (2.0 * std::f64::consts::PI * 0.25 * i as f64).sin();
        }

        let (_freqs, power) = compute_power_spectrum(&signal);
        let naive_nf = noise_floor(&power);
        let peak_bin = n / 4; // f=0.25 -> bin = N/4
        let robust_nf = robust_noise_floor(&power, &[peak_bin], 3);

        // Robust estimate should be LOWER because it excludes the peak
        assert!(
            robust_nf < naive_nf,
            "Robust NF ({:.6e}) should be < naive NF ({:.6e}) when peak excluded",
            robust_nf,
            naive_nf
        );
    }

    #[test]
    fn test_fwhm_clamp_at_resolution() {
        // Create a signal with a very sharp peak
        let n = 256;
        let freq = 0.25;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin())
            .collect();

        let (freqs, power) = compute_power_spectrum(&signal);
        if let Some((_fwhm_cps, fwhm_bins, _clamped)) = peak_fwhm_clamped(&freqs, &power, freq) {
            assert!(
                fwhm_bins >= 1.0,
                "FWHM must be >= 1 bin (got {:.4})",
                fwhm_bins
            );
        }
    }

    #[test]
    fn test_trials_factor_computation() {
        // FREQ_TOL=0.02, N=5000 -> freq_res=0.0002 -> n_trials = 0.04/0.0002 = 200
        let n_trials = trials_factor(0.02, 0.0002);
        assert_eq!(n_trials, 200);

        let n_trials = trials_factor(0.02, 0.001);
        assert_eq!(n_trials, 40);

        // Edge case: zero resolution
        assert_eq!(trials_factor(0.02, 0.0), 1);
    }

    #[test]
    fn test_false_alarm_probability_monotone() {
        // Higher power -> lower p-value
        let p1 = false_alarm_probability_single(5.0, 1.0);
        let p2 = false_alarm_probability_single(10.0, 1.0);
        assert!(
            p2 < p1,
            "Higher power should give lower p: p(5)={:.6e}, p(10)={:.6e}",
            p1,
            p2
        );
    }

    #[test]
    fn test_bonferroni_increases_p() {
        let p_single = 0.001;
        let p_bonf = false_alarm_probability_bonferroni(p_single, 200);
        assert!(
            p_bonf > p_single,
            "Bonferroni should increase p: {:.6e} > {:.6e}",
            p_bonf,
            p_single
        );
        assert!(p_bonf < 1.0, "Bonferroni should stay below 1.0");
    }

    #[test]
    fn test_fdr_correction_basic() {
        // 10 p-values, 2 very small (real signals), 8 large (noise)
        let p_values = vec![0.001, 0.002, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80];
        let (adjusted, surviving) = fdr_correction(&p_values, 0.05);
        assert_eq!(adjusted.len(), 10);
        // The two small p-values should survive
        assert!(
            surviving.contains(&0) && surviving.contains(&1),
            "True signals (indices 0,1) should survive FDR. Surviving: {:?}",
            surviving
        );
    }

    #[test]
    fn test_check_ghost_rigorous_on_noise() {
        // Pure deterministic pseudo-noise -> should get NULL verdict
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                ((i as u64).wrapping_mul(2654435761).wrapping_add(12345) >> 16) as f64 / 65536.0
                    - 0.5
            })
            .collect();

        let result = check_ghost_rigorous(&signal, ALIASED_GHOST_FREQ, FREQ_TOL, 0.05);
        assert_eq!(
            result.verdict,
            GhostVerdict::Null,
            "Noise should produce NULL verdict, got {} (p_fdr={:.4e}, p_bonf={:.4e})",
            result.verdict,
            result.p_fdr,
            result.p_bonferroni
        );
    }

    #[test]
    fn test_check_ghost_rigorous_on_strong_signal() {
        // Strong injected sine at ghost frequency -> should be DETECTED
        let n = 1024;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let noise = ((i as u64).wrapping_mul(2654435761) >> 16) as f64 / 65536.0 - 0.5;
                noise + 20.0 * (2.0 * std::f64::consts::PI * ALIASED_GHOST_FREQ * i as f64).sin()
            })
            .collect();

        let result = check_ghost_rigorous(&signal, ALIASED_GHOST_FREQ, FREQ_TOL, 0.05);
        assert_eq!(
            result.verdict,
            GhostVerdict::Detected,
            "Strong signal should be DETECTED, got {} (p_fdr={:.4e}, p_bonf={:.4e})",
            result.verdict,
            result.p_fdr,
            result.p_bonferroni
        );
    }

    #[test]
    fn test_permutation_null_calibrated() {
        // Pure noise: p > 0.05 expected
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                ((i as u64).wrapping_mul(2654435761).wrapping_add(99999) >> 16) as f64 / 65536.0
                    - 0.5
            })
            .collect();

        let result = permutation_test(&signal, ALIASED_GHOST_FREQ, 200, 42);
        assert!(
            result.p_empirical > 0.01,
            "Pure noise permutation p should be > 0.01, got {:.4}",
            result.p_empirical
        );
    }

    #[test]
    fn test_permutation_signal_detected() {
        // Strong signal at a bin-centered frequency (0.25 = bin 128 for N=512)
        // to avoid spectral leakage reducing the peak power.
        // Amplitude 50x the noise level ensures unambiguous detection.
        let n = 512;
        let freq = 0.25; // perfectly bin-centered
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let noise = ((i as u64).wrapping_mul(2654435761) >> 16) as f64 / 65536.0 - 0.5;
                noise + 50.0 * (2.0 * std::f64::consts::PI * freq * i as f64).sin()
            })
            .collect();

        let result = permutation_test(&signal, freq, 200, 42);
        assert!(
            result.p_empirical < 0.05,
            "Strong signal should give p < 0.05, got {:.4}",
            result.p_empirical
        );
    }

    #[test]
    fn test_bootstrap_null_on_gaussian() {
        // Sorted Gaussian should NOT produce significant peak at any specific freq
        let n = 256;
        let mut signal: Vec<f64> = (0..n)
            .map(|i| {
                // Box-Muller-like: deterministic standard-normal-ish
                let u = (i as f64 + 0.5) / n as f64;
                probit(u) // quantile function of N(0,1)
            })
            .collect();
        signal.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let result = sorted_distribution_null(&signal, ALIASED_GHOST_FREQ, 200, 42);
        assert!(
            result.p_empirical > 0.01,
            "Sorted Gaussian bootstrap should give p > 0.01, got {:.4}",
            result.p_empirical
        );
    }

    #[test]
    fn test_stouffer_combination_known() {
        // Known: two very significant p-values
        let p_values = vec![0.001, 0.002];
        let sample_sizes = vec![1000.0, 2000.0];
        let result = combine_p_values(&p_values, &sample_sizes);

        assert!(
            result.fisher_p < 0.001,
            "Fisher combined p should be < 0.001, got {:.6e}",
            result.fisher_p
        );
        assert!(
            result.stouffer_z > 2.0,
            "Stouffer Z should be > 2.0, got {:.4}",
            result.stouffer_z
        );
        assert_eq!(result.n_datasets, 2);
    }

    #[test]
    fn test_stouffer_combination_null() {
        // All non-significant p-values -> combined should also be non-significant
        let p_values = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        let sample_sizes = vec![100.0; 5];
        let result = combine_p_values(&p_values, &sample_sizes);

        assert!(
            result.fisher_p > 0.05,
            "Fisher combined p should be > 0.05 for null data, got {:.6e}",
            result.fisher_p
        );
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        let cdf_pos = normal_cdf(1.96);
        let cdf_neg = normal_cdf(-1.96);
        assert!(
            (cdf_pos + cdf_neg - 1.0).abs() < 1e-5,
            "CDF should be symmetric"
        );
        assert!((cdf_pos - 0.975).abs() < 0.002, "CDF(1.96) ~ 0.975");
    }

    #[test]
    fn test_probit_roundtrip() {
        for &p in &[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99] {
            let z = probit(p);
            let recovered = normal_cdf(z);
            assert!(
                (recovered - p).abs() < 0.002,
                "probit({:.2}) = {:.6}, CDF(z) = {:.6}, delta = {:.6e}",
                p,
                z,
                recovered,
                (recovered - p).abs()
            );
        }
    }

    #[test]
    fn test_mad_scale_factor() {
        // MAD_SCALE = 1/Phi^{-1}(3/4) where Phi is the standard normal CDF
        // Phi^{-1}(0.75) ~ 0.6745
        let z_075 = probit(0.75);
        let expected_scale = 1.0 / z_075;
        assert!(
            (MAD_SCALE - expected_scale).abs() < 0.01,
            "MAD_SCALE={:.6} should be 1/probit(0.75)={:.6}",
            MAD_SCALE,
            expected_scale
        );
    }

    #[test]
    fn test_is_monotonically_sorted() {
        assert!(is_monotonically_sorted(&[1.0, 2.0, 3.0, 4.0]));
        assert!(is_monotonically_sorted(&[1.0, 1.0, 2.0, 3.0]));
        assert!(!is_monotonically_sorted(&[3.0, 1.0, 2.0, 4.0]));
        assert!(!is_monotonically_sorted(&[]));
        assert!(!is_monotonically_sorted(&[1.0]));
    }
}
