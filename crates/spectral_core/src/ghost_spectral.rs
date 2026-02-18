//! Ghost frequency spectral analysis.
//!
//! Reusable FFT-based spectral analysis for detecting the phi^{-1/2} ghost
//! frequency in LBM rho_mean time series. Extracted from the rho-ghost-fft
//! binary for use across multiple analysis pipelines.
//!
//! The "ghost frequency" hypothesis: Sedenion zero-divisor modulated viscosity
//! imprints a spectral peak at phi^{-1/2} ~ 0.786 cycles/sample. Nyquist
//! folding maps this to 1 - phi^{-1/2} ~ 0.214 for unit-rate sampled data.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Golden ratio.
pub const PHI: f64 = 1.618_033_988_749_895;

/// Target ghost frequency: phi^{-1/2}.
pub const GHOST_FREQ: f64 = 0.786_151_377_757_423;

/// Aliased ghost frequency: 1.0 - phi^{-1/2} (Nyquist folding for unit sampling).
/// When GHOST_FREQ > 0.5, discrete FFT reports the peak here instead.
pub const ALIASED_GHOST_FREQ: f64 = 1.0 - GHOST_FREQ;

/// Frequency matching tolerance (cycles/sample).
pub const FREQ_TOL: f64 = 0.02;

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

    // Apply Hann window to reduce spectral leakage
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

/// Compute the full set of Nyquist aliases for a given true frequency.
///
/// When subsampling by stride S, the effective sampling rate is 1/S and
/// the true frequency in the subsampled domain is f_true * S. The observed
/// alias folds into [0, 0.5] via: f_obs = |frac(f_true * S) - round(frac(f_true * S))|
pub fn ghost_aliases(stride: usize) -> Vec<f64> {
    let f_scaled = GHOST_FREQ * stride as f64;
    // Fold into [0, 1) then into [0, 0.5]
    let f_mod = f_scaled - f_scaled.floor();
    let f_alias = if f_mod > 0.5 { 1.0 - f_mod } else { f_mod };
    vec![f_alias]
}

/// Check if any peak is near the ghost frequency or ANY of its Nyquist aliases
/// for a given subsampling stride.
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
    aliases
        .iter()
        .any(|&alias| (freq - alias).abs() < FREQ_TOL)
}

/// Compute the full-width at half-maximum (FWHM) of the peak nearest to
/// the given frequency. Returns FWHM in cycles/sample.
///
/// A razor-sharp peak (algebraic origin) has FWHM ~ 1/N (one FFT bin).
/// A spectrally blurred peak (physical/chirping) has FWHM >> 1/N.
pub fn peak_fwhm(freqs: &[f64], power: &[f64], target_freq: f64) -> Option<f64> {
    // Find the bin closest to target_freq
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

    // Walk left from peak to find half-max crossing
    let mut left_freq = freqs[peak_idx];
    for i in (1..=peak_idx).rev() {
        if power[i] < half_max {
            // Linear interpolation between bins i and i+1
            let frac = (half_max - power[i]) / (power[i + 1] - power[i]);
            left_freq = freqs[i] + frac * (freqs[i + 1] - freqs[i]);
            break;
        }
    }

    // Walk right from peak
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

/// Compute noise floor as mean PSD excluding DC bin.
pub fn noise_floor(power: &[f64]) -> f64 {
    if power.len() <= 1 {
        return 0.0;
    }
    let total: f64 = power[1..].iter().sum();
    total / (power.len() - 1) as f64
}

/// Compute signal-to-noise ratio for a peak given the noise floor.
pub fn peak_snr(peak: &SpectralPeak, noise: f64) -> f64 {
    if noise > 0.0 {
        peak.power / noise
    } else {
        0.0
    }
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
            if peaks.is_empty() {
                0.0
            } else {
                peaks[0].freq
            }
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
        // Synthetic test: injected frequency should not depend on a "tau" parameter.
        // This validates the analysis pipeline, not physics.
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
}
