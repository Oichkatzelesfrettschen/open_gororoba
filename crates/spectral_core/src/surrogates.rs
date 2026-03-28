//! Surrogate data testing for spectral significance.
//!
//! Provides:
//! - Phase-randomization surrogates (preserve power spectrum)
//! - IAAFT (Iterative Amplitude Adjusted Fourier Transform) surrogates
//!   (preserve both amplitude distribution and power spectrum)
//! - Empirical p-value estimation
//!
//! # Literature
//! - Theiler et al. (1992): Testing for nonlinearity in time series
//! - Schreiber & Schmitz (1996): Improved surrogate data for nonlinearity tests

use rand::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex64};
use std::f64::consts::PI;

/// Phase-randomization surrogate: preserve power spectrum, randomize phases.
pub fn phase_randomize<R: Rng>(signal: &[f64], rng: &mut R) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut spectrum: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft.process(&mut spectrum);

    let n_half = n / 2;
    for i in 1..n_half + 1 {
        let phase = rng.r#gen::<f64>() * 2.0 * PI;
        let mag = spectrum[i].norm();
        let new_val = Complex64::from_polar(mag, phase);

        spectrum[i] = new_val;
        if i > 0 && i < n - i {
            spectrum[n - i] = new_val.conj();
        }
    }

    // DC and Nyquist (if even) must be real
    spectrum[0] = Complex64::new(spectrum[0].re, 0.0);
    if n.is_multiple_of(2) {
        spectrum[n_half] = Complex64::new(spectrum[n_half].re, 0.0);
    }

    ifft.process(&mut spectrum);
    let scale = 1.0 / n as f64;
    spectrum.iter().map(|c| c.re * scale).collect()
}

/// IAAFT surrogate: preserve both amplitude distribution and power spectrum.
///
/// Algorithm (Schreiber & Schmitz 1996).
pub fn iaaft_surrogate<R: Rng>(signal: &[f64], rng: &mut R, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }

    let mut sorted_original = signal.to_vec();
    sorted_original.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut original_spectrum: Vec<Complex64> =
        signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft.process(&mut original_spectrum);
    let original_amplitudes: Vec<f64> = original_spectrum.iter().map(|c| c.norm()).collect();

    // Start with phase-randomized version
    let mut surrogate = phase_randomize(signal, rng);

    for _ in 0..max_iter {
        // Step A: Rank-order matching (impose original distribution)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| surrogate[a].partial_cmp(&surrogate[b]).unwrap());

        let mut amplitude_matched = vec![0.0; n];
        for (i, &idx) in indices.iter().enumerate() {
            amplitude_matched[idx] = sorted_original[i];
        }

        // Step B: Impose original power spectrum
        let mut spectrum: Vec<Complex64> = amplitude_matched
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        fft.process(&mut spectrum);

        for i in 0..n {
            let phase = spectrum[i].arg();
            spectrum[i] = Complex64::from_polar(original_amplitudes[i], phase);
        }

        ifft.process(&mut spectrum);
        let scale = 1.0 / n as f64;
        let surrogate_new: Vec<f64> = spectrum.iter().map(|c| c.re * scale).collect();

        // Check convergence
        let mut max_diff = 0.0;
        for i in 0..n {
            let diff = (surrogate_new[i] - surrogate[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        surrogate = surrogate_new;
        if max_diff < tol {
            break;
        }
    }

    // Final rank-order matching to ensure exact distribution match
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| surrogate[a].partial_cmp(&surrogate[b]).unwrap());
    let mut final_surrogate = vec![0.0; n];
    for (i, &idx) in indices.iter().enumerate() {
        final_surrogate[idx] = sorted_original[i];
    }

    final_surrogate
}

/// Empirical p-value: fraction of surrogates with statistic >= observed.
pub fn empirical_p_value(observed: f64, surrogates: &[f64]) -> f64 {
    let count_ge = surrogates.iter().filter(|&&s| s >= observed).count();
    (count_ge as f64 + 1.0) / (surrogates.len() as f64 + 1.0)
}

// ---------------------------------------------------------------------------
// Block shuffle surrogate
// ---------------------------------------------------------------------------

/// Block-shuffle surrogate: randomly permute blocks of consecutive samples.
///
/// Preserves within-block autocorrelation while destroying inter-block temporal
/// order. `block_size` controls the trade-off: large blocks preserve more
/// structure; `block_size=1` degenerates to a full shuffle.
pub fn block_shuffle<R: Rng>(signal: &[f64], block_size: usize, rng: &mut R) -> Vec<f64> {
    if signal.is_empty() || block_size == 0 {
        return signal.to_vec();
    }
    let n_blocks = signal.len().div_ceil(block_size);
    let mut block_indices: Vec<usize> = (0..n_blocks).collect();
    block_indices.shuffle(rng);

    let mut result = Vec::with_capacity(signal.len());
    for &bi in &block_indices {
        let start = bi * block_size;
        let end = (start + block_size).min(signal.len());
        result.extend_from_slice(&signal[start..end]);
    }
    result.truncate(signal.len());
    result
}

// ---------------------------------------------------------------------------
// Multivariate phase-randomization surrogates
// ---------------------------------------------------------------------------

/// Multivariate phase-randomization with SHARED random phases across channels.
///
/// Preserves cross-channel coherence structure while randomizing absolute phases.
/// Each channel's FFT magnitudes are preserved, but the same random phase offset
/// is applied to all channels at each frequency bin.
pub fn phase_randomize_mv_shared<R: Rng>(
    channels: &[Vec<f64>],
    rng: &mut R,
) -> Vec<Vec<f64>> {
    if channels.is_empty() {
        return vec![];
    }
    let n = channels[0].len();
    if n == 0 {
        return channels.to_vec();
    }

    // Generate shared random phases
    let n_half = n / 2;
    let random_phases: Vec<f64> = (0..n_half + 1).map(|_| rng.r#gen::<f64>() * 2.0 * PI).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    channels
        .iter()
        .map(|ch| {
            let mut spectrum: Vec<Complex64> =
                ch.iter().map(|&x| Complex64::new(x, 0.0)).collect();
            fft.process(&mut spectrum);

            for i in 1..n_half + 1 {
                let mag = spectrum[i].norm();
                let orig_phase = spectrum[i].arg();
                let new_phase = orig_phase + random_phases[i];
                spectrum[i] = Complex64::from_polar(mag, new_phase);
                if i < n - i {
                    spectrum[n - i] = spectrum[i].conj();
                }
            }
            spectrum[0] = Complex64::new(spectrum[0].re, 0.0);
            if n.is_multiple_of(2) {
                spectrum[n_half] = Complex64::new(spectrum[n_half].re, 0.0);
            }

            ifft.process(&mut spectrum);
            let scale = 1.0 / n as f64;
            spectrum.iter().map(|c| c.re * scale).collect()
        })
        .collect()
}

/// Multivariate phase-randomization with INDEPENDENT random phases per channel.
///
/// Destroys cross-channel phase coupling. Each channel gets its own independent
/// random phase at each frequency. This is the strongest multivariate null:
/// any surviving associator signal after this treatment is a spectral artifact.
pub fn phase_randomize_mv_independent<R: Rng>(
    channels: &[Vec<f64>],
    rng: &mut R,
) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|ch| phase_randomize(ch, rng))
        .collect()
}

// ---------------------------------------------------------------------------
// Circular bootstrap
// ---------------------------------------------------------------------------

/// Circular bootstrap: resample by choosing a random start point and wrapping.
///
/// Preserves the spectral structure (autocorrelation, periodicity) of the
/// original signal by treating it as circular. Returns a signal of the same
/// length starting at a random offset.
pub fn circular_bootstrap<R: Rng>(signal: &[f64], rng: &mut R) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let offset = rng.gen_range(0..n);
    let mut result = Vec::with_capacity(n);
    result.extend_from_slice(&signal[offset..]);
    result.extend_from_slice(&signal[..offset]);
    result
}

/// Circular bootstrap confidence interval for a scalar statistic.
///
/// Computes `n_boot` circular bootstrap replicates, applies `statistic_fn`
/// to each, and returns (lower, upper) percentile CI at the given `alpha`
/// level (e.g., alpha=0.05 for 95% CI).
pub fn circular_bootstrap_ci<R: Rng, F>(
    signal: &[f64],
    statistic_fn: F,
    n_boot: usize,
    alpha: f64,
    rng: &mut R,
) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let mut stats: Vec<f64> = (0..n_boot)
        .map(|_| {
            let replicate = circular_bootstrap(signal, rng);
            statistic_fn(&replicate)
        })
        .collect();
    stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lo_idx = ((alpha / 2.0) * stats.len() as f64).floor() as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * stats.len() as f64).ceil() as usize;
    let lo = stats.get(lo_idx).copied().unwrap_or(f64::NAN);
    let hi = stats.get(hi_idx.min(stats.len() - 1)).copied().unwrap_or(f64::NAN);
    (lo, hi)
}

// ---------------------------------------------------------------------------
// Bayesian change-point detection
// ---------------------------------------------------------------------------

/// Bayesian change-point detection via penalized Gaussian log-likelihood.
///
/// Finds the single change-point that maximizes the total log-likelihood
/// of two independent Gaussian segments, with a BIC-like penalty for the
/// extra parameters. Returns `(change_index, pre_mean, post_mean, delta_bic)`.
///
/// `min_segment` is the minimum number of points in each segment.
pub fn detect_change_point(
    signal: &[f64],
    min_segment: usize,
) -> Option<(usize, f64, f64, f64)> {
    let n = signal.len();
    if n < 2 * min_segment {
        return None;
    }

    let global_mean = signal.iter().sum::<f64>() / n as f64;
    let global_var = signal.iter().map(|&x| (x - global_mean).powi(2)).sum::<f64>() / n as f64;
    if global_var < 1e-30 {
        return None;
    }
    let null_ll = -0.5 * n as f64 * (global_var.ln() + 1.0);

    let mut best_cp = 0;
    let mut best_ll = f64::NEG_INFINITY;

    for cp in min_segment..=(n - min_segment) {
        let pre = &signal[..cp];
        let post = &signal[cp..];

        let pre_mean = pre.iter().sum::<f64>() / pre.len() as f64;
        let post_mean = post.iter().sum::<f64>() / post.len() as f64;

        let pre_var = pre.iter().map(|&x| (x - pre_mean).powi(2)).sum::<f64>() / pre.len() as f64;
        let post_var = post.iter().map(|&x| (x - post_mean).powi(2)).sum::<f64>() / post.len() as f64;

        // Avoid log(0)
        let pre_var = pre_var.max(1e-30);
        let post_var = post_var.max(1e-30);

        let ll = -0.5 * pre.len() as f64 * (pre_var.ln() + 1.0)
            - 0.5 * post.len() as f64 * (post_var.ln() + 1.0);

        if ll > best_ll {
            best_ll = ll;
            best_cp = cp;
        }
    }

    // BIC penalty: 2 extra parameters (second mean + second variance)
    let penalty = 2.0 * (n as f64).ln();
    let delta_bic = 2.0 * (best_ll - null_ll) - penalty;

    let pre_mean = signal[..best_cp].iter().sum::<f64>() / best_cp as f64;
    let post_mean = signal[best_cp..].iter().sum::<f64>() / (n - best_cp) as f64;

    Some((best_cp, pre_mean, post_mean, delta_bic))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_phase_randomize_preserves_mean() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let surr = phase_randomize(&signal, &mut rng);

        let mean_orig = signal.iter().sum::<f64>() / 5.0;
        let mean_surr = surr.iter().sum::<f64>() / 5.0;
        assert!((mean_orig - mean_surr).abs() < 1e-10);
    }

    #[test]
    fn test_iaaft_preserves_distribution() {
        let signal = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut surr = iaaft_surrogate(&signal, &mut rng, 100, 1e-8);

        let mut sorted_orig = signal.clone();
        sorted_orig.sort_by(|a, b| a.partial_cmp(b).unwrap());
        surr.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (a, b) in sorted_orig.iter().zip(surr.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
