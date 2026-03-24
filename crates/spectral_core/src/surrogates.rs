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
