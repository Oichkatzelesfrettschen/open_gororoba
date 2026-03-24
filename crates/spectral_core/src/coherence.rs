//! Magnitude-squared coherence and cross-spectral methods.
//!
//! Provides:
//! - Welch's method for PSD and CPSD
//! - Magnitude-squared coherence (MSC)
//! - Binomial test for pairwise significance

use rustfft::{FftPlanner, num_complex::Complex64};
use std::f64::consts::PI;

/// Compute the cross-power spectral density (CPSD) using Welch's method.
pub fn welch_cpsd(
    x: &[f64],
    y: &[f64],
    nperseg: usize,
    noverlap: usize,
) -> (Vec<f64>, Vec<Complex64>) {
    let n = x.len().min(y.len());
    if n < nperseg {
        return (vec![], vec![]);
    }

    let step = nperseg - noverlap;
    let n_freq = nperseg / 2 + 1;
    let mut sum_cpsd = vec![Complex64::new(0.0, 0.0); n_freq];
    let mut n_segments = 0;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nperseg);

    // Hann window
    let window: Vec<f64> = (0..nperseg)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / nperseg as f64).cos()))
        .collect();
    let win_norm = window.iter().map(|&w| w * w).sum::<f64>();

    let mut i = 0;
    while i + nperseg <= n {
        let mut x_seg: Vec<Complex64> = x[i..i + nperseg]
            .iter()
            .zip(window.iter())
            .map(|(&xi, &wi)| Complex64::new(xi * wi, 0.0))
            .collect();
        let mut y_seg: Vec<Complex64> = y[i..i + nperseg]
            .iter()
            .zip(window.iter())
            .map(|(&yi, &wi)| Complex64::new(yi * wi, 0.0))
            .collect();

        fft.process(&mut x_seg);
        fft.process(&mut y_seg);

        for j in 0..n_freq {
            sum_cpsd[j] += x_seg[j] * y_seg[j].conj();
        }
        n_segments += 1;
        i += step;
    }

    let scale = 1.0 / (n_segments as f64 * win_norm);
    for val in &mut sum_cpsd {
        *val *= scale;
    }

    let frequencies: Vec<f64> = (0..n_freq).map(|j| j as f64 / nperseg as f64).collect();
    (frequencies, sum_cpsd)
}

/// Compute magnitude-squared coherence (MSC).
///
/// MSC(f) = |CPSD(f)|^2 / (PSD_x(f) * PSD_y(f))
pub fn magnitude_squared_coherence(
    x: &[f64],
    y: &[f64],
    nperseg: usize,
    noverlap: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (freqs, cpsd) = welch_cpsd(x, y, nperseg, noverlap);
    let (_, psd_x_complex) = welch_cpsd(x, x, nperseg, noverlap);
    let (_, psd_y_complex) = welch_cpsd(y, y, nperseg, noverlap);

    let mut msc = Vec::with_capacity(cpsd.len());
    for i in 0..cpsd.len() {
        let den = psd_x_complex[i].re * psd_y_complex[i].re;
        if den > 1e-15 {
            msc.push(cpsd[i].norm_sqr() / den);
        } else {
            msc.push(0.0);
        }
    }

    (freqs, msc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_coherence_identical_signals() {
        let n = 1024;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let (freqs, msc) = magnitude_squared_coherence(&x, &x, 256, 128);

        assert_eq!(freqs.len(), 129);
        // Coherence of identical signals should be 1.0
        for &val in &msc {
            if val > 1e-10 {
                // some bins might be zero if signal has no power there
                assert!((val - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_coherence_independent_noise() {
        let n = 1024;
        let mut rng = rand::thread_rng();
        let x: Vec<f64> = (0..n).map(|_| rng.r#gen()).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.r#gen()).collect();

        let (_, msc) = magnitude_squared_coherence(&x, &y, 256, 128);

        let mean_msc = msc.iter().sum::<f64>() / msc.len() as f64;
        // For independent signals, coherence should be small (~ 1/L where L=n_segments)
        assert!(mean_msc < 0.5);
    }
}
