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

/// Cross-spectral partial coherence: C_xy.z
///
/// Measures the coherence between x and y after removing the linear
/// influence of a control variable z. Uses the standard formula:
///   C_xy.z = (C_xy - C_xz * conj(C_yz)) / sqrt((1-|C_xz|^2)(1-|C_yz|^2))
///
/// Returns magnitude-squared partial coherence per frequency bin.
pub fn partial_coherence(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    nperseg: usize,
    noverlap: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (freqs, cpsd_xy) = welch_cpsd(x, y, nperseg, noverlap);
    let (_, cpsd_xz) = welch_cpsd(x, z, nperseg, noverlap);
    let (_, cpsd_yz) = welch_cpsd(y, z, nperseg, noverlap);
    let (_, psd_x) = welch_cpsd(x, x, nperseg, noverlap);
    let (_, psd_y) = welch_cpsd(y, y, nperseg, noverlap);
    let (_, psd_z) = welch_cpsd(z, z, nperseg, noverlap);

    let mut pcoh = Vec::with_capacity(freqs.len());
    for i in 0..freqs.len() {
        let sx = psd_x[i].re;
        let sy = psd_y[i].re;
        let sz = psd_z[i].re;
        if sx < 1e-30 || sy < 1e-30 || sz < 1e-30 {
            pcoh.push(0.0);
            continue;
        }

        let c_xy = cpsd_xy[i] / (sx * sy).sqrt();
        let c_xz = cpsd_xz[i] / (sx * sz).sqrt();
        let c_yz = cpsd_yz[i] / (sy * sz).sqrt();

        let numerator = c_xy - c_xz * c_yz.conj();
        let denom_sq = (1.0 - c_xz.norm_sqr()) * (1.0 - c_yz.norm_sqr());
        if denom_sq < 1e-30 {
            pcoh.push(0.0);
        } else {
            pcoh.push(numerator.norm_sqr() / denom_sq);
        }
    }

    (freqs, pcoh)
}

/// Decompose spectral power into field-parallel and field-perpendicular
/// fractions relative to a reference direction b_hat.
///
/// For each frequency bin, computes the fraction of B-field spectral power
/// in the direction of b_hat (parallel/compressive) vs perpendicular
/// (transverse/Alfvenic-like).
///
/// Returns (frequencies, parallel_fraction, perpendicular_fraction).
pub fn field_aligned_spectral_fractions(
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    b_hat: [f64; 3],
    nperseg: usize,
    noverlap: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = bx.len().min(by.len()).min(bz.len());
    if n < nperseg {
        return (vec![], vec![], vec![]);
    }

    let step = nperseg - noverlap;
    let n_freq = nperseg / 2 + 1;
    let mut par_power = vec![0.0; n_freq];
    let mut total_power = vec![0.0; n_freq];

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nperseg);

    let window: Vec<f64> = (0..nperseg)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / nperseg as f64).cos()))
        .collect();

    let mut i = 0;
    while i + nperseg <= n {
        let mut fx: Vec<Complex64> = bx[i..i + nperseg]
            .iter().zip(window.iter())
            .map(|(&v, &w)| Complex64::new(v * w, 0.0)).collect();
        let mut fy: Vec<Complex64> = by[i..i + nperseg]
            .iter().zip(window.iter())
            .map(|(&v, &w)| Complex64::new(v * w, 0.0)).collect();
        let mut fz: Vec<Complex64> = bz[i..i + nperseg]
            .iter().zip(window.iter())
            .map(|(&v, &w)| Complex64::new(v * w, 0.0)).collect();

        fft.process(&mut fx);
        fft.process(&mut fy);
        fft.process(&mut fz);

        for j in 0..n_freq {
            let bvec = [fx[j], fy[j], fz[j]];
            let par_component = bvec[0] * b_hat[0] + bvec[1] * b_hat[1] + bvec[2] * b_hat[2];
            let par_p = par_component.norm_sqr();
            let total_p = bvec[0].norm_sqr() + bvec[1].norm_sqr() + bvec[2].norm_sqr();
            par_power[j] += par_p;
            total_power[j] += total_p;
        }
        i += step;
    }

    let frequencies: Vec<f64> = (0..n_freq).map(|j| j as f64 / nperseg as f64).collect();
    let mut par_frac = Vec::with_capacity(n_freq);
    let mut perp_frac = Vec::with_capacity(n_freq);
    for j in 0..n_freq {
        if total_power[j] > 1e-30 {
            let pf = par_power[j] / total_power[j];
            par_frac.push(pf);
            perp_frac.push(1.0 - pf);
        } else {
            par_frac.push(0.0);
            perp_frac.push(0.0);
        }
    }

    (frequencies, par_frac, perp_frac)
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
