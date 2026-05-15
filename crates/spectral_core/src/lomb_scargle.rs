//! Lomb-Scargle periodogram for irregularly sampled data.
//!
//! Provides the Generalized Lomb-Scargle periodogram (with floating mean)
//! and analytic false alarm probability (FAP) estimators, including Baluev (2008).
//!
//! # Literature
//! - Lomb (1976): Least-squares frequency analysis of unequally spaced data
//! - Scargle (1982): Studies in astronomical time series analysis
//! - Zechmeister & Kurster (2009): A generalized Lomb-Scargle periodogram
//! - Baluev (2008): Assessing the statistical significance of periodogram peaks

use std::f64::consts::PI;

/// Lomb-Scargle periodogram result.
#[derive(Debug, Clone)]
pub struct LombScargleResult {
    /// Frequencies at which the periodogram was evaluated.
    pub frequencies: Vec<f64>,
    /// Periodogram power at each frequency.
    pub power: Vec<f64>,
}

impl LombScargleResult {
    /// Find the peak with maximum power.
    pub fn max_peak(&self) -> (f64, f64) {
        let mut max_p = -1.0;
        let mut max_f = 0.0;
        for (&f, &p) in self.frequencies.iter().zip(self.power.iter()) {
            if p > max_p {
                max_p = p;
                max_f = f;
            }
        }
        (max_f, max_p)
    }
}

/// Compute the Generalized Lomb-Scargle periodogram with unit weights.
///
/// This version includes a floating mean offset (Zechmeister & Kurster
/// 2009). For per-sample uncertainties use
/// [`compute_lomb_scargle_weighted`] instead.
///
/// # Arguments
/// * `t` - Observation times
/// * `y` - Observed values
/// * `freqs` - Frequencies to evaluate
pub fn compute_lomb_scargle(t: &[f64], y: &[f64], freqs: &[f64]) -> LombScargleResult {
    let n = t.len();
    let w = vec![1.0_f64; n];
    compute_lomb_scargle_weighted(t, y, &w, freqs)
}

/// Compute the Generalized Lomb-Scargle periodogram with per-sample
/// weights (Zechmeister & Kurster 2009, full Eq. 5 form).
///
/// # Weight convention
///
/// `weights[i]` is the *statistical* weight of sample `i`, typically
/// `1 / sigma_i^2` for Gaussian measurement uncertainties `sigma_i`.
/// The internal normalization divides by `sum(weights)`, so absolute
/// scaling is irrelevant -- only the ratio between samples matters.
/// Pass all-`1.0` weights to recover the unit-weight variant
/// [`compute_lomb_scargle`].
///
/// # Why this exists
///
/// The original `compute_lomb_scargle` carried a "Assuming unit weights
/// for now" comment because the `(t, y, freqs)` signature has no slot
/// for sigmas. For heteroscedastic data (NANOGrav residuals, Pantheon+
/// SNe distance moduli, MaNGA per-spaxel uncertainties), uniform-weight
/// LS over-counts noisy samples and under-weights precise ones. The
/// Zechmeister & Kurster Eq. 5 form replaces every plain sum
/// `sum_i f(t_i, y_i)` with the weighted sum
/// `sum_i w_i f(t_i, y_i) / sum_i w_i` and reduces to the unit-weight
/// formula when all `w_i` are equal.
///
/// # Panics
///
/// Panics if `t`, `y`, and `weights` do not have equal length, or if
/// any weight is negative.
pub fn compute_lomb_scargle_weighted(
    t: &[f64],
    y: &[f64],
    weights: &[f64],
    freqs: &[f64],
) -> LombScargleResult {
    let n = t.len();
    assert_eq!(n, y.len(), "t and y must have equal length");
    assert_eq!(n, weights.len(), "t and weights must have equal length");
    for (i, &w) in weights.iter().enumerate() {
        assert!(w >= 0.0, "weight at index {} is negative: {}", i, w);
    }

    let w_sum: f64 = weights.iter().sum();
    if w_sum <= 0.0 {
        // All weights zero -> ill-posed; return zero-power result so
        // callers can detect the degenerate case via the output instead
        // of via a panic.
        return LombScargleResult {
            frequencies: freqs.to_vec(),
            power: vec![0.0; freqs.len()],
        };
    }

    let y_mean: f64 = y
        .iter()
        .zip(weights.iter())
        .map(|(&yi, &wi)| wi * yi)
        .sum::<f64>()
        / w_sum;
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();
    let yy_sum: f64 = y_centered
        .iter()
        .zip(weights.iter())
        .map(|(&yi, &wi)| wi * yi * yi)
        .sum();

    let mut power = Vec::with_capacity(freqs.len());

    for &f in freqs {
        let omega = 2.0 * PI * f;

        let mut s = 0.0;
        let mut c = 0.0;
        let mut ss = 0.0;
        let mut cc = 0.0;
        let mut sc = 0.0;
        let mut yc = 0.0;
        let mut ys = 0.0;

        for (i, &ti) in t.iter().enumerate() {
            let (si, ci) = (omega * ti).sin_cos();
            let wi = weights[i];
            let yi = y_centered[i];

            s += wi * si;
            c += wi * ci;
            ss += wi * si * si;
            cc += wi * ci * ci;
            sc += wi * si * ci;
            yc += wi * yi * ci;
            ys += wi * yi * si;
        }

        // Floating mean normalization (Zechmeister & Kurster Eq. 5).
        // The weighted means s_hat = sum(w_i sin)/sum(w_i),
        // c_hat = sum(w_i cos)/sum(w_i) shift the basis so the residual
        // 2x2 system has determinant `det` and the periodogram power
        // follows directly.
        let s_hat = s / w_sum;
        let c_hat = c / w_sum;

        let cc_tilde = cc - c * c_hat;
        let ss_tilde = ss - s * s_hat;
        let sc_tilde = sc - c * s_hat;
        // y_centered already has weighted mean 0, so the cross terms
        // collapse to plain weighted sums.
        let yc_tilde = yc;
        let ys_tilde = ys;

        // Determinant of the 2x2 reduced system.
        let det = cc_tilde * ss_tilde - sc_tilde * sc_tilde;

        if det.abs() < 1e-15 || yy_sum.abs() < 1e-15 {
            power.push(0.0);
            continue;
        }

        let p = (ss_tilde * yc_tilde * yc_tilde + cc_tilde * ys_tilde * ys_tilde
            - 2.0 * sc_tilde * yc_tilde * ys_tilde)
            / (det * yy_sum);

        power.push(p.clamp(0.0, 1.0));
    }

    LombScargleResult {
        frequencies: freqs.to_vec(),
        power,
    }
}

/// Compute the False Alarm Probability (FAP) using the Baluev (2008) method.
///
/// This provides a tight upper bound for the FAP of the maximum peak in the periodogram.
///
/// # Arguments
/// * `z` - Maximum observed power (normalized to [0, 1])
/// * `n` - Number of observations
/// * `f_max` - Maximum frequency in the search range
/// * `t_range` - Range of observation times (max(t) - min(t))
pub fn false_alarm_probability_baluev(z: f64, n: usize, f_max: f64, t_range: f64) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }
    if z >= 1.0 {
        return 0.0;
    }

    let n_f = n as f64;

    // P(Z > z) for a single frequency
    // For normalized power z in [0, 1], the null distribution is Beta((1, (N-3)/2))
    // P(Z > z) = (1 - z)^((N-3)/2)
    let p_single = (1.0 - z).powf((n_f - 3.0) / 2.0);

    // Alias factor / effective number of independent frequencies
    // W = f_max * t_range
    let w = f_max * t_range;
    let tau = (8.0 * PI).sqrt() * w * (1.0 - z).powf((n_f - 4.0) / 2.0) * z.sqrt();

    let fap = p_single + tau;
    fap.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lomb_scargle_sine() {
        let n = 100;
        let freq_true = 0.234;
        let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * freq_true * ti).sin())
            .collect();

        let freqs: Vec<f64> = (1..500).map(|i| i as f64 / 1000.0).collect();
        let result = compute_lomb_scargle(&t, &y, &freqs);

        let (max_f, max_p) = result.max_peak();
        assert!((max_f - freq_true).abs() < 0.01);
        assert!(max_p > 0.9);
    }

    #[test]
    fn test_baluev_fap_high_snr() {
        let fap = false_alarm_probability_baluev(0.95, 100, 0.5, 100.0);
        println!("Baluev FAP (0.95, 100, 0.5, 100.0) = {:.6e}", fap);
        assert!(fap < 1e-5);
    }

    #[test]
    fn test_baluev_fap_low_snr() {
        let fap = false_alarm_probability_baluev(0.05, 100, 0.5, 100.0);
        assert!(fap > 0.1);
    }

    #[test]
    fn weighted_with_unit_weights_matches_unweighted() {
        // Unit weights on the weighted variant must recover the unweighted
        // periodogram to floating-point precision.
        let n = 64;
        let freq_true = 0.117;
        let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * freq_true * ti).cos())
            .collect();
        let freqs: Vec<f64> = (1..200).map(|i| i as f64 / 1000.0).collect();

        let unweighted = compute_lomb_scargle(&t, &y, &freqs);
        let weights = vec![1.0_f64; n];
        let weighted = compute_lomb_scargle_weighted(&t, &y, &weights, &freqs);

        assert_eq!(unweighted.power.len(), weighted.power.len());
        for (a, b) in unweighted.power.iter().zip(weighted.power.iter()) {
            assert!((a - b).abs() < 1e-12, "unweighted={} weighted={}", a, b);
        }
    }

    #[test]
    fn weighted_downweights_noisy_samples() {
        // Construct a signal sin(2 pi f t) and add a large outlier to one
        // sample. With uniform weights the periodogram peak is shifted
        // by the outlier; with that sample given a near-zero weight, the
        // recovered peak frequency is closer to the true value.
        let n = 256;
        let freq_true = 0.073;
        let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut y: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * freq_true * ti).sin())
            .collect();
        y[42] += 50.0;

        let freqs: Vec<f64> = (1..500).map(|i| i as f64 / 5000.0).collect();
        let uniform = compute_lomb_scargle(&t, &y, &freqs);
        let mut weights = vec![1.0_f64; n];
        weights[42] = 1.0e-6;
        let downweighted = compute_lomb_scargle_weighted(&t, &y, &weights, &freqs);

        let (f_uniform, _) = uniform.max_peak();
        let (f_downweighted, _) = downweighted.max_peak();
        let err_uniform = (f_uniform - freq_true).abs();
        let err_downweighted = (f_downweighted - freq_true).abs();
        assert!(
            err_downweighted <= err_uniform,
            "downweighted err {} should be <= uniform err {} (f_dw={}, f_uni={})",
            err_downweighted,
            err_uniform,
            f_downweighted,
            f_uniform,
        );
    }

    #[test]
    fn weighted_zero_weight_sample_is_ignored() {
        // Marking a single sample with weight zero must produce the same
        // periodogram as omitting that sample entirely.
        let n = 32;
        let freq_true = 0.21;
        let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * freq_true * ti).cos())
            .collect();
        let freqs: Vec<f64> = (1..200).map(|i| i as f64 / 1000.0).collect();

        // Mask sample 7.
        let mut weights = vec![1.0_f64; n];
        weights[7] = 0.0;
        let masked = compute_lomb_scargle_weighted(&t, &y, &weights, &freqs);

        // Drop sample 7 from the input.
        let t_dropped: Vec<f64> = t
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if i == 7 { None } else { Some(v) })
            .collect();
        let y_dropped: Vec<f64> = y
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if i == 7 { None } else { Some(v) })
            .collect();
        let dropped = compute_lomb_scargle(&t_dropped, &y_dropped, &freqs);

        for (a, b) in masked.power.iter().zip(dropped.power.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "masked={} dropped={} diff={}",
                a,
                b,
                (a - b).abs()
            );
        }
    }
}
