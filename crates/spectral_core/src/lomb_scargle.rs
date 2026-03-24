//! Lomb-Scargle periodogram for irregularly sampled data.
//!
//! Provides the Generalized Lomb-Scargle periodogram (with floating mean)
//! and analytic false alarm probability (FAP) estimators, including Baluev (2008).
//!
//! # Literature
//! - Lomb (1976): Least-squares frequency analysis of unequally spaced data
//! - Scargle (1982): Studies in astronomical time series analysis
//! - Zechmeister & Kürster (2009): A generalized Lomb-Scargle periodogram
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

/// Compute the Generalized Lomb-Scargle periodogram.
///
/// This version includes a floating mean offset (Zechmeister & Kürster 2009).
///
/// # Arguments
/// * `t` - Observation times
/// * `y` - Observed values
/// * `freqs` - Frequencies to evaluate
pub fn compute_lomb_scargle(t: &[f64], y: &[f64], freqs: &[f64]) -> LombScargleResult {
    let n = t.len();
    assert_eq!(n, y.len(), "t and y must have equal length");

    let w_sum = n as f64; // Assuming unit weights for now
    let y_mean = y.iter().sum::<f64>() / w_sum;
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();
    let yy_sum: f64 = y_centered.iter().map(|&yi| yi * yi).sum();

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
            let yi = y_centered[i];

            s += si;
            c += ci;
            ss += si * si;
            cc += ci * ci;
            sc += si * ci;
            yc += yi * ci;
            ys += yi * si;
        }

        // Floating mean normalization (Zechmeister & Kürster Eq. 5)
        // We solve the 3x3 system for [offset, A, B] where y ~ offset + A*cos + B*sin
        // But centered y_mean=0 and unit weights simplifies it slightly.

        let s_hat = s / w_sum;
        let c_hat = c / w_sum;

        let cc_tilde = cc - c * c_hat;
        let ss_tilde = ss - s * s_hat;
        let sc_tilde = sc - c * s_hat;
        let yc_tilde = yc - y_mean * 0.0; // y_centered already has mean 0
        let ys_tilde = ys - y_mean * 0.0;

        // Determinant of the 2x2 reduced system
        let det = cc_tilde * ss_tilde - sc_tilde * sc_tilde;

        if det.abs() < 1e-15 {
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
}
