//! Radial survival spectrum and latency-law classification.

/// One radial bin in the collision/latency spectrum.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumBin {
    pub radius: f64,
    pub mean_latency: f64,
    pub n_samples: usize,
}

/// Fit classification for latency curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyLaw {
    InverseSquare,
    /// General power law: latency ~ r^gamma (gamma is fitted exponent)
    PowerLaw,
    Linear,
    Exponential,
    Uniform,
    Undetermined,
}

/// Build radial bins from `(radius, latency)` observations.
pub fn radial_bins(samples: &[(f64, f64)], n_bins: usize) -> Vec<SpectrumBin> {
    if samples.is_empty() || n_bins == 0 {
        return Vec::new();
    }
    let r_max = samples.iter().map(|(r, _)| *r).fold(0.0_f64, f64::max);
    let dr = (r_max / n_bins as f64).max(1e-9);

    let mut sums = vec![0.0; n_bins];
    let mut counts = vec![0usize; n_bins];
    let mut radii = vec![0.0; n_bins];

    for &(r, l) in samples {
        let mut b = (r / dr).floor() as usize;
        if b >= n_bins {
            b = n_bins - 1;
        }
        sums[b] += l;
        counts[b] += 1;
        radii[b] += r;
    }

    let mut out = Vec::new();
    for i in 0..n_bins {
        if counts[i] > 0 {
            out.push(SpectrumBin {
                radius: radii[i] / counts[i] as f64,
                mean_latency: sums[i] / counts[i] as f64,
                n_samples: counts[i],
            });
        }
    }
    out
}

/// R^2 fit for model `latency ~ a / r^2 + b`.
pub fn inverse_square_r2(samples: &[(f64, f64)]) -> f64 {
    if samples.len() < 3 {
        return 0.0;
    }
    let x = samples
        .iter()
        .map(|(r, _)| 1.0 / (r * r).max(1e-12))
        .collect::<Vec<_>>();
    let y = samples.iter().map(|(_, l)| *l).collect::<Vec<_>>();
    linear_r2_xy(&x, &y)
}

/// R^2 fit for model `latency ~ a * r + b`.
pub fn linear_r2(samples: &[(f64, f64)]) -> f64 {
    if samples.len() < 3 {
        return 0.0;
    }
    let x = samples.iter().map(|(r, _)| *r).collect::<Vec<_>>();
    let y = samples.iter().map(|(_, l)| *l).collect::<Vec<_>>();
    linear_r2_xy(&x, &y)
}

/// R^2 fit for general power law `latency ~ r^gamma` via log-log regression.
///
/// Returns `(r2, gamma)` where gamma is the fitted exponent.
/// Filters to samples with r > 0 and latency > 0 for log transform.
pub fn power_law_r2(samples: &[(f64, f64)]) -> (f64, f64) {
    let valid: Vec<(f64, f64)> = samples
        .iter()
        .filter(|&&(r, l)| r > 1e-12 && l > 1e-12)
        .map(|&(r, l)| (r.ln(), l.ln()))
        .collect();
    if valid.len() < 3 {
        return (0.0, 0.0);
    }
    let x: Vec<f64> = valid.iter().map(|(lnr, _)| *lnr).collect();
    let y: Vec<f64> = valid.iter().map(|(_, lnl)| *lnl).collect();
    let r2 = linear_r2_xy(&x, &y);

    // Recover slope (= gamma exponent)
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - x_mean;
        sxx += dx * dx;
        sxy += dx * (y[i] - y_mean);
    }
    let gamma = if sxx > 1e-12 { sxy / sxx } else { 0.0 };

    (r2, gamma)
}

/// R^2 fit for exponential model `latency ~ exp(a * r + b)`.
///
/// Fits via `ln(latency) = a * r + b`.
pub fn exponential_r2(samples: &[(f64, f64)]) -> f64 {
    let valid: Vec<(f64, f64)> = samples
        .iter()
        .filter(|&&(_, l)| l > 1e-12)
        .map(|&(r, l)| (r, l.ln()))
        .collect();
    if valid.len() < 3 {
        return 0.0;
    }
    let x: Vec<f64> = valid.iter().map(|(r, _)| *r).collect();
    let y: Vec<f64> = valid.iter().map(|(_, lnl)| *lnl).collect();
    linear_r2_xy(&x, &y)
}

/// Classify which latency law best explains the data.
///
/// Compares five models: inverse-square, general power-law, linear,
/// exponential, and uniform. Picks the best fit by R^2, with preference
/// for inverse-square when its R^2 is close to the power-law R^2
/// (since inverse-square is a special case of power-law with gamma=-2).
pub fn classify_latency_law(samples: &[(f64, f64)]) -> LatencyLaw {
    if samples.len() < 3 {
        return LatencyLaw::Undetermined;
    }

    let mean = samples.iter().map(|(_, y)| *y).sum::<f64>() / samples.len() as f64;
    let var = samples.iter().map(|(_, y)| (y - mean).powi(2)).sum::<f64>() / samples.len() as f64;

    if var < 1e-10 {
        return LatencyLaw::Uniform;
    }

    let r2_inv = inverse_square_r2(samples);
    let r2_lin = linear_r2(samples);
    let (r2_pow, gamma) = power_law_r2(samples);
    let r2_exp = exponential_r2(samples);

    // Collect candidates with their R^2 values
    let threshold = 0.85;

    // If inverse-square is strong AND power-law gamma is near -2, prefer inverse-square
    if r2_inv > threshold && (gamma + 2.0).abs() < 0.5 {
        return LatencyLaw::InverseSquare;
    }

    // Find best R^2
    let candidates = [
        (r2_inv, LatencyLaw::InverseSquare),
        (r2_pow, LatencyLaw::PowerLaw),
        (r2_lin, LatencyLaw::Linear),
        (r2_exp, LatencyLaw::Exponential),
    ];

    let best = candidates
        .iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(&(r2, law)) = best {
        if r2 > threshold {
            return law;
        }
    }

    LatencyLaw::Undetermined
}

/// Detailed latency law classification with fitted parameters.
///
/// Companion to `LatencyLaw` that preserves the fitted exponent and R^2
/// for each model. Created to address STPT-009 refutation: the simple
/// inverse-square classification was insufficient, and we need the actual
/// power-law exponent to characterize the latency behavior.
#[derive(Debug, Clone, Copy)]
pub struct LatencyLawDetail {
    /// Best-fit classification
    pub law: LatencyLaw,
    /// R^2 for inverse-square model
    pub r2_inverse_square: f64,
    /// R^2 for general power law
    pub r2_power_law: f64,
    /// Fitted power-law exponent (gamma in latency ~ r^gamma)
    pub power_law_exponent: f64,
    /// R^2 for linear model
    pub r2_linear: f64,
    /// R^2 for exponential model
    pub r2_exponential: f64,
}

/// Classify latency law with full fitted parameters.
///
/// Returns both the classification and all R^2 values plus the fitted
/// power-law exponent. Use this when you need the actual exponent
/// (e.g. to test whether gamma = -2 for inverse-square hypothesis).
pub fn classify_latency_law_detailed(samples: &[(f64, f64)]) -> LatencyLawDetail {
    if samples.len() < 3 {
        return LatencyLawDetail {
            law: LatencyLaw::Undetermined,
            r2_inverse_square: 0.0,
            r2_power_law: 0.0,
            power_law_exponent: 0.0,
            r2_linear: 0.0,
            r2_exponential: 0.0,
        };
    }

    let r2_inv = inverse_square_r2(samples);
    let r2_lin = linear_r2(samples);
    let (r2_pow, gamma) = power_law_r2(samples);
    let r2_exp = exponential_r2(samples);
    let law = classify_latency_law(samples);

    LatencyLawDetail {
        law,
        r2_inverse_square: r2_inv,
        r2_power_law: r2_pow,
        power_law_exponent: gamma,
        r2_linear: r2_lin,
        r2_exponential: r2_exp,
    }
}

fn linear_r2_xy(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 3 {
        return 0.0;
    }
    let n_f = n as f64;
    let x_mean = x.iter().take(n).sum::<f64>() / n_f;
    let y_mean = y.iter().take(n).sum::<f64>() / n_f;

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx <= 1e-12 || syy <= 1e-12 {
        return 0.0;
    }
    let slope = sxy / sxx;
    let intercept = y_mean - slope * x_mean;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let y_hat = slope * x[i] + intercept;
        ss_res += (y[i] - y_hat).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }
    if ss_tot <= 1e-12 {
        0.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_square_classification() {
        let samples = (1..60)
            .map(|r| {
                let rf = r as f64;
                (rf, 2.0 / (rf * rf) + 0.1)
            })
            .collect::<Vec<_>>();
        assert_eq!(classify_latency_law(&samples), LatencyLaw::InverseSquare);
    }

    #[test]
    fn test_uniform_classification() {
        let samples = (1..30).map(|r| (r as f64, 2.5)).collect::<Vec<_>>();
        assert_eq!(classify_latency_law(&samples), LatencyLaw::Uniform);
    }

    #[test]
    fn test_power_law_classification() {
        // Pure power law y = r^(-1.5) without offset (offset breaks log-log linearity)
        let samples = (1..60)
            .map(|r| {
                let rf = r as f64;
                (rf, rf.powf(-1.5))
            })
            .collect::<Vec<_>>();
        let (r2, gamma) = power_law_r2(&samples);
        assert!(r2 > 0.99, "Power law R^2 should be high: {}", r2);
        assert!(
            (gamma + 1.5).abs() < 0.05,
            "Gamma should be near -1.5: {}",
            gamma
        );
    }

    #[test]
    fn test_exponential_classification() {
        // y = exp(-0.1 * r)
        let samples = (1..50)
            .map(|r| {
                let rf = r as f64;
                (rf, (-0.1 * rf).exp())
            })
            .collect::<Vec<_>>();
        let r2 = exponential_r2(&samples);
        assert!(r2 > 0.95, "Exponential R^2 should be high: {}", r2);
    }

    #[test]
    fn test_linear_classification() {
        let samples = (1..40)
            .map(|r| {
                let rf = r as f64;
                (rf, 3.0 * rf + 2.0)
            })
            .collect::<Vec<_>>();
        assert_eq!(classify_latency_law(&samples), LatencyLaw::Linear);
    }

    #[test]
    fn test_inverse_square_vs_power_law_distinction() {
        // True inverse-square: gamma should be near -2
        let samples = (1..60)
            .map(|r| {
                let rf = r as f64;
                (rf, 100.0 / (rf * rf) + 0.5)
            })
            .collect::<Vec<_>>();
        let law = classify_latency_law(&samples);
        assert_eq!(law, LatencyLaw::InverseSquare);
    }

    #[test]
    fn test_detailed_inverse_square() {
        // Pure inverse-square: y = 2/r^2 (no offset for clean log-log fit)
        let samples = (1..60)
            .map(|r| {
                let rf = r as f64;
                (rf, 2.0 / (rf * rf))
            })
            .collect::<Vec<_>>();
        let detail = classify_latency_law_detailed(&samples);
        assert!(detail.r2_inverse_square > 0.95);
        assert!(
            (detail.power_law_exponent + 2.0).abs() < 0.1,
            "Exponent should be near -2: {}",
            detail.power_law_exponent
        );
    }

    #[test]
    fn test_detailed_power_law_exponent() {
        // Pure power law y = r^(-1.5) (no offset)
        let samples = (1..60)
            .map(|r| {
                let rf = r as f64;
                (rf, rf.powf(-1.5))
            })
            .collect::<Vec<_>>();
        let detail = classify_latency_law_detailed(&samples);
        assert!(detail.r2_power_law > 0.99);
        assert!(
            (detail.power_law_exponent + 1.5).abs() < 0.05,
            "Exponent should be near -1.5: {}",
            detail.power_law_exponent
        );
    }

    #[test]
    fn test_detailed_undetermined_short_data() {
        let samples = [(1.0, 2.0), (2.0, 3.0)];
        let detail = classify_latency_law_detailed(&samples);
        assert_eq!(detail.law, LatencyLaw::Undetermined);
        assert_eq!(detail.r2_inverse_square, 0.0);
    }
}
