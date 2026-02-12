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
    Linear,
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

/// Classify which latency law best explains the data.
pub fn classify_latency_law(samples: &[(f64, f64)]) -> LatencyLaw {
    if samples.len() < 3 {
        return LatencyLaw::Undetermined;
    }
    let r2_inv = inverse_square_r2(samples);
    let r2_lin = linear_r2(samples);

    let mean = samples.iter().map(|(_, y)| *y).sum::<f64>() / samples.len() as f64;
    let var = samples.iter().map(|(_, y)| (y - mean).powi(2)).sum::<f64>() / samples.len() as f64;

    if var < 1e-10 {
        return LatencyLaw::Uniform;
    }
    if r2_inv > 0.85 && r2_inv >= r2_lin {
        return LatencyLaw::InverseSquare;
    }
    if r2_lin > 0.85 && r2_lin > r2_inv {
        return LatencyLaw::Linear;
    }
    LatencyLaw::Undetermined
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
}
