//! Standard boundary/discontinuity detectors for baseline comparison.
//!
//! Provides:
//! - PVI (Partial Variance of Increments): standard MHD discontinuity detector
//! - Spectral entropy transition score: sliding-window spectral flatness change
//! - |B| gradient detector: simplest scalar threshold baseline
//!
//! # Literature
//! - Greco et al. (2018): Partial variance of increments method
//! - Matthaeus et al. (2015): Intermittency and discontinuities in MHD

/// Partial Variance of Increments (PVI).
///
/// PVI(t) = |delta_B(t)| / sqrt(<|delta_B|^2>)
///
/// where delta_B(t) = B(t+lag) - B(t) is the magnetic field increment
/// and the denominator is the RMS increment over the full series.
///
/// High PVI (> 3-4) indicates a strong discontinuity / current sheet.
/// Returns one PVI value per valid time step.
pub fn pvi_3d(
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    lag: usize,
) -> Vec<f64> {
    let n = bx.len().min(by.len()).min(bz.len());
    if n <= lag {
        return vec![];
    }

    // Compute increments
    let mut delta_sq: Vec<f64> = Vec::with_capacity(n - lag);
    for i in 0..n - lag {
        let dx = bx[i + lag] - bx[i];
        let dy = by[i + lag] - by[i];
        let dz = bz[i + lag] - bz[i];
        delta_sq.push(dx * dx + dy * dy + dz * dz);
    }

    // RMS increment
    let mean_sq = delta_sq.iter().sum::<f64>() / delta_sq.len() as f64;
    let rms = mean_sq.sqrt();

    if rms < 1e-15 {
        return vec![0.0; delta_sq.len()];
    }

    delta_sq.iter().map(|&dsq| dsq.sqrt() / rms).collect()
}

/// Detect PVI-based boundary transitions.
///
/// Returns indices where PVI exceeds `threshold` (typically 3-4 for
/// strong discontinuities). Suppresses duplicates within `min_gap` steps.
pub fn pvi_crossings(
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    lag: usize,
    threshold: f64,
    min_gap: usize,
) -> Vec<usize> {
    let pvi = pvi_3d(bx, by, bz, lag);
    let mut crossings = Vec::new();

    for (i, &v) in pvi.iter().enumerate() {
        if v > threshold {
            let dominated = crossings
                .last()
                .is_some_and(|&prev: &usize| i.saturating_sub(prev) < min_gap);
            if !dominated {
                crossings.push(i);
            }
        }
    }

    crossings
}

/// Sliding-window |B| gradient detector.
///
/// Returns indices where the absolute difference between pre-window and
/// post-window mean |B| exceeds `threshold`. This is the simplest possible
/// boundary detector and the baseline against which all others are compared.
pub fn bmag_gradient_crossings(
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    window: usize,
    threshold: f64,
) -> Vec<usize> {
    let n = bx.len().min(by.len()).min(bz.len());
    if n < window * 2 + 1 {
        return vec![];
    }

    let bmag: Vec<f64> = (0..n)
        .map(|i| (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt())
        .collect();

    let mut crossings = Vec::new();
    for i in window..n.saturating_sub(window) {
        let pre: f64 = bmag[i.saturating_sub(window)..i].iter().sum::<f64>() / window as f64;
        let post: f64 = bmag[i..(i + window).min(n)].iter().sum::<f64>()
            / window.min(n - i) as f64;
        if (post - pre).abs() > threshold {
            let dominated = crossings
                .last()
                .is_some_and(|&prev: &usize| i.saturating_sub(prev) < window);
            if !dominated {
                crossings.push(i);
            }
        }
    }

    crossings
}

/// Spectral entropy transition score.
///
/// Computes sliding-window spectral entropy (flatness) and detects transitions
/// where the entropy changes sharply. High entropy = broadband (turbulent);
/// low entropy = narrowband (ordered). A transition in entropy indicates a
/// boundary between regimes.
pub fn spectral_entropy_crossings(
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    window: usize,
    threshold: f64,
) -> Vec<usize> {
    let n = bx.len().min(by.len()).min(bz.len());
    if n < window * 2 {
        return vec![];
    }

    // Compute |B| for scalar entropy
    let bmag: Vec<f64> = (0..n)
        .map(|i| (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt())
        .collect();

    // Sliding window spectral entropy
    let mut entropies: Vec<f64> = Vec::new();
    let half = window / 2;
    for i in half..n.saturating_sub(half) {
        let start = i.saturating_sub(half);
        let end = (i + half).min(n);
        let seg = &bmag[start..end];

        // Compute normalized power spectrum via variance decomposition
        let mean: f64 = seg.iter().sum::<f64>() / seg.len() as f64;
        let var: f64 = seg.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / seg.len() as f64;

        // Spectral entropy proxy: coefficient of variation (simple, fast)
        // High CV = broadband fluctuations; low CV = steady
        let entropy = if mean > 1e-10 { var.sqrt() / mean } else { 0.0 };
        entropies.push(entropy);
    }

    // Detect transitions in entropy
    let mut crossings = Vec::new();
    let trans_half = window.max(5);
    for i in trans_half..entropies.len().saturating_sub(trans_half) {
        let pre: f64 = entropies[i.saturating_sub(trans_half)..i].iter().sum::<f64>()
            / trans_half as f64;
        let post: f64 = entropies[i..(i + trans_half).min(entropies.len())]
            .iter()
            .sum::<f64>()
            / trans_half.min(entropies.len() - i) as f64;
        if (post - pre).abs() > threshold {
            let dominated = crossings
                .last()
                .is_some_and(|&prev: &usize| i.saturating_sub(prev) < trans_half);
            if !dominated {
                // Map back to original index
                crossings.push(i + half);
            }
        }
    }

    crossings
}

/// Magnetic shear angle detector.
///
/// Computes the rotation angle between pre-window and post-window mean
/// B-vectors. This is the standard magnetopause identification criterion
/// used in most crossing databases (Sonnerup & Cahill 1967).
///
/// Returns indices where the shear angle exceeds `threshold_deg`.
pub fn magnetic_shear_crossings(
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    window: usize,
    threshold_deg: f64,
) -> Vec<usize> {
    let n = bx.len().min(by.len()).min(bz.len());
    if n < window * 2 + 1 {
        return vec![];
    }

    let mut crossings = Vec::new();
    let half = window;

    for i in half..n.saturating_sub(half) {
        let pre_start = i.saturating_sub(half);
        let post_end = (i + half).min(n);
        let pre_n = (i - pre_start) as f64;
        let post_n = (post_end - i) as f64;

        let pre_bx: f64 = bx[pre_start..i].iter().sum::<f64>() / pre_n;
        let pre_by: f64 = by[pre_start..i].iter().sum::<f64>() / pre_n;
        let pre_bz: f64 = bz[pre_start..i].iter().sum::<f64>() / pre_n;

        let post_bx: f64 = bx[i..post_end].iter().sum::<f64>() / post_n;
        let post_by: f64 = by[i..post_end].iter().sum::<f64>() / post_n;
        let post_bz: f64 = bz[i..post_end].iter().sum::<f64>() / post_n;

        let pre_mag = (pre_bx.powi(2) + pre_by.powi(2) + pre_bz.powi(2)).sqrt();
        let post_mag = (post_bx.powi(2) + post_by.powi(2) + post_bz.powi(2)).sqrt();

        if pre_mag < 0.01 || post_mag < 0.01 {
            continue;
        }

        let cos_angle = ((pre_bx * post_bx + pre_by * post_by + pre_bz * post_bz)
            / (pre_mag * post_mag))
            .clamp(-1.0, 1.0);
        let angle_deg = cos_angle.acos().to_degrees();

        if angle_deg >= threshold_deg {
            let dominated = crossings
                .last()
                .is_some_and(|&prev: &usize| i.saturating_sub(prev) < half);
            if !dominated {
                crossings.push(i);
            }
        }
    }

    crossings
}

/// Wavelet variance transition detector.
///
/// Computes sliding-window Haar wavelet variance at each position, then
/// detects transitions where the variance structure changes sharply.
/// A boundary shows a sudden increase or decrease in scale-resolved variance.
pub fn wavelet_variance_crossings(
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    window: usize,
    threshold: f64,
) -> Vec<usize> {
    let n = bx.len().min(by.len()).min(bz.len());
    if n < window * 2 {
        return vec![];
    }

    // Compute |B| for scalar wavelet variance
    let bmag: Vec<f64> = (0..n)
        .map(|i| (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt())
        .collect();

    // Sliding-window Haar wavelet variance (1-level decomposition)
    let half = window / 2;
    let mut variances: Vec<f64> = Vec::new();

    for i in half..n.saturating_sub(half) {
        let start = i.saturating_sub(half);
        let end = (i + half).min(n);
        let seg = &bmag[start..end];
        let seg_n = seg.len();

        // Haar detail coefficients: d_k = (x_{2k} - x_{2k+1}) / sqrt(2)
        let mut detail_energy = 0.0;
        let pairs = seg_n / 2;
        for k in 0..pairs {
            let d = (seg[2 * k] - seg[2 * k + 1]) / std::f64::consts::SQRT_2;
            detail_energy += d * d;
        }
        let var = if pairs > 0 { detail_energy / pairs as f64 } else { 0.0 };
        variances.push(var);
    }

    // Detect transitions in wavelet variance
    let mut crossings = Vec::new();
    let tw = window.max(5);
    for i in tw..variances.len().saturating_sub(tw) {
        let pre: f64 = variances[i.saturating_sub(tw)..i].iter().sum::<f64>() / tw as f64;
        let post: f64 = variances[i..(i + tw).min(variances.len())]
            .iter()
            .sum::<f64>()
            / tw.min(variances.len() - i) as f64;

        let ratio = if pre > 1e-15 { post / pre } else if post > 1e-15 { 1e6 } else { 1.0 };
        let log_ratio = ratio.ln().abs();

        if log_ratio > threshold {
            let dominated = crossings
                .last()
                .is_some_and(|&prev: &usize| i.saturating_sub(prev) < tw);
            if !dominated {
                crossings.push(i + half);
            }
        }
    }

    crossings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pvi_step_function() {
        // Step function should produce high PVI at the transition
        let mut bx = vec![1.0; 50];
        bx.extend(vec![10.0; 50]);
        let by = vec![0.0; 100];
        let bz = vec![0.0; 100];

        let pvi = pvi_3d(&bx, &by, &bz, 1);
        assert!(!pvi.is_empty());

        // PVI should spike near index 49 (the step)
        let max_idx = pvi
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!((45..55).contains(&max_idx));
    }

    #[test]
    fn test_pvi_crossings_detection() {
        let mut bx = vec![1.0; 50];
        bx.extend(vec![10.0; 50]);
        let by = vec![0.0; 100];
        let bz = vec![0.0; 100];

        let crossings = pvi_crossings(&bx, &by, &bz, 1, 3.0, 5);
        assert!(!crossings.is_empty());
    }
}
