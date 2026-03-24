//! Orthonormal Haar discrete wavelet transform and wavelet bridge reservoir.
//!
//! Provides:
//! - `haar_dwt`: forward orthonormal Haar DWT (energy-preserving)
//! - `haar_idwt`: inverse of `haar_dwt`
//! - `hard_threshold`: hard-threshold sparsification
//! - `concurrency`: count of surviving coefficients after threshold
//! - `WaveletBridge`: negative-dimensional reservoir accumulator

use std::f64::consts::SQRT_2;

/// Orthonormal Haar DWT in-place butterfly.
///
/// Transforms `signal` of length N (must be a power of 2) into wavelet
/// coefficients in the standard layout:
///   [scaling_coeff | detail_coarsest | ... | detail_finest]
///
/// Energy is preserved: `||output||_2 == ||input||_2`.
pub fn haar_dwt(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(
        n.is_power_of_two(),
        "haar_dwt: length must be a power of 2, got {}",
        n
    );
    let mut buf = signal.to_vec();
    let inv_sqrt2 = 1.0 / SQRT_2;
    let mut len = n;
    while len > 1 {
        let half = len / 2;
        let mut tmp = vec![0.0_f64; len];
        for i in 0..half {
            let a = buf[2 * i];
            let b = buf[2 * i + 1];
            tmp[i] = (a + b) * inv_sqrt2;
            tmp[half + i] = (a - b) * inv_sqrt2;
        }
        buf[..len].copy_from_slice(&tmp);
        len = half;
    }
    buf
}

/// Inverse orthonormal Haar DWT.
///
/// Recovers the original signal from coefficients produced by `haar_dwt`.
pub fn haar_idwt(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    assert!(
        n.is_power_of_two(),
        "haar_idwt: length must be a power of 2, got {}",
        n
    );
    let mut buf = coeffs.to_vec();
    let inv_sqrt2 = 1.0 / SQRT_2;
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let mut tmp = vec![0.0_f64; len];
        for i in 0..half {
            let s = buf[i];
            let d = buf[half + i];
            tmp[2 * i] = (s + d) * inv_sqrt2;
            tmp[2 * i + 1] = (s - d) * inv_sqrt2;
        }
        buf[..len].copy_from_slice(&tmp);
        len *= 2;
    }
    buf
}

/// Hard threshold: zero out coefficients where `|c_i| < eps * ||c||_inf`.
///
/// Returns a new `Vec`; never mutates `coeffs`.
/// With `eps = 0.0`: no coefficients are zeroed.
/// With `eps = 1.0`: all coefficients are zeroed (||c||_inf * 1.0 >= every |c_i|).
pub fn hard_threshold(coeffs: &[f64], eps: f64) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![];
    }
    let c_inf = coeffs.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if c_inf == 0.0 {
        return vec![0.0; coeffs.len()];
    }
    let cutoff = eps * c_inf;
    coeffs
        .iter()
        .map(|&c| if c.abs() <= cutoff { 0.0 } else { c })
        .collect()
}

/// Count of surviving (nonzero) coefficients after `hard_threshold`.
pub fn concurrency(coeffs: &[f64], eps: f64) -> usize {
    let thresholded = hard_threshold(coeffs, eps);
    thresholded.iter().filter(|&&x| x != 0.0).count()
}

/// Negative-dimensional reservoir for wavelet bridge energy accumulation.
///
/// Accumulates sub-threshold wavelet residuals and reinjects them into the
/// active coefficient vector with exponential decay weight `alpha`.
/// This implements a discrete Mori-Zwanzig memory closure (see I-070).
pub struct WaveletBridge {
    buffer: Vec<f64>,
}

impl WaveletBridge {
    /// Create a new bridge reservoir of length `n`, initialized to zero.
    pub fn new(n: usize) -> Self {
        Self {
            buffer: vec![0.0; n],
        }
    }

    /// Accumulate a residual vector into the reservoir (`buffer += residual`).
    pub fn accumulate(&mut self, residual: &[f64]) {
        assert_eq!(
            self.buffer.len(),
            residual.len(),
            "accumulate: residual length {} != buffer length {}",
            residual.len(),
            self.buffer.len()
        );
        for (b, &r) in self.buffer.iter_mut().zip(residual.iter()) {
            *b += r;
        }
    }

    /// Reinject `alpha * buffer` into `coeffs`, then decay buffer by `(1 - alpha)`.
    ///
    /// `coeffs += alpha * buffer; buffer *= (1 - alpha)`
    pub fn reinject(&mut self, coeffs: &mut [f64], alpha: f64) {
        assert_eq!(
            self.buffer.len(),
            coeffs.len(),
            "reinject: coeffs length {} != buffer length {}",
            coeffs.len(),
            self.buffer.len()
        );
        let decay = 1.0 - alpha;
        for (c, b) in coeffs.iter_mut().zip(self.buffer.iter_mut()) {
            *c += alpha * (*b);
            *b *= decay;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.7 + 1.3).sin()).collect()
    }

    #[test]
    fn test_haar_roundtrip() {
        for &n in &[2usize, 4, 8, 16, 64, 512] {
            let u = make_signal(n);
            let recovered = haar_idwt(&haar_dwt(&u));
            for (&orig, &rec) in u.iter().zip(recovered.iter()) {
                assert!(
                    (orig - rec).abs() < 1e-12,
                    "round-trip failed at n={n}: orig={orig} rec={rec}"
                );
            }
        }
    }

    #[test]
    fn test_haar_energy_preserving() {
        for &n in &[4usize, 8, 32, 256] {
            let u = make_signal(n);
            let c = haar_dwt(&u);
            let norm_u: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_c: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert_relative_eq!(norm_u, norm_c, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hard_threshold_extremes() {
        let c = vec![1.0, -0.5, 0.25, -0.1];
        // eps=1.0: cutoff = 1.0 * 1.0 = 1.0; all |c_i| < 1.0 so all zeroed
        let zeroed = hard_threshold(&c, 1.0);
        for &v in &zeroed {
            assert_eq!(v, 0.0, "eps=1.0 should zero all coefficients");
        }
        // eps=0.0: cutoff = 0.0; no coefficient < 0.0 so none zeroed
        let none_zeroed = hard_threshold(&c, 0.0);
        for (&orig, &kept) in c.iter().zip(none_zeroed.iter()) {
            assert_relative_eq!(orig, kept, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_bridge_accumulate_reinject() {
        // alpha=1.0: reinject transfers entire buffer to coeffs
        let n = 8;
        let residual: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut bridge = WaveletBridge::new(n);
        bridge.accumulate(&residual);

        let mut coeffs = vec![0.0_f64; n];
        bridge.reinject(&mut coeffs, 1.0);

        for (i, (&c, &r)) in coeffs.iter().zip(residual.iter()).enumerate() {
            assert!(
                (c - r).abs() < 1e-12,
                "reinject with alpha=1.0 failed at index {i}: c={c} r={r}"
            );
        }
        // buffer should now be zero (decayed by 1-1=0)
        for &b in &bridge.buffer {
            assert_relative_eq!(b, 0.0, epsilon = 1e-15);
        }
    }
}

/// Fractal kernel using wavelet transforms for multi-scale analysis.
pub struct WaveletFractalKernel {
    /// Fractal scaling parameter (typically alpha ~ 0.618)
    pub alpha: f64,
    /// Number of decomposition levels
    pub levels: usize,
}

impl WaveletFractalKernel {
    /// Create a new fractal kernel.
    pub fn new(levels: usize, alpha: f64) -> Self {
        Self { levels, alpha }
    }

    /// Perform multi-scale decomposition and apply fractal transformation.
    ///
    /// Currently uses Haar DWT as the base transform.
    pub fn multi_scale_transform(&self, signal: &[f64]) -> Vec<f64> {
        let mut coeffs = haar_dwt(signal);
        let n = coeffs.len();

        // Apply fractal map at each scale: detail_coeff = scale * tanh(coeff / scale)
        // Coeffs layout: [scaling | detail_level_0 | detail_level_1 | ... | detail_level_L]
        // where L = log2(n) - 1.
        let mut current_len = 1;
        let mut level = 0;

        while current_len < n {
            let scale_factor = self.alpha.powi(level);
            for coeff in coeffs.iter_mut().take(current_len * 2).skip(current_len) {
                if level == 0 {
                    *coeff *= scale_factor;
                } else {
                    *coeff = scale_factor * (*coeff / scale_factor).tanh();
                }
            }
            current_len *= 2;
            level += 1;
        }

        haar_idwt(&coeffs)
    }
}

/// Estimate fractal dimension using box counting on a 1D signal.
pub fn compute_fractal_dimension(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let n = signal.len();
    let min = signal.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = signal.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;
    if range == 0.0 {
        return 0.0;
    }

    let normalized: Vec<f64> = signal.iter().map(|&x| (x - min) / range).collect();

    // Box sizes: powers of 2 from 1 to n/4
    let mut box_sizes = Vec::new();
    let mut s = 1;
    while s <= n / 4 && s < 1024 {
        box_sizes.push(s);
        s *= 2;
    }

    if box_sizes.len() < 2 {
        return 1.0; // Minimal dimension
    }

    let mut log_sizes = Vec::new();
    let mut log_counts = Vec::new();

    for &size in &box_sizes {
        let n_boxes = n / size;
        let mut count = 0.0;
        for i in 0..n_boxes {
            let start = i * size;
            let end = (i + 1) * size;
            let segment = &normalized[start..end];
            let s_min = segment.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let s_max = segment.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let s_range = s_max - s_min;
            count += (s_range * size as f64).ceil().max(1.0);
        }
        log_sizes.push((size as f64).ln());
        log_counts.push(count.ln());
    }

    // Linear regression: log(count) = -D * log(size) + C
    // D = (N * sum(xy) - sum(x)*sum(y)) / (N * sum(x^2) - (sum(x))^2)
    let m = log_sizes.len() as f64;
    let sum_x: f64 = log_sizes.iter().sum();
    let sum_y: f64 = log_counts.iter().sum();
    let sum_xx: f64 = log_sizes.iter().map(|x| x * x).sum();
    let sum_xy: f64 = log_sizes
        .iter()
        .zip(log_counts.iter())
        .map(|(x, y)| x * y)
        .sum();

    let denom = m * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return 1.0;
    }

    let slope = (m * sum_xy - sum_x * sum_y) / denom;
    -slope
}

/// Ricker wavelet (Mexican Hat).
///
/// Returns the values of the Ricker wavelet at `points` positions for scale `a`.
/// A = 2 / (sqrt(3 * a) * pi^0.25)
/// w(x) = A * (1 - (x/a)^2) * exp(-x^2 / (2*a^2))
pub fn ricker_wavelet(points: usize, a: f64) -> Vec<f64> {
    if points == 0 {
        return vec![];
    }
    let n_f = points as f64;
    let a2 = a * a;
    let pi_pow_025 = std::f64::consts::PI.powf(0.25);
    let norm = 2.0 / ((3.0 * a).sqrt() * pi_pow_025);

    (0..points)
        .map(|i| {
            let x = i as f64 - (n_f - 1.0) / 2.0;
            let x2 = x * x;
            let x_a_2 = x2 / a2;
            norm * (1.0 - x_a_2) * (-x2 / (2.0 * a2)).exp()
        })
        .collect()
}

/// Simple Continuous Wavelet Transform (CWT) using Ricker wavelet.
///
/// Computes the CWT for a signal at given widths (scales).
/// Result is a matrix of size `widths.len() x signal.len()`.
pub fn continuous_wavelet_transform(signal: &[f64], widths: &[f64]) -> Vec<Vec<f64>> {
    let n = signal.len();
    let mut result = Vec::with_capacity(widths.len());

    for &a in widths {
        // Kernel length: typically 10 * width
        let points = ((10.0 * a).ceil() as usize).min(n).max(3);
        let kernel = ricker_wavelet(points, a);

        // Convolution (mode='same')
        let mut conv = vec![0.0; n];
        let half_k = points / 2;

        for (i, conv_value) in conv.iter_mut().enumerate().take(n) {
            let mut sum = 0.0;
            for (j, kernel_value) in kernel.iter().enumerate().take(points) {
                let sig_idx = i as i64 + j as i64 - half_k as i64;
                if sig_idx >= 0 && sig_idx < n as i64 {
                    sum += signal[sig_idx as usize] * kernel_value;
                }
            }
            *conv_value = sum;
        }
        result.push(conv);
    }
    result
}

#[cfg(test)]
mod cwt_tests {
    use super::*;

    #[test]
    fn test_ricker_wavelet_symmetry() {
        let w = ricker_wavelet(101, 5.0);
        for i in 0..50 {
            assert!((w[i] - w[100 - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cwt_dimensions() {
        let signal = vec![0.0; 100];
        let widths = vec![1.0, 2.0, 5.0];
        let result = continuous_wavelet_transform(&signal, &widths);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 100);
    }
}
