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
    assert!(n.is_power_of_two(), "haar_dwt: length must be a power of 2, got {}", n);
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
    assert!(n.is_power_of_two(), "haar_idwt: length must be a power of 2, got {}", n);
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
    coeffs.iter().map(|&c| if c.abs() <= cutoff { 0.0 } else { c }).collect()
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
        Self { buffer: vec![0.0; n] }
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
