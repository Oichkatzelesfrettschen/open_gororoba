//! Thomson Multitaper spectral estimation.
//!
//! Provides:
//! - DPSS (Slepian) tapers via tridiagonal eigenproblem
//! - Multitaper PSD estimation
//! - Thomson F-test for line components

use faer::{Mat, Side};
use rustfft::{FftPlanner, num_complex::Complex64};
use std::f64::consts::PI;

/// Compute Discrete Prolate Spheroidal Sequences (DPSS) tapers.
///
/// Solves the tridiagonal eigenproblem for the Slepian tapers.
///
/// # Arguments
/// * `n` - Signal length
/// * `nw` - Time-bandwidth product
/// * `k` - Number of tapers (typically 2*NW - 1)
pub fn dpss_tapers(n: usize, nw: f64, k: usize) -> Vec<Vec<f64>> {
    if n == 0 || k == 0 {
        return vec![];
    }
    let k = k.min(n);
    let w = nw / n as f64;
    let cos_2pi_w = (2.0 * PI * w).cos();

    // Construct the symmetric tridiagonal matrix L (Slepian 1978)
    // L[i, i] = ((N-1-2i)/2)^2 * cos(2pi*W)
    // L[i, i+1] = 0.5 * (i+1)*(N-1-i)
    
    let mut diag = Vec::with_capacity(n);
    let mut off_diag = Vec::with_capacity(n - 1);
    
    for i in 0..n {
        let term = (n as f64 - 1.0 - 2.0 * i as f64) / 2.0;
        diag.push(term * term * cos_2pi_w);
        if i < n - 1 {
            off_diag.push(0.5 * (i as f64 + 1.0) * (n as f64 - 1.0 - i as f64));
        }
    }

    // Solve the eigenproblem. Since L is symmetric tridiagonal,
    // we use faer's specialized solvers if available, or general symmetric.
    
    let mut mat = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        mat.write(i, i, diag[i]);
        if i < n - 1 {
            mat.write(i, i + 1, off_diag[i]);
            mat.write(i + 1, i, off_diag[i]);
        }
    }

    let evd = mat.selfadjoint_eigendecomposition(Side::Upper);
    let eigenvalues = evd.s();
    let eigenvectors = evd.u();

    // Sort eigenvalues descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let val_a = eigenvalues.column_vector().read(a);
        let val_b = eigenvalues.column_vector().read(b);
        val_b.partial_cmp(&val_a).unwrap()
    });

    let mut tapers = Vec::with_capacity(k);
    for &idx in indices.iter().take(k) {
        let mut taper = Vec::with_capacity(n);
        for i in 0..n {
            taper.push(eigenvectors.read(i, idx));
        }
        
        // Ensure the first lobe is positive (convention)
        let sum: f64 = taper.iter().sum();
        if sum < 0.0 {
            for val in &mut taper {
                *val = -*val;
            }
        }
        tapers.push(taper);
    }

    tapers
}

/// Result of multitaper spectral analysis.
pub struct MultitaperResult {
    pub frequencies: Vec<f64>,
    pub psd: Vec<f64>,
    /// Eigencoefficients [k][freq_bin]
    pub eigencoefficients: Vec<Vec<Complex64>>,
}

/// Compute Thomson Multitaper PSD estimate.
pub fn multitaper_psd(signal: &[f64], nw: f64, k: usize) -> MultitaperResult {
    let n = signal.len();
    if n == 0 {
        return MultitaperResult {
            frequencies: vec![],
            psd: vec![],
            eigencoefficients: vec![],
        };
    }

    let tapers = dpss_tapers(n, nw, k);
    let k = tapers.len();

    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    
    let mut eigencoeffs = Vec::with_capacity(k);
    let n_freq = n / 2 + 1;

    for taper in &tapers {
        let mut buffer: Vec<Complex64> = centered.iter()
            .zip(taper.iter())
            .map(|(&x, &w)| Complex64::new(x * w, 0.0))
            .collect();
        fft.process(&mut buffer);
        eigencoeffs.push(buffer[..n_freq].to_vec());
    }

    let mut psd = vec![0.0; n_freq];
    for j in 0..n_freq {
        let mut sum_p = 0.0;
        for coeffs in eigencoeffs.iter().take(k) {
            sum_p += coeffs[j].norm_sqr();
        }
        psd[j] = sum_p / (k as f64 * n as f64);
    }

    let frequencies: Vec<f64> = (0..n_freq).map(|i| i as f64 / n as f64).collect();

    MultitaperResult {
        frequencies,
        psd,
        eigencoefficients: eigencoeffs,
    }
}

/// Thomson F-test for line components.
///
/// Returns F-statistic for each frequency bin.
/// d.o.f. are (2, 2*K-2).
pub fn thomson_f_test(eigencoeffs: &[Vec<Complex64>], tapers: &[Vec<f64>]) -> Vec<f64> {
    if eigencoeffs.is_empty() || tapers.is_empty() {
        return vec![];
    }
    let k = eigencoeffs.len();
    let n_freq = eigencoeffs[0].len();
    
    let taper_sums: Vec<f64> = tapers.iter().map(|t| t.iter().sum::<f64>()).collect();
    let denom_norm: f64 = taper_sums.iter().map(|&s| s * s).sum();

    let mut f_stats = vec![0.0; n_freq];

    for j in 0..n_freq {
        if denom_norm < 1e-30 {
            continue;
        }

        let mut numerator_complex = Complex64::new(0.0, 0.0);
        for i in 0..k {
            numerator_complex += eigencoeffs[i][j] * taper_sums[i];
        }

        let mu_hat = numerator_complex / denom_norm;

        let mut residual_power = 0.0;
        for i in 0..k {
            let res = eigencoeffs[i][j] - mu_hat * taper_sums[i];
            residual_power += res.norm_sqr();
        }

        let line_power = mu_hat.norm_sqr() * denom_norm;

        if residual_power < 1e-30 {
            f_stats[j] = 1000.0; // effectively infinity
        } else {
            f_stats[j] = (k as f64 - 1.0) * line_power / residual_power;
        }
    }

    f_stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpss_tapers_orthogonality() {
        let n = 64;
        let tapers = dpss_tapers(n, 4.0, 4);
        assert_eq!(tapers.len(), 4);
        
        // Check orthogonality
        for i in 0..tapers.len() {
            for j in 0..tapers.len() {
                let dot: f64 = tapers[i].iter().zip(tapers[j].iter()).map(|(a, b)| a * b).sum();
                if i == j {
                    assert!((dot - 1.0).abs() < 1e-10);
                } else {
                    assert!(dot.abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_multitaper_sine() {
        let n = 256;
        let freq = 0.2;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64).sin()).collect();
        
        let res = multitaper_psd(&signal, 4.0, 7);
        let mut max_p = -1.0;
        let mut max_f = 0.0;
        for (f, p) in res.frequencies.iter().zip(res.psd.iter()) {
            if *p > max_p {
                max_p = *p;
                max_f = *f;
            }
        }
        assert!((max_f - freq).abs() < 0.01);
    }
}
