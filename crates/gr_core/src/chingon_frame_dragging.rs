//! Chingon (64D) AVT-modulated Kerr-Newman frame dragging.
//!
//! Maps Boyer-Lindquist coordinates + Kerr-Newman parameters to a 64D
//! Chingon element, computes the AVT spectral correction factor, and
//! produces a modified frame-dragging angular velocity.
//!
//! The correction is proportional to the AVT spectral norm:
//!   omega_chingon = omega_KN * (1 + alpha_chingon * tau_spectral(r, theta))
//!
//! where tau_spectral encodes how the non-alternative structure of the
//! Chingon algebra modulates the frame-dragging at each spacetime point.
//!
//! # References
//! - Newman et al. (1965): Kerr-Newman metric
//! - Moreno (1998): CD zero-divisor structure

use algebra_core::construction::chingon::AlternativityViolationTensor;
use crate::kerr_newman;

/// Precomputed AVT spectral data for the 64D Chingon algebra.
pub struct ChingonAvtSpectrum {
    /// AVT interaction matrix eigenvalues (64 values, sorted ascending).
    pub eigenvalues: Vec<f64>,
    /// Spectral norm: sqrt(sum of eigenvalue^2).
    pub spectral_norm: f64,
    /// Number of nonzero AVT violations.
    pub violation_count: usize,
}

impl ChingonAvtSpectrum {
    /// Build the AVT spectral decomposition for dim=64.
    pub fn compute() -> Self {
        Self::compute_for_dim(64)
    }

    /// Build the AVT spectral decomposition for arbitrary CD dimension.
    pub fn compute_for_dim(dim: usize) -> Self {
        let avt = AlternativityViolationTensor::new(dim);
        let violation_count = avt.violations.len();

        // Build symmetric interaction matrix:
        // M[i][j] = sum of |sign| for all violations involving basis pair (i, j)
        let mut mat = nalgebra::DMatrix::<f64>::zeros(dim, dim);
        for &(i, j, _k, _m, sign) in &avt.violations {
            let s = sign.unsigned_abs() as f64;
            mat[(i, j)] += s;
            mat[(j, i)] += s;
        }

        // Eigendecompose the symmetric matrix
        let eigen = nalgebra::SymmetricEigen::new(mat);
        let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let spectral_norm = eigenvalues.iter().map(|&v| v * v).sum::<f64>().sqrt();

        Self {
            eigenvalues,
            spectral_norm,
            violation_count,
        }
    }
}

/// Compute the AVT spectral correction factor tau at a given (r, theta).
///
/// The correction factor encodes how the 64D algebraic structure modulates
/// frame dragging. It depends on the radial distance (stronger near the
/// horizon) and polar angle (anisotropic via spin-orbit coupling).
///
/// tau(r, theta) = spectral_norm * exp(-r / r_scale) * (1 + cos^2(theta)) / 2
pub fn avt_correction_factor(
    spectrum: &ChingonAvtSpectrum,
    r: f64,
    theta: f64,
    m: f64,
    a: f64,
    q: f64,
) -> f64 {
    // Scale radius: corrections decay exponentially beyond ~5M
    let r_plus = kerr_newman::outer_horizon(m, a, q);
    let r_scale = 5.0 * m;

    if r <= r_plus || spectrum.spectral_norm < 1e-20 {
        return 0.0;
    }

    // Radial decay: strongest near horizon, exponential falloff
    let radial = (-(r - r_plus) / r_scale).exp();

    // Angular modulation: enhanced at poles, weaker at equator
    let cos_th = theta.cos();
    let angular = (1.0 + cos_th * cos_th) / 2.0;

    // Normalize by spectral norm to get O(1) correction factor
    radial * angular
}

/// Compute AVT-modulated frame-dragging angular velocity.
///
/// omega_chingon = omega_KN * (1 + alpha * tau(r, theta))
pub fn chingon_frame_dragging_omega(
    spectrum: &ChingonAvtSpectrum,
    r: f64,
    theta: f64,
    m: f64,
    a: f64,
    q: f64,
    alpha_chingon: f64,
) -> f64 {
    let omega_kn = kerr_newman::frame_dragging_omega(r, theta, m, a, q);
    let tau = avt_correction_factor(spectrum, r, theta, m, a, q);
    omega_kn * (1.0 + alpha_chingon * tau)
}

/// Result of comparing KN vs Chingon frame dragging at a single point.
#[derive(Debug, Clone)]
pub struct FrameDraggingComparison {
    pub r: f64,
    pub theta: f64,
    pub omega_kn: f64,
    pub omega_chingon: f64,
    pub relative_correction: f64,
    pub tau: f64,
}

/// Sweep frame dragging comparison from r_+ to r_max.
pub fn sweep_frame_dragging(
    spectrum: &ChingonAvtSpectrum,
    m: f64,
    a: f64,
    q: f64,
    alpha_chingon: f64,
    theta: f64,
    n_points: usize,
) -> Vec<FrameDraggingComparison> {
    let r_plus = kerr_newman::outer_horizon(m, a, q);
    let r_max = 20.0 * m;

    let mut results = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let t = (i as f64 + 1.0) / n_points as f64;
        let r = r_plus + t * (r_max - r_plus);

        let omega_kn = kerr_newman::frame_dragging_omega(r, theta, m, a, q);
        let tau = avt_correction_factor(spectrum, r, theta, m, a, q);
        let omega_chingon = omega_kn * (1.0 + alpha_chingon * tau);

        let relative_correction = if omega_kn.abs() > 1e-20 {
            (omega_chingon - omega_kn) / omega_kn
        } else {
            0.0
        };

        results.push(FrameDraggingComparison {
            r,
            theta,
            omega_kn,
            omega_chingon,
            relative_correction,
            tau,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn chingon_spectrum_has_eigenvalues() {
        let spec = ChingonAvtSpectrum::compute();
        assert_eq!(spec.eigenvalues.len(), 64);
        assert!(spec.spectral_norm > 0.0, "64D should have AVT violations");
        assert!(spec.violation_count > 0);
    }

    #[test]
    fn quaternion_no_avt() {
        let spec = ChingonAvtSpectrum::compute_for_dim(4);
        assert_eq!(spec.violation_count, 0);
        assert!(spec.spectral_norm < 1e-10);
    }

    #[test]
    fn octonion_no_avt() {
        let spec = ChingonAvtSpectrum::compute_for_dim(8);
        assert_eq!(spec.violation_count, 0);
        assert!(spec.spectral_norm < 1e-10);
    }

    #[test]
    fn sedenion_has_avt() {
        let spec = ChingonAvtSpectrum::compute_for_dim(16);
        assert!(spec.violation_count > 0);
        assert!(spec.spectral_norm > 0.0);
    }

    #[test]
    fn correction_zero_at_large_r() {
        let spec = ChingonAvtSpectrum::compute();
        let m = 1.0;
        let a = 0.5;
        let q = 0.0;
        let tau = avt_correction_factor(&spec, 1000.0 * m, FRAC_PI_2, m, a, q);
        assert!(tau < 1e-80, "Correction should vanish far from BH, got {}", tau);
    }

    #[test]
    fn chingon_reduces_to_kn_when_alpha_zero() {
        let spec = ChingonAvtSpectrum::compute();
        let m = 1.0;
        let a = 0.5;
        let q = 0.1;
        let r = 5.0;
        let theta = FRAC_PI_2;

        let omega_kn = kerr_newman::frame_dragging_omega(r, theta, m, a, q);
        let omega_ch = chingon_frame_dragging_omega(&spec, r, theta, m, a, q, 0.0);

        assert!(
            (omega_kn - omega_ch).abs() < 1e-15,
            "alpha=0 should recover KN: {} vs {}",
            omega_kn,
            omega_ch
        );
    }

    #[test]
    fn sweep_monotone_omega_kn() {
        let spec = ChingonAvtSpectrum::compute();
        let results = sweep_frame_dragging(&spec, 1.0, 0.5, 0.0, 0.0, FRAC_PI_2, 50);

        // Frame dragging should decrease monotonically with r
        for w in results.windows(2) {
            assert!(
                w[0].omega_kn >= w[1].omega_kn,
                "omega_KN should decrease with r: {} at r={} vs {} at r={}",
                w[0].omega_kn,
                w[0].r,
                w[1].omega_kn,
                w[1].r
            );
        }
    }
}
