//! Lattice-cutoff Hawking spectrum.
//!
//! Computes thermal spectra with lattice UV cutoff from the D3Q19 LBM
//! dispersion relation.  The key prediction: the LBM lattice spacing
//! acts as a UV cutoff on analog Hawking radiation, bounding the
//! effective temperature by the lattice's kinematic viscosity.
//!
//! The lattice dispersion relation is:
//!   omega_lattice = (2/dx) * sin(k * dx / 2)
//!
//! This gives a maximum frequency omega_max = 2/dx (the Nyquist frequency).
//! Viscous dissipation further cuts off at omega_visc = nu / dx^2.
//!
//! References:
//!   - Unruh (1995): "Sonic analogue of black holes and the effects of
//!     high frequencies on black hole evaporation" PRD 51, 2827
//!   - Jacobson (1991): "Black-hole evaporation and ultrashort distances"
//!     PRD 44, 1731

use std::f64::consts::PI;

/// Ideal Planck/Bose-Einstein distribution for the Hawking spectrum.
///
/// n(omega) = 1 / (exp(hbar * omega / (k_B * T)) - 1)
///
/// In lattice units (hbar = k_B = 1):
/// n(omega) = 1 / (exp(omega / T) - 1)
pub fn ideal_hawking_spectrum(t_h: f64, omega: f64) -> f64 {
    if t_h <= 0.0 || omega <= 0.0 {
        return 0.0;
    }
    let x = omega / t_h;
    if x > 500.0 {
        return 0.0; // avoid overflow
    }
    1.0 / (x.exp() - 1.0)
}

/// Lattice-modified Hawking spectrum with sinc^2 window.
///
/// The lattice dispersion relation omega = (2/dx)*sin(k*dx/2) introduces
/// a sinc^2 modification to the density of states:
///
///   n_lattice(omega) = n_ideal(omega) * sinc^2(omega * dx / (2 * pi))
///
/// where sinc(x) = sin(pi*x)/(pi*x).
pub fn lattice_cutoff_spectrum(t_h: f64, omega: f64, dx: f64) -> f64 {
    let n_ideal = ideal_hawking_spectrum(t_h, omega);

    // sinc^2 window from lattice dispersion
    let arg = omega * dx / (2.0 * PI);
    let sinc = if arg.abs() < 1e-10 {
        1.0
    } else {
        (PI * arg).sin() / (PI * arg)
    };

    n_ideal * sinc * sinc
}

/// Viscosity cutoff frequency.
///
/// The kinematic viscosity nu determines a dissipation scale:
///   omega_visc = nu / dx^2
///
/// Above this frequency, viscous damping exponentially suppresses phonons.
pub fn viscosity_cutoff(nu: f64, dx: f64) -> f64 {
    nu / (dx * dx)
}

/// Viscosity-cutoff Hawking spectrum.
///
/// Combines the lattice sinc^2 window with exponential viscous damping:
///   n_visc(omega) = n_lattice(omega) * exp(-omega / omega_visc)
pub fn viscosity_cutoff_spectrum(t_h: f64, omega: f64, dx: f64, nu: f64) -> f64 {
    let n_lattice = lattice_cutoff_spectrum(t_h, omega, dx);
    let omega_visc = viscosity_cutoff(nu, dx);

    if omega_visc <= 0.0 {
        return 0.0;
    }

    n_lattice * (-omega / omega_visc).exp()
}

/// Compute the spectral chi-squared between two spectra.
///
/// chi2 = sum_i [(n1(omega_i) - n2(omega_i))^2 / (n1(omega_i) + eps)]
pub fn spectral_chi_squared(
    n1: &[f64],
    n2: &[f64],
    eps: f64,
) -> f64 {
    assert_eq!(n1.len(), n2.len());
    n1.iter()
        .zip(n2.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff / (a.abs() + eps)
        })
        .sum()
}

/// Effective Hawking temperature from the lattice-cutoff spectrum.
///
/// Fits the low-frequency part of the spectrum (omega << omega_visc)
/// to a Planck distribution and extracts the effective temperature.
///
/// Uses least-squares fit of ln(1 + 1/n) vs omega to get 1/T.
pub fn effective_temperature(omegas: &[f64], spectrum: &[f64]) -> f64 {
    // ln(1 + 1/n(omega)) = omega / T for ideal Planck
    // Fit: y = omega / T, so T = sum(omega^2) / sum(omega * y)
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;

    for (&omega, &n) in omegas.iter().zip(spectrum.iter()) {
        if n > 1e-10 && omega > 1e-10 {
            let y = (1.0 + 1.0 / n).ln();
            sum_xy += omega * y;
            sum_xx += omega * omega;
        }
    }

    if sum_xy.abs() < 1e-30 {
        return 0.0;
    }

    sum_xx / sum_xy
}

/// Full spectrum computation at a given grid resolution.
///
/// Returns (omegas, ideal, lattice, viscous) arrays.
pub fn compute_spectra(
    t_h: f64,
    n_grid: usize,
    nu: f64,
    n_omega: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let dx = 1.0 / n_grid as f64;
    let omega_max = 2.0 / dx; // Nyquist

    let omegas: Vec<f64> = (1..=n_omega)
        .map(|i| omega_max * i as f64 / (n_omega as f64 + 1.0))
        .collect();

    let ideal: Vec<f64> = omegas.iter().map(|&w| ideal_hawking_spectrum(t_h, w)).collect();
    let lattice: Vec<f64> = omegas.iter().map(|&w| lattice_cutoff_spectrum(t_h, w, dx)).collect();
    let viscous: Vec<f64> = omegas
        .iter()
        .map(|&w| viscosity_cutoff_spectrum(t_h, w, dx, nu))
        .collect();

    (omegas, ideal, lattice, viscous)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ideal_spectrum_thermal() {
        // At omega = T, n = 1/(e-1) ~ 0.582
        let t = 1.0;
        let n = ideal_hawking_spectrum(t, t);
        let expected = 1.0 / (1.0_f64.exp() - 1.0);
        assert!((n - expected).abs() < 1e-10);
    }

    #[test]
    fn test_lattice_cutoff_reduces_spectrum() {
        let t = 0.1;
        let omega = 1.0;
        let dx = 0.1;
        let n_ideal = ideal_hawking_spectrum(t, omega);
        let n_lattice = lattice_cutoff_spectrum(t, omega, dx);
        // Lattice cutoff should reduce or equal the ideal spectrum
        assert!(n_lattice <= n_ideal + 1e-15);
    }

    #[test]
    fn test_viscosity_cutoff_further_reduces() {
        let t = 0.1;
        let omega = 1.0;
        let dx = 0.1;
        let nu = 0.01;
        let n_lattice = lattice_cutoff_spectrum(t, omega, dx);
        let n_viscous = viscosity_cutoff_spectrum(t, omega, dx, nu);
        assert!(n_viscous <= n_lattice + 1e-15);
    }

    #[test]
    fn test_viscosity_cutoff_frequency() {
        let nu = 0.01;
        let dx = 0.1;
        let omega_v = viscosity_cutoff(nu, dx);
        assert!((omega_v - 1.0).abs() < 1e-10, "nu/dx^2 = 0.01/0.01 = 1.0");
    }

    #[test]
    fn test_effective_temperature_recovery() {
        // Generate ideal spectrum and recover T
        let t_true = 0.5;
        let n_omega = 50;
        let omegas: Vec<f64> = (1..=n_omega)
            .map(|i| 0.1 * i as f64)
            .collect();
        let spectrum: Vec<f64> = omegas
            .iter()
            .map(|&w| ideal_hawking_spectrum(t_true, w))
            .collect();
        let t_fit = effective_temperature(&omegas, &spectrum);
        assert!(
            (t_fit - t_true).abs() / t_true < 0.05,
            "recovered T={:.3} vs true T={:.3}",
            t_fit, t_true
        );
    }

    #[test]
    fn test_compute_spectra_lengths() {
        let (omegas, ideal, lattice, viscous) = compute_spectra(0.1, 32, 0.01, 100);
        assert_eq!(omegas.len(), 100);
        assert_eq!(ideal.len(), 100);
        assert_eq!(lattice.len(), 100);
        assert_eq!(viscous.len(), 100);
    }
}
