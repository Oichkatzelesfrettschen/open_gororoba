//! Ellipsometry, thermal emission, epsilon-near-zero, and Reststrahlen
//! band methods on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split.
//! Seven methods grouped by experimental probe:
//!
//! Ellipsometry: `psi_delta` returns `(psi, delta)` such that
//! `rho = r_p/r_s = tan(psi) * exp(i*delta)`.
//!
//! Thermal emission (Kirchhoff's law on opaque slabs):
//! - `emissivity`: 1 - reflectivity_normal.
//! - `spectral_emittance`: emissivity * Planck blackbody.
//! - `integrated_emissivity`: Planck-weighted average over a frequency band.
//!
//! Epsilon-near-zero (ENZ):
//! - `enz_frequency`: first `Re[eps] = 0` crossing via bisection.
//! - `enz_group_velocity`: `v_g/c = 1/Re[n_g]` at the ENZ point.
//!
//! Reststrahlen band: `(omega_TO, omega_LO)` for polar dielectrics
//! via Lyddane-Sachs-Teller `omega_LO^2 = omega_TO^2 * eps_s / eps_inf`.
//!
//! References: Fujiwara (2007) "Spectroscopic Ellipsometry"; Reststrahlen
//! and LST in Born & Huang (1954) "Dynamical Theory of Crystal Lattices".

use num_complex::Complex64;

use super::{C, DrudeLorentzParams, EV_TO_RADS, HBAR_EV_S, K_B_EV};

impl DrudeLorentzParams {
    /// Spectroscopic ellipsometry angles `(psi, delta)` at frequency
    /// `omega` and incidence angle `theta_i` (rad). Ellipsometry measures
    /// `rho = r_p / r_s = tan(psi) * exp(i*delta)`.
    pub fn psi_delta(&self, omega: f64, theta_i: f64) -> (f64, f64) {
        let rs = self.fresnel_rs(omega, theta_i, 1.0);
        let rp = self.fresnel_rp(omega, theta_i, 1.0);
        let rho = if rs.norm() < 1e-30 {
            Complex64::new(0.0, 0.0)
        } else {
            rp / rs
        };
        let psi = rho.norm().atan();
        let delta = rho.arg();
        (psi, delta)
    }

    /// Emissivity = absorptivity = `1 - reflectivity_normal` for an opaque
    /// slab (Kirchhoff's law, normal-incidence hemispherical).
    pub fn emissivity(&self, omega: f64) -> f64 {
        1.0 - self.reflectivity_normal(omega)
    }

    /// Spectral radiance `L(omega, T) = emissivity(omega) * B(omega, T)`
    /// in W / m^2 / sr / (rad/s), with Planck function
    /// `B = hbar * omega^3 / (4*pi^3 * c^2 * (exp(hbar*omega/k_B/T) - 1))`.
    pub fn spectral_emittance(&self, omega: f64, temperature_k: f64) -> f64 {
        if temperature_k < 1e-10 || omega < 1e-10 {
            return 0.0;
        }
        let hbar = HBAR_EV_S * 1.602_176_634e-19;
        let k_b = K_B_EV * 1.602_176_634e-19;
        let x = hbar * omega / (k_b * temperature_k);
        if x > 500.0 {
            return 0.0;
        }
        let planck =
            hbar * omega.powi(3) / (4.0 * std::f64::consts::PI.powi(3) * C * C * (x.exp() - 1.0));
        self.emissivity(omega) * planck
    }

    /// Planck-weighted integrated emissivity `eta_total = integral[eps(omega)*
    /// B(omega,T)] / integral[B(omega,T)]` over `[omega_min, omega_max]`
    /// via trapezoidal rule with `n_steps` intervals.
    pub fn integrated_emissivity(
        &self,
        temperature_k: f64,
        omega_min: f64,
        omega_max: f64,
        n_steps: usize,
    ) -> f64 {
        if n_steps < 2 || omega_max <= omega_min {
            return 0.0;
        }
        let hbar = HBAR_EV_S * 1.602_176_634e-19;
        let k_b = K_B_EV * 1.602_176_634e-19;
        let d_omega = (omega_max - omega_min) / n_steps as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..=n_steps {
            let omega = omega_min + i as f64 * d_omega;
            let x = hbar * omega / (k_b * temperature_k);
            if !(1e-30..=500.0).contains(&x) {
                continue;
            }
            let planck = omega.powi(3) / (x.exp() - 1.0);
            let w = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
            num += w * self.emissivity(omega) * planck;
            den += w * planck;
        }
        if den < 1e-30 {
            return 0.0;
        }
        num / den
    }

    /// Epsilon-near-zero frequency: first `Re[eps] = 0` crossing via
    /// bisection in `[scan_min, scan_max]` (rad/s). For metals this is
    /// the screened plasma frequency.
    pub fn enz_frequency(&self, scan_min: f64, scan_max: f64) -> Option<f64> {
        let n_scan = 2000;
        let d_omega = (scan_max - scan_min) / n_scan as f64;
        let mut prev_re = self.epsilon(scan_min).re;
        let mut crossing_omega = None;
        for i in 1..=n_scan {
            let omega = scan_min + i as f64 * d_omega;
            let re = self.epsilon(omega).re;
            if prev_re * re < 0.0 {
                let mut lo = omega - d_omega;
                let mut hi = omega;
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    let mid_re = self.epsilon(mid).re;
                    if prev_re * mid_re < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        prev_re = mid_re;
                    }
                }
                crossing_omega = Some(0.5 * (lo + hi));
                break;
            }
            prev_re = re;
        }
        crossing_omega
    }

    /// Group velocity at the ENZ frequency, normalized to c: `v_g/c = 1/Re[n_g]`.
    /// At ENZ, `Re[eps] ~ 0` so the phase velocity diverges, but the group
    /// velocity remains finite and can be very slow (slow light). Returns
    /// `None` if no ENZ crossing exists.
    pub fn enz_group_velocity(&self, scan_min: f64, scan_max: f64) -> Option<f64> {
        let omega_enz = self.enz_frequency(scan_min, scan_max)?;
        let n_g = self.group_refractive_index(omega_enz);
        if n_g.abs() < 1e-30 {
            return None;
        }
        Some(1.0 / n_g)
    }

    /// Reststrahlen band edges `(omega_TO, omega_LO)` for polar dielectrics.
    /// The band `omega_TO < omega < omega_LO` has `Re[eps] < 0` (metallic-like).
    /// `omega_LO` is estimated via Lyddane-Sachs-Teller:
    /// `omega_LO^2 = omega_TO^2 * eps_static / eps_inf` using the strongest
    /// Lorentz oscillator. Returns `None` if no oscillators exist or
    /// `eps_static <= eps_inf`.
    pub fn reststrahlen_band(&self) -> Option<(f64, f64)> {
        if self.oscillators.is_empty() {
            return None;
        }
        let strongest = self.oscillators.iter().max_by(|a, b| {
            a.strength
                .partial_cmp(&b.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

        let omega_to = strongest.omega_0_ev * EV_TO_RADS;
        let eps_s = self.eps_inf + strongest.strength;
        if eps_s < 1e-10 || self.eps_inf < 1e-10 {
            return None;
        }
        let lst_ratio = eps_s / self.eps_inf;
        if lst_ratio <= 1.0 {
            return None;
        }
        let omega_lo = omega_to * lst_ratio.sqrt();
        Some((omega_to, omega_lo))
    }
}
