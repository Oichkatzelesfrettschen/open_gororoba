//! Phonon polariton and IR spectroscopy methods on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 16d). Six methods covering the polar-phonon Reststrahlen physics:
//!
//! - `surface_phonon_polariton_frequency`: `Re[eps] = -eps_d` crossing inside
//!   the Reststrahlen band (the phonon analogue of the surface plasmon).
//! - `phonon_polariton_wavevector`: same `k_PhP` formula as SPP, evaluated
//!   in the polariton band.
//! - `polariton_group_velocity`: `v_g = d_omega / d_k` -- can be `~ c/100`.
//! - `ir_activity_proxy`: `S_j * omega_j^2` ~ Born effective charge squared.
//! - `isotope_shift_estimate`: harmonic `delta_omega/omega = 1 - 1/sqrt(M_new/M)`.
//! - `bose_einstein_occupation`: thermal occupation `n_BE`.

use num_complex::Complex64;

use super::{DrudeLorentzParams, EV_TO_RADS, K_B_EV, ev_to_omega};

impl DrudeLorentzParams {
    /// Surface phonon-polariton frequency from `Re[eps(omega)] = -eps_d`,
    /// scanned inside the Reststrahlen band (or 0.01-1 eV fallback) and
    /// refined by bisection. Returns `None` if no crossing exists.
    pub fn surface_phonon_polariton_frequency(&self, eps_dielectric: f64) -> Option<f64> {
        let (scan_min, scan_max) = self
            .reststrahlen_band()
            .unwrap_or((ev_to_omega(0.01), ev_to_omega(1.0)));
        let n_scan = 2000;
        let d_omega = (scan_max - scan_min) / n_scan as f64;

        for i in 0..n_scan {
            let omega_a = scan_min + i as f64 * d_omega;
            let omega_b = omega_a + d_omega;
            let val_a = self.epsilon(omega_a).re + eps_dielectric;
            let val_b = self.epsilon(omega_b).re + eps_dielectric;

            if val_a * val_b < 0.0 {
                let mut lo = omega_a;
                let mut hi = omega_b;
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    let val_mid = self.epsilon(mid).re + eps_dielectric;
                    if val_a * val_mid < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return Some(0.5 * (lo + hi));
            }
        }
        None
    }

    /// Phonon-polariton wavevector (identical SPP formula evaluated in the
    /// Reststrahlen band instead of the metallic regime).
    pub fn phonon_polariton_wavevector(&self, omega: f64, eps_dielectric: f64) -> Complex64 {
        self.spp_wavevector(omega, eps_dielectric)
    }

    /// Polariton group velocity from `v_g = d_omega/d_k`, estimated by
    /// finite difference of the inverse dispersion `k(omega)`.
    pub fn polariton_group_velocity(&self, omega: f64, eps_dielectric: f64) -> f64 {
        let dw = omega * 1e-6;
        let k1 = self.spp_wavevector(omega - dw, eps_dielectric).re;
        let k2 = self.spp_wavevector(omega + dw, eps_dielectric).re;
        let dk = k2 - k1;
        if dk.abs() < 1e-30 {
            return 0.0;
        }
        2.0 * dw / dk
    }

    /// IR activity proxy `S_j * omega_j^2` for the j-th Lorentz oscillator
    /// (~ Born effective charge squared). Returns `None` if the index is
    /// out of range.
    pub fn ir_activity_proxy(&self, oscillator_index: usize) -> Option<f64> {
        let osc = self.oscillators.get(oscillator_index)?;
        Some(osc.strength * osc.omega_0_ev * osc.omega_0_ev)
    }

    /// Harmonic isotope frequency shift `delta_omega/omega = 1 - 1/sqrt(M_new/M)`
    /// from `omega ~ 1/sqrt(M)`. `mass_ratio = M_new / M_original`.
    pub fn isotope_shift_estimate(mass_ratio: f64) -> f64 {
        if mass_ratio <= 0.0 {
            return 0.0;
        }
        1.0 - (1.0 / mass_ratio).sqrt()
    }

    /// Bose-Einstein phonon occupation `n_BE = 1/(exp(hbar*omega/kT) - 1)`.
    /// Returns 0 if `T = 0` or `omega = 0`.
    pub fn bose_einstein_occupation(omega: f64, temperature_k: f64) -> f64 {
        if temperature_k < 1e-10 || omega < 1e-10 {
            return 0.0;
        }
        let hbar_omega_ev = omega / EV_TO_RADS;
        let kt_ev = K_B_EV * temperature_k;
        let x = hbar_omega_ev / kt_ev;
        if x > 500.0 {
            return 0.0;
        }
        1.0 / (x.exp() - 1.0)
    }
}
