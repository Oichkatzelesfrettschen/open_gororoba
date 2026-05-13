//! Plasmonic sensing and SERS metrics on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 16b). Seven methods covering LSPR-based sensors + SERS substrates:
//!
//! - `refractive_index_sensitivity`: d(lambda_LSPR)/d(n) in nm/RIU.
//! - `figure_of_merit_sensor`: sensitivity / FWHM (sharper = better).
//! - `field_enhancement_factor`: |E_loc/E_0| from Clausius-Mossotti.
//! - `sers_enhancement_factor`: 4th power of the field enhancement.
//! - `decay_rate_enhancement`: Gamma/Gamma_0 near a planar surface.
//! - `quantum_efficiency_near_surface`: free-space QY transformed by F_nr.
//! - `hot_electron_generation_proxy`: `|Im[eps]|` material factor.

use std::f64::consts::PI;

use num_complex::Complex64;

use super::{C, DrudeLorentzParams, EV_TO_RADS};

impl DrudeLorentzParams {
    /// Refractive-index sensitivity: shift of LSPR wavelength per RIU.
    /// Finite-difference `d(lambda_LSPR)/d(n)` at `eps_d` and `eps_d + delta`.
    /// Returns `None` if no LSPR is found. Result in nm/RIU.
    pub fn refractive_index_sensitivity(&self, eps_dielectric: f64) -> Option<f64> {
        let dn = 0.01;
        let n_d = eps_dielectric.sqrt();
        let omega1 = self.lspr_frequency(eps_dielectric)?;
        let omega2 = self.lspr_frequency((n_d + dn) * (n_d + dn))?;
        let lambda1 = 2.0 * PI * C / omega1 * 1e9;
        let lambda2 = 2.0 * PI * C / omega2 * 1e9;
        Some((lambda2 - lambda1) / dn)
    }

    /// Sensor FoM = sensitivity / FWHM with FWHM estimated from the Drude
    /// damping rate. Higher FoM means sharper resonances and better detection.
    pub fn figure_of_merit_sensor(&self, eps_dielectric: f64) -> Option<f64> {
        let sensitivity = self.refractive_index_sensitivity(eps_dielectric)?;
        let omega_lspr = self.lspr_frequency(eps_dielectric)?;
        let gamma = self.drude.as_ref()?.gamma_ev * EV_TO_RADS;
        let lambda_lspr = 2.0 * PI * C / omega_lspr * 1e9;
        let fwhm_nm = gamma * lambda_lspr * lambda_lspr / (2.0 * PI * C) * 1e9;
        if fwhm_nm.abs() < 1e-30 {
            return None;
        }
        Some(sensitivity.abs() / fwhm_nm)
    }

    /// Quasistatic field enhancement |E_loc/E_0| at a nanoparticle surface.
    /// Clausius-Mossotti: `alpha = 3V*eps_0*(eps - eps_d)/(eps + 2*eps_d)`;
    /// surface enhancement at the equator includes the dipole + incident
    /// field, giving `1 + 2 * |ratio|`.
    pub fn field_enhancement_factor(&self, omega: f64, eps_dielectric: f64) -> f64 {
        let eps = self.epsilon(omega);
        let eps_d = Complex64::new(eps_dielectric, 0.0);
        let ratio = (eps - eps_d) / (eps + 2.0 * eps_d);
        1.0 + 2.0 * ratio.norm()
    }

    /// SERS electromagnetic enhancement factor `|E_loc/E_0|^4` (two field
    /// factors each for excitation and emission). Chemical enhancement
    /// (10-100x) is additional and separately modelled.
    pub fn sers_enhancement_factor(&self, omega: f64, eps_dielectric: f64) -> f64 {
        let fe = self.field_enhancement_factor(omega, eps_dielectric);
        fe * fe * fe * fe
    }

    /// Total decay-rate enhancement `Gamma/Gamma_0 = 1 + 3/(2*(kd)^3) *
    /// Im[(eps-1)/(eps+1)]` in the near-field (`kd << 1`). Includes both
    /// radiative and non-radiative channels.
    pub fn decay_rate_enhancement(&self, omega: f64, distance_m: f64) -> f64 {
        let eps = self.epsilon(omega);
        let k = omega / C;
        let kd = k * distance_m;
        let ratio = (eps - 1.0) / (eps + 1.0);
        1.0 + 1.5 / (kd * kd * kd) * ratio.im
    }

    /// Quantum efficiency of an emitter near a planar surface:
    /// `eta = QY_free * F_rad / (QY_free * F_rad + (1 - QY_free) + F_nr)`.
    /// Here `F_rad ~ 1` (far-field) and
    /// `F_nr ~ 3/(4*(kd)^3) * Im[(eps-1)/(eps+1)]`.
    pub fn quantum_efficiency_near_surface(
        &self,
        omega: f64,
        distance_m: f64,
        qy_free: f64,
    ) -> f64 {
        let eps = self.epsilon(omega);
        let k = omega / C;
        let kd = k * distance_m;
        let ratio = (eps - 1.0) / (eps + 1.0);
        let f_nr = 0.75 / (kd * kd * kd) * ratio.im;
        let f_nr_abs = f_nr.abs();
        let numerator = qy_free;
        let denominator = qy_free + (1.0 - qy_free) + qy_free * f_nr_abs;
        if denominator < 1e-30 {
            return 0.0;
        }
        numerator / denominator
    }

    /// Hot-electron generation rate proxy `|Im[eps]|`. The full rate scales
    /// as `Im[eps(omega)] * |E|^2`; this returns the material factor alone.
    pub fn hot_electron_generation_proxy(&self, omega: f64) -> f64 {
        self.epsilon(omega).im.abs()
    }
}
