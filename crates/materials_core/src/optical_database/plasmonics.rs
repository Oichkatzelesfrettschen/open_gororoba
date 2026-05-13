//! Surface plasmon polariton (SPP) and localized surface plasmon resonance
//! methods on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split.
//! Four methods covering surface-bound electromagnetic modes:
//!
//! - `spp_wavevector`: complex `k_spp` for a flat metal/dielectric interface.
//! - `spp_propagation_length`: `L_spp = 1 / (2 * Im[k_spp])`.
//! - `evanescent_decay_length`: vacuum/dielectric penetration depth in
//!   the metallic regime (`Re[eps] < 0`).
//! - `lspr_frequency`: Frohlich condition for a sphere
//!   (`Re[eps] = -2 * eps_d`); locates the LSPR by scan from 0.1-15 eV.
//!
//! References: Maier (2007) "Plasmonics: Fundamentals and Applications";
//! Raether (1988) "Surface Plasmons on Smooth and Rough Surfaces and on
//! Gratings".

use num_complex::Complex64;

use super::{C, DrudeLorentzParams, EV_TO_RADS};

impl DrudeLorentzParams {
    /// Surface plasmon polariton wavevector `k_spp` in 1/m.
    /// `k_spp = (omega/c) * sqrt(eps_m * eps_d / (eps_m + eps_d))` where
    /// `eps_d` is the dielectric medium permittivity (default vacuum = 1).
    /// SPPs exist when `Re[eps_m] < -Re[eps_d]`. Returns the complex
    /// `k_spp`; `Re` gives the spatial wavelength, `Im` the decay.
    pub fn spp_wavevector(&self, omega: f64, eps_dielectric: f64) -> Complex64 {
        let eps_m = self.epsilon(omega);
        let eps_d = Complex64::new(eps_dielectric, 0.0);
        let ratio = (eps_m * eps_d) / (eps_m + eps_d);
        (omega / C) * ratio.sqrt()
    }

    /// SPP propagation length in meters: `L_spp = 1 / (2 * Im[k_spp])`.
    /// The 1/e decay length of SPP intensity along the surface. For gold
    /// at 633 nm, `L_spp ~ 10 um`. Returns `None` if `Im[k_spp]` is
    /// non-positive (unphysical: no damping).
    pub fn spp_propagation_length(&self, omega: f64, eps_dielectric: f64) -> Option<f64> {
        let k_spp = self.spp_wavevector(omega, eps_dielectric);
        let k_im = k_spp.im.abs();
        if k_im < 1e-30 {
            return None;
        }
        Some(1.0 / (2.0 * k_im))
    }

    /// Evanescent decay length into vacuum/dielectric in meters:
    /// `delta = c / (omega * sqrt(-Re[eps_m]))`. Valid only when
    /// `Re[eps_m] < 0` (metallic regime). Determines how deeply evanescent
    /// fields penetrate the medium -- critical for Casimir proximity effects.
    pub fn evanescent_decay_length(&self, omega: f64) -> Option<f64> {
        let eps_re = self.epsilon(omega).re;
        if eps_re >= 0.0 {
            return None;
        }
        Some(C / (omega * (-eps_re).sqrt()))
    }

    /// Localized surface plasmon resonance (LSPR) frequency in rad/s.
    /// Solves the Frohlich condition `Re[eps_m(omega)] = -2 * eps_d` for a
    /// spherical nanoparticle in a dielectric medium. Scans 0.1 - 15 eV.
    /// Returns `None` if no crossing is found (pure dielectrics).
    pub fn lspr_frequency(&self, eps_dielectric: f64) -> Option<f64> {
        let target = -2.0 * eps_dielectric;
        let steps: usize = 3000;
        let ev_min = 0.1;
        let ev_max = 15.0;
        let dev = (ev_max - ev_min) / steps as f64;
        let mut prev_re = self.epsilon(ev_min * EV_TO_RADS).re;
        for i in 1..=steps {
            let ev = ev_min + i as f64 * dev;
            let omega = ev * EV_TO_RADS;
            let re = self.epsilon(omega).re;
            if prev_re < target && re >= target {
                let frac = (target - prev_re) / (re - prev_re);
                return Some(((ev - dev) + frac * dev) * EV_TO_RADS);
            }
            prev_re = re;
        }
        None
    }
}
