//! EELS, photonic density of states, and absorption engineering methods on
//! `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split.
//! Seven methods covering near-field cavity QED and absorber design:
//!
//! EELS / loss functions:
//! - `surface_loss_function`: `Im[-1/(1+eps)]` -- peaks at surface plasmon.
//! - `volume_loss_weighted`: `omega * Im[-1/eps]` -- bulk EELS spectral weight.
//!
//! Cavity QED / Purcell:
//! - `purcell_factor`: LDOS enhancement near a planar surface.
//! - `lamb_shift_fractional`: dipole frequency shift near the surface.
//!
//! Absorber engineering:
//! - `absorption_per_pass`: Beer-Lambert through a thin film.
//! - `optimal_absorber_thickness`: 1/alpha penetration depth (63.2% absorption).
//! - `impedance_mismatch`: `|Z_surface/Z_0 - 1|` for normal-incidence design.
//!
//! References: Egerton (2011) "Electron Energy-Loss Spectroscopy in the
//! Electron Microscope" 3rd ed.; Novotny & Hecht (2012) "Principles of
//! Nano-Optics" 2nd ed. (Purcell and Lamb shift near surfaces).

use num_complex::Complex64;

use super::{C, DrudeLorentzParams};

impl DrudeLorentzParams {
    /// Surface EELS loss function `Im[-1/(1+eps(omega))]`. Peaks at the
    /// surface plasmon frequency where `Re[eps] = -1`. Used to probe
    /// surface excitations in low-energy electron scattering.
    pub fn surface_loss_function(&self, omega: f64) -> f64 {
        let eps = self.epsilon(omega);
        let denom = Complex64::new(1.0, 0.0) + eps;
        if denom.norm() < 1e-30 {
            return 0.0;
        }
        (-1.0 / denom).im
    }

    /// Bulk EELS spectral weight `omega * Im[-1/eps(omega)]`. Proportional
    /// to the differential EELS cross-section in the optical (q -> 0)
    /// limit; the `omega` weighting follows from the fluctuation-dissipation
    /// theorem.
    pub fn volume_loss_weighted(&self, omega: f64) -> f64 {
        omega * self.loss_function(omega)
    }

    /// Purcell factor for a dipole emitter at distance `d` from a planar
    /// surface: `F_P = 1 + (3/(4*(k*d)^3)) * Im[(eps-1)/(eps+1)]` with
    /// `k = omega/c`. Gives the LDOS enhancement relative to free space.
    /// Valid in the near-field regime (`k*d << 1`).
    pub fn purcell_factor(&self, omega: f64, distance_m: f64) -> f64 {
        let eps = self.epsilon(omega);
        let k = omega / C;
        let kd = k * distance_m;
        if kd < 1e-30 {
            return 1.0;
        }
        let reflection_factor = (eps - Complex64::new(1.0, 0.0)) / (eps + Complex64::new(1.0, 0.0));
        1.0 + 3.0 / (4.0 * kd.powi(3)) * reflection_factor.im
    }

    /// Fractional Lamb shift `delta_omega/omega = -(3/(8*(k*d)^3)) *
    /// Re[(eps-1)/(eps+1)]` for a dipole emitter near a planar surface.
    /// Negative = redshift (towards surface plasmon); positive = blueshift.
    pub fn lamb_shift_fractional(&self, omega: f64, distance_m: f64) -> f64 {
        let eps = self.epsilon(omega);
        let k = omega / C;
        let kd = k * distance_m;
        if kd < 1e-30 {
            return 0.0;
        }
        let reflection_factor = (eps - Complex64::new(1.0, 0.0)) / (eps + Complex64::new(1.0, 0.0));
        -3.0 / (8.0 * kd.powi(3)) * reflection_factor.re
    }

    /// Single-pass absorption fraction through a thin film:
    /// `A = 1 - exp(-alpha * thickness)`. Thin-film limit (`alpha*d << 1`):
    /// `A ~ alpha * d` (Beer-Lambert).
    pub fn absorption_per_pass(&self, omega: f64, thickness_m: f64) -> f64 {
        let alpha = self.absorption_coefficient(omega);
        1.0 - (-alpha * thickness_m).exp()
    }

    /// One penetration depth `d = 1/alpha`. Yields `A = 1 - 1/e ~ 63.2%`
    /// absorption. Returns `None` if `alpha < 1e-10` (transparent material).
    pub fn optimal_absorber_thickness(&self, omega: f64) -> Option<f64> {
        let alpha = self.absorption_coefficient(omega);
        if alpha < 1e-10 {
            return None;
        }
        Some(1.0 / alpha)
    }

    /// Impedance mismatch parameter `|Z_surface / Z_0 - 1|`. `Z_0 = 377 Ohm`
    /// (free-space impedance); `Z_surface = Z_0 / n` for normal incidence
    /// on a half-space. Returns 0 for perfect impedance match (zero
    /// reflection), larger values for high-reflectivity materials.
    pub fn impedance_mismatch(&self, omega: f64) -> f64 {
        let n = self.refractive_index(omega);
        let z_ratio = 1.0 / n;
        (z_ratio - Complex64::new(1.0, 0.0)).norm()
    }
}
