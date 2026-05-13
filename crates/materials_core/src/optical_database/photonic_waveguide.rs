//! Photonic crystal and waveguide metrics on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 16a). Seven methods for fiber + waveguide design:
//!
//! - `numerical_aperture`: `NA = sqrt(n_core^2 - n_clad^2)`.
//! - `v_parameter`: normalized frequency `V = (2*pi*a/lambda) * NA`.
//! - `confinement_factor`: Marcuse Gaussian power fraction in core.
//! - `effective_mode_area`: `pi * w^2` (Gaussian mode-field radius).
//! - `modal_birefringence`: `2 * Im[n]` for absorption-induced PDL.
//! - `bend_loss_critical_radius`: Unger formula `lambda * n_eff / (pi * NA^3)`.
//! - `chromatic_dispersion_ps_nm_km`: fiber-convention dispersion.

use std::f64::consts::PI;

use super::{C, DrudeLorentzParams};

impl DrudeLorentzParams {
    /// Step-index fiber numerical aperture `NA = sqrt(n_core^2 - n_clad^2)`.
    /// Returns `None` if `n_core <= n_clad` (no guiding).
    pub fn numerical_aperture(&self, omega: f64, n_cladding: f64) -> Option<f64> {
        let n_core = self.refractive_index(omega).re;
        let diff = n_core * n_core - n_cladding * n_cladding;
        if diff > 0.0 { Some(diff.sqrt()) } else { None }
    }

    /// V-parameter (normalized frequency) for a step-index fiber:
    /// `V = (2*pi/lambda) * a * NA`. Single-mode cutoff at `V = 2.405` (LP11).
    pub fn v_parameter(&self, omega: f64, core_radius_m: f64, n_cladding: f64) -> Option<f64> {
        let na = self.numerical_aperture(omega, n_cladding)?;
        let lambda = 2.0 * PI * C / omega;
        Some(2.0 * PI * core_radius_m / lambda * na)
    }

    /// Confinement factor (fraction of optical power within the fiber core).
    /// Gaussian approximation: `Gamma = 1 - exp(-2*(a/w)^2)` with the
    /// Marcuse formula `w/a = 0.65 + 1.619/V^1.5 + 2.879/V^6`. Returns
    /// `None` if `V < 0.8` (formula invalid) or no guiding.
    pub fn confinement_factor(
        &self,
        omega: f64,
        core_radius_m: f64,
        n_cladding: f64,
    ) -> Option<f64> {
        let v = self.v_parameter(omega, core_radius_m, n_cladding)?;
        if v < 0.8 {
            return None;
        }
        let w_over_a = 0.65 + 1.619 / v.powf(1.5) + 2.879 / v.powi(6);
        let gamma = 1.0 - (-2.0 / (w_over_a * w_over_a)).exp();
        Some(gamma)
    }

    /// Effective mode area `A_eff = pi * w^2` for single-mode fiber
    /// (Gaussian approximation). Returns `None` if `V < 0.8` or no guiding.
    pub fn effective_mode_area(
        &self,
        omega: f64,
        core_radius_m: f64,
        n_cladding: f64,
    ) -> Option<f64> {
        let v = self.v_parameter(omega, core_radius_m, n_cladding)?;
        if v < 0.8 {
            return None;
        }
        let w = core_radius_m * (0.65 + 1.619 / v.powf(1.5) + 2.879 / v.powi(6));
        Some(PI * w * w)
    }

    /// Effective modal birefringence `2 * |Im[n]|` -- characterises
    /// polarisation-dependent loss for absorbing materials. Zero for
    /// isotropic transparent DL materials.
    pub fn modal_birefringence(&self, omega: f64) -> f64 {
        let n = self.refractive_index(omega);
        2.0 * n.im.abs()
    }

    /// Critical bend radius below which radiation loss dominates (Unger):
    /// `R_c = lambda * n_eff / (pi * NA^3)`. Returns `None` if no guiding.
    pub fn bend_loss_critical_radius(&self, omega: f64, n_cladding: f64) -> Option<f64> {
        let na = self.numerical_aperture(omega, n_cladding)?;
        let n_core = self.refractive_index(omega).re;
        let lambda = 2.0 * PI * C / omega;
        Some(lambda * n_core / (PI * na * na * na))
    }

    /// Chromatic dispersion in fiber units ps/(nm*km):
    /// `D = -(2*pi*c/lambda^2) * beta_2` then converted from SI s/m^2
    /// to ps/(nm*km) via factor 1e6.
    pub fn chromatic_dispersion_ps_nm_km(&self, omega: f64) -> f64 {
        let beta2 = self.gvd_beta2(omega);
        let lambda = 2.0 * PI * C / omega;
        let d_si = -(2.0 * PI * C / (lambda * lambda)) * beta2;
        d_si * 1e6
    }
}
