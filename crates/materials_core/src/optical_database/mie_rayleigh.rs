//! Mie and Rayleigh scattering methods on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 17a). Six methods for small-particle scattering in the
//! Clausius-Mossotti polarizability regime:
//!
//! - `polarizability_clausius_mossotti`: `alpha = 4*pi*a^3 * (eps - 1)/(eps + 2)`.
//! - `rayleigh_cross_section`: `C_sca = (8*pi/3) k^4 a^6 |K|^2`.
//! - `rayleigh_scattering_efficiency`: `Q_sca = C_sca / (pi*a^2)`.
//! - `mie_extinction_efficiency`: `Q_ext = 4x * Im[K]` (small-particle limit).
//! - `mie_scattering_albedo`: `Q_sca / Q_ext`.
//! - `absorption_cross_section_mie`: `C_abs = C_ext - C_sca`.
//! - `radiation_pressure_efficiency`: Rayleigh limit g~0 so `Q_pr ~ Q_ext`.

use num_complex::Complex64;

use super::{C, DrudeLorentzParams};

impl DrudeLorentzParams {
    /// Clausius-Mossotti polarizability `alpha = 4*pi*a^3 * (eps - 1)/(eps + 2)`
    /// in m^3 for a sphere of radius `a`.
    pub fn polarizability_clausius_mossotti(&self, omega: f64, radius_m: f64) -> Complex64 {
        let eps = self.epsilon(omega);
        let ratio = (eps - 1.0) / (eps + 2.0);
        4.0 * std::f64::consts::PI * radius_m.powi(3) * ratio
    }

    /// Rayleigh scattering cross section `C_sca = (8*pi/3) * k^4 * a^6 * |K|^2`
    /// with `K = (eps - 1)/(eps + 2)`.
    pub fn rayleigh_cross_section(&self, omega: f64, radius_m: f64) -> f64 {
        let k = omega / C;
        let eps = self.epsilon(omega);
        let k_factor = (eps - 1.0) / (eps + 2.0);
        (8.0 * std::f64::consts::PI / 3.0) * k.powi(4) * radius_m.powi(6) * k_factor.norm_sqr()
    }

    /// Rayleigh scattering efficiency `Q_sca = C_sca / (pi * a^2)`.
    pub fn rayleigh_scattering_efficiency(&self, omega: f64, radius_m: f64) -> f64 {
        let c_sca = self.rayleigh_cross_section(omega, radius_m);
        c_sca / (std::f64::consts::PI * radius_m * radius_m)
    }

    /// Mie extinction efficiency in the small-particle limit (`x = k*a << 1`):
    /// `Q_ext = 4x * Im[(eps - 1)/(eps + 2)]`.
    pub fn mie_extinction_efficiency(&self, omega: f64, radius_m: f64) -> f64 {
        let k = omega / C;
        let x = k * radius_m;
        let eps = self.epsilon(omega);
        let k_factor = (eps - 1.0) / (eps + 2.0);
        4.0 * x * k_factor.im
    }

    /// Mie scattering albedo `Q_sca / Q_ext` clamped to `[0, 1]`. Near 0
    /// for absorbing particles; near 1 for transparent dielectrics.
    pub fn mie_scattering_albedo(&self, omega: f64, radius_m: f64) -> f64 {
        let q_ext = self.mie_extinction_efficiency(omega, radius_m);
        if q_ext.abs() < 1e-30 {
            return 0.0;
        }
        let q_sca = self.rayleigh_scattering_efficiency(omega, radius_m);
        (q_sca / q_ext).clamp(0.0, 1.0)
    }

    /// Mie absorption cross section `C_abs = C_ext - C_sca` (small-particle
    /// limit). Clamped to non-negative.
    pub fn absorption_cross_section_mie(&self, omega: f64, radius_m: f64) -> f64 {
        let k = omega / C;
        let x = k * radius_m;
        let eps = self.epsilon(omega);
        let k_factor = (eps - 1.0) / (eps + 2.0);
        let c_ext = 4.0 * std::f64::consts::PI * radius_m * radius_m * x * k_factor.im;
        let c_sca = self.rayleigh_cross_section(omega, radius_m);
        (c_ext - c_sca).max(0.0)
    }

    /// Radiation pressure efficiency `Q_pr = Q_ext - g * Q_sca`. In the
    /// Rayleigh limit the asymmetry parameter `g ~ 0` so `Q_pr ~ Q_ext`.
    pub fn radiation_pressure_efficiency(&self, omega: f64, radius_m: f64) -> f64 {
        self.mie_extinction_efficiency(omega, radius_m)
    }
}
