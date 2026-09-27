//! Thin-film interference and coating-design metrics on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 16c). Six methods covering single-layer film optics + colorimetry:
//!
//! - `thin_film_reflectance` / `thin_film_transmittance`: Airy formulas.
//! - `thin_film_phase_shift`: single-traversal phase.
//! - `constructive_interference_orders`: integer m where `2*n*d ~ m*lambda`.
//! - `fabry_perot_finesse`: `F = pi*sqrt(R) / (1 - R)`.
//! - `color_coordinates_cie`: CIE 1931 (x, y, Y) from spectral reflectance.

use std::f64::consts::PI;

use num_complex::Complex64;

use super::{C, DrudeLorentzParams, ev_to_omega, omega_to_ev};

impl DrudeLorentzParams {
    fn thin_film_reflectance_with_substrate_index(
        &self,
        omega: f64,
        thickness_m: f64,
        substrate_index: Complex64,
    ) -> f64 {
        assert!(
            omega.is_finite() && omega > 0.0,
            "angular frequency must be finite and positive"
        );
        assert!(
            thickness_m.is_finite() && thickness_m >= 0.0,
            "film thickness must be finite and nonnegative"
        );
        let film_index = self.refractive_index(omega);
        let incident_index = Complex64::new(1.0, 0.0);
        let incident_film = (incident_index - film_index) / (incident_index + film_index);
        let film_substrate = (film_index - substrate_index) / (film_index + substrate_index);
        let phase = Complex64::new(0.0, 2.0) * film_index * omega * thickness_m / C;
        let round_trip = phase.exp();
        let amplitude = (incident_film + film_substrate * round_trip)
            / (1.0 + incident_film * film_substrate * round_trip);
        amplitude.norm_sqr()
    }

    /// Single-layer thin-film reflectance on a substrate via the coherent
    /// Airy formula. Normal incidence from air; film of thickness `d` and
    /// refractive index `n_film(omega)` on a substrate with index `n_sub`.
    pub fn thin_film_reflectance(&self, omega: f64, thickness_m: f64, n_substrate: f64) -> f64 {
        self.thin_film_reflectance_with_substrate_index(
            omega,
            thickness_m,
            Complex64::new(n_substrate, 0.0),
        )
    }

    /// Single-layer reflectance on a dispersive, absorbing substrate.
    ///
    /// The film and substrate responses are evaluated at the same angular
    /// frequency and use the same passive Fourier convention.
    pub fn thin_film_reflectance_on_material(
        &self,
        omega: f64,
        thickness_m: f64,
        substrate: &Self,
    ) -> f64 {
        self.thin_film_reflectance_with_substrate_index(
            omega,
            thickness_m,
            substrate.refractive_index(omega),
        )
    }

    /// Single-layer thin-film transmittance on a substrate via the Airy
    /// formula. For non-absorbing films `T = 1 - R`; absorbing films have
    /// `T < 1 - R` due to film extinction.
    pub fn thin_film_transmittance(&self, omega: f64, thickness_m: f64, n_substrate: f64) -> f64 {
        let n_film = self.refractive_index(omega);
        let n_i = Complex64::new(1.0, 0.0);
        let n_s = Complex64::new(n_substrate, 0.0);
        let r12 = (n_i - n_film) / (n_i + n_film);
        let t12 = 2.0 * n_i / (n_i + n_film);
        let r23 = (n_film - n_s) / (n_film + n_s);
        let t23 = 2.0 * n_film / (n_film + n_s);
        let delta = n_film * thickness_m * omega / C;
        let exp_phase = Complex64::new(0.0, delta.re).exp() * (-delta.im).exp();
        let t_total = (t12 * t23 * exp_phase) / (1.0 + r12 * r23 * exp_phase * exp_phase);
        (n_s.re / n_i.re) * t_total.norm_sqr()
    }

    /// Single-traversal phase shift `phi = Re[n] * omega * d / c` in radians.
    pub fn thin_film_phase_shift(&self, omega: f64, thickness_m: f64) -> f64 {
        let n = self.refractive_index(omega);
        n.re * omega * thickness_m / C
    }

    /// Constructive interference orders `m = 1, 2, ..., floor(2*n*d/lambda)`
    /// for a thin film of thickness `d`.
    pub fn constructive_interference_orders(&self, omega: f64, thickness_m: f64) -> Vec<u32> {
        let n = self.refractive_index(omega).re;
        let lambda = 2.0 * PI * C / omega;
        let max_order = (2.0 * n * thickness_m / lambda).floor() as u32;
        (1..=max_order).collect()
    }

    /// Fabry-Perot finesse `F = pi * sqrt(R) / (1 - R)` for a symmetric
    /// etalon (identical media on both sides, or computed from the air-film
    /// reflectance).
    pub fn fabry_perot_finesse(&self, omega: f64) -> f64 {
        let r = self.reflectivity_normal(omega);
        if r >= 1.0 - 1e-15 {
            return f64::INFINITY;
        }
        PI * r.sqrt() / (1.0 - r)
    }

    /// CIE 1931 chromaticity coordinates `(x, y, Y_luminance)` from
    /// spectral reflectance. Integrates `R(omega)` against Gaussian
    /// approximations of the CIE color-matching functions over 1.55-3.10 eV.
    pub fn color_coordinates_cie(&self, n_steps: usize) -> (f64, f64, f64) {
        let omega_min = ev_to_omega(1.55);
        let omega_max = ev_to_omega(3.10);
        let d_omega = (omega_max - omega_min) / n_steps as f64;
        let mut x_sum = 0.0_f64;
        let mut y_sum = 0.0_f64;
        let mut z_sum = 0.0_f64;
        for i in 0..n_steps {
            let omega = omega_min + (i as f64 + 0.5) * d_omega;
            let ev = omega_to_ev(omega);
            let r = self.reflectivity_normal(omega);
            let x_bar = 1.056 * (-(ev - 1.82_f64).powi(2) / (2.0 * 0.12)).exp()
                + 0.362 * (-(ev - 2.24_f64).powi(2) / (2.0 * 0.07)).exp();
            let y_bar = 0.821 * (-(ev - 2.23_f64).powi(2) / (2.0 * 0.08)).exp()
                + 0.286 * (-(ev - 2.06_f64).powi(2) / (2.0 * 0.14)).exp();
            let z_bar = 1.217 * (-(ev - 2.72_f64).powi(2) / (2.0 * 0.08)).exp()
                + 0.681 * (-(ev - 2.98_f64).powi(2) / (2.0 * 0.12)).exp();
            x_sum += r * x_bar * d_omega;
            y_sum += r * y_bar * d_omega;
            z_sum += r * z_bar * d_omega;
        }
        let total = x_sum + y_sum + z_sum;
        if total < 1e-30 {
            return (0.333, 0.333, 0.0);
        }
        (x_sum / total, y_sum / total, y_sum)
    }
}
