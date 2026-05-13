//! Photovoltaic + solar energy + thermophotovoltaic metrics on
//! `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 15). Seven methods for solar-cell + selective-emitter design:
//!
//! - `solar_absorptance`: AM1.5G-weighted absorption (5800 K approximation).
//! - `solar_reflectance`: `1 - solar_absorptance`.
//! - `antireflection_thickness`: quarter-wave layer for given frequency.
//! - `wien_peak_omega` / `wien_peak_ev`: peak blackbody frequency.
//! - `luminous_reflectance`: photopic-V-weighted reflectance.
//! - `selective_emitter_efficiency`: TPV figure-of-merit.

use super::{C, DrudeLorentzParams, HBAR_EV_S, K_B_EV, ev_to_omega};

impl DrudeLorentzParams {
    /// Solar-weighted absorptance using a 5800 K blackbody envelope as a
    /// simplified AM1.5G spectrum over 0.3 - 4.0 eV. Trapezoidal integral
    /// with `n_steps` panels.
    pub fn solar_absorptance(&self, n_steps: usize) -> f64 {
        if n_steps < 2 {
            return 0.0;
        }
        let e_min = 0.3;
        let e_max = 4.0;
        let de = (e_max - e_min) / n_steps as f64;
        let t_sun = 5800.0;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..=n_steps {
            let e_ev = e_min + i as f64 * de;
            let omega = ev_to_omega(e_ev);
            let x = e_ev / (K_B_EV * t_sun);
            if x > 500.0 {
                continue;
            }
            let weight = e_ev.powi(3) / (x.exp() - 1.0);
            let absorptance = self.emissivity(omega);
            let w = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
            num += w * absorptance * weight;
            den += w * weight;
        }
        if den < 1e-30 {
            return 0.0;
        }
        num / den
    }

    /// Solar-weighted reflectance: `1 - solar_absorptance` (opaque slab).
    pub fn solar_reflectance(&self, n_steps: usize) -> f64 {
        1.0 - self.solar_absorptance(n_steps)
    }

    /// Quarter-wave antireflection coating thickness. Ideal coating index
    /// `n_coat = sqrt(n_sub)`; thickness `d = lambda / (4 * n_coat)`.
    /// Returns the thickness in meters at the given frequency.
    pub fn antireflection_thickness(&self, omega: f64) -> f64 {
        let n_sub = self.refractive_index(omega).re;
        let n_coating = n_sub.abs().sqrt();
        let lambda = 2.0 * std::f64::consts::PI * C / omega;
        lambda / (4.0 * n_coating)
    }

    /// Wien displacement law for the peak of a blackbody:
    /// `omega_peak = alpha * k_B * T / hbar` with `alpha ~ 2.8214` (root of
    /// `x = 3*(1 - exp(-x))`). Returns rad/s.
    pub fn wien_peak_omega(temperature_k: f64) -> f64 {
        let alpha = 2.821_439_372;
        let hbar_j = HBAR_EV_S * 1.602_176_634e-19;
        let k_b_j = K_B_EV * 1.602_176_634e-19;
        alpha * k_b_j * temperature_k / hbar_j
    }

    /// Wien peak energy in eV for a blackbody at temperature `T`.
    pub fn wien_peak_ev(temperature_k: f64) -> f64 {
        2.821_439_372 * K_B_EV * temperature_k
    }

    /// Luminous reflectance: CIE photopic luminosity-weighted reflectance.
    /// Approximates the V(lambda) curve as a Gaussian centered at 2.23 eV
    /// (555 nm) with sigma 0.34 eV (~0.8 eV FWHM); integrates over
    /// 1.59 - 3.26 eV (380 - 780 nm).
    pub fn luminous_reflectance(&self, n_steps: usize) -> f64 {
        if n_steps < 2 {
            return 0.0;
        }
        let e_min = 1.59;
        let e_max = 3.26;
        let de = (e_max - e_min) / n_steps as f64;
        let center_ev = 2.23;
        let sigma = 0.34;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..=n_steps {
            let e_ev = e_min + i as f64 * de;
            let omega = ev_to_omega(e_ev);
            let v = (-0.5 * ((e_ev - center_ev) / sigma).powi(2)).exp();
            let r = self.reflectivity_normal(omega);
            let w = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
            num += w * r * v;
            den += w * v;
        }
        if den < 1e-30 {
            return 0.0;
        }
        num / den
    }

    /// Selective-emitter efficiency for thermophotovoltaics:
    /// `eta = integral[e(omega) * B(omega,T) d_omega, omega > omega_gap]
    ///       / integral[e(omega) * B(omega,T) d_omega, all omega]`.
    /// Measures the fraction of thermal emission above the PV bandgap
    /// (useful) versus below (waste).
    pub fn selective_emitter_efficiency(
        &self,
        temperature_k: f64,
        omega_gap: f64,
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
        let mut above_gap = 0.0;
        let mut total = 0.0;
        for i in 0..=n_steps {
            let omega = omega_min + i as f64 * d_omega;
            let x = hbar * omega / (k_b * temperature_k);
            if !(1e-30..=500.0).contains(&x) {
                continue;
            }
            let planck = omega.powi(3) / (x.exp() - 1.0);
            let e = self.emissivity(omega);
            let w = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
            let contribution = w * e * planck;
            total += contribution;
            if omega >= omega_gap {
                above_gap += contribution;
            }
        }
        if total < 1e-30 {
            return 0.0;
        }
        above_gap / total
    }
}
