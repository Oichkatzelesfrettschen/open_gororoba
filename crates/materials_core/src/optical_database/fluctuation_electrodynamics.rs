//! Fluctuation electrodynamics + thermal/quantum noise methods on
//! `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 17b). Seven methods covering the FDT + near-field thermal +
//! Casimir-Lifshitz physics:
//!
//! - `fluctuation_dissipation_spectral`: `(2*hbar*omega/pi)*Im[eps]*(n_BE+1/2)`.
//! - `thermal_noise_power_density`: `hbar*omega * Im[eps] * coth(x/2)`.
//! - `zero_point_energy_density`: `E_0 = hbar*omega/2`.
//! - `spectral_energy_density`: Planck `u(omega, T)`.
//! - `near_field_thermal_emission`: evanescent enhancement at `d << lambda`.
//! - `photon_tunneling_probability`: Fresnel * exp(-2*kappa*d).
//! - `fluctuation_induced_force_integrand`: Lifshitz `r_TM^2 * exp(-2*xi*d/c)`.

use super::{C, DrudeLorentzParams, HBAR_EV_S, K_B_EV};

impl DrudeLorentzParams {
    /// Fluctuation-dissipation spectral density
    /// `S(omega,T) = (2*hbar*omega/pi) * |Im[eps]| * (n_BE + 1/2)`
    /// in eV^2/(rad/s) units.
    pub fn fluctuation_dissipation_spectral(&self, omega: f64, temperature_k: f64) -> f64 {
        let eps_im = self.epsilon(omega).im;
        let hbar_omega_ev = HBAR_EV_S * omega;
        let n_be = if temperature_k > 0.0 && hbar_omega_ev > 0.0 {
            let x = hbar_omega_ev / (K_B_EV * temperature_k);
            if x > 500.0 {
                0.0
            } else {
                1.0 / (x.exp() - 1.0)
            }
        } else {
            0.0
        };
        (2.0 * hbar_omega_ev / std::f64::consts::PI) * eps_im.abs() * (n_be + 0.5)
    }

    /// Thermal noise power density
    /// `P(omega) = hbar*omega * |Im[eps]| * coth(hbar*omega / (2*k_B*T))`.
    pub fn thermal_noise_power_density(&self, omega: f64, temperature_k: f64) -> f64 {
        let eps_im = self.epsilon(omega).im;
        let hbar_omega_ev = HBAR_EV_S * omega;
        let coth = if temperature_k > 0.0 && hbar_omega_ev > 0.0 {
            let x = hbar_omega_ev / (2.0 * K_B_EV * temperature_k);
            if x > 500.0 { 1.0 } else { x.cosh() / x.sinh() }
        } else {
            1.0
        };
        hbar_omega_ev * eps_im.abs() * coth
    }

    /// Zero-point energy density per mode `E_0 = hbar*omega / 2`.
    pub fn zero_point_energy_density(omega: f64) -> f64 {
        HBAR_EV_S * omega / 2.0
    }

    /// Planck spectral energy density
    /// `u(omega, T) = (hbar*omega^3/(pi^2 c^3)) * n_BE(omega, T)`.
    pub fn spectral_energy_density(omega: f64, temperature_k: f64) -> f64 {
        let hbar_omega_ev = HBAR_EV_S * omega;
        let n_be = if temperature_k > 0.0 && hbar_omega_ev > 0.0 {
            let x = hbar_omega_ev / (K_B_EV * temperature_k);
            if x > 500.0 {
                0.0
            } else {
                1.0 / (x.exp() - 1.0)
            }
        } else {
            0.0
        };
        hbar_omega_ev * omega * omega * n_be
            / (std::f64::consts::PI * std::f64::consts::PI * C * C * C)
    }

    /// Near-field thermal emission factor: evanescent modes scale as
    /// `1/(k*d)^2` for `d << lambda`. Returns the dimensionless enhancement
    /// `|Im[eps]| * (n_BE + 1/2) * (1/(kd)^2)` (or 1 when far-field).
    pub fn near_field_thermal_emission(
        &self,
        omega: f64,
        distance_m: f64,
        temperature_k: f64,
    ) -> f64 {
        let eps_im = self.epsilon(omega).im;
        let k = omega / C;
        let kd = k * distance_m;
        let n_be = if temperature_k > 0.0 {
            let x = HBAR_EV_S * omega / (K_B_EV * temperature_k);
            if x > 500.0 {
                0.0
            } else {
                1.0 / (x.exp() - 1.0)
            }
        } else {
            0.0
        };
        let evanescent = if kd > 1e-10 && kd < 1.0 {
            1.0 / (kd * kd)
        } else {
            1.0
        };
        eps_im.abs() * (n_be + 0.5) * evanescent
    }

    /// Photon tunneling probability through a vacuum gap of width `d`:
    /// `T ~ exp(-2*kappa*d)` modulated by Fresnel transmission at the
    /// air-material interface.
    pub fn photon_tunneling_probability(&self, omega: f64, kappa_m: f64) -> f64 {
        if kappa_m <= 0.0 {
            return 1.0;
        }
        let eps = self.epsilon(omega);
        let r_fresnel = ((eps.sqrt() - 1.0) / (eps.sqrt() + 1.0)).norm_sqr();
        let transmission = 1.0 - r_fresnel;
        transmission * (-2.0 * kappa_m).exp()
    }

    /// Casimir-Lifshitz force integrand at imaginary frequency `xi`:
    /// `r_TM^2 * exp(-2*xi*d/c)` for two identical half-spaces.
    pub fn fluctuation_induced_force_integrand(&self, xi: f64, distance_m: f64) -> f64 {
        let eps_xi = self.epsilon_imaginary(xi);
        let r_tm = (eps_xi - 1.0) / (eps_xi + 1.0);
        let decay = (-2.0 * xi * distance_m / C).exp();
        r_tm * r_tm * decay
    }
}
