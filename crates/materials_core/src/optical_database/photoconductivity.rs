//! Photoconductivity and transient carrier dynamics on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 16e). Five methods for pump-probe photo-injected-carrier physics:
//!
//! - `plasma_frequency_shift`: `Delta_omega_p` from injected `delta_n`.
//! - `photo_induced_absorption`: `Delta_alpha` from injected-carrier Drude.
//! - `transient_reflectivity_change`: `Delta_R/R` via modified-Drude finite diff.
//! - `drude_smith_mobility`: `mu * (1 + c)` Drude-Smith parametrization.
//! - `carrier_recombination_time`: `tau_rec = delta_n / G`.

use num_complex::Complex64;

use super::{C, DrudeLorentzParams, E_CHARGE, EPS_0, EV_TO_RADS, M_E_KG};

impl DrudeLorentzParams {
    /// Plasma frequency shift `Delta_omega_p` in eV from optically injected
    /// carriers: `omega_p_new = sqrt(omega_p^2 + delta_n*e^2/(eps_0*m*))`.
    /// Returns `None` if no Drude component.
    pub fn plasma_frequency_shift(&self, delta_n: f64, m_star_ratio: f64) -> Option<f64> {
        let drude = self.drude.as_ref()?;
        let omega_p = drude.omega_p_ev * EV_TO_RADS;
        let m_star = m_star_ratio * M_E_KG;
        let delta_wp_sq = delta_n * E_CHARGE * E_CHARGE / (EPS_0 * m_star);
        let new_omega_p = (omega_p * omega_p + delta_wp_sq).sqrt();
        Some((new_omega_p - omega_p) / EV_TO_RADS)
    }

    /// Photo-induced absorption change `Delta_alpha = omega * Im[delta_eps]
    /// / (c * Re[n])` from a transient carrier population. Returns `None`
    /// if no Drude component.
    pub fn photo_induced_absorption(
        &self,
        omega: f64,
        delta_n: f64,
        m_star_ratio: f64,
    ) -> Option<f64> {
        let m_star = m_star_ratio * M_E_KG;
        let delta_wp_sq = delta_n * E_CHARGE * E_CHARGE / (EPS_0 * m_star);
        let gamma = self.drude.as_ref()?.gamma_ev * EV_TO_RADS;
        let denom = Complex64::new(-(omega * omega) + gamma * gamma, omega * gamma);
        let delta_eps = Complex64::new(-delta_wp_sq, 0.0)
            / Complex64::new(omega * omega + gamma * gamma, 0.0)
            * Complex64::new(1.0, gamma / omega);
        let n_re = self.refractive_index(omega).re;
        if n_re < 1e-10 {
            return None;
        }
        let _ = denom;
        Some(omega * delta_eps.im.abs() / (C * n_re))
    }

    /// Transient reflectivity change `Delta_R/R` from pump-induced
    /// shifted plasma frequency. Uses a finite-difference between baseline
    /// and shifted-Drude reflectivity. Returns `None` if no Drude or
    /// baseline `R < 1e-15`.
    pub fn transient_reflectivity_change(
        &self,
        omega: f64,
        delta_n: f64,
        m_star_ratio: f64,
    ) -> Option<f64> {
        let r0 = self.reflectivity_normal(omega);
        if r0 < 1e-15 {
            return None;
        }
        let drude = self.drude.as_ref()?;
        let omega_p = drude.omega_p_ev * EV_TO_RADS;
        let m_star = m_star_ratio * M_E_KG;
        let delta_wp_sq = delta_n * E_CHARGE * E_CHARGE / (EPS_0 * m_star);
        let new_omega_p = (omega_p * omega_p + delta_wp_sq).sqrt();
        let mut modified = self.clone();
        if let Some(ref mut d) = modified.drude {
            d.omega_p_ev = new_omega_p / EV_TO_RADS;
        }
        let r1 = modified.reflectivity_normal(omega);
        Some((r1 - r0) / r0)
    }

    /// Drude-Smith mobility `mu_DS = mu_Drude * (1 + c)` with persistence
    /// parameter `c in [-1, 0]`: `c = 0` standard ballistic Drude;
    /// `c = -1` complete backscattering. Returns `None` if no Drude.
    pub fn drude_smith_mobility(&self, c_parameter: f64, carrier_density: f64) -> Option<f64> {
        let drude = self.drude.as_ref()?;
        let tau = 1.0 / (drude.gamma_ev * EV_TO_RADS);
        let mu_drude = E_CHARGE * tau / (carrier_density * M_E_KG);
        Some(mu_drude * (1.0 + c_parameter))
    }

    /// Effective carrier recombination time `tau_rec = delta_n / G`.
    /// Returns infinity if `G = 0`.
    pub fn carrier_recombination_time(delta_n: f64, generation_rate: f64) -> f64 {
        if generation_rate.abs() < 1e-30 {
            return f64::INFINITY;
        }
        delta_n / generation_rate
    }
}
