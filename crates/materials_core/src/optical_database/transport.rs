//! DC + AC carrier transport methods on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split.
//! Seven methods covering free-carrier transport, magneto-optics, and the
//! Drude-derived spectral / scattering observables:
//!
//! - `drude_weight`: integrated spectral weight under the Drude peak.
//! - `carrier_mobility`: `mu = e / (m* * gamma)`.
//! - `plasma_frequency_from_density`: inverse of optical effective mass.
//! - `voigt_eps_xy`: off-diagonal magneto-optical tensor element for MOKE.
//! - `faraday_rotation`: rotation per unit length from Voigt eps_xy.
//! - `dc_resistivity`: `rho = gamma / (eps_0 * omega_p^2)`.
//! - `scattering_time`: `tau = 1 / gamma`.
//!
//! References: Dressel & Gruner (2002) "Electrodynamics of Solids"
//! Chs. 5-6; Wegener (2005) "Extreme Nonlinear Optics" Ch. 4 (MOKE).

use std::f64::consts::PI;

use num_complex::Complex64;

use super::{C, DrudeLorentzParams, E_CHARGE, EPS_0, EV_TO_RADS, M_E_KG};

impl DrudeLorentzParams {
    /// Drude weight `D = (pi/2) * omega_p^2 * eps_0` in S*rad/(m*s).
    /// Equals `pi * n * e^2 / (2 * m*)`; the integrated spectral weight
    /// under the Drude peak. Returns `None` for non-metallic materials.
    pub fn drude_weight(&self) -> Option<f64> {
        let omega_p = if let Some(ext) = &self.extended_drude {
            ext.omega_p_ev * EV_TO_RADS
        } else if let Some(drude) = &self.drude {
            drude.omega_p_ev * EV_TO_RADS
        } else {
            return None;
        };
        Some(PI / 2.0 * omega_p * omega_p * EPS_0)
    }

    /// Carrier mobility `mu = e / (m* * gamma)` in m^2/(V*s) from Drude
    /// parameters. For gold (n ~ 5.9e28 m^-3), mu ~ 0.004 m^2/(V*s) at room T.
    /// Returns `None` if no Drude term or if the carrier density gives an
    /// unphysical effective mass.
    pub fn carrier_mobility(&self, carrier_density: f64) -> Option<f64> {
        let gamma_ev = if let Some(ext) = &self.extended_drude {
            ext.scattering.gamma_at_ev(0.0)
        } else if let Some(drude) = &self.drude {
            drude.gamma_ev
        } else {
            return None;
        };
        let m_star = self.optical_effective_mass(carrier_density)?;
        let m_star_kg = m_star * M_E_KG;
        let gamma = gamma_ev * EV_TO_RADS;
        if gamma < 1e-30 {
            return None;
        }
        Some(E_CHARGE / (m_star_kg * gamma))
    }

    /// Plasma frequency from carrier density and effective mass ratio:
    /// `omega_p = sqrt(n * e^2 / (eps_0 * m*))` in rad/s. The inverse
    /// of `optical_effective_mass()`: given `n` and `m*`, compute `omega_p`.
    pub fn plasma_frequency_from_density(carrier_density: f64, m_star_ratio: f64) -> f64 {
        let m_star = m_star_ratio * M_E_KG;
        (carrier_density * E_CHARGE * E_CHARGE / (EPS_0 * m_star)).sqrt()
    }

    /// Off-diagonal Voigt dielectric tensor element `eps_xy` for MOKE.
    /// `eps_xy(omega) = i * omega_c * omega_p^2 / (omega * (-omega^2 + i*gamma*omega))`
    /// with cyclotron frequency `omega_c = e*B / m*`. The lowest-order
    /// magneto-optical response (free-electron). Returns `None` if no Drude
    /// term (no free carriers to precess).
    pub fn voigt_eps_xy(
        &self,
        omega: f64,
        b_field: f64,
        carrier_density: f64,
    ) -> Option<Complex64> {
        let (omega_p_ev, gamma_ev) = if let Some(ext) = &self.extended_drude {
            (ext.omega_p_ev, ext.scattering.gamma_at_ev(0.0))
        } else if let Some(drude) = &self.drude {
            (drude.omega_p_ev, drude.gamma_ev)
        } else {
            return None;
        };
        let m_star = self.optical_effective_mass(carrier_density)?;
        let m_star_kg = m_star * M_E_KG;
        let omega_c = E_CHARGE * b_field / m_star_kg;
        let omega_p = omega_p_ev * EV_TO_RADS;
        let gamma = gamma_ev * EV_TO_RADS;
        let denom = Complex64::new(-omega * omega, gamma * omega);
        let numerator = Complex64::new(0.0, omega_c * omega_p * omega_p);
        Some(numerator / (omega * denom))
    }

    /// Faraday rotation per unit length in rad/m:
    /// `theta_F = omega * Re[eps_xy] / (2 * n * c)`. Returns `None` if
    /// no Drude term.
    pub fn faraday_rotation(&self, omega: f64, b_field: f64, carrier_density: f64) -> Option<f64> {
        let eps_xy = self.voigt_eps_xy(omega, b_field, carrier_density)?;
        let n = self.refractive_index(omega).re;
        if n < 1e-10 {
            return None;
        }
        Some(omega * eps_xy.re / (2.0 * n * C))
    }

    /// DC resistivity in Ohm*m: `rho = gamma / (eps_0 * omega_p^2)`. For
    /// gold: `rho ~ 2.2e-8 Ohm*m` at room temperature. Returns `None`
    /// if no Drude term.
    pub fn dc_resistivity(&self) -> Option<f64> {
        let (omega_p_ev, gamma_ev) = if let Some(ext) = &self.extended_drude {
            (ext.omega_p_ev, ext.scattering.gamma_at_ev(0.0))
        } else if let Some(drude) = &self.drude {
            (drude.omega_p_ev, drude.gamma_ev)
        } else {
            return None;
        };
        let omega_p = omega_p_ev * EV_TO_RADS;
        let gamma = gamma_ev * EV_TO_RADS;
        Some(gamma / (EPS_0 * omega_p * omega_p))
    }

    /// Scattering time (Drude relaxation time) `tau = 1/gamma` in seconds.
    /// For gold at 300 K: `tau ~ 9.5 fs`. Returns `None` if no Drude term.
    pub fn scattering_time(&self) -> Option<f64> {
        let gamma_ev = if let Some(ext) = &self.extended_drude {
            ext.scattering.gamma_at_ev(0.0)
        } else if let Some(drude) = &self.drude {
            drude.gamma_ev
        } else {
            return None;
        };
        let gamma = gamma_ev * EV_TO_RADS;
        if gamma < 1e-30 {
            return None;
        }
        Some(1.0 / gamma)
    }
}
