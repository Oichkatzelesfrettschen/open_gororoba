//! Linear and quadratic electro-optic, photoelastic, and acousto-optic
//! response methods on `DrudeLorentzParams`.
//!
//! Extracted from the monolithic `optical_database.rs` impl block as part of
//! the #138 PH-MOD split. Rust permits the `impl` block to be split across
//! multiple files within the same crate; calls to `params.pockels_delta_n(...)`
//! and the other 5 methods in this module continue to resolve transparently
//! after relocation -- no signature or call-site changes.
//!
//! All 6 methods depend on `self.refractive_index(omega)` from the parent
//! impl block plus the parent-module constants `C`, `HBAR_EV_S`,
//! `E_CHARGE`, `M_E_KG`. The Franz-Keldysh absorption uses the textbook
//! Wentzel-Kramers-Brillouin tunneling exponent (Yu & Cardona 2010
//! "Fundamentals of Semiconductors" Sec. 6.3.3).
//!
//! Reference: Yariv & Yeh (2007) "Photonics", chapters 7 (Pockels) and
//! 9 (acousto-optic).

use super::{C, DrudeLorentzParams, E_CHARGE, HBAR_EV_S, M_E_KG};

impl DrudeLorentzParams {
    /// Pockels (linear electro-optic) refractive-index change:
    /// `delta_n = -0.5 * n^3 * r * E`.
    /// `r_eo` is the electro-optic coefficient in m/V (typical 1e-12..30e-12).
    pub fn pockels_delta_n(&self, omega: f64, electric_field_v_m: f64, r_eo: f64) -> f64 {
        let n = self.refractive_index(omega).re;
        -0.5 * n.powi(3) * r_eo * electric_field_v_m
    }

    /// Kerr (quadratic electro-optic) refractive-index change:
    /// `delta_n = -0.5 * n^3 * s * E^2`.
    /// `s_eo` is the Kerr coefficient in m^2/V^2 (typical 1e-20..1e-18).
    pub fn kerr_electro_optic(&self, omega: f64, electric_field_v_m: f64, s_eo: f64) -> f64 {
        let n = self.refractive_index(omega).re;
        -0.5 * n.powi(3) * s_eo * electric_field_v_m * electric_field_v_m
    }

    /// Half-wave voltage for a Pockels modulator:
    /// `V_pi = lambda / (2 * n^3 * r * L)`.
    /// Returns the modulator's half-wave drive voltage in Volts.
    pub fn half_wave_voltage(&self, omega: f64, r_eo: f64, crystal_length_m: f64) -> f64 {
        let n = self.refractive_index(omega).re;
        let lambda = 2.0 * std::f64::consts::PI * C / omega;
        lambda / (2.0 * n.powi(3) * r_eo * crystal_length_m)
    }

    /// Franz-Keldysh sub-gap absorption: field-enhanced tunneling absorption
    /// below the band edge. Returns alpha in 1/m at photon energy hbar*omega
    /// below the gap, with `gap_ev` in eV and an applied electric field.
    /// `alpha ~ exp(-4*sqrt(2*m_eff) * (Eg - hbar*omega)^(3/2) / (3*e*E*hbar))`.
    pub fn franz_keldysh_absorption(
        &self,
        omega: f64,
        electric_field_v_m: f64,
        gap_ev: f64,
    ) -> f64 {
        let hbar_j_s = HBAR_EV_S * E_CHARGE; // J*s
        let hbar_omega_ev = HBAR_EV_S * omega;
        if hbar_omega_ev >= gap_ev || electric_field_v_m <= 0.0 {
            return 0.0;
        }
        let delta_e_j = (gap_ev - hbar_omega_ev) * E_CHARGE; // Joules
        let m_star = 0.1 * M_E_KG; // generic effective-mass estimate
        let exponent = -4.0_f64 * (2.0_f64 * m_star).sqrt() * delta_e_j.powf(1.5)
            / (3.0 * E_CHARGE * electric_field_v_m * hbar_j_s);
        1e6 * exponent.exp()
    }

    /// Photoelastic refractive-index change: `delta_n = -0.5 * n^3 * p * S`.
    /// `p_ij` is the (dimensionless) photoelastic coefficient (typical 0.1..0.3).
    pub fn photoelastic_delta_n(&self, omega: f64, strain: f64, p_ij: f64) -> f64 {
        let n = self.refractive_index(omega).re;
        -0.5 * n.powi(3) * p_ij * strain
    }

    /// Acousto-optic figure of merit:
    /// `M2 = n^6 * p^2 / (rho * v^3)`.
    /// Units: M2 returned in s^3/kg.
    pub fn acoustooptic_figure_of_merit(
        &self,
        omega: f64,
        p_ij: f64,
        v_sound: f64,
        density: f64,
    ) -> f64 {
        let n = self.refractive_index(omega).re;
        n.powi(6) * p_ij * p_ij / (density * v_sound.powi(3))
    }
}
