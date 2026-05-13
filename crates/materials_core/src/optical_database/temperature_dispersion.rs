//! Temperature-dependent broadening + dispersion-engineering + nonlinear
//! optics estimates on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split.
//! Nine methods cover three tightly-coupled physics threads:
//!
//! Temperature broadening:
//! - `at_temperature`: Bose-Einstein coth() broadening of phonon oscillators
//!   + Bloch-Gruneisen T^2 correction for Drude damping.
//! - `optical_effective_mass`: `m* = n*e^2/(eps_0*omega_p^2)` from carrier
//!   density and Drude plasma frequency.
//!
//! Dispersion engineering (Part 9):
//! - `gvd_beta2`: group velocity dispersion in s^2/m via finite differences.
//! - `gvd_fs2_per_mm`: same in ultrafast-optics units.
//! - `dispersion_regime`: classifier (-1 anomalous, 0 zero, +1 normal).
//! - `zero_dispersion_omega`: frequency where beta_2 crosses zero.
//!
//! Nonlinear optics (Part 9 cont.):
//! - `chi3_miller_estimate`: third-order susceptibility via Miller's rule.
//! - `kerr_n2_estimate`: nonlinear refractive index `n_2`.
//! - `beta_tpa_estimate`: Sheik-Bahae two-photon-absorption coefficient.
//!
//! References: Sheik-Bahae et al. (1991) IEEE J. Quantum Electron. 27, 1296;
//! Miller (1964) Appl. Phys. Lett. 5, 17; Boyd (2008) "Nonlinear Optics" 3rd ed.

use super::{C, DrudeLorentzParams, DrudeParams, EPS_0, EV_TO_RADS, E_CHARGE, K_B_EV, LorentzOscillator, M_E_KG};

impl DrudeLorentzParams {
    /// Return a new `DrudeLorentzParams` with thermally broadened
    /// oscillators. Phonon damping uses the Bose-Einstein occupation
    /// `gamma_j(T) = gamma_j(0) * coth(hbar*omega_0j / (2*k_B*T))`.
    /// Drude damping uses a Bloch-Gruneisen `T^2` correction with the
    /// user-supplied Debye temperature.
    pub fn at_temperature(&self, temperature_k: f64, debye_t_k: Option<f64>) -> Self {
        let broadened_oscs: Vec<LorentzOscillator> = self
            .oscillators
            .iter()
            .map(|osc| {
                let x = osc.omega_0_ev / (2.0 * K_B_EV * temperature_k);
                let coth = if x > 20.0 {
                    1.0
                } else if x < 0.01 {
                    1.0 / x
                } else {
                    (x.exp() + (-x).exp()) / (x.exp() - (-x).exp())
                };
                LorentzOscillator {
                    strength: osc.strength,
                    omega_0_ev: osc.omega_0_ev,
                    gamma_ev: osc.gamma_ev * coth,
                }
            })
            .collect();

        let broadened_drude = self.drude.map(|d| {
            let t_ratio_sq = if let Some(t_d) = debye_t_k {
                (temperature_k / t_d).powi(2)
            } else {
                0.0
            };
            DrudeParams {
                omega_p_ev: d.omega_p_ev,
                gamma_ev: d.gamma_ev * (1.0 + t_ratio_sq),
                eps_inf: d.eps_inf,
            }
        });

        DrudeLorentzParams {
            drude: broadened_drude,
            oscillators: broadened_oscs,
            eps_inf: self.eps_inf,
            extended_drude: self.extended_drude.clone(),
        }
    }

    /// Optical effective mass `m*/m_e = n*e^2 / (eps_0 * omega_p^2 * m_e)`
    /// in units of free-electron mass. Requires the carrier density `n`
    /// (m^-3). Returns `None` for non-metallic materials.
    pub fn optical_effective_mass(&self, carrier_density: f64) -> Option<f64> {
        let omega_p = if let Some(ext) = &self.extended_drude {
            ext.omega_p_ev * EV_TO_RADS
        } else if let Some(drude) = &self.drude {
            drude.omega_p_ev * EV_TO_RADS
        } else {
            return None;
        };
        let m_star = carrier_density * E_CHARGE * E_CHARGE / (EPS_0 * omega_p * omega_p);
        Some(m_star / M_E_KG)
    }

    /// Group velocity dispersion `beta_2 = d^2 k / d_omega^2 = (1/c) * d(n_g)/d_omega`
    /// in s^2/m. Positive = normal dispersion (red faster); negative =
    /// anomalous. Computed via finite differences on the group index.
    pub fn gvd_beta2(&self, omega: f64) -> f64 {
        let delta = omega * 1e-4;
        let ng_plus = self.group_refractive_index(omega + delta);
        let ng_minus = self.group_refractive_index(omega - delta);
        let dng_domega = (ng_plus - ng_minus) / (2.0 * delta);
        dng_domega / C
    }

    /// GVD in fs^2/mm (ultrafast optics convention). Typical: silica at
    /// 800 nm `~ +36 fs^2/mm` (normal); at 1550 nm `~ -26 fs^2/mm` (anomalous).
    pub fn gvd_fs2_per_mm(&self, omega: f64) -> f64 {
        self.gvd_beta2(omega) * 1e27
    }

    /// Dispersion classifier: `+1` normal (`beta_2 > 0`), `-1` anomalous,
    /// `0` if `|beta_2| < 1e-30` (effectively zero dispersion).
    pub fn dispersion_regime(&self, omega: f64) -> i32 {
        let beta2 = self.gvd_beta2(omega);
        if beta2 > 1e-30 {
            1
        } else if beta2 < -1e-30 {
            -1
        } else {
            0
        }
    }

    /// Zero-dispersion frequency finder in rad/s. Scans `[omega_min,
    /// omega_max]` for the first `beta_2` zero-crossing. For silica this
    /// lies near 1.27 um wavelength (1.49e15 rad/s).
    pub fn zero_dispersion_omega(&self, omega_min: f64, omega_max: f64) -> Option<f64> {
        let steps: usize = 2000;
        let domega = (omega_max - omega_min) / steps as f64;
        let mut prev_beta2 = self.gvd_beta2(omega_min);
        for i in 1..=steps {
            let omega = omega_min + i as f64 * domega;
            let beta2 = self.gvd_beta2(omega);
            if (prev_beta2 > 0.0 && beta2 <= 0.0) || (prev_beta2 < 0.0 && beta2 >= 0.0) {
                let frac = prev_beta2.abs() / (prev_beta2.abs() + beta2.abs());
                return Some(omega - domega + frac * domega);
            }
            prev_beta2 = beta2;
        }
        None
    }

    /// Third-order nonlinear susceptibility `chi^(3)(omega)` estimate via
    /// Miller's rule generalization: `chi^(3) ~ delta * [chi^(1)]^4` with
    /// `delta = 4.52e-24 m^2/V^2`. Semi-empirical scaling; actual chi^(3)
    /// may differ by an order of magnitude near resonances. Returns the
    /// positive magnitude. Reference: Miller (1964) Appl. Phys. Lett. 5, 17.
    pub fn chi3_miller_estimate(&self, omega: f64) -> f64 {
        let miller_delta: f64 = 4.52e-24;
        let chi1 = self.epsilon(omega) - 1.0;
        let chi1_sq = chi1.norm_sqr();
        miller_delta * chi1_sq * chi1_sq
    }

    /// Kerr nonlinear refractive index `n_2 = 3 * chi^(3) / (4 * eps_0 * c * n^2)`
    /// in m^2/W. Typical: silica `~ 2.2e-20 m^2/W`; CS2 `~ 3e-18 m^2/W`.
    pub fn kerr_n2_estimate(&self, omega: f64) -> f64 {
        let chi3 = self.chi3_miller_estimate(omega);
        let n = self.refractive_index(omega).re;
        if n < 1e-10 {
            return 0.0;
        }
        3.0 * chi3 / (4.0 * EPS_0 * C * n * n)
    }

    /// Two-photon absorption coefficient `beta_TPA` in m/W (Sheik-Bahae).
    /// `beta_TPA = K * sqrt(E_p) * F_2(x) / (n^2 * E_g^3)` with `E_p = 21 eV`
    /// (Kane energy) and `F_2(x) = (2x-1)^(3/2) / (2x)^5` for `x = 2hv/E_g > 0.5`.
    /// Returns `None` if no Tauc gap or below two-photon threshold.
    /// Reference: Sheik-Bahae et al. (1991) IEEE JQE 27, 1296.
    pub fn beta_tpa_estimate(&self, omega: f64) -> Option<f64> {
        let e_g = self.tauc_gap_ev(2.0)?;
        let hv = omega / EV_TO_RADS;
        let x = 2.0 * hv / e_g;
        if x <= 0.5 {
            return None;
        }
        let n = self.refractive_index(omega).re;
        if n < 1e-10 {
            return None;
        }
        let e_p: f64 = 21.0;
        let f2 = (2.0 * x - 1.0).powf(1.5) / (2.0 * x).powi(5);
        let k_si: f64 = 1.94e-8;
        let beta = k_si * e_p.sqrt() * f2 / (n * n * e_g.powi(3));
        Some(beta)
    }
}
