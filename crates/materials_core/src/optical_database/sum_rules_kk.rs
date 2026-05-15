//! Sum rules + Kramers-Kronig validation methods on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split.
//! These 8 methods establish the consistency of the modelled dielectric
//! response with the physical sum rules and the causality requirement.
//!
//! Methods:
//! - `n_eff`: partial f-sum (effective electrons per unit volume below cutoff).
//! - `f_sum_ratio`: Drude-normalised f-sum convergence check.
//! - `plasmon_energy_ev`: bulk plasmon from loss-function peak.
//! - `loss_spectral_weight`: omega-weighted integral of -Im[1/eps].
//! - `screened_plasma_ev`: first Re[eps] = 0 crossing.
//! - `intraband_weight`: Drude contribution to f-sum (pi/2 * omega_p^2 * eps_0).
//! - `interband_weight`: Lorentz oscillator contribution.
//! - `kramers_kronig_error`: RMS relative error of KK-reconstructed eps_1
//!   compared to model eps_1; expected to be small for any causal model.
//!
//! References: Wooten (1972) "Optical Properties of Solids" Ch. 7
//! (sum rules); Lucarini et al. (2005) "Kramers-Kronig Relations in
//! Optical Materials Research" (Springer).

use std::f64::consts::PI;

use super::{DrudeLorentzParams, E_CHARGE, EPS_0, EV_TO_RADS, M_E_KG};

impl DrudeLorentzParams {
    /// f-sum effective number of electrons:
    /// `N_eff = (2 * m_e * eps_0) / (pi * e^2) * integral_0^omega_c sigma_1 d_omega`.
    /// Counts the effective number of electrons per unit volume contributing
    /// to optical transitions below `cutoff_ev`. Trapezoidal integration with
    /// `n_steps` points from `0.001 eV` to `cutoff_ev`.
    pub fn n_eff(&self, cutoff_ev: f64, n_steps: usize) -> f64 {
        let prefactor = 2.0 * M_E_KG / (PI * E_CHARGE * E_CHARGE);
        let dw = cutoff_ev * EV_TO_RADS / n_steps as f64;
        let mut integral = 0.0;
        let mut prev_sigma = 0.0;
        for i in 1..=n_steps {
            let omega = i as f64 * dw;
            let sigma = self.optical_conductivity_re(omega);
            integral += 0.5 * (prev_sigma + sigma) * dw;
            prev_sigma = sigma;
        }
        prefactor * integral
    }

    /// Verify the f-sum rule against the known Drude plasma frequency.
    /// Returns `Some((N_eff_computed, N_eff_drude))` where the ratio
    /// approaches 1.0 as the cutoff -> infinity. Returns `None` for
    /// materials without a Drude term.
    pub fn f_sum_ratio(&self, cutoff_ev: f64, n_steps: usize) -> Option<(f64, f64)> {
        let drude = self.drude.as_ref()?;
        let omega_p = drude.omega_p_ev * EV_TO_RADS;
        let n_drude = EPS_0 * M_E_KG * omega_p * omega_p / (E_CHARGE * E_CHARGE);
        let n_eff = self.n_eff(cutoff_ev, n_steps);
        Some((n_eff, n_drude))
    }

    /// Find the bulk plasmon energy from the loss-function peak.
    /// Scans from `scan_min_ev` to `scan_max_ev` in 0.01 eV steps to find
    /// the maximum of `-Im[1/eps]`. Returns the energy in eV.
    pub fn plasmon_energy_ev(&self, scan_min_ev: f64, scan_max_ev: f64) -> f64 {
        let steps = ((scan_max_ev - scan_min_ev) / 0.01) as usize;
        let mut max_loss = 0.0_f64;
        let mut max_ev = scan_min_ev;
        for i in 0..=steps {
            let ev = scan_min_ev + i as f64 * 0.01;
            let omega = ev * EV_TO_RADS;
            let loss = self.loss_function(omega);
            if loss > max_loss {
                max_loss = loss;
                max_ev = ev;
            }
        }
        max_ev
    }

    /// Loss-function spectral weight (partial sum rule).
    /// `integral_0^omega_c omega * (-Im[1/eps]) d_omega` returned as
    /// `omega_p_eff^2 = integral / (pi/2)`. Approaches `omega_p^2` as the
    /// cutoff -> infinity for a Drude metal.
    pub fn loss_spectral_weight(&self, cutoff_ev: f64, n_steps: usize) -> f64 {
        let dw = cutoff_ev * EV_TO_RADS / n_steps as f64;
        let mut integral = 0.0;
        let mut prev_val = 0.0;
        for i in 1..=n_steps {
            let omega = i as f64 * dw;
            let val = omega * self.loss_function(omega);
            integral += 0.5 * (prev_val + val) * dw;
            prev_val = val;
        }
        integral / (PI / 2.0)
    }

    /// Screened plasma frequency from the first `Re[eps] = 0` crossing.
    /// Returns the energy (eV) of the crossing or `None` if no crossing
    /// is found in the scan window.
    pub fn screened_plasma_ev(&self, scan_min_ev: f64, scan_max_ev: f64) -> Option<f64> {
        let steps = ((scan_max_ev - scan_min_ev) / 0.01) as usize;
        let mut prev_re = self.epsilon(scan_min_ev * EV_TO_RADS).re;
        for i in 1..=steps {
            let ev = scan_min_ev + i as f64 * 0.01;
            let omega = ev * EV_TO_RADS;
            let re = self.epsilon(omega).re;
            if prev_re < 0.0 && re >= 0.0 {
                let ev_prev = ev - 0.01;
                let frac = (0.0 - prev_re) / (re - prev_re);
                return Some(ev_prev + frac * 0.01);
            }
            prev_re = re;
        }
        None
    }

    /// Intraband spectral weight from Drude parameters:
    /// `W_intra = (pi/2) * omega_p^2 * eps_0` (SI units S/(m*s)).
    /// Returns `None` for materials without a Drude term.
    pub fn intraband_weight(&self) -> Option<f64> {
        let omega_p = if let Some(ext) = &self.extended_drude {
            ext.omega_p_ev * EV_TO_RADS
        } else if let Some(drude) = &self.drude {
            drude.omega_p_ev * EV_TO_RADS
        } else {
            return None;
        };
        Some(PI / 2.0 * omega_p * omega_p * EPS_0)
    }

    /// Interband spectral weight from Lorentz oscillators:
    /// `W_inter = sum_j (pi/2) * S_j * omega_0j^2 * eps_0` (SI units).
    pub fn interband_weight(&self) -> f64 {
        let mut w = 0.0;
        for osc in &self.oscillators {
            let omega_0 = osc.omega_0_ev * EV_TO_RADS;
            w += PI / 2.0 * osc.strength * omega_0 * omega_0 * EPS_0;
        }
        w
    }

    /// Kramers-Kronig consistency check: numerical error metric.
    /// Reconstructs `eps_1(omega)` from `eps_2(omega)` via the subtracted
    /// KK relation and returns the RMS relative error compared to the model.
    /// For a causal Drude-Lorentz model, the error should be small,
    /// limited only by numerical quadrature accuracy and finite cutoff.
    pub fn kramers_kronig_error(&self, cutoff_ev: f64, n_steps: usize) -> f64 {
        let lambda = cutoff_ev * EV_TO_RADS;
        let domega = lambda / n_steps as f64;

        let f_table: Vec<f64> = (1..=n_steps)
            .map(|j| {
                let omega_p = j as f64 * domega;
                omega_p * self.epsilon(omega_p).im.abs()
            })
            .collect();

        let n_probe: usize = 50;
        let probe_step = n_steps / n_probe;
        let mut sum_sq = 0.0;
        let mut count = 0;

        for i in 1..n_probe {
            let idx = i * probe_step;
            let omega = idx as f64 * domega;
            let eps1_model = self.epsilon(omega).re;
            let f_omega = omega * self.epsilon(omega).im.abs();

            let mut integral_sub = 0.0;
            for j in 1..=n_steps {
                let omega_p = j as f64 * domega;
                let diff_sq = omega_p * omega_p - omega * omega;
                if diff_sq.abs() < 1e-30 {
                    let f_plus = if j < n_steps {
                        f_table[j]
                    } else {
                        f_table[j - 1]
                    };
                    let f_minus = if j > 1 {
                        f_table[j - 2]
                    } else {
                        f_table[j - 1]
                    };
                    let f_prime = (f_plus - f_minus) / (2.0 * domega);
                    integral_sub += f_prime / (2.0 * omega) * domega;
                } else {
                    integral_sub += (f_table[j - 1] - f_omega) / diff_sq * domega;
                }
            }

            let pv_log = ((lambda - omega) / (lambda + omega)).abs().ln() / (2.0 * omega);
            let eps1_kk = self.eps_inf + 2.0 / PI * (integral_sub + f_omega * pv_log);

            if eps1_model.abs() > 0.1 {
                let rel_err = (eps1_model - eps1_kk) / eps1_model.abs();
                sum_sq += rel_err * rel_err;
                count += 1;
            }
        }

        if count == 0 {
            return 1.0;
        }
        (sum_sq / count as f64).sqrt()
    }
}
