//! Anharmonic + multiphonon spectroscopy methods on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 17c). Four methods covering temperature-dependent phonon
//! broadening and multi-phonon process spectroscopy:
//!
//! - `anharmonic_linewidth`: Klemens-type cubic anharmonic
//!   `gamma(T) = gamma_0 + A * (1 + 2*n_BE(omega/2, T))`.
//! - `multiphonon_absorption`: Urbach-like exponential tail above the
//!   one-phonon cutoff.
//! - `two_phonon_density_of_states`: self-convolution of the Lorentz
//!   spectrum (Lorentzian-broadened pair sum).
//! - `infrared_combination_bands`: sorted unique sum and difference
//!   frequencies of oscillator pairs.

use super::{DrudeLorentzParams, K_B_EV};

impl DrudeLorentzParams {
    /// Anharmonic linewidth broadening
    /// `gamma(T) = gamma_0 + A * (1 + 2*n_BE(omega/2, T))` -- Klemens cubic
    /// anharmonic decay of a phonon at omega into two phonons at omega/2.
    /// `A` is the cubic anharmonic coupling. Returns `None` if index OOB.
    pub fn anharmonic_linewidth(
        &self,
        oscillator_index: usize,
        temperature_k: f64,
        coupling_a: f64,
    ) -> Option<f64> {
        let osc = self.oscillators.get(oscillator_index)?;
        let half_omega_ev = osc.omega_0_ev / 2.0;
        let n_be = if temperature_k > 0.0 && half_omega_ev > 0.0 {
            let x = half_omega_ev / (K_B_EV * temperature_k);
            if x > 500.0 {
                0.0
            } else {
                1.0 / (x.exp() - 1.0)
            }
        } else {
            0.0
        };
        Some(osc.gamma_ev + coupling_a * (1.0 + 2.0 * n_be))
    }

    /// Multiphonon absorption coefficient above the one-phonon cutoff.
    /// Urbach-like tail
    /// `alpha ~ exp(-beta * (omega - omega_max) / omega_max)`. Returns
    /// 0 if `omega_ev <= omega_max` (in the one-phonon band) or no
    /// oscillators. Uses the highest oscillator frequency as `omega_max`.
    pub fn multiphonon_absorption(&self, omega_ev: f64, temperature_k: f64, beta: f64) -> f64 {
        if self.oscillators.is_empty() {
            return 0.0;
        }
        let omega_max_ev = self
            .oscillators
            .iter()
            .map(|o| o.omega_0_ev)
            .fold(0.0_f64, f64::max);
        if omega_ev <= omega_max_ev || omega_max_ev <= 0.0 {
            return 0.0;
        }
        let t_factor = if temperature_k > 0.0 && omega_max_ev > 0.0 {
            let x = omega_max_ev / (K_B_EV * temperature_k);
            if x > 500.0 {
                1.0
            } else {
                1.0 + 1.0 / (x.exp() - 1.0)
            }
        } else {
            1.0
        };
        let excess = (omega_ev - omega_max_ev) / omega_max_ev;
        1e4 * t_factor * (-beta * excess).exp()
    }

    /// Two-phonon density of states (self-convolution of the Lorentz
    /// spectrum). Returns the combined DOS at `omega_ev` (eV) summed over
    /// oscillator pairs `(i, j)` with `omega_i + omega_j ~ omega`,
    /// Lorentzian-broadened by `(gamma_i + gamma_j)/2`.
    pub fn two_phonon_density_of_states(&self, omega_ev: f64) -> f64 {
        let mut dos = 0.0;
        for i in &self.oscillators {
            for j in &self.oscillators {
                let sum_ev = i.omega_0_ev + j.omega_0_ev;
                let width = (i.gamma_ev + j.gamma_ev) * 0.5;
                if width > 0.0 {
                    let delta = omega_ev - sum_ev;
                    dos += i.strength * j.strength / (delta * delta + width * width);
                }
            }
        }
        dos
    }

    /// Infrared combination-band frequencies: sorted unique sum and
    /// difference frequencies of all oscillator pairs (in eV).
    pub fn infrared_combination_bands(&self) -> Vec<f64> {
        let mut bands: Vec<f64> = Vec::new();
        for i in 0..self.oscillators.len() {
            for j in i..self.oscillators.len() {
                let sum = self.oscillators[i].omega_0_ev + self.oscillators[j].omega_0_ev;
                bands.push(sum);
                let diff = (self.oscillators[i].omega_0_ev - self.oscillators[j].omega_0_ev).abs();
                if diff > 0.0 {
                    bands.push(diff);
                }
            }
        }
        bands.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        bands.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        bands
    }
}
