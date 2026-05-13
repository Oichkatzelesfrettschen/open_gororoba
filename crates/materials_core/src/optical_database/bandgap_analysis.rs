//! Bandgap analysis methods for `DrudeLorentzParams`: Tauc, Urbach,
//! Penn, absorption onset, joint density of states.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split.
//! These 5 methods all return derived "electronic transitions" properties
//! computed by post-processing the underlying epsilon(omega) and
//! absorption_coefficient(omega) curves of the parent impl. They are
//! gated on metals (return `None` when a Drude term is present) since
//! the bandgap concept does not apply to free-electron-dominated systems.
//!
//! References:
//! - Tauc (1968) Mater. Res. Bull. 3, 37 (Tauc plot for amorphous Ge).
//! - Urbach (1953) Phys. Rev. 92, 1324 (exponential absorption tail in
//!   disordered semiconductors).
//! - Penn (1962) Phys. Rev. 128, 2093 (single-oscillator approximation
//!   of semiconductor dielectric response).

use super::{DrudeLorentzParams, EV_TO_RADS};

impl DrudeLorentzParams {
    /// Tauc plot gap extraction in eV.
    ///
    /// Fits `(alpha * hv)^exponent vs hv` to find the band gap energy
    /// by linear extrapolation. Standard exponents:
    /// - `2.0`: direct allowed transition
    /// - `0.5`: indirect allowed transition
    /// - `2.0/3.0`: direct forbidden transition
    /// - `1.0/3.0`: indirect forbidden transition
    ///
    /// Returns `None` for metals (Drude-like, no gap) or if no linear region
    /// is found.
    pub fn tauc_gap_ev(&self, exponent: f64) -> Option<f64> {
        if self.drude.is_some() || self.extended_drude.is_some() {
            return None;
        }
        if self.oscillators.is_empty() {
            return None;
        }

        let n_pts: usize = 1000;
        let e_min = 0.1_f64;
        let e_max = 15.0_f64;
        let de = (e_max - e_min) / n_pts as f64;

        let tauc_data: Vec<(f64, f64)> = (0..n_pts)
            .map(|i| {
                let ev = e_min + (i as f64 + 0.5) * de;
                let omega = ev * EV_TO_RADS;
                let alpha = self.absorption_coefficient(omega);
                let y = (alpha * ev).powf(exponent);
                (ev, y)
            })
            .collect();

        let window: usize = 30;
        let mut best_slope = 0.0_f64;
        let mut best_start = 0_usize;
        for start in 0..n_pts.saturating_sub(window) {
            let end = start + window;
            let n_w = (end - start) as f64;
            let mut sx = 0.0;
            let mut sy = 0.0;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for &(x, y) in &tauc_data[start..end] {
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
            let slope = (n_w * sxy - sx * sy) / (n_w * sxx - sx * sx);
            if slope > best_slope {
                best_slope = slope;
                best_start = start;
            }
        }

        if best_slope <= 0.0 {
            return None;
        }

        let best_end = best_start + window;
        let n_w = window as f64;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        for &(x, y) in &tauc_data[best_start..best_end] {
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let slope = (n_w * sxy - sx * sy) / (n_w * sxx - sx * sx);
        let intercept = (sy - slope * sx) / n_w;

        let gap = -intercept / slope;
        if gap > 0.0 && gap < e_max {
            Some(gap)
        } else {
            None
        }
    }

    /// Urbach energy (exponential absorption-tail parameter) in eV.
    ///
    /// In the sub-gap absorption tail, `alpha ~ exp(E / E_u)` where `E_u`
    /// is the Urbach energy characterising disorder. Extracted by fitting
    /// `ln(alpha) vs E` in the region below the main absorption onset.
    ///
    /// For Lorentz oscillators, the tail is algebraic (not truly exponential),
    /// so this returns an effective `E_u` that should be compared cautiously
    /// with experimental Urbach energies from real semiconductors.
    ///
    /// Returns `None` for metals or if no meaningful fit region is found
    /// (R^2 < 0.8 or slope <= 0.1 per eV).
    pub fn urbach_energy_ev(&self) -> Option<f64> {
        if self.drude.is_some() || self.extended_drude.is_some() {
            return None;
        }
        if self.oscillators.is_empty() {
            return None;
        }

        let n_pts = 200;
        let e_min = 0.5;
        let e_max = 5.0;
        let de = (e_max - e_min) / n_pts as f64;

        let mut data: Vec<(f64, f64)> = Vec::new();
        for i in 0..n_pts {
            let ev = e_min + (i as f64 + 0.5) * de;
            let omega = ev * EV_TO_RADS;
            let alpha = self.absorption_coefficient(omega);
            if alpha > 1e2 {
                data.push((ev, alpha.ln()));
            }
        }

        if data.len() < 20 {
            return None;
        }

        let window = 15;
        let mut best_slope = 0.0_f64;
        let mut best_start = 0_usize;
        for start in 0..data.len().saturating_sub(window) {
            let end = start + window;
            let n_w = window as f64;
            let mut sx = 0.0;
            let mut sy = 0.0;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for &(x, y) in &data[start..end] {
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
            let slope = (n_w * sxy - sx * sy) / (n_w * sxx - sx * sx);
            if slope > best_slope {
                best_slope = slope;
                best_start = start;
            }
        }

        if best_slope > 0.1 {
            let best_end = (best_start + window).min(data.len());
            let n_w = (best_end - best_start) as f64;
            let mut sx = 0.0;
            let mut sy = 0.0;
            for &(ex, ey) in &data[best_start..best_end] {
                sx += ex;
                sy += ey;
            }
            let mean_x = sx / n_w;
            let mean_y = sy / n_w;
            let mut ss_res = 0.0;
            let mut ss_tot = 0.0;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for &(ex, ey) in &data[best_start..best_end] {
                let dx = ex - mean_x;
                let dy = ey - mean_y;
                sxx += dx * dx;
                sxy += dx * dy;
                ss_tot += dy * dy;
            }
            let slope = sxy / sxx;
            let intercept = mean_y - slope * mean_x;
            for &(ex, ey) in &data[best_start..best_end] {
                let pred = slope * ex + intercept;
                let residual = ey - pred;
                ss_res += residual * residual;
            }
            let r_sq = 1.0 - ss_res / ss_tot;

            if r_sq > 0.8 && slope > 0.1 {
                Some(1.0 / slope)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Penn model gap energy in eV.
    ///
    /// The Penn (1962) model approximates a semiconductor's dielectric
    /// response as a single oscillator, giving
    /// `E_g_Penn = hbar * omega_p_eff / sqrt(eps_s - 1)`
    /// where `omega_p_eff` is computed from the total oscillator strength
    /// and `eps_s` is the static dielectric constant.
    ///
    /// Returns `None` for metals or if `eps_static <= 1`.
    pub fn penn_gap_ev(&self) -> Option<f64> {
        let eps_s = self.static_dielectric()?;
        if eps_s <= 1.0 {
            return None;
        }

        let mut omega_p_sq = 0.0;
        for osc in &self.oscillators {
            let omega_0 = osc.omega_0_ev * EV_TO_RADS;
            omega_p_sq += osc.strength * omega_0 * omega_0;
        }

        if omega_p_sq <= 0.0 {
            return None;
        }

        let omega_p_eff = omega_p_sq.sqrt();
        let gap_rads = omega_p_eff / (eps_s - 1.0).sqrt();
        let gap_ev = gap_rads / EV_TO_RADS;

        if gap_ev > 0.0 && gap_ev < 50.0 {
            Some(gap_ev)
        } else {
            None
        }
    }

    /// Absorption onset energy where alpha reaches a fraction of its
    /// maximum. Scans from 0.1 to 15 eV and finds the first energy where
    /// the absorption coefficient reaches `threshold_fraction * alpha_max`.
    /// Useful for comparing with Tauc and Penn gap methods.
    ///
    /// Returns `None` for metals or if alpha is below threshold everywhere.
    pub fn absorption_onset_ev(&self, threshold_fraction: f64) -> Option<f64> {
        if self.drude.is_some() || self.extended_drude.is_some() {
            return None;
        }

        let n_pts = 1500;
        let e_min = 0.1;
        let e_max = 15.0;
        let de = (e_max - e_min) / n_pts as f64;

        let mut alpha_max = 0.0_f64;
        let alphas: Vec<(f64, f64)> = (0..n_pts)
            .map(|i| {
                let ev = e_min + i as f64 * de;
                let omega = ev * EV_TO_RADS;
                let alpha = self.absorption_coefficient(omega);
                if alpha > alpha_max {
                    alpha_max = alpha;
                }
                (ev, alpha)
            })
            .collect();

        if alpha_max < 1.0 {
            return None;
        }

        let threshold = threshold_fraction * alpha_max;

        for &(ev, alpha) in &alphas {
            if alpha >= threshold {
                return Some(ev);
            }
        }
        None
    }

    /// Joint density of states (JDOS) proxy in arbitrary units.
    ///
    /// `JDOS(omega) ~ omega * |eps_2(omega)|` for direct transitions.
    /// Proportional to the number of electronic states available for
    /// vertical transitions at energy `hbar*omega`; van Hove singularities
    /// appear as peaks.
    pub fn joint_density_of_states(&self, omega: f64) -> f64 {
        let eps = self.epsilon(omega);
        omega * eps.im.abs()
    }
}
