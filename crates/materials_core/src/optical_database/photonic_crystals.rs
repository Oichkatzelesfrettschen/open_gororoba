//! 1D quarter-wave stack (Bragg mirror) properties on `DrudeLorentzParams`.
//!
//! Extracted from the monolithic `optical_database.rs` impl block as part of
//! the #138 PH-MOD split. Rust permits the `impl` block to be split across
//! multiple files within the same crate; calls to
//! `params.quarter_wave_stack_gap(...)` continue to resolve transparently
//! after this relocation -- no signature or call-site changes.
//!
//! All 6 methods in this module depend on only one method from the parent
//! impl block: `refractive_index(omega)` to obtain the high-index material's
//! n at the band-center frequency. The low-index partner is passed in as the
//! caller's responsibility (the canonical workflow pairs a high-n metal-
//! oxide film with a low-n SiO2 spacer).
//!
//! References: Fink et al. (1998) Science 282, 1679 (omnidirectional gap
//! criterion); Joannopoulos et al. (2008) "Photonic Crystals" 2nd ed.

use super::DrudeLorentzParams;

impl DrudeLorentzParams {
    /// Quarter-wave stack stop band edges for this material (high-n) with a
    /// low-n partner. Returns (omega_low, omega_high) in rad/s for the
    /// first-order stop band. The gap width:
    /// `delta_omega/omega_0 = (4/pi)*arcsin(|n_h - n_l|/(n_h + n_l))`.
    pub fn quarter_wave_stack_gap(&self, omega_center: f64, n_low: f64) -> (f64, f64) {
        let n_h = self.refractive_index(omega_center).re;
        let n_l = n_low.max(1.0);
        let ratio = ((n_h - n_l) / (n_h + n_l)).abs();
        let half_gap = (2.0 / std::f64::consts::PI) * ratio.asin();
        (
            omega_center * (1.0 - half_gap),
            omega_center * (1.0 + half_gap),
        )
    }

    /// Quarter-wave stack peak reflectivity for N pairs.
    /// `R = [(n_h/n_l)^(2N) - 1]^2 / [(n_h/n_l)^(2N) + 1]^2`.
    pub fn quarter_wave_stack_reflectivity(
        &self,
        omega_center: f64,
        n_low: f64,
        n_pairs: u32,
    ) -> f64 {
        let n_h = self.refractive_index(omega_center).re;
        let n_l = n_low.max(1.0);
        let r = (n_h / n_l).powi(2 * n_pairs as i32);
        let num = r - 1.0;
        let den = r + 1.0;
        (num / den).powi(2)
    }

    /// Photonic band gap fractional width:
    /// `delta_omega/omega_0 = (4/pi)*arcsin(|n_h-n_l|/(n_h+n_l))`.
    pub fn photonic_band_gap_ratio(&self, omega: f64, n_low: f64) -> f64 {
        let n_h = self.refractive_index(omega).re;
        let n_l = n_low.max(1.0);
        let ratio = ((n_h - n_l) / (n_h + n_l)).abs();
        (4.0 / std::f64::consts::PI) * ratio.asin()
    }

    /// Bragg wavelength for a given period: `lambda_B = 2 * d * n_eff`.
    /// Returns wavelength in meters.
    pub fn bragg_wavelength(&self, period_m: f64, omega: f64) -> f64 {
        let n = self.refractive_index(omega).re;
        2.0 * period_m * n
    }

    /// Group velocity at band edge (fraction of c). Near a stop band edge,
    /// `v_g -> 0` due to Bragg reflection; for finite stacks
    /// `v_g/c ~ sqrt(1 - R_peak)`.
    pub fn group_velocity_at_band_edge(&self, omega_center: f64, n_low: f64, n_pairs: u32) -> f64 {
        let r = self.quarter_wave_stack_reflectivity(omega_center, n_low, n_pairs);
        (1.0 - r).sqrt()
    }

    /// Omnidirectional gap condition: the gap survives at all incidence
    /// angles when the index contrast clears the Fink et al. (1998)
    /// criterion `(n_h * n_l)^2 > n_h^2 + n_l^2` (equivalent to
    /// `n_h/n_l > (1 + sin^2(theta_B))/(cos^2(theta_B))` for the Brewster
    /// angle `theta_B`).
    pub fn omnidirectional_gap_condition(&self, omega: f64, n_low: f64) -> bool {
        let n_h = self.refractive_index(omega).re;
        let n_l = n_low.max(1.0);
        (n_h * n_l).powi(2) > n_h * n_h + n_l * n_l
    }
}
