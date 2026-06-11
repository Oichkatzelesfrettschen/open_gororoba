//! Coherence, quality, and spectral characterization methods on
//! `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split
//! (Part 14). Six methods covering temporal/spectral coherence and the
//! Q-factor diagnostics for resonators built from this material:
//!
//! - `oscillator_quality_factor`: Q = omega_0/gamma of the strongest
//!   Lorentz oscillator.
//! - `drude_quality`: Q_Drude = omega/gamma for the free-carrier response.
//! - `figure_of_merit_spp`: `Re[k_spp] / (2*Im[k_spp])` -- number of SPP
//!   wavelengths before 1/e decay.
//! - `spectral_weight_window`: partial sigma_1 integral over a frequency band.
//! - `optical_path_length`: (n*d, kappa*d) for thin-film interference.
//! - `coherence_length`: c / (n * delta_omega).
//! - `penetration_depth_ratio`: skin-depth / wavelength.

use super::{C, DrudeLorentzParams, EV_TO_RADS};

impl DrudeLorentzParams {
    /// Quality factor of the strongest Lorentz oscillator: `Q = omega_0 / gamma`.
    /// High Q means a narrow resonance (long-lived excitation).
    /// Returns `None` if no oscillators.
    pub fn oscillator_quality_factor(&self) -> Option<f64> {
        let strongest = self.oscillators.iter().max_by(|a, b| {
            a.strength
                .partial_cmp(&b.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if strongest.gamma_ev < 1e-30 {
            return None;
        }
        Some(strongest.omega_0_ev / strongest.gamma_ev)
    }

    /// Drude quality factor `Q_Drude = omega / gamma` at the given frequency.
    /// `Q >> 1` means the material is a good conductor at that frequency
    /// (coherent carrier response). Returns `None` if no Drude term.
    pub fn drude_quality(&self, omega: f64) -> Option<f64> {
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
        Some(omega / gamma)
    }

    /// SPP propagation figure of merit: `FoM = Re[k_spp] / (2*Im[k_spp])`,
    /// the number of wavelengths an SPP propagates before decaying to 1/e.
    /// Returns `None` for dielectric materials (no SPP).
    pub fn figure_of_merit_spp(&self, omega: f64, eps_dielectric: f64) -> Option<f64> {
        let k = self.spp_wavevector(omega, eps_dielectric);
        if k.im.abs() < 1e-30 {
            return None;
        }
        Some(k.re / (2.0 * k.im.abs()))
    }

    /// Partial spectral weight `SW = integral[sigma_1(omega) d_omega]`
    /// over `[omega_min, omega_max]` via trapezoidal rule. Partial
    /// oscillator-strength sum rule diagnostic.
    pub fn spectral_weight_window(&self, omega_min: f64, omega_max: f64, n_steps: usize) -> f64 {
        if n_steps < 2 || omega_max <= omega_min {
            return 0.0;
        }
        let d_omega = (omega_max - omega_min) / n_steps as f64;
        let mut sum = 0.0;
        for i in 0..=n_steps {
            let omega = omega_min + i as f64 * d_omega;
            let sigma_1 = self.optical_conductivity_re(omega);
            let w = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
            sum += w * sigma_1;
        }
        sum * d_omega
    }

    /// Optical path length `(Re[n] * d, Im[n] * d)` in meters. The real
    /// part controls interference phase; the imaginary part the per-pass
    /// extinction.
    pub fn optical_path_length(&self, omega: f64, thickness_m: f64) -> (f64, f64) {
        let n = self.refractive_index(omega);
        (n.re * thickness_m, n.im * thickness_m)
    }

    /// Temporal coherence length `l_c = c / (n * delta_omega)`. For a
    /// monochromatic source (`delta_omega -> 0`), `l_c -> infinity`.
    pub fn coherence_length(&self, omega: f64, bandwidth_rad_s: f64) -> f64 {
        if bandwidth_rad_s < 1e-30 {
            return f64::INFINITY;
        }
        let n = self.refractive_index(omega);
        C / (n.re.abs() * bandwidth_rad_s)
    }

    /// Penetration-depth-to-wavelength ratio `delta/lambda`. `<< 1` means
    /// the material is opaque within a wavelength (good metal); `>> 1`
    /// means transparent over many wavelengths.
    pub fn penetration_depth_ratio(&self, omega: f64) -> f64 {
        let alpha = self.absorption_coefficient(omega);
        if alpha < 1e-30 {
            return f64::INFINITY;
        }
        let delta = 1.0 / alpha;
        let lambda = 2.0 * std::f64::consts::PI * C / omega;
        delta / lambda
    }
}
