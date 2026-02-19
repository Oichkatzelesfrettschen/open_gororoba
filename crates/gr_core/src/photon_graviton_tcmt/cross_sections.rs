//! Cross-section calculations from TCMT parameters
//!
//! Maps TCMT parameters to observable scattering, absorption, and extinction
//! cross-sections using Ruan-Fan Eqs. 14, 21, 23 extended to gravitational system.

use crate::photon_graviton_tcmt::{CrossSections, GravitationalCoupling, AsymmetryParameter};

/// Compute total cross-section tensor from coupling and frequency sweep
///
/// Integrates scattering response over frequency range to get effective cross-section.
pub struct CrossSectionComputer {
    coupling: GravitationalCoupling,
    asymmetry: AsymmetryParameter,
}

impl CrossSectionComputer {
    /// Create new cross-section computer
    pub fn new(coupling: GravitationalCoupling) -> Self {
        let asymmetry = AsymmetryParameter::from_coupling(&coupling);
        CrossSectionComputer { coupling, asymmetry }
    }

    /// Compute cross-sections at specific frequency
    ///
    /// # Arguments
    /// * `frequency` - Probe frequency (units: same as resonance_frequency)
    /// * `loss_ratio` - γ_nr / γ_total (non-radiative loss fraction)
    ///
    /// # Returns
    /// CrossSections struct with C_sct, C_abs, C_ext
    pub fn at_frequency(&self, frequency: f64, loss_ratio: f64) -> CrossSections {
        let detuning = frequency - self.coupling.resonance_frequency;
        let gamma_total = self.coupling.total_decay_rate();
        let x = detuning / gamma_total;  // Normalized detuning

        // Scattering: Fano lineshape
        let x_sq = x * x;
        let q_sq = self.asymmetry.q * self.asymmetry.q;
        let c_sct = (x_sq + q_sq) / (x_sq + 1.0);

        // Absorption: Loss-dependent
        let c_abs = loss_ratio * (1.0 / (x_sq + 1.0)) * (1.0 + 2.0 * self.asymmetry.q * x / (x_sq + 1.0)).max(0.0);

        CrossSections::new(c_sct, c_abs)
    }

    /// Compute peak cross-section (at resonance)
    pub fn peak_scattering(&self) -> f64 {
        let c_sct = (self.asymmetry.q * self.asymmetry.q) / 1.0;  // x=0: (q²) / 1
        c_sct
    }

    /// Compute quality factor Q from FWHM
    ///
    /// Q = ω₀ / Δω_FWHM where Δω_FWHM is full width at half maximum
    /// For Fano resonance: Q = 1 / (2 * gamma / omega_0)
    pub fn quality_factor(&self) -> f64 {
        let gamma_total = self.coupling.total_decay_rate();
        self.coupling.resonance_frequency / (2.0 * gamma_total)
    }

    /// Compute resonance linewidth
    pub fn linewidth(&self) -> f64 {
        self.coupling.total_decay_rate()
    }

    /// Verify energy conservation: C_ext = C_sct + C_abs
    pub fn verify_optical_theorem(&self, frequency: f64, loss_ratio: f64, tolerance: f64) -> bool {
        let cs = self.at_frequency(frequency, loss_ratio);
        cs.verify_optical_theorem(tolerance)
    }
}

/// Compute cross-sections for a frequency sweep
pub fn sweep_cross_sections(
    coupling: &GravitationalCoupling,
    frequencies: &[f64],
    loss_ratio: f64,
) -> Vec<CrossSections> {
    let computer = CrossSectionComputer::new(*coupling);
    frequencies.iter().map(|&f| computer.at_frequency(f, loss_ratio)).collect()
}

/// Check if resonance is in valid range (weak-field approximation)
pub fn is_weak_field_valid(coupling: &GravitationalCoupling) -> bool {
    coupling.weak_field_parameter() < 0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_section_computer_creation() {
        let coupling = GravitationalCoupling::new(1e-3, 1e-6, 1e-7, 1e-8, 1e15, 0.5);
        let computer = CrossSectionComputer::new(coupling);
        assert!(computer.coupling.is_weak_field());
    }

    #[test]
    fn optical_theorem_at_resonance() {
        let coupling = GravitationalCoupling::new(1e-3, 1e-6, 1e-7, 1e-8, 1e15, 0.5);
        let computer = CrossSectionComputer::new(coupling);
        let cs = computer.at_frequency(coupling.resonance_frequency, 0.5);
        assert!(cs.verify_optical_theorem(1e-10));
    }

    #[test]
    fn quality_factor_positive() {
        let coupling = GravitationalCoupling::new(1e-3, 1e-6, 1e-7, 1e-8, 1e15, 0.5);
        let computer = CrossSectionComputer::new(coupling);
        assert!(computer.quality_factor() > 0.0);
    }

    #[test]
    fn peak_scattering_normalized() {
        let coupling = GravitationalCoupling::new(1e-3, 1e-6, 0.0, 1e-8, 1e15, 0.5);
        let computer = CrossSectionComputer::new(coupling);
        let peak = computer.peak_scattering();
        assert!(peak > 0.0);
    }

    #[test]
    fn sweep_produces_correct_length() {
        let coupling = GravitationalCoupling::new(1e-3, 1e-6, 1e-7, 1e-8, 1e15, 0.5);
        let freqs = vec![1e14, 1e15, 1e16];
        let cs = sweep_cross_sections(&coupling, &freqs, 0.5);
        assert_eq!(cs.len(), 3);
    }
}
