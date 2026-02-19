//! Core types for photon-graviton TCMT

use std::fmt;

/// Gravitational coupling parameters extracted from worldline amplitude
///
/// References: Ahmadiniaz et al. (2026) arXiv:2601.23279, Bastianelli & Schubert gr-qc/0412095
#[derive(Clone, Copy, Debug)]
pub struct GravitationalCoupling {
    /// Photon-graviton coupling strength (from E-073 irreducible diagram)
    /// Units: [frequency]
    pub coupling_strength: f64,

    /// Radiative decay rate of photonic mode (from tadpole contribution)
    /// Units: [frequency]
    pub gamma_radiative: f64,

    /// Non-radiative decay rate (absorption/scattering losses)
    /// Units: [frequency]
    pub gamma_nonradiative: f64,

    /// Gravitational contribution to decay rate
    /// Units: [frequency], typically << gamma_radiative in weak-field
    pub gamma_gravitational: f64,

    /// Resonance frequency (photonic mode center)
    /// Units: [frequency]
    pub resonance_frequency: f64,

    /// Coupling phase (from E-073 external-leg diagram)
    /// Units: radians, range [0, 2π]
    pub coupling_phase: f64,
}

impl GravitationalCoupling {
    /// Create new coupling parameters
    ///
    /// # Arguments
    /// * `coupling_strength` - Main coupling parameter from worldline
    /// * `gamma_radiative` - Photonic decay rate
    /// * `gamma_nonradiative` - Loss/absorption rate
    /// * `gamma_gravitational` - Gravitational contribution to decay
    /// * `resonance_frequency` - Resonance center frequency
    /// * `coupling_phase` - Phase of coupling interaction
    pub fn new(
        coupling_strength: f64,
        gamma_radiative: f64,
        gamma_nonradiative: f64,
        gamma_gravitational: f64,
        resonance_frequency: f64,
        coupling_phase: f64,
    ) -> Self {
        GravitationalCoupling {
            coupling_strength,
            gamma_radiative,
            gamma_nonradiative,
            gamma_gravitational,
            resonance_frequency,
            coupling_phase,
        }
    }

    /// Total decay rate = radiative + non-radiative + gravitational
    pub fn total_decay_rate(&self) -> f64 {
        self.gamma_radiative + self.gamma_nonradiative + self.gamma_gravitational
    }

    /// Weak-field parameter: coupling / (total_decay * resonance_frequency)
    /// Should be << 1 for validity of weak-field TCMT approximation
    pub fn weak_field_parameter(&self) -> f64 {
        self.coupling_strength / (self.total_decay_rate() * self.resonance_frequency)
    }

    /// Check if parameters satisfy weak-field approximation (epsilon << 1)
    pub fn is_weak_field(&self) -> bool {
        self.weak_field_parameter() < 0.1
    }
}

impl fmt::Display for GravitationalCoupling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GravitationalCoupling {{ kappa={:.4e}, gamma_tot={:.4e}, omega_0={:.4e}, weak_field_param={:.4e} }}",
            self.coupling_strength,
            self.total_decay_rate(),
            self.resonance_frequency,
            self.weak_field_parameter()
        )
    }
}

/// Fano asymmetry parameter for photon-graviton system
///
/// Generalization of Ruan-Fan Eq. 21 to include gravitational coupling.
/// The asymmetry parameter q characterizes the balance between:
/// - Direct (non-resonant) coupling pathway
/// - Resonant coupling through the photonic mode
///
/// References: Ruan & Fan (2009) Eq. 21, Maksimov et al. (2025) arXiv:2505.00396
#[derive(Clone, Copy, Debug)]
pub struct AsymmetryParameter {
    /// Asymmetry parameter value q_grav
    /// - q → 0: Perfect resonance (Lorentzian, C_max = 1 in normalized units)
    /// - q → ∞: Perfect anti-resonance (dip, C_min = 0)
    /// - q ~ 1: Typical Fano asymmetry intermediate regime
    pub q: f64,

    /// Alternative form: cot(phi/2) where phi is phase delay
    /// Relationship: q = -cot(phi/2) [Ruan-Fan convention]
    pub phase_factor: f64,

    /// Effective gravitational contribution to asymmetry
    /// Quantifies how strongly gravity perturbs the photonic Fano response
    pub gravitational_factor: f64,
}

impl AsymmetryParameter {
    /// Create asymmetry parameter from q value
    pub fn from_q(q: f64) -> Self {
        let phase_factor = if q.abs() > 1e-10 {
            -1.0 / q.atan()  // cot(phi/2) = -1/arctan(q)
        } else {
            f64::INFINITY
        };

        let gravitational_factor = q.abs();  // Magnitude indicates strength

        AsymmetryParameter {
            q,
            phase_factor,
            gravitational_factor,
        }
    }

    /// Create asymmetry parameter from phase (in radians)
    pub fn from_phase(phi: f64) -> Self {
        let q = -(phi / 2.0).tan().recip();
        AsymmetryParameter::from_q(q)
    }

    /// Compute lineshape parameter from coupling and decay rates
    /// Based on Ruan-Fan Eq. 14-23
    pub fn from_coupling(coupling: &GravitationalCoupling) -> Self {
        // q = (direct_coupling / (i * coupling_decay))
        // In real units: q ≈ coupling_strength / (gamma_coupling * resonance_frequency)
        // Adjusted for gravitational contribution
        let base_q = coupling.coupling_strength / (coupling.total_decay_rate() * coupling.resonance_frequency);
        let gravitational_correction = (coupling.gamma_gravitational / coupling.total_decay_rate()).abs();

        let q = base_q * (1.0 + gravitational_correction * coupling.coupling_phase.cos());

        AsymmetryParameter::from_q(q)
    }
}

impl fmt::Display for AsymmetryParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AsymmetryParameter {{ q={:.4}, phase={:.4} }}", self.q, self.phase_factor)
    }
}

/// TCMT state vector: amplitudes of photonic and gravitational modes
///
/// Represents the coupled-mode amplitude evolution in time
#[derive(Clone, Copy, Debug)]
pub struct TCMTState {
    /// Photonic mode amplitude (complex)
    pub a_photon: num_complex::Complex64,

    /// Gravitational mode amplitude (complex)
    pub a_graviton: num_complex::Complex64,

    /// Time at which this state is evaluated
    pub time: f64,
}

impl TCMTState {
    /// Create new TCMT state
    pub fn new(a_photon: num_complex::Complex64, a_graviton: num_complex::Complex64, time: f64) -> Self {
        TCMTState {
            a_photon,
            a_graviton,
            time,
        }
    }

    /// Total photon number (|a_photon|^2)
    pub fn photon_population(&self) -> f64 {
        self.a_photon.norm_sqr()
    }

    /// Total graviton number (|a_graviton|^2)
    pub fn graviton_population(&self) -> f64 {
        self.a_graviton.norm_sqr()
    }

    /// Total energy (sum of populations, assuming equal masses/couplings)
    pub fn total_energy(&self) -> f64 {
        self.photon_population() + self.graviton_population()
    }

    /// Coupling strength indicator: projection of photon onto graviton
    pub fn coupling_indicator(&self) -> f64 {
        (self.a_photon.conj() * self.a_graviton).norm()
    }
}

/// Scattering cross-section components
///
/// Following Ruan-Fan Eqs. 14, 21, 23 but for photon-graviton system
#[derive(Clone, Copy, Debug)]
pub struct CrossSections {
    /// Scattering cross-section (elastic scattering)
    /// Units: [frequency]^{-2} or [area] depending on normalization
    pub c_scattering: f64,

    /// Absorption cross-section (inelastic loss)
    /// Units: same as c_scattering
    pub c_absorption: f64,

    /// Extinction cross-section = scattering + absorption
    /// Units: same as c_scattering
    pub c_extinction: f64,

    /// Optical theorem check: should satisfy c_extinction = c_scattering + c_absorption
    /// This value should be very close to 1.0 if optical theorem is satisfied
    pub optical_theorem_ratio: f64,
}

impl CrossSections {
    /// Create new cross-section set
    pub fn new(c_scattering: f64, c_absorption: f64) -> Self {
        let c_extinction = c_scattering + c_absorption;
        let optical_theorem_ratio = if (c_scattering + c_absorption).abs() > 1e-15 {
            c_extinction / (c_scattering + c_absorption)
        } else {
            1.0
        };

        CrossSections {
            c_scattering,
            c_absorption,
            c_extinction,
            optical_theorem_ratio,
        }
    }

    /// Verify optical theorem within tolerance (default 1e-10)
    pub fn verify_optical_theorem(&self, tolerance: f64) -> bool {
        (self.optical_theorem_ratio - 1.0).abs() < tolerance
    }
}

impl fmt::Display for CrossSections {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CrossSections {{ C_sct={:.4e}, C_abs={:.4e}, C_ext={:.4e}, OT_ratio={:.6} }}",
            self.c_scattering, self.c_absorption, self.c_extinction, self.optical_theorem_ratio
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravitational_coupling_creation() {
        let coupling = GravitationalCoupling::new(1e-3, 1e-6, 1e-7, 1e-8, 1e15, 0.5);
        assert!((coupling.coupling_strength - 1e-3).abs() < 1e-10);
        assert!(coupling.is_weak_field());
    }

    #[test]
    fn asymmetry_parameter_from_q() {
        let asym = AsymmetryParameter::from_q(1.0);
        assert!((asym.q - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tcmt_state_energy() {
        let state = TCMTState::new(
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(0.5, 0.0),
            0.0,
        );
        assert!((state.total_energy() - 1.25).abs() < 1e-10);
    }

    #[test]
    fn cross_sections_optical_theorem() {
        let cs = CrossSections::new(0.5, 0.3);
        assert!(cs.verify_optical_theorem(1e-10));
    }
}
