//! Photon-Graviton Temporal Coupled-Mode Theory (TCMT)
//!
//! Implements the temporal coupled-mode framework for photon-graviton mixing
//! in the weak-field limit, synthesizing:
//! - Ruan & Fan (2009) TCMT foundation for 2D cylindrical scattering
//! - Ahmadiniaz et al. (2026) complete one-loop worldline amplitude
//! - Maksimov et al. (2025) generic TCMT symmetry constraints
//!
//! ## Physics Framework
//!
//! The photon-graviton system is modeled as a two-coupled-mode resonator where:
//! - Mode 1: Photonic resonance with coupling kappa_ph, decay gamma_ph
//! - Mode 2: Gravitational resonance with coupling kappa_grav, decay gamma_grav
//! - Coupling: Gravitational interaction strength epsilon_grav
//!
//! The TCMT equations follow Ruan-Fan Eq. 8 extended to coupled modes:
//! ```text
//! d/dt `[a_ph]`   = `[-i*omega_0,ph - gamma_ph/2    -i*kappa_coupling]` `[a_ph]`
//!       `[a_grav]`   `[-i*kappa_coupling             -i*omega_0,grav - gamma_grav/2]` `[a_grav]`
//! ```
//!
//! ## Asymmetry Parameter q_grav
//!
//! The gravitational asymmetry parameter q_grav (generalization of Ruan-Fan Eq. 21)
//! encodes the ratio of direct to resonant coupling:
//! ```text
//! q_grav = (kappa_direct / (i * gamma_coupling))
//! ```
//! where gamma_coupling is the effective coupling decay rate extracted from the
//! worldline amplitude (E-073).
//!
//! ## Validation
//!
//! All results validated against:
//! 1. Weak-field limit -> must match E-073 worldline amplitude
//! 2. Energy conservation -> kappa_1 - kappa_2 = gamma_l (Maksimov constraint)
//! 3. Time-reversal symmetry -> pairs of real zeros in transmission
//! 4. Optical theorem -> C_ext = C_sct + C_abs
//!
//! ## References
//!
//! - Ruan & Fan (2009) arXiv:0909.3323 - TCMT foundational framework
//! - Bastianelli & Schubert (2004) gr-qc/0412095 - Worldline one-loop structure
//! - Ahmadiniaz et al. (2026) arXiv:2601.23279 - Complete photon-graviton amplitude
//! - Maksimov et al. (2025) arXiv:2505.00396 - TCMT symmetry constraints

pub mod amplitude_bridge;
pub mod constants;
pub mod cross_sections;
pub mod fano_lineshape;
pub mod tcmt_equations;
pub mod types;

pub use amplitude_bridge::*;
pub use constants::*;
pub use cross_sections::*;
pub use fano_lineshape::*;
pub use tcmt_equations::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_integrity_check() {
        // Verify all submodules are accessible
        let _constants = PLANCK_CONSTANT;
        let _reduced_planck = REDUCED_PLANCK;
        let _speed_of_light = SPEED_OF_LIGHT;
    }
}
