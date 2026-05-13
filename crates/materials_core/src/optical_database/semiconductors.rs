//! Semiconductor + dielectric optical models: silicon (intrinsic),
//! amorphous silica (room-T + Casimir-experiment variant), silicon
//! nitride, germanium. All return DrudeLorentzParams with no Drude
//! response (drude: None) -- pure interband-oscillator models.
//!
//! Re-exported from optical_database via `pub use` so external paths
//! materials_core::optical_database::{silicon_optical, silica_optical,
//! silica_casimir_optical, silicon_nitride_optical, germanium_optical}
//! stay stable.
//!
//! All five constructors source their parameters from
//! `crates/materials_data/data/optical/lorentz_models.toml` via the
//! build-time codegen path -- the literal arrays no longer live in
//! Rust source (Phase 5 / task #127).

use super::{DrudeLorentzParams, LorentzOscillator};

/// Build a `DrudeLorentzParams` from the codegen'd `<NAME>_EPS_INF`
/// scalar and `<NAME>_OSCILLATORS: &[[f64; 3]]` array constants.
/// Local helper -- not exported.
fn from_codegen(
    eps_inf: f64,
    oscillators: &'static [[f64; 3]],
) -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: oscillators
            .iter()
            .map(|&[strength, omega_0_ev, gamma_ev]| LorentzOscillator {
                strength,
                omega_0_ev,
                gamma_ev,
            })
            .collect(),
        eps_inf,
        extended_drude: None,
    }
}

/// Silicon (intrinsic) optical model. Bandgap ~1.1 eV; E0/E1/E2 critical points.
pub fn silicon_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SILICON_EPS_INF,
        materials_data::SILICON_OSCILLATORS,
    )
}

/// Silica (SiO2) optical model -- wide-bandgap dielectric; IR phonon + UV edge.
pub fn silica_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SILICA_EPS_INF,
        materials_data::SILICA_OSCILLATORS,
    )
}

/// Silica (SiO2) optical model calibrated for Casimir-Lifshitz calculations.
///
/// Three-oscillator IR model from Lambrecht and Reynaud (2000) and Parsegian (2006).
/// Reproduces the known static permittivity eps_static = 3.80:
///   eps_inf=2.1 (optical, n=1.45) + sum(S_i)=1.700 (IR phonons) = 3.800.
///
/// # IR phonon assignments (Palik 1998 Table II; Parsegian 2006 Table B.2)
/// - IR1: Si-O rocking,    460 cm^{-1} = 0.057 eV, S=0.185
/// - IR2: Si-O bending,    800 cm^{-1} = 0.099 eV, S=0.115
/// - IR3: Si-O stretching, 1075 cm^{-1} = 0.133 eV, S=1.400 (dominant mode)
///
/// No explicit UV oscillator: eps_inf=2.1 already encodes the UV edge contribution.
pub fn silica_casimir_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SILICA_CASIMIR_EPS_INF,
        materials_data::SILICA_CASIMIR_OSCILLATORS,
    )
}

/// Silicon Nitride (Si3N4) optical model -- n ~ 2.0, single IR resonance.
pub fn silicon_nitride_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SILICON_NITRIDE_EPS_INF,
        materials_data::SILICON_NITRIDE_OSCILLATORS,
    )
}

/// Germanium optical model -- E0 gap + E1 critical point.
pub fn germanium_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::GERMANIUM_EPS_INF,
        materials_data::GERMANIUM_OSCILLATORS,
    )
}
