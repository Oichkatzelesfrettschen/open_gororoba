//! Semiconductor + dielectric optical models: silicon (intrinsic),
//! amorphous silica (room-T + Casimir-experiment variant), silicon
//! nitride, germanium. All return DrudeLorentzParams with no Drude
//! response (drude: None) -- pure interband-oscillator models.
//!
//! Re-exported from optical_database via `pub use` so external paths
//! materials_core::optical_database::{silicon_optical, silica_optical,
//! silica_casimir_optical, silicon_nitride_optical, germanium_optical}
//! stay stable.

use super::{DrudeLorentzParams, LorentzOscillator};

/// Silicon (intrinsic) optical model.
///
/// Semiconductor with bandgap at ~1.1 eV.
pub fn silicon_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // E0 critical point (direct gap at ~3.4 eV)
            LorentzOscillator {
                strength: 29.0,
                omega_0_ev: 3.40,
                gamma_ev: 0.1,
            },
            // E1 critical point
            LorentzOscillator {
                strength: 6.0,
                omega_0_ev: 3.74,
                gamma_ev: 0.25,
            },
            // E2 critical point
            LorentzOscillator {
                strength: 3.0,
                omega_0_ev: 4.40,
                gamma_ev: 0.2,
            },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

/// Silica (SiO2) optical model.
///
/// Wide-bandgap dielectric.
pub fn silica_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // IR phonon resonance
            LorentzOscillator {
                strength: 1.0,
                omega_0_ev: 0.064, // ~8 microns
                gamma_ev: 0.005,
            },
            // UV absorption edge
            LorentzOscillator {
                strength: 1.0,
                omega_0_ev: 11.0,
                gamma_ev: 2.0,
            },
        ],
        eps_inf: 2.1, // n = 1.45 -> eps = 2.1
        extended_drude: None,
    }
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
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // IR1: Si-O rocking mode (460 cm^{-1} = 0.057 eV)
            LorentzOscillator {
                strength: 0.185,
                omega_0_ev: 0.057,
                gamma_ev: 0.003,
            },
            // IR2: Si-O bending mode (800 cm^{-1} = 0.099 eV)
            LorentzOscillator {
                strength: 0.115,
                omega_0_ev: 0.099,
                gamma_ev: 0.005,
            },
            // IR3: Si-O stretching mode (1075 cm^{-1} = 0.133 eV), dominant
            LorentzOscillator {
                strength: 1.400,
                omega_0_ev: 0.133,
                gamma_ev: 0.012,
            },
        ],
        // eps_inf = n^2 = 1.45^2 = 2.10; UV edge already encoded.
        // Verify: eps(0) = 2.1 + 0.185 + 0.115 + 1.400 = 3.800
        eps_inf: 2.1,
        extended_drude: None,
    }
}

/// Silicon Nitride (Si3N4) optical model.
pub fn silicon_nitride_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // IR resonance
            LorentzOscillator {
                strength: 1.5,
                omega_0_ev: 0.11,
                gamma_ev: 0.01,
            },
        ],
        eps_inf: 4.0, // n ~ 2.0
        extended_drude: None,
    }
}

/// Germanium optical model.
pub fn germanium_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // E0 gap
            LorentzOscillator {
                strength: 25.0,
                omega_0_ev: 2.1,
                gamma_ev: 0.15,
            },
            // E1 critical point
            LorentzOscillator {
                strength: 8.0,
                omega_0_ev: 2.3,
                gamma_ev: 0.2,
            },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}
