//! Tungstate optical models: WO3, WO3-x (oxygen-deficient),
//! Cs-doped WO3 (isotropic + uniaxial), CaWO4, PbWO4.
//!
//! These six material constructors were originally inlined at the
//! tail of optical_database.rs. Lifted to a sibling submodule and
//! re-exported via `pub use` so the canonical paths
//! `materials_core::optical_database::{wo3_optical, cs_wo3_optical,
//! cawo4_optical, pbwo4_optical, ...}` remain stable for callers
//! across the casimir / lifshitz crates.

use super::{DrudeLorentzParams, DrudeParams, LorentzOscillator, UniaxialOptical};

/// Tungsten Trioxide (WO3) stoichiometric wide-gap semiconductor.
///
/// Band gap ~2.6-3.0 eV. No free carriers: purely Lorentz oscillator model
/// with IR phonon modes and UV absorption edge.
/// From: Granqvist (2000), Niklasson & Granqvist (2007).
pub fn wo3_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // W-O stretching mode
            LorentzOscillator {
                strength: 1.5,
                omega_0_ev: 0.085,
                gamma_ev: 0.008,
            },
            // W-O-W bending mode
            LorentzOscillator {
                strength: 0.8,
                omega_0_ev: 0.042,
                gamma_ev: 0.005,
            },
            // Higher phonon
            LorentzOscillator {
                strength: 0.4,
                omega_0_ev: 0.120,
                gamma_ev: 0.010,
            },
            // Band-edge absorption
            LorentzOscillator {
                strength: 6.0,
                omega_0_ev: 3.5,
                gamma_ev: 0.8,
            },
            // Higher UV transition
            LorentzOscillator {
                strength: 3.0,
                omega_0_ev: 5.5,
                gamma_ev: 1.5,
            },
        ],
        eps_inf: 4.5,
        extended_drude: None,
    }
}

/// Oxygen-deficient Tungsten Oxide (WO3-x) plasmonic conductor.
///
/// Oxygen vacancies create free carriers (n ~ 1e21 cm^-3 for x ~ 0.1),
/// turning stoichiometric WO3 into a plasmonic material with NIR crossover.
/// omega_p = sqrt(n*e^2 / (eps_0*m*)) for m* = 1.2*m_e gives ~1.07 eV.
/// From: Garcia et al. Nano Lett. 11(10), 2011.
pub fn wo3_x_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 1.07,
            gamma_ev: 0.20,
            eps_inf: 1.0,
        }),
        oscillators: vec![
            // Residual phonon (reduced by vacancy disorder)
            LorentzOscillator {
                strength: 0.8,
                omega_0_ev: 0.085,
                gamma_ev: 0.015,
            },
            // Band-edge (blue-shifted by Burstein-Moss effect)
            LorentzOscillator {
                strength: 5.0,
                omega_0_ev: 3.8,
                gamma_ev: 1.0,
            },
        ],
        eps_inf: 5.88,
        extended_drude: None,
    }
}

/// Cesium Tungsten Bronze (Cs0.33WO3) polycrystalline scalar average.
///
/// Orientation-averaged Drude parameters from the anisotropic tensor.
/// eps_avg = (eps_par + 2*eps_perp) / 3.
/// From: Lynch & Hunter (1991), Handbook of Optical Constants II.
pub fn cs_wo3_optical() -> DrudeLorentzParams {
    // Polycrystalline average of anisotropic Drude weights:
    // omega_p_avg^2 = (omega_p_par^2 + 2*omega_p_perp^2) / 3
    //               = (4.664^2 + 2*3.180^2) / 3 = (21.75 + 20.22) / 3 = 13.99
    // omega_p_avg = sqrt(13.99) = 3.741 eV
    // gamma_avg = (gamma_par + 2*gamma_perp) / 3
    //           = (0.217 + 2*0.335) / 3 = 0.296 eV
    // eps_inf_avg = (6.3 + 2*5.8) / 3 = 5.97
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 3.741,
            gamma_ev: 0.296,
            eps_inf: 1.0,
        }),
        oscillators: vec![
            // Interband transition
            LorentzOscillator {
                strength: 2.0,
                omega_0_ev: 4.0,
                gamma_ev: 1.5,
            },
        ],
        eps_inf: 5.97,
        extended_drude: None,
    }
}

/// Cesium Tungsten Bronze (Cs0.33WO3) uniaxial tensor parameters.
///
/// Hexagonal tungsten bronze structure: c-axis (parallel) has higher Drude weight
/// than the a-b plane (perpendicular), reflecting anisotropic effective mass.
/// From: Lynch & Hunter (1991), Handbook of Optical Constants II.
pub fn cs_wo3_uniaxial() -> UniaxialOptical {
    UniaxialOptical {
        parallel: DrudeLorentzParams {
            drude: Some(DrudeParams {
                omega_p_ev: 4.664,
                gamma_ev: 0.217,
                eps_inf: 1.0,
            }),
            oscillators: vec![LorentzOscillator {
                strength: 2.0,
                omega_0_ev: 4.0,
                gamma_ev: 1.5,
            }],
            eps_inf: 6.3,
            extended_drude: None,
        },
        perpendicular: DrudeLorentzParams {
            drude: Some(DrudeParams {
                omega_p_ev: 3.180,
                gamma_ev: 0.335,
                eps_inf: 1.0,
            }),
            oscillators: vec![LorentzOscillator {
                strength: 2.0,
                omega_0_ev: 4.0,
                gamma_ev: 1.5,
            }],
            eps_inf: 5.8,
            extended_drude: None,
        },
        axis_description: "c-axis (parallel) vs a-b plane (perpendicular), hexagonal bronze",
    }
}

/// Calcium Tungstate (CaWO4) scheelite-structure scintillator.
///
/// Wide-gap dielectric (Eg ~ 5.0 eV) used in cryogenic dark matter detectors
/// (CRESST experiment). Scheelite structure with Ca2+ and WO4^2- tetrahedra.
/// From: Nikl et al. Rad. Meas. 33(5), 2000.
pub fn cawo4_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // WO4 internal stretching modes
            LorentzOscillator {
                strength: 1.0,
                omega_0_ev: 0.110,
                gamma_ev: 0.006,
            },
            // Lattice phonon
            LorentzOscillator {
                strength: 0.6,
                omega_0_ev: 0.045,
                gamma_ev: 0.004,
            },
            // UV absorption edge
            LorentzOscillator {
                strength: 4.0,
                omega_0_ev: 5.5,
                gamma_ev: 1.0,
            },
        ],
        eps_inf: 3.7,
        extended_drude: None,
    }
}

/// Lead Tungstate (PbWO4) fast scintillator crystal.
///
/// Wide-gap dielectric (Eg ~ 4.2 eV) used as electromagnetic calorimeter
/// in CMS at CERN. Fast scintillation decay (~6 ns). Scheelite structure.
/// From: Nikl et al. Rad. Meas. 33(5), 2000.
pub fn pbwo4_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // WO4 internal modes
            LorentzOscillator {
                strength: 1.2,
                omega_0_ev: 0.100,
                gamma_ev: 0.008,
            },
            // Pb-O lattice mode
            LorentzOscillator {
                strength: 0.8,
                omega_0_ev: 0.035,
                gamma_ev: 0.005,
            },
            // UV absorption edge (lower than CaWO4 due to Pb 6s states)
            LorentzOscillator {
                strength: 5.0,
                omega_0_ev: 4.5,
                gamma_ev: 1.2,
            },
        ],
        eps_inf: 4.8,
        extended_drude: None,
    }
}
