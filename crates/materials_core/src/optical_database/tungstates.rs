//! Tungstate optical models: WO3, WO3-x (oxygen-deficient),
//! Cs-doped WO3 (isotropic average + uniaxial pair), CaWO4, PbWO4.
//!
//! All parameters are sourced from
//! `crates/materials_data/data/optical/lorentz_models.toml` via the
//! build-time codegen path (#127 phase 5). The literal float arrays
//! that previously lived in this file have been moved entirely to
//! the TOML registry; this file now only assembles
//! DrudeLorentzParams / UniaxialOptical struct instances from the
//! codegen'd const slices.
//!
//! Re-exported from optical_database via `pub use` so external paths
//! materials_core::optical_database::{wo3_optical, cs_wo3_optical,
//! cawo4_optical, pbwo4_optical, ...} remain stable.

use super::{DrudeLorentzParams, DrudeParams, LorentzOscillator, UniaxialOptical};

/// Build a `DrudeLorentzParams` from the codegen'd consts. Handles
/// the optional Drude free-carrier component pattern uniformly.
fn from_codegen(
    eps_inf: f64,
    drude: Option<[f64; 3]>,
    oscillators: &'static [[f64; 3]],
) -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: drude.map(|[omega_p_ev, gamma_ev, eps_inf]| DrudeParams {
            omega_p_ev,
            gamma_ev,
            eps_inf,
        }),
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

/// Tungsten Trioxide (WO3) -- stoichiometric wide-gap semiconductor.
/// Band gap ~2.6-3.0 eV; pure Lorentz oscillator model (no free carriers).
pub fn wo3_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::WO3_EPS_INF,
        materials_data::WO3_DRUDE,
        materials_data::WO3_OSCILLATORS,
    )
}

/// Oxygen-deficient WO3-x plasmonic conductor (Garcia 2011).
pub fn wo3_x_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::WO3_X_EPS_INF,
        materials_data::WO3_X_DRUDE,
        materials_data::WO3_X_OSCILLATORS,
    )
}

/// Cesium Tungsten Bronze Cs0.33WO3 -- polycrystalline scalar average.
pub fn cs_wo3_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::CS_WO3_EPS_INF,
        materials_data::CS_WO3_DRUDE,
        materials_data::CS_WO3_OSCILLATORS,
    )
}

/// Cesium Tungsten Bronze Cs0.33WO3 -- uniaxial tensor parameters.
/// c-axis (parallel) has higher Drude weight than the a-b plane.
pub fn cs_wo3_uniaxial() -> UniaxialOptical {
    UniaxialOptical {
        parallel: from_codegen(
            materials_data::CS_WO3_PARALLEL_EPS_INF,
            materials_data::CS_WO3_PARALLEL_DRUDE,
            materials_data::CS_WO3_PARALLEL_OSCILLATORS,
        ),
        perpendicular: from_codegen(
            materials_data::CS_WO3_PERPENDICULAR_EPS_INF,
            materials_data::CS_WO3_PERPENDICULAR_DRUDE,
            materials_data::CS_WO3_PERPENDICULAR_OSCILLATORS,
        ),
        axis_description: "c-axis (parallel) vs a-b plane (perpendicular), hexagonal bronze",
    }
}

/// Calcium Tungstate (CaWO4) -- scheelite-structure cryogenic scintillator
/// (CRESST dark-matter experiment).
pub fn cawo4_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::CAWO4_EPS_INF,
        materials_data::CAWO4_DRUDE,
        materials_data::CAWO4_OSCILLATORS,
    )
}

/// Lead Tungstate (PbWO4) -- fast scintillator crystal (CMS calorimeter at CERN).
pub fn pbwo4_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::PBWO4_EPS_INF,
        materials_data::PBWO4_DRUDE,
        materials_data::PBWO4_OSCILLATORS,
    )
}
