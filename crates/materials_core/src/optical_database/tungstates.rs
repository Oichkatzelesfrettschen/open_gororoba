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

use super::{DrudeLorentzParams, DrudeParams, LorentzOscillator, MineralMetadata, OpticSign, UniaxialOptical};

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

// ============================================================================
// MineralMetadata accessors for the tungstate family (task #140 medium-priority
// remediation from the 2026-05-13 audit). All tungstates share scheelite
// structure (CaWO4, PbWO4) or are tungsten bronzes (WO3, Cs-WO3) with
// well-defined polymorphism.
// ============================================================================

/// gamma-WO3 (room-temperature monoclinic polymorph). Electrochromic;
/// reversibly turns blue when intercalated with Li+ or H+.
pub fn wo3_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "tungsten_trioxide_gamma",
        formula: "WO3",
        crystal_system: "monoclinic",
        space_group: "P2_1/n (14) gamma-WO3 (RT polymorph)",
        n_omega: 2.50,
        n_epsilon: 2.50,
        birefringence: 0.0,
        optic_sign: OpticSign::Biaxial,
        hardness_mohs: 2.75,
        density_g_cm3: 7.16,
        color: "lemon-yellow polycrystalline; thin films electrochromic (yellow->deep blue)",
        reference: "Tilley (1995) Defect Crystal Chemistry p.114; canonical WO3 polymorphism review.",
    }
}

/// Cesium tungsten bronze Cs_xWO3 (0.18 <= x <= 0.33): hexagonal P6/mmm
/// bronze with delocalized W 5d electrons. Used as a near-IR shielding
/// transparent conductor.
pub fn cs_wo3_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "cesium_tungsten_bronze",
        formula: "Cs_xWO3 (0.18..=0.33)",
        crystal_system: "hexagonal",
        space_group: "P6/mmm (191) hexagonal tungsten bronze (HTB)",
        n_omega: 1.90,
        n_epsilon: 1.95,
        birefringence: 0.05,
        optic_sign: OpticSign::Positive,
        hardness_mohs: 3.0,
        density_g_cm3: 6.6,
        color: "deep blue to violet (carrier-induced plasma resonance)",
        reference: "Magneli (1953) Acta Chem. Scand. 7, 315 (HTB structure); Takeda et al. (2008) Cs-WO3 IR shielding.",
    }
}

/// Scheelite CaWO4: tetragonal I4_1/a. Used historically as a primary
/// X-ray fluorescence screen ("scheelite green" line at 425 nm).
pub fn cawo4_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "scheelite_cawo4",
        formula: "CaWO4",
        crystal_system: "tetragonal",
        space_group: "I4_1/a (88) scheelite",
        n_omega: 1.918,
        n_epsilon: 1.934,
        birefringence: 0.016,
        optic_sign: OpticSign::Positive,
        hardness_mohs: 4.75,
        density_g_cm3: 6.12,
        color: "colorless to pale yellow; characteristic blue-white fluorescence under SW UV",
        reference: "Nikl (2000) Phys. Status Solidi A 178, 595; mindat.org canonical scheelite entry.",
    }
}

/// Substoichiometric WO3-x: shares the parent monoclinic gamma-WO3 lattice
/// (P2_1/n #14) but with random oxygen vacancies acting as color centers.
pub fn wo3_x_metadata() -> MineralMetadata {
    let parent = wo3_metadata();
    MineralMetadata {
        species_name: "tungsten_trioxide_substoichiometric",
        formula: "WO3-x (x ~ 0.05 to 0.30 oxygen vacancies)",
        color: "intense blue to navy (oxygen-vacancy color centers)",
        ..parent
    }
}

/// Stolzite-structured PbWO4: tetragonal I4_1/a. The fastest commercial
/// scintillator (~10 ns decay); chosen as the CMS / ALICE ECAL calorimeter
/// crystal at CERN. Uniaxial NEGATIVE.
pub fn pbwo4_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "stolzite_pbwo4",
        formula: "PbWO4",
        crystal_system: "tetragonal",
        space_group: "I4_1/a (88) stolzite (scheelite-type)",
        n_omega: 2.27,
        n_epsilon: 2.24,
        birefringence: 0.03,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 3.0,
        density_g_cm3: 8.28,
        color: "pale yellow-green when pure; doped with Mo for radiation hardness in HEP calorimetry",
        reference: "Nikl (2000) Phys. Status Solidi A 178, 595; LHC CMS PWO calorimeter design literature.",
    }
}
