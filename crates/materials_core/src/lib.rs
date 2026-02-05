//! materials_core: Metamaterial absorbers, Tang mass predictions, refractive index.
//!
//! This crate provides:
//! - Zero-divisor to metamaterial layer mapping
//! - Tang-style lepton mass ratio predictions
//! - Perfect absorption via critical coupling
//!
//! # Literature
//! - Tang & Tang 2023: Sedenion-SU(5) mapping
//! - Gresnigt 2023: Unified sedenion lepton model
//! - Landy et al. 2008: Perfect metamaterial absorber

pub mod metamaterial;
pub mod tang_mass;

pub use metamaterial::{
    map_zd_to_refractive_index, map_zd_norm_to_thickness, classify_material_type,
    map_zd_pair_to_layer, build_absorber_stack, verify_physical_realizability,
    canonical_sedenion_zd_pairs, MetamaterialLayer, ZdToLayerMapping,
    MaterialType, VerificationResult,
};

pub use tang_mass::{
    M_ELECTRON, M_MUON, M_TAU, RATIO_E_MU, RATIO_MU_TAU, RATIO_E_TAU,
    GenerationAssignment, MassRatioPrediction, MassNullTestResult,
    DimensionScalingResult, basis_associator_norm, canonical_sedenion_assignments,
    predict_mass_ratios, find_best_assignment, mass_ratio_null_test,
    dimension_scaling_analysis,
};
