//! materials_core: Metamaterial absorbers, effective medium theory, Tang mass predictions.
//!
//! This crate provides:
//! - Zero-divisor to metamaterial layer mapping
//! - Effective medium theory (Maxwell-Garnett, Bruggeman)
//! - Drude-Lorentz dielectric models
//! - Transfer Matrix Method for thin-film optics
//! - Kramers-Kronig consistency checks
//! - Tang-style lepton mass ratio predictions
//! - Periodic table element properties
//! - Optical properties database for Casimir physics
//!
//! # Literature
//! - Sihvola (1999), Electromagnetic Mixing Formulas
//! - Born & Wolf (2019), Principles of Optics
//! - Tang & Tang 2023: Sedenion-SU(5) mapping
//! - Gresnigt 2023: Unified sedenion lepton model
//! - Palik (1998): Handbook of Optical Constants

pub mod metamaterial;
pub mod effective_medium;
pub mod tang_mass;
pub mod periodic_table;
pub mod optical_database;
pub mod featurizer;
pub mod baselines;

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

pub use effective_medium::{
    maxwell_garnett, maxwell_garnett_array,
    bruggeman, bruggeman_array,
    drude, drude_lorentz, LorentzOscillator,
    kramers_kronig_check, KramersKronigResult,
    tmm_reflection, tmm_spectrum, TmmResult,
};

pub use periodic_table::{
    Element, CrystalStructure,
    get_element, get_element_by_z,
};

pub use featurizer::{
    parse_formula, composition_fractions, featurize,
    CompositionFeatures, PropertyStats, feature_vector, feature_names,
};

pub use baselines::{
    train_test_split, ols_fit, run_baseline, RegressionResult,
};

pub use optical_database::{
    // Constants
    EV_TO_RADS, C, HBAR_EV_S,
    // Conversions
    wavelength_to_omega, ev_to_omega, omega_to_ev,
    // Models
    DrudeParams, DrudeLorentzParams, LorentzOscillator as OpticalLorentzOscillator,
    // Pre-defined materials
    gold_drude, gold_drude_lorentz, silver_drude, silver_drude_lorentz,
    copper_drude, aluminum_drude, silicon_optical, silica_optical,
    germanium_optical, silicon_nitride_optical,
    // Casimir utilities
    reflection_te, reflection_tm, lifshitz_integrand_te,
    // Database
    MaterialEntry, MaterialType as OpticalMaterialType, get_material, list_materials,
};
