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

pub mod baselines;
pub mod effective_medium;
pub mod featurizer;
pub mod metamaterial;
pub mod optical_database;
pub mod periodic_table;
pub mod tang_mass;

pub use metamaterial::{
    build_absorber_stack, canonical_sedenion_zd_pairs, classify_material_type,
    map_zd_norm_to_thickness, map_zd_pair_to_layer, map_zd_to_refractive_index,
    verify_physical_realizability, MaterialType, MetamaterialLayer, VerificationResult,
    ZdToLayerMapping,
};

pub use tang_mass::{
    basis_associator_norm, canonical_sedenion_assignments, dimension_scaling_analysis,
    find_best_assignment, mass_ratio_null_test, predict_mass_ratios, DimensionScalingResult,
    GenerationAssignment, MassNullTestResult, MassRatioPrediction, M_ELECTRON, M_MUON, M_TAU,
    RATIO_E_MU, RATIO_E_TAU, RATIO_MU_TAU,
};

pub use effective_medium::{
    bruggeman, bruggeman_array, drude, drude_lorentz, kramers_kronig_check, maxwell_garnett,
    maxwell_garnett_array, tmm_reflection, tmm_spectrum, KramersKronigResult, LorentzOscillator,
    TmmResult,
};

pub use periodic_table::{get_element, get_element_by_z, CrystalStructure, Element};

pub use featurizer::{
    composition_fractions, feature_names, feature_vector, featurize, parse_formula,
    CompositionFeatures, PropertyStats,
};

pub use baselines::{ols_fit, run_baseline, train_test_split, RegressionResult};

pub use optical_database::{
    aluminum_drude,
    copper_drude,
    ev_to_omega,
    germanium_optical,
    get_material,
    // Pre-defined materials
    gold_drude,
    gold_drude_lorentz,
    lifshitz_integrand_te,
    list_materials,
    omega_to_ev,
    // Casimir utilities
    reflection_te,
    reflection_tm,
    silica_optical,
    silicon_nitride_optical,
    silicon_optical,
    silver_drude,
    silver_drude_lorentz,
    // Conversions
    wavelength_to_omega,
    DrudeLorentzParams,
    // Models
    DrudeParams,
    LorentzOscillator as OpticalLorentzOscillator,
    // Database
    MaterialEntry,
    MaterialType as OpticalMaterialType,
    C,
    // Constants
    EV_TO_RADS,
    HBAR_EV_S,
};
