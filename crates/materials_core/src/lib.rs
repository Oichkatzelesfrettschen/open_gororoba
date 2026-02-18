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
pub mod crystal_symmetry;
pub mod effective_medium;
pub mod featurizer;
pub mod metamaterial;
pub mod optical_database;
pub mod periodic_table;
pub mod tabulated_nk;
pub mod tang_mass;
pub mod viscosity_database;

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

pub use viscosity_database::{
    from_tau, get_coupling_regime, get_lambda, get_material as get_viscosity_material,
    get_materials_by_phase, get_quantum_fluids, list_lambda_materials,
    list_materials as list_viscosity_materials, load_lambda_results, load_viscosity_database,
    reynolds_number, to_lattice_units, LambdaResult, MaterialViscosity,
};

pub use optical_database::{
    // Constants
    C, EPS_0, EV_TO_RADS, HBAR_EV_S, K_B_EV, E_CHARGE, M_E_KG,
    // Conversions
    ev_to_omega, omega_to_ev, wavelength_to_omega,
    // Models
    DrudeLorentzParams, DrudeParams,
    ExtendedDrudeParams, ScatteringModel,
    LorentzOscillator as OpticalLorentzOscillator,
    UniaxialOptical,
    // Database types
    CasimirModelFlag, MaterialEntry, MaterialType as OpticalMaterialType,
    // Database access
    get_material, list_materials,
    // Pre-defined metals (original)
    gold_drude, gold_drude_lorentz, gold_rakic_ld,
    silver_drude, silver_drude_lorentz,
    copper_drude, copper_drude_lorentz,
    aluminum_drude, aluminum_drude_lorentz,
    // Rakic metals
    beryllium_drude, beryllium_drude_lorentz,
    chromium_drude, chromium_drude_lorentz,
    nickel_drude, nickel_drude_lorentz,
    palladium_drude, palladium_drude_lorentz,
    platinum_drude, platinum_drude_lorentz,
    titanium_drude, titanium_drude_lorentz,
    tungsten_drude, tungsten_drude_lorentz,
    // Semiconductors / dielectrics
    silicon_optical, germanium_optical, silica_optical, silica_casimir_optical, silicon_nitride_optical,
    // C-418 gap materials
    alumina_optical, diamond_optical, quartz_optical, tio2_optical,
    // Titanates
    tio_optical, srtio3_optical, srtio3_doped_optical, latio3_optical,
    // TCOs
    azo_optical, doped_silicon_optical,
    // Tungsten oxide family
    wo3_optical, wo3_x_optical, cs_wo3_optical, cs_wo3_uniaxial,
    cawo4_optical, pbwo4_optical,
    // Casimir utilities (legacy)
    lifshitz_integrand_te, reflection_te, reflection_tm,
    casimir_energy_density, casimir_force_density, casimir_energy_ideal, casimir_eta,
    // Casimir utilities (correct Lifshitz formula, Sprint 45)
    casimir_lifshitz_energy, casimir_lifshitz_force, casimir_lifshitz_eta,
    casimir_drude_plasma_discrepancy,
};

pub use tabulated_nk::{
    // Tabulated n,k data structs
    TabulatedNK, PhysicalProperties,
    // J&C 1972 datasets
    gold_jc_nk, silver_jc_nk, copper_jc_nk,
    // Database access
    get_tabulated_nk, get_physical_properties,
    // Casimir utilities using tabulated data + KK transform
    casimir_lifshitz_energy_tabulated,
};
