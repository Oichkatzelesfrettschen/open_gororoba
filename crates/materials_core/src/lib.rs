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
//!
//! # Canonical docs
//!
//! For materials and metamaterial work, the canonical documentation surfaces
//! are crate rustdoc, focused Markdown docs under `docs/`, and the claim and
//! experiment registries. Generated LaTeX under `docs/latex/` is publication
//! packaging, not the primary source of current implementation truth.

pub mod baselines;
pub mod cif_parser;
pub mod crystal_symmetry;
pub mod e8_crystal_bridge;
pub mod effective_medium;
pub mod featurizer;
pub mod landy_absorber;
pub mod liquid_crystal;
pub mod metamaterial;
pub mod nonlocal_metamaterial;
pub mod optical_database;
pub mod pathion_toy_mapping;
pub mod periodic_table;
pub mod tabulated_nk;
pub mod tang_mass;
pub mod viscosity_database;

pub use metamaterial::{
    MaterialType, MetamaterialLayer, VerificationResult, ZdToLayerMapping, build_absorber_stack,
    canonical_sedenion_zd_pairs, classify_material_type, map_zd_norm_to_thickness,
    map_zd_pair_to_layer, map_zd_to_refractive_index, verify_physical_realizability,
};

pub use nonlocal_metamaterial::{
    AssessorNode, AssessorTopology, FloquetEffectiveModel, LcAdmittanceModel, M3ProjectionConfig,
    MaterialCalibrationRecord, SyntheticCouplingModel, SyntheticCouplingReport,
    find_calibration_record, load_calibration_records,
};

pub use pathion_toy_mapping::{
    PathionToyLayer, build_pathion_diagonal, default_c053_layers, diagonal_to_layers,
    write_c053_summary,
};

pub use tang_mass::{
    DimensionScalingResult, GenerationAssignment, M_ELECTRON, M_MUON, M_TAU, MassNullTestResult,
    MassRatioPrediction, RATIO_E_MU, RATIO_E_TAU, RATIO_MU_TAU, basis_associator_norm,
    canonical_sedenion_assignments, dimension_scaling_analysis, find_best_assignment,
    mass_ratio_null_test, predict_mass_ratios,
};

pub use effective_medium::{
    KramersKronigResult, LorentzOscillator, TmmResult, bruggeman, bruggeman_array, drude,
    drude_lorentz, kramers_kronig_check, maxwell_garnett, maxwell_garnett_array, tmm_reflection,
    tmm_spectrum,
};

pub use periodic_table::{CrystalStructure, Element, get_element, get_element_by_z};

pub use featurizer::{
    CompositionFeatures, PropertyStats, composition_fractions, feature_names, feature_vector,
    featurize, parse_formula,
};

pub use baselines::{RegressionResult, ols_fit, run_baseline, train_test_split};

pub use landy_absorber::{
    LandyParams, MagneticSlabResult, absorber_at_frequency,
    absorption_spectrum as landy_absorption_spectrum, effective_n as landy_effective_n,
    impedance as landy_impedance, landy_2008_params, landy_n_layers, lorentz_epsilon, lorentz_mu,
};

pub use viscosity_database::{
    LambdaResult, MaterialViscosity, from_tau, get_coupling_regime, get_lambda,
    get_material as get_viscosity_material, get_materials_by_phase, get_quantum_fluids,
    list_lambda_materials, list_materials as list_viscosity_materials, load_lambda_results,
    load_viscosity_database, reynolds_number, to_lattice_units,
};

pub use optical_database::{
    // Constants
    C,
    // Database types
    CasimirModelFlag,
    // Models
    DrudeLorentzParams,
    DrudeParams,
    E_CHARGE,
    EPS_0,
    EV_TO_RADS,
    ExtendedDrudeParams,
    HBAR_EV_S,
    K_B_EV,
    LorentzOscillator as OpticalLorentzOscillator,
    M_E_KG,
    MaterialEntry,
    MaterialType as OpticalMaterialType,
    ScatteringModel,
    // Sellmeier dispersion (Son & Chekhova 2026)
    SellmeierParams,
    UniaxialOptical,
    // C-418 gap materials
    alumina_optical,
    aluminum_drude,
    aluminum_drude_lorentz,
    // TCOs
    azo_optical,
    // Rakic metals
    beryllium_drude,
    beryllium_drude_lorentz,
    casimir_drude_plasma_discrepancy,
    casimir_energy_density,
    casimir_energy_ideal,
    casimir_eta,
    casimir_force_density,
    // Casimir utilities (correct Lifshitz formula, Sprint 45)
    casimir_lifshitz_energy,
    casimir_lifshitz_eta,
    casimir_lifshitz_force,
    cawo4_optical,
    chromium_drude,
    chromium_drude_lorentz,
    copper_drude,
    copper_drude_lorentz,
    cs_wo3_optical,
    cs_wo3_uniaxial,
    diamond_optical,
    doped_silicon_optical,
    // Conversions
    ev_to_omega,
    fused_silica_sellmeier,
    germanium_optical,
    // Database access
    get_material,
    // Pre-defined metals (original)
    gold_drude,
    gold_drude_lorentz,
    gold_rakic_ld,
    latio3_optical,
    // Casimir utilities (legacy)
    lifshitz_integrand_te,
    linbo3_extraordinary_sellmeier,
    linbo3_ordinary_sellmeier,
    list_materials,
    nickel_drude,
    nickel_drude_lorentz,
    omega_to_ev,
    palladium_drude,
    palladium_drude_lorentz,
    pbwo4_optical,
    platinum_drude,
    platinum_drude_lorentz,
    quartz_optical,
    reflection_te,
    reflection_tm,
    silica_casimir_optical,
    silica_optical,
    silicon_nitride_optical,
    // Semiconductors / dielectrics
    silicon_optical,
    silver_drude,
    silver_drude_lorentz,
    srtio3_doped_optical,
    srtio3_optical,
    // Titanates
    tio_optical,
    tio2_optical,
    titanium_drude,
    titanium_drude_lorentz,
    tungsten_drude,
    tungsten_drude_lorentz,
    wavelength_to_omega,
    // Tungsten oxide family
    wo3_optical,
    wo3_x_optical,
};

pub use tabulated_nk::{
    PhysicalProperties,
    // Tabulated n,k data structs
    TabulatedNK,
    // Casimir utilities using tabulated data + KK transform
    casimir_lifshitz_energy_tabulated,
    copper_jc_nk,
    get_physical_properties,
    // Database access
    get_tabulated_nk,
    // J&C 1972 datasets
    gold_jc_nk,
    silver_jc_nk,
};
