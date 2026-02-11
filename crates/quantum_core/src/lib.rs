//! quantum_core: MERA tensor networks, holographic entropy, fractional Schrodinger, Casimir, Grover.
//!
//! This crate provides:
//! - MERA (Multi-scale Entanglement Renormalization Ansatz) implementation
//! - von Neumann entropy calculations
//! - Ryu-Takayanagi min-cut entropy
//! - Bekenstein bound verification
//! - Fractional Schrodinger equation solver (Levy propagator, split-operator)
//! - Casimir sphere-plate-sphere system (Xu et al. 2022 transistor architecture)
//! - Grover's quantum search and amplitude amplification
//!
//! # Literature
//! - Vidal (2007): MERA original proposal
//! - Swingle (2012): MERA/AdS correspondence
//! - Ryu & Takayanagi (2006): Holographic entropy formula
//! - Bekenstein (1981): Entropy bounds
//! - Laskin (2000, 2002): Fractional Schrodinger equation
//! - Xu et al., Nature Communications 13, 6148 (2022): Casimir transistor
//! - Grover (1996): Quantum search algorithm

pub mod casimir;
pub mod fractional_schrodinger;
pub mod gross_pitaevskii;
pub mod grover;
pub mod hamiltonian_evolution;
pub mod hardware;
pub mod harper_chern;
pub mod holographic;
pub mod hypothesis_search;
pub mod mera;
pub mod mps;
pub mod peps;
pub mod qua_ten_net_bridge;
pub mod spinor_mechanics;
pub mod superfluid;
pub mod tensor_network_classical;
pub mod tensor_networks;
pub mod two_fluid;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use mera::{
    bootstrap_slope_ci, build_mera_structure, fit_log_scaling, mera_entropy_estimate,
    mera_entropy_scaling_analysis, von_neumann_entropy, MeraLayer, MeraScalingResult,
};

pub use holographic::{
    absorber_channel_capacity, absorber_effective_radius, absorber_energy, analyze_entropy_scaling,
    bekenstein_bound_bits, compute_min_cut, verify_area_law, verify_bekenstein_bound,
    AbsorberLayer, AreaLawResult, BekensteinBoundResult, EntropyScalingResult, MinCutResult,
    RTLattice,
};

pub use fractional_schrodinger::{
    gaussian_propagator, imaginary_time_ground_state, levy_propagator, propagator_l2_error,
    split_operator_evolve, variational_ground_state, EvolutionResult, PropagatorResult,
    VariationalResult,
};

pub use tensor_network_classical::{
    bell_state_entropy, ghz_state_entropy, prepare_bell_state, prepare_ghz_state,
    simulate_random_circuit, CircuitEvolutionResult, EntropyResult, TensorNetworkState,
};

pub use harper_chern::{
    fhs_chern_numbers, harper_hamiltonian, hofstadter_chern_map, reduced_fractions,
    verify_chern_sum_zero, verify_diophantine, ButterflyResult, ChernResult,
};

pub use mps::{MatrixProductState, MpsTensor};

pub use peps::{Peps, PepsTensor};

pub use tensor_networks::{estimate_memory_bytes, suggest_representation, EntanglementMeasure};

pub use qua_ten_net_bridge::{
    contract_network, estimate_contraction_cost, tensor_contract, truncate_mps_bond, truncated_svd,
    TruncatedSVD,
};

pub use casimir::{
    analyze_transistor,
    casimir_energy_pfa,
    casimir_force_guarded,
    casimir_force_pfa,
    casimir_force_with_corrections,
    casimir_force_with_de,
    casimir_force_with_validity,
    casimir_spring_constant_guarded,
    check_pfa_validity,
    compute_casimir_forces,
    cross_coupling_additive,
    estimate_de_error,
    finite_conductivity_correction,
    force_sps_additive,
    fresnel_te_imaginary,
    fresnel_tm_imaginary,
    lifshitz_force_ratio,
    lifshitz_force_sphere_plate,
    lifshitz_pressure_plates,
    lifshitz_sphere_plate,
    matsubara_frequency,
    max_gap_for_error,
    nonadditivity_correction,
    pfa_is_valid,
    pfa_is_valid_at_accuracy,
    spring_constant_strict,
    spring_constant_with_diagnostics,
    sweep_source_gap,
    thermal_correction,
    thermal_wavelength,
    three_body_casimir_dynamics,
    three_body_gain_quasistatic,
    three_body_gain_strict,
    transistor_gain_additive,
    transistor_gain_strict,
    AdditivityResult,
    // PFA validity guard system (Emig et al. 2006)
    CasimirError,
    CasimirForceResult,
    // Derivative expansion error estimates (Fosco et al. 2024)
    DeCoefficients,
    DerivativeExpansionResult,
    // Lifshitz theory with dielectric functions
    DielectricModel,
    LifshitzResult,
    // Additivity API (Xu et al. 2022)
    PfaAccuracy,
    PfaValidityInfo,
    Plate,
    Sphere,
    SpherePlateSphere,
    SpringConstantResult,
    SweepResult,
    // Three-body transistor dynamics
    ThreeBodyResult,
    TransistorResult,
    C,
    CASIMIR_COEFF,
    HBAR,
    // Strict spring constant / gain modes with error amplification
    SPRING_CONSTANT_ERROR_FACTOR,
};

pub use grover::{
    amplitude_amplification, apply_diffusion, apply_oracle, grover_iterate, grover_search,
    grover_search_indices, optimal_iterations, success_probability, theoretical_amplitude,
    theoretical_success_probability, top_candidates, uniform_superposition, GroverConfig,
    GroverResult,
};

pub use hypothesis_search::{
    quantum_grid_search, quantum_hypothesis_search, Hypothesis, HypothesisSearchResult,
    OraclePredicate, QuantumHypothesisSearch, ThresholdOracle,
};

pub use hardware::{
    CoherenceTimes, ErrorRates, GateTiming, HardwareProfile, IdealHardware, NativeGate,
    NeutralAtomProfile, QubitTopology, SuperconductingProfile, SuperconductingVendor,
    TrappedIonProfile,
};

pub use hamiltonian_evolution::HamiltonianND;
