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

pub mod mera;
pub mod holographic;
pub mod fractional_schrodinger;
pub mod tensor_network_classical;
pub mod harper_chern;
pub mod mps;
pub mod casimir;
pub mod grover;

pub use mera::{
    build_mera_structure, von_neumann_entropy, mera_entropy_estimate,
    fit_log_scaling, mera_entropy_scaling_analysis, bootstrap_slope_ci,
    MeraLayer, MeraScalingResult,
};

pub use holographic::{
    bekenstein_bound_bits, verify_bekenstein_bound,
    absorber_channel_capacity, absorber_effective_radius, absorber_energy,
    RTLattice, compute_min_cut, analyze_entropy_scaling, verify_area_law,
    BekensteinBoundResult, MinCutResult, EntropyScalingResult, AreaLawResult,
    AbsorberLayer,
};

pub use fractional_schrodinger::{
    levy_propagator, gaussian_propagator, propagator_l2_error,
    split_operator_evolve, imaginary_time_ground_state, variational_ground_state,
    EvolutionResult, VariationalResult, PropagatorResult,
};

pub use tensor_network_classical::{
    TensorNetworkState, EntropyResult, CircuitEvolutionResult,
    simulate_random_circuit, prepare_bell_state, prepare_ghz_state,
    bell_state_entropy, ghz_state_entropy,
};

pub use harper_chern::{
    ChernResult, ButterflyResult,
    reduced_fractions, harper_hamiltonian, fhs_chern_numbers,
    hofstadter_chern_map, verify_chern_sum_zero, verify_diophantine,
};

pub use mps::{
    MatrixProductState, MpsTensor,
};

pub use casimir::{
    Sphere, Plate, SpherePlateSphere,
    CasimirForceResult, TransistorResult, SweepResult,
    casimir_force_pfa, casimir_energy_pfa, pfa_is_valid, pfa_is_valid_at_accuracy,
    compute_casimir_forces, analyze_transistor, sweep_source_gap,
    casimir_force_with_corrections, finite_conductivity_correction, thermal_correction,
    // Additivity API (Xu et al. 2022)
    PfaAccuracy, AdditivityResult,
    force_sps_additive, cross_coupling_additive, transistor_gain_additive,
    nonadditivity_correction,
    // PFA validity guard system (Emig et al. 2006)
    CasimirError, PfaValidityInfo,
    check_pfa_validity, casimir_force_guarded, casimir_force_with_validity,
    casimir_spring_constant_guarded,
    // Derivative expansion error estimates (Fosco et al. 2024)
    DeCoefficients, DerivativeExpansionResult,
    casimir_force_with_de, estimate_de_error, max_gap_for_error,
    // Strict spring constant / gain modes with error amplification
    SPRING_CONSTANT_ERROR_FACTOR, SpringConstantResult,
    spring_constant_strict, transistor_gain_strict,
    spring_constant_with_diagnostics,
    // Lifshitz theory with dielectric functions
    DielectricModel, LifshitzResult,
    fresnel_tm_imaginary, fresnel_te_imaginary,
    lifshitz_pressure_plates, lifshitz_force_sphere_plate,
    lifshitz_force_ratio, lifshitz_sphere_plate,
    matsubara_frequency, thermal_wavelength,
    CASIMIR_COEFF, HBAR, C,
};

pub use grover::{
    GroverResult, GroverConfig,
    optimal_iterations, uniform_superposition,
    apply_oracle, apply_diffusion, grover_iterate,
    success_probability, top_candidates,
    grover_search, grover_search_indices, amplitude_amplification,
    theoretical_amplitude, theoretical_success_probability,
};
