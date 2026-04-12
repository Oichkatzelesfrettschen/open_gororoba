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
pub mod chrono_turbulence;
pub mod correlation_measures;
pub mod deka_voudon_qec;
pub mod diamond_nv;
pub mod fractional_schrodinger;
pub mod gross_pitaevskii;
pub mod grover;
pub mod hamiltonian_evolution;
#[cfg(feature = "sparse-hamiltonians")]
pub mod hamiltonian_sparse;
pub mod hardware;
pub mod harper_chern;
pub mod holographic;
pub mod holographic_entropy;
pub mod hybrid_network;
pub mod hypothesis_search;
pub mod intention_operator;
pub(crate) mod kahan;
pub mod magnonic;
pub mod magnonic_crystal;
pub mod measurement_formalism;
pub mod mera;
pub mod mps;
pub mod orca_memory;
pub mod peps;
pub mod pseudospectrum;
pub mod qua_ten_net_bridge;
pub mod quantum_frequency_conversion;
pub mod scaling_laws;
pub mod spinor_mechanics;
pub mod superfluid;
pub mod tensor_network_classical;
pub mod tensor_networks;
pub mod tight_binding;
pub mod two_fluid;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use faer::c64 as Complex64;

pub use pseudospectrum::{PseudospectrumResult, fractional_laplacian_pseudospectrum};

pub use mera::{
    MeraLayer, MeraScalingResult, bootstrap_slope_ci, build_mera_structure, fit_log_scaling,
    mera_entropy_estimate, mera_entropy_scaling_analysis, von_neumann_entropy,
};

pub use holographic::{
    AbsorberLayer, AreaLawResult, BekensteinBoundResult, EntropyScalingResult, MinCutResult,
    RTLattice, absorber_channel_capacity, absorber_effective_radius, absorber_energy,
    analyze_entropy_scaling, bekenstein_bound_bits, compute_min_cut, verify_area_law,
    verify_bekenstein_bound,
};

pub use fractional_schrodinger::{
    EvolutionResult, PropagatorResult, VariationalResult, gaussian_propagator,
    imaginary_time_ground_state, levy_propagator, propagator_l2_error, split_operator_evolve,
    variational_ground_state,
};

pub use tensor_network_classical::{
    CircuitEvolutionResult, EntropyResult, TensorNetworkState, bell_state_entropy,
    ghz_state_entropy, prepare_bell_state, prepare_ghz_state, simulate_random_circuit,
};

pub use harper_chern::{
    ButterflyResult, ChernResult, fhs_chern_numbers, harper_hamiltonian, hofstadter_chern_map,
    reduced_fractions, verify_chern_sum_zero, verify_diophantine,
};

pub use tight_binding::{
    BravaisLattice2D, FlatBandInfo, Hopping, OrbitalSite, TightBindingModel, Valley, Vec2,
    band_chern_number, detect_flat_bands, fhs_berry_curvature, hexagonal_high_symmetry_path,
    hexagonal_symmetry_labels, valley_chern_number,
};

pub use magnonic::{
    MagnonicCrystalGeometry, YigParams, ghz_to_rad_per_s, hp_validity_bound,
    magnon_frequency_homogeneous, rad_per_s_to_ghz, yig_effective_spin,
};

pub use magnonic_crystal::{
    InversionBreakingParams, MagnonicBandResult, MagnonicTBParams, build_domain_wall_supercell,
    build_magnonic_9band, compute_magnonic_bands, point_defect_modes,
};

pub use mps::{MatrixProductState, MpsTensor};

pub use peps::{Peps, PepsTensor};

pub use tensor_networks::{EntanglementMeasure, estimate_memory_bytes, suggest_representation};

pub use qua_ten_net_bridge::{
    TruncatedSVD, contract_network, estimate_contraction_cost, tensor_contract, truncate_mps_bond,
    truncated_svd,
};

pub use casimir::{
    AdditivityResult,
    C,
    CASIMIR_COEFF,
    // PFA validity guard system (Emig et al. 2006)
    CasimirError,
    CasimirForceResult,
    // Derivative expansion error estimates (Fosco et al. 2024)
    DeCoefficients,
    DerivativeExpansionResult,
    // Lifshitz theory with dielectric functions
    DielectricModel,
    HBAR,
    LifshitzResult,
    // Additivity API (Xu et al. 2022)
    PfaAccuracy,
    PfaValidityInfo,
    Plate,
    // Strict spring constant / gain modes with error amplification
    SPRING_CONSTANT_ERROR_FACTOR,
    Sphere,
    SpherePlateSphere,
    SpringConstantResult,
    SweepResult,
    // Three-body transistor dynamics
    ThreeBodyResult,
    TransistorResult,
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
};

pub use grover::{
    GroverConfig, GroverResult, amplitude_amplification, apply_diffusion, apply_oracle,
    grover_iterate, grover_search, grover_search_indices, optimal_iterations, success_probability,
    theoretical_amplitude, theoretical_success_probability, top_candidates, uniform_superposition,
};

pub use hypothesis_search::{
    Hypothesis, HypothesisSearchResult, OraclePredicate, QuantumHypothesisSearch, ThresholdOracle,
    quantum_grid_search, quantum_hypothesis_search,
};

pub use hardware::{
    CoherenceTimes, ErrorRates, GateTiming, HardwareProfile, IdealHardware, NativeGate,
    NeutralAtomProfile, QubitTopology, SuperconductingProfile, SuperconductingVendor,
    TrappedIonProfile,
};

pub use hamiltonian_evolution::HamiltonianND;
#[cfg(feature = "sparse-hamiltonians")]
pub use hamiltonian_sparse::{build_sparse_hamiltonian, build_sparse_hamiltonian_coo};
pub mod lattice_qec_bridge;
pub mod qec_boxkite;
pub mod stabilizer_like;

pub use verified_core::coupler_manifold::{
    CouplerJacobian, CouplerPoint, IdentifiabilityAudit, mipt, qec, suppression_elasticity,
    suppression_factor, tree_geometry,
};
