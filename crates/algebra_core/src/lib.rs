//! algebra_core: Cayley-Dickson algebras, Clifford algebras, and algebraic structures.
//!
//! This crate provides high-performance implementations of:
//! - Cayley-Dickson multiplication for any power-of-2 dimension
//! - Associator computation and batch operations
//! - Zero-divisor search algorithms
//! - Clifford algebra Cl(8) for particle physics
//! - E8 lattice and root system computations
//! - Box-kite symmetry structures (de Marrais)
//!
//! # Literature
//! - de Marrais (2000): Box-kite structure of sedenion zero-divisors
//! - Furey et al. (2024): Cl(8) -> 3 generations
//! - Reggiani (2024): Geometry of sedenion zero divisors

pub mod cayley_dickson;
pub mod clifford;
pub mod zd_graphs;
pub mod e8_lattice;
pub mod boxkites;
pub mod octonion_field;

// Re-export core algebra functions
pub use cayley_dickson::{
    cd_multiply, cd_conjugate, cd_norm_sq, cd_associator, cd_associator_norm,
    batch_associator_norms, batch_associator_norms_sq, batch_associator_norms_parallel,
    left_mult_operator, find_zero_divisors, measure_associator_density,
    zd_spectrum_analysis, count_pathion_zero_divisors,
};

pub use clifford::{
    pauli_matrices, gamma_matrices_cl8, verify_clifford_relation,
    GammaMatrix, CliffordAlgebra,
};

pub use zd_graphs::{
    build_zd_interaction_graph, analyze_zd_graph, analyze_basis_participation,
    build_associator_graph, analyze_associator_graph, zd_shortest_path,
    zd_graph_diameter, ZdGraphAnalysis, BasisParticipationResult,
    AssociatorGraphResult,
};

pub use e8_lattice::{
    E8Lattice, E8Root, generate_e8_roots, e8_cartan_matrix,
    e8_weyl_group_order, compute_e8_inner_products,
};

pub use boxkites::{
    BoxKite, find_box_kites, analyze_box_kite_symmetry,
    BoxKiteSymmetryResult,
};

pub use octonion_field::{
    Octonion, FieldParams, EvolutionResult, DispersionResult,
    FANO_TRIPLES, build_structure_constants,
    oct_multiply, oct_conjugate, oct_norm_sq,
    hamiltonian, force, stormer_verlet_step, noether_charges,
    evolve, gaussian_wave_packet, standing_wave, measure_dispersion,
};
