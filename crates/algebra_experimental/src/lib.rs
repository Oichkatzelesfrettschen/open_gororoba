//! algebra_experimental: Experimental algebraic structures.
//!
//! Emanation tables (de Marrais), E10-octonion bridge, billiard statistics,
//! Monstrous Moonshine, Golay code, Leech lattice, algebraic dynamics,
//! SO(7) drift analysis, and external CD data cross-validation.
//!
//! # Canonical docs
//!
//! This crate's rustdoc is part of the canonical repo documentation surface.
//! Prefer it, the mdBook, and the relevant registries over generated LaTeX
//! mirrors when you need the current meaning of an algebraic experiment or
//! boundary note.

pub mod algebraic_dynamics;
pub mod bell_inequality;
pub mod billiard_stats;
pub mod cd_external;
pub mod cd_tower_violations;
pub mod e10_octonion;
pub mod emanation;
pub mod golay_code;
pub mod higher_cd;
pub mod speculative;
pub mod leech_lattice;
pub mod leech_pathion;
pub mod majorana_braiding;
pub mod moonshine;
pub mod moonshine_embedding;
pub mod particle_physics;
pub mod quantum_geometry;
pub mod recursive_frameworks;
pub mod representation_closure;
pub mod so7_drift;
pub mod superpartner_pairing;
pub mod topological_associator_flux;
pub mod voudon_stabilizer;
pub mod wow_signal;
pub mod zd_flat_band_fraction;
pub mod fractal_dimension;
pub mod exact_chsh_bound;
pub mod f4_casimir;
pub mod three_fermion_generations;
pub mod lepton_mass_hierarchy;
pub mod su_n_generators;
pub mod sedenion_subalgebras;
pub mod matrix_representation;
pub mod cayley_dickson_structs;
pub mod quantum_state;
pub mod neutrino_sector;
#[cfg(test)]
mod invariance_tests;
pub mod renormalization_group;
pub mod quark_sector;
pub mod cd_graph_imbalance;
pub mod dark_sector_quantum_numbers;
pub mod sieve_scan;
pub mod dark_inflaton;
pub mod cd_black_hole;
pub mod su5_gut;
