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
//!
//! Higher Cayley-Dickson and speculative lanes in this crate are maintained as
//! control, falsification, and regression infrastructure. They remain useful,
//! but they are not the primary forward bridge architecture.

pub mod algebraic_dynamics;
pub mod bell_inequality;
pub mod billiard_stats;
pub mod cayley_dickson_structs;
pub mod cd_black_hole;
pub mod cd_external;
pub mod cd_graph_imbalance;
pub mod cd_tower_violations;
pub mod dark_inflaton;
pub mod dark_sector_quantum_numbers;
pub mod e10_octonion;
pub mod emanation;
pub mod exact_chsh_bound;
pub mod experimental_predictions;
pub mod f4_casimir;
pub mod fractal_dimension;
pub mod golay_code;
pub mod higher_cd;
#[cfg(test)]
mod invariance_tests;
pub mod leech_lattice;
pub mod leech_pathion;
pub mod lepton_mass_hierarchy;
pub mod majorana_braiding;
pub mod matrix_representation;
pub mod moonshine;
pub mod moonshine_embedding;
pub mod neutrino_sector;
pub mod novel_algorithms;
pub mod particle_physics;
pub mod quantum_geometry;
pub mod quantum_state;
pub mod quark_sector;
pub mod recursive_frameworks;
pub mod renormalization_group;
pub mod representation_closure;
pub mod sedenion_subalgebras;
pub mod sieve_scan;
pub mod so7_drift;
pub mod speculative;
pub mod su5_gut;
pub mod su_n_generators;
pub mod superpartner_pairing;
pub mod three_fermion_generations;
pub mod topological_associator_flux;
pub mod voudon_stabilizer;
pub mod wow_signal;
pub mod zd_flat_band_fraction;
