//! quantum_core: MERA tensor networks, holographic entropy, Ryu-Takayanagi lattice.
//!
//! This crate provides:
//! - MERA (Multi-scale Entanglement Renormalization Ansatz) implementation
//! - von Neumann entropy calculations
//! - Ryu-Takayanagi min-cut entropy
//! - Bekenstein bound verification
//!
//! # Literature
//! - Vidal (2007): MERA original proposal
//! - Swingle (2012): MERA/AdS correspondence
//! - Ryu & Takayanagi (2006): Holographic entropy formula
//! - Bekenstein (1981): Entropy bounds

pub mod mera;
pub mod holographic;

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
