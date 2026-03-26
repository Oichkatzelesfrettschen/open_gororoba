#![allow(clippy::items_after_test_module)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]

//! algebra_analysis: Zero-divisor analysis, box-kites, codebook/lattice,
//! projective geometry, homotopy algebra, grassmannian, fractal/stochastic
//! for Cayley-Dickson algebras.

#[allow(dead_code)]
pub mod dd_jacobi;
#[allow(dead_code)]
pub mod double_double;
mod jacobi_shared;
pub mod precision_policy;
pub mod reference_jacobi;
#[cfg(target_arch = "x86_64")]
pub mod x87_jacobi;

pub mod annihilator;
pub mod associator_entropy;
pub mod avt;
pub mod avt_entropy;
pub mod block_jacobi;
pub mod boxkite_alignment;
pub mod boxkites;
pub mod codebook;
pub mod crystal_bands;
pub mod entropy_census;
pub mod flat_band_localization;
pub mod fractal_analysis;
pub mod graph_projections;
pub mod grassmannian;
pub mod homotopy_algebra;
pub mod legacy_crossval;
pub mod moreno;
pub mod partial_spectrum;
pub mod phase_transition;
pub mod prefix_chain_theorem;
pub mod projective_geometry;
pub mod quantum_chaos;
pub mod reggiani;
pub mod riemann_resonance;
pub mod sedenion_lifting;
pub mod sky_mapping;
pub mod sparse;
pub mod spectral_dimension;
pub mod spectrum_solvers;
pub mod stiefel;
pub mod stochastic;
pub mod subalgebra;
pub mod test_support;
pub mod zd_graphs;
