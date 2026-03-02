//! algebra_analysis: Zero-divisor analysis, box-kites, codebook/lattice,
//! projective geometry, homotopy algebra, grassmannian, fractal/stochastic
//! for Cayley-Dickson algebras.
//!
//! This crate depends on `cd_kernel` for core arithmetic and provides the
//! analytical layer: graph-based zero-divisor analysis, de Marrais box-kite
//! structures, codebook/lattice theory, projective geometry, A-infinity and
//! L-infinity homotopy algebras, Grassmannian manifolds, fractal analysis,
//! and stochastic processes.

pub mod annihilator;
pub mod boxkites;
pub mod codebook;
pub mod entropy_census;
pub mod fractal_analysis;
pub mod graph_projections;
pub mod grassmannian;
pub mod homotopy_algebra;
pub mod legacy_crossval;
pub mod prefix_chain_theorem;
pub mod projective_geometry;
pub mod reggiani;
pub mod stiefel;
pub mod stochastic;
pub mod subalgebra;
pub mod zd_graphs;
