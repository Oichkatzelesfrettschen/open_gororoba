//! Analysis modules for Cayley-Dickson algebras.
//!
//! Most modules are re-exported from the `algebra_analysis` crate.
//! Two modules remain in algebra_core due to cross-module dependencies:
//! - `test_wedged_validation`: depends on `construction::clifford`
//! - `legacy_crossval`: uses `construction::cayley_dickson`; filtration test in algebra_analysis

// Re-export all algebra_analysis modules at their original paths.
pub use algebra_analysis::annihilator;
pub use algebra_analysis::boxkites;
pub use algebra_analysis::codebook;
pub use algebra_analysis::entropy_census;
pub use algebra_analysis::fractal_analysis;
pub use algebra_analysis::graph_projections;
pub use algebra_analysis::grassmannian;
pub use algebra_analysis::homotopy_algebra;
pub use algebra_analysis::prefix_chain_theorem;
pub use algebra_analysis::projective_geometry;
pub use algebra_analysis::reggiani;
pub use algebra_analysis::stiefel;
pub use algebra_analysis::stochastic;
pub use algebra_analysis::subalgebra;
pub use algebra_analysis::zd_graphs;

// These modules stay in algebra_core (cross-module deps on construction/experimental).
pub mod test_wedged_validation;
pub mod legacy_crossval;
