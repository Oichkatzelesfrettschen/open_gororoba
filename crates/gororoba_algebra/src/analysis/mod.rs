//! Analysis modules for Cayley-Dickson algebras.
//!
//! Most modules are re-exported from the `algebra_analysis` crate.
//! Two modules remain in gororoba_algebra due to cross-module dependencies:
//! - `test_wedged_validation`: depends on `construction::clifford`
//! - `legacy_crossval`: uses `construction::cayley_dickson`; filtration test in algebra_analysis

// Re-export all algebra_analysis modules at their original paths.
pub use algebra_analysis::{
    annihilator, boxkites, codebook, entropy_census, fractal_analysis, graph_projections,
    grassmannian, homotopy_algebra, prefix_chain_theorem, projective_geometry, reggiani, stiefel,
    stochastic, subalgebra, zd_graphs,
};

// These modules stay in gororoba_algebra (cross-module deps on construction/experimental).
pub mod legacy_crossval;
pub mod test_wedged_validation;
