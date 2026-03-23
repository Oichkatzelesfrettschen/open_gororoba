//! Analysis modules for Cayley-Dickson algebras.
//!
//! Most modules are re-exported from the `algebra_analysis` crate.
//! One module remains in gororoba_algebra due to cross-module dependencies:
//! - `test_wedged_validation`: depends on `construction::clifford`
//!
//! `legacy_crossval` now delegates entirely to `algebra_analysis::legacy_crossval`.
//! The local copy used `crate::construction::cayley_dickson` which is a pure
//! re-export of `cd_kernel::cayley_dickson`, so both versions were identical
//! modulo import path. The algebra_analysis version is canonical.

// Re-export all algebra_analysis modules at their original paths.
pub use algebra_analysis::{
    annihilator, boxkites, codebook, entropy_census, fractal_analysis, graph_projections,
    grassmannian, homotopy_algebra, legacy_crossval, prefix_chain_theorem, projective_geometry,
    reggiani, stiefel, stochastic, subalgebra, zd_graphs,
};

// These modules stay in gororoba_algebra (cross-module deps on construction/experimental).
pub mod annihilators;
pub mod motif_summary;
pub mod numerical_stability;
pub mod test_wedged_validation;
pub mod zd_ecology;
