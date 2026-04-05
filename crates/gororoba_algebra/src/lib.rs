#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]

//! gororoba_algebra: algebra facade over the Cayley-Dickson kernel and
//! higher-level analysis/physics modules.

// -- core ----------------------------------------------------------------
#[cfg(feature = "core")]
pub mod construction;
#[cfg(feature = "core")]
pub mod error;
#[cfg(feature = "core")]
pub mod traits;
#[cfg(feature = "core")]
pub mod universal_algebra;

#[cfg(feature = "core")]
pub use error::{AlgebraError, AlgebraResult};
#[cfg(feature = "core")]
pub use traits::Hypercomplex;

// Re-export common functions from cd_kernel for convenience
pub use cd_kernel::{
    batch_sedenion_associator_norms, batch_sedenion_associator_norms_parallel, cd_associator,
    cd_associator_norm, cd_conjugate, cd_multiply, cd_multiply_simd, cd_norm_sq,
};

pub use cd_kernel::cayley_dickson::{SignTable, cd_basis_mul_sign, cd_multiply_into};

// -- analysis ------------------------------------------------------------
#[cfg(feature = "analysis")]
pub mod analysis;

#[cfg(feature = "analysis")]
pub use analysis::fractal_analysis::hurst_rs_analysis;

// -- physics -------------------------------------------------------------
#[cfg(feature = "physics")]
pub mod physics;

#[cfg(feature = "physics")]
pub use physics::clifford::{kron, kron2, pauli_matrices};

// -- lie -----------------------------------------------------------------
#[cfg(feature = "lie")]
pub mod lie;

// -- gpu -----------------------------------------------------------------
#[cfg(feature = "gpu")]
pub mod gpu;

// -- types ---------------------------------------------------------------
pub use construction::auxiliary::{Rational, padic_distance};
