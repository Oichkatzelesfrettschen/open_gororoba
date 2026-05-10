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
    batch_associator_norms, batch_associator_norms_parallel, batch_sedenion_associator_norms,
    batch_sedenion_associator_norms_parallel, cd_associator, cd_associator_norm, cd_conjugate,
    cd_multiply, cd_multiply_simd, cd_norm_sq, find_zero_divisors,
};

pub use cd_kernel::cayley_dickson::{SignTable, cd_basis_mul_sign, cd_multiply_into};

// -- analysis ------------------------------------------------------------
#[cfg(feature = "analysis")]
pub mod analysis;

#[cfg(feature = "analysis")]
pub use analysis::fractal_analysis::{calculate_hurst, generate_fbm, hurst_rs_analysis};

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
pub use construction::{
    auxiliary::{Rational, padic_distance},
    padic::vp_int,
};

// -- construction re-exports (core) -------------------------------------
#[cfg(feature = "core")]
pub use construction::exotic_octonions::{Bioctonion, DualOctonion, ParaOctonion};
#[cfg(feature = "core")]
pub use construction::signature_observables::{ObservableReading, ObservableSignatureRegime};
#[cfg(feature = "core")]
pub use construction::symmetric_composition::{OkuboElement, TrialityAction};

// -- lie re-exports ------------------------------------------------------
#[cfg(feature = "lie")]
pub use lie::e8::root_system::{E8Root, generate_e8_roots};
#[cfg(feature = "lie")]
pub use lie::group_theory::order_psl2_q;
#[cfg(feature = "lie")]
pub use lie::nilpotent_orbits::{jordan_type_nilpotent, nilpotency_index};

// -- physics re-exports --------------------------------------------------
#[cfg(feature = "physics")]
pub use physics::octonion_field::{
    FieldParams, Octonion, gaussian_wave_packet, oct_conjugate, oct_multiply, oct_norm_sq,
    stormer_verlet_step,
};
