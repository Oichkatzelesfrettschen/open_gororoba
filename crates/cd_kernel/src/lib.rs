//! cd_kernel: Pure Cayley-Dickson arithmetic kernel.
//!
//! Recursive multiplication, conjugation, norm, zero-divisor search,
//! associator computation, basis product signs, and SIMD variants for
//! hypercomplex algebras of any power-of-two dimension.
//!
//! This crate is the foundational layer of the algebra stack.  It has
//! no dependency on analysis, physics, lie, or experimental modules.

pub mod cayley_dickson;
pub mod error;
pub mod mult_table;
pub mod traits;

// Re-export core types at crate root for ergonomic imports.
pub use error::{AlgebraError, AlgebraResult};
pub use traits::Hypercomplex;

pub use cayley_dickson::{
    batch_associator_norms, batch_associator_norms_parallel, batch_associator_norms_sq,
    cd_associator, cd_associator_norm, cd_basis_mul_sign, cd_conjugate, cd_multiply,
    cd_multiply_simd, cd_norm_sq, cd_norm_sq_simd, count_pathion_zero_divisors, find_zero_divisors,
    left_mult_operator, measure_associator_density, zd_spectrum_analysis,
};
