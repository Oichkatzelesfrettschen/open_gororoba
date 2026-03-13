//! cd_kernel: Pure Cayley-Dickson arithmetic kernel.
//!
//! Recursive multiplication, conjugation, norm, zero-divisor search,
//! associator computation, basis product signs, and SIMD variants for
//! hypercomplex algebras of any power-of-two dimension.
//!
//! This crate is the foundational layer of the algebra stack.  It has
//! no dependency on analysis, physics, lie, or experimental modules.

pub mod avx2_primitives;
pub mod cayley_dickson;
pub mod error;
pub mod mult_table;
pub mod parallel_primitives;
pub mod traits;
#[cfg(target_arch = "x86_64")]
pub mod x87_ext80;
#[cfg(target_arch = "x86_64")]
pub mod x87_jacobi_kernels;
pub mod x87_primitives;
#[cfg(target_arch = "x86_64")]
pub mod x87_transcendentals;

// Re-export core types at crate root for ergonomic imports.
pub use error::{AlgebraError, AlgebraResult};
pub use parallel_primitives::{
    ParallelReductionStrategy, PhysicalCorePlan, parallel_dot, parallel_sum, physical_core_ids,
};
pub use traits::Hypercomplex;

pub use avx2_primitives::{avx2_dot, avx2_norm_sq, avx2_norm_sq_16, avx2_sum};
#[cfg(target_arch = "x86_64")]
pub use x87_ext80::{
    Ext80, PrecisionControl, RoundingControl, X87ControlGuard, X87ControlWord, X87StatusWord,
    X87ValueStatus,
};
#[cfg(target_arch = "x86_64")]
pub use x87_jacobi_kernels::{
    X87JacobiRotation80, givens_sincos_ext80, givens_sincos_f64, x87_atan2_sincos,
    x87_givens_diagonal_update, x87_givens_sincos,
};
pub use x87_primitives::{x87_dot, x87_horner, x87_norm_sq, x87_norm_sq_16, x87_sum};
#[cfg(target_arch = "x86_64")]
pub use x87_transcendentals::{
    X87ReductionResult, atan2_ext80, fprem1_ext80, pi_ext80, reduce_trig_argument_ext80,
    sincos_ext80, sincos_reduced_ext80, two_pi_ext80,
};

pub use cayley_dickson::{
    batch_associator_norms, batch_associator_norms_parallel, batch_associator_norms_sq,
    cd_associator, cd_associator_norm, cd_basis_mul_sign, cd_conjugate, cd_multiply,
    cd_multiply_simd, cd_norm_sq, cd_norm_sq_simd, count_pathion_zero_divisors, find_zero_divisors,
    left_mult_operator, measure_associator_density, zd_spectrum_analysis,
};
