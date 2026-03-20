//! Cayley-Dickson algebra operations.
//!
//! Implements recursive multiplication for hypercomplex algebras of any
//! power-of-two dimension: reals (1), complex (2), quaternions (4),
//! octonions (8), sedenions (16), pathions (32), etc.
//!
//! The multiplication formula (a,b)(c,d) = (ac - d*b, da + bc*) follows
//! the standard Cayley-Dickson doubling construction.

mod arith;
mod associator;
pub mod sedenion;
mod signature;
mod signs;
mod simd;
mod zero_divisors;

pub use arith::{
    cd_conjugate, cd_multiply, cd_multiply_conjugated, cd_multiply_into, cd_multiply_mut,
    cd_norm_sq, left_mult_operator,
};
pub use associator::{
    AssociatorStats, associator_independence_stats, batch_associator_norms,
    batch_associator_norms_parallel, batch_associator_norms_sq, cd_associator, cd_associator_norm,
    measure_associator_density,
};
pub use signature::{
    CdSignature, SplitSignTable, cd_basis_mul_sign_split, cd_basis_mul_sign_split_iter,
    cd_mul_table_split, cd_multiply_split,
};
pub use signs::{SignTable, cd_basis_mul_sign, cd_basis_mul_sign_iter};
pub use simd::{
    cd_multiply_simd, cd_norm_sq_simd, octonion_multiply_flat, quaternion_multiply_flat,
    sedenion_multiply_flat,
};
pub use zero_divisors::{
    GeneralFormZD, count_pathion_zero_divisors, find_zero_divisors, find_zero_divisors_3blade,
    find_zero_divisors_general_form, find_zero_divisors_parallel, zd_spectrum_analysis,
};

#[cfg(test)]
mod tests;
