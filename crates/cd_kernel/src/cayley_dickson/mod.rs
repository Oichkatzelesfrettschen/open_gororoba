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
pub mod cariow_factorization;
pub mod fast_associator;
pub mod predicates;
pub mod sedenion;
mod signature;
mod signs;
pub mod simd;
pub mod soa_cache;
mod symmetry;
pub mod trigintaduonion;
mod zero_divisors;

pub use arith::{
    cd_conjugate, cd_conjugate_into, cd_multiply, cd_multiply_conjugated, cd_multiply_into,
    cd_multiply_mut, cd_multiply_workspace_len, cd_norm_sq, left_mult_operator,
};
pub use associator::{
    AssociatorStats, associator_independence_stats, batch_associator_norms,
    batch_associator_norms_parallel, batch_associator_norms_sq, batch_octonion_basis_associators,
    batch_sedenion_associator_norms, batch_sedenion_associator_norms_parallel,
    batch_sliding_associator_norms, batch_sliding_associator_norms_parallel, cd_associator,
    cd_associator_norm, measure_associator_density, octonion_associator_simd,
};
pub use fast_associator::{
    AssociatorWorkspace, batch_fast_associator_norms_f32, fast_associator_norm_f32,
};
pub use signature::{
    CdSignature, SplitSignTable, cd_basis_mul_sign_split, cd_basis_mul_sign_split_iter,
    cd_mul_table_split, cd_multiply_split,
};
pub use signs::{SignTable, cd_basis_mul_sign, cd_basis_mul_sign_iter};
pub use simd::{
    // f32 quantized CD multiply for high-dimensional (256D+) performance
    batch_sliding_associator_norms_f32,
    cd_associator_norm_f32,
    // Zero-alloc fused CD multiply (steinmarder Instant-NGP pattern)
    cd_multiply_f32_fused,
    cd_multiply_f32_into,
    cd_multiply_f32_workspace_size,
    cd_multiply_flat_into,
    cd_multiply_simd,
    cd_norm_sq_simd,
    octonion_multiply_flat,
    octonion_multiply_scalar_flat,
    quaternion_multiply_flat,
    quaternion_multiply_scalar_flat,
    sedenion_multiply_flat,
};
pub use symmetry::{cross_generational_friction, gourlay_epsilon, gourlay_psi, gourlay_psi_n};
pub use zero_divisors::{
    GeneralFormZD, count_pathion_zero_divisors, find_zero_divisors, find_zero_divisors_3blade,
    find_zero_divisors_3blade_sign_table, find_zero_divisors_general_form,
    find_zero_divisors_parallel, find_zero_divisors_sign_table, is_zero_divisor_koebisu,
    koebisu_d1, koebisu_d2, zd_spectrum_analysis,
};

#[cfg(test)]
pub(crate) fn assert_batch_sedenion_associator_matches_recursive() {
    let mut vectors = Vec::new();
    for i in 0..10 {
        let mut v = [0.0_f64; 16];
        for (j, item) in v.iter_mut().enumerate() {
            *item = (i * 16 + j) as f64 * 0.1;
        }
        vectors.push(v);
    }

    let batch_results = batch_sedenion_associator_norms(&vectors);
    let parallel_results = batch_sedenion_associator_norms_parallel(&vectors);

    assert_eq!(batch_results.len(), 8);
    assert_eq!(parallel_results.len(), 8);

    for i in 0..8 {
        let expected = cd_associator_norm(&vectors[i], &vectors[i + 1], &vectors[i + 2]);
        assert!(
            (batch_results[i] - expected).abs() < 1e-10,
            "at index {}, got {}, expected {}",
            i,
            batch_results[i],
            expected
        );
        assert!((parallel_results[i] - expected).abs() < 1e-10);
    }
}

#[cfg(test)]
mod tests;
