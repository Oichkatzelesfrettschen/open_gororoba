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
pub mod lattice_codebook;
pub mod lloyd_max;
pub mod mult_table;
pub mod parallel_primitives;
pub mod traits;
pub mod turboquant;
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
    X87ReductionResult, angular_separation_arcsec_ext80_deg, atan2_ext80, fprem1_ext80, pi_ext80,
    reduce_trig_argument_ext80, sincos_ext80, sincos_reduced_ext80, two_pi_ext80,
};

pub use cayley_dickson::{
    batch_associator_norms, batch_associator_norms_parallel, batch_associator_norms_sq,
    batch_octonion_basis_associators, batch_sedenion_associator_norms,
    batch_sedenion_associator_norms_parallel, batch_sliding_associator_norms,
    batch_sliding_associator_norms_parallel, cd_associator, cd_associator_norm, cd_basis_mul_sign,
    cd_conjugate, cd_multiply, cd_multiply_simd, cd_norm_sq, cd_norm_sq_simd,
    count_pathion_zero_divisors, cross_generational_friction, find_zero_divisors, gourlay_epsilon,
    gourlay_psi, gourlay_psi_n, is_zero_divisor_koebisu, koebisu_d1, koebisu_d2,
    left_mult_operator, measure_associator_density, zd_spectrum_analysis,
    // f32 quantized CD (37x faster at 256D+, from TurboQuant precision insight)
    batch_sliding_associator_norms_f32, cd_associator_norm_f32, cd_multiply_f32_into,
    // Workspace-based fast associator (3-buffer reuse, 3300x less allocation at 4096D)
    AssociatorWorkspace, batch_fast_associator_norms_f32, fast_associator_norm_f32,
    // Zero-alloc fused CD multiply (steinmarder Instant-NGP pattern)
    cd_multiply_f32_fused, cd_multiply_f32_workspace_size,
};

/// Unified dispatch: compute sliding associator norms at the requested precision.
/// Returns Vec<f64> regardless of internal precision (f32 results promoted to f64).
///
/// The f32 path uses the workspace-based fast associator (AssociatorWorkspace +
/// cd_multiply_f32_fused) for zero heap allocation during the hot loop. At 128D
/// this is 20-37x faster than f64 with identical results at 6 significant figures.
///
/// The f64 path uses rayon-parallel sliding norms (standard, well-tested).
pub fn batch_sliding_associator_norms_dispatch(
    embedded: &[Vec<f64>],
    dim: usize,
    precision: &str,
) -> Vec<f64> {
    match precision {
        "f32" => {
            let embedded_f32: Vec<Vec<f32>> = embedded
                .iter()
                .map(|v| v.iter().map(|&x| x as f32).collect())
                .collect();
            // Use workspace-based fast associator: zero alloc per triplet,
            // fused CD multiply, rayon map_init per-thread workspace.
            batch_fast_associator_norms_f32(&embedded_f32, dim)
                .into_iter()
                .map(|x| x as f64)
                .collect()
        }
        _ => batch_sliding_associator_norms_parallel(embedded, dim),
    }
}
