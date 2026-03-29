//! Cayley-Dickson Paper Formalizations -- Meta-crate.
//!
//! Re-exports all paper-specific crates for convenient one-import access
//! to the full CD algebra formalization library.
//!
//! # Paper Chain (chronological)
//!
//! 1. **Hurwitz (1898)**: Composition of quadratic forms. Only dims 1,2,4,8.
//! 2. **Dickson (1919)**: The CD doubling construction. Property hierarchy.
//! 3. **Schafer (1945)**: Division algebras of order 16. Sedenions fail.
//! 4. **Brown (1967)**: Generalized CD algebras with arbitrary gamma parameters.
//! 5. **Brown (1972)**: Zero-divisor structure in higher-dimensional CD algebras.
//! 6. **Moreno (1997)**: Operator-theoretic zero-divisor criteria and decomposition.
//! 7. **de Marrais (2000)**: Assessors, box-kites, and XOR-organized ZD structure.
//! 8. **Wilmot (2025)**: Full CD structure. Triads, cycles, modes, ZD formula.
//!
//! # Usage
//!
//! ```rust
//! use cd_papers::prelude::*;
//!
//! // Classify dimension 8
//! let h = hurwitz_1898::classify(8);
//! assert!(h.is_composition);
//!
//! // Check property hierarchy
//! let level = CayleyDicksonLevel::Octonion;
//! assert!(level.is_alternative());
//! assert!(!level.is_associative());
//!
//! // Test division
//! let d = schafer_1945::test_division(16);
//! assert!(!d.is_division);
//!
//! // Moreno's decomposition computes operator-theoretic structure
//! let mut a = vec![0.0; 8];
//! a[1] = 1.0;
//! let decomp = moreno_1997::moreno_decomposition(&a, 8, 1e-10);
//! assert!(decomp.total_dim_check);
//!
//! // Count sedenion assessors
//! assert_eq!(de_marrais_2000::TOTAL_ASSESSORS, 42);
//!
//! // Count zero divisors
//! let z = wilmot_2025::zero_divisors::zero_divisor_pair_count(1);
//! assert_eq!(z, 1260);
//! ```

/// Re-exports from all paper crates.
pub mod prelude {
    pub use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};
    pub use de_marrais_2000::{all_assessors, O_TRIPS, S_TRIPS, TOTAL_ASSESSORS, TOTAL_ZD_PAIRS};
    pub use dickson_1919::{
        cd_inverse, verify_inverse, CayleyDicksonLevel, PropertyReport,
    };
    pub use hurwitz_1898::{
        classify, CliffordSystem, CliffordVerification, HurwitzClassification,
    };
    pub use moreno_1997::{
        find_zd_witness_moreno, has_eigenvalue_minus_2, is_zd_moreno, moreno_decomposition,
        verify_corollary_1_17, MorenoDecomposition,
    };
    pub use schafer_1945::{test_division, DivisionTest, ModifiedCDAlgebra};
}

// Full crate re-exports for qualified access
pub use dickson_1919;
pub use hurwitz_1898;
pub use moreno_1997;
pub use schafer_1945;
pub use wilmot_2025;
pub use brown_1967;
pub use brown_1972;
pub use de_marrais_2000;

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn hurwitz_dim8_is_composition() {
        let h = classify(8);
        assert!(h.is_composition);
    }

    #[test]
    fn sedenion_assessor_count() {
        assert_eq!(TOTAL_ASSESSORS, 42);
    }
}
