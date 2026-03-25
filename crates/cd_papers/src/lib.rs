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
//! 4. **Wilmot (2025)**: Full CD structure. Triads, cycles, modes, ZD formula.
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
//! // Count zero divisors
//! let z = wilmot_2025::zero_divisors::zero_divisor_pair_count(1);
//! assert_eq!(z, 1260);
//! ```

/// Re-exports from all paper crates.
pub mod prelude {
    pub use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};
    pub use dickson_1919::{
        cd_inverse, verify_inverse, CayleyDicksonLevel, PropertyReport,
    };
    pub use hurwitz_1898::{
        classify, CliffordSystem, CliffordVerification, HurwitzClassification,
    };
    pub use schafer_1945::{test_division, DivisionTest, ModifiedCDAlgebra};
}

// Full crate re-exports for qualified access
pub use dickson_1919;
pub use hurwitz_1898;
pub use schafer_1945;
pub use wilmot_2025;
pub use brown_1967;
pub use brown_1972;
