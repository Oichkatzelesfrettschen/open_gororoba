//! Schafer (1945) -- Division Algebras of Order 16.
//!
//! Implements the mathematical content from:
//! Schafer, R.D. "On a Construction for Division Algebras of Order 16",
//! Bull. Amer. Math. Soc., 51, 1945, pp. 532-534.
//!
//! Mirrors: proofs/theories/SchaferDivAlg16.v (0 Admitted)
//!
//! # Library Usage
//!
//! ```rust
//! use schafer_1945::types::test_division;
//!
//! let t = test_division(8);
//! assert!(t.is_division);
//!
//! let t = test_division(16);
//! assert!(!t.is_division);
//! assert!(t.zd_witness.is_some());
//! ```
//!
//! # Key results
//!
//! - The standard CD process over R does NOT yield a division algebra at dim 16
//! - A modified CD process with non-scalar g CAN yield division algebras over
//!   other fields (e.g., Q) when n(g) is not a perfect square

pub mod division_test;
pub mod modified_cd;
pub mod types;

// Top-level re-exports
pub use types::{test_division, DivisionTest, ModifiedCDAlgebra};
