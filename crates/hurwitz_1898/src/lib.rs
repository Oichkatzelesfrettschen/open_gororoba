//! Hurwitz (1898) -- Composition of Quadratic Forms.
//!
//! Implements all computable content from:
//! Hurwitz, A. "Ueber die Composition der quadratischen Formen von beliebig
//! vielen Variablen", Nachr. Ges. Wiss. Goettingen, 1898, pp. 309-316.
//!
//! Mirrors: proofs/theories/HurwitzTheorem.v (443 lines, 0 Admitted)
//!
//! # Library Usage
//!
//! ```rust
//! use hurwitz_1898::types::{classify, HurwitzClassification};
//!
//! let c = classify(8);
//! assert!(c.is_composition);
//! assert_eq!(c.hurwitz_radon, 8);
//! ```
//!
//! # Modules
//!
//! - [`types`]: Core types (HurwitzClassification, CliffordSystem) + cd_kernel re-exports
//! - [`composition`]: Norm multiplicativity verification at dims 1, 2, 4, 8, 16
//! - [`dimension_bound`]: Hurwitz's 2^{n-2} <= n^2 and refined skew-symmetric bound
//! - [`clifford`]: Clifford relation structure (eq.9) for the B_i generators
//! - [`hurwitz_radon`]: The Hurwitz-Radon function rho(n) and generalized composition

pub mod clifford;
pub mod composition;
pub mod dimension_bound;
pub mod hurwitz_radon;
pub mod types;

// Top-level re-exports for convenience
pub use types::{CliffordSystem, CliffordVerification, HurwitzClassification, classify};
