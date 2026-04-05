//! Dickson (1919) -- The Cayley-Dickson Construction.
//!
//! Implements all computable content from sections 1-5 of:
//! Dickson, L.E. "On Quaternions and Their Generalization and the History
//! of the Eight Square Theorem", Annals of Math., 20(3), 1919, pp. 155-171.
//!
//! Mirrors: proofs/theories/DicksonCDProcess.v (239 lines, 0 Admitted)
//!
//! # Library Usage
//!
//! ```rust
//! use dickson_1919::types::{CayleyDicksonLevel, PropertyReport, cd_inverse};
//!
//! let level = CayleyDicksonLevel::Octonion;
//! assert!(level.is_alternative());
//! assert!(level.is_division());
//! assert!(!level.is_associative());
//!
//! let report = PropertyReport::for_level(level);
//! println!("{report}");
//!
//! let q = vec![1.0, 2.0, 3.0, 4.0];
//! let inv = cd_inverse(&q).unwrap();
//! assert!(inv.len() == 4);
//! ```
//!
//! # Modules
//!
//! - [`types`]: Core types (CayleyDicksonLevel, PropertyReport, cd_inverse) + hurwitz integration
//! - [`cd_process`]: The CD doubling construction (eq.6)
//! - [`property_hierarchy`]: Computational property tests at each dimension

pub mod algebraic_identities;
pub mod cd_process;
pub mod property_hierarchy;
pub mod types;

// Top-level re-exports
pub use types::{CayleyDicksonLevel, PropertyReport, cd_inverse, verify_inverse};
