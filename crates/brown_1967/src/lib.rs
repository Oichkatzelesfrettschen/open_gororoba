//! Brown (1967) -- Generalized Cayley-Dickson Algebras.
//!
//! Extends the CD construction with arbitrary gamma parameters, enabling
//! division algebras at dim 16+ over suitable fields. Classifies algebras
//! up to isomorphism at dims 16, 32, 64.
//!
//! Mirrors: proofs/theories/BrownGeneralizedCD.v (0 Admitted)
//!
//! # Key results
//!
//! - Generalized CD product with gamma parameters (eq.2)
//! - Norm formula for generalized algebras (eq.1)
//! - Division criterion: gamma not a norm AND -gamma not a trace-zero norm (Thm 3)
//! - Classification: A_t iso A_t' iff base algebras iso and gammas related by squares (Thm 2)

pub mod division_criterion;
pub mod generalized_cd;
