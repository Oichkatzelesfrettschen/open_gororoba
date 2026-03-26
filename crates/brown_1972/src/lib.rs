//! Brown (1972) -- Zero Divisor Structure in Cayley-Dickson Algebras.
//!
//! THE foundational dissertation for zero divisor theory in CD algebras.
//! 89 pages, 9 chapters, the Major Theorem (7.15) that characterizes
//! exactly when AB = 0 in the sedenions.
//!
//! This is the paper that de Marrais (2000), Moreno (1997), and all
//! subsequent ZD work builds upon.
//!
//! # Key results
//!
//! - Norm defect: N(A)N(B) - N(AB) expressed in terms of octonionic halves
//! - ZD symmetry: AB=0 <=> BA=0 <=> conj(A)B=0 <=> ... (11 equivalences)
//! - ZD subspace: <A,B> is 3-dimensional, commutative Jordan, non-alternative
//! - Major Theorem: AB=0 iff N(a1)=N(a2) AND b2=[(a1*b1)*a2]/N(a1) AND )a1,b1,a2(=0
//! - Star operator: A* = a1 - ea2 (inner doubling conjugation)
//!
//! # Chapter map
//!
//! - Chapter III, pp. 15-16: norm symmetry and involution identities.
//!   Rust lane: `norm_symmetry.rs`. Dedicated Rocq lane still open.
//! - Chapter IV, pp. 20-22: flexibility and associator formulas.
//!   Rocq support exists, but Brown-numbered theorem surfacing is still partial.
//! - Chapter V, pp. 27-30: exponent properties and power laws.
//!   Rust lane: `exponent_properties.rs`. Dedicated Rocq lane still open.
//! - Chapter VI, pp. 30-37: basis element identities and restricted Moufang laws.
//!   Rust lane: `basis_element_properties.rs`. Direct Rocq chapter lane still open.
//! - Chapter VII, pp. 45-56: zero-divisor structure, star operator, and the
//!   Major Theorem. Rust lanes: `star_operator.rs`, `zd_criterion.rs`.
//!   Rocq lane: `proofs/theories/Brown1972.v` plus companion theorem files.
//! - Appendix C, pp. 78-89: PL/1 search program.
//!   Rust lane: `pl1_emulator.rs`. Rocq extraction bridge still open.
//!
//! Mirrors: Brown1972.v, ZD_Criterion.v, BrownAssessorEquivalence.v,
//! C1538_MorZDSymmetry.v. Brown-adjacent support reused by these lanes includes
//! CDPowerAssociative.v and CDTraceZero.v.

pub mod associator_properties;
pub mod basis_element_properties;
pub mod exponent_properties;
pub mod norm_defect;
pub mod norm_symmetry;
pub mod pl1_emulator;
pub mod star_operator;
pub mod zd_criterion;
