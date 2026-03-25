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
//! Mirrors: C1538_MorZDSymmetry.v, CDTraceZero.v, CDPowerAssociative.v

pub mod exponent_properties;
pub mod norm_defect;
pub mod star_operator;
pub mod zd_criterion;
