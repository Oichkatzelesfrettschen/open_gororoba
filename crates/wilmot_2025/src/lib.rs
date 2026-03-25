//! Wilmot (2025) -- Structure of the Cayley-Dickson Algebras.
//!
//! Implements the graded CD construction, associativity classification,
//! cycle/mode theory, and zero divisor counting from arXiv:2505.11747v3.
//!
//! Mirrors: proofs/theories/WilmotPathionRetraction.v,
//!          proofs/theories/WilmotRetractionTheorem.v
//!
//! # Key formulas
//!
//! - N_m = 2^{m+3} - 1 blades at level U_m
//! - H_n = N_n(N_n-1)/6 quaternionic subalgebras
//! - Z_m = N_m(N_m-1)(N_m-3)(N_m-7)/16 zero divisor pairs
//! - Total triads = C(N_m, 3) = N_m(N_m-1)(N_m-2)/6

pub mod tower_counting;
pub mod triad_classification;
pub mod zero_divisors;
