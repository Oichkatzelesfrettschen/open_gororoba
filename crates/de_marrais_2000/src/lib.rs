//! de Marrais (2000) -- The 42 Assessors and the Box-Kites they fly.
//!
//! R.P.C. de Marrais, "The 42 Assessors and the Box-Kites they fly:
//! Diagonal Axis-Pair Systems of Zero-Divisors in the Sedenions'
//! 16 Dimensions", arXiv:math/0011260.
//!
//! This crate formalizes the enumerable content of Sections I and II:
//!
//! - **Section I**: O-trips (7 octonion associative triples) and S-trips
//!   (28 sedenion associative triples), all satisfying a XOR b = c.
//!
//! - **Section II**: The 42 primitive assessor planes, co-assessor predicate,
//!   Production Rule #1 (three-ring circuits), and the 168 = 42 * 4
//!   zero-divisor count decomposition.
//!
//! Rocq companion: proofs/theories/DeMarraisAssessors.v (256 lines, 0 Admitted).

pub mod assessors;
pub mod count;
pub mod trips;

// Flatten the most-used types to crate root for ergonomic imports.
pub use assessors::{
    all_assessors, co_assessor_buckets, co_assessors, production_rule_1,
    verify_production_rule_1, Assessor, ThreeRingCircuit,
};
pub use count::{
    ASSESSORS_PER_BOX_KITE, NUM_BOX_KITES, TOTAL_ASSESSORS, TOTAL_ZD_PAIRS,
    ZD_PAIRS_PER_ASSESSOR, ZD_PAIRS_PER_BOX_KITE,
};
pub use trips::{AssocTriple, O_TRIPS, S_TRIPS, TOTAL_TRIP_COUNT};
