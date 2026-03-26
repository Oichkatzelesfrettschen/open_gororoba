//! Section II: The 168 = 42 * 4 zero-divisor count decomposition.
//!
//! de Marrais (2000) Section II: each assessor plane contains 4 primitive unit
//! zero-divisors (the two diagonals '/' and '\', each traversable in two signed
//! directions).
//!
//! The full count:
//!   168 = 42 assessors * 4 primitive ZD pairs per assessor
//!       = 7 box-kites * 6 assessors * 4 ZD pairs
//!       = 7 * 24
//!
//! Mirrors: DeMarraisAssessors.v zd_count_decomposition, zd_per_boxkite,
//!          zd_total_from_boxkites.

/// Number of box-kites in sedenion space.
pub const NUM_BOX_KITES: usize = 7;

/// Number of assessors per box-kite (one per valid sedenion slot).
pub const ASSESSORS_PER_BOX_KITE: usize = 6;

/// Total primitive assessor planes: 7 * 6 = 42.
pub const TOTAL_ASSESSORS: usize = NUM_BOX_KITES * ASSESSORS_PER_BOX_KITE;

/// Primitive unit zero-divisor pairs per assessor plane (two diagonals, each
/// traversable in two signed directions: +/-).
pub const ZD_PAIRS_PER_ASSESSOR: usize = 4;

/// Total primitive unit zero-divisor pairs: 168 = 42 * 4.
pub const TOTAL_ZD_PAIRS: usize = TOTAL_ASSESSORS * ZD_PAIRS_PER_ASSESSOR;

/// ZD pairs per box-kite: 6 assessors * 4 pairs = 24.
pub const ZD_PAIRS_PER_BOX_KITE: usize = ASSESSORS_PER_BOX_KITE * ZD_PAIRS_PER_ASSESSOR;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_assessors_is_42() {
        assert_eq!(TOTAL_ASSESSORS, 42);
    }

    #[test]
    fn total_zd_pairs_is_168() {
        // Mirrors: DeMarraisAssessors.v zd_count_decomposition: 42 * 4 = 168.
        assert_eq!(TOTAL_ZD_PAIRS, 168);
    }

    #[test]
    fn zd_per_boxkite_is_24() {
        // Mirrors: DeMarraisAssessors.v zd_per_boxkite: 6 * 4 = 24.
        assert_eq!(ZD_PAIRS_PER_BOX_KITE, 24);
    }

    #[test]
    fn total_from_boxkites() {
        // Mirrors: DeMarraisAssessors.v zd_total_from_boxkites: 7 * 24 = 168.
        assert_eq!(NUM_BOX_KITES * ZD_PAIRS_PER_BOX_KITE, TOTAL_ZD_PAIRS);
    }
}
