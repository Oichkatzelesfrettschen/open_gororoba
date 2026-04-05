//! Section I: O-trips and S-trips.
//!
//! An **O-trip** (a, b, c) is an octonion associative triple: e_a * e_b = +/- e_c
//! in the XOR labeling convention.  All seven satisfy a XOR b = c.
//!
//! An **S-trip** (a, b, c) is a sedenion associative triple crossing the
//! octonion/sedenion boundary: one low index (1..7) and two high indices (8..15).
//! All twenty-eight also satisfy a XOR b = c.
//!
//! Mirrors: DeMarraisAssessors.v o_trips, s_trips, o_trip_count, s_trip_count,
//!          o_trips_xor_consistent, s_trips_xor_consistent, total_trip_count.

/// An associative triple (a, b, c) satisfying a XOR b = c.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssocTriple {
    pub a: u8,
    pub b: u8,
    pub c: u8,
}

impl AssocTriple {
    pub const fn new(a: u8, b: u8, c: u8) -> Self {
        Self { a, b, c }
    }

    /// Check the XOR rule: a XOR b = c.
    pub const fn is_xor_consistent(&self) -> bool {
        self.a ^ self.b == self.c
    }

    /// The XOR G-index = a XOR b (= c when consistent).
    pub const fn g_index(&self) -> u8 {
        self.a ^ self.b
    }
}

/// The 7 O-trips (octonion associative triples).
///
/// de Marrais (2000) Section I:
///   (1,2,3); (1,4,5); (1,7,6); (2,4,6); (2,5,7); (3,4,7); (3,6,5)
///
/// Each triple corresponds to an imaginary basis multiplication in the
/// XOR-indexed octonion: e_a * e_b = +/- e_{a XOR b}.
///
/// Mirrors: DeMarraisAssessors.v o_trips.
pub const O_TRIPS: [AssocTriple; 7] = [
    AssocTriple::new(1, 2, 3),
    AssocTriple::new(1, 4, 5),
    AssocTriple::new(1, 7, 6),
    AssocTriple::new(2, 4, 6),
    AssocTriple::new(2, 5, 7),
    AssocTriple::new(3, 4, 7),
    AssocTriple::new(3, 6, 5),
];

/// The 28 S-trips (sedenion associative triples).
///
/// Each has one low index (1..7) and two high indices (8..15).
/// All satisfy the XOR rule a XOR b = c.
///
/// Mirrors: DeMarraisAssessors.v s_trips.
pub const S_TRIPS: [AssocTriple; 28] = [
    // Low index 1
    AssocTriple::new(1, 8, 9),
    AssocTriple::new(1, 11, 10),
    AssocTriple::new(1, 13, 12),
    AssocTriple::new(1, 14, 15),
    // Low index 2
    AssocTriple::new(2, 8, 10),
    AssocTriple::new(2, 9, 11),
    AssocTriple::new(2, 14, 12),
    AssocTriple::new(2, 15, 13),
    // Low index 3
    AssocTriple::new(3, 8, 11),
    AssocTriple::new(3, 10, 9),
    AssocTriple::new(3, 15, 12),
    AssocTriple::new(3, 13, 14),
    // Low index 4
    AssocTriple::new(4, 8, 12),
    AssocTriple::new(4, 9, 13),
    AssocTriple::new(4, 10, 14),
    AssocTriple::new(4, 11, 15),
    // Low index 5
    AssocTriple::new(5, 8, 13),
    AssocTriple::new(5, 12, 9),
    AssocTriple::new(5, 10, 15),
    AssocTriple::new(5, 14, 11),
    // Low index 6
    AssocTriple::new(6, 8, 14),
    AssocTriple::new(6, 15, 9),
    AssocTriple::new(6, 12, 10),
    AssocTriple::new(6, 11, 13),
    // Low index 7
    AssocTriple::new(7, 8, 15),
    AssocTriple::new(7, 9, 14),
    AssocTriple::new(7, 13, 10),
    AssocTriple::new(7, 12, 11),
];

/// Total associative triples in sedenion space: 7 + 28 = 35.
pub const TOTAL_TRIP_COUNT: usize = O_TRIPS.len() + S_TRIPS.len();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn o_trip_count() {
        assert_eq!(O_TRIPS.len(), 7);
    }

    #[test]
    fn s_trip_count() {
        assert_eq!(S_TRIPS.len(), 28);
    }

    #[test]
    fn total_trip_count() {
        assert_eq!(TOTAL_TRIP_COUNT, 35);
    }

    #[test]
    fn o_trips_xor_consistent() {
        for t in &O_TRIPS {
            assert!(
                t.is_xor_consistent(),
                "O-trip ({},{},{}) fails XOR rule",
                t.a,
                t.b,
                t.c
            );
        }
    }

    #[test]
    fn s_trips_xor_consistent() {
        for t in &S_TRIPS {
            assert!(
                t.is_xor_consistent(),
                "S-trip ({},{},{}) fails XOR rule",
                t.a,
                t.b,
                t.c
            );
        }
    }

    #[test]
    fn o_trips_low_indices_in_range() {
        for t in &O_TRIPS {
            assert!(t.a >= 1 && t.a <= 7, "a={} out of range", t.a);
            assert!(t.b >= 1 && t.b <= 7, "b={} out of range", t.b);
            assert!(t.c >= 1 && t.c <= 7, "c={} out of range", t.c);
        }
    }

    #[test]
    fn s_trips_boundary_condition() {
        // Each S-trip has exactly one index in 1..7 and two in 8..15
        for t in &S_TRIPS {
            let low_count = [t.a, t.b, t.c]
                .iter()
                .filter(|&&x| (1..=7).contains(&x))
                .count();
            let high_count = [t.a, t.b, t.c]
                .iter()
                .filter(|&&x| (8..=15).contains(&x))
                .count();
            assert_eq!(
                low_count, 1,
                "S-trip ({},{},{}) should have 1 low index",
                t.a, t.b, t.c
            );
            assert_eq!(
                high_count, 2,
                "S-trip ({},{},{}) should have 2 high indices",
                t.a, t.b, t.c
            );
        }
    }
}
