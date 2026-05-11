//! de Marrais production rules and Fano-plane automorpheme machinery.
//!
//! Implements the three canonical production rules from
//! de Marrais (2000) plus the seven Fano-plane O-trips and the
//! automorpheme construction that PR#3 navigates.
//!
//! - `O_TRIPS`: the seven Fano-plane lines (each a 3-subset of 1..=7
//!   with each pair of points on exactly one line).
//! - `production_rule_1`: three-ring circuit -- given co-assessors a, b,
//!   returns the unique third assessor completing the triple via XOR.
//! - `production_rule_2`: skew-symmetric twisting -- creates two new
//!   assessors via pair-swapping, with exactly one of the two
//!   candidate swaps valid by construction.
//! - `production_rule_3`: automorpheme uniqueness -- given an O-trip
//!   and an assessor it contains, returns the unique OTHER O-trip
//!   whose automorpheme also contains that assessor (Fano-plane
//!   incidence: each primitive assessor belongs to exactly 2 trips).
//! - `automorpheme_assessors`: builds the 12 assessors for one O-trip
//!   via the "Behind the 8-Ball Theorem" exclusion of
//!   {8, 8^o1, 8^o2, 8^o3} from the allowed highs.
//! - `automorphemes`: applies the above to all seven O-trips.
//! - `automorphemes_containing_assessor`: reverse lookup -- returns
//!   the O-trips whose automorphemes contain a given assessor.

use std::collections::HashSet;

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

use super::Assessor;

/// The seven Fano-plane O-trips (3-element subsets of 1..=7).
///
/// Properties: 7 lines, 7 points, each point on exactly 3 lines,
/// each pair of points on exactly 1 line.
///
/// Used by de Marrais' "GoTo listings" for automorpheme construction.
pub const O_TRIPS: [[usize; 3]; 7] = [
    [1, 2, 3],
    [1, 4, 5],
    [1, 6, 7],
    [2, 4, 6],
    [2, 5, 7],
    [3, 4, 7],
    [3, 5, 6],
];

/// de Marrais Production Rule #1 ("Three-Ring Circuits").
///
/// Given two co-assessors a=(A,B) and b=(C,D), constructs a third assessor
/// (E,F) via XOR:
///   E = A ^ C = B ^ D   (low index)
///   F = A ^ D = B ^ C   (high index)
///
/// The result is the unique third assessor completing the co-assessor triple.
///
/// # Panics
/// Panics if the XOR invariants fail or the result is degenerate.
pub fn production_rule_1(a: &Assessor, b: &Assessor) -> Assessor {
    let (big_a, big_b) = (a.low, a.high);
    let (big_c, big_d) = (b.low, b.high);

    let e = big_a ^ big_c;
    let f = big_a ^ big_d;

    assert_eq!(
        e,
        big_b ^ big_d,
        "PR#1 invariant failed: A^C != B^D for ({},{}) and ({},{})",
        big_a,
        big_b,
        big_c,
        big_d
    );
    assert_eq!(
        f,
        big_b ^ big_c,
        "PR#1 invariant failed: A^D != B^C for ({},{}) and ({},{})",
        big_a,
        big_b,
        big_c,
        big_d
    );
    assert_ne!(
        e, f,
        "PR#1 degenerate: equal indices {} from ({},{}) and ({},{})",
        e, big_a, big_b, big_c, big_d
    );

    // For sedenion cross-assessors: e is in 1..7 (XOR of two lows),
    // f is in 8..15 (XOR of low and high), so e < f always.
    let (low, high) = if e < f { (e, f) } else { (f, e) };
    Assessor::new(low, high)
}

/// Helper: create a diagonal zero-divisor vector from raw index pair.
/// Used internally by production_rule_2 for candidate pairs that may
/// not satisfy the Assessor invariants.
fn raw_diagonal(i: usize, j: usize, sign: f64) -> Vec<f64> {
    let mut v = vec![0.0; 16];
    let norm = 2.0_f64.sqrt();
    v[i] = 1.0 / norm;
    v[j] = sign / norm;
    v
}

/// Helper: diagonal zero-products for raw index pairs (not necessarily valid Assessors).
fn raw_all_diagonal_zero_products(
    a: (usize, usize),
    b: (usize, usize),
    atol: f64,
) -> Vec<(i8, i8)> {
    let mut results = Vec::new();
    for s in [-1.0_f64, 1.0] {
        for t in [-1.0_f64, 1.0] {
            let v1 = raw_diagonal(a.0, a.1, s);
            let v2 = raw_diagonal(b.0, b.1, t);
            let product = cd_multiply(&v1, &v2);
            let norm = cd_norm_sq(&product).sqrt();
            if norm < atol {
                results.push((s as i8, t as i8));
            }
        }
    }
    results
}

/// de Marrais Production Rule #2 ("Skew-Symmetric Twisting").
///
/// Given co-assessors a=(A,B) and b=(C,D), creates two new assessors via
/// pair swapping. Two candidate swaps exist:
///   - Candidate 1: (A,D) and (C,B)  -- cross-index swap
///   - Candidate 2: (B,D) and (A,C)  -- same-range swap
///
/// Exactly one candidate satisfies the defining property: the outputs are
/// co-assessors with each other, but not with either input.
///
/// For sedenion cross-assessors, candidate 1 is always the valid one
/// (candidate 2 produces non-cross-assessor pairs), but both are checked
/// for mathematical rigor.
///
/// # Panics
/// Panics if exactly one valid swap cannot be found.
pub fn production_rule_2(a: &Assessor, b: &Assessor, atol: f64) -> (Assessor, Assessor) {
    let (big_a, big_b) = (a.low, a.high);
    let (big_c, big_d) = (b.low, b.high);

    // Candidate 1: cross swap -> (A,D) and (C,B)
    let c1_p = (big_a.min(big_d), big_a.max(big_d));
    let c1_q = (big_c.min(big_b), big_c.max(big_b));

    // Candidate 2: same-range swap -> (B,D) and (A,C)
    let c2_p = (big_b.min(big_d), big_b.max(big_d));
    let c2_q = (big_a.min(big_c), big_a.max(big_c));

    let a_raw = (a.low, a.high);
    let b_raw = (b.low, b.high);

    let valid_pair = |p: (usize, usize), q: (usize, usize)| -> bool {
        if p == q {
            return false;
        }
        // p and q must be co-assessors
        if raw_all_diagonal_zero_products(p, q, atol).is_empty() {
            return false;
        }
        // Neither must be co-assessor with either input
        if !raw_all_diagonal_zero_products(p, a_raw, atol).is_empty() {
            return false;
        }
        if !raw_all_diagonal_zero_products(p, b_raw, atol).is_empty() {
            return false;
        }
        if !raw_all_diagonal_zero_products(q, a_raw, atol).is_empty() {
            return false;
        }
        if !raw_all_diagonal_zero_products(q, b_raw, atol).is_empty() {
            return false;
        }
        true
    };

    let val1 = valid_pair(c1_p, c1_q);
    let val2 = valid_pair(c2_p, c2_q);

    assert_ne!(
        val1, val2,
        "PR#2 expected exactly one valid swap for ({},{}) and ({},{}), got val1={}, val2={}",
        big_a, big_b, big_c, big_d, val1, val2
    );

    let (p, q) = if val1 { (c1_p, c1_q) } else { (c2_p, c2_q) };

    // Verify the result forms valid cross-assessors
    assert!(
        (1..=7).contains(&p.0) && (8..=15).contains(&p.1),
        "PR#2 produced invalid assessor indices ({},{})",
        p.0,
        p.1
    );
    assert!(
        (1..=7).contains(&q.0) && (8..=15).contains(&q.1),
        "PR#2 produced invalid assessor indices ({},{})",
        q.0,
        q.1
    );

    (Assessor::new(p.0, p.1), Assessor::new(q.0, q.1))
}

/// Compute the 12 assessors for a de Marrais automorpheme (GoTo listing).
///
/// For each Fano-plane O-trip (o1, o2, o3), the "Behind the 8-Ball Theorem"
/// implies excluded sedenion high indices are {8, 8^o1, 8^o2, 8^o3},
/// leaving exactly 4 allowed highs to pair with each of the 3 low indices.
///
/// # Panics
/// Panics if `o_trip` is not one of the 7 canonical O_TRIPS.
pub fn automorpheme_assessors(o_trip: &[usize; 3]) -> HashSet<Assessor> {
    assert!(O_TRIPS.contains(o_trip), "Unknown O-trip: {:?}", o_trip);

    let excluded_highs: HashSet<usize> = std::iter::once(8)
        .chain(o_trip.iter().map(|&o| 8 ^ o))
        .collect();

    let allowed_highs: Vec<usize> = (8..=15).filter(|h| !excluded_highs.contains(h)).collect();
    debug_assert_eq!(allowed_highs.len(), 4);

    let mut result = HashSet::new();
    for &o in o_trip {
        for &h in &allowed_highs {
            result.insert(Assessor::new(o, h));
        }
    }
    debug_assert_eq!(result.len(), 12);
    result
}

/// Return all 7 automorpheme assessor sets (one per Fano-plane O-trip).
pub fn automorphemes() -> Vec<HashSet<Assessor>> {
    O_TRIPS.iter().map(automorpheme_assessors).collect()
}

/// Return all O-trips whose automorphemes contain the given assessor.
///
/// For a valid primitive assessor, this returns exactly 2 O-trips.
/// For excluded assessors (high=8 or high=8^low), returns empty.
pub fn automorphemes_containing_assessor(a: &Assessor) -> Vec<[usize; 3]> {
    O_TRIPS
        .iter()
        .filter(|t| automorpheme_assessors(t).contains(a))
        .copied()
        .collect()
}

/// de Marrais Production Rule #3 (Automorpheme Uniqueness).
///
/// Given an automorpheme (by O-trip) and an assessor it contains, returns the
/// unique OTHER O-trip whose automorpheme also contains that assessor.
///
/// Each primitive assessor belongs to exactly 2 automorphemes (by Fano-plane
/// incidence). PR#3 finds the other one.
///
/// # Panics
/// Panics if the assessor is not in the given automorpheme, or if the
/// expected 2-membership property fails.
pub fn production_rule_3(o_trip: &[usize; 3], a: &Assessor) -> [usize; 3] {
    assert!(
        automorpheme_assessors(o_trip).contains(a),
        "Assessor ({},{}) not in automorpheme for {:?}",
        a.low,
        a.high,
        o_trip
    );

    let candidates = automorphemes_containing_assessor(a);
    assert_eq!(
        candidates.len(),
        2,
        "Expected exactly 2 automorphemes for ({},{}), got {}",
        a.low,
        a.high,
        candidates.len()
    );

    if candidates[1] == *o_trip {
        candidates[0]
    } else {
        candidates[1]
    }
}
