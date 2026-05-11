//! Forbidden-prefix enumeration for the Lambda_2048 cut, plus generic
//! lattice predicate-driven enumeration utilities.
//!
//! The 2048D cut removes points matching one of three forbidden prefix
//! families: (0, 1, 1), (0, 1, 0, 1, 1), (0, 1, 0, 1, 0, 1). The families
//! are mutually exclusive by construction. The exhaustive enumerator
//! scans the 6561 vectors of {-1, 0, 1}^8 and tags each excluded point
//! with its family.
//!
//! `enumerate_lattice_by_predicate` is the generic helper used by
//! `enumerate_lambda_256` here and by `enumerate_lambda_4096` in the
//! `lambda_predicates` sibling.

use super::{LatticeVector, is_in_base_universe, is_in_lambda_256};

// ============================================================================
// 2048D Forbidden Prefix Enumeration
// ============================================================================

/// Which of the 3 forbidden prefix families a base universe point matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForbiddenFamily {
    /// Prefix (0, 1, 1): 3-coordinate prefix, free tail of length 5.
    Prefix011,
    /// Prefix (0, 1, 0, 1, 1): 5-coordinate prefix, free tail of length 3.
    Prefix01011,
    /// Prefix (0, 1, 0, 1, 0, 1): 6-coordinate prefix, free tail of length 2.
    Prefix010101,
}

/// A base universe point excluded from Lambda_2048, tagged with its family.
#[derive(Debug, Clone)]
pub struct ForbiddenPoint {
    /// The lattice vector.
    pub vector: LatticeVector,
    /// Which forbidden prefix family it belongs to.
    pub family: ForbiddenFamily,
}

/// Classify which forbidden family a base universe point belongs to.
/// Returns `None` if the point is in Lambda_2048 (not forbidden).
pub fn classify_forbidden(v: &LatticeVector) -> Option<ForbiddenFamily> {
    if !is_in_base_universe(v) {
        return None;
    }
    // Check the 3 forbidden prefix patterns (mutually exclusive by construction:
    // Pattern 1 requires l_2=1, patterns 2 & 3 require l_2=0;
    // Pattern 2 requires l_4=1, pattern 3 requires l_4=0.)
    if v[0] == 0 && v[1] == 1 && v[2] == 1 {
        return Some(ForbiddenFamily::Prefix011);
    }
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 1 {
        return Some(ForbiddenFamily::Prefix01011);
    }
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 0 && v[5] == 1 {
        return Some(ForbiddenFamily::Prefix010101);
    }
    None
}

/// Enumerate all base universe points excluded from Lambda_2048.
///
/// Exhaustively scans {-1, 0, 1}^8 (6561 vectors), filters for base
/// universe membership, then identifies points matching one of the
/// 3 forbidden prefix families. Returns tagged forbidden points.
pub fn enumerate_forbidden_2048() -> Vec<ForbiddenPoint> {
    let mut forbidden = Vec::new();
    // Enumerate all {-1, 0, 1}^8 vectors via base-3 encoding
    for code in 0..3u32.pow(8) {
        let mut v = [0i8; 8];
        let mut c = code;
        for coord in &mut v {
            *coord = (c % 3) as i8 - 1; // map 0->-1, 1->0, 2->1
            c /= 3;
        }
        if let Some(family) = classify_forbidden(&v) {
            forbidden.push(ForbiddenPoint { vector: v, family });
        }
    }
    forbidden
}

// ============================================================================
// Lambda Enumeration from Predicates
// ============================================================================

/// Enumerate all lattice vectors in {-1, 0, 1}^8 satisfying a predicate.
///
/// Performs exhaustive scan of 3^8 = 6561 trinary vectors, returning those
/// that pass `pred`. Results are sorted in lexicographic order (with -1 < 0 < 1).
pub fn enumerate_lattice_by_predicate(pred: impl Fn(&LatticeVector) -> bool) -> Vec<LatticeVector> {
    let mut result = Vec::new();
    for code in 0..3u32.pow(8) {
        let mut v = [0i8; 8];
        let mut c = code;
        for coord in &mut v {
            *coord = (c % 3) as i8 - 1; // map 0->-1, 1->0, 2->1
            c /= 3;
        }
        if pred(&v) {
            result.push(v);
        }
    }
    result.sort();
    result
}

/// Enumerate all Lambda_256 lattice vectors from predicates alone.
///
/// Returns exactly those {-1, 0, 1}^8 vectors passing `is_in_lambda_256`.
/// This is the pure-predicate analog of loading from CSV data files.
pub fn enumerate_lambda_256() -> Vec<LatticeVector> {
    enumerate_lattice_by_predicate(is_in_lambda_256)
}
