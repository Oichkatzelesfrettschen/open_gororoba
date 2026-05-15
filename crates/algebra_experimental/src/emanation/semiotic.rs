//! Semiotic Square Mapping (ZD-Net Hypothesis, MIL 18, 19).
//!
//! De Marrais maps each box-kite to a Greimas semiotic square:
//! - S (strut) link: strut-opposite assessors
//! - G (generator) link: related by the generator index
//! - X = G XOR S link: composite relation
//!
//! Each box-kite has 3 strut axes; one square is produced per axis,
//! and `verify_semiotic_completeness` checks that the 3 squares
//! together cover every assessor of the box-kite.

use std::collections::HashSet;

use algebra_analysis::boxkites::{
    Assessor, BoxKite, EdgeSignType, canonical_strut_table, edge_sign_type,
};

/// Strut link type in the semiotic square.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrutLinkType {
    /// S-link: strut-opposite pair.
    Strut,
    /// G-link: generator relation.
    Generator,
    /// X-link: G XOR S composite.
    Composite,
}

/// A semiotic square derived from a box-kite strut pair.
///
/// Maps the 4 assessors around a strut axis to Greimas positions:
/// A, B (contraries on the same zigzag face),
/// ~A = F (strut-opposite of A), ~B (derived via generator).
#[derive(Debug, Clone)]
pub struct SemioticSquare {
    /// The "A" assessor (from zigzag face).
    pub a: Assessor,
    /// The "B" assessor (co-assessor of A on zigzag face).
    pub b: Assessor,
    /// The "not-A" assessor (strut-opposite of A).
    pub not_a: Assessor,
    /// The "not-B" assessor (strut-opposite of B).
    pub not_b: Assessor,
    /// Edge sign between A and B.
    pub ab_sign: EdgeSignType,
    /// Edge sign between not-A and not-B.
    pub not_ab_sign: EdgeSignType,
}

/// Map a box-kite to its semiotic squares (one per strut axis).
///
/// Each box-kite has 3 strut axes. For each axis, the 4 assessors adjacent
/// to both strut endpoints form a semiotic square:
///
///    A -------- B          (contraries: zigzag edge)
///    |          |
///  not-A ---- not-B        (sub-contraries: opposite zigzag edge)
///
/// The vertical links are strut-opposites (S-links).
pub fn map_boxkite_to_semiotic(bk: &BoxKite) -> Vec<SemioticSquare> {
    let atol = 1e-10;
    let tab = canonical_strut_table(bk, atol);

    // The 3 strut pairs are: (A,F), (B,E), (C,D)
    // For each strut pair, the other 4 assessors form the semiotic square.

    // Strut axis 1: (A,F) is the axis -> square from {B, C, E, D}
    // B,C are on the zigzag face with A -> they're contraries
    // E,D are on the opposite zigzag face
    //
    // Strut axis 2: (B,E) is the axis -> square from {A, C, F, D}
    // Strut axis 3: (C,D) is the axis -> square from {A, B, F, E}
    vec![
        SemioticSquare {
            a: tab.b,
            b: tab.c,
            not_a: tab.e,
            not_b: tab.d,
            ab_sign: edge_sign_type(&tab.b, &tab.c, atol),
            not_ab_sign: edge_sign_type(&tab.e, &tab.d, atol),
        },
        SemioticSquare {
            a: tab.a,
            b: tab.c,
            not_a: tab.f,
            not_b: tab.d,
            ab_sign: edge_sign_type(&tab.a, &tab.c, atol),
            not_ab_sign: edge_sign_type(&tab.f, &tab.d, atol),
        },
        SemioticSquare {
            a: tab.a,
            b: tab.b,
            not_a: tab.f,
            not_b: tab.e,
            ab_sign: edge_sign_type(&tab.a, &tab.b, atol),
            not_ab_sign: edge_sign_type(&tab.f, &tab.e, atol),
        },
    ]
}

/// Verify that the semiotic square mapping covers all assessors.
///
/// For a complete box-kite, every assessor should appear in at least one
/// semiotic square position (as A, B, ~A, or ~B).
pub fn verify_semiotic_completeness(bk: &BoxKite, squares: &[SemioticSquare]) -> bool {
    let all_assessors: HashSet<Assessor> = bk.assessors.iter().copied().collect();
    let mut covered: HashSet<Assessor> = HashSet::new();

    for sq in squares {
        covered.insert(sq.a);
        covered.insert(sq.b);
        covered.insert(sq.not_a);
        covered.insert(sq.not_b);
    }

    all_assessors == covered
}
