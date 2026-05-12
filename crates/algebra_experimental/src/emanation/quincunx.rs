//! Quincunx and Bicycle Chain explicit assessor paths (L12).
//!
//! Quincunx: 5-vertex cycle bypassing the Royal Hunt "top edge"
//! obstacle.
//!   Feet: detour via Zigzag endpoint -> "/////\\\\\" strings
//!   Hands: detour via Vent endpoint -> "/\\//\//\\" strings
//!
//! Bicycle Chain: 12-diagonal Hamiltonian cycle via 3/4-tray-rack
//! scans linked by minus-edge jumps.
//!
//! De Marrais: 2 types x 3 axes x 10 readings x 2 reversals = 120 = |H3|.

use algebra_analysis::boxkites::{Assessor, BoxKite, canonical_strut_table};

/// A Quincunx path: 5-vertex cycle through a box-kite.
#[derive(Debug, Clone)]
pub struct QuincunxPath {
    /// The 5 assessor indices visited (in order).
    pub assessor_indices: Vec<usize>,
    /// Foot (detour via Zigzag) or Hand (detour via Vent).
    pub path_type: QuincunxType,
    /// Which strut axis this quincunx bypasses.
    pub strut_axis: usize,
}

/// Quincunx path type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuincunxType {
    Foot,
    Hand,
}

/// Enumerate 6 quincunx paths for a box-kite (2 types x 3 strut axes).
///
/// For each of the 3 strut axes (AF, BE, CD in canonical labeling):
/// - The tray-rack orthogonal to strut XY is the square of the 4 remaining vertices
/// - The "top edge" is the reversed (Opposite) edge in the tray-rack
/// - Foot bypasses via the X endpoint, Hand bypasses via the Y endpoint
pub fn enumerate_quincunx_paths(bk: &BoxKite) -> Vec<QuincunxPath> {
    let atol = 1e-10;
    let tab = canonical_strut_table(bk, atol);

    // Map canonical labels A-F to assessor indices
    let labels = [tab.a, tab.b, tab.c, tab.d, tab.e, tab.f];
    let find_idx = |target: &Assessor| -> usize {
        bk.assessors
            .iter()
            .position(|a| a.low == target.low && a.high == target.high)
            .expect("Assessor must be in box-kite")
    };
    let a = find_idx(&labels[0]);
    let b = find_idx(&labels[1]);
    let c = find_idx(&labels[2]);
    let d = find_idx(&labels[3]);
    let e = find_idx(&labels[4]);
    let f = find_idx(&labels[5]);

    // Three strut axes: AF, BE, CD (A<->F, B<->E, C<->D are strut-opposites)
    // For strut axis AF: tray-rack is BCED, top edge is DE
    //   Foot(AF): B->C->E->A->D->B (detour via A, the Zigzag endpoint)
    //   Hand(AF): B->C->E->F->D->B (detour via F, the Vent endpoint)
    //
    // For strut axis BE: tray-rack is ACFD, top edge is FD
    //   Foot(BE): A->C->F->B->D->A
    //   Hand(BE): A->C->F->E->D->A
    //
    // For strut axis CD: tray-rack is ABFE, top edge is EF
    //   Foot(CD): A->B->E->C->F->A (note: not F->A since we close the 5-cycle)
    //   Hand(CD): A->B->E->D->F->A

    vec![
        QuincunxPath {
            assessor_indices: vec![b, c, e, a, d],
            path_type: QuincunxType::Foot,
            strut_axis: 0, // AF
        },
        QuincunxPath {
            assessor_indices: vec![b, c, e, f, d],
            path_type: QuincunxType::Hand,
            strut_axis: 0, // AF
        },
        QuincunxPath {
            assessor_indices: vec![a, c, f, b, d],
            path_type: QuincunxType::Foot,
            strut_axis: 1, // BE
        },
        QuincunxPath {
            assessor_indices: vec![a, c, f, e, d],
            path_type: QuincunxType::Hand,
            strut_axis: 1, // BE
        },
        QuincunxPath {
            assessor_indices: vec![a, b, e, c, f],
            path_type: QuincunxType::Foot,
            strut_axis: 2, // CD
        },
        QuincunxPath {
            assessor_indices: vec![a, b, e, d, f],
            path_type: QuincunxType::Hand,
            strut_axis: 2, // CD
        },
    ]
}

/// Count total quincunx string-readings for a box-kite.
///
/// 6 paths x 10 readings x 2 reversals = 120 = |H3|.
pub fn quincunx_string_count(_bk: &BoxKite) -> usize {
    // Each of 6 quincunx paths has 10 start points, each reversible
    6 * 10 * 2
}

/// A Bicycle Chain: 12-diagonal Hamiltonian cycle through all diagonals.
#[derive(Debug, Clone)]
pub struct BicycleChain {
    /// Sequence of (assessor_index, diagonal_orientation) states.
    /// Orientation: true = "/" (forward), false = "\" (backward).
    pub steps: Vec<(usize, bool)>,
}

/// Construct a canonical Bicycle Chain for a box-kite.
///
/// Threads all 12 diagonals via three 3/4-tray-rack scans linked by
/// minus-edge jumps. De Marrais (Presto I, Section on lanyards).
pub fn bicycle_chain(bk: &BoxKite) -> BicycleChain {
    let atol = 1e-10;
    let tab = canonical_strut_table(bk, atol);

    let find_idx = |target: &Assessor| -> usize {
        bk.assessors
            .iter()
            .position(|a| a.low == target.low && a.high == target.high)
            .expect("Assessor must be in box-kite")
    };
    let b = find_idx(&tab.b);
    let c = find_idx(&tab.c);
    let e = find_idx(&tab.e);
    let d = find_idx(&tab.d);
    let f = find_idx(&tab.f);
    let a = find_idx(&tab.a);

    // Canonical Bicycle Chain:
    // 1. AF 3/4-scan: B/ -> C\ -> E\ -> D/
    // 2. Minus-edge jump (DF): D/ -> F\
    // 3. CD 3/4-scan: F\ -> E/ -> A/ -> B\
    // 4. Minus-edge jump (BC): B\ -> C/
    // 5. BE 3/4-scan: C/ -> F/ -> D\ -> A\
    // 6. Minus-edge jump (AB): A\ -> B/ (closing)
    BicycleChain {
        steps: vec![
            (b, true),
            (c, false),
            (e, false),
            (d, true),  // AF 3/4-scan
            (f, false), // jump DF
            (e, true),
            (a, true),
            (b, false), // CD 3/4-scan
            (c, true),  // jump BC
            (f, true),
            (d, false),
            (a, false), // BE 3/4-scan
        ],
    }
}
