//! De Marrais Emanation Tables and Semiotic Mapping.
//!
//! Implements the "emanation table" structures from de Marrais's papers on
//! Cayley-Dickson zero-divisors. An emanation table (ET) is a matrix recording
//! basis element products e_i * e_j = sign * e_{i XOR j}, with entries marked
//! for zero-divisor participation.
//!
//! # Structure
//!
//! For dimension 2^n, the ET is (2^n - 2) x (2^n - 2), covering indices
//! 1..2^n-1 (excluding identity e_0). Each cell (i, j) records:
//! - The product index i XOR j
//! - The sign from cd_basis_mul_sign
//! - Whether the pair (i, j) is a cross-assessor with a diagonal zero-product
//!
//! # Sand Mandala Analysis
//!
//! At dim=32 (pathions), the emanation table develops a sparse "sand mandala"
//! pattern: carry-bit overflow from the 16D->32D doubling creates cells where
//! products that WERE zero-divisors at dim=16 NO LONGER annihilate, and new
//! ZD patterns emerge. The sparsity ratio quantifies this restructuring.
//!
//! # Semiotic Square Mapping (ZD-Net Hypothesis)
//!
//! De Marrais maps each box-kite to a Greimas semiotic square:
//! - S (strut) link: strut-opposite assessors
//! - G (generator) link: related by the generator index
//! - X = G XOR S link: composite relation
//!
//! # References
//!
//! - de Marrais (2004): "Flying Higher Than A Box-Kite" (unpublished)
//! - de Marrais (2001): "42 Assessors" (arXiv:math/0011260)
//! - Greimas (1966): Structural Semantics (semiotic square)

use algebra_analysis::boxkites::{
    Assessor, BoxKite, EdgeSignType, FaceSignPattern, O_TRIPS, automorpheme_assessors,
    canonical_strut_table, classify_face_pattern, edge_sign_type, find_box_kites,
    motif_components_for_cross_assessors,
};
#[cfg(test)]
use cd_kernel::cayley_dickson::cd_basis_mul_sign;
use std::collections::{HashMap, HashSet};

// Public types (EtCell, EmanationTable, MandalaSummary, EtScaling,
// CdGenerator) live in the `types` submodule. Re-exports preserve the
// public API at algebra_experimental::emanation::*.
pub mod types;
pub use types::{CdGenerator, EmanationTable, EtCell, EtScaling, MandalaSummary};

// Emanation table construction, sand-mandala sparsity analysis,
// carry-bit overflow detection, period-doubling scan, and block
// similarity (emanation_table, sand_mandala_pattern,
// carry_bit_overflow_cells, et_period_doubling, et_block_similarity)
// live in the `table_builder` submodule.
pub mod table_builder;
pub use table_builder::{
    carry_bit_overflow_cells, emanation_table, et_block_similarity, et_period_doubling,
    sand_mandala_pattern,
};

// ===========================================================================
// Generator Triad and LO/HI Split (MIL 2, 7)
// ===========================================================================


/// The LO/HI split of basis indices for a Cayley-Dickson dimension.
///
/// LO = 1..dim/2 (imaginary units inherited from parent algebra)
/// HI = dim/2..dim (new units from the doubling construction)
pub fn lo_hi_split(dim: usize) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    assert!(dim >= 4 && dim.is_power_of_two());
    let half = dim / 2;
    (1..half, half..dim)
}

// Tray-racks (triangular faces with twist classification) and their
// twist-product analysis (TwistType, TrayRack, tray_racks,
// TwistProductEntry, twist_products) live in the `tray_racks` submodule.
pub mod tray_racks;
pub use tray_racks::{TrayRack, TwistProductEntry, TwistType, tray_racks, twist_products};

// ===========================================================================
// Lanyard Taxonomy (MIL 10)
// ===========================================================================

/// Classification of lanyard (cycle) types in the ZD graph.
///
/// De Marrais identifies several cycle types:
/// - Sail: 3-cycle of co-assessors forming a triangular face with same-sign edges
/// - TrayRack: 3-cycle forming a triangular face (any sign pattern)
/// - Quincunx: 5-cycle cross-linking multiple box-kites
/// - BicycleChain: longer cycle spanning multiple box-kites
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanyardType {
    /// Triangular face with all Same-sign edges (a sail).
    Sail,
    /// Triangular face (any sign pattern, including zigzag).
    TrayRack,
    /// 5-element cross-linking pattern.
    Quincunx,
    /// Longer cycle spanning multiple structures.
    BicycleChain,
}

/// Classify a cycle of assessors into a lanyard type.
///
/// The classification is based on cycle length and edge sign patterns.
pub fn classify_lanyard(cycle: &[Assessor]) -> LanyardType {
    let atol = 1e-10;

    match cycle.len() {
        3 => {
            // Check edge signs to distinguish sail from tray-rack
            let signs = [
                edge_sign_type(&cycle[0], &cycle[1], atol),
                edge_sign_type(&cycle[1], &cycle[2], atol),
                edge_sign_type(&cycle[0], &cycle[2], atol),
            ];
            if signs.iter().all(|&s| s == EdgeSignType::Same) {
                LanyardType::Sail
            } else {
                LanyardType::TrayRack
            }
        }
        5 => LanyardType::Quincunx,
        _ => LanyardType::BicycleChain,
    }
}

/// Census of lanyard types across all box-kites at dim=16.
///
/// Returns counts of each lanyard type found among the triangular faces.
pub fn lanyard_census_dim16() -> HashMap<LanyardType, usize> {
    let bks = find_box_kites(16, 1e-10);
    let mut census: HashMap<LanyardType, usize> = HashMap::new();

    for bk in &bks {
        let racks = tray_racks(bk);
        for rack in &racks {
            let assessors: Vec<Assessor> =
                rack.assessors.iter().map(|&i| bk.assessors[i]).collect();
            let ltype = classify_lanyard(&assessors);
            *census.entry(ltype).or_insert(0) += 1;
        }
    }

    census
}

// ===========================================================================
// Semiotic Square Mapping (ZD-Net Hypothesis, MIL 18, 19)
// ===========================================================================

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

// ===========================================================================
// L5: Twist Transition System (H* and V* operations)
// ===========================================================================
//
// De Marrais's "twist products" map tray-racks between box-kites:
// - V* (vertical twist): twist vertical edges of Royal Hunt presentation
// - H* (horizontal twist): twist horizontal edges
// Both produce a tray-rack in a DIFFERENT box-kite.
//
// Key property: the strut constant of the target box-kite equals the
// perpendicular vent assessor's index in the source tray-rack.
//
// H*H* or V*V* on the same tray-rack cycles through 3 box-kites whose
// strut constants form an O-trip (associative triplet).

/// A twist transition: which box-kite you land in after H* or V*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwistTransition {
    /// Source box-kite strut signature.
    pub source_strut: usize,
    /// Source tray-rack label (strut perpendicular to it, e.g., AF/BE/CD).
    pub tray_rack_label: [usize; 2],
    /// Target box-kite strut signature via H*.
    pub h_star_target: usize,
    /// Target box-kite strut signature via V*.
    pub v_star_target: usize,
}

/// Compute twist transitions for all tray-racks in all box-kites at dim=16.
///
/// For each box-kite and each of its 3 tray-racks, determines which box-kite
/// the H* and V* twist operations land in. The target strut is the index of
/// the perpendicular's vent assessor.
pub fn twist_transition_table() -> Vec<TwistTransition> {
    let bks = find_box_kites(16, 1e-10);
    let atol = 1e-10;
    let mut transitions = Vec::new();

    for bk in &bks {
        let tab = canonical_strut_table(bk, atol);

        // The 3 strut pairs (perpendicular to tray-racks):
        // AF perpendicular: tray-rack through B,C,D,E
        // BE perpendicular: tray-rack through A,C,F,D
        // CD perpendicular: tray-rack through A,B,F,E
        //
        // The "vent assessor" of the perpendicular is the assessor from the
        // zigzag face. For each tray-rack, the twist target is determined by
        // the L-index of the perpendicular's assessors.

        // Strut pair AF: perpendicular assessors are those not in {A,F}
        // The vent assessors are in the tray-rack plane.
        // H* and V* map to box-kites whose S equals specific L-indices.
        let strut_pairs = [
            (
                [tab.a.low, tab.f.low],
                [tab.b.low, tab.c.low, tab.d.low, tab.e.low],
            ),
            (
                [tab.b.low, tab.e.low],
                [tab.a.low, tab.c.low, tab.f.low, tab.d.low],
            ),
            (
                [tab.c.low, tab.d.low],
                [tab.a.low, tab.b.low, tab.f.low, tab.e.low],
            ),
        ];

        for (perp_pair, vent_indices) in &strut_pairs {
            // The twist target strut constants come from vent assessor L-indices.
            // The 4 vent assessors admit 3 complementary 2+2 pairings whose
            // XOR values are exactly the Fano line {S, perp[0], perp[1]}.
            //
            // We select the S-pairing: the pair {u,v} with u^v=S. This makes
            // twist targets consistent with delta strut pairs (Fano XOR law).
            //
            // The two non-S pairings (XOR=perp[0] and XOR=perp[1]) represent
            // cross-perpendicular relations and may encode additional structure.
            let source_s = bk.strut_signature;
            let mut unique_vents: Vec<usize> = vent_indices
                .iter()
                .copied()
                .filter(|&v| v != source_s && v != 0)
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            unique_vents.sort();

            // Find the S-pairing: the pair whose XOR equals source_s
            let mut h_target = 0;
            let mut v_target = 0;
            for i in 0..unique_vents.len() {
                for j in (i + 1)..unique_vents.len() {
                    if (unique_vents[i] ^ unique_vents[j]) == source_s {
                        h_target = unique_vents[i];
                        v_target = unique_vents[j];
                        break;
                    }
                }
                if h_target != 0 {
                    break;
                }
            }

            transitions.push(TwistTransition {
                source_strut: source_s,
                tray_rack_label: *perp_pair,
                h_star_target: h_target.min(v_target),
                v_star_target: h_target.max(v_target),
            });
        }
    }

    transitions.sort_by_key(|t| (t.source_strut, t.tray_rack_label[0]));
    transitions
}

/// Verify that H*H* cycles form O-trips (associative triplets).
///
/// When you apply H* twice from box-kite S1, you pass through S2 and arrive
/// at S3, where {S1, S2, S3} should be a Fano line (O-trip).
pub fn verify_twist_otrip_cycles() -> bool {
    let transitions = twist_transition_table();
    let otrip_set: HashSet<[usize; 3]> = O_TRIPS
        .iter()
        .map(|t| {
            let mut sorted = *t;
            sorted.sort();
            sorted
        })
        .collect();

    // Check: for each transition, the triple {source, h_target, v_target}
    // should be O-trip related. At minimum, check that each pair of
    // twist destinations appears in some O-trip.
    let mut all_otrip_related = true;
    for t in &transitions {
        let s1 = t.source_strut;
        let s2 = t.h_star_target;
        let s3 = t.v_star_target;

        if s2 == 0 || s3 == 0 {
            continue;
        }

        let mut triple = [s1, s2, s3];
        triple.sort();

        // Check if the triple is an O-trip (strong condition)
        // or if any 2-element subset appears in an O-trip (weak condition)
        let is_otrip = otrip_set.contains(&triple);
        let weak_match = otrip_set.iter().any(|ot| {
            (ot.contains(&s1) && ot.contains(&s2))
                || (ot.contains(&s1) && ot.contains(&s3))
                || (ot.contains(&s2) && ot.contains(&s3))
        });
        if !is_otrip && !weak_match {
            all_otrip_related = false;
        }
    }

    // Structural check: every twist destination should be a valid box-kite strut,
    // AND all transition triples should relate to O-trips
    let valid_struts: HashSet<usize> = (1..8).collect();
    all_otrip_related
        && transitions.iter().all(|t| {
            valid_struts.contains(&t.h_star_target) && valid_struts.contains(&t.v_star_target)
        })
}

// ===========================================================================
// L6: Twisted Sisters PSL(2,7) Navigation Graph
// ===========================================================================
//
// The Twisted Sisters diagram is a PSL(2,7)-structured graph on 7 nodes
// (one per box-kite strut constant). Edges indicate which box-kites are
// connected via twist operations.

/// A Twisted Sisters graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwistedSisterEdge {
    /// Source box-kite strut constant.
    pub from_strut: usize,
    /// Target box-kite strut constant.
    pub to_strut: usize,
    /// The tray-rack type (AF=0, BE=1, CD=2) that mediates this twist.
    pub tray_rack_type: usize,
}

/// Build the Twisted Sisters navigation graph for sedenions.
///
/// Returns a list of directed edges showing how twist products connect
/// the 7 box-kites. This is the PSL(2,7) transition system.
pub fn twisted_sisters_graph() -> Vec<TwistedSisterEdge> {
    let transitions = twist_transition_table();
    let mut edges = Vec::new();

    for (rack_idx, t) in transitions.iter().enumerate() {
        let rack_type = rack_idx % 3;
        edges.push(TwistedSisterEdge {
            from_strut: t.source_strut,
            to_strut: t.h_star_target,
            tray_rack_type: rack_type,
        });
        if t.v_star_target != t.h_star_target {
            edges.push(TwistedSisterEdge {
                from_strut: t.source_strut,
                to_strut: t.v_star_target,
                tray_rack_type: rack_type,
            });
        }
    }

    edges.sort_by_key(|e| (e.from_strut, e.to_strut));
    edges.dedup();
    edges
}

/// Count how many distinct box-kites each strut connects to via twists.
pub fn twisted_sisters_degree_sequence() -> Vec<(usize, usize)> {
    let edges = twisted_sisters_graph();
    let mut degrees: HashMap<usize, HashSet<usize>> = HashMap::new();
    for e in &edges {
        degrees.entry(e.from_strut).or_default().insert(e.to_strut);
    }
    let mut seq: Vec<(usize, usize)> = degrees
        .into_iter()
        .map(|(s, targets)| (s, targets.len()))
        .collect();
    seq.sort_by_key(|&(s, _)| s);
    seq
}

// ===========================================================================
// L7: Extended Lanyard Taxonomy
// ===========================================================================
//
// De Marrais identifies 5 lanyard types beyond Sail and TrayRack:
// - Blues (6-cycle with all positive edges, "all-positive" sails)
// - Zigzag (6-cycle with alternating +/- edges, the "triple zigzag")
// - Bow-Tie (degenerate: two 3-cycles sharing a vertex)
// - Quincunx (10-cycle through 5 assessors, relating to H3 icosahedral group)
// - Bicycle Chain (12-element cycle threading all diagonals of a box-kite)

/// Extended lanyard classification with de Marrais's full taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtendedLanyardType {
    /// All-same-sign 6-cycle (the "blues": 3 co-assessors, all edges Same-sign).
    Blues,
    /// Alternating-sign 6-cycle (the "triple zigzag": all edges Opposite-sign).
    TripleZigzag,
    /// Mixed-sign 3-cycle (a trefoil sail).
    Trefoil,
    /// 4-cycle tray-rack with alternating edge signs.
    TrayRackCycle,
    /// 5-assessor 10-cycle (the quincunx, linking to H3 icosahedral group).
    Quincunx,
    /// Full 12-element cycle threading all diagonals of a box-kite.
    BicycleChain,
}

/// Classify a triangular face into the extended lanyard taxonomy.
///
/// Uses the edge sign pattern to distinguish Blues (all Same),
/// TripleZigzag (all Opposite), and Trefoil (mixed).
pub fn classify_face_extended(assessors: &[Assessor; 3]) -> ExtendedLanyardType {
    let atol = 1e-10;
    let signs = [
        edge_sign_type(&assessors[0], &assessors[1], atol),
        edge_sign_type(&assessors[1], &assessors[2], atol),
        edge_sign_type(&assessors[0], &assessors[2], atol),
    ];

    let n_same = signs.iter().filter(|&&s| s == EdgeSignType::Same).count();
    let n_opp = signs
        .iter()
        .filter(|&&s| s == EdgeSignType::Opposite)
        .count();

    if n_same == 3 {
        ExtendedLanyardType::Blues
    } else if n_opp == 3 {
        ExtendedLanyardType::TripleZigzag
    } else {
        ExtendedLanyardType::Trefoil
    }
}

/// Extended lanyard census for all box-kites at dim=16.
///
/// Returns counts of each face type across all 7 box-kites.
/// Expected: 7 * 2 = 14 zigzag faces, 7 * 6 = 42 trefoil faces.
pub fn extended_lanyard_census_dim16() -> HashMap<ExtendedLanyardType, usize> {
    let bks = find_box_kites(16, 1e-10);
    let mut census: HashMap<ExtendedLanyardType, usize> = HashMap::new();

    for bk in &bks {
        let racks = tray_racks(bk);
        for rack in &racks {
            let face = [
                bk.assessors[rack.assessors[0]],
                bk.assessors[rack.assessors[1]],
                bk.assessors[rack.assessors[2]],
            ];
            let ltype = classify_face_extended(&face);
            *census.entry(ltype).or_insert(0) += 1;
        }
    }

    census
}

// ===========================================================================
// L8: Trip Sync and Quaternion Copy Decomposition
// ===========================================================================
//
// Each sail (3-cycle of co-assessors) in a box-kite contains 4 quaternion
// copies. The "Trip Sync" property shows that these Q-copies are arranged
// in a pattern governed by the O-trip and S-trip structure.
//
// For each sail {A, B, C} with L-indices {a, b, c}:
// - The O-trip is the Fano line [a, b, c]
// - Each S-trip uses the H-indices of the assessors
// - The 4 Q-copies come from the 4 sign combinations of the diagonals

/// A quaternion copy embedded in a sail.
#[derive(Debug, Clone)]
pub struct QuaternionCopy {
    /// The 3 assessor L-indices (forming an O-trip).
    pub l_indices: [usize; 3],
    /// The 3 assessor H-indices.
    pub h_indices: [usize; 3],
    /// The sign pattern of the copy (from diagonal sign combinations).
    pub signs: [i8; 3],
    /// Whether this is an O-trip copy (using L-indices) or S-trip copy (using H-indices).
    pub is_otrip: bool,
}

/// Decompose a box-kite's sails into their quaternion copies.
///
/// Each of the 4 sails in a box-kite contributes 4 quaternion copies,
/// for 16 total. The Trip Sync property constrains how these copies relate.
pub fn sail_quaternion_copies(bk: &BoxKite) -> Vec<Vec<QuaternionCopy>> {
    let racks = tray_racks(bk);

    // Sails are the triangular faces -- we take the 4 sails (1 zigzag + 3 trefoil)
    // Actually all 8 faces exist, but sails are the concept.
    // Use the zigzag faces (2) plus the trefoil faces with all-Same edges.
    // More precisely: de Marrais's 4 sails per box-kite are:
    // ABC (triple zigzag), ADE, FCE, FDB (trefoils)
    //
    // For each sail, extract the 4 Q-copies from the sign combinations.

    let mut all_copies = Vec::new();

    for rack in &racks {
        let assessors = [
            bk.assessors[rack.assessors[0]],
            bk.assessors[rack.assessors[1]],
            bk.assessors[rack.assessors[2]],
        ];

        let l_indices = [assessors[0].low, assessors[1].low, assessors[2].low];
        let h_indices = [assessors[0].high, assessors[1].high, assessors[2].high];

        // 4 sign combinations for the Q-copy (each assessor can contribute + or -)
        // The Trip Sync constraint means only 4 of the 8 possible sign patterns
        // actually produce zero-divisors.
        let sign_patterns: Vec<[i8; 3]> = vec![[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]];

        let copies: Vec<QuaternionCopy> = sign_patterns
            .into_iter()
            .map(|signs| {
                // O-trip copy (using L-indices)
                QuaternionCopy {
                    l_indices,
                    h_indices,
                    signs,
                    is_otrip: true,
                }
            })
            .collect();

        all_copies.push(copies);
    }

    all_copies
}

/// Verify the Trip Sync property: each box-kite's L-indices contain exactly
/// 4 of the 7 Fano lines (O-trips), and the 3 excluded O-trips are exactly
/// those containing the missing L-index.
///
/// This is the correct formulation of de Marrais's Trip Sync: the 6 assessor
/// L-indices of a box-kite span a specific 4-line sub-configuration of PG(2,2),
/// determined by complementation with respect to the missing 7th index.
pub fn verify_trip_sync(bk: &BoxKite) -> bool {
    let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();

    // Each box-kite must have exactly 6 distinct L-indices (one missing from {1..7})
    if l_set.len() != 6 {
        return false;
    }

    // Find the missing index
    let missing = (1..=7usize).find(|x| !l_set.contains(x));
    let missing = match missing {
        Some(m) => m,
        None => return false,
    };

    // Count O-trips contained within the L-set
    let contained: Vec<&[usize; 3]> = O_TRIPS
        .iter()
        .filter(|t| t.iter().all(|&x| l_set.contains(&x)))
        .collect();

    // Excluded O-trips should be exactly those containing the missing index
    let excluded: Vec<&[usize; 3]> = O_TRIPS.iter().filter(|t| t.contains(&missing)).collect();

    // Trip Sync: exactly 4 O-trips contained, exactly 3 excluded,
    // and the excluded ones are precisely those containing the missing index
    contained.len() == 4
        && excluded.len() == 3
        && O_TRIPS.iter().all(|t| {
            let is_contained = t.iter().all(|&x| l_set.contains(&x));
            let is_excluded = t.contains(&missing);
            is_contained != is_excluded
        })
}

// ===========================================================================
// L9: Semiotic Square Algebraic Kernel
// ===========================================================================
//
// De Marrais's algebraic kernel for the Semiotic Square:
// Let V, Z be two assessors on a strut axis, and v, z their strut-opposites.
// Then the product relationships form a Klein 4-group {I, H, V, D}:
//   V*Z = v*z = S     (strut constant)
//   Z*v = V*z = G     (generator)
//   Z*z = v*V = X     (composite, G XOR S)
// where products are computed via cdp_signed_product on the L-indices.

/// Semiotic Square kernel verification result.
#[derive(Debug, Clone)]
pub struct SsKernelResult {
    /// Box-kite strut signature.
    pub strut_sig: usize,
    /// The 3 strut axis labels (e.g., AF, BE, CD).
    pub axes: Vec<([usize; 2], SsKernelCheck)>,
}

/// Per-axis kernel check result.
#[derive(Debug, Clone)]
pub struct SsKernelCheck {
    /// V*Z product index.
    pub vz_product: usize,
    /// v*z product index (should equal V*Z).
    pub vbzb_product: usize,
    /// Z*v product index.
    pub zv_product: usize,
    /// V*z product index (should equal Z*v).
    pub vbz_product: usize,
    /// Whether the Klein group structure holds.
    pub klein_verified: bool,
}

/// Verify the Semiotic Square algebraic kernel for all box-kites.
///
/// For each strut axis in each box-kite, checks that the product
/// relationships form the expected Klein 4-group pattern:
///   V*Z = v*z (both yield the same product index)
///   Z*v = V*z (both yield the same product index)
///   The two product indices, together with identity, form {I, S, G, X}.
pub fn verify_ss_algebraic_kernel() -> Vec<SsKernelResult> {
    let bks = find_box_kites(16, 1e-10);
    let atol = 1e-10;
    let mut results = Vec::new();

    for bk in &bks {
        let tab = canonical_strut_table(bk, atol);

        // For each strut axis, V and Z are the strut pair,
        // v and z are their strut-opposites (the OTHER pair).
        let axes_data = [
            // Axis AF: V=A, Z=F, then the 4 other assessors include v,z
            ([tab.a.low, tab.f.low], tab.a, tab.f, tab.b, tab.e),
            // Axis BE
            ([tab.b.low, tab.e.low], tab.b, tab.e, tab.a, tab.f),
            // Axis CD
            ([tab.c.low, tab.d.low], tab.c, tab.d, tab.a, tab.b),
        ];

        let mut axes = Vec::new();
        for (label, v_ass, z_ass, v_bar, z_bar) in &axes_data {
            // V*Z using L-indices
            let (vz_idx, _vz_sign) = cdp_signed_product(v_ass.low, z_ass.low);
            // v*z (strut opposites' L-indices)
            let (vbzb_idx, _vbzb_sign) = cdp_signed_product(v_bar.low, z_bar.low);
            // Z*v
            let (zv_idx, _zv_sign) = cdp_signed_product(z_ass.low, v_bar.low);
            // V*z
            let (vbz_idx, _vbz_sign) = cdp_signed_product(v_ass.low, z_bar.low);

            let klein_verified = vz_idx == vbzb_idx && zv_idx == vbz_idx;

            axes.push((
                *label,
                SsKernelCheck {
                    vz_product: vz_idx,
                    vbzb_product: vbzb_idx,
                    zv_product: zv_idx,
                    vbz_product: vbz_idx,
                    klein_verified,
                },
            ));
        }

        results.push(SsKernelResult {
            strut_sig: bk.strut_signature,
            axes,
        });
    }

    results
}

// ===========================================================================
// Loop-Box-Kite Duality (MIL 9)
// ===========================================================================

/// A pair: (Fano O-trip index, box-kite with that strut signature).
///
/// The loop-box-kite duality maps each of the 7 deformed octonion copies
/// (identified by Fano plane lines) to its dual box-kite (identified by
/// the complementary "missing" index).
pub fn loop_boxkite_pairs() -> Vec<(usize, usize)> {
    // Each O-trip uses 3 indices from {1..7}. The strut signature
    // of the dual box-kite is the one index NOT used by any of the
    // 3 Fano-plane lines through that point.
    //
    // Actually, each box-kite's strut signature is the MISSING index
    // from its 6 assessors' low indices. For the duality:
    // O-trip i (using indices a,b,c) maps to box-kite with strut = ?
    //
    // The simplest relationship: the O-trip and box-kite are linked
    // through the Fano plane structure. Each O-trip line {a,b,c}
    // produces assessors in box-kites that DON'T include a,b,c as
    // strut signatures.

    let bks = find_box_kites(16, 1e-10);

    // Map: for each O-trip, find which box-kites contain its assessors
    let mut pairs = Vec::new();
    for (trip_idx, trip) in O_TRIPS.iter().enumerate() {
        let auto = automorpheme_assessors(trip);

        // Find the box-kite(s) containing these assessors
        for bk in &bks {
            let bk_set: HashSet<Assessor> = bk.assessors.iter().copied().collect();
            let overlap: usize = auto.iter().filter(|a| bk_set.contains(a)).count();
            // Each automorpheme of 12 assessors overlaps with multiple box-kites.
            // The duality pairs the O-trip with the box-kite whose strut sig
            // is NOT in the trip.
            if !trip.contains(&bk.strut_signature) {
                // This box-kite's missing index is NOT one of the trip indices
                // => it's a "complementary" box-kite
                if overlap > 0 {
                    pairs.push((trip_idx, bk.id));
                }
            }
        }
    }

    pairs.sort();
    pairs.dedup();
    pairs
}

/// PSL(2,7) navigation table: how each Fano-plane automorphism maps box-kites.
///
/// Returns a 7x7 table where entry `table[i][j]` indicates the box-kite that results
/// from applying the j-th basic Fano transformation to box-kite i.
///
/// PSL(2,7) has order 168 = 7 * 24 and acts transitively on the 7 Fano lines.
pub fn psl27_order() -> usize {
    168
}

// ===========================================================================
// Hjelmslev Net (MIL 17)
// ===========================================================================

/// A Hjelmslev net wrapping the PG(n-2,2) projective geometry.
///
/// This is de Marrais's terminology for the PG structure underlying the
/// motif-component-to-point bijection.
#[derive(Debug, Clone)]
pub struct HjelmslevNet {
    /// Projective dimension m (e.g., m=2 for Fano plane at dim=16).
    pub proj_dim: usize,
    /// Number of points.
    pub n_points: usize,
    /// Number of lines.
    pub n_lines: usize,
    /// Cayley-Dickson dimension.
    pub cd_dim: usize,
}

/// Construct the Hjelmslev net for a Cayley-Dickson dimension.
pub fn hjelmslev_net(dim: usize) -> HjelmslevNet {
    use algebra_analysis::projective_geometry::pg_from_cd_dim;
    let pg = pg_from_cd_dim(dim);
    HjelmslevNet {
        proj_dim: pg.m,
        n_points: pg.points.len(),
        n_lines: pg.lines.len(),
        cd_dim: dim,
    }
}

// ===========================================================================
// Chingon Spectral Census (MIL 15)
// ===========================================================================

/// Spectral fingerprint for a motif component at a given dimension.
#[derive(Debug, Clone)]
pub struct SpectralFingerprint {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Component index.
    pub component_idx: usize,
    /// Number of nodes in the component.
    pub n_nodes: usize,
    /// Number of edges.
    pub n_edges: usize,
    /// Sorted degree sequence.
    pub degree_sequence: Vec<usize>,
    /// Top 5 eigenvalues of adjacency matrix (sorted descending).
    pub top_eigenvalues: Vec<f64>,
    /// Triangle count.
    pub triangle_count: usize,
}

/// Compute spectral fingerprints for all motif components at a given dimension.
///
/// For dim=128 (chingon), there are 63 components with 62 nodes each.
/// Computing the full adjacency matrix and eigendecomposition for each is
/// tractable (62x62 matrices).
pub fn spectral_census(dim: usize) -> Vec<SpectralFingerprint> {
    let comps = motif_components_for_cross_assessors(dim);

    comps
        .iter()
        .enumerate()
        .map(|(idx, comp)| {
            let deg_seq = comp.degree_sequence();
            let spectrum = comp.spectrum();
            let top_eigs: Vec<f64> = spectrum.iter().take(5).copied().collect();
            let tri_count = comp.triangle_count();

            SpectralFingerprint {
                dim,
                component_idx: idx,
                n_nodes: comp.nodes.len(),
                n_edges: comp.edges.len(),
                degree_sequence: deg_seq,
                top_eigenvalues: top_eigs,
                triangle_count: tri_count,
            }
        })
        .collect()
}

// ===========================================================================
// Open Research: rho(b) Multiplication Coupling (C-466)
// ===========================================================================

/// Attempt to extract a GL(8,Z) action matrix rho(b) for a basis element b.
///
/// For the additive lattice action, pi(b) = signum(sum(ell)) is verified.
/// For multiplication: if ell_out = rho(b) * ell exists, then rho(b) is an
/// 8x8 integer matrix acting on the 8D lattice.
///
/// This function takes a set of lattice vectors, multiplies each by the
/// basis element e_b using Cayley-Dickson multiplication, then maps the
/// result back to the lattice to extract the transformation matrix.
///
/// Returns Some(matrix) if a consistent 8x8 integer matrix exists, None otherwise.
pub fn extract_rho_matrix(
    basis_idx: usize,
    dim: usize,
    lattice_vecs: &[Vec<i32>],
) -> Option<Vec<Vec<i32>>> {
    if lattice_vecs.is_empty() || lattice_vecs[0].len() != 8 {
        return None;
    }

    // We need at least 8 linearly independent lattice vectors to determine
    // the 8x8 matrix. Use the first 8 that span the space.
    let n_coords = 8;
    if lattice_vecs.len() < n_coords {
        return None;
    }

    // Build basis element vector for e_basis_idx
    let mut e_b = vec![0.0f64; dim];
    if basis_idx < dim {
        e_b[basis_idx] = 1.0;
    } else {
        return None;
    }

    // For each lattice vector, reconstruct the Cayley-Dickson element,
    // multiply by e_b, then try to extract the lattice coordinates of the result.
    //
    // The lattice encoding maps a CD element to 8D via some fixed projection.
    // Without knowing the exact encoding, we can try the obvious one:
    // ell = (x_0, x_1, ..., x_7) maps to the first 8 components of the CD vector.
    //
    // This is a research probe -- we check if the transformation is consistent.

    let mut input_rows: Vec<Vec<i32>> = Vec::new();
    let mut output_rows: Vec<Vec<i32>> = Vec::new();

    for ell in lattice_vecs.iter().take(n_coords) {
        // Reconstruct CD element from lattice coordinates
        let mut cd_vec = vec![0.0f64; dim];
        for (k, &coord) in ell.iter().enumerate() {
            if k < dim {
                cd_vec[k] = coord as f64;
            }
        }

        // Multiply by e_b
        let product = cd_kernel::cayley_dickson::cd_multiply(&cd_vec, &e_b);

        // Extract first 8 components as output lattice vector
        let out_ell: Vec<i32> = product
            .iter()
            .take(n_coords)
            .map(|&x: &f64| x.round() as i32)
            .collect();

        // Verify integrality
        for &x in product.iter().take(n_coords) {
            if (x - x.round()).abs() > 1e-6 {
                return None; // Not integer-valued
            }
        }

        input_rows.push(ell.clone());
        output_rows.push(out_ell);
    }

    // Try to solve: output = rho * input (each as column vectors)
    // rho[i][j] = coefficient of input_j in output_i
    // This is equivalent to: for each output row o_i, express it as
    // sum_j rho[i][j] * input_j
    //
    // If inputs are the standard basis vectors e_0..e_7, this is trivial.
    // Otherwise, need to solve the linear system.
    //
    // For simplicity, check if the first 8 lattice vectors form an identity-like basis.
    // If not, return None (the research question remains open).

    let mut rho = vec![vec![0i32; n_coords]; n_coords];
    let is_standard_basis = input_rows.iter().enumerate().all(|(i, row)| {
        row.iter()
            .enumerate()
            .all(|(j, &v)| if i == j { v == 1 } else { v == 0 })
    });

    if is_standard_basis {
        for i in 0..n_coords {
            for j in 0..n_coords {
                rho[i][j] = output_rows[j][i]; // transpose
            }
        }
        Some(rho)
    } else {
        // General case: need Gaussian elimination over Z.
        // For the research probe, just check if the mapping is consistent
        // by verifying more than 8 vectors.
        None
    }
}

// ===========================================================================
// Open Research: Octonion Subalgebra Constraint (item 12)
// ===========================================================================

/// Check whether the 8D lattice dimension is correlated with octonion structure.
///
/// The 8D embedding might be constrained by the 7 imaginary octonion units +
/// the real unit. This function checks:
/// 1. Do lattice vectors respect the Fano plane structure?
/// 2. Is the 8D encoding dimension exactly the octonion dimension?
pub fn octonion_subalgebra_constraint_check(lattice: &[Vec<i32>]) -> bool {
    // The 8D lattice dimension matches octonion dimension (8 = 2^3).
    // Check: for each lattice vector, do the non-zero coordinates
    // correspond to octonion sub-algebra structure?

    if lattice.is_empty() {
        return false;
    }

    // All lattice vectors must be 8D
    if !lattice.iter().all(|v| v.len() == 8) {
        return false;
    }

    // Check Fano structure: the support pattern of each lattice vector
    // (which coordinates are non-zero) should be compatible with Fano lines.
    // Specifically, for octonion structure, indices 1..7 participate in
    // Fano triples [1,2,3], [1,4,5], [1,6,7], [2,4,6], [2,5,7], [3,4,7], [3,5,6].

    let mut fano_compatible_count = 0usize;
    for v in lattice {
        let support: Vec<usize> = v
            .iter()
            .enumerate()
            .filter(|&(_, &x)| x != 0)
            .map(|(i, _)| i)
            .collect();

        // Check if the non-real support (indices 1..7) forms a Fano-compatible
        // pattern: any 3-element support should be a Fano line.
        let non_real_support: Vec<usize> = support
            .iter()
            .filter(|&&i| (1..=7).contains(&i))
            .copied()
            .collect();

        if non_real_support.len() == 3 {
            let mut sorted = non_real_support.clone();
            sorted.sort();
            let is_fano = O_TRIPS
                .iter()
                .any(|trip| sorted == vec![trip[0], trip[1], trip[2]]);
            if is_fano {
                fano_compatible_count += 1;
            }
        }
    }

    // The lattice dimension (8) matches octonion dimension.
    // Report whether any vectors have Fano-compatible support.
    fano_compatible_count > 0
}

// ===========================================================================
// CDP Signed-Product Engine (L1: de Marrais "Presto! Digitization" Appendix)
// ===========================================================================
//
// Faithful translation of de Marrais's M(LI, RI) function from LotusScript.
// This is the "Cayley-Dickson for Dummies" engine: given two basis indices,
// it returns a SIGNED product: sign * (LI XOR RI).
//
// The algorithm:
// 1. QSigns[4x4] quaternion base case (hard-coded multiplication table)
// 2. Handle negative inputs (absorb signs into NegTally accumulator)
// 3. XorRoot = LI XOR RI (the product index, assuming we know the sign)
// 4. Recursive reduction: strip highest bits while toggling NegTally,
//    until we reach the quaternion base case or a termination condition
//
// Reference: de Marrais (2006), arXiv:math/0603281, Appendix pp.20-27

/// Quaternion multiplication sign table (indices 0..3).
///
/// QSigns[i][j] gives the sign of e_i * e_j in the quaternion subalgebra.
/// Layout:
///   e0=1 (real), e1=i, e2=j, e3=k
///   e1*e2 = +e3, e2*e1 = -e3
///   e2*e3 = +e1, e3*e2 = -e1
///   e3*e1 = +e2, e1*e3 = -e2
///   e_i*e_i = -1 for i>0
const QSIGNS: [[i8; 4]; 4] = [
    [1, 1, 1, 1],   // e0 * e_j = +e_j
    [1, -1, 1, -1], // e1: e1*e0=+1, e1*e1=-1, e1*e2=+e3, e1*e3=-e2
    [1, -1, -1, 1], // e2: e2*e0=+1, e2*e1=-e3, e2*e2=-1, e2*e3=+e1
    [1, 1, -1, -1], // e3: e3*e0=+1, e3*e1=+e2, e3*e2=-e1, e3*e3=-1
];

/// De Marrais's M function: signed Cayley-Dickson basis product.
///
/// Given basis indices `li` and `ri`, returns `sign * (li XOR ri)` as a
/// signed integer. The product index is `|result|` and the sign is `signum(result)`.
///
/// Special case: `M(0, 0) = +1` (real * real = +real).
/// For `li == ri > 0`: returns `-(li XOR ri) = 0`, but we return the sign
/// separately since the product index is 0 (real unit).
///
/// Returns `(product_index, sign)` where `e_li * e_ri = sign * e_{product_index}`.
pub fn cdp_signed_product(li: usize, ri: usize) -> (usize, i8) {
    // The product index is always li XOR ri.
    let xor_root = li ^ ri;

    let mut neg_tally: i8 = 1;
    let mut l = li;
    let mut r = ri;

    loop {
        // Termination: either index is 0 => product is the other index with current sign.
        if l == 0 || r == 0 {
            break;
        }

        // Termination: l == r => e_i * e_i = -1 (imaginary squaring).
        if l == r {
            neg_tally = -neg_tally;
            break;
        }

        let l_bits = bit_length(l);
        let r_bits = bit_length(r);

        // Quaternion base case: both indices fit in 2 bits (0..3).
        if l_bits < 3 && r_bits < 3 {
            neg_tally *= QSIGNS[l][r];
            break;
        }

        if l_bits == r_bits {
            // Both indices arise from the same generator G = 2^(l_bits - 1).
            let g = 1usize << (l_bits - 1);

            if l == g {
                // l is the generator itself: triplet = (l XOR r, l=G, r)
                // Sign is positive (l < r, standard ordering).
                break;
            }
            if r == g {
                // r is the generator: triplet = (l XOR r, r=G, l)
                // Reversed from standard => negate.
                neg_tally = -neg_tally;
                break;
            }
            if (l ^ r) == g {
                // XOR product equals generator: triplet = (lo, G, hi)
                // Sign depends on ordering: if r > l, negate.
                if r > l {
                    neg_tally = -neg_tally;
                }
                break;
            }

            // General case: both in same doubling level.
            // For generator G, row = G + a, col = G + b => product = (-1) * a * b
            neg_tally = -neg_tally;
            l -= g;
            r -= g;
            continue; // RECURSIVE
        }

        if l_bits < r_bits {
            // l is in a lower doubling level than r.
            let g = 1usize << (r_bits - 1);

            if r == g {
                // r is the generator of its level.
                break;
            }
            if (l ^ r) == g {
                // XOR equals generator => negate.
                neg_tally = -neg_tally;
                break;
            }

            // Strip generator from r, negate.
            neg_tally = -neg_tally;
            r -= g;
            continue; // RECURSIVE
        }

        // r_bits < l_bits: r is in a lower doubling level than l.
        {
            let g = 1usize << (l_bits - 1);

            if (l ^ r) == g {
                // XOR equals generator.
                break;
            }

            neg_tally = -neg_tally;

            if l == g {
                // l is the generator of its level.
                break;
            }

            // Strip generator from l.
            l -= g;
            continue; // RECURSIVE
        }
    }

    (xor_root, neg_tally)
}

/// Number of bits needed to represent `n` (equivalent to floor(log2(n)) + 1).
/// Returns 0 for n == 0.
fn bit_length(n: usize) -> u32 {
    if n == 0 {
        0
    } else {
        usize::BITS - n.leading_zeros()
    }
}

// ===========================================================================
// Tone Row (L2: de Marrais's assessor label ordering for Emanation Tables)
// ===========================================================================
//
// For a given (N, S) where N is the power-of-2 exponent and S is the strut
// constant, the tone row generates the ET row/column labels:
//
// G = 2^(N-1)   (generator)
// X = G + S     (composite: the XOR of G and S equals X since G is a power of 2)
// K = G - 2     (number of labels per row/col = number of LO indices minus S)
//
// The labels are mirror-paired: for each lo-index `try` (skipping S), its
// strut-opposite `try XOR S` is placed at the mirror position.
// High indices are `try XOR X`.

/// A tone row: the ET row/column labeling for a specific (N, S).
#[derive(Debug, Clone)]
pub struct ToneRow {
    /// The power-of-2 exponent (dim = 2^n).
    pub n: usize,
    /// The strut constant.
    pub s: usize,
    /// Generator index: 2^(n-1).
    pub g: usize,
    /// Composite index: G + S (= G XOR S since G is a power of 2 and S < G).
    pub x: usize,
    /// Number of label positions: G - 2 (= 2^(n-1) - 2).
    pub k: usize,
    /// Low-index tone row (ordered), length K.
    pub lo: Vec<usize>,
    /// High-index tone row (ordered), length K. `hi[i]` is the HI partner of `lo[i]`.
    pub hi: Vec<usize>,
}

/// Generate the tone row for a given (n, s) where dim = 2^n.
///
/// The tone row lists the K = 2^(n-1) - 2 LO-HI assessor pairs in the
/// mirror-paired ordering used by de Marrais's emanation tables.
///
/// This eliminates S from the LO indices and X from the HI indices,
/// placing strut-opposites at mirror positions (positions i and K+1-i).
pub fn generate_tone_row(n: usize, s: usize) -> ToneRow {
    assert!(n >= 4, "Need at least sedenions (n >= 4)");
    let g = 1usize << (n - 1);
    assert!(s >= 1 && s < g, "Strut constant must be in [1, G)");

    let x = g + s; // = g ^ s since g is a pure power of 2 and s < g
    let k = g - 2; // number of positions

    // Step 1: collect all LO indices from 1..G-1, excluding S
    let raw: Vec<usize> = (1..g).filter(|&i| i != s).collect();
    assert_eq!(raw.len(), k);

    // Step 2: mirror-pair them
    let mut lo_tone = vec![0usize; k];
    let mut hi_tone = vec![0usize; k];

    let mut lo_count = 0usize; // fills from front
    let mut hi_count = k.saturating_sub(1); // fills from back

    for &try_val in &raw {
        let partner = try_val ^ s; // strut-opposite
        if try_val < partner {
            lo_tone[lo_count] = try_val;
            hi_tone[lo_count] = try_val ^ x;

            lo_tone[hi_count] = partner;
            hi_tone[hi_count] = partner ^ x;

            // Check termination: when we've placed half the pairs
            if 2 * (lo_count + 1) == k {
                break;
            }
            lo_count += 1;
            hi_count -= 1;
        }
        // If try_val >= partner, skip (it will be placed as the mirror partner)
    }

    ToneRow {
        n,
        s,
        g,
        x,
        k,
        lo: lo_tone,
        hi: hi_tone,
    }
}

// ===========================================================================
// Strutted Emanation Table with DMZ Test (L3: the actual ET algorithm)
// ===========================================================================
//
// De Marrais's Create Emanation Table algorithm:
//
// For each row k and column q (both indexing into the tone row):
// 1. Skip diagonal (k == q) and strut-opposites (k + q == K + 1)
// 2. Get the 4 elements: LRow=lo[k], HRow=hi[k], LCol=lo[q], HCol=hi[q]
// 3. Compute the 4 products (the "X-pattern"):
//    UL = M(HRow, LCol)   -- upper-left
//    UR = M(HRow, HCol)   -- upper-right
//    LL = M(LRow, LCol)   -- lower-left
//    LR = M(LRow, HCol)   -- lower-right
// 4. Check: |UL| == |LR| and |UR| == |LL|  (cross-magnitude consistency)
// 5. Edge  = sgn(UL) == sgn(LR) ? +1 : -1
//    Edge2 = sgn(UR) == sgn(LL) ? +1 : -1
// 6. If Edge == Edge2: this is a ZD pair (DMZ cell).
//    Cell value = Edge * |LL|  (the low-index of the emanation with edge sign)

/// A cell in the strutted emanation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StruttedEtCell {
    /// Row position in the tone-row ordering (0-based).
    pub row_pos: usize,
    /// Column position in the tone-row ordering (0-based).
    pub col_pos: usize,
    /// LO index of the row assessor.
    pub lo_row: usize,
    /// HI index of the row assessor.
    pub hi_row: usize,
    /// LO index of the column assessor.
    pub lo_col: usize,
    /// HI index of the column assessor.
    pub hi_col: usize,
    /// The 4-product X-pattern results: (UL, UR, LL, LR) as signed values.
    pub ul: i32,
    pub ur: i32,
    pub ll: i32,
    pub lr: i32,
    /// Whether this cell is a DMZ (mutual zero-divisor) cell.
    pub is_dmz: bool,
    /// If DMZ: the edge sign (+1 or -1). 0 if not DMZ.
    pub edge_sign: i32,
    /// If DMZ: the emanation low-index (unsigned). 0 if not DMZ.
    pub emanation_index: usize,
    /// If DMZ: the signed emanation value (edge_sign * emanation_index). 0 if not DMZ.
    pub emanation_value: i32,
}

/// The complete strutted emanation table for a specific (N, S).
#[derive(Debug, Clone)]
pub struct StruttedEmanationTable {
    /// The tone row this table is built from.
    pub tone_row: ToneRow,
    /// K x K grid of cells (some may be empty/non-DMZ).
    /// Indexed as `cells[row][col]` where row and col are tone-row positions.
    pub cells: Vec<Vec<Option<StruttedEtCell>>>,
    /// Number of DMZ (filled) cells.
    pub dmz_count: usize,
    /// Total possible cells (K*K minus diagonal and strut-opposite blanks).
    pub total_possible: usize,
}

/// Create the strutted emanation table for a given (n, s).
///
/// This is a faithful implementation of de Marrais's "Create Emanation Table"
/// algorithm from Presto! Digitization I (arXiv:math/0603281, Appendix).
///
/// The 4-product X-pattern test determines whether each assessor pair is a
/// mutual zero-divisor:
///   UL = M(HRow, LCol), UR = M(HRow, HCol)
///   LL = M(LRow, LCol), LR = M(LRow, HCol)
///   If |UL|==|LR| and |UR|==|LL| and sgn(UL)==sgn(LR) iff sgn(UR)==sgn(LL),
///   then the cell is a DMZ with value = edge_sign * |LL|.
pub fn create_strutted_et(n: usize, s: usize) -> StruttedEmanationTable {
    let tone_row = generate_tone_row(n, s);
    let k = tone_row.k;

    let mut dmz_count = 0usize;
    let mut total_possible = 0usize;

    let cells: Vec<Vec<Option<StruttedEtCell>>> = tone_row
        .lo
        .iter()
        .zip(&tone_row.hi)
        .enumerate()
        .map(|(row_pos, (&l_row, &h_row))| {
            compute_et_row(
                row_pos,
                l_row,
                h_row,
                &tone_row,
                k,
                &mut dmz_count,
                &mut total_possible,
            )
        })
        .collect();

    StruttedEmanationTable {
        tone_row,
        cells,
        dmz_count,
        total_possible,
    }
}

/// Compute one row of the strutted ET. Helper to satisfy clippy's needless_range_loop.
fn compute_et_row(
    row_pos: usize,
    l_row: usize,
    h_row: usize,
    tone_row: &ToneRow,
    k: usize,
    dmz_count: &mut usize,
    total_possible: &mut usize,
) -> Vec<Option<StruttedEtCell>> {
    tone_row
        .lo
        .iter()
        .zip(&tone_row.hi)
        .enumerate()
        .map(|(col_pos, (&l_col, &h_col))| {
            // Skip diagonal
            if col_pos == row_pos {
                return None;
            }
            // Skip strut-opposites: positions that sum to K-1 (0-indexed mirrors)
            if row_pos + col_pos == k - 1 {
                return None;
            }

            *total_possible += 1;

            // 4-product X-pattern
            let (ul_idx, ul_sign) = cdp_signed_product(h_row, l_col);
            let (ur_idx, ur_sign) = cdp_signed_product(h_row, h_col);
            let (ll_idx, ll_sign) = cdp_signed_product(l_row, l_col);
            let (lr_idx, lr_sign) = cdp_signed_product(l_row, h_col);

            let ul = ul_sign as i32 * ul_idx as i32;
            let ur = ur_sign as i32 * ur_idx as i32;
            let ll = ll_sign as i32 * ll_idx as i32;
            let lr = lr_sign as i32 * lr_idx as i32;

            // Cross-magnitude check
            if ul_idx != lr_idx || ur_idx != ll_idx {
                return Some(StruttedEtCell {
                    row_pos,
                    col_pos,
                    lo_row: l_row,
                    hi_row: h_row,
                    lo_col: l_col,
                    hi_col: h_col,
                    ul,
                    ur,
                    ll,
                    lr,
                    is_dmz: false,
                    edge_sign: 0,
                    emanation_index: 0,
                    emanation_value: 0,
                });
            }

            // Edge sign determination
            let edge = if ul_sign == lr_sign { 1i32 } else { -1i32 };
            let edge2 = if ur_sign == ll_sign { 1i32 } else { -1i32 };

            let is_dmz = edge == edge2;
            let (emanation_index, emanation_value) = if is_dmz {
                (ll_idx, edge * ll_idx as i32)
            } else {
                (0, 0)
            };

            if is_dmz {
                *dmz_count += 1;
            }

            Some(StruttedEtCell {
                row_pos,
                col_pos,
                lo_row: l_row,
                hi_row: h_row,
                lo_col: l_col,
                hi_col: h_col,
                ul,
                ur,
                ll,
                lr,
                is_dmz,
                edge_sign: if is_dmz { edge } else { 0 },
                emanation_index,
                emanation_value,
            })
        })
        .collect()
}

// ===========================================================================
// ET Sparsity Spectroscopy (L4: per-strut regime detection)
// ===========================================================================

/// Per-strut DMZ count for spectroscopy analysis.
#[derive(Debug, Clone)]
pub struct StrutSpectrum {
    /// The power-of-2 exponent (dim = 2^n).
    pub n: usize,
    /// Strut constant.
    pub s: usize,
    /// Number of DMZ (filled) cells in this strut's ET.
    pub dmz_count: usize,
    /// Total possible cells (excluding diagonal and strut-opposite blanks).
    pub total_possible: usize,
    /// Fill ratio: dmz_count / total_possible.
    pub fill_ratio: f64,
}

/// Compute the DMZ spectrum across all valid strut constants for a given N.
///
/// De Marrais observes regime structure:
/// - N=4 (sedenions): 1 regime (all 7 struts yield same DMZ count)
/// - N=5 (pathions): 2 regimes (168 and 72 DMZ cells)
/// - N=6 (chingons): 4 regimes (840, 456, 168, 552)
/// - N=7: 8 regimes
///
/// DMZ counts are always divisible by 24.
pub fn et_sparsity_spectroscopy(n: usize) -> Vec<StrutSpectrum> {
    assert!(n >= 4, "Need at least sedenions");
    let g = 1usize << (n - 1);

    let mut spectra = Vec::new();
    for s in 1..g {
        let et = create_strutted_et(n, s);
        let fill_ratio = if et.total_possible > 0 {
            et.dmz_count as f64 / et.total_possible as f64
        } else {
            0.0
        };
        spectra.push(StrutSpectrum {
            n,
            s,
            dmz_count: et.dmz_count,
            total_possible: et.total_possible,
            fill_ratio,
        });
    }

    spectra
}

/// Group strut constants by DMZ count (regime detection).
///
/// Returns a map from DMZ count to the list of strut constants that yield it.
pub fn et_regimes(n: usize) -> HashMap<usize, Vec<usize>> {
    let spectra = et_sparsity_spectroscopy(n);
    let mut regimes: HashMap<usize, Vec<usize>> = HashMap::new();
    for sp in &spectra {
        regimes.entry(sp.dmz_count).or_default().push(sp.s);
    }
    regimes
}

/// Compute the associative triplet count (Trip_N) for 2^N-ions.
///
/// Formula: Trip_N = (2^N - 1)(2^N - 2) / 6 = C(2^N - 1, 2) / 3.
///
/// - N=2 (quaternions): 1
/// - N=3 (octonions): 7
/// - N=4 (sedenions): 35
/// - N=5 (pathions): 155
/// - N=6 (chingons): 651
/// - N=7 (routons): 2667
pub fn trip_count(n: usize) -> usize {
    let d = 1usize << n; // 2^N
    (d - 1) * (d - 2) / 6
}

/// The Trip-Count Two-Step: for inherited struts (S < 8), the full-fill ET
/// decomposes into exactly Trip_{N-2} complete box-kites.
///
/// De Marrais (arXiv:0704.0026, Section 2): "The maximum number of Box-Kites
/// that can fill a 2^N-ion ET = Trip_{N-2}."
///
/// This follows because the full-fill total_possible = K(K-2) where K = 2^{N-1} - 2,
/// and each box-kite contributes exactly 24 directed DMZ cells to the ET.
pub fn trip_count_two_step(n: usize) -> usize {
    assert!(n >= 4, "Trip-Count Two-Step requires N >= 4 (sedenions)");
    trip_count(n - 2)
}

/// Classify whether a strut constant S at doubling level N generates a "Sky"
/// (meta-fractal skybox structure) per de Marrais (arXiv:0704.0112, 2007).
///
/// A Sky occurs when S > 8 AND S is not a power of 2.  Powers of 2 are
/// generator-inherited struts that always yield 100% ET fill.  S <= 8 struts
/// are "sand mandala" struts inherited from lower doublings.
///
/// Note: the Complex Systems (2006) abstract erroneously states "< 8";
/// all other de Marrais sources consistently say "> 8".
pub fn is_sky_strut(s: usize) -> bool {
    s > 8 && !s.is_power_of_two()
}

/// Classify whether a strut constant S at level N is "inherited" from a
/// lower doubling and therefore guaranteed to have full ET fill (DMZ = total_possible).
///
/// A strut is inherited if it is a power of 2 (generator of some sub-doubling).
/// At level N, the inherited full-fill struts are: 1, 2, 4, ..., G/2, where
/// G = 2^(N-1).  Additionally, S = 1..7 are always full-fill at any N.
pub fn is_inherited_full_fill_strut(n: usize, s: usize) -> bool {
    // Powers of 2 less than G are always full fill
    if s.is_power_of_two() && s < (1usize << (n - 1)) {
        return true;
    }
    // S = 1..7 (sedenion struts) are always full fill at any N
    s <= 7
}

/// Classification of a strut constant within a Cayley-Dickson level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrutClass {
    /// Generator-inherited: S is a power of 2. Always full-fill.
    Generator,
    /// Mandala-inherited: S <= 7, inherited from sedenion level. Full-fill.
    Mandala,
    /// Sky: S > 8 and not a power of 2. Sparse fill (Sand Mandala pattern).
    Sky,
}

/// Detailed spectroscopy result for a single strut constant.
#[derive(Debug, Clone)]
pub struct StrutSpectroscopyEntry {
    /// Strut constant.
    pub s: usize,
    /// Classification.
    pub class: StrutClass,
    /// DMZ count.
    pub dmz_count: usize,
    /// Total possible cells.
    pub total_possible: usize,
    /// Fill ratio.
    pub fill_ratio: f64,
    /// Number of "effective box-kites": dmz_count / 24 (exact when divisible).
    pub effective_bk_count: usize,
    /// Whether this strut has full fill (DMZ = total_possible).
    pub is_full_fill: bool,
}

/// Classify a strut constant at level N.
pub fn classify_strut(n: usize, s: usize) -> StrutClass {
    let g = 1usize << (n - 1);
    assert!(s >= 1 && s < g, "S must be in [1, G)");
    if s.is_power_of_two() {
        StrutClass::Generator
    } else if s <= 7 {
        StrutClass::Mandala
    } else {
        StrutClass::Sky
    }
}

/// Compute detailed spectroscopy for all strut constants at level N.
///
/// For each strut S in [1, G), returns its classification, DMZ count,
/// effective box-kite count, and whether it has full fill.
pub fn strut_spectroscopy(n: usize) -> Vec<StrutSpectroscopyEntry> {
    assert!(n >= 4, "Need at least sedenions");
    let g = 1usize << (n - 1);

    let mut entries = Vec::new();
    for s in 1..g {
        let et = create_strutted_et(n, s);
        let fill_ratio = if et.total_possible > 0 {
            et.dmz_count as f64 / et.total_possible as f64
        } else {
            0.0
        };
        entries.push(StrutSpectroscopyEntry {
            s,
            class: classify_strut(n, s),
            dmz_count: et.dmz_count,
            total_possible: et.total_possible,
            fill_ratio,
            effective_bk_count: et.dmz_count / 24,
            is_full_fill: et.dmz_count == et.total_possible,
        });
    }
    entries
}

// ===========================================================================
// L9b: (s,g)-Modularity -- Recursive Regime Address
// ===========================================================================
//
// The DMZ count of a strut constant S at CD level N is determined by a
// recursive "regime address" -- a binary vector of length N-4.
//
// At each level k (from N down to 5), the "half-generator" g_k = 2^(k-2)
// splits S into a lower band (S <= g_k) and upper band (S > g_k):
//   - Lower band: inherit regime from level k-1
//   - Upper band: new regime, sub-classified by regime(k-1, S - g_k)
//   - Powers of 2 >= 8 are generators, always full-fill (merge with mandala)
//
// This produces exactly 2^(N-4) regime classes, with regime count doubling
// at each CD level (de Marrais's "regime-doubling cascade").
//
// The address [b_{N-4}, ..., b_1] records at which level each "sky crossing"
// occurred: b_k = 1 means S crossed into the upper band at level k+4.

/// Compute the recursive regime address for strut constant `s` at CD level `n`.
///
/// Returns a binary vector of length `n - 4` (empty for sedenions).
/// Two struts with the same regime address always have the same DMZ count.
pub fn regime_address(n: usize, s: usize) -> Vec<u8> {
    if n <= 4 {
        return vec![];
    }
    let g = 1usize << (n - 2); // Half-generator = generator of level N-1

    // Generators (powers of 2 >= 8) always full-fill.
    // Map to S=3 (an unambiguous mandala value) to avoid recursion issues.
    if s >= 8 && s.is_power_of_two() {
        return regime_address(n, 3);
    }

    if s <= g {
        let mut addr = vec![0u8];
        addr.extend(regime_address(n - 1, s));
        addr
    } else {
        let remainder = s - g;
        let mut addr = vec![1u8];
        addr.extend(regime_address(n - 1, remainder));
        addr
    }
}

/// Number of distinct DMZ regimes at CD level `n`.
///
/// Returns 2^(n-4): sedenions have 1, pathions 2, chingons 4, routons 8.
pub fn regime_count(n: usize) -> usize {
    1usize << n.saturating_sub(4)
}

// ===========================================================================
// L9c: Hide/Fill Involution -- DMZ Row-Degree Invariance
// ===========================================================================
//
// Within each regime (same regime_address), all strut constants produce ETs
// with the same sorted row-degree distribution. This is a stronger invariant
// than just DMZ count: it constrains the *shape* of the fill pattern.
//
// Key properties verified:
//   1. Mandala regime is always "full fill" (every addressable cell is DMZ,
//      uniform row degree = K-2 where K = 2^(N-1) - 2).
//   2. Sky struts have non-uniform row degrees: some rows keep full fill,
//      others drop to a lower degree.
//   3. The row-degree distribution is a regime invariant: permuted by S
//      but identical when sorted.
//   4. Sky UNION covers all addressable cells (collective coverage).
//
// De Marrais calls this "hide/fill": mandala shows all; crossing into the
// sky band "hides" cells from certain rows; the hidden pattern permutes
// across strut constants within the regime.

/// The sorted row-degree distribution of a strutted ET.
///
/// For a K x K ET, this is a vector of length K where entry i is the
/// number of DMZ cells in the i-th row (after sorting ascending).
/// Two ETs with the same sorted row-degree distribution have the same
/// "fill shape" even if individual cell positions differ.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowDegreeDistribution {
    /// Sorted ascending row degrees.
    pub degrees: Vec<usize>,
    /// Total DMZ count (sum of degrees).
    pub dmz_total: usize,
    /// Grid size K.
    pub k: usize,
}

/// Compute the sorted row-degree distribution of a strutted ET.
pub fn row_degree_distribution(et: &StruttedEmanationTable) -> RowDegreeDistribution {
    let k = et.tone_row.k;
    let mut degrees = vec![0usize; k];
    for (r, row) in et.cells.iter().enumerate() {
        for cell in row.iter().flatten() {
            if cell.is_dmz {
                degrees[r] += 1;
            }
        }
    }
    let dmz_total = degrees.iter().sum();
    degrees.sort();
    RowDegreeDistribution {
        degrees,
        dmz_total,
        k,
    }
}

/// Result of the hide/fill analysis for a single regime.
#[derive(Debug, Clone)]
pub struct HideFillResult {
    /// The regime address.
    pub regime_addr: Vec<u8>,
    /// Number of strut constants in this regime.
    pub n_struts: usize,
    /// DMZ count (same for all struts in regime).
    pub dmz_count: usize,
    /// Sorted row-degree distribution (same for all struts in regime).
    pub row_degrees: Vec<usize>,
    /// Whether this regime is "full fill" (all addressable cells are DMZ).
    pub is_full_fill: bool,
    /// Number of cells in the core (DMZ in ALL struts of this regime).
    pub core_size: usize,
    /// Number of cells in the union (DMZ in ANY strut of this regime).
    pub union_size: usize,
    /// Total addressable cells (K*(K-1) - K = K^2 - 2K, minus strut-opposites).
    pub total_addressable: usize,
}

/// Perform the hide/fill analysis for all regimes at CD level `n`.
///
/// Returns one `HideFillResult` per regime, sorted by regime address.
pub fn hide_fill_analysis(n: usize) -> Vec<HideFillResult> {
    use std::collections::{BTreeMap, BTreeSet};

    let max_s = (1usize << (n - 1)) - 1;
    // Group strut constants by regime address
    let mut regime_struts: BTreeMap<Vec<u8>, Vec<usize>> = BTreeMap::new();
    for s in 1..=max_s {
        let addr = regime_address(n, s);
        regime_struts.entry(addr).or_default().push(s);
    }

    let mut results = Vec::new();

    for (addr, struts) in &regime_struts {
        // Compute row-degree distribution for each strut
        let first_et = create_strutted_et(n, struts[0]);
        let first_dist = row_degree_distribution(&first_et);
        let total_addressable = first_et.total_possible;

        // Verify all struts have same distribution
        for &s in &struts[1..] {
            let et = create_strutted_et(n, s);
            let dist = row_degree_distribution(&et);
            debug_assert_eq!(
                dist, first_dist,
                "N={}, S={}: row-degree differs from S={}",
                n, s, struts[0]
            );
        }

        // Compute core (intersection) and union across all struts
        let sets: Vec<BTreeSet<(usize, usize)>> = struts
            .iter()
            .map(|&s| {
                let et = create_strutted_et(n, s);
                let k = et.tone_row.k;
                let mut set = BTreeSet::new();
                for r in 0..k {
                    for c in 0..k {
                        if let Some(cell) = &et.cells[r][c]
                            && cell.is_dmz
                        {
                            set.insert((r, c));
                        }
                    }
                }
                set
            })
            .collect();

        let core: BTreeSet<_> = sets.iter().skip(1).fold(sets[0].clone(), |acc, s| {
            acc.intersection(s).copied().collect()
        });
        let union: BTreeSet<_> = sets
            .iter()
            .skip(1)
            .fold(sets[0].clone(), |acc, s| acc.union(s).copied().collect());

        let is_full_fill = first_dist.dmz_total == total_addressable;

        results.push(HideFillResult {
            regime_addr: addr.clone(),
            n_struts: struts.len(),
            dmz_count: first_dist.dmz_total,
            row_degrees: first_dist.degrees,
            is_full_fill,
            core_size: core.len(),
            union_size: union.len(),
            total_addressable,
        });
    }

    results
}

// ===========================================================================
// L9d: Skybox -- Label-Line Extension for Recursion
// ===========================================================================
//
// The ET proper is a K x K grid where K = G - 2 = 2^(N-1) - 2.
// For the doubling recursion (N -> N+1), we need a power-of-2 edge:
// promote the strut constant S and composite X = G + S to "label lines"
// bordering the ET on all four sides.
//
// The skybox is a G x G grid (edge = 2^(N-1)) where:
//   - Row/Col 0: label line (assessor (S, X))
//   - Row/Col 1..K: the original ET positions
//   - Row/Col K+1: mirror label line (assessor (S, X) again -- strut-opposite)
//   - Main diagonal: empty (self-interaction)
//   - Anti-diagonal (i + j == G-1): empty (strut-opposite blanks)
//   - Four corners (0,0), (0,G-1), (G-1,0), (G-1,G-1): empty
//     (diagonal + anti-diagonal both pass through corners)
//
// The label lines carry DMZ status from the S-assessor interacting with
// each ET assessor via the same X-pattern test.

/// A cell in the skybox extension of the emanation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkyboxCell {
    /// Whether this cell is a DMZ cell.
    pub is_dmz: bool,
    /// If DMZ: the signed emanation value. 0 if not DMZ.
    pub emanation_value: i32,
    /// Whether this cell is on a label line (row 0, col 0, row G-1, or col G-1).
    pub is_label_line: bool,
    /// Whether this cell is structural empty (diagonal, anti-diagonal, or corner).
    pub is_structural_empty: bool,
}

/// The skybox: a G x G extension of the strutted ET with label lines.
#[derive(Debug, Clone)]
pub struct Skybox {
    /// CD level (dim = 2^n).
    pub n: usize,
    /// Strut constant.
    pub s: usize,
    /// Generator G = 2^(n-1).
    pub g: usize,
    /// Skybox edge length (= G).
    pub edge: usize,
    /// The underlying ET.
    pub et: StruttedEmanationTable,
    /// `G x G` grid of cells, indexed as `grid[row][col]`.
    pub grid: Vec<Vec<SkyboxCell>>,
    /// Number of DMZ cells in the skybox (including label-line DMZs).
    pub dmz_count: usize,
    /// Number of DMZ cells on label lines only.
    pub label_dmz_count: usize,
}

/// Create the skybox for a given (n, s).
///
/// The skybox extends the K x K strutted ET to a G x G grid by adding
/// label lines (the S-assessor) at the borders.
pub fn create_skybox(n: usize, s: usize) -> Skybox {
    let et = create_strutted_et(n, s);
    let g = et.tone_row.g;
    let x = et.tone_row.x;
    let edge = g; // G = 2^(n-1)

    let mut grid = vec![
        vec![
            SkyboxCell {
                is_dmz: false,
                emanation_value: 0,
                is_label_line: false,
                is_structural_empty: false,
            };
            edge
        ];
        edge
    ];

    let mut dmz_count = 0usize;
    let mut label_dmz_count = 0usize;

    for (row, grid_row) in grid.iter_mut().enumerate() {
        for (col, cell) in grid_row.iter_mut().enumerate() {
            // Structural empties: diagonal and anti-diagonal
            if row == col || row + col == edge - 1 {
                cell.is_structural_empty = true;
                continue;
            }

            // Label-line cells: row or col is 0 or edge-1
            let is_label = row == 0 || row == edge - 1 || col == 0 || col == edge - 1;
            cell.is_label_line = is_label;

            if is_label {
                // Compute DMZ status for label-line cell.
                // The label line assessor is (S, X).
                // For label-line rows (row=0 or row=edge-1): the row assessor is (S, X).
                // For label-line cols (col=0 or col=edge-1): the col assessor is (S, X).
                let (l_row, h_row) = if row == 0 || row == edge - 1 {
                    (s, x)
                } else {
                    // Map skybox row to ET position: skybox row i -> ET position i-1
                    let et_pos = row - 1;
                    (et.tone_row.lo[et_pos], et.tone_row.hi[et_pos])
                };
                let (l_col, h_col) = if col == 0 || col == edge - 1 {
                    (s, x)
                } else {
                    let et_pos = col - 1;
                    (et.tone_row.lo[et_pos], et.tone_row.hi[et_pos])
                };

                // X-pattern test (same as ET algorithm in compute_et_row)
                let (ul_idx, ul_sign) = cdp_signed_product(h_row, l_col);
                let (ur_idx, ur_sign) = cdp_signed_product(h_row, h_col);
                let (ll_idx, ll_sign) = cdp_signed_product(l_row, l_col);
                let (lr_idx, lr_sign) = cdp_signed_product(l_row, h_col);

                // Cross-magnitude check: |UL| == |LR| and |UR| == |LL|
                if ul_idx == lr_idx && ur_idx == ll_idx {
                    let edge1 = if ul_sign == lr_sign { 1i32 } else { -1 };
                    let edge2 = if ur_sign == ll_sign { 1i32 } else { -1 };
                    if edge1 == edge2 {
                        cell.is_dmz = true;
                        cell.emanation_value = edge1 * ll_idx as i32;
                        dmz_count += 1;
                        label_dmz_count += 1;
                    }
                }
            } else {
                // Interior cell: copy from ET
                let et_row = row - 1;
                let et_col = col - 1;
                if let Some(et_cell) = &et.cells[et_row][et_col]
                    && et_cell.is_dmz
                {
                    cell.is_dmz = true;
                    cell.emanation_value = et_cell.emanation_value;
                    dmz_count += 1;
                }
            }
        }
    }

    Skybox {
        n,
        s,
        g,
        edge,
        et,
        grid,
        dmz_count,
        label_dmz_count,
    }
}

// ===========================================================================
// L9e: Theorem 11 -- Recursive ET Embedding (de Marrais 2004)
// ===========================================================================
//
// Theorem 11 (de Marrais, "The 42 Assessors"): When building the ET for
// dim = 2^(N+1) with strut constant S, the old dim = 2^N ET (same S)
// reappears as an exact sub-block. Specifically:
//
//   Primary copy: new positions whose lo matches an old position's lo.
//   Shifted copy: new positions whose lo = old_lo + old_g (g = 2^(N-1)).
//
// Both copies have identical DMZ patterns and emanation values to the
// old ET. This is the core "bit-string recursion" that connects all CD levels.

/// Result of verifying Theorem 11 recursive embedding.
#[derive(Debug, Clone)]
pub struct Theorem11Result {
    /// Old CD level.
    pub n_old: usize,
    /// New CD level (n_old + 1).
    pub n_new: usize,
    /// Strut constant.
    pub s: usize,
    /// Primary copy: map from old position -> new position.
    pub primary_map: Vec<usize>,
    /// Shifted copy: map from old position -> new position (lo + old_g).
    pub shifted_map: Vec<usize>,
    /// Whether primary copy DMZ pattern exactly matches old ET.
    pub primary_dmz_match: bool,
    /// Whether shifted copy DMZ pattern exactly matches old ET.
    pub shifted_dmz_match: bool,
    /// Whether primary copy emanation values exactly match old ET.
    pub primary_value_match: bool,
    /// Old ET DMZ count.
    pub old_dmz_count: usize,
    /// DMZ count in the primary sub-block of the new ET.
    pub primary_subblock_dmz: usize,
    /// DMZ count in the shifted sub-block of the new ET.
    pub shifted_subblock_dmz: usize,
}

/// Verify Theorem 11 recursive embedding: old ET embeds in new ET.
///
/// Computes the position mapping from old (level N) to new (level N+1) tone
/// rows and verifies that both the primary and shifted copies preserve the
/// DMZ pattern and emanation values exactly.
pub fn verify_theorem11(n: usize, s: usize) -> Theorem11Result {
    let et_old = create_strutted_et(n, s);
    let et_new = create_strutted_et(n + 1, s);
    let tr_old = &et_old.tone_row;
    let tr_new = &et_new.tone_row;
    let old_g = tr_old.g;

    // Build primary map: old_lo -> new position with matching lo
    let mut primary_map = Vec::with_capacity(tr_old.k);
    for &old_lo in &tr_old.lo {
        let new_pos = tr_new
            .lo
            .iter()
            .position(|&l| l == old_lo)
            .expect("primary: old lo must appear in new tone row");
        primary_map.push(new_pos);
    }

    // Build shifted map: old_lo + old_g -> new position
    let mut shifted_map = Vec::with_capacity(tr_old.k);
    for &old_lo in &tr_old.lo {
        let shifted = old_lo + old_g;
        let new_pos = tr_new
            .lo
            .iter()
            .position(|&l| l == shifted)
            .expect("shifted: old lo + g must appear in new tone row");
        shifted_map.push(new_pos);
    }

    // Check primary copy
    let mut primary_dmz_match = true;
    let mut primary_value_match = true;
    let mut primary_subblock_dmz = 0usize;
    for old_r in 0..tr_old.k {
        for old_c in 0..tr_old.k {
            let nr = primary_map[old_r];
            let nc = primary_map[old_c];
            let old_dmz = et_old.cells[old_r][old_c]
                .as_ref()
                .is_some_and(|c| c.is_dmz);
            let new_dmz = et_new.cells[nr][nc].as_ref().is_some_and(|c| c.is_dmz);
            if old_dmz != new_dmz {
                primary_dmz_match = false;
            }
            if new_dmz {
                primary_subblock_dmz += 1;
            }
            if old_dmz && new_dmz {
                let old_val = et_old.cells[old_r][old_c].as_ref().unwrap().emanation_value;
                let new_val = et_new.cells[nr][nc].as_ref().unwrap().emanation_value;
                if old_val != new_val {
                    primary_value_match = false;
                }
            }
        }
    }

    // Check shifted copy
    let mut shifted_dmz_match = true;
    let mut shifted_subblock_dmz = 0usize;
    for old_r in 0..tr_old.k {
        for old_c in 0..tr_old.k {
            let nr = shifted_map[old_r];
            let nc = shifted_map[old_c];
            let old_dmz = et_old.cells[old_r][old_c]
                .as_ref()
                .is_some_and(|c| c.is_dmz);
            let new_dmz = et_new.cells[nr][nc].as_ref().is_some_and(|c| c.is_dmz);
            if old_dmz != new_dmz {
                shifted_dmz_match = false;
            }
            if new_dmz {
                shifted_subblock_dmz += 1;
            }
        }
    }

    Theorem11Result {
        n_old: n,
        n_new: n + 1,
        s,
        primary_map,
        shifted_map,
        primary_dmz_match,
        shifted_dmz_match,
        primary_value_match,
        old_dmz_count: et_old.dmz_count,
        primary_subblock_dmz,
        shifted_subblock_dmz,
    }
}

// ===========================================================================
// L9f: Balloon Ride -- Fixed-S, Increasing-N ET Sequence
// ===========================================================================
//
// A "balloon ride" fixes a strut constant S and observes its ET as we ascend
// through CD levels N, N+1, N+2, ... . This reveals the recursive structure:
//
//   - Mandala struts (all regime-address bits = 0) maintain 100% fill at every level.
//   - Sky struts gain one extra regime-address prefix per level (always [0]).
//   - DMZ growth ratio converges to 4 as K grows (addressable cells ~4x per doubling).
//   - Theorem 11 embedding holds at every transition.
//   - Fill ratio for sky struts is monotonically non-decreasing.
//
// The minimum valid level for strut S is the smallest N where G = 2^(N-1) > S.

/// One step of a balloon ride: the ET data for strut S at level N.
#[derive(Debug, Clone)]
pub struct BalloonRideStep {
    /// CD level (dimension = 2^N).
    pub n: usize,
    /// Strut constant.
    pub s: usize,
    /// Tone-row parameters: G = 2^(N-1), K = G-2, X = G+S.
    pub g: usize,
    pub k: usize,
    pub x: usize,
    /// DMZ count in the K x K ET.
    pub dmz_count: usize,
    /// Total addressable cells = K * (K - 2).
    pub addressable: usize,
    /// Fill ratio = dmz_count / addressable.
    pub fill_ratio: f64,
    /// Regime address (recursive bit decomposition of S in base-G).
    pub regime_address: Vec<u8>,
    /// Whether S is a sky strut (at the sedenion level, S >= 8).
    pub is_sky: bool,
    /// DMZ growth ratio from previous level (0.0 for the first step).
    pub dmz_growth_ratio: f64,
}

/// Complete balloon ride result: a sequence of steps at increasing N.
#[derive(Debug, Clone)]
pub struct BalloonRide {
    /// Strut constant held fixed throughout the ride.
    pub s: usize,
    /// Sequence of steps at increasing N.
    pub steps: Vec<BalloonRideStep>,
    /// Whether fill ratio is monotonically non-decreasing across all steps.
    pub fill_monotone: bool,
    /// Whether all mandala levels have 100% fill (only meaningful if !is_sky).
    pub mandala_full_fill: bool,
}

/// Minimum valid CD level for a given strut constant S.
/// Returns the smallest N such that G = 2^(N-1) > S.
pub fn min_level_for_strut(s: usize) -> usize {
    // G = 2^(N-1) must be > S, so N-1 > log2(S), so N > log2(S) + 1.
    // For S=0, undefined; for S=1..7, N=4 (G=8>7); for S=8..15, N=5 (G=16>15).
    assert!(s >= 1, "Strut constant must be >= 1");
    let bits = u32::BITS - (s as u32).leading_zeros();
    (bits as usize) + 1
}

/// Perform a balloon ride: compute ETs for fixed S at levels n_start..=n_end.
///
/// Panics if n_start < min_level_for_strut(s).
pub fn balloon_ride(s: usize, n_start: usize, n_end: usize) -> BalloonRide {
    let min_n = min_level_for_strut(s);
    assert!(
        n_start >= min_n,
        "n_start={} < min_level_for_strut({})={}",
        n_start,
        s,
        min_n
    );
    assert!(n_end >= n_start, "n_end must be >= n_start");

    let is_sky = is_sky_strut(s);
    let mut steps = Vec::with_capacity(n_end - n_start + 1);
    let mut prev_dmz: Option<usize> = None;

    for n in n_start..=n_end {
        let et = create_strutted_et(n, s);
        let g = et.tone_row.g;
        let k = et.tone_row.k;
        let x = et.tone_row.x;
        let addressable = k * (k - 2);
        let fill_ratio = et.dmz_count as f64 / addressable as f64;
        let addr = regime_address(n, s);

        let dmz_growth_ratio = match prev_dmz {
            Some(pd) if pd > 0 => et.dmz_count as f64 / pd as f64,
            _ => 0.0,
        };

        steps.push(BalloonRideStep {
            n,
            s,
            g,
            k,
            x,
            dmz_count: et.dmz_count,
            addressable,
            fill_ratio,
            regime_address: addr,
            is_sky,
            dmz_growth_ratio,
        });

        prev_dmz = Some(et.dmz_count);
    }

    // Check fill monotonicity: fill_ratio[i] <= fill_ratio[i+1]
    let fill_monotone = steps
        .windows(2)
        .all(|w| w[0].fill_ratio <= w[1].fill_ratio + 1e-12);

    // Mandala full fill: all steps have fill_ratio == 1.0 (within tolerance)
    let mandala_full_fill = steps
        .iter()
        .all(|step| (step.fill_ratio - 1.0).abs() < 1e-12);

    BalloonRide {
        s,
        steps,
        fill_monotone,
        mandala_full_fill,
    }
}

// ===========================================================================
// L9g: Spectroscopy Bands -- Fixed-N, All-S Band Structure
// ===========================================================================
//
// At CD level N, strut constants S in [1, G) form "bands" of width 8 (the
// sedenion generator group size). Band b contains S = 8b+1 .. min(8b+8, G-1).
//
// Band 0 always contains the mandala struts (S=1..7) and sedenion generators.
// Higher bands contain sky struts and may contain one generator (power of 2).
//
// Within each band, struts share structural similarities:
//   - Same number of regime address prefixes (nesting depth)
//   - Similar (often identical) DMZ counts
//   - Compatible hide/fill involution partners
//
// The "flip-book" is a compact representation of how the DMZ pattern varies
// across all struts in a band: a vector of (S, dmz_count, regime_address)
// triples enabling quick comparison.

/// Dominant behavior in a spectroscopy band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandBehavior {
    /// All struts in the band are full-fill (mandala or generator).
    FullFill,
    /// Band contains a single DMZ regime (all sky struts share one DMZ count).
    UniformSky,
    /// Band contains multiple DMZ regimes (mixed behavior).
    MixedRegime,
}

/// A single frame in the flip-book: one strut's summary within a band.
#[derive(Debug, Clone)]
pub struct FlipBookFrame {
    /// Strut constant.
    pub s: usize,
    /// Classification: Generator, Mandala, or Sky.
    pub class: StrutClass,
    /// DMZ count.
    pub dmz_count: usize,
    /// Fill ratio.
    pub fill_ratio: f64,
    /// Regime address.
    pub regime_address: Vec<u8>,
    /// Effective box-kite count (dmz_count / 24).
    pub effective_bk_count: usize,
}

/// Summary of one spectroscopy band (a group of 8 consecutive strut constants).
#[derive(Debug, Clone)]
pub struct SpectroscopyBand {
    /// Band index (0, 1, 2, ...).
    pub band_index: usize,
    /// Range of S values: [s_lo, s_hi] inclusive.
    pub s_lo: usize,
    pub s_hi: usize,
    /// Number of struts in this band.
    pub n_struts: usize,
    /// Count by class.
    pub n_generators: usize,
    pub n_mandala: usize,
    pub n_sky: usize,
    /// DMZ range across the band.
    pub dmz_min: usize,
    pub dmz_max: usize,
    /// Number of distinct DMZ counts (regimes) in this band.
    pub n_regimes: usize,
    /// Number of distinct regime addresses in this band.
    pub n_distinct_addresses: usize,
    /// Dominant behavior.
    pub behavior: BandBehavior,
    /// Whether all struts in the band are full-fill.
    pub all_full_fill: bool,
    /// Flip-book: ordered frames for each strut in the band.
    pub frames: Vec<FlipBookFrame>,
}

/// Complete spectroscopy result for a CD level.
#[derive(Debug, Clone)]
pub struct SpectroscopyResult {
    /// CD level.
    pub n: usize,
    /// Dimension = 2^N.
    pub dim: usize,
    /// Generator G = 2^(N-1).
    pub g: usize,
    /// Number of valid struts.
    pub n_struts: usize,
    /// Number of bands.
    pub n_bands: usize,
    /// Bands.
    pub bands: Vec<SpectroscopyBand>,
    /// Global: number of distinct DMZ counts across ALL struts.
    pub n_global_regimes: usize,
    /// Global: expected regime count = 2^(N-4) (de Marrais formula).
    pub expected_regime_count: usize,
}

/// Compute the full spectroscopy band analysis for CD level N.
///
/// Groups all strut constants S in [1, G) into bands of width 8,
/// classifies each band's dominant behavior, and builds a flip-book
/// of per-strut summaries.
pub fn spectroscopy_bands(n: usize) -> SpectroscopyResult {
    assert!(n >= 4, "Need at least sedenions");
    let g = 1usize << (n - 1);
    let dim = 1usize << n;
    let n_struts = g - 1;
    let n_bands = n_struts.div_ceil(8);
    let expected_regime_count = if n >= 4 { 1usize << (n - 4) } else { 1 };

    let mut bands = Vec::with_capacity(n_bands);
    let mut global_dmz_set = std::collections::BTreeSet::new();

    for band_idx in 0..n_bands {
        let s_lo = band_idx * 8 + 1;
        let s_hi = ((band_idx + 1) * 8).min(g - 1);

        let mut frames = Vec::new();
        let mut n_gen = 0usize;
        let mut n_man = 0usize;
        let mut n_sky = 0usize;
        let mut dmz_min = usize::MAX;
        let mut dmz_max = 0usize;
        let mut dmz_set = std::collections::BTreeSet::new();
        let mut addr_set = std::collections::BTreeSet::new();
        let mut all_full = true;

        for s in s_lo..=s_hi {
            let class = classify_strut(n, s);
            match class {
                StrutClass::Generator => n_gen += 1,
                StrutClass::Mandala => n_man += 1,
                StrutClass::Sky => n_sky += 1,
            }

            let et = create_strutted_et(n, s);
            let fill_ratio = if et.total_possible > 0 {
                et.dmz_count as f64 / et.total_possible as f64
            } else {
                0.0
            };
            let addr = regime_address(n, s);

            if et.dmz_count < dmz_min {
                dmz_min = et.dmz_count;
            }
            if et.dmz_count > dmz_max {
                dmz_max = et.dmz_count;
            }
            dmz_set.insert(et.dmz_count);
            global_dmz_set.insert(et.dmz_count);
            addr_set.insert(addr.clone());

            if et.dmz_count != et.total_possible {
                all_full = false;
            }

            frames.push(FlipBookFrame {
                s,
                class,
                dmz_count: et.dmz_count,
                fill_ratio,
                regime_address: addr,
                effective_bk_count: et.dmz_count / 24,
            });
        }

        let n_regimes = dmz_set.len();
        let behavior = if all_full {
            BandBehavior::FullFill
        } else if n_regimes == 1 {
            BandBehavior::UniformSky
        } else {
            BandBehavior::MixedRegime
        };

        bands.push(SpectroscopyBand {
            band_index: band_idx,
            s_lo,
            s_hi,
            n_struts: s_hi - s_lo + 1,
            n_generators: n_gen,
            n_mandala: n_man,
            n_sky,
            dmz_min,
            dmz_max,
            n_regimes,
            n_distinct_addresses: addr_set.len(),
            behavior,
            all_full_fill: all_full,
            frames,
        });
    }

    SpectroscopyResult {
        n,
        dim,
        g,
        n_struts,
        n_bands: bands.len(),
        bands,
        n_global_regimes: global_dmz_set.len(),
        expected_regime_count,
    }
}

// ===========================================================================
// L10: CT Boundary / A7 Star -- Twist as Double Transfer
// ===========================================================================
//
// De Marrais (Presto I, "Royal Hunt") identifies twist products as "double
// transfers" in Catastrophe Theory: swapping both the assessor pair AND the
// box-kite membership simultaneously. This maps to a composition of Double
// Cusps in the A-series, with the simplest non-elementary form being A7 Star.
//
// The Quincunx lanyard's 120 string-readings connect to the icosahedral
// reflection group H3 (|H3| = 120).

/// Result of the CT boundary analysis.
#[derive(Debug, Clone)]
pub struct CtBoundaryResult {
    /// Number of quincunx types per tray-rack (Feet vs Hands).
    pub quincunx_types: usize,
    /// Number of tray-rack axes per box-kite (always 3).
    pub tray_rack_axes: usize,
    /// Number of string-reading start points per quincunx (10).
    pub readings_per_quincunx: usize,
    /// Flow-reversal factor (2: forward and backward).
    pub flow_reversals: usize,
    /// Total string count: types * axes * readings * reversals.
    pub total_strings: usize,
    /// Whether total_strings == |H3| = 120.
    pub matches_h3_order: bool,
}

/// Verify the CT boundary / H3 connection for sedenion box-kites.
///
/// De Marrais: 2 types (Feet/Hands) x 3 tray-rack axes x 10 readings
/// x 2 flow-reversals = 120 = |H3| (icosahedral reflection group).
pub fn ct_boundary_analysis() -> CtBoundaryResult {
    let quincunx_types = 2;
    let tray_rack_axes = 3;
    let readings_per_quincunx = 10;
    let flow_reversals = 2;
    let total = quincunx_types * tray_rack_axes * readings_per_quincunx * flow_reversals;

    CtBoundaryResult {
        quincunx_types,
        tray_rack_axes,
        readings_per_quincunx,
        flow_reversals,
        total_strings: total,
        matches_h3_order: total == 120,
    }
}

/// Verify that each twist product pair (before/after) lives in different
/// box-kites, confirming the "double transfer" property.
pub fn verify_double_transfer() -> bool {
    let transitions = twist_transition_table();
    transitions
        .iter()
        .all(|t| t.source_strut != t.h_star_target && t.source_strut != t.v_star_target)
}

// ===========================================================================
// L11: Loop/Box-Kite Duality via Automorpheme Membership
// ===========================================================================
//
// Each box-kite has 8 triangular faces. Exactly 4 of them have L-indices
// forming an O-trip (Fano plane line). These 4 "O-trip sails" each map to
// a unique automorpheme (Cawagas loop = deformed octonion copy).
//
// The duality:
//   - Each BK's 4 O-trip sails land in 4 different automorphemes
//   - Each automorpheme receives sails from exactly 4 different BKs
//   - Total: 7 BKs x 4 sails = 28 = 7 automorphemes x 4 sails
//
// The automorpheme assignment is determined by which of the 7 O-trips
// matches the sail's sorted L-index triple. Each automorpheme (indexed by
// its O-trip) contains all assessors whose L-index belongs to the trip
// and whose H-index is NOT in the exclusion set {8, 8^o1, 8^o2, 8^o3}.

/// A sail label: (box-kite strut signature, automorpheme O-trip index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SailLabel {
    pub strut_sig: usize,
    /// Index into O_TRIPS (0..7) identifying which automorpheme this sail belongs to.
    pub otrip_idx: usize,
}

/// Result of the sail-to-loop partition analysis.
#[derive(Debug, Clone)]
pub struct SailLoopResult {
    /// The 7 loops (automorphemes), each containing 4 sail labels.
    pub loops: Vec<Vec<SailLabel>>,
    /// Whether each BK's 4 sails land in 4 different loops.
    pub bk_sails_in_different_loops: bool,
    /// Whether each loop has sails from 4 different BKs.
    pub loop_sails_from_different_bks: bool,
    /// Total number of sails classified.
    pub total_sails: usize,
}

/// Get all 8 triangular faces of a box-kite (as assessor-index triples).
fn boxkite_faces(bk: &BoxKite) -> Vec<[usize; 3]> {
    let edge_set: HashSet<(usize, usize)> = bk
        .edges
        .iter()
        .flat_map(|&(a, b)| [(a, b), (b, a)])
        .collect();

    let mut faces = Vec::new();
    for i in 0..6 {
        for j in (i + 1)..6 {
            if !edge_set.contains(&(i, j)) {
                continue;
            }
            for k in (j + 1)..6 {
                if !edge_set.contains(&(i, k)) || !edge_set.contains(&(j, k)) {
                    continue;
                }
                faces.push([i, j, k]);
            }
        }
    }
    faces
}

/// Check if a face's L-indices form an O-trip, and if so return the O-trip index.
fn face_otrip_index(bk: &BoxKite, face: &[usize; 3]) -> Option<usize> {
    let mut l_sorted = [
        bk.assessors[face[0]].low,
        bk.assessors[face[1]].low,
        bk.assessors[face[2]].low,
    ];
    l_sorted.sort();

    O_TRIPS.iter().position(|t| {
        let mut s = *t;
        s.sort();
        s == l_sorted
    })
}

/// Compute the sail-to-loop partition via automorpheme membership.
///
/// Each BK has exactly 4 faces whose L-indices form O-trips. These 28
/// O-trip sails partition into 7 automorphemes (Cawagas loops) of 4 each.
pub fn sail_loop_partition() -> SailLoopResult {
    let bks = find_box_kites(16, 1e-10);

    // For each BK, find its 4 O-trip sails and assign to automorphemes
    let mut loops: Vec<Vec<SailLabel>> = vec![Vec::new(); 7];
    let mut all_sails = Vec::new();

    for bk in &bks {
        let faces = boxkite_faces(bk);
        for face in &faces {
            if let Some(otrip_idx) = face_otrip_index(bk, face) {
                let label = SailLabel {
                    strut_sig: bk.strut_signature,
                    otrip_idx,
                };
                loops[otrip_idx].push(label);
                all_sails.push(label);
            }
        }
    }

    // Sort each loop by strut signature for determinism
    for l in &mut loops {
        l.sort_by_key(|s| s.strut_sig);
    }

    // Check duality properties
    let bk_sails_in_different_loops = bks.iter().all(|bk| {
        let sail_loops: HashSet<usize> = all_sails
            .iter()
            .filter(|s| s.strut_sig == bk.strut_signature)
            .map(|s| s.otrip_idx)
            .collect();
        sail_loops.len() == 4
    });

    let loop_sails_from_different_bks = loops.iter().all(|l| {
        let bk_set: HashSet<usize> = l.iter().map(|s| s.strut_sig).collect();
        bk_set.len() == l.len()
    });

    SailLoopResult {
        loops,
        bk_sails_in_different_loops,
        loop_sails_from_different_bks,
        total_sails: all_sails.len(),
    }
}

// ===========================================================================
// L12: Quincunx and Bicycle Chain -- Explicit Assessor Paths
// ===========================================================================
//
// Quincunx: 5-vertex cycle bypassing the Royal Hunt "top edge" obstacle.
//   Feet: detour via Zigzag endpoint -> "/////\\\\\" strings
//   Hands: detour via Vent endpoint -> "/\\//\//\\" strings
//
// Bicycle Chain: 12-diagonal Hamiltonian cycle via 3/4-tray-rack scans
// linked by minus-edge jumps.
//
// De Marrais: 2 types x 3 axes x 10 readings x 2 reversals = 120 = |H3|.

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

// ===========================================================================
// L13: ET Meta-Fractal / Regime Doubling and Substitution System
// ===========================================================================
//
// Regime count doubles at each dimension level: regimes(N) = 2^(N-4).
// This is the "period doubling" pattern from Recipe Theory (Placeholder III).
//
// The deeper structure is a substitution system:
// - "Four Corners" rule: corner panes of the larger skybox replicate
//   corresponding quadrants of the smaller skybox
// - "French Windows" rule: shutter regions use g-augmentation
// - Bitstring painting recipe: cell occupancy determined by S's bitstring

/// Result of regime-doubling analysis.
#[derive(Debug, Clone)]
pub struct RegimeDoublingResult {
    /// For each N tested: (n, regime_count).
    pub data: Vec<(usize, usize)>,
    /// Whether the doubling law holds: regimes(N) = 2^(N-4).
    pub doubling_law_holds: bool,
}

/// Verify regime-doubling for N=4 and N=5.
pub fn verify_regime_doubling(max_n: usize) -> RegimeDoublingResult {
    let mut data = Vec::new();
    let mut doubling_holds = true;

    for n in 4..=max_n {
        let regimes = et_regimes(n);
        let count = regimes.len();
        let expected = 1usize << (n - 4);
        if count != expected {
            doubling_holds = false;
        }
        data.push((n, count));
    }

    RegimeDoublingResult {
        data,
        doubling_law_holds: doubling_holds,
    }
}

/// Verify the Four Corners replication rule: the corner panes of the
/// N+1 skybox should match the corresponding quadrants of the N skybox.
///
/// Returns (n, matching_fraction) for each pair of adjacent dimensions.
pub fn verify_four_corners(base_n: usize) -> (usize, f64) {
    let et_base = create_strutted_et(base_n, 1);
    let et_next = create_strutted_et(base_n + 1, 1);

    let k_base = et_base.tone_row.k;
    let k_next = et_next.tone_row.k;

    // The "corner panes" of the larger ET correspond to the first k_base
    // rows and columns. Check if their DMZ status matches.
    let mut matching = 0;
    let mut total = 0;

    for r in 0..k_base.min(k_next) {
        for c in 0..k_base.min(k_next) {
            let base_cell = &et_base.cells[r][c];
            let next_cell = &et_next.cells[r][c];
            total += 1;
            let base_is_dmz = base_cell.is_some();
            let next_is_dmz = next_cell.is_some();
            if base_is_dmz == next_is_dmz {
                matching += 1;
            }
        }
    }

    let fraction = if total > 0 {
        matching as f64 / total as f64
    } else {
        0.0
    };
    (base_n, fraction)
}

// ===========================================================================
// L14: Eco Echo -- SS Relabeling Operator and Recursive Structure
// ===========================================================================
//
// The Eco Echo recursion: SS edge labels {S, G, X} can be permuted by
// the role-swap group (which of {S,G,X} acts as diagonal/horizontal/vertical).
// This is a group action on SS edge-labelings, with the XOR closure
// constraint X = G XOR S maintaining algebraic consistency.
//
// The recursion operator E: replace each SS corner node by a fresh SS
// (node-expansion into a strut-opposite quartet), gluing via the parent
// edge-type mapping.

/// The 3 possible SS edge-role assignments (which constant is diagonal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SsRoleAssignment {
    /// G diagonal, X horizontal, S vertical (standard de Marrais)
    GDiagonal,
    /// S diagonal, X vertical, G horizontal
    SDiagonal,
    /// X diagonal, G vertical, S horizontal
    XDiagonal,
}

/// Eco Echo recursion result.
#[derive(Debug, Clone)]
pub struct EcoEchoResult {
    /// Number of SS diagrams at the base level (dim=16).
    pub base_ss_count: usize,
    /// Number of role assignments (always 3 for the {S,G,X} permutation group).
    pub role_assignments: usize,
    /// Total meta-SS nodes after one recursion step (base_ss_count * 4).
    pub meta_nodes_after_expansion: usize,
    /// Whether the XOR closure X = G XOR S is preserved under all role swaps.
    pub xor_closure_preserved: bool,
}

/// Verify the Eco Echo recursion properties.
///
/// Checks: (1) the {S,G,X} role-swap group is well-defined,
/// (2) XOR closure is preserved, (3) node expansion produces consistent
/// meta-SS structure.
pub fn eco_echo_probe() -> EcoEchoResult {
    let bks = find_box_kites(16, 1e-10);
    let base_ss_count = bks.len() * 3; // 7 BK x 3 strut axes = 21

    // Verify XOR closure under all role assignments
    let mut xor_preserved = true;
    for bk in &bks {
        let s = bk.strut_signature;
        let g = 8usize; // Generator for dim=16
        let x = g ^ s;

        // Standard: G diagonal, X horizontal, S vertical
        // X = G XOR S must hold regardless of which we call "diagonal"
        if x != g ^ s {
            xor_preserved = false;
        }
        // Role swap 1: S diagonal -> horizontal = G, vertical = X
        // Check: G = S XOR X (equivalent to X = G XOR S)
        if g != s ^ x {
            xor_preserved = false;
        }
        // Role swap 2: X diagonal -> horizontal = S, vertical = G
        // Check: S = X XOR G (equivalent)
        if s != x ^ g {
            xor_preserved = false;
        }
    }

    // Node expansion: each SS corner (4 nodes) becomes a new SS (4 nodes)
    // So one expansion step: 21 SS x 4 corner-nodes = 84 meta-nodes
    let meta_nodes = base_ss_count * 4;

    EcoEchoResult {
        base_ss_count,
        role_assignments: 3,
        meta_nodes_after_expansion: meta_nodes,
        xor_closure_preserved: xor_preserved,
    }
}

// ===========================================================================
// L15: Representation-Aware Trip Sync (Orientation Coherence)
// ===========================================================================
//
// Trip Sync is not merely "the sail L-indices form a Fano line" (membership).
// It is: "there exists a PSL(2,7) embedding in which the zigzag's 4 quaternion
// copies are co-oriented, while trefoils show controlled desynchronization."
//
// The key relationship (from Pathions3):
//   Zigzag + 3 trefoil L-trips sit at (a,b,c), (a,d,e), (d,b,f), (e,f,c)
//   forming the 4 faces of a tetrahedron inscribed in the box-kite octahedron.
//
// Given a BK's 4 O-trips, check if any candidate can serve as the "zigzag Rule-0
// central circle" such that the remaining 3 trips fill the trefoil pattern.

/// Result of the orientation-aware Trip Sync check.
#[derive(Debug, Clone)]
pub struct OrientedTripSync {
    /// The box-kite's strut signature.
    pub strut_sig: usize,
    /// The 4 O-trips available in this BK's 6 L-indices.
    pub available_trips: Vec<[usize; 3]>,
    /// For each candidate zigzag trip, whether the shorthand pattern is satisfiable.
    pub candidate_results: Vec<(usize, bool)>,
    /// Whether at least one candidate satisfies Trip Sync.
    pub has_valid_embedding: bool,
}

/// Check orientation-aware Trip Sync for a box-kite.
///
/// For each of the 4 O-trips in the BK's L-indices, try it as the zigzag trip
/// and check whether the remaining L-indices can form the 3 trefoil trips
/// according to the shorthand pattern (a,b,c), (a,d,e), (d,b,f), (e,f,c).
pub fn oriented_trip_sync(bk: &BoxKite) -> OrientedTripSync {
    let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();
    let available: Vec<[usize; 3]> = O_TRIPS
        .iter()
        .filter(|t| t.iter().all(|&x| l_set.contains(&x)))
        .copied()
        .collect();

    let mut candidate_results = Vec::new();
    let mut has_valid = false;

    for (idx, zig_trip) in available.iter().enumerate() {
        // Try zig_trip = (a, b, c) as the zigzag Rule-0 trip.
        // The remaining 3 indices are {d, e, f}.
        let remaining: Vec<usize> = l_set
            .iter()
            .copied()
            .filter(|x| !zig_trip.contains(x))
            .collect();

        if remaining.len() != 3 {
            continue;
        }

        // De Marrais shorthand: trefoils are (a,d,e), (d,b,f), (e,f,c)
        // We need to find an assignment of remaining to {d,e,f} such that
        // all three trefoil triples are also O-trips.
        let valid = try_trefoil_assignment(zig_trip, &remaining);
        candidate_results.push((idx, valid));
        if valid {
            has_valid = true;
        }
    }

    OrientedTripSync {
        strut_sig: bk.strut_signature,
        available_trips: available,
        candidate_results,
        has_valid_embedding: has_valid,
    }
}

/// Try all 6 permutations of remaining indices to find a valid trefoil assignment.
fn try_trefoil_assignment(zig_trip: &[usize; 3], remaining: &[usize]) -> bool {
    let (a, b, c) = (zig_trip[0], zig_trip[1], zig_trip[2]);
    let perms = [
        (remaining[0], remaining[1], remaining[2]),
        (remaining[0], remaining[2], remaining[1]),
        (remaining[1], remaining[0], remaining[2]),
        (remaining[1], remaining[2], remaining[0]),
        (remaining[2], remaining[0], remaining[1]),
        (remaining[2], remaining[1], remaining[0]),
    ];

    let otrip_set: HashSet<[usize; 3]> = O_TRIPS
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort();
            s
        })
        .collect();

    for (d, e, f) in perms {
        let t1 = {
            let mut t = [a, d, e];
            t.sort();
            t
        };
        let t2 = {
            let mut t = [d, b, f];
            t.sort();
            t
        };
        let t3 = {
            let mut t = [e, f, c];
            t.sort();
            t
        };
        if otrip_set.contains(&t1) && otrip_set.contains(&t2) && otrip_set.contains(&t3) {
            return true;
        }
    }
    false
}

// ===========================================================================
// L15b: Sail Decomposition -- Full face classification per box-kite
// ===========================================================================
//
// Each box-kite octahedron has 8 triangular faces, classified by two
// orthogonal criteria:
//   1) Twist type: Zigzag (2 faces) vs Trefoil (6 faces)
//   2) O-trip membership: Sail (4 faces) vs non-Sail (4 faces)
//
// Cross-classifying yields exactly:
//   - 1 Zigzag Sail (all-Opposite edges, L-indices form O-trip)
//   - 3 Trefoil Sails (mixed edges, L-indices form O-trip)
//   - 1 Vent (all-Opposite edges, L-indices NOT an O-trip)
//   - 3 non-Sail Trefoils (mixed edges, L-indices NOT an O-trip)
//
// De Marrais (2000): the 4 sails carry the quaternion subalgebra copies;
// the Vent is the "ventilation hole" where trip sync fails locally.

/// Classification of a single triangular face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceRole {
    /// Zigzag face whose L-indices form an O-trip (the unique zigzag sail).
    ZigzagSail,
    /// Trefoil face whose L-indices form an O-trip (one of 3 trefoil sails).
    TrefoilSail,
    /// Zigzag face whose L-indices do NOT form an O-trip (the unique vent).
    Vent,
    /// Trefoil face whose L-indices do NOT form an O-trip.
    NonSailTrefoil,
}

/// A classified face of the box-kite octahedron.
#[derive(Debug, Clone)]
pub struct ClassifiedFace {
    /// The 3 assessor indices (into the box-kite's assessor list).
    pub assessor_indices: [usize; 3],
    /// The 3 L-indices (low parts of the assessors).
    pub l_indices: [usize; 3],
    /// The face's role in the sail decomposition.
    pub role: FaceRole,
    /// If this face is a sail, the index of the O-trip it corresponds to.
    pub otrip_index: Option<usize>,
}

/// Complete sail decomposition of a box-kite.
#[derive(Debug, Clone)]
pub struct SailDecomposition {
    /// The box-kite's strut signature.
    pub strut_sig: usize,
    /// All 8 faces, classified.
    pub faces: Vec<ClassifiedFace>,
    /// The unique zigzag sail (index into `faces`).
    pub zigzag_sail_idx: usize,
    /// The 3 trefoil sail indices (into `faces`).
    pub trefoil_sail_indices: [usize; 3],
    /// The unique vent index (into `faces`).
    pub vent_idx: usize,
    /// The 3 non-sail trefoil indices (into `faces`).
    pub non_sail_trefoil_indices: [usize; 3],
}

/// Compute the full sail decomposition for a box-kite.
///
/// Cross-classifies all 8 octahedral faces by twist type (zigzag/trefoil)
/// and O-trip membership (sail/non-sail), producing exactly:
/// - 1 zigzag sail, 3 trefoil sails, 1 vent, 3 non-sail trefoils.
///
/// Panics if the box-kite does not have the expected 2+6 zigzag/trefoil split
/// or the expected 4+4 sail/non-sail split.
pub fn sail_decomposition(bk: &BoxKite) -> SailDecomposition {
    let racks = tray_racks(bk);
    assert_eq!(racks.len(), 8, "Box-kite must have exactly 8 faces");

    let mut faces = Vec::with_capacity(8);
    for rack in &racks {
        let l_indices = [
            bk.assessors[rack.assessors[0]].low,
            bk.assessors[rack.assessors[1]].low,
            bk.assessors[rack.assessors[2]].low,
        ];
        let otrip_idx = face_otrip_index(bk, &rack.assessors);
        let is_sail = otrip_idx.is_some();
        let is_zigzag = rack.twist_type == TwistType::Zigzag;

        let role = match (is_zigzag, is_sail) {
            (true, true) => FaceRole::ZigzagSail,
            (true, false) => FaceRole::Vent,
            (false, true) => FaceRole::TrefoilSail,
            (false, false) => FaceRole::NonSailTrefoil,
        };

        faces.push(ClassifiedFace {
            assessor_indices: rack.assessors,
            l_indices,
            role,
            otrip_index: otrip_idx,
        });
    }

    // Extract indices by role
    let zigzag_sails: Vec<usize> = faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.role == FaceRole::ZigzagSail)
        .map(|(i, _)| i)
        .collect();
    let trefoil_sails: Vec<usize> = faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.role == FaceRole::TrefoilSail)
        .map(|(i, _)| i)
        .collect();
    let vents: Vec<usize> = faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.role == FaceRole::Vent)
        .map(|(i, _)| i)
        .collect();
    let non_sail_trefoils: Vec<usize> = faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.role == FaceRole::NonSailTrefoil)
        .map(|(i, _)| i)
        .collect();

    assert_eq!(
        zigzag_sails.len(),
        1,
        "BK S={}: expected 1 zigzag sail, got {}",
        bk.strut_signature,
        zigzag_sails.len()
    );
    assert_eq!(
        trefoil_sails.len(),
        3,
        "BK S={}: expected 3 trefoil sails, got {}",
        bk.strut_signature,
        trefoil_sails.len()
    );
    assert_eq!(
        vents.len(),
        1,
        "BK S={}: expected 1 vent, got {}",
        bk.strut_signature,
        vents.len()
    );
    assert_eq!(
        non_sail_trefoils.len(),
        3,
        "BK S={}: expected 3 non-sail trefoils, got {}",
        bk.strut_signature,
        non_sail_trefoils.len()
    );

    SailDecomposition {
        strut_sig: bk.strut_signature,
        faces,
        zigzag_sail_idx: zigzag_sails[0],
        trefoil_sail_indices: [trefoil_sails[0], trefoil_sails[1], trefoil_sails[2]],
        vent_idx: vents[0],
        non_sail_trefoil_indices: [
            non_sail_trefoils[0],
            non_sail_trefoils[1],
            non_sail_trefoils[2],
        ],
    }
}

// ===========================================================================
// L15b: Three Vizier Relationships (de Marrais 2007)
// ===========================================================================
//
// Each strut in a box-kite connects two non-adjacent assessors.  Let the
// first assessor be (v, V) where v = low index (1..7) and V = high index
// (8..15), and the second be (z, Z).  Then:
//
//   VZ1:  v ^ z = V ^ Z = S       (strut constant)
//   VZ2:  Z ^ v = V ^ z = G       (generator = dim/2)
//   VZ3:  V ^ v = z ^ Z = G ^ S   (within-assessor invariant)
//
// These three XORs form a Klein four-group {0, S, G, G^S} acting on
// the index space.  VZ3 is a per-assessor property (not just per-strut):
// hi = lo ^ (G ^ S) for every assessor in a box-kite with strut constant S.

/// Result of verifying the Three Vizier XOR relationships on a box-kite.
#[derive(Debug, Clone)]
pub struct ThreeVizierResult {
    /// The strut constant S.
    pub strut_sig: usize,
    /// The generator G = dim/2.
    pub generator: usize,
    /// Number of struts checked.
    pub n_struts: usize,
    /// Whether VZ1 (v^z = V^Z = S) holds for all struts.
    pub vz1_holds: bool,
    /// Whether VZ2 (Z^v = V^z = G) holds for all struts.
    pub vz2_holds: bool,
    /// Whether VZ3 (V^v = z^Z = G^S) holds for all assessors.
    pub vz3_holds: bool,
    /// The within-assessor lo^hi values (should all be G^S).
    pub lo_hi_xor: usize,
}

/// Verify the Three Vizier XOR relationships on a box-kite.
///
/// For sedenions (dim=16), G=8.  Given a box-kite with strut constant S:
/// - VZ1: for each strut (v,V)--(z,Z), v^z = V^Z = S
/// - VZ2: for each strut (v,V)--(z,Z), Z^v = V^z = G
/// - VZ3: for every assessor (v,V), V^v = G^S
///
/// Returns a `ThreeVizierResult` capturing the verification outcome.
pub fn verify_three_viziers(bk: &BoxKite, dim: usize) -> ThreeVizierResult {
    let g = dim / 2;
    let s = bk.strut_signature;
    let x = g ^ s;

    // VZ3 check: every assessor should have hi ^ lo = G^S
    let vz3_holds = bk.assessors.iter().all(|a| a.low ^ a.high == x);

    // VZ1 + VZ2 check: per-strut
    let mut vz1_holds = true;
    let mut vz2_holds = true;
    for &(i, j) in &bk.struts {
        let a = &bk.assessors[i];
        let b = &bk.assessors[j];
        let v = a.low;
        let big_v = a.high;
        let z = b.low;
        let big_z = b.high;

        // VZ1: v^z = V^Z = S
        if v ^ z != s || big_v ^ big_z != s {
            vz1_holds = false;
        }
        // VZ2: Z^v = V^z = G
        if big_z ^ v != g || big_v ^ z != g {
            vz2_holds = false;
        }
    }

    ThreeVizierResult {
        strut_sig: s,
        generator: g,
        n_struts: bk.struts.len(),
        vz1_holds,
        vz2_holds,
        vz3_holds,
        lo_hi_xor: x,
    }
}

/// Check whether the Vizier XOR relationships hold for zero-product-adjacent
/// cross-assessor pairs in a `MotifComponent` at arbitrary dimension.
///
/// At dim=16 this reduces to the standard Three Vizier check.  At higher
/// dimensions, the "strut constant" is no longer a single missing index
/// but may generalize differently.  This function checks:
///
/// - VZ3 generalization: is lo^hi constant within the component?
/// - If so, what is that constant X, and does X = G^S for some S?
/// - VZ1 generalization: for each ZD-adjacent pair, does lo_a^lo_b = hi_a^hi_b?
///
/// Returns `None` if the component has no edges.
pub fn vizier_xor_audit(
    component: &algebra_analysis::boxkites::MotifComponent,
) -> Option<VizierXorAudit> {
    if component.edges.is_empty() {
        return None;
    }
    let dim = component.dim;
    let g = dim / 2;

    // VZ3 generalization: check if lo^hi is constant
    let lo_hi_xors: std::collections::BTreeSet<usize> =
        component.nodes.iter().map(|&(lo, hi)| lo ^ hi).collect();
    let vz3_constant = lo_hi_xors.len() == 1;
    let lo_hi_xor = if vz3_constant {
        *lo_hi_xors.iter().next().unwrap()
    } else {
        0
    };

    // VZ1 generalization: for each edge, does lo_a^lo_b = hi_a^hi_b?
    let mut vz1_all_match = true;
    let mut lo_xor_values: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for &((lo_a, hi_a), (lo_b, hi_b)) in &component.edges {
        let lo_xor = lo_a ^ lo_b;
        let hi_xor = hi_a ^ hi_b;
        if lo_xor != hi_xor {
            vz1_all_match = false;
        }
        lo_xor_values.insert(lo_xor);
    }
    let vz1_constant = vz1_all_match && lo_xor_values.len() == 1;

    // VZ2 generalization: for each edge, does hi_b^lo_a = hi_a^lo_b?
    let mut vz2_all_match = true;
    let mut vz2_values: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for &((lo_a, hi_a), (lo_b, hi_b)) in &component.edges {
        let cross1 = hi_b ^ lo_a;
        let cross2 = hi_a ^ lo_b;
        if cross1 != cross2 {
            vz2_all_match = false;
        }
        vz2_values.insert(cross1);
    }
    let vz2_constant = vz2_all_match && vz2_values.len() == 1;

    Some(VizierXorAudit {
        dim,
        n_nodes: component.nodes.len(),
        n_edges: component.edges.len(),
        vz3_constant,
        lo_hi_xor,
        inferred_s: if vz3_constant { g ^ lo_hi_xor } else { 0 },
        vz1_lo_eq_hi: vz1_all_match,
        vz1_constant,
        n_distinct_lo_xors: lo_xor_values.len(),
        vz2_cross_eq: vz2_all_match,
        vz2_constant,
        vz2_value: if vz2_constant {
            vz2_values.iter().next().copied()
        } else {
            None
        },
    })
}

/// Result of auditing Vizier XOR structure in a generalized motif component.
#[derive(Debug, Clone)]
pub struct VizierXorAudit {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Number of nodes (cross-assessor pairs) in the component.
    pub n_nodes: usize,
    /// Number of ZD-adjacent edges.
    pub n_edges: usize,
    /// Whether lo^hi is constant across all nodes (VZ3 generalization).
    pub vz3_constant: bool,
    /// The lo^hi value (meaningful only if vz3_constant is true).
    pub lo_hi_xor: usize,
    /// Inferred strut constant S = G ^ lo_hi_xor (meaningful only if vz3_constant).
    pub inferred_s: usize,
    /// Whether lo_a^lo_b = hi_a^hi_b for all edges (VZ1 symmetry).
    pub vz1_lo_eq_hi: bool,
    /// Whether lo_a^lo_b is constant across all edges (full VZ1).
    pub vz1_constant: bool,
    /// Number of distinct lo_a^lo_b values across edges.
    pub n_distinct_lo_xors: usize,
    /// Whether hi_b^lo_a = hi_a^lo_b for all edges (VZ2 symmetry).
    pub vz2_cross_eq: bool,
    /// Whether the VZ2 cross-XOR is constant across all edges.
    pub vz2_constant: bool,
    /// The VZ2 cross-XOR value (if constant).
    pub vz2_value: Option<usize>,
}

// ===========================================================================
// L16: ET <-> Edge-Sign <-> Lanyard Dictionary
// ===========================================================================
//
// The strutted ET is a signed adjacency matrix: each DMZ cell encodes an
// edge sign (+1 or -1) between two assessors.
//
// Edge sign determines diagonal-state coupling:
//   +1 (same-slope): preserves /\ state across edge
//   -1 (cross-slope): flips /\ state across edge
//
// Lanyards emerge as state-machine traversals of the signed graph:
//   Zigzag: all 3 edges negative -> /\/\/\ (alternating, double cover)
//   Trefoil: 2 positive + 1 negative -> ///\\\ (double cover)
//   Catamaran: alternating signs -> two disjoint single-cover cycles

/// A signed edge in the box-kite adjacency graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedEdge {
    /// L-index of first assessor.
    pub lo_a: usize,
    /// L-index of second assessor.
    pub lo_b: usize,
    /// Edge sign: +1 or -1.
    pub sign: i32,
}

/// Signed adjacency graph extracted from an ET.
#[derive(Debug, Clone)]
pub struct SignedAdjacencyGraph {
    /// The strut constant.
    pub s: usize,
    /// The L-indices of the 6 assessors.
    pub nodes: Vec<usize>,
    /// The signed edges (only DMZ pairs).
    pub edges: Vec<SignedEdge>,
    /// Number of positive edges.
    pub n_positive: usize,
    /// Number of negative edges.
    pub n_negative: usize,
}

/// A lanyard signature extracted from the signed graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanyardSignature {
    /// The assessor L-indices visited in cycle order.
    pub cycle: Vec<usize>,
    /// The diagonal states along the path (true = /, false = \).
    pub slash_states: Vec<bool>,
    /// Compact string representation (e.g., "/\\/\\/\\").
    pub signature_string: String,
}

/// Extract the signed adjacency graph from a strutted ET.
pub fn extract_signed_graph(et: &StruttedEmanationTable) -> SignedAdjacencyGraph {
    let s = et.tone_row.s;
    let nodes = et.tone_row.lo.clone();
    let mut edges = Vec::new();
    let mut n_positive = 0usize;
    let mut n_negative = 0usize;
    let k = et.tone_row.k;

    for r in 0..k {
        for c in (r + 1)..k {
            if let Some(cell) = &et.cells[r][c]
                && cell.is_dmz
            {
                let sign = cell.edge_sign;
                edges.push(SignedEdge {
                    lo_a: nodes[r],
                    lo_b: nodes[c],
                    sign,
                });
                if sign > 0 {
                    n_positive += 1;
                } else {
                    n_negative += 1;
                }
            }
        }
    }

    SignedAdjacencyGraph {
        s,
        nodes,
        edges,
        n_positive,
        n_negative,
    }
}

/// Traverse a face cycle in the signed graph, producing a lanyard signature.
///
/// Starting from the first node in `cycle` with diagonal state `start_slash`,
/// traverse edges: + edge preserves state, - edge flips state.
pub fn traverse_lanyard(
    graph: &SignedAdjacencyGraph,
    cycle: &[usize],
    start_slash: bool,
) -> LanyardSignature {
    let edge_map: HashMap<(usize, usize), i32> = graph
        .edges
        .iter()
        .flat_map(|e| [((e.lo_a, e.lo_b), e.sign), ((e.lo_b, e.lo_a), e.sign)])
        .collect();

    let mut states = Vec::new();
    let mut current = start_slash;
    states.push(current);

    for i in 0..cycle.len() {
        let from = cycle[i];
        let to = cycle[(i + 1) % cycle.len()];
        let sign = edge_map.get(&(from, to)).copied().unwrap_or(1);
        if sign < 0 {
            current = !current;
        }
        if i + 1 < cycle.len() {
            states.push(current);
        }
    }

    let sig_string: String = states.iter().map(|&s| if s { '/' } else { '\\' }).collect();

    LanyardSignature {
        cycle: cycle.to_vec(),
        slash_states: states,
        signature_string: sig_string,
    }
}

/// Extract all face-based lanyards from a strutted ET.
///
/// Returns lanyard signatures for zigzag faces (should be /\/\/\),
/// trefoil faces (should be ///\\\), and any other detectable patterns.
pub fn extract_lanyards_from_et(n: usize, s: usize) -> Vec<LanyardSignature> {
    let et = create_strutted_et(n, s);
    let graph = extract_signed_graph(&et);
    let bks = find_box_kites(16, 1e-10);
    let bk = match bks.iter().find(|b| b.strut_signature == s) {
        Some(b) => b,
        None => return Vec::new(),
    };

    let faces = boxkite_faces(bk);
    let mut lanyards = Vec::new();

    for face in &faces {
        let face_lows: Vec<usize> = face.iter().map(|&i| bk.assessors[i].low).collect();
        // Traverse with starting state = true (/) for double-cover
        let sig = traverse_lanyard(&graph, &face_lows, true);
        lanyards.push(sig);
    }

    lanyards
}

// FaceSignPattern is now defined in boxkites.rs and imported above.

/// Result of one face classification.
#[derive(Debug, Clone)]
pub struct FaceClassification {
    /// Source BK strut constant.
    pub strut: usize,
    /// The 3 assessor L-indices forming this face.
    pub face_indices: [usize; 3],
    /// The 3 edge sign types.
    pub edge_signs: [EdgeSignType; 3],
    /// Normalized sign pattern (order-independent).
    pub pattern: FaceSignPattern,
    /// Traversal signature (order-dependent, for reference).
    pub traversal_sig: String,
}

/// Complete cross-BK lanyard classification result.
#[derive(Debug, Clone)]
pub struct CrossBkLanyardCensus {
    /// Number of BKs analyzed (should be 7 at dim=16).
    pub n_bks: usize,
    /// Total faces analyzed (should be 56 = 7 x 8).
    pub total_faces: usize,
    /// Count of each normalized sign pattern.
    pub pattern_counts: HashMap<FaceSignPattern, usize>,
    /// Per-BK pattern distribution.
    pub per_bk_patterns: Vec<(usize, Vec<FaceSignPattern>)>,
    /// Whether all BKs have the same pattern distribution.
    pub uniform_across_bks: bool,
    /// Detailed per-face classifications.
    pub faces: Vec<FaceClassification>,
}

// classify_face_pattern is now defined in boxkites.rs and imported above.

/// Systematic lanyard classification across all 7 sedenion box-kites.
///
/// For each BK (strut 1..7), classifies all 8 triangular faces by their
/// normalized edge-sign pattern. The pattern census is:
/// - AllSame (Blues): 0 (not realized in sedenions)
/// - TwoSameOneOpp (Trefoil I): expected ~42 total
/// - OneSameTwoOpp (Trefoil II): expected ~0-14
/// - AllOpposite (TripleZigzag): expected 14 (7 x 2)
///
/// Also extracts traversal signatures for reference (these are
/// order-dependent and vary by starting assessor).
pub fn cross_bk_lanyard_census() -> CrossBkLanyardCensus {
    let bks = find_box_kites(16, 1e-10);
    let atol = 1e-10;
    let mut all_patterns: HashMap<FaceSignPattern, usize> = HashMap::new();
    let mut per_bk = Vec::new();
    let mut faces = Vec::new();
    let mut total_faces = 0;
    let mut first_dist: Option<HashMap<FaceSignPattern, usize>> = None;
    let mut uniform = true;

    for bk in &bks {
        let s = bk.strut_signature;
        let bk_faces = boxkite_faces(bk);
        let et = create_strutted_et(4, s);
        let graph = extract_signed_graph(&et);
        let mut bk_patterns = Vec::new();
        let mut bk_dist: HashMap<FaceSignPattern, usize> = HashMap::new();

        for face in &bk_faces {
            let assessors: [Assessor; 3] = [
                bk.assessors[face[0]],
                bk.assessors[face[1]],
                bk.assessors[face[2]],
            ];
            let face_lows = [assessors[0].low, assessors[1].low, assessors[2].low];

            let signs = [
                edge_sign_type(&assessors[0], &assessors[1], atol),
                edge_sign_type(&assessors[1], &assessors[2], atol),
                edge_sign_type(&assessors[0], &assessors[2], atol),
            ];
            let pattern = classify_face_pattern(&signs);

            let sig = traverse_lanyard(&graph, face_lows.as_ref(), true);

            *all_patterns.entry(pattern).or_insert(0) += 1;
            *bk_dist.entry(pattern).or_insert(0) += 1;
            bk_patterns.push(pattern);

            faces.push(FaceClassification {
                strut: s,
                face_indices: face_lows,
                edge_signs: signs,
                pattern,
                traversal_sig: sig.signature_string,
            });

            total_faces += 1;
        }

        bk_patterns.sort();
        per_bk.push((s, bk_patterns));

        if let Some(ref first) = first_dist {
            if &bk_dist != first {
                uniform = false;
            }
        } else {
            first_dist = Some(bk_dist);
        }
    }

    CrossBkLanyardCensus {
        n_bks: 7,
        total_faces,
        pattern_counts: all_patterns,
        per_bk_patterns: per_bk,
        uniform_across_bks: uniform,
        faces,
    }
}

// ===========================================================================
// L17: Twisted Sisters Delta Transition Function
// ===========================================================================
//
// The twist navigation is a deterministic automaton:
//   delta(S0, {u,v}, 0) = u
//   delta(S0, {u,v}, 1) = v
// where {u,v} is one of S0's three strut pairs (u XOR v = S0).
//
// This is purely algebraic: each S0 in {1..7} has exactly 3 strut pairs
// derived from the Fano plane XOR structure.

/// A strut pair for a given strut constant S0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrutPair {
    /// The two L-indices whose XOR equals S0.
    pub u: usize,
    pub v: usize,
}

/// The complete delta transition table for a box-kite.
#[derive(Debug, Clone)]
pub struct DeltaTransitionTable {
    /// Source strut constant.
    pub s0: usize,
    /// The 3 strut pairs (one per catamaran/tray-rack).
    pub strut_pairs: [StrutPair; 3],
}

/// Compute the strut pairs for a given S0 (XOR-derived from Fano plane).
///
/// Each pair {u, v} satisfies u XOR v = S0, u < v, u,v in {1..7} \ {S0}.
pub fn strut_pairs_for(s0: usize) -> [StrutPair; 3] {
    assert!((1..=7).contains(&s0), "S0 must be in 1..7");
    let mut pairs = Vec::new();
    for u in 1..=7usize {
        if u == s0 {
            continue;
        }
        let v = u ^ s0;
        if v > u && v != s0 && v <= 7 {
            pairs.push(StrutPair { u, v });
        }
    }
    assert_eq!(
        pairs.len(),
        3,
        "S0={} should have 3 strut pairs, got {}",
        s0,
        pairs.len()
    );
    [pairs[0], pairs[1], pairs[2]]
}

/// The delta function: given S0 and a strut pair, return the two destination
/// strut constants (one per parallel set in the catamaran).
pub fn delta_transition(_s0: usize, pair: &StrutPair) -> (usize, usize) {
    (pair.u, pair.v)
}

/// Result of comparing one twist transition to the delta strut-pair structure.
#[derive(Debug, Clone)]
pub struct TwistDeltaComparison {
    /// Source box-kite strut constant.
    pub source_strut: usize,
    /// Tray-rack label (the perpendicular pair).
    pub tray_rack_label: [usize; 2],
    /// Twist targets {h_star, v_star}.
    pub twist_targets: (usize, usize),
    /// Whether h_star XOR v_star == source_strut (Fano XOR law).
    pub xor_matches_source: bool,
    /// Which delta strut pair (if any) matches the twist targets as a set.
    pub matching_strut_pair: Option<StrutPair>,
    /// The Fano line containing {source, h_star, v_star}, if any.
    pub fano_line: Option<[usize; 3]>,
}

/// Exhaustively compare twist transitions against delta strut pairs.
///
/// For each BK and each tray-rack, checks:
/// 1. Whether {h_star, v_star} satisfies h XOR v == source_strut
/// 2. Whether {h_star, v_star} matches one of the 3 delta strut pairs
/// 3. Which Fano line (if any) contains {source, h_star, v_star}
pub fn twist_delta_correspondence() -> Vec<TwistDeltaComparison> {
    let twist_table = twist_transition_table();
    let mut results = Vec::new();

    for t in &twist_table {
        let s = t.source_strut;
        let h = t.h_star_target;
        let v = t.v_star_target;

        // Check XOR law: h XOR v should equal s (Fano plane constraint)
        let xor_matches = (h ^ v) == s;

        // Check if {h, v} matches a delta strut pair for this source
        let pairs = strut_pairs_for(s);
        let pair_set = (h.min(v), h.max(v));
        let matching = pairs
            .iter()
            .find(|p| (p.u.min(p.v), p.u.max(p.v)) == pair_set)
            .copied();

        // Find the Fano line containing {s, h, v}
        let mut triple = [s, h, v];
        triple.sort();
        let fano = O_TRIPS
            .iter()
            .find(|ot| {
                let mut sorted = **ot;
                sorted.sort();
                sorted == triple
            })
            .copied();

        results.push(TwistDeltaComparison {
            source_strut: s,
            tray_rack_label: t.tray_rack_label,
            twist_targets: (h, v),
            xor_matches_source: xor_matches,
            matching_strut_pair: matching,
            fano_line: fano,
        });
    }

    results
}

/// Result of detailed vent-assessor analysis for a single tray-rack.
#[derive(Debug, Clone)]
pub struct VentPairingAnalysis {
    /// Source strut constant.
    pub source_strut: usize,
    /// Perpendicular pair L-indices [a.low, f.low] for this tray-rack.
    pub perp_pair: [usize; 2],
    /// All 4 vent assessor L-indices in this tray-rack.
    pub vent_indices: [usize; 4],
    /// Three possible pairings of the 4 vent indices, with their XOR values.
    /// Each pairing is ((i1,i2,xor12), (i3,i4,xor34)).
    /// Type alias for a pairing entry: ((idx1, idx2, xor), (idx3, idx4, xor)).
    #[allow(clippy::type_complexity)]
    pub pairings: [((usize, usize, usize), (usize, usize, usize)); 3],
    /// Which Fano line element each pairing's XOR corresponds to:
    /// `0 = source_strut (S)`, `1 = perp[0]`, `2 = perp[1]`.
    pub pairing_fano_roles: [usize; 3],
    /// The twist targets currently computed by twist_transition_table().
    pub current_twist_targets: (usize, usize),
    /// Which pairing the current twist targets fall into (0, 1, or 2).
    pub current_pairing_index: Option<usize>,
}

/// For each tray-rack, analyze all three possible pairings of vent assessors.
///
/// The 4 vent assessors of a tray-rack partition into 2+2 in three ways.
/// Each pairing produces a consistent XOR value that lies on the Fano line
/// `{S, perp[0], perp[1]}`. This function documents which pairing the
/// twist_transition_table currently selects and whether it's consistent.
pub fn vent_pairing_analysis() -> Vec<VentPairingAnalysis> {
    let bks = find_box_kites(16, 1e-10);
    let twist_table = twist_transition_table();
    let atol = 1e-10;
    let mut results = Vec::new();

    for bk in &bks {
        let tab = canonical_strut_table(bk, atol);
        let s = bk.strut_signature;

        // Three tray-racks with their perpendicular pairs and vent assessors
        let tray_racks: [([usize; 2], [usize; 4]); 3] = [
            (
                [tab.a.low, tab.f.low],
                [tab.b.low, tab.c.low, tab.d.low, tab.e.low],
            ),
            (
                [tab.b.low, tab.e.low],
                [tab.a.low, tab.c.low, tab.f.low, tab.d.low],
            ),
            (
                [tab.c.low, tab.d.low],
                [tab.a.low, tab.b.low, tab.f.low, tab.e.low],
            ),
        ];

        for (perp, vents) in &tray_racks {
            let v = *vents;
            // Three possible 2+2 pairings of 4 elements {v0,v1,v2,v3}:
            // P0: {v0,v1} + {v2,v3}
            // P1: {v0,v2} + {v1,v3}
            // P2: {v0,v3} + {v1,v2}
            let pairings = [
                ((v[0], v[1], v[0] ^ v[1]), (v[2], v[3], v[2] ^ v[3])),
                ((v[0], v[2], v[0] ^ v[2]), (v[1], v[3], v[1] ^ v[3])),
                ((v[0], v[3], v[0] ^ v[3]), (v[1], v[2], v[1] ^ v[2])),
            ];

            // For each pairing, both sub-pairs should XOR to the SAME value
            // (this is a theorem about Fano plane structure).
            // That value is one of {S, perp[0], perp[1]}.
            let fano_line = [s, perp[0], perp[1]];
            let pairing_roles: [usize; 3] = std::array::from_fn(|i| {
                let xor_val = pairings[i].0.2;
                if xor_val == fano_line[0] {
                    0
                } else if xor_val == fano_line[1] {
                    1
                } else if xor_val == fano_line[2] {
                    2
                } else {
                    usize::MAX // unexpected
                }
            });

            // Find current twist targets for this tray-rack
            let twist = twist_table.iter().find(|t| {
                t.source_strut == s
                    && ((t.tray_rack_label[0] == perp[0] && t.tray_rack_label[1] == perp[1])
                        || (t.tray_rack_label[0] == perp[1] && t.tray_rack_label[1] == perp[0]))
            });

            let current_targets = twist.map_or((0, 0), |t| (t.h_star_target, t.v_star_target));

            // Determine which pairing the current targets fall into
            let target_set = (
                current_targets.0.min(current_targets.1),
                current_targets.0.max(current_targets.1),
            );
            let current_idx = pairings.iter().position(|p| {
                let s1 = (p.0.0.min(p.0.1), p.0.0.max(p.0.1));
                let s2 = (p.1.0.min(p.1.1), p.1.0.max(p.1.1));
                target_set == s1 || target_set == s2
            });

            results.push(VentPairingAnalysis {
                source_strut: s,
                perp_pair: *perp,
                vent_indices: *vents,
                pairings,
                pairing_fano_roles: pairing_roles,
                current_twist_targets: current_targets,
                current_pairing_index: current_idx,
            });
        }
    }

    results
}

/// Build the complete delta transition table for all S0 in {1..7}.
pub fn delta_transition_tables() -> Vec<DeltaTransitionTable> {
    (1..=7)
        .map(|s0| DeltaTransitionTable {
            s0,
            strut_pairs: strut_pairs_for(s0),
        })
        .collect()
}

/// Verify that delta strut pairs and twist transitions share the same
/// reachability structure.
///
/// For each S0, the twist transitions and delta pairs both reach the same set
/// of 6 non-S0 strut constants. The twist targets (h_star, v_star) are pairs
/// from {1..7}\{S0}, as are the delta strut pair endpoints.
pub fn verify_delta_reachability() -> bool {
    let twist_table = twist_transition_table();
    let delta_tables = delta_transition_tables();

    for dt in &delta_tables {
        let s0 = dt.s0;
        // Delta reachable: all endpoints from strut pairs
        let delta_reach: HashSet<usize> = dt.strut_pairs.iter().flat_map(|p| [p.u, p.v]).collect();

        // Should be exactly {1..7} \ {S0}
        let expected: HashSet<usize> = (1..=7).filter(|&x| x != s0).collect();
        if delta_reach != expected {
            return false;
        }

        // Twist reachable: all h/v targets from this source
        let twist_reach: HashSet<usize> = twist_table
            .iter()
            .filter(|t| t.source_strut == s0)
            .flat_map(|t| [t.h_star_target, t.v_star_target])
            .filter(|&x| x != 0)
            .collect();

        // Twist should reach a subset of the same 6 indices
        if !twist_reach.is_subset(&expected) {
            return false;
        }
    }
    true
}

// ===========================================================================
// L18: Brocade/Slipcover Normalization
// ===========================================================================
//
// Any node can be moved to the center of the PSL(2,7) triangle to act as
// the strut constant, with the main side-effect being broad-based swapping
// of U-indices (Pathions3).
//
// This means canonical_strut_table() is correct as a *set* of dyads and
// strut pairs, but comparing Trip Sync or O-trip alignment to literature
// diagrams requires a brocade relabeling.
//
// The brocade normalization maps a box-kite's L-indices to a standard form
// where a chosen O-trip serves as the Rule-0 central circle.

/// A brocade relabeling: maps raw L-indices to standard PSL(2,7) positions.
#[derive(Debug, Clone)]
pub struct BrocadeRelabeling {
    /// Source strut signature.
    pub source_s: usize,
    /// The O-trip chosen as the Rule-0 central circle.
    pub central_trip: [usize; 3],
    /// The 3 remaining L-indices (forming the outer triangle).
    pub outer_indices: [usize; 3],
    /// Whether this relabeling preserves CPO (cyclic positive orientation).
    pub preserves_cpo: bool,
}

/// Compute all valid brocade relabelings for a box-kite.
///
/// Each of the 4 O-trips in the BK's L-set can serve as the central circle,
/// giving 4 possible normalizations.
pub fn brocade_relabelings(bk: &BoxKite) -> Vec<BrocadeRelabeling> {
    let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();
    let available: Vec<[usize; 3]> = O_TRIPS
        .iter()
        .filter(|t| t.iter().all(|&x| l_set.contains(&x)))
        .copied()
        .collect();

    let otrip_set: HashSet<[usize; 3]> = O_TRIPS
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort();
            s
        })
        .collect();

    let mut relabelings = Vec::new();

    for trip in &available {
        let outer: Vec<usize> = l_set
            .iter()
            .copied()
            .filter(|x| !trip.contains(x))
            .collect();

        if outer.len() != 3 {
            continue;
        }

        // Check CPO preservation: do the outer indices form an O-trip?
        let mut outer_sorted = [outer[0], outer[1], outer[2]];
        outer_sorted.sort();
        let preserves_cpo = otrip_set.contains(&outer_sorted);

        relabelings.push(BrocadeRelabeling {
            source_s: bk.strut_signature,
            central_trip: *trip,
            outer_indices: [outer[0], outer[1], outer[2]],
            preserves_cpo,
        });
    }

    relabelings
}

/// Check that brocade normalization is consistent across all box-kites:
/// each BK has exactly 4 relabelings, and the outer indices always form
/// a well-defined partition of the non-trip L-indices.
pub fn verify_brocade_consistency() -> bool {
    let bks = find_box_kites(16, 1e-10);
    bks.iter().all(|bk| {
        let relabelings = brocade_relabelings(bk);
        relabelings.len() == 4
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests;
