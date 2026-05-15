//! ET <-> Edge-Sign <-> Lanyard Dictionary (L16).
//!
//! The strutted ET is a signed adjacency matrix: each DMZ cell encodes
//! an edge sign (+1 or -1) between two assessors.
//!
//! Edge sign determines diagonal-state coupling:
//!   +1 (same-slope): preserves /\ state across edge
//!   -1 (cross-slope): flips /\ state across edge
//!
//! Lanyards emerge as state-machine traversals of the signed graph:
//!   Zigzag: all 3 edges negative -> /\/\/\ (alternating, double cover)
//!   Trefoil: 2 positive + 1 negative -> ///\\\ (double cover)
//!   Catamaran: alternating signs -> two disjoint single-cover cycles

use std::collections::HashMap;

use algebra_analysis::boxkites::{
    Assessor, EdgeSignType, FaceSignPattern, classify_face_pattern, edge_sign_type, find_box_kites,
};

use super::{
    sail_loop::boxkite_faces,
    strutted_et::{StruttedEmanationTable, create_strutted_et},
};

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
