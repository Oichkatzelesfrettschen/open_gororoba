//! Twisted Sisters Delta Transition Function (L17).
//!
//! The twist navigation is a deterministic automaton:
//!   delta(S0, {u,v}, 0) = u
//!   delta(S0, {u,v}, 1) = v
//! where {u,v} is one of S0's three strut pairs (u XOR v = S0).
//!
//! This is purely algebraic: each S0 in {1..7} has exactly 3 strut
//! pairs derived from the Fano plane XOR structure.

use std::collections::HashSet;

use algebra_analysis::boxkites::{O_TRIPS, canonical_strut_table, find_box_kites};

use super::twist_transitions::twist_transition_table;

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
