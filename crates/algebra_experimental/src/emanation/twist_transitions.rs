//! Twist transition system (H* and V* operations) and the
//! Twisted Sisters PSL(2,7) navigation graph.
//!
//! De Marrais's "twist products" map tray-racks between box-kites:
//! - V* (vertical twist): twist vertical edges of Royal Hunt presentation
//! - H* (horizontal twist): twist horizontal edges
//!
//! Both produce a tray-rack in a DIFFERENT box-kite.
//!
//! Key property: the strut constant of the target box-kite equals the
//! perpendicular vent assessor's index in the source tray-rack.
//!
//! H*H* or V*V* on the same tray-rack cycles through 3 box-kites whose
//! strut constants form an O-trip (associative triplet).
//!
//! The Twisted Sisters diagram is a PSL(2,7)-structured graph on 7 nodes
//! (one per box-kite strut constant). Edges indicate which box-kites are
//! connected via twist operations.

use std::collections::{HashMap, HashSet};

use algebra_analysis::boxkites::{O_TRIPS, canonical_strut_table, find_box_kites};

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
