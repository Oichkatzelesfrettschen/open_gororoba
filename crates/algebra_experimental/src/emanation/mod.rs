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

use algebra_analysis::boxkites::{BoxKite, O_TRIPS, canonical_strut_table, find_box_kites};
#[cfg(test)]
use algebra_analysis::boxkites::{
    Assessor, EdgeSignType, FaceSignPattern, edge_sign_type, motif_components_for_cross_assessors,
};
#[cfg(test)]
use cd_kernel::cayley_dickson::cd_basis_mul_sign;
use std::collections::HashSet;
#[cfg(test)]
use std::collections::HashMap;

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

// Lanyard (cycle) classification (LanyardType, classify_lanyard,
// lanyard_census_dim16) lives in the `lanyard` submodule.
pub mod lanyard;
pub use lanyard::{LanyardType, classify_lanyard, lanyard_census_dim16};

// Greimas semiotic square mapping (StrutLinkType, SemioticSquare,
// map_boxkite_to_semiotic, verify_semiotic_completeness) lives in
// the `semiotic` submodule.
pub mod semiotic;
pub use semiotic::{SemioticSquare, StrutLinkType, map_boxkite_to_semiotic, verify_semiotic_completeness};

// Twist transition system (H*/V* operations) and the Twisted Sisters
// PSL(2,7) navigation graph (TwistTransition, twist_transition_table,
// verify_twist_otrip_cycles, TwistedSisterEdge, twisted_sisters_graph,
// twisted_sisters_degree_sequence) live in the `twist_transitions`
// submodule.
pub mod twist_transitions;
pub use twist_transitions::{
    TwistTransition, TwistedSisterEdge, twist_transition_table, twisted_sisters_degree_sequence,
    twisted_sisters_graph, verify_twist_otrip_cycles,
};

// Extended lanyard taxonomy (ExtendedLanyardType, classify_face_extended,
// extended_lanyard_census_dim16) lives in the `extended_lanyard` submodule.
pub mod extended_lanyard;
pub use extended_lanyard::{ExtendedLanyardType, classify_face_extended, extended_lanyard_census_dim16};

// Trip Sync property and Quaternion Copy decomposition
// (QuaternionCopy, sail_quaternion_copies, verify_trip_sync) live
// in the `trip_sync` submodule.
pub mod trip_sync;
pub use trip_sync::{QuaternionCopy, sail_quaternion_copies, verify_trip_sync};

// Semiotic Square algebraic kernel (SsKernelResult, SsKernelCheck,
// verify_ss_algebraic_kernel) lives in the `ss_kernel` submodule.
pub mod ss_kernel;
pub use ss_kernel::{SsKernelCheck, SsKernelResult, verify_ss_algebraic_kernel};

// Fano-plane / projective-geometry derived structures: loop-box-kite
// duality, Hjelmslev net wrapping PG(n-2,2), and chingon spectral
// census (loop_boxkite_pairs, psl27_order, HjelmslevNet, hjelmslev_net,
// SpectralFingerprint, spectral_census) live in the `fano_duality`
// submodule.
pub mod fano_duality;
pub use fano_duality::{
    HjelmslevNet, SpectralFingerprint, hjelmslev_net, loop_boxkite_pairs, psl27_order,
    spectral_census,
};

// Open-research probes (extract_rho_matrix for the C-466 rho(b)
// multiplication coupling; octonion_subalgebra_constraint_check for
// item 12) live in the `open_research` submodule.
pub mod open_research;
pub use open_research::{extract_rho_matrix, octonion_subalgebra_constraint_check};

// CDP signed-product engine (L1: de Marrais's M(LI, RI) translated
// from LotusScript) lives in the `cdp` submodule.
pub mod cdp;
pub use cdp::cdp_signed_product;
#[cfg(test)]
pub(crate) use cdp::{QSIGNS, bit_length};

// Strutted Emanation Table: tone row generation (L2) and the DMZ
// X-pattern test (L3). Public types and the create entry point live
// in the `strutted_et` submodule.
pub mod strutted_et;
pub use strutted_et::{
    StruttedEmanationTable, StruttedEtCell, ToneRow, create_strutted_et, generate_tone_row,
};

// ET sparsity spectroscopy (L4) and strut-class spectroscopy (L4b)
// live in the `strut_spectroscopy` submodule.
pub mod strut_spectroscopy;
pub use strut_spectroscopy::{
    StrutClass, StrutSpectroscopyEntry, StrutSpectrum, classify_strut, et_regimes,
    et_sparsity_spectroscopy, is_inherited_full_fill_strut, is_sky_strut, strut_spectroscopy,
    trip_count, trip_count_two_step,
};

// (s,g)-modularity recursive regime address (L9b) and the hide/fill
// row-degree-invariance analysis (L9c) live in the `regime_address`
// submodule. The `regime_address` function shares its name with the
// module; Rust's separate namespaces for modules and values allow both
// to be re-exported without conflict.
pub mod regime_address;
pub use regime_address::{
    HideFillResult, RowDegreeDistribution, hide_fill_analysis, regime_count,
    row_degree_distribution,
};
pub use regime_address::regime_address;

// Skybox label-line extension (L9d) for the doubling recursion lives
// in the `skybox` submodule.
pub mod skybox;
pub use skybox::{Skybox, SkyboxCell, create_skybox};

// Theorem 11 recursive ET embedding (L9e: de Marrais 2004 "The 42
// Assessors") lives in the `theorem11` submodule.
pub mod theorem11;
pub use theorem11::{Theorem11Result, verify_theorem11};

// Balloon ride (L9f) -- fixed-S, increasing-N ET sequence -- lives in
// the `balloon_ride` submodule. The function `balloon_ride` shares its
// name with the module; Rust's separate type/value namespaces allow
// both re-exports.
pub mod balloon_ride;
pub use balloon_ride::{BalloonRide, BalloonRideStep, min_level_for_strut};
pub use balloon_ride::balloon_ride;

// Spectroscopy bands (L9g) -- fixed-N, all-S band structure -- live
// in the `spectroscopy_bands` submodule. The function
// `spectroscopy_bands` shares its name with the module; Rust's
// separate type/value namespaces allow both re-exports.
pub mod spectroscopy_bands;
pub use spectroscopy_bands::{
    BandBehavior, FlipBookFrame, SpectroscopyBand, SpectroscopyResult,
};
pub use spectroscopy_bands::spectroscopy_bands;

// CT Boundary / A7 Star (L10: twist as double transfer) lives in
// the `ct_boundary` submodule.
pub mod ct_boundary;
pub use ct_boundary::{CtBoundaryResult, ct_boundary_analysis, verify_double_transfer};

// Sail-to-loop duality via automorpheme membership (L11) lives in the
// `sail_loop` submodule.
pub mod sail_loop;
pub use sail_loop::{SailLabel, SailLoopResult, sail_loop_partition};

// Quincunx + Bicycle Chain explicit assessor paths (L12) live in the
// `quincunx` submodule.
pub mod quincunx;
pub use quincunx::{
    BicycleChain, QuincunxPath, QuincunxType, bicycle_chain, enumerate_quincunx_paths,
    quincunx_string_count,
};

// ET meta-fractal regime doubling (L13) and Eco Echo recursion (L14)
// live in the `meta_fractal` submodule.
pub mod meta_fractal;
pub use meta_fractal::{
    EcoEchoResult, RegimeDoublingResult, SsRoleAssignment, eco_echo_probe, verify_four_corners,
    verify_regime_doubling,
};

// Representation-aware Trip Sync (L15) and full sail decomposition
// (L15b) live in the `sail_classification` submodule.
pub mod sail_classification;
pub use sail_classification::{
    ClassifiedFace, FaceRole, OrientedTripSync, SailDecomposition, oriented_trip_sync,
    sail_decomposition,
};

// Three Vizier Relationships (L15b, de Marrais 2007) live in the
// `three_viziers` submodule.
pub mod three_viziers;
pub use three_viziers::{
    ThreeVizierResult, VizierXorAudit, verify_three_viziers, vizier_xor_audit,
};

// ET <-> Edge-Sign <-> Lanyard Dictionary (L16) lives in the
// `lanyard_dictionary` submodule.
pub mod lanyard_dictionary;
pub use lanyard_dictionary::{
    CrossBkLanyardCensus, FaceClassification, LanyardSignature, SignedAdjacencyGraph, SignedEdge,
    cross_bk_lanyard_census, extract_lanyards_from_et, extract_signed_graph, traverse_lanyard,
};


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
