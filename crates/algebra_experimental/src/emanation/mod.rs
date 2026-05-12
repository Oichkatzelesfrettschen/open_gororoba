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
    Assessor, BoxKite, EdgeSignType, FaceSignPattern, O_TRIPS, canonical_strut_table,
    classify_face_pattern, edge_sign_type, find_box_kites,
};
#[cfg(test)]
use algebra_analysis::boxkites::motif_components_for_cross_assessors;
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
