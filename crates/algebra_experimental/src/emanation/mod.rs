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

#[cfg(test)]
use algebra_analysis::boxkites::{
    Assessor, EdgeSignType, FaceSignPattern, O_TRIPS, edge_sign_type, find_box_kites,
    motif_components_for_cross_assessors,
};
#[cfg(test)]
use cd_kernel::cayley_dickson::cd_basis_mul_sign;
#[cfg(test)]
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

// Twisted Sisters Delta Transition Function (L17) lives in the
// `delta_transition` submodule. The function `delta_transition` shares
// its name with the module; Rust's separate type/value namespaces allow
// both re-exports.
pub mod delta_transition;
pub use delta_transition::{
    DeltaTransitionTable, StrutPair, TwistDeltaComparison, VentPairingAnalysis,
    delta_transition_tables, strut_pairs_for, twist_delta_correspondence, vent_pairing_analysis,
    verify_delta_reachability,
};
pub use delta_transition::delta_transition;

// Brocade / Slipcover normalization (L18 PSL(2,7) relabeling) lives in
// the `brocade` submodule.
pub mod brocade;
pub use brocade::{BrocadeRelabeling, brocade_relabelings, verify_brocade_consistency};


// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests;
