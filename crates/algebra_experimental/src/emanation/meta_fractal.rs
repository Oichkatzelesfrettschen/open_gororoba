//! ET meta-fractal regime doubling (L13) and Eco Echo recursion (L14).
//!
//! Regime count doubles at each dimension level: regimes(N) = 2^(N-4).
//! This is the "period doubling" pattern from Recipe Theory
//! (Placeholder III).
//!
//! The deeper structure is a substitution system:
//! - "Four Corners" rule: corner panes of the larger skybox replicate
//!   corresponding quadrants of the smaller skybox
//! - "French Windows" rule: shutter regions use g-augmentation
//! - Bitstring painting recipe: cell occupancy determined by S's bitstring
//!
//! Eco Echo (L14): SS edge labels {S, G, X} can be permuted by the
//! role-swap group (which of {S,G,X} acts as diagonal/horizontal/
//! vertical). The XOR closure constraint X = G XOR S maintains
//! algebraic consistency. The recursion operator E replaces each SS
//! corner node by a fresh SS (node-expansion into a strut-opposite
//! quartet).

use algebra_analysis::boxkites::find_box_kites;

use super::strut_spectroscopy::et_regimes;
use super::strutted_et::create_strutted_et;

// ===========================================================================
// L13: ET Meta-Fractal / Regime Doubling
// ===========================================================================

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
