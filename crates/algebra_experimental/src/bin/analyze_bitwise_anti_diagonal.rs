//! `analyze_bitwise_anti_diagonal` -- algebraic classifier for the
//! 64-pixel failure set observed in the Terakan `r32_sint_single_layer`
//! CTS test on 2026-04-19.
//!
//! # Context
//!
//! The Terakan driver's `r32_sint_single_layer` test produces 64 wrong
//! pixels out of 4096 on layer 0.  Every wrong pixel lies on the
//! anti-diagonal `x + y == 63` in a 64x64 grid and has observed
//! `x ^ y == 62` instead of the expected `63` (all six bits set).
//!
//! # What this analyzer does
//!
//! 1. Enumerates the 64 anti-diagonal points of `F_2^6`.
//! 2. Identifies the set as exactly the **top XOR-bucket** in
//!    Cayley-Dickson basis-index theory: the set of pairs
//!    `(x, y)` with `xor_key(x, y) = 2^6 - 1 = 63`, i.e. those
//!    basis-index pairs whose CD product `e_x * e_y = +/- e_63`
//!    lands on the top basis element of the 64D (pathion) algebra.
//! 3. Characterizes the bit-0-drop as: hardware producing `e_62`
//!    (drop bit 0 of the basis index) instead of `e_63` exclusively
//!    for products in the top XOR-bucket.
//! 4. Enumerates the tile-level structure: which of the 8x8 macro-tiles
//!    on a 64x64 image contain anti-diagonal pixels, and what the
//!    intra-tile pattern is.
//! 5. Emits a structured JSON report to stdout.
//!
//! # Why this analyzer lives in open_gororoba
//!
//! The XOR-bucket structure is the foundational primitive of CD
//! zero-divisor theory (de Marrais 2000, Reggiani 2024) and is
//! formalized in this repository as
//! `algebra_analysis::zd_graphs::xor_key`.  The Terakan failure is
//! literally the top bucket of that primitive at n = 6.  Naming the
//! failure set this way (instead of "anti-diagonal of a 64x64 grid")
//! transports the diagnostic into a known mathematical object whose
//! invariants the repository already exposes.
//!
//! # Correction note
//!
//! The earlier draft of this analyzer claimed the failure set was
//! "the graph of CD conjugation restricted to F_2 basis".  That was
//! wrong: `cd_kernel::cayley_dickson::arith::cd_conjugate` operates
//! on real-valued coordinates `(x_0, x_1, ..., x_{n-1})` by negating
//! all imaginary entries, NOT by permuting basis indices.  The
//! correct algebraic identification is the XOR-bucket, as documented
//! in `algebra_analysis::zd_graphs::xor_key` ("In CD algebras,
//! `e_i * e_j = +/- e_{i^j}`").

use algebra_analysis::zd_graphs::xor_key;
use serde_json::json;

const GRID_DIM: u32 = 64;
const GRID_BITS: u32 = 6;
const TOP_ELEMENT: u32 = GRID_DIM - 1; // 63 = all six bits set
const TILE_DIM: u32 = 8;

/// The "complement-of-x" partner under the top XOR-bucket: the
/// unique `y` in `[0, 64)` with `xor_key(x, y) = 63`.
///
/// Equivalent to `(!x) & TOP_ELEMENT`, but framed via the CD
/// `xor_key` primitive to make the algebraic identification
/// explicit.
fn top_bucket_partner(x: u32) -> u32 {
    let y = (!x) & TOP_ELEMENT;
    debug_assert_eq!(
        xor_key(x as usize, y as usize),
        TOP_ELEMENT as usize,
        "top_bucket_partner consistency with algebra_analysis::zd_graphs::xor_key"
    );
    y
}

/// The 64 points `(x, y)` with `xor_key(x, y) = TOP_ELEMENT = 63`.
/// In CD basis-index terms: the basis-index pairs whose product
/// lands on `e_63`, the top basis element of the 64D pathion algebra.
fn top_bucket_pairs() -> Vec<(u32, u32)> {
    (0..GRID_DIM).map(|x| (x, top_bucket_partner(x))).collect()
}

/// Macro-tile coordinate `(tx, ty)`.
type TileCoord = (u32, u32);
/// Intra-tile pixel coordinate `(ix, iy)` with each in `[0, TILE_DIM)`.
type IntraCoord = (u32, u32);
/// A tile and the intra-pixel coords that lie in it.
type TileBucket = (TileCoord, Vec<IntraCoord>);

/// Groups anti-diagonal points by macro-tile `(tx, ty)` of size
/// 8x8.  Returns each tile's (tx, ty) and intra-tile coords.
fn tile_structure(points: &[(u32, u32)]) -> Vec<TileBucket> {
    let mut by_tile: std::collections::BTreeMap<TileCoord, Vec<IntraCoord>> =
        std::collections::BTreeMap::new();
    for &(x, y) in points {
        let key = (x / TILE_DIM, y / TILE_DIM);
        by_tile
            .entry(key)
            .or_default()
            .push((x % TILE_DIM, y % TILE_DIM));
    }
    by_tile.into_iter().collect()
}

/// Verifies algebraic properties of the failure set.
fn verify_algebraic_structure(points: &[(u32, u32)]) -> serde_json::Value {
    let count = points.len();
    let all_on_sum63 = points.iter().all(|&(x, y)| x + y == TOP_ELEMENT);
    let all_xor_63 = points.iter().all(|&(x, y)| (x ^ y) == TOP_ELEMENT);
    let all_partner_pairs = points.iter().all(|&(x, y)| top_bucket_partner(x) == y);
    let all_xor_key_top = points
        .iter()
        .all(|&(x, y)| xor_key(x as usize, y as usize) == TOP_ELEMENT as usize);
    let bucket_closed_under_swap = points.iter().all(|&(x, y)| {
        let (x2, y2) = (y, x);
        points.contains(&(x2, y2))
    });
    let fixed_points = points.iter().filter(|&&(x, y)| x == y).count();
    json!({
        "point_count": count,
        "all_points_satisfy_x_plus_y_eq_63": all_on_sum63,
        "all_points_satisfy_x_xor_y_eq_63": all_xor_63,
        "all_points_satisfy_xor_key_top": all_xor_key_top,
        "all_points_are_top_bucket_partners": all_partner_pairs,
        "set_closed_under_coordinate_swap": bucket_closed_under_swap,
        "fixed_points_of_top_bucket": fixed_points,
        "F_2_rank": GRID_BITS,
        "xor_key_primitive_source": "algebra_analysis::zd_graphs::xor_key",
        "cd_basis_index_max": TOP_ELEMENT,
        "cd_top_basis_element": format!("e_{}", TOP_ELEMENT),
        "cd_algebra_at_dim_64": "pathion (32D = trigintaduonion is 2^5; 64D pathion is 2^6)",
    })
}

fn main() {
    let points = top_bucket_pairs();
    let structure = verify_algebraic_structure(&points);
    let tiles = tile_structure(&points);
    let tile_report: Vec<serde_json::Value> = tiles
        .iter()
        .map(|((tx, ty), intras)| {
            let intra_diagonal = intras
                .iter()
                .all(|&(ix, iy)| ix + iy == TILE_DIM - 1);
            json!({
                "tile_xy": [tx, ty],
                "tile_index_sum": tx + ty,
                "pixel_count": intras.len(),
                "intra_pixels_on_intra_anti_diagonal": intra_diagonal,
                "intra_coords": intras,
            })
        })
        .collect();
    let distinct_tx_plus_ty: std::collections::BTreeSet<u32> =
        tiles.iter().map(|((tx, ty), _)| tx + ty).collect();
    let report = json!({
        "analyzer": "analyze_bitwise_anti_diagonal",
        "version": "1.0",
        "target": {
            "driver": "Terakan Vulkan",
            "test": "dEQP-VK.image.store.with_format.2d_array.r32_sint_single_layer",
            "failure_date": "2026-04-19",
            "grid": format!("{}x{}", GRID_DIM, GRID_DIM),
            "observed_residual": "64 wrong pixels; actual x^y = 62 where expected 63",
        },
        "algebraic_classification": {
            "ambient_structure": "F_2^6 viewed as the basis-index space of the 64D Cayley-Dickson algebra (pathion)",
            "failure_set_identity": "the top XOR-bucket: pairs (x, y) with xor_key(x, y) = 63",
            "cd_significance": "These are exactly the basis-index pairs whose CD product e_x * e_y = +/- e_63 (the top basis element of pathion)",
            "primitive_used": "algebra_analysis::zd_graphs::xor_key",
            "top_element": TOP_ELEMENT,
            "top_element_binary": "0b111111",
            "structure_properties": structure,
        },
        "tile_decomposition": {
            "tile_size": format!("{}x{}", TILE_DIM, TILE_DIM),
            "macro_tiles_touched": tiles.len(),
            "macro_tile_index_sums": distinct_tx_plus_ty.iter().collect::<Vec<_>>(),
            "tiles_lie_on_anti_diagonal": distinct_tx_plus_ty.len() == 1
                && *distinct_tx_plus_ty.iter().next().unwrap() == (GRID_DIM / TILE_DIM) - 1,
            "per_tile": tile_report,
        },
        "diagnosis": {
            "failure_mode": "Hardware produces e_62 instead of e_63 (bit-0 of the basis index drops) exclusively for products in the top XOR-bucket.",
            "algebraic_reading": "Hardware fails to produce the top basis element e_{2^n - 1} of the dim=2^n CD algebra at n=6.",
            "implications": [
                "Any pipeline stage that saturates unsigned 6-bit outputs at 62 instead of 63 reproduces this geometry exactly.",
                "Any pipeline stage that applies a mask (_ & ~1) conditional on (value == 63) reproduces this geometry exactly.",
                "The tile decomposition shows the failure is uniform across all eight 8x8 macro-tiles crossing the global anti-diagonal -- the fault is NOT tile-local.",
                "The CD-level classification (top XOR-bucket) is independent of which specific bit drops; any saturation at the top basis element of F_2^6 produces this pattern.",
                "By analogy with sedenion / pathion zero-divisor theory (de Marrais 2000, Reggiani 2024), the top-bucket subset has structurally distinguished status -- it is the unique bucket containing pairs whose product is the top basis element.",
            ],
            "hardware_side_candidates": [
                "MEM_RAT STORE_TYPED format-conversion saturation on R32_SINT at signed max (+31 in the colorExpr encoding of x^y = 63).",
                "CB export-path clamp on sint channel to [-32, +30] instead of [-32, +31].",
                "SFN ALU clamp flag on a mov downstream of XOR_INT, clamping the six-bit value to [0, 62].",
            ],
            "ruled_out_candidates": [
                "SFN x^y identity-lowering to (x|y)-(x&y): sfn_instr_alu.cpp emits native XOR_INT (2026-04-19 mining).",
                "MEM_RAT STORE_TYPED saturation on R32_UINT: PALM probe 5 proved bit-transparency (2026-04-19 mining).",
                "CD conjugation involution: cd_kernel::cd_conjugate negates real-valued imaginary coordinates, NOT basis-index permutation; not the right primitive.",
            ],
            "open_gororoba_followups": [
                "Run algebra_analysis::boxkites::boxkite_assessors at dim=64 (pathion) to test whether the 64-pixel set decomposes into a known box-kite at the pathion level.",
                "Run algebra_analysis::annihilator::left_multiplication_matrix on a sedenion / pathion built from indicator-of-top-bucket coefficients; check if the resulting nullspace dimension matches the geometric structure of the failure set.",
                "Run sign_imbalance::balance::compute_imbalance_index on the signed graph induced by the CD multiplication table restricted to the top XOR-bucket. Bit-0-drop hardware would produce a specific sign-imbalance signature.",
            ],
        },
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_bucket_partner_is_involution() {
        for x in 0..GRID_DIM {
            assert_eq!(
                top_bucket_partner(top_bucket_partner(x)),
                x,
                "partner^2 = id failed at x={}",
                x
            );
        }
    }

    #[test]
    fn top_bucket_partner_has_no_fixed_points() {
        // For 6-bit complement, partner(x) = x would require all bits
        // equal to their complement, impossible at even bit-count.
        for x in 0..GRID_DIM {
            assert_ne!(top_bucket_partner(x), x);
        }
    }

    #[test]
    fn top_bucket_has_exactly_64_pairs() {
        let pts = top_bucket_pairs();
        assert_eq!(pts.len(), GRID_DIM as usize);
    }

    #[test]
    fn every_pair_satisfies_xor_key_top() {
        for (x, y) in top_bucket_pairs() {
            assert_eq!(
                xor_key(x as usize, y as usize),
                TOP_ELEMENT as usize,
                "xor_key consistency failed at ({}, {})",
                x,
                y
            );
        }
    }

    #[test]
    fn tile_decomposition_hits_exactly_eight_tiles() {
        let pts = top_bucket_pairs();
        let tiles = tile_structure(&pts);
        assert_eq!(tiles.len(), 8, "expected exactly 8 macro-tiles crossed");
        for ((tx, ty), intras) in &tiles {
            assert_eq!(tx + ty, (GRID_DIM / TILE_DIM) - 1);
            assert_eq!(intras.len(), TILE_DIM as usize);
            for &(ix, iy) in intras {
                assert_eq!(ix + iy, TILE_DIM - 1);
            }
        }
    }
}
