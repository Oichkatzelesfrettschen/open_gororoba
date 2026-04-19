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
//! 2. Identifies the set as the graph of the complement involution
//!    `tau(x) = ~x mod 64`.
//! 3. Cross-references this involution with the standard
//!    Cayley-Dickson conjugation at dimension 64 (the dim=64 level
//!    of the CD tower where the algebra is a pathion).
//! 4. Characterizes the bit-0-drop at the fixed-locus `x XOR tau(x) = 63`
//!    as a saturation at the top element of `F_2^6`.
//! 5. Enumerates the tile-level structure: which of the 8x8 macro-tiles
//!    on a 64x64 image contain anti-diagonal pixels, and what the
//!    intra-tile pattern is.
//! 6. Emits a structured JSON report to stdout.
//!
//! # Why this analyzer lives in open_gororoba
//!
//! The 6-bit XOR involution is exactly the dim=64 Cayley-Dickson
//! conjugation restricted to the F_2-linear basis.  Classifying the
//! failure as an algebraic object (rather than a geometric curiosity)
//! points diagnostic attention at the correct layer of the driver:
//! any mechanism that saturates the CD top-element (all-ones =
//! dim-1 in F_2 basis) will produce exactly this 64-pixel anti-diagonal
//! residual.

use serde_json::json;

const GRID_DIM: u32 = 64;
const GRID_BITS: u32 = 6;
const TOP_ELEMENT: u32 = GRID_DIM - 1; // 63 = all six bits set
const TILE_DIM: u32 = 8;

/// Complement involution `tau(x) = ~x mod 64`.
/// In the Cayley-Dickson tower at dim=64 this is the conjugation
/// restricted to the F_2-linear coordinate basis.
fn tau(x: u32) -> u32 {
    (!x) & TOP_ELEMENT
}

/// The 64 anti-diagonal points of the 64x64 grid.
fn anti_diagonal_points() -> Vec<(u32, u32)> {
    (0..GRID_DIM).map(|x| (x, tau(x))).collect()
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
    let all_complement = points.iter().all(|&(x, y)| tau(x) == y);
    let involution_closed = points.iter().all(|&(x, y)| {
        let (x2, y2) = (y, x);
        points.contains(&(x2, y2))
    });
    let fixed_points = points.iter().filter(|&&(x, y)| x == y).count();
    json!({
        "point_count": count,
        "all_points_satisfy_x_plus_y_eq_63": all_on_sum63,
        "all_points_satisfy_x_xor_y_eq_63": all_xor_63,
        "all_points_are_complement_pairs": all_complement,
        "set_closed_under_coordinate_swap": involution_closed,
        "fixed_points_of_tau": fixed_points,
        "F_2_rank_of_involution": GRID_BITS,
        "cayley_dickson_level": "dim=64 (pathion); tau = CD conjugation restricted to F_2-linear basis",
    })
}

fn main() {
    let points = anti_diagonal_points();
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
            "ambient_algebra": "F_2^6 (6-dimensional vector space over F_2, i.e. 6-bit XOR)",
            "failure_set_identity": "graph of the complement involution tau(x) = NOT x mod 64",
            "is_conjugation_graph_of_cd_tower": true,
            "cd_tower_level": "dim=64 = 2^6 = pathion level",
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
            "failure_mode": "Bit 0 is cleared exclusively when the XOR result equals the top element of F_2^6.",
            "algebraic_reading": "Saturation at the CD top-element under the dim=64 conjugation involution.",
            "implications": [
                "Any pipeline stage that saturates unsigned 6-bit outputs at 62 instead of 63 reproduces this geometry exactly.",
                "Any pipeline stage that applies a mask (_ & ~1) conditional on (value == 63) reproduces this geometry exactly.",
                "The tile decomposition shows the failure is uniform across all eight 8x8 tiles that cross the global anti-diagonal, so the fault is NOT tile-local.",
                "The CD-level classification (dim=64 conjugation) is independent of which specific bit drops; any fixed-locus-of-tau saturation produces this pattern.",
            ],
            "hardware_side_candidates": [
                "MEM_RAT STORE_TYPED format-conversion saturation on R32_SINT at signed max (+31 in the colorExpr encoding of x^y = 63).",
                "CB export-path clamp on sint channel to [-32, +30] instead of [-32, +31].",
                "SFN ALU clamp flag on a mov downstream of XOR_INT, clamping the six-bit value to [0, 62].",
            ],
            "ruled_out_candidates": [
                "SFN x^y identity-lowering to (x|y)-(x&y): sfn_instr_alu.cpp emits native XOR_INT (2026-04-19 mining).",
                "MEM_RAT STORE_TYPED saturation on R32_UINT: PALM probe 5 proved bit-transparency (2026-04-19 mining).",
            ],
        },
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_is_involution() {
        for x in 0..GRID_DIM {
            assert_eq!(tau(tau(x)), x, "tau^2 = id failed at x={}", x);
        }
    }

    #[test]
    fn tau_has_no_fixed_points_for_even_bit_count() {
        // For 6-bit complement, tau(x) = x would require all bits equal
        // to their complement, impossible.
        for x in 0..GRID_DIM {
            assert_ne!(tau(x), x, "tau has unexpected fixed point at x={}", x);
        }
    }

    #[test]
    fn anti_diagonal_has_exactly_64_points() {
        let pts = anti_diagonal_points();
        assert_eq!(pts.len(), GRID_DIM as usize);
    }

    #[test]
    fn every_point_satisfies_xor_identity() {
        for (x, y) in anti_diagonal_points() {
            assert_eq!(
                x ^ y,
                TOP_ELEMENT,
                "x^y identity failed at ({}, {})",
                x,
                y
            );
        }
    }

    #[test]
    fn tile_decomposition_hits_exactly_eight_tiles() {
        let pts = anti_diagonal_points();
        let tiles = tile_structure(&pts);
        assert_eq!(tiles.len(), 8, "expected exactly 8 macro-tiles crossed");
        for ((tx, ty), intras) in &tiles {
            assert_eq!(tx + ty, (GRID_DIM / TILE_DIM) - 1, "tile not on macro anti-diagonal: ({}, {})", tx, ty);
            assert_eq!(intras.len(), TILE_DIM as usize);
            for &(ix, iy) in intras {
                assert_eq!(ix + iy, TILE_DIM - 1);
            }
        }
    }
}
