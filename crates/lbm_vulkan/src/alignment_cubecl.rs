// Copyright (c) 2026 Terascale Functionalists
// SPDX-License-Identifier: GPL-2.0-or-later

//! cubecl-wgpu backend for box-kite alignment scanning.
//!
//! Each input sedenion (16-component f32) is scored against every orientation
//! in a PSL(2,7) permutation table; for each (vector, orientation) pair the
//! maximum projection weight across all 7 box-kite assessor subspaces is
//! stored.  A host-side argmax pass then finds the best orientation per vector.
//!
//! Input layout (all flat arrays):
//!   vectors:      f32[n_vectors * 16]   -- sedenion components, row-major
//!   orientations: u32[n_orientations * 16]   -- permutation table
//!   bk_basis:     `u32[84]`  -- 7 box-kites * 12 unique basis indices each
//! Intermediate (GPU):
//!   out_scores:   f32[n_vectors * n_orientations]  -- max bk weight per pair
//! Output (host argmax):
//!   (max_align, best_orient): (`Vec<f32>`, `Vec<u32>`) per vector
//!
//! Kernel design: one thread per (vector, orientation) pair.  The 7 * 12 = 84
//! basis reads and the 16 norm components are unrolled as immutable let bindings
//! to remain compatible with cubecl 0.10's expand IR, which does not support
//! mutable f32 accumulator variables inside #[cube] bodies.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

// ---- kernel -----------------------------------------------------------------

/// Compute the max box-kite projection weight for a single (vector, orientation)
/// pair.  One thread per pair; layout is `pair_idx = v_idx * n_orientations + o_idx`.
///
/// All intermediate f32 values are immutable let bindings so the expand IR
/// does not need to track phi-nodes for f32 mutable variables.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)] // cubecl launch ABI: one kernel arg per GPU buffer/scalar; no struct packing under #[cube]
pub fn box_kite_score_kernel(
    vectors: &Array<f32>,
    orientations: &Array<u32>,
    bk_basis: &Array<u32>,
    out_scores: &mut Array<f32>,
    #[comptime] n_pairs: u32,
    #[comptime] n_orientations: u32,
    #[comptime] n_vectors: u32,
) {
    let _ = n_vectors; // used by host only; declared comptime for documentation
    let pair_idx = ABSOLUTE_POS;
    if pair_idx >= n_pairs as usize {
        terminate!();
    }

    let v_idx = pair_idx / n_orientations as usize;
    let o_idx = pair_idx % n_orientations as usize;

    let v_off = v_idx * 16usize;
    let o_off = o_idx * 16usize;

    // Norm-squared: unroll 16 components (avoids mutable f32 accumulator).
    let c0 = vectors[v_off];
    let c1 = vectors[v_off + 1usize];
    let c2 = vectors[v_off + 2usize];
    let c3 = vectors[v_off + 3usize];
    let c4 = vectors[v_off + 4usize];
    let c5 = vectors[v_off + 5usize];
    let c6 = vectors[v_off + 6usize];
    let c7 = vectors[v_off + 7usize];
    let c8 = vectors[v_off + 8usize];
    let c9 = vectors[v_off + 9usize];
    let c10 = vectors[v_off + 10usize];
    let c11 = vectors[v_off + 11usize];
    let c12 = vectors[v_off + 12usize];
    let c13 = vectors[v_off + 13usize];
    let c14 = vectors[v_off + 14usize];
    let c15 = vectors[v_off + 15usize];
    let norm_sq = c0 * c0
        + c1 * c1
        + c2 * c2
        + c3 * c3
        + c4 * c4
        + c5 * c5
        + c6 * c6
        + c7 * c7
        + c8 * c8
        + c9 * c9
        + c10 * c10
        + c11 * c11
        + c12 * c12
        + c13 * c13
        + c14 * c14
        + c15 * c15;

    // Tiny epsilon avoids div-by-zero for zero vectors;
    // their score is ~0 regardless of orientation.
    let inv_norm_sq = 1.0_f32 / (norm_sq + 1e-30_f32);

    // Box-kite 0 (basis indices bk_basis[0..12]).
    let k0b0 = bk_basis[0usize] as usize;
    let k0b1 = bk_basis[1usize] as usize;
    let k0b2 = bk_basis[2usize] as usize;
    let k0b3 = bk_basis[3usize] as usize;
    let k0b4 = bk_basis[4usize] as usize;
    let k0b5 = bk_basis[5usize] as usize;
    let k0b6 = bk_basis[6usize] as usize;
    let k0b7 = bk_basis[7usize] as usize;
    let k0b8 = bk_basis[8usize] as usize;
    let k0b9 = bk_basis[9usize] as usize;
    let k0b10 = bk_basis[10usize] as usize;
    let k0b11 = bk_basis[11usize] as usize;
    let k0p0 = orientations[o_off + k0b0] as usize;
    let k0p1 = orientations[o_off + k0b1] as usize;
    let k0p2 = orientations[o_off + k0b2] as usize;
    let k0p3 = orientations[o_off + k0b3] as usize;
    let k0p4 = orientations[o_off + k0b4] as usize;
    let k0p5 = orientations[o_off + k0b5] as usize;
    let k0p6 = orientations[o_off + k0b6] as usize;
    let k0p7 = orientations[o_off + k0b7] as usize;
    let k0p8 = orientations[o_off + k0b8] as usize;
    let k0p9 = orientations[o_off + k0b9] as usize;
    let k0p10 = orientations[o_off + k0b10] as usize;
    let k0p11 = orientations[o_off + k0b11] as usize;
    let k0w = vectors[v_off + k0p0] * vectors[v_off + k0p0]
        + vectors[v_off + k0p1] * vectors[v_off + k0p1]
        + vectors[v_off + k0p2] * vectors[v_off + k0p2]
        + vectors[v_off + k0p3] * vectors[v_off + k0p3]
        + vectors[v_off + k0p4] * vectors[v_off + k0p4]
        + vectors[v_off + k0p5] * vectors[v_off + k0p5]
        + vectors[v_off + k0p6] * vectors[v_off + k0p6]
        + vectors[v_off + k0p7] * vectors[v_off + k0p7]
        + vectors[v_off + k0p8] * vectors[v_off + k0p8]
        + vectors[v_off + k0p9] * vectors[v_off + k0p9]
        + vectors[v_off + k0p10] * vectors[v_off + k0p10]
        + vectors[v_off + k0p11] * vectors[v_off + k0p11];

    // Box-kite 1 (basis indices bk_basis[12..24]).
    let k1b0 = bk_basis[12usize] as usize;
    let k1b1 = bk_basis[13usize] as usize;
    let k1b2 = bk_basis[14usize] as usize;
    let k1b3 = bk_basis[15usize] as usize;
    let k1b4 = bk_basis[16usize] as usize;
    let k1b5 = bk_basis[17usize] as usize;
    let k1b6 = bk_basis[18usize] as usize;
    let k1b7 = bk_basis[19usize] as usize;
    let k1b8 = bk_basis[20usize] as usize;
    let k1b9 = bk_basis[21usize] as usize;
    let k1b10 = bk_basis[22usize] as usize;
    let k1b11 = bk_basis[23usize] as usize;
    let k1p0 = orientations[o_off + k1b0] as usize;
    let k1p1 = orientations[o_off + k1b1] as usize;
    let k1p2 = orientations[o_off + k1b2] as usize;
    let k1p3 = orientations[o_off + k1b3] as usize;
    let k1p4 = orientations[o_off + k1b4] as usize;
    let k1p5 = orientations[o_off + k1b5] as usize;
    let k1p6 = orientations[o_off + k1b6] as usize;
    let k1p7 = orientations[o_off + k1b7] as usize;
    let k1p8 = orientations[o_off + k1b8] as usize;
    let k1p9 = orientations[o_off + k1b9] as usize;
    let k1p10 = orientations[o_off + k1b10] as usize;
    let k1p11 = orientations[o_off + k1b11] as usize;
    let k1w = vectors[v_off + k1p0] * vectors[v_off + k1p0]
        + vectors[v_off + k1p1] * vectors[v_off + k1p1]
        + vectors[v_off + k1p2] * vectors[v_off + k1p2]
        + vectors[v_off + k1p3] * vectors[v_off + k1p3]
        + vectors[v_off + k1p4] * vectors[v_off + k1p4]
        + vectors[v_off + k1p5] * vectors[v_off + k1p5]
        + vectors[v_off + k1p6] * vectors[v_off + k1p6]
        + vectors[v_off + k1p7] * vectors[v_off + k1p7]
        + vectors[v_off + k1p8] * vectors[v_off + k1p8]
        + vectors[v_off + k1p9] * vectors[v_off + k1p9]
        + vectors[v_off + k1p10] * vectors[v_off + k1p10]
        + vectors[v_off + k1p11] * vectors[v_off + k1p11];

    // Box-kite 2 (basis indices bk_basis[24..36]).
    let k2b0 = bk_basis[24usize] as usize;
    let k2b1 = bk_basis[25usize] as usize;
    let k2b2 = bk_basis[26usize] as usize;
    let k2b3 = bk_basis[27usize] as usize;
    let k2b4 = bk_basis[28usize] as usize;
    let k2b5 = bk_basis[29usize] as usize;
    let k2b6 = bk_basis[30usize] as usize;
    let k2b7 = bk_basis[31usize] as usize;
    let k2b8 = bk_basis[32usize] as usize;
    let k2b9 = bk_basis[33usize] as usize;
    let k2b10 = bk_basis[34usize] as usize;
    let k2b11 = bk_basis[35usize] as usize;
    let k2p0 = orientations[o_off + k2b0] as usize;
    let k2p1 = orientations[o_off + k2b1] as usize;
    let k2p2 = orientations[o_off + k2b2] as usize;
    let k2p3 = orientations[o_off + k2b3] as usize;
    let k2p4 = orientations[o_off + k2b4] as usize;
    let k2p5 = orientations[o_off + k2b5] as usize;
    let k2p6 = orientations[o_off + k2b6] as usize;
    let k2p7 = orientations[o_off + k2b7] as usize;
    let k2p8 = orientations[o_off + k2b8] as usize;
    let k2p9 = orientations[o_off + k2b9] as usize;
    let k2p10 = orientations[o_off + k2b10] as usize;
    let k2p11 = orientations[o_off + k2b11] as usize;
    let k2w = vectors[v_off + k2p0] * vectors[v_off + k2p0]
        + vectors[v_off + k2p1] * vectors[v_off + k2p1]
        + vectors[v_off + k2p2] * vectors[v_off + k2p2]
        + vectors[v_off + k2p3] * vectors[v_off + k2p3]
        + vectors[v_off + k2p4] * vectors[v_off + k2p4]
        + vectors[v_off + k2p5] * vectors[v_off + k2p5]
        + vectors[v_off + k2p6] * vectors[v_off + k2p6]
        + vectors[v_off + k2p7] * vectors[v_off + k2p7]
        + vectors[v_off + k2p8] * vectors[v_off + k2p8]
        + vectors[v_off + k2p9] * vectors[v_off + k2p9]
        + vectors[v_off + k2p10] * vectors[v_off + k2p10]
        + vectors[v_off + k2p11] * vectors[v_off + k2p11];

    // Box-kite 3 (basis indices bk_basis[36..48]).
    let k3b0 = bk_basis[36usize] as usize;
    let k3b1 = bk_basis[37usize] as usize;
    let k3b2 = bk_basis[38usize] as usize;
    let k3b3 = bk_basis[39usize] as usize;
    let k3b4 = bk_basis[40usize] as usize;
    let k3b5 = bk_basis[41usize] as usize;
    let k3b6 = bk_basis[42usize] as usize;
    let k3b7 = bk_basis[43usize] as usize;
    let k3b8 = bk_basis[44usize] as usize;
    let k3b9 = bk_basis[45usize] as usize;
    let k3b10 = bk_basis[46usize] as usize;
    let k3b11 = bk_basis[47usize] as usize;
    let k3p0 = orientations[o_off + k3b0] as usize;
    let k3p1 = orientations[o_off + k3b1] as usize;
    let k3p2 = orientations[o_off + k3b2] as usize;
    let k3p3 = orientations[o_off + k3b3] as usize;
    let k3p4 = orientations[o_off + k3b4] as usize;
    let k3p5 = orientations[o_off + k3b5] as usize;
    let k3p6 = orientations[o_off + k3b6] as usize;
    let k3p7 = orientations[o_off + k3b7] as usize;
    let k3p8 = orientations[o_off + k3b8] as usize;
    let k3p9 = orientations[o_off + k3b9] as usize;
    let k3p10 = orientations[o_off + k3b10] as usize;
    let k3p11 = orientations[o_off + k3b11] as usize;
    let k3w = vectors[v_off + k3p0] * vectors[v_off + k3p0]
        + vectors[v_off + k3p1] * vectors[v_off + k3p1]
        + vectors[v_off + k3p2] * vectors[v_off + k3p2]
        + vectors[v_off + k3p3] * vectors[v_off + k3p3]
        + vectors[v_off + k3p4] * vectors[v_off + k3p4]
        + vectors[v_off + k3p5] * vectors[v_off + k3p5]
        + vectors[v_off + k3p6] * vectors[v_off + k3p6]
        + vectors[v_off + k3p7] * vectors[v_off + k3p7]
        + vectors[v_off + k3p8] * vectors[v_off + k3p8]
        + vectors[v_off + k3p9] * vectors[v_off + k3p9]
        + vectors[v_off + k3p10] * vectors[v_off + k3p10]
        + vectors[v_off + k3p11] * vectors[v_off + k3p11];

    // Box-kite 4 (basis indices bk_basis[48..60]).
    let k4b0 = bk_basis[48usize] as usize;
    let k4b1 = bk_basis[49usize] as usize;
    let k4b2 = bk_basis[50usize] as usize;
    let k4b3 = bk_basis[51usize] as usize;
    let k4b4 = bk_basis[52usize] as usize;
    let k4b5 = bk_basis[53usize] as usize;
    let k4b6 = bk_basis[54usize] as usize;
    let k4b7 = bk_basis[55usize] as usize;
    let k4b8 = bk_basis[56usize] as usize;
    let k4b9 = bk_basis[57usize] as usize;
    let k4b10 = bk_basis[58usize] as usize;
    let k4b11 = bk_basis[59usize] as usize;
    let k4p0 = orientations[o_off + k4b0] as usize;
    let k4p1 = orientations[o_off + k4b1] as usize;
    let k4p2 = orientations[o_off + k4b2] as usize;
    let k4p3 = orientations[o_off + k4b3] as usize;
    let k4p4 = orientations[o_off + k4b4] as usize;
    let k4p5 = orientations[o_off + k4b5] as usize;
    let k4p6 = orientations[o_off + k4b6] as usize;
    let k4p7 = orientations[o_off + k4b7] as usize;
    let k4p8 = orientations[o_off + k4b8] as usize;
    let k4p9 = orientations[o_off + k4b9] as usize;
    let k4p10 = orientations[o_off + k4b10] as usize;
    let k4p11 = orientations[o_off + k4b11] as usize;
    let k4w = vectors[v_off + k4p0] * vectors[v_off + k4p0]
        + vectors[v_off + k4p1] * vectors[v_off + k4p1]
        + vectors[v_off + k4p2] * vectors[v_off + k4p2]
        + vectors[v_off + k4p3] * vectors[v_off + k4p3]
        + vectors[v_off + k4p4] * vectors[v_off + k4p4]
        + vectors[v_off + k4p5] * vectors[v_off + k4p5]
        + vectors[v_off + k4p6] * vectors[v_off + k4p6]
        + vectors[v_off + k4p7] * vectors[v_off + k4p7]
        + vectors[v_off + k4p8] * vectors[v_off + k4p8]
        + vectors[v_off + k4p9] * vectors[v_off + k4p9]
        + vectors[v_off + k4p10] * vectors[v_off + k4p10]
        + vectors[v_off + k4p11] * vectors[v_off + k4p11];

    // Box-kite 5 (basis indices bk_basis[60..72]).
    let k5b0 = bk_basis[60usize] as usize;
    let k5b1 = bk_basis[61usize] as usize;
    let k5b2 = bk_basis[62usize] as usize;
    let k5b3 = bk_basis[63usize] as usize;
    let k5b4 = bk_basis[64usize] as usize;
    let k5b5 = bk_basis[65usize] as usize;
    let k5b6 = bk_basis[66usize] as usize;
    let k5b7 = bk_basis[67usize] as usize;
    let k5b8 = bk_basis[68usize] as usize;
    let k5b9 = bk_basis[69usize] as usize;
    let k5b10 = bk_basis[70usize] as usize;
    let k5b11 = bk_basis[71usize] as usize;
    let k5p0 = orientations[o_off + k5b0] as usize;
    let k5p1 = orientations[o_off + k5b1] as usize;
    let k5p2 = orientations[o_off + k5b2] as usize;
    let k5p3 = orientations[o_off + k5b3] as usize;
    let k5p4 = orientations[o_off + k5b4] as usize;
    let k5p5 = orientations[o_off + k5b5] as usize;
    let k5p6 = orientations[o_off + k5b6] as usize;
    let k5p7 = orientations[o_off + k5b7] as usize;
    let k5p8 = orientations[o_off + k5b8] as usize;
    let k5p9 = orientations[o_off + k5b9] as usize;
    let k5p10 = orientations[o_off + k5b10] as usize;
    let k5p11 = orientations[o_off + k5b11] as usize;
    let k5w = vectors[v_off + k5p0] * vectors[v_off + k5p0]
        + vectors[v_off + k5p1] * vectors[v_off + k5p1]
        + vectors[v_off + k5p2] * vectors[v_off + k5p2]
        + vectors[v_off + k5p3] * vectors[v_off + k5p3]
        + vectors[v_off + k5p4] * vectors[v_off + k5p4]
        + vectors[v_off + k5p5] * vectors[v_off + k5p5]
        + vectors[v_off + k5p6] * vectors[v_off + k5p6]
        + vectors[v_off + k5p7] * vectors[v_off + k5p7]
        + vectors[v_off + k5p8] * vectors[v_off + k5p8]
        + vectors[v_off + k5p9] * vectors[v_off + k5p9]
        + vectors[v_off + k5p10] * vectors[v_off + k5p10]
        + vectors[v_off + k5p11] * vectors[v_off + k5p11];

    // Box-kite 6 (basis indices bk_basis[72..84]).
    let k6b0 = bk_basis[72usize] as usize;
    let k6b1 = bk_basis[73usize] as usize;
    let k6b2 = bk_basis[74usize] as usize;
    let k6b3 = bk_basis[75usize] as usize;
    let k6b4 = bk_basis[76usize] as usize;
    let k6b5 = bk_basis[77usize] as usize;
    let k6b6 = bk_basis[78usize] as usize;
    let k6b7 = bk_basis[79usize] as usize;
    let k6b8 = bk_basis[80usize] as usize;
    let k6b9 = bk_basis[81usize] as usize;
    let k6b10 = bk_basis[82usize] as usize;
    let k6b11 = bk_basis[83usize] as usize;
    let k6p0 = orientations[o_off + k6b0] as usize;
    let k6p1 = orientations[o_off + k6b1] as usize;
    let k6p2 = orientations[o_off + k6b2] as usize;
    let k6p3 = orientations[o_off + k6b3] as usize;
    let k6p4 = orientations[o_off + k6b4] as usize;
    let k6p5 = orientations[o_off + k6b5] as usize;
    let k6p6 = orientations[o_off + k6b6] as usize;
    let k6p7 = orientations[o_off + k6b7] as usize;
    let k6p8 = orientations[o_off + k6b8] as usize;
    let k6p9 = orientations[o_off + k6b9] as usize;
    let k6p10 = orientations[o_off + k6b10] as usize;
    let k6p11 = orientations[o_off + k6b11] as usize;
    let k6w = vectors[v_off + k6p0] * vectors[v_off + k6p0]
        + vectors[v_off + k6p1] * vectors[v_off + k6p1]
        + vectors[v_off + k6p2] * vectors[v_off + k6p2]
        + vectors[v_off + k6p3] * vectors[v_off + k6p3]
        + vectors[v_off + k6p4] * vectors[v_off + k6p4]
        + vectors[v_off + k6p5] * vectors[v_off + k6p5]
        + vectors[v_off + k6p6] * vectors[v_off + k6p6]
        + vectors[v_off + k6p7] * vectors[v_off + k6p7]
        + vectors[v_off + k6p8] * vectors[v_off + k6p8]
        + vectors[v_off + k6p9] * vectors[v_off + k6p9]
        + vectors[v_off + k6p10] * vectors[v_off + k6p10]
        + vectors[v_off + k6p11] * vectors[v_off + k6p11];

    // Max over 7 box-kites via f32::max chains (no mutable accumulator).
    let max_raw = f32::max(
        k0w,
        f32::max(
            k1w,
            f32::max(k2w, f32::max(k3w, f32::max(k4w, f32::max(k5w, k6w)))),
        ),
    );
    out_scores[pair_idx] = max_raw * inv_norm_sq;
}

// ---- public API -------------------------------------------------------------

/// Probe whether a cubecl-wgpu adapter is reachable on this host.
pub fn is_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

/// Errors from the cubecl alignment launcher.
#[derive(Debug, thiserror::Error)]
pub enum CubeclAlignmentError {
    #[error("vectors length {got} is not a multiple of 16")]
    VectorsNotAligned { got: usize },
    #[error("orientations length {got} is not a multiple of 16")]
    OrientationsNotAligned { got: usize },
    #[error("bk_basis length {got} != 84 (7 box-kites * 12 basis each)")]
    BkBasisWrongLength { got: usize },
    #[error("cubecl adapter not available on this host")]
    AdapterUnavailable,
}

/// Compute box-kite alignment scan via cubecl-wgpu.
///
/// `vectors` is a flat f32 array of n_vectors * 16 components (row-major).
/// `orientations` is a flat u32 array of n_orientations * 16 permutation
/// entries (each entry is the permuted component index for that slot).
/// `bk_basis` is a flat u32 array of 84 entries: for box-kite k, the 12
/// basis indices occupy positions [k*12 .. k*12+12].
///
/// Returns `(max_align, best_orient)` where each element corresponds to one
/// input vector.  Uses an intermediate per-(v, o) score buffer; the host-side
/// argmax avoids mutable f32 in the GPU kernel body.
pub fn box_kite_alignment_scan_cubecl(
    vectors: &[f32],
    orientations: &[u32],
    bk_basis: &[u32],
) -> Result<(Vec<f32>, Vec<u32>), CubeclAlignmentError> {
    if !vectors.len().is_multiple_of(16) {
        return Err(CubeclAlignmentError::VectorsNotAligned { got: vectors.len() });
    }
    if !orientations.len().is_multiple_of(16) {
        return Err(CubeclAlignmentError::OrientationsNotAligned {
            got: orientations.len(),
        });
    }
    if bk_basis.len() != 84 {
        return Err(CubeclAlignmentError::BkBasisWrongLength {
            got: bk_basis.len(),
        });
    }

    let n_vectors = vectors.len() / 16;
    let n_orientations = orientations.len() / 16;

    if n_vectors == 0 {
        return Ok((vec![], vec![]));
    }

    if !is_available() {
        return Err(CubeclAlignmentError::AdapterUnavailable);
    }

    let n_pairs = n_vectors * n_orientations;

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let v_bytes: &[u8] = bytemuck::cast_slice(vectors);
    let o_bytes: &[u8] = bytemuck::cast_slice(orientations);
    let b_bytes: &[u8] = bytemuck::cast_slice(bk_basis);

    let v_handle = client.create_from_slice(v_bytes);
    let o_handle = client.create_from_slice(o_bytes);
    let b_handle = client.create_from_slice(b_bytes);
    let scores_handle = client.empty(n_pairs * std::mem::size_of::<f32>());
    let scores_rb = scores_handle.clone();

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::new_1d(n_pairs.div_ceil(256) as u32);

    // SAFETY: launch_unchecked requires the caller to vouch for buffer sizes:
    //   vectors     = n_vectors * 16 f32
    //   orientations = n_orientations * 16 u32
    //   bk_basis    = 84 u32
    //   out_scores  = n_pairs f32
    // Threads beyond n_pairs terminate immediately.
    unsafe {
        box_kite_score_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(v_handle, vectors.len()),
            ArrayArg::from_raw_parts(o_handle, orientations.len()),
            ArrayArg::from_raw_parts(b_handle, bk_basis.len()),
            ArrayArg::from_raw_parts(scores_handle, n_pairs),
            n_pairs as u32,
            n_orientations as u32,
            n_vectors as u32,
        );
    }

    let scores_bytes = client.read_one_unchecked(scores_rb);
    let scores: &[f32] = bytemuck::cast_slice(&scores_bytes);

    // Host argmax: for each vector, find the orientation with the highest score.
    let mut max_align = Vec::with_capacity(n_vectors);
    let mut best_orient = Vec::with_capacity(n_vectors);
    for v in 0..n_vectors {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_o = 0u32;
        for o in 0..n_orientations {
            let s = scores[v * n_orientations + o];
            if s > best_score {
                best_score = s;
                best_o = o as u32;
            }
        }
        max_align.push(best_score.max(0.0));
        best_orient.push(best_o);
    }

    Ok((max_align, best_orient))
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_does_not_panic() {
        let _ = is_available();
    }

    #[test]
    fn rejects_vectors_not_multiple_of_16() {
        let bad_vecs = vec![0.0_f32; 17];
        let orients = vec![0u32; 16];
        let basis = vec![0u32; 84];
        match box_kite_alignment_scan_cubecl(&bad_vecs, &orients, &basis) {
            Err(CubeclAlignmentError::VectorsNotAligned { got: 17 }) => {}
            other => panic!("expected VectorsNotAligned, got {:?}", other),
        }
    }

    #[test]
    fn rejects_wrong_bk_basis_length() {
        let vecs = vec![0.0_f32; 16];
        let orients = vec![0u32; 16];
        let bad_basis = vec![0u32; 80];
        match box_kite_alignment_scan_cubecl(&vecs, &orients, &bad_basis) {
            Err(CubeclAlignmentError::BkBasisWrongLength { got: 80 }) => {}
            other => panic!("expected BkBasisWrongLength, got {:?}", other),
        }
    }

    #[test]
    fn empty_vectors_returns_empty_without_gpu() {
        let orients = vec![0u32; 16];
        let basis = vec![0u32; 84];
        let (a, b) = box_kite_alignment_scan_cubecl(&[], &orients, &basis).unwrap();
        assert!(a.is_empty());
        assert!(b.is_empty());
    }
}
