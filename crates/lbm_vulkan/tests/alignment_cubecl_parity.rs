// Copyright (c) 2026 Terascale Functionalists
// SPDX-License-Identifier: GPL-2.0-or-later

//! CPU (algebra_analysis, f64) vs cubecl-wgpu (f32) parity test for the
//! box-kite alignment scan.
//!
//! The test generates N=64 random 16-component sedenion vectors, prepares the
//! PSL(2,7) orientation table and box-kite basis arrays, runs both the CPU
//! reference and the cubecl path, and checks that:
//!
//!   1. Every `best_orient` index agrees exactly unless both chosen
//!      orientations cross-score as ties within f32 tolerance.
//!   2. Every `max_align` value agrees within f32 single-precision tolerance
//!      (abs_tol=1e-5, rel_tol=1e-4).
//!
//! Tolerances are wider than integer-exact because the cubecl path works in
//! f32 while the CPU oracle uses f64.
//!
//! Gated `#[ignore = "gpu (cubecl-wgpu adapter required)"]`; skipped cleanly
//! when no wgpu adapter is present.
//!
//! Run with:
//!   cargo test -p lbm_vulkan --features cubecl --release \
//!     alignment_cubecl_parity -- --ignored --nocapture

#![cfg(feature = "cubecl")]

use algebra_analysis::{
    boxkite_alignment::{box_kite_alignment_scan_cpu, generate_psl_2_7_permutations_16d},
    boxkites::{BoxKite, cached_sedenion_boxkites},
};
use lbm_vulkan::alignment_cubecl::{box_kite_alignment_scan_cubecl, is_available};
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
};
use rand_chacha::ChaCha20Rng;

const N_VECTORS: usize = 64;
const ABS_TOL: f64 = 1e-5;
const REL_TOL: f64 = 1e-4;
const SEED: u64 = 0x00A1_19B0_7CA1_6EDC;

fn boxkite_basis_sets(boxkites: &[BoxKite]) -> Vec<Vec<usize>> {
    let mut basis_sets = Vec::with_capacity(boxkites.len());
    for (boxkite_idx, boxkite) in boxkites.iter().enumerate() {
        let mut indices = std::collections::BTreeSet::new();
        for assessor in &boxkite.assessors {
            indices.insert(assessor.low);
            indices.insert(assessor.high);
        }
        let boxkite_basis: Vec<usize> = indices.into_iter().collect();
        assert_eq!(
            boxkite_basis.len(),
            12,
            "box-kite {boxkite_idx} must have exactly 12 unique basis indices"
        );
        basis_sets.push(boxkite_basis);
    }
    basis_sets
}

fn flatten_boxkite_basis(basis_sets: &[Vec<usize>]) -> Vec<u32> {
    basis_sets
        .iter()
        .flat_map(|basis_set| basis_set.iter().map(|&i| i as u32))
        .collect()
}

fn scores_close(lhs: f64, rhs: f64) -> bool {
    let err = (lhs - rhs).abs();
    let scale = lhs.abs().max(rhs.abs()).max(1e-12);
    err <= ABS_TOL || err / scale <= REL_TOL
}

fn alignment_score_f64(
    vector: &[f64; 16],
    orientation: &[usize; 16],
    basis_sets: &[Vec<usize>],
) -> f64 {
    let norm_sq: f64 = vector.iter().map(|value| value * value).sum();
    if norm_sq < 1e-30 {
        return 0.0;
    }

    basis_sets
        .iter()
        .map(|basis_set| {
            let proj_sq: f64 = basis_set
                .iter()
                .map(|&basis_idx| {
                    let permuted_idx = orientation[basis_idx];
                    vector[permuted_idx] * vector[permuted_idx]
                })
                .sum();
            proj_sq / norm_sq
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

fn alignment_score_f32(
    vector: &[f32],
    orientation: &[usize; 16],
    basis_sets: &[Vec<usize>],
) -> f64 {
    let norm_sq: f32 = vector.iter().map(|value| value * value).sum();
    let inv_norm_sq = 1.0_f32 / (norm_sq + 1e-30_f32);

    basis_sets
        .iter()
        .map(|basis_set| {
            let proj_sq: f32 = basis_set
                .iter()
                .map(|&basis_idx| {
                    let permuted_idx = orientation[basis_idx];
                    vector[permuted_idx] * vector[permuted_idx]
                })
                .sum();
            proj_sq * inv_norm_sq
        })
        .fold(f32::NEG_INFINITY, f32::max)
        .max(0.0) as f64
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn cpu_vs_cubecl_boxkite_alignment_64vectors() {
    if !is_available() {
        eprintln!("skip: cubecl wgpu adapter not available");
        return;
    }

    let mut rng = ChaCha20Rng::seed_from_u64(SEED);
    let dist = Uniform::new(-1.0_f64, 1.0_f64).expect("range valid");

    // Generate random 16-component sedenion vectors (f64 for CPU oracle).
    let vectors_f64: Vec<[f64; 16]> = (0..N_VECTORS)
        .map(|_| {
            let mut v = [0.0_f64; 16];
            for x in &mut v {
                *x = dist.sample(&mut rng);
            }
            v
        })
        .collect();

    // PSL(2,7) orientations from algebra_analysis.
    let orientations_usize = generate_psl_2_7_permutations_16d();
    let n_orientations = orientations_usize.len();

    // Box-kite structures from algebra_analysis.
    let boxkites = cached_sedenion_boxkites();
    assert_eq!(boxkites.len(), 7, "expected exactly 7 sedenion box-kites");
    let basis_sets = boxkite_basis_sets(boxkites);

    // CPU oracle.
    let (cpu_max, cpu_best) =
        box_kite_alignment_scan_cpu(&vectors_f64, &orientations_usize, boxkites);
    assert_eq!(cpu_max.len(), N_VECTORS);
    assert_eq!(cpu_best.len(), N_VECTORS);

    // Prepare flat f32 arrays for the cubecl kernel.
    let vectors_f32: Vec<f32> = vectors_f64.iter().flatten().map(|&x| x as f32).collect();

    // orientations: flat u32, 16 per orientation (permuted index per slot).
    let orientations_u32: Vec<u32> = orientations_usize
        .iter()
        .flat_map(|perm| perm.iter().map(|&i| i as u32))
        .collect();

    let bk_basis = flatten_boxkite_basis(&basis_sets);
    assert_eq!(bk_basis.len(), 84);

    // cubecl path.
    let (cl_max, cl_best) =
        box_kite_alignment_scan_cubecl(&vectors_f32, &orientations_u32, &bk_basis)
            .expect("cubecl alignment scan succeeds");
    assert_eq!(cl_max.len(), N_VECTORS);
    assert_eq!(cl_best.len(), N_VECTORS);

    let mut max_align_err = 0.0_f64;
    let mut orientation_tie_mismatches = 0usize;
    for i in 0..N_VECTORS {
        let cpu_o = cpu_best[i];
        let cl_o = cl_best[i];

        let cpu_m = cpu_max[i];
        let cl_m = cl_max[i] as f64;

        let align_err = (cpu_m - cl_m).abs();
        let align_rel = align_err / cpu_m.abs().max(cl_m.abs()).max(1e-12);
        let scores_match = scores_close(cpu_m, cl_m);

        if cpu_o != cl_o {
            let cpu_score_at_cubecl_orient = alignment_score_f64(
                &vectors_f64[i],
                &orientations_usize[cl_o as usize],
                &basis_sets,
            );
            let vector_f32 = &vectors_f32[i * 16..(i + 1) * 16];
            let cubecl_score_at_cpu_orient =
                alignment_score_f32(vector_f32, &orientations_usize[cpu_o as usize], &basis_sets);
            let cpu_tie = scores_close(cpu_m, cpu_score_at_cubecl_orient);
            let cubecl_tie = scores_close(cl_m, cubecl_score_at_cpu_orient);

            assert!(
                scores_match && cpu_tie && cubecl_tie,
                "vector {i}: orient mismatch without verified tie: cpu={cpu_o}, cubecl={cl_o}, \
                 cpu_max={cpu_m:.6e}, cl_max={cl_m:.6e}, \
                 cpu_score_at_cubecl_orient={cpu_score_at_cubecl_orient:.6e}, \
                 cubecl_score_at_cpu_orient={cubecl_score_at_cpu_orient:.6e}, \
                 scores_match={scores_match}, cpu_tie={cpu_tie}, cubecl_tie={cubecl_tie}"
            );

            orientation_tie_mismatches += 1;
        }

        assert!(
            scores_match,
            "vector {i}: max_align mismatch: cpu={cpu_m:.6e}, cubecl={cl_m:.6e}, \
             cpu_orient={cpu_o}, cubecl_orient={cl_o}, \
             abs_err={align_err:.3e}, rel_err={align_rel:.3e} \
             (abs_tol={ABS_TOL:.1e}, rel_tol={REL_TOL:.1e})"
        );

        max_align_err = max_align_err.max(align_err);
    }

    eprintln!(
        "CPU-vs-cubecl alignment OK: {} vectors, {} orientations, max_align_err={max_align_err:.3e}, orientation_tie_mismatches={orientation_tie_mismatches}",
        N_VECTORS, n_orientations,
    );
}
