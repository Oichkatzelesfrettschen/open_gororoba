// Copyright (c) 2026 Terascale Functionalists
// SPDX-License-Identifier: GPL-2.0-or-later

//! Parity tests for the Besag-Clifford cubecl shuffle + transform pipeline.
//!
//! Two tests, both gated `#[ignore = "gpu"]`:
//!
//! 1. `shuffle_matches_cpu_pcg`: verifies the PCG shuffle kernel produces
//!    bit-identical output to the CPU reference for the same seed and batch_idx.
//!    PCG hash uses only integer arithmetic (exact on all platforms), so the
//!    shuffled values must match exactly.
//!
//! 2. `shuffle_and_transform_matches_cpu`: verifies the combined GPU shuffle +
//!    transform_viscosity pipeline matches the CPU reference within f32 exp()
//!    precision (abs_tol=1e-5, rel_tol=1e-4).
//!
//! Run with:
//!   cargo test -p lbm_vulkan --features cubecl --release \
//!     besag_clifford_cubecl_parity -- --ignored --nocapture

#![cfg(feature = "cubecl")]

use lbm_vulkan::besag_clifford_cubecl::{
    is_available, shuffle_and_transform_cubecl, shuffle_imbalance_cubecl,
};
use lbm_vulkan::transform_viscosity_cpu::{transform_viscosity_cpu, ViscosityTransformInputs};
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand_chacha::ChaCha20Rng;

const NX: u32 = 16;
const NY: u32 = 16;
const NZ: u32 = 16;
const N_CELLS: usize = (NX * NY * NZ) as usize;
const SEED: u32 = 0xCA7E_C0FF;
const BATCH_IDX: u32 = 7;
const NU_BASE: f32 = 0.01;
const LAMBDA: f32 = 1.5;
const PHI_MEAN: f32 = 0.375;
const RNG_SEED: u64 = 0x1337_BEEF_DEAD_C0DE;
const ABS_TOL: f32 = 1e-5;
const REL_TOL: f32 = 1e-4;

/// CPU reference: PCG hash matching besag_clifford.wgsl::pcg_hash.
fn pcg_hash_cpu(input: u32) -> u32 {
    let state = input.wrapping_mul(747796405_u32).wrapping_add(2891336453_u32);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737_u32);
    (word >> 22) ^ word
}

/// CPU reference shuffle: output[i] = imbalance[pcg_hash(seed ^ batch_idx ^ i) % n].
/// Matches the direct-read approach in pcg_shuffle_kernel.
fn cpu_pcg_shuffle(imbalance: &[f32], seed: u32, batch_idx: u32) -> Vec<f32> {
    let n = imbalance.len() as u32;
    (0..imbalance.len())
        .map(|i| {
            let rng_in = seed ^ batch_idx ^ (i as u32);
            let partner = (pcg_hash_cpu(rng_in) % n) as usize;
            imbalance[partner]
        })
        .collect()
}

fn random_imbalance(n: usize) -> Vec<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(RNG_SEED);
    let dist = Uniform::new(-1.0_f32, 1.0_f32).expect("range valid");
    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn shuffle_matches_cpu_pcg() {
    if !is_available() {
        eprintln!("skip: cubecl wgpu adapter not available");
        return;
    }

    let imbalance = random_imbalance(N_CELLS);
    let cpu_shuffled = cpu_pcg_shuffle(&imbalance, SEED, BATCH_IDX);

    let gpu_shuffled =
        shuffle_imbalance_cubecl(&imbalance, SEED, BATCH_IDX).expect("shuffle cubecl succeeds");

    assert_eq!(
        cpu_shuffled.len(),
        gpu_shuffled.len(),
        "output length mismatch"
    );

    let mismatches: Vec<usize> = cpu_shuffled
        .iter()
        .zip(gpu_shuffled.iter())
        .enumerate()
        .filter(|(_, (c, g))| c.to_bits() != g.to_bits())
        .map(|(i, _)| i)
        .collect();

    assert!(
        mismatches.is_empty(),
        "PCG shuffle mismatch at {} / {} cells: first few indices={:?}",
        mismatches.len(),
        N_CELLS,
        &mismatches[..mismatches.len().min(5)]
    );
    eprintln!(
        "shuffle_matches_cpu_pcg OK: {} cells, 0 bit mismatches",
        N_CELLS
    );
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn shuffle_and_transform_matches_cpu() {
    if !is_available() {
        eprintln!("skip: cubecl wgpu adapter not available");
        return;
    }

    let imbalance = random_imbalance(N_CELLS);

    // CPU reference: PCG shuffle then transform_viscosity
    let cpu_shuffled = cpu_pcg_shuffle(&imbalance, SEED, BATCH_IDX);
    let cpu_tau = transform_viscosity_cpu(&ViscosityTransformInputs {
        phi: cpu_shuffled,
        nu_base: NU_BASE,
        lambda: LAMBDA,
        phi_mean: PHI_MEAN,
    });

    let gpu_tau = shuffle_and_transform_cubecl(
        &imbalance, NU_BASE, LAMBDA, PHI_MEAN, SEED, BATCH_IDX,
    )
    .expect("shuffle_and_transform_cubecl succeeds");

    assert_eq!(cpu_tau.len(), gpu_tau.len(), "output length mismatch");

    let mut max_abs_err = 0.0f32;
    let mut n_fail = 0usize;
    for (i, (&c, &g)) in cpu_tau.iter().zip(gpu_tau.iter()).enumerate() {
        let abs_err = (c - g).abs();
        let rel_err = abs_err / c.abs().max(1e-9_f32);
        if abs_err > ABS_TOL && rel_err > REL_TOL {
            n_fail += 1;
            if n_fail <= 3 {
                eprintln!(
                    "cell {i}: cpu={c:.8e} gpu={g:.8e} abs={abs_err:.3e} rel={rel_err:.3e}"
                );
            }
        }
        max_abs_err = max_abs_err.max(abs_err);
    }
    assert_eq!(
        n_fail, 0,
        "{n_fail} / {} cells exceed tolerance (abs_tol={ABS_TOL:.1e}, rel_tol={REL_TOL:.1e})",
        N_CELLS
    );
    eprintln!(
        "shuffle_and_transform_matches_cpu OK: {} cells, max_abs_err={max_abs_err:.3e}",
        N_CELLS
    );
}
