//! Parity test: chingon AVT contraction CPU oracle vs cubecl path.
//!
//! WHY: Closes another cell of the cubecl + Vulkan + CUDA parity matrix
//! (#136 Phase 2). Vulkan and CUDA chingon paths exist already; the
//! cubecl path needs validation against the CPU reference before it can
//! be wired into the production dispatcher.
//!
//! WHAT: Three integration tests gated `#[ignore = "gpu"]`:
//!
//! 1. Random sparse violations on a 16-D state with seeded ChaCha20Rng.
//! 2. Random dense violations on a 64-D state (~256 violations).
//! 3. Diagonal-only violations (each thread writes to its own m slot).
//!
//! HOW: Each test builds a `ChingonInputs`, runs the CPU oracle, runs the
//! cubecl path, asserts element-wise within tolerance:
//!     |force_gpu[m] - force_cpu[m]| < eps_abs + eps_rel * |force_cpu[m]|
//! with `eps_abs = 1e-5` and `eps_rel = 1e-5` (the algorithm accumulates
//! n_violations f32 products so a single-precision rounding error of
//! sqrt(n_viol) * eps_machine is expected).
//!
//! Run command:
//!
//! ```sh
//! cargo test -p lbm_vulkan --features cubecl --release \
//!   --test chingon_cubecl_parity -- --ignored --nocapture
//! ```

#![cfg(feature = "cubecl")]

use lbm_vulkan::chingon_cpu::{ChingonInputs, chingon_contract_cpu};
use lbm_vulkan::chingon_cubecl::{chingon_contract_cubecl, is_available};
use lbm_vulkan::chingon_vulkan::{index_bits_for_dim, pack_violation};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::RngExt;
use rand_chacha::ChaCha20Rng;

const EPS_ABS: f32 = 1e-5;
const EPS_REL: f32 = 1e-5;

fn assert_force_close(cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "force length mismatch");
    for (idx, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let tol = EPS_ABS + EPS_REL * c.abs();
        let diff = (g - c).abs();
        assert!(
            diff <= tol,
            "force[{}] mismatch: cpu={} gpu={} diff={} > tol={}",
            idx,
            c,
            g,
            diff,
            tol
        );
    }
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn chingon_parity_16d_sparse() {
    if !is_available() {
        eprintln!("skip: cubecl-wgpu adapter not reachable on this host.");
        return;
    }
    let dim: u32 = 16;
    let ib = index_bits_for_dim(dim);
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let v_nd: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
    let n_viol = 32;
    let packed_avt: Vec<u32> = (0..n_viol)
        .map(|_| {
            let i = rng.random_range(0..dim);
            let j = rng.random_range(0..dim);
            let m = rng.random_range(0..dim);
            let sign = rng.random::<bool>();
            pack_violation(i, j, m, sign, ib)
        })
        .collect();

    let inputs = ChingonInputs {
        v_nd,
        packed_avt,
        alpha: 0.5,
        inv_n_viol: 1.0 / (n_viol as f32),
        index_bits: ib,
        dim,
    };
    let cpu = chingon_contract_cpu(&inputs);
    let gpu = chingon_contract_cubecl(&inputs).expect("cubecl should succeed");
    assert_force_close(&cpu, &gpu);
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn chingon_parity_64d_dense() {
    if !is_available() {
        eprintln!("skip: cubecl-wgpu adapter not reachable on this host.");
        return;
    }
    let dim: u32 = 64;
    let ib = index_bits_for_dim(dim);
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let v_nd: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
    let n_viol = 256;
    let packed_avt: Vec<u32> = (0..n_viol)
        .map(|_| {
            let i = rng.random_range(0..dim);
            let j = rng.random_range(0..dim);
            let m = rng.random_range(0..dim);
            let sign = rng.random::<bool>();
            pack_violation(i, j, m, sign, ib)
        })
        .collect();

    let inputs = ChingonInputs {
        v_nd,
        packed_avt,
        alpha: 1.0,
        inv_n_viol: 1.0 / (n_viol as f32),
        index_bits: ib,
        dim,
    };
    let cpu = chingon_contract_cpu(&inputs);
    let gpu = chingon_contract_cubecl(&inputs).expect("cubecl should succeed");
    assert_force_close(&cpu, &gpu);
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn chingon_parity_diagonal_no_collisions() {
    if !is_available() {
        eprintln!("skip: cubecl-wgpu adapter not reachable on this host.");
        return;
    }
    let dim: u32 = 8;
    let ib = index_bits_for_dim(dim);
    let v_nd: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 * 0.1).collect();
    // 8 diagonal violations, each writing to a different m slot:
    let mut packed_avt: Vec<u32> = (0..dim)
        .map(|m| pack_violation(m, m, m, true, ib))
        .collect();
    // Shuffle so the dispatch order doesn't reveal trivial structure.
    let mut rng = ChaCha20Rng::seed_from_u64(13);
    packed_avt.shuffle(&mut rng);

    let inputs = ChingonInputs {
        v_nd,
        packed_avt,
        alpha: 1.0,
        inv_n_viol: 1.0,
        index_bits: ib,
        dim,
    };
    let cpu = chingon_contract_cpu(&inputs);
    let gpu = chingon_contract_cubecl(&inputs).expect("cubecl should succeed");
    assert_force_close(&cpu, &gpu);
}
