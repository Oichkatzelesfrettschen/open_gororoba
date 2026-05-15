//! Parity test: CPU oracle vs cubecl box-counting.
//!
//! WHY: Closes one cell of the cubecl + Vulkan + CUDA parity matrix
//! (docs/engineering/cubecl_vulkan_cuda_parity_matrix.md). The cubecl
//! path must produce bit-identical u32 counts to the CPU reference at
//! every scale, for the same input grid and threshold.
//!
//! WHAT: A single integration test that:
//!   1. Builds a 32x32x32 density grid populated with a fixed-seed
//!      ChaCha20Rng pattern.
//!   2. Runs the CPU oracle at scales 1, 2, 4, 8, 16 -> (size, count) pairs.
//!   3. Runs the cubecl path at the same scales -> (size, count) pairs.
//!   4. Asserts pairs are EQUAL (bit-identical -- counts are integers).
//!
//! HOW: Gated `#[ignore = "gpu"]` because the test requires a cubecl-wgpu
//! reachable adapter (Vulkan / Metal / DX12 / WebGPU). Runtime adapter
//! probe via `box_counting_cubecl::is_available()` skips cleanly if
//! absent. Local run:
//!
//! ```sh
//! cargo test -p lbm_vulkan --features cubecl --release \
//!   --test box_counting_cubecl_parity -- --ignored --nocapture
//! ```

#![cfg(feature = "cubecl")]

use lbm_vulkan::{
    box_counting_cpu::{count_occupied_boxes, default_box_sizes},
    box_counting_cubecl::{count_occupied_boxes_cubecl, is_available},
};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha20Rng;

fn build_test_grid(
    seed: u64,
    nx: usize,
    ny: usize,
    nz: usize,
    occupancy_fraction: f64,
) -> Vec<f32> {
    let total = nx * ny * nz;
    let n_occupied = (total as f64 * occupancy_fraction) as usize;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..total).collect();
    indices.shuffle(&mut rng);

    let mut rho = vec![0.0f32; total];
    for &i in indices.iter().take(n_occupied) {
        rho[i] = 1.0;
    }
    rho
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn cubecl_parity_32cube_random() {
    if !is_available() {
        eprintln!(
            "skip: cubecl-wgpu adapter not reachable on this host (headless container?). \
             Set up libvulkan / Metal / DX12 adapter to enable this test."
        );
        return;
    }

    let nx = 32;
    let ny = 32;
    let nz = 32;
    let threshold = 0.5;

    // Three random occupancy fractions chosen so each scale gives a
    // meaningfully different count.
    for &occupancy in &[0.10, 0.25, 0.50] {
        let rho = build_test_grid(42 + (occupancy * 100.0) as u64, nx, ny, nz, occupancy);

        for &s in &default_box_sizes(nx.min(ny).min(nz)) {
            let cpu = count_occupied_boxes(&rho, threshold, nx, ny, nz, s);
            let gpu = count_occupied_boxes_cubecl(&rho, threshold, nx, ny, nz, s)
                .expect("cubecl path must succeed when is_available() is true");
            assert_eq!(
                cpu, gpu,
                "mismatch at occupancy={} box_size={}: cpu={} gpu={}",
                occupancy, s, cpu, gpu
            );
        }
    }
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn cubecl_parity_filled_cube_at_every_scale() {
    if !is_available() {
        eprintln!("skip: cubecl-wgpu adapter not reachable on this host.");
        return;
    }

    let n = 16;
    let rho = vec![1.0f32; n * n * n];
    let threshold = 0.5;

    for &s in &default_box_sizes(n) {
        let cpu = count_occupied_boxes(&rho, threshold, n, n, n, s);
        let gpu = count_occupied_boxes_cubecl(&rho, threshold, n, n, n, s).unwrap();
        let expected = (n / s).pow(3) as u32;
        assert_eq!(cpu, expected, "cpu oracle wrong for filled cube at s={}", s);
        assert_eq!(gpu, expected, "gpu wrong for filled cube at s={}", s);
    }
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn cubecl_parity_empty_grid_zero_count() {
    if !is_available() {
        eprintln!("skip: cubecl-wgpu adapter not reachable on this host.");
        return;
    }

    let n = 16;
    let rho = vec![0.0f32; n * n * n];
    for &s in &default_box_sizes(n) {
        let cpu = count_occupied_boxes(&rho, 0.5, n, n, n, s);
        let gpu = count_occupied_boxes_cubecl(&rho, 0.5, n, n, n, s).unwrap();
        assert_eq!(cpu, 0);
        assert_eq!(gpu, 0);
    }
}
