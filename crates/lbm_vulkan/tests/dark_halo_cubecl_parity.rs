//! CPU parity for the dark-halo cubecl ZD viscosity hash.
//!
//! The cubecl path ports `dark_halo_viscosity.wgsl`. This test mirrors the
//! Vulkan parity lane's deterministic ZD-count case: velocity and density are
//! treated as trivially passing, so only `(tau - tau_base) / tau_amp` controls
//! the count.

#![cfg(feature = "cubecl")]

use lbm_vulkan::{
    dark_halo_cubecl::{count_zd_cubecl, is_available},
    dark_halo_vulkan::DarkHaloConfig,
};

const NX: usize = 16;
const NY: usize = 16;
const NZ: usize = 16;
const SEED: u32 = 0xDEAD_C0DE;
const K_DIM: u32 = 16;
const TAU_BASE: f32 = 1.0;
const TAU_AMP: f32 = 0.3;
const STEPS: usize = 5;

fn cpu_tau(
    nx: usize,
    ny: usize,
    nz: usize,
    seed: u32,
    k_dim: u32,
    tau_base: f32,
    tau_amp: f32,
) -> Vec<f32> {
    let lambda = (k_dim as f32).ln();
    let n_cells = nx * ny * nz;
    let mut tau = vec![0.0_f32; n_cells];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let cell = x + nx * (y + ny * z);
                let mut hash: u32 = seed
                    ^ (x as u32).wrapping_mul(73_856_093)
                    ^ (y as u32).wrapping_mul(19_349_663)
                    ^ (z as u32).wrapping_mul(83_492_791);
                hash = (hash >> 13) ^ hash;
                hash = hash.wrapping_mul(0x5bd1_e995);
                hash = (hash >> 15) ^ hash;
                let noise = (hash & 0xffff) as f32 / 65_535.0;
                tau[cell] = tau_base + tau_amp * (lambda * noise).sin();
            }
        }
    }
    tau
}

fn cpu_zd_count(tau: &[f32], tau_base: f32, tau_amp: f32, zd_threshold: f32) -> u32 {
    tau.iter()
        .map(|&value| {
            let proxy = if tau_amp > 0.0 {
                (value - tau_base) / tau_amp
            } else {
                0.0
            };
            u32::from(proxy > zd_threshold)
        })
        .sum()
}

#[test]
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn deterministic_zd_count_matches_cpu() {
    if !is_available() {
        eprintln!("skip: cubecl wgpu adapter not available");
        return;
    }

    let n_cells = NX * NY * NZ;
    let zd_threshold = 0.0_f32;
    let tau_cpu = cpu_tau(NX, NY, NZ, SEED, K_DIM, TAU_BASE, TAU_AMP);
    let expected_count = cpu_zd_count(&tau_cpu, TAU_BASE, TAU_AMP, zd_threshold);

    let config = DarkHaloConfig {
        k_dim: K_DIM,
        steps: STEPS,
        seed: SEED,
        tau_base: TAU_BASE,
        tau_amp: TAU_AMP,
        zd_threshold,
        velocity_epsilon: 2.0,
        density_factor: 0.0,
    };
    let result = count_zd_cubecl(NX, NY, NZ, &config).expect("dark-halo cubecl ZD count succeeds");
    assert_eq!(result.n_cells, n_cells);

    let tolerance = (n_cells / 100).max(4) as u32;
    let diff = (result.halo_count as i64 - expected_count as i64).unsigned_abs() as u32;
    assert!(
        diff <= tolerance,
        "halo_count mismatch: cubecl={}, cpu={}, diff={} (tol={})",
        result.halo_count,
        expected_count,
        diff,
        tolerance
    );
    eprintln!(
        "dark-halo cubecl ZD parity OK on {}x{}x{}: cubecl={}, cpu={}, diff={}, fraction={:.3}",
        NX, NY, NZ, result.halo_count, expected_count, diff, result.halo_fraction
    );
}
