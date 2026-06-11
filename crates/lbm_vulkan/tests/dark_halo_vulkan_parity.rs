//! Parity tests for `DarkHaloVulkan` against CPU reference implementations.
//!
//! Two test strategies:
//!
//! 1. `deterministic_zd_count`: disables the LBM-sensitive velocity and
//!    density criteria (velocity_epsilon=2.0, density_factor=0.0) so the
//!    detector fires purely on the ZD proxy, which is computed from the
//!    deterministic spatial hash.  The CPU reference replicates the exact
//!    same Murmur-inspired hash and f32 sin, so GPU and CPU halo counts
//!    should agree to within the precision of f32::sin vs WGSL sin
//!    (tolerance: <= n_cells / 100 cells differ).
//!
//! 2. `all_pass_all_fail`: trivial boundary smoke test.  Runs with
//!    thresholds that force all cells to be halos, then all cells to be
//!    non-halos, verifying halo_count = n_cells and halo_count = 0.
//!
//! Both tests are gated `#[ignore = "gpu"]` and require a working Vulkan ICD.
//!
//! Run with:
//!   cargo test -p lbm_vulkan --release dark_halo_vulkan_parity \
//!     -- --ignored --nocapture

use lbm_vulkan::dark_halo_vulkan::{DarkHaloConfig, DarkHaloVulkan};

const NX: usize = 16;
const NY: usize = 16;
const NZ: usize = 16;
const SEED: u32 = 0xDEAD_C0DE;
const K_DIM: u32 = 16;
const TAU_BASE: f32 = 1.0;
const TAU_AMP: f32 = 0.3;
const STEPS: usize = 5;

/// Replicate dark_halo_viscosity.wgsl `pos_hash` + tau formula on CPU.
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
                let mut h: u32 = seed
                    ^ (x as u32).wrapping_mul(73856093)
                    ^ (y as u32).wrapping_mul(19349663)
                    ^ (z as u32).wrapping_mul(83492791);
                h = (h >> 13) ^ h;
                h = h.wrapping_mul(0x5bd1e995);
                h = (h >> 15) ^ h;
                let noise = (h & 0xFFFF) as f32 / 65535.0;
                tau[cell] = tau_base + tau_amp * (lambda * noise).sin();
            }
        }
    }
    tau
}

/// Count cells where the ZD proxy exceeds zd_threshold (CPU reference).
fn cpu_zd_count(tau: &[f32], tau_base: f32, tau_amp: f32, zd_threshold: f32) -> u32 {
    tau.iter()
        .map(|&t| {
            let proxy = if tau_amp > 0.0 {
                (t - tau_base) / tau_amp
            } else {
                0.0
            };
            u32::from(proxy > zd_threshold)
        })
        .sum()
}

#[test]
#[ignore = "gpu (Vulkan ICD + compute-capable adapter required)"]
fn deterministic_zd_count() {
    let n_cells = NX * NY * NZ;
    let zd_threshold = 0.0_f32; // threshold at the midpoint of [-1, 1]

    // CPU reference: deterministic ZD count using the same hash formula.
    let tau_cpu = cpu_tau(NX, NY, NZ, SEED, K_DIM, TAU_BASE, TAU_AMP);
    let expected_count = cpu_zd_count(&tau_cpu, TAU_BASE, TAU_AMP, zd_threshold);

    // GPU path: velocity_epsilon and density_factor set so those criteria
    // trivially pass, leaving only the ZD proxy as the active gate.
    let config = DarkHaloConfig {
        k_dim: K_DIM,
        steps: STEPS,
        seed: SEED,
        tau_base: TAU_BASE,
        tau_amp: TAU_AMP,
        zd_threshold,
        velocity_epsilon: 2.0, // all cells pass (max LBM |u| << 1)
        density_factor: 0.0,   // all cells pass (rho > 0)
    };

    let mut solver = match DarkHaloVulkan::new(NX, NY, NZ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skip: DarkHaloVulkan init failed ({e}); no compute adapter?");
            return;
        }
    };
    let result = solver
        .run(None, &config)
        .expect("dark-halo Vulkan run succeeds");

    // f32 sin() precision may differ by ~1 ulp between host and WGSL,
    // which can shift a small number of cells near the threshold. Allow
    // a tolerance of 1% of n_cells.
    let tolerance = (n_cells / 100).max(4) as u32;
    let diff = (result.halo_count as i64 - expected_count as i64).unsigned_abs() as u32;
    assert!(
        diff <= tolerance,
        "halo_count mismatch: gpu={}, cpu={}, diff={} (tol={})",
        result.halo_count,
        expected_count,
        diff,
        tolerance
    );
    eprintln!(
        "ZD parity OK on {}x{}x{} after {} steps: \
         gpu_halo_count={}, cpu_halo_count={}, diff={}, fraction={:.3}",
        NX, NY, NZ, STEPS, result.halo_count, expected_count, diff, result.halo_fraction
    );
}

#[test]
#[ignore = "gpu (Vulkan ICD + compute-capable adapter required)"]
fn all_pass_all_fail() {
    let n_cells = NX * NY * NZ;

    let mut solver = match DarkHaloVulkan::new(NX, NY, NZ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skip: DarkHaloVulkan init failed ({e}); no compute adapter?");
            return;
        }
    };

    // All cells should be halo candidates: ZD proxy always >= -1 > -2.0,
    // velocity always < 2.0, density always > 0.0 * rho_mean = 0.
    let all_pass_config = DarkHaloConfig {
        k_dim: K_DIM,
        steps: STEPS,
        seed: SEED,
        tau_base: TAU_BASE,
        tau_amp: TAU_AMP,
        zd_threshold: -2.0,
        velocity_epsilon: 2.0,
        density_factor: 0.0,
    };
    let all_pass = solver
        .run(None, &all_pass_config)
        .expect("all-pass run succeeds");
    assert_eq!(
        all_pass.halo_count, n_cells as u32,
        "expected all {} cells to be halos with trivially-passing thresholds, \
         got {}",
        n_cells, all_pass.halo_count
    );
    eprintln!(
        "all-pass: halo_count = {} / {} = 1.000",
        all_pass.halo_count, n_cells
    );

    // No cells should be halo candidates: ZD proxy always <= 1 < 2.0.
    let all_fail_config = DarkHaloConfig {
        k_dim: K_DIM,
        steps: STEPS,
        seed: SEED,
        tau_base: TAU_BASE,
        tau_amp: TAU_AMP,
        zd_threshold: 2.0,
        velocity_epsilon: 2.0,
        density_factor: 0.0,
    };
    let all_fail = solver
        .run(None, &all_fail_config)
        .expect("all-fail run succeeds");
    assert_eq!(
        all_fail.halo_count, 0,
        "expected 0 halo cells with impossibly-high ZD threshold, got {}",
        all_fail.halo_count
    );
    eprintln!("all-fail: halo_count = 0 / {} = 0.000", n_cells);
}
