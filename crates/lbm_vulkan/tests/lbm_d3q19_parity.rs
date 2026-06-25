//! 3-way parity test: CPU `lbm_3d::LbmSolver3D` (f64 BGK) vs Vulkan
//! `LbmD3Q19Vulkan` (f32 WGSL PUSH BGK) vs cubecl-wgpu
//! `evolve_d3q19_cubecl` (f32 cubecl PUSH BGK).
//!
//! All three solvers are seeded from the same ChaCha20-derived smooth
//! macroscopic field projected onto the BGK equilibrium manifold, evolved
//! for N=10 steps at 16^3, tau=1.0, and compared cell-by-cell (rho, ux,
//! uy, uz) within abs_tol=5e-4 / rel_tol=5e-3.
//!
//! Gated `#[ignore = "gpu"]` because the test needs a working Vulkan ICD and a
//! cubecl-wgpu adapter.  Each backend failing at init is logged and skipped
//! individually.  If both skip, the test returns after logging the soft skip so
//! headless hosts can run the ignored test suite without a false failure.
//!
//! Run with:
//!   cargo test -p lbm_vulkan --features cubecl --release \
//!     lbm_d3q19_parity -- --ignored --nocapture

#![cfg(feature = "cubecl")]

use lbm_3d::{lattice::D3Q19Lattice, solver::LbmSolver3D};
use lbm_vulkan::{
    lbm_d3q19_cubecl::{evolve_d3q19_cubecl, is_available},
    lbm_d3q19_vulkan::LbmD3Q19Vulkan,
};
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
};
use rand_chacha::ChaCha20Rng;

const NX: usize = 16;
const NY: usize = 16;
const NZ: usize = 16;
const TAU: f64 = 1.0;
const STEPS: usize = 10;
const REL_TOL: f64 = 5e-3;
const ABS_TOL: f64 = 5e-4;

// Seed chosen to differ from the two per-backend parity seeds
// (0x10110D3_00000019 for Vulkan, 0xCBC_0019 for cubecl).
const SEED: u64 = 0x00D3_193A_F0CF;

#[test]
#[ignore = "gpu (Vulkan ICD + cubecl-wgpu adapter required)"]
fn three_way_parity_16cubed_10steps() {
    // 1. Smooth, non-trivial initial macroscopic field.
    let mut rng = ChaCha20Rng::seed_from_u64(SEED);
    let rho_dist = Uniform::new(0.95_f64, 1.05_f64).expect("rho range valid");
    let u_dist = Uniform::new(-0.05_f64, 0.05_f64).expect("u range valid");
    let n_cells = NX * NY * NZ;
    let mut rho0 = vec![0.0_f64; n_cells];
    let mut u0 = vec![[0.0_f64; 3]; n_cells];
    for cell in 0..n_cells {
        rho0[cell] = rho_dist.sample(&mut rng);
        u0[cell] = [
            u_dist.sample(&mut rng),
            u_dist.sample(&mut rng),
            u_dist.sample(&mut rng),
        ];
    }

    // 2. Build the f32 BGK equilibrium once; all three paths share it
    //    (CPU path gets the f64 version via reinitialize_from_macroscopic).
    let lattice = D3Q19Lattice::new();
    let mut f_init = vec![0.0_f32; n_cells * 19];
    for cell in 0..n_cells {
        let rho = rho0[cell] as f32;
        let ux = u0[cell][0] as f32;
        let uy = u0[cell][1] as f32;
        let uz = u0[cell][2] as f32;
        let u_sq = ux * ux + uy * uy + uz * uz;
        for i in 0..19 {
            let c = lattice.velocity(i);
            let cu = (c[0] as f32) * ux + (c[1] as f32) * uy + (c[2] as f32) * uz;
            let w_i = lattice.weight(i) as f32;
            f_init[i * n_cells + cell] = w_i * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
        }
    }

    // 3. CPU oracle.
    let mut cpu = LbmSolver3D::new(NX, NY, NZ, TAU);
    cpu.initialize_uniform(1.0, [0.0, 0.0, 0.0]);
    cpu.rho[..n_cells].copy_from_slice(&rho0[..n_cells]);
    cpu.u[..n_cells].copy_from_slice(&u0[..n_cells]);
    cpu.reinitialize_from_macroscopic();
    cpu.evolve(STEPS);

    // 4. Vulkan path.
    let vulkan_result: Option<Vec<(f64, [f64; 3])>> =
        match LbmD3Q19Vulkan::new(NX, NY, NZ, TAU as f32) {
            Err(e) => {
                eprintln!("skip Vulkan path: init failed ({e})");
                None
            }
            Ok(mut gpu) => {
                gpu.upload_state(&f_init)
                    .expect("upload_state matches solver dimensions");
                gpu.evolve(STEPS).expect("Vulkan dispatch succeeds");
                let m = gpu.compute_macroscopic().expect("GPU readback");
                Some(
                    (0..n_cells)
                        .map(|c| {
                            (
                                m.rho[c] as f64,
                                [m.ux[c] as f64, m.uy[c] as f64, m.uz[c] as f64],
                            )
                        })
                        .collect(),
                )
            }
        };

    // 5. cubecl path.
    let cubecl_result: Option<Vec<(f64, [f64; 3])>> = if !is_available() {
        eprintln!("skip cubecl path: wgpu adapter not available");
        None
    } else {
        match evolve_d3q19_cubecl(NX, NY, NZ, TAU as f32, &f_init, STEPS) {
            Err(e) => {
                eprintln!("skip cubecl path: evolve failed ({e})");
                None
            }
            Ok(f_out) => Some(
                (0..n_cells)
                    .map(|cell| {
                        let mut r = 0.0_f32;
                        for i in 0..19 {
                            r += f_out[i * n_cells + cell];
                        }
                        let inv_r = 1.0 / r;
                        let g = |i: usize| f_out[i * n_cells + cell];
                        let mx = g(1) - g(2) + g(7) - g(8) + g(9) - g(10) + g(11) - g(12) + g(13)
                            - g(14);
                        let my = g(3) - g(4) + g(7) - g(8) - g(9) + g(10) + g(15) - g(16) + g(17)
                            - g(18);
                        let mz =
                            g(5) - g(6) + g(11) - g(12) - g(13) + g(14) + g(15) - g(16) - g(17)
                                + g(18);
                        (
                            r as f64,
                            [
                                (mx * inv_r) as f64,
                                (my * inv_r) as f64,
                                (mz * inv_r) as f64,
                            ],
                        )
                    })
                    .collect(),
            ),
        }
    };

    // 6. When neither GPU path is available, emit a diagnostic and return; this
    // mirrors the soft-skip contract of the per-backend parity tests so that
    // `cargo test -- --ignored` on a headless host does not fail the suite.
    if vulkan_result.is_none() && cubecl_result.is_none() {
        eprintln!("skip: neither Vulkan nor cubecl backend available; 3-way parity test skipped");
        return;
    }

    // 7. Compare each available GPU path against the CPU oracle.
    let mut max_rho_err_v = 0.0_f64;
    let mut max_u_err_v = 0.0_f64;
    let mut max_rho_err_c = 0.0_f64;
    let mut max_u_err_c = 0.0_f64;

    for cell in 0..n_cells {
        let cpu_rho = cpu.rho[cell];
        let cpu_u = cpu.u[cell];

        if let Some(ref vr) = vulkan_result {
            let (gpu_rho, gpu_u) = vr[cell];
            check_cell(
                "Vulkan",
                cell,
                cpu_rho,
                cpu_u,
                gpu_rho,
                gpu_u,
                &mut max_rho_err_v,
                &mut max_u_err_v,
            );
        }

        if let Some(ref cr) = cubecl_result {
            let (gpu_rho, gpu_u) = cr[cell];
            check_cell(
                "cubecl",
                cell,
                cpu_rho,
                cpu_u,
                gpu_rho,
                gpu_u,
                &mut max_rho_err_c,
                &mut max_u_err_c,
            );
        }
    }

    // 8. If both paths ran, check that Vulkan and cubecl agree with each other.
    if let (Some(vr), Some(cr)) = (&vulkan_result, &cubecl_result) {
        let mut max_vv_rho = 0.0_f64;
        let mut max_vv_u = 0.0_f64;
        for cell in 0..n_cells {
            let (v_rho, v_u) = vr[cell];
            let (c_rho, c_u) = cr[cell];
            check_cell(
                "Vulkan-vs-cubecl",
                cell,
                v_rho,
                v_u,
                c_rho,
                c_u,
                &mut max_vv_rho,
                &mut max_vv_u,
            );
        }
        eprintln!("Vulkan-vs-cubecl: max_rho_err={max_vv_rho:.3e}, max_u_err={max_vv_u:.3e}");
    }

    if vulkan_result.is_some() {
        eprintln!(
            "CPU-vs-Vulkan OK on {}x{}x{} after {} steps: \
             max_rho_err={max_rho_err_v:.3e}, max_u_err={max_u_err_v:.3e}",
            NX, NY, NZ, STEPS
        );
    }
    if cubecl_result.is_some() {
        eprintln!(
            "CPU-vs-cubecl OK on {}x{}x{} after {} steps: \
             max_rho_err={max_rho_err_c:.3e}, max_u_err={max_u_err_c:.3e}",
            NX, NY, NZ, STEPS
        );
    }
}

#[allow(clippy::too_many_arguments)] // parity helper compares rho, velocity components, and per-channel values per cell
fn check_cell(
    label: &str,
    cell: usize,
    cpu_rho: f64,
    cpu_u: [f64; 3],
    gpu_rho: f64,
    gpu_u: [f64; 3],
    max_rho_err: &mut f64,
    max_u_err: &mut f64,
) {
    let rho_err = (cpu_rho - gpu_rho).abs();
    let rho_rel = rho_err / cpu_rho.abs().max(1e-12);
    assert!(
        rho_err <= ABS_TOL || rho_rel <= REL_TOL,
        "{label} rho mismatch at cell {cell}: cpu={cpu_rho:.6e}, gpu={gpu_rho:.6e}, \
         abs_err={rho_err:.3e}, rel_err={rho_rel:.3e} \
         (abs_tol={ABS_TOL:.1e}, rel_tol={REL_TOL:.1e})"
    );
    *max_rho_err = max_rho_err.max(rho_err);

    for axis in 0..3 {
        let u_err = (cpu_u[axis] - gpu_u[axis]).abs();
        let u_rel = u_err / cpu_u[axis].abs().max(1e-12);
        assert!(
            u_err <= ABS_TOL || u_rel <= REL_TOL,
            "{label} u[{axis}] mismatch at cell {cell}: cpu={:.6e}, gpu={:.6e}, \
             abs_err={u_err:.3e}, rel_err={u_rel:.3e} \
             (abs_tol={ABS_TOL:.1e}, rel_tol={REL_TOL:.1e})",
            cpu_u[axis],
            gpu_u[axis]
        );
        *max_u_err = max_u_err.max(u_err);
    }
}
