//! Parity test: CPU `lbm_3d::LbmSolver3D` (f64 BGK) vs Vulkan
//! `LbmD3Q19Vulkan` (f32 fused PUSH BGK) on a 16x16x16 periodic grid
//! for 10 timesteps.
//!
//! Strategy:
//!   1. Generate a smooth, non-trivial macroscopic field on the host
//!      (rho_0(x) and u_0(x)) with a ChaCha20-seeded PRNG.
//!   2. Build the BGK equilibrium f_eq(rho, u) for both solvers from
//!      the SAME macroscopic field, so the initial f arrays carry
//!      bit-identical macroscopic content modulo f64 -> f32 rounding.
//!   3. Evolve both for N=10 steps with the same tau.
//!   4. Compare per-cell (rho, u_x, u_y, u_z) within an f32-mixed-with-
//!      f64 tolerance (relative + absolute fall-through).
//!
//! Gated `#[ignore = "gpu"]` because the test needs a working Vulkan
//! ICD + compute-capable adapter. CI without GPU support runs with
//! `cargo test -- --ignored` opt-in.
//!
//! Run with:
//!   cargo test -p lbm_vulkan --release lbm_d3q19_vulkan_parity \
//!     -- --ignored --nocapture

use lbm_3d::lattice::D3Q19Lattice;
use lbm_3d::solver::LbmSolver3D;
use lbm_vulkan::lbm_d3q19_vulkan::LbmD3Q19Vulkan;
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand_chacha::ChaCha20Rng;

const NX: usize = 16;
const NY: usize = 16;
const NZ: usize = 16;
const TAU: f64 = 1.0;
const STEPS: usize = 10;
const REL_TOL: f64 = 5e-3;
const ABS_TOL: f64 = 5e-4;

#[test]
#[ignore = "gpu (Vulkan ICD + compute-capable adapter required)"]
fn cpu_vs_vulkan_parity_16cubed_10steps() {
    // 1. Build a smooth, non-trivial initial (rho, u) field.
    let mut rng = ChaCha20Rng::seed_from_u64(0x10110D3_00000019);
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

    // 2a. Seed the CPU solver: initialize_uniform(1, 0) lays down
    //     equilibrium f at rest, then write rho/u and call
    //     reinitialize_from_macroscopic to project f back onto the
    //     equilibrium manifold for the requested (rho, u).
    let mut cpu = LbmSolver3D::new(NX, NY, NZ, TAU);
    cpu.initialize_uniform(1.0, [0.0, 0.0, 0.0]);
    cpu.rho[..n_cells].copy_from_slice(&rho0[..n_cells]);
    cpu.u[..n_cells].copy_from_slice(&u0[..n_cells]);
    cpu.reinitialize_from_macroscopic();

    // 2b. Build the same f_eq on the host with f32 arithmetic and
    //     upload to the GPU. SoA layout matches the WGSL kernel.
    let lattice = D3Q19Lattice::new();
    let mut f_gpu = vec![0.0_f32; n_cells * 19];
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
            let f_eq =
                w_i * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
            f_gpu[i * n_cells + cell] = f_eq;
        }
    }

    let mut gpu = match LbmD3Q19Vulkan::new(NX, NY, NZ, TAU as f32) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: Vulkan solver init failed ({e}); no compute adapter?");
            return;
        }
    };
    gpu.upload_state(&f_gpu)
        .expect("upload_state matches solver dimensions");

    // 3. Evolve.
    cpu.evolve(STEPS);
    gpu.evolve(STEPS).expect("Vulkan dispatch succeeds");

    // 4. Compare macroscopic moments.
    let gpu_macro = gpu.compute_macroscopic().expect("GPU readback");
    let mut max_rho_err = 0.0_f64;
    let mut max_u_err = 0.0_f64;
    for cell in 0..n_cells {
        let cpu_rho = cpu.rho[cell];
        let gpu_rho = gpu_macro.rho[cell] as f64;
        let cpu_u = cpu.u[cell];
        let gpu_u = [
            gpu_macro.ux[cell] as f64,
            gpu_macro.uy[cell] as f64,
            gpu_macro.uz[cell] as f64,
        ];
        let rho_err = (cpu_rho - gpu_rho).abs();
        let rho_rel = rho_err / cpu_rho.abs().max(1e-12);
        let pass_rho = rho_err <= ABS_TOL || rho_rel <= REL_TOL;
        assert!(
            pass_rho,
            "rho mismatch at cell {cell}: cpu={cpu_rho:.6e}, gpu={gpu_rho:.6e}, \
             abs_err={rho_err:.3e}, rel_err={rho_rel:.3e} (abs_tol={ABS_TOL:.1e}, \
             rel_tol={REL_TOL:.1e})"
        );
        max_rho_err = max_rho_err.max(rho_err);

        for axis in 0..3 {
            let u_err = (cpu_u[axis] - gpu_u[axis]).abs();
            let u_rel = u_err / cpu_u[axis].abs().max(1e-12);
            let pass_u = u_err <= ABS_TOL || u_rel <= REL_TOL;
            assert!(
                pass_u,
                "u[{axis}] mismatch at cell {cell}: cpu={:.6e}, gpu={:.6e}, \
                 abs_err={u_err:.3e}, rel_err={u_rel:.3e} (abs_tol={ABS_TOL:.1e}, \
                 rel_tol={REL_TOL:.1e})",
                cpu_u[axis], gpu_u[axis]
            );
            max_u_err = max_u_err.max(u_err);
        }
    }
    eprintln!(
        "PARITY OK on {}x{}x{} after {} steps: max_rho_err={max_rho_err:.3e}, \
         max_u_err={max_u_err:.3e}",
        NX, NY, NZ, STEPS
    );
}
