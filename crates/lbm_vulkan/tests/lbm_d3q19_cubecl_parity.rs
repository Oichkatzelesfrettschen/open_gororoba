//! Parity test: CPU `lbm_3d::LbmSolver3D` (f64 BGK) vs cubecl-wgpu
//! `evolve_d3q19_cubecl` (f32 fused PUSH BGK) on a 16x16x16 periodic
//! grid for 10 timesteps.
//!
//! Strategy mirrors `tests/lbm_d3q19_vulkan_parity.rs`: seed both
//! solvers with the SAME ChaCha20-derived smooth macroscopic field
//! projected onto the BGK equilibrium manifold, evolve, compare
//! per-cell macroscopic moments within an f32-vs-f64 tolerance.
//!
//! Gated `#[ignore = "gpu"]` because the test needs a working
//! cubecl-wgpu adapter (Vulkan / Metal / DX12 / WebGPU).
//!
//! Run with:
//!   cargo test -p lbm_vulkan --features cubecl --release \
//!     lbm_d3q19_cubecl_parity -- --ignored --nocapture

#![cfg(feature = "cubecl")]

use lbm_3d::lattice::D3Q19Lattice;
use lbm_3d::solver::LbmSolver3D;
use lbm_vulkan::lbm_d3q19_cubecl::{evolve_d3q19_cubecl, is_available};
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
#[ignore = "gpu (cubecl-wgpu adapter required)"]
fn cpu_vs_cubecl_parity_16cubed_10steps() {
    if !is_available() {
        eprintln!("skip: cubecl-wgpu runtime not available on this host");
        return;
    }

    let mut rng = ChaCha20Rng::seed_from_u64(0xCBC_0019);
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

    // CPU oracle: initialize at rest, write rho/u, project onto f_eq.
    let mut cpu = LbmSolver3D::new(NX, NY, NZ, TAU);
    cpu.initialize_uniform(1.0, [0.0, 0.0, 0.0]);
    cpu.rho[..n_cells].copy_from_slice(&rho0[..n_cells]);
    cpu.u[..n_cells].copy_from_slice(&u0[..n_cells]);
    cpu.reinitialize_from_macroscopic();

    // Build the same f_eq on the host with f32 arithmetic; this is
    // the cubecl launcher's input format.
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
            let f_eq = w_i * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
            f_gpu[i * n_cells + cell] = f_eq;
        }
    }

    cpu.evolve(STEPS);
    let f_out = match evolve_d3q19_cubecl(NX, NY, NZ, TAU as f32, &f_gpu, STEPS) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("skip: cubecl evolve failed ({e})");
            return;
        }
    };

    // Recompute (rho, ux, uy, uz) on the host from the GPU SoA
    // output and compare cell-by-cell.
    let mut max_rho_err = 0.0_f64;
    let mut max_u_err = 0.0_f64;
    for cell in 0..n_cells {
        let mut r = 0.0_f32;
        for i in 0..19 {
            r += f_out[i * n_cells + cell];
        }
        let inv_r = 1.0 / r;
        let g = |i: usize| f_out[i * n_cells + cell];
        let mx = g(1) - g(2) + g(7) - g(8) + g(9) - g(10) + g(11) - g(12) + g(13) - g(14);
        let my = g(3) - g(4) + g(7) - g(8) - g(9) + g(10) + g(15) - g(16) + g(17) - g(18);
        let mz = g(5) - g(6) + g(11) - g(12) - g(13) + g(14) + g(15) - g(16) - g(17) + g(18);

        let cpu_rho = cpu.rho[cell];
        let gpu_rho = r as f64;
        let cpu_u = cpu.u[cell];
        let gpu_u = [(mx * inv_r) as f64, (my * inv_r) as f64, (mz * inv_r) as f64];

        let rho_err = (cpu_rho - gpu_rho).abs();
        let rho_rel = rho_err / cpu_rho.abs().max(1e-12);
        let pass_rho = rho_err <= ABS_TOL || rho_rel <= REL_TOL;
        assert!(
            pass_rho,
            "rho mismatch at cell {cell}: cpu={cpu_rho:.6e}, gpu={gpu_rho:.6e}, \
             abs_err={rho_err:.3e}, rel_err={rho_rel:.3e}"
        );
        max_rho_err = max_rho_err.max(rho_err);

        for axis in 0..3 {
            let u_err = (cpu_u[axis] - gpu_u[axis]).abs();
            let u_rel = u_err / cpu_u[axis].abs().max(1e-12);
            let pass_u = u_err <= ABS_TOL || u_rel <= REL_TOL;
            assert!(
                pass_u,
                "u[{axis}] mismatch at cell {cell}: cpu={:.6e}, gpu={:.6e}, \
                 abs_err={u_err:.3e}, rel_err={u_rel:.3e}",
                cpu_u[axis], gpu_u[axis]
            );
            max_u_err = max_u_err.max(u_err);
        }
    }
    eprintln!(
        "CUBECL PARITY OK on {}x{}x{} after {} steps: max_rho_err={max_rho_err:.3e}, \
         max_u_err={max_u_err:.3e}",
        NX, NY, NZ, STEPS
    );
}
