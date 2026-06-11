//! CPU (`lbm_3d::LbmSolver3D` MRT f64) vs Vulkan `LbmMrtD3Q19Vulkan` (f32)
//! parity test for the D3Q19 MRT collision operator.
//!
//! Seeded from a ChaCha20-derived smooth macroscopic field, evolved for
//! N=10 steps at 16^3, tau=1.0. MRT accumulates more FMA operations than
//! BGK (~722 vs ~57 per cell), so tolerances are wider:
//! abs_tol=2e-3 / rel_tol=2e-2.
//!
//! Gated `#[ignore = "gpu"]`; skipped gracefully when no Vulkan ICD is
//! present.
//!
//! Run with:
//!   cargo test -p lbm_vulkan --release lbm_mrt_d3q19_vulkan_parity -- --ignored --nocapture

use lbm_3d::{lattice::D3Q19Lattice, solver::LbmSolver3D};
use lbm_vulkan::lbm_mrt_d3q19_vulkan::LbmMrtD3Q19Vulkan;
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
const REL_TOL: f64 = 2e-2;
const ABS_TOL: f64 = 2e-3;

// Seed distinct from the BGK parity seeds (0x10110D3_00000019, 0xCBC_0019,
// 0xD3_19_3A_F0CF) to ensure independent initial conditions.
const SEED: u64 = 0xD319_4D52_5400_0000;

#[test]
#[ignore = "gpu (Vulkan ICD required)"]
fn cpu_vs_vulkan_mrt_16cubed_10steps() {
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

    // CPU oracle (f64 MRT).
    let mut cpu = LbmSolver3D::new_mrt(NX, NY, NZ, TAU);
    cpu.rho[..n_cells].copy_from_slice(&rho0[..n_cells]);
    cpu.u[..n_cells].copy_from_slice(&u0[..n_cells]);
    cpu.reinitialize_from_macroscopic();
    cpu.evolve(STEPS);

    // Vulkan path.
    let mut gpu = match LbmMrtD3Q19Vulkan::new(NX, NY, NZ, TAU as f32) {
        Err(e) => {
            eprintln!("skip: Vulkan init failed ({e})");
            return;
        }
        Ok(g) => g,
    };
    gpu.upload_state(&f_init)
        .expect("upload_state matches solver dimensions");
    gpu.evolve(STEPS).expect("Vulkan dispatch succeeds");
    let m = gpu.compute_macroscopic().expect("GPU readback");

    let mut max_rho_err = 0.0_f64;
    let mut max_u_err = 0.0_f64;
    for cell in 0..n_cells {
        let cpu_rho = cpu.rho[cell];
        let cpu_u = cpu.u[cell];
        let gpu_rho = m.rho[cell] as f64;
        let gpu_u = [m.ux[cell] as f64, m.uy[cell] as f64, m.uz[cell] as f64];

        let rho_err = (cpu_rho - gpu_rho).abs();
        let rho_rel = rho_err / cpu_rho.abs().max(1e-12);
        assert!(
            rho_err <= ABS_TOL || rho_rel <= REL_TOL,
            "rho mismatch at cell {cell}: cpu={cpu_rho:.6e}, gpu={gpu_rho:.6e}, \
             abs={rho_err:.3e}, rel={rho_rel:.3e} \
             (abs_tol={ABS_TOL:.1e}, rel_tol={REL_TOL:.1e})"
        );
        max_rho_err = max_rho_err.max(rho_err);

        for axis in 0..3 {
            let u_err = (cpu_u[axis] - gpu_u[axis]).abs();
            let u_rel = u_err / cpu_u[axis].abs().max(1e-12);
            assert!(
                u_err <= ABS_TOL || u_rel <= REL_TOL,
                "u[{axis}] mismatch at cell {cell}: cpu={:.6e}, gpu={:.6e}, \
                 abs={u_err:.3e}, rel={u_rel:.3e} \
                 (abs_tol={ABS_TOL:.1e}, rel_tol={REL_TOL:.1e})",
                cpu_u[axis],
                gpu_u[axis]
            );
            max_u_err = max_u_err.max(u_err);
        }
    }
    eprintln!(
        "CPU-vs-Vulkan-MRT OK on {NX}x{NY}x{NZ} after {STEPS} steps: \
         max_rho_err={max_rho_err:.3e}, max_u_err={max_u_err:.3e}"
    );
}
