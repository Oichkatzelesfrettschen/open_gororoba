use lbm_3d::solver::LbmSolver3D;
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};

fn rms_error_f64_f32(a: &[f64], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - *y as f64).powi(2))
        .sum();
    (sum_sq / a.len() as f64).sqrt()
}

fn rms_error_velocity(a: &[[f64; 3]], b: &[[f32; 3]]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(va, vb)| {
            (va[0] - vb[0] as f64).powi(2)
                + (va[1] - vb[1] as f64).powi(2)
                + (va[2] - vb[2] as f64).powi(2)
        })
        .sum();
    (sum_sq / (3 * a.len()) as f64).sqrt()
}

#[test]
fn equilibrium_cpu_gpu_remain_consistent_fp32() {
    let (nx, ny, nz) = (8, 8, 8);
    let steps = 5usize;
    let tau = 0.6f64;

    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, tau);
    cpu_solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);
    cpu_solver.evolve(steps);

    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, tau, Precision::FP32) {
        Ok(solver) => solver,
        Err(err) => {
            eprintln!("Skipping GPU integration test: {err}");
            return;
        }
    };
    for _ in 0..steps {
        gpu_solver.step().expect("GPU step should succeed");
    }
    gpu_solver
        .sync_to_host()
        .expect("GPU host sync should succeed");

    let rho_err = rms_error_f64_f32(&cpu_solver.rho, &gpu_solver.rho);
    let u_err = rms_error_velocity(&cpu_solver.u, &gpu_solver.u);

    assert!(rho_err.is_finite());
    assert!(u_err.is_finite());
    assert!(rho_err < 1.0e-2, "rho rms error too large: {rho_err}");
    assert!(u_err < 1.0e-2, "u rms error too large: {u_err}");
}
