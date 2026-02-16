use lbm_3d_cuda::{LbmSolver3DCuda, Precision};

fn maybe_solver(precision: Precision) -> Option<LbmSolver3DCuda> {
    match LbmSolver3DCuda::new(4, 4, 4, 0.6, precision) {
        Ok(solver) => Some(solver),
        Err(err) => {
            eprintln!("Skipping GPU integration test: {err}");
            None
        }
    }
}

#[test]
fn gpu_init_and_step_remain_finite_fp32() {
    let Some(mut solver) = maybe_solver(Precision::FP32) else { return };
    solver.step().expect("step should succeed");
    let mean = solver
        .calculate_mean_density()
        .expect("mean density should compute");
    assert!(mean.is_finite());
    assert!(mean > 0.0);
}

#[test]
fn gpu_init_and_step_remain_finite_bf16() {
    let Some(mut solver) = maybe_solver(Precision::BF16) else { return };
    solver.step().expect("step should succeed");
    let mean = solver
        .calculate_mean_density()
        .expect("mean density should compute");
    assert!(mean.is_finite());
    assert!(mean > 0.0);
}
