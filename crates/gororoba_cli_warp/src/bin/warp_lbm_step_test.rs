use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let res: usize = 128;
    let tau: f64 = 0.6;
    let prec = Precision::FP32;
    let steps: usize = 100;

    println!("Initializing LbmSolver3DCuda for {} FP32...", res);
    let mut solver = LbmSolver3DCuda::new(res, res, res, tau, prec)?;
    println!("Solver initialized. Running {} steps...", steps);

    let start = Instant::now();
    for _ in 0..steps {
        solver.step()?;
    }
    let elapsed = start.elapsed();

    println!("{} steps took: {:?}", steps, elapsed);
    println!("LBM_STEP_TEST_COMPLETE");

    Ok(())
}
