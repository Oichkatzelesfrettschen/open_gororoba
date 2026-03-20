use cudarc::driver::CudaContext;
use gororoba_algebra::gpu::voudon::Cd256FrustrationKernel;
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("--- Voudon-LBM Bridge: CUDA-Accelerated 256D Turbulence ---");

    if !gororoba_algebra::gpu::is_gpu_available() {
        println!("GPU not available. This example requires CUDA.");
        return Ok(());
    }

    let nx = 128;
    let ny = 128;
    let nz = 128;
    let n_cells = nx * ny * nz;

    // 1. Compute Voudon (256D) Frustration Field on GPU
    println!(
        "Computing 256D Voudon frustration field ({} cells)...",
        n_cells
    );
    let frustration_host =
        Cd256FrustrationKernel::compute_field(nx, ny, nz, 42).map_err(|e| anyhow::anyhow!(e))?;

    // 2. Initialize CUDA LBM Solver
    println!("Initializing CUDA LBM solver (FP32)...");
    let mut solver = LbmSolver3DCuda::new(nx, ny, nz, 0.8, Precision::FP32)?;

    // 3. Upload Frustration Field to GPU for LBM modulation
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream();
    let d_frustration = stream.clone_htod(&frustration_host)?;

    // 4. Modulate Viscosity via Voudon Frustration
    let alpha_voudon = 0.5; // Coupling strength
    let tau_base = 0.8;
    println!(
        "Applying Voudon-driven viscosity modulation (alpha = {})...",
        alpha_voudon
    );
    solver.update_tau_from_voudon(&d_frustration, tau_base as f32, alpha_voudon as f32)?;

    // 5. Run Simulation Steps
    let n_steps = 100;
    println!("Running {} simulation steps...", n_steps);
    for step in 1..=n_steps {
        solver.step()?;
        if step % 20 == 0 {
            println!("  Step {}/{}", step, n_steps);
        }
    }

    println!("\n--- Simulation Complete ---");
    println!("Voudon-coupled turbulence successfully simulated on GPU.");

    Ok(())
}
