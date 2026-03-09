use algebra_experimental::voudon_stabilizer::VoudonStabilizerGpu;
use cudarc::driver::CudaContext;
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("--- Voudon Stabilizer Demo: GPU-Accelerated Topological Search ---");

    if !gororoba_algebra::gpu::is_gpu_available() {
        println!("GPU not available. This example requires CUDA.");
        return Ok(());
    }

    // 1. Search for Stable Voudon Cycles (Associative Triples) on GPU
    let max_triples = 1000;
    println!(
        "Searching for up to {} stable Voudon cycles in 256D manifold...",
        max_triples
    );
    let stable_cycles =
        VoudonStabilizerGpu::find_stable_cycles(max_triples).map_err(|e| anyhow::anyhow!(e))?;

    println!("Found {} stable cycles.", stable_cycles.len());
    if !stable_cycles.is_empty() {
        println!("Example stable triple: {:?}", stable_cycles[0]);
    }

    // 2. Initialize CUDA LBM Solver
    let nx = 64;
    let ny = 64;
    let nz = 64;
    println!(
        "\nInitializing CUDA LBM solver for stability analysis ({}x{}x{})...",
        nx, ny, nz
    );
    let mut solver = LbmSolver3DCuda::new(nx, ny, nz, 0.6, Precision::FP32)?;

    // 3. Map Stable Cycles to LBM Stabilizer Seeds
    // (Simulated: regions with high density of stable cycles suppress turbulence)
    println!("Seeding topological stabilizers into the fluid grid...");

    // Create a mock stabilizer field based on cycle density
    let mut stabilizer_field = vec![0.0f32; nx * ny * nz];
    for (i, _cycle) in stable_cycles.iter().enumerate() {
        let idx = (i * 137) % (nx * ny * nz);
        stabilizer_field[idx] = 1.0;
    }

    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream();
    let d_stabilizer = stream.clone_htod(&stabilizer_field)?;

    // 4. Update tau (relaxation time) to suppress turbulence in stable zones
    // Using the same bridge kernel but with negative alpha to decrease viscosity (tau -> 0.5)
    // Decreasing tau near 0.5 usually increases turbulence, but here we model
    // "algebraic rigidity" where stable cycles force laminar flow.
    // Actually, increasing tau (higher viscosity) is a better proxy for stabilization.
    let alpha_stable = 2.0;
    let tau_base = 0.6;
    println!(
        "Applying topological stabilization (alpha = {})...",
        alpha_stable
    );
    solver.update_tau_from_voudon(&d_stabilizer, tau_base as f32, alpha_stable as f32)?;

    // 5. Run simulation
    println!("Running simulation steps...");
    for step in 1..=50 {
        solver.step()?;
        if step % 10 == 0 {
            println!("  Step {}/50 complete", step);
        }
    }

    println!("\n--- Demo Complete ---");
    println!("Topological Voudon stabilizers successfully integrated into CUDA LBM.");

    Ok(())
}
