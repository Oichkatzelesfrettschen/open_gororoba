use algebra_core::gpu::voudon::VoudonFrustrationGpu;
use algebra_analysis::phase_transition::{PhaseTransitionAnalyzer, AlgebraicPhase};
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use cudarc::driver::CudaContext;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("--- 256D Algebraic Phase Transition Analysis: GPU-Accelerated ---");

    if !algebra_core::gpu::is_gpu_available() {
        println!("GPU not available. This example requires CUDA.");
        return Ok(());
    }

    let nx = 64;
    let ny = 64;
    let nz = 64;
    let n_cells = nx * ny * nz;

    // 1. Initialize Analyzer with critical threshold phi_c = 0.25
    let analyzer = PhaseTransitionAnalyzer::with_critical_density(256, 0.25);
    println!("Analyzer initialized: Critical defect density threshold = 0.25");

    // 2. Compute 256D Non-Associativity Density Field on GPU
    println!("Computing 256D non-associativity density (frustration) field...");
    let frustration_host = VoudonFrustrationGpu::compute_field(nx, ny, nz, 42)
        .map_err(|e| anyhow::anyhow!(e))?;

    // 3. Statistical Analysis of Phases
    let mut coherent_cells = 0;
    let mut total_order_param = 0.0;

    for &density in &frustration_host {
        let phase = analyzer.evaluate_phase(density as f64);
        if phase == AlgebraicPhase::Coherent {
            coherent_cells += 1;
        }
        total_order_param += analyzer.calculate_order_parameter(density as f64);
    }

    let coherence_fraction = coherent_cells as f64 / n_cells as f64;
    let mean_order_param = total_order_param / n_cells as f64;

    println!("\nStatistical Summary:");
    println!("  Coherent fraction: {:.2}%", coherence_fraction * 100.0);
    println!("  Mean order parameter: {:.4}", mean_order_param);
    println!("  Dominant phase: {}", if coherence_fraction > 0.5 { "Coherent (Superfluid-like)" } else { "Dissipative" });

    // 4. LBM Modulation Demo
    println!("\nInitializing CUDA LBM for phase-modulated flow...");
    let mut solver = LbmSolver3DCuda::new(nx, ny, nz, 0.8, Precision::FP32)?;
    
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream();
    let d_frustration = stream.clone_htod(&frustration_host)?;

    // Coupling: Coherent regions (low frustration) maintain low viscosity.
    // Dissipative regions (high frustration) increase viscosity.
    let coupling_strength = 1.0;
    solver.update_tau_from_voudon(&d_frustration, 0.8, coupling_strength)?;

    println!("Running LBM steps with phase modulation...");
    for step in 1..=50 {
        solver.step()?;
        if step % 10 == 0 {
            println!("  Step {}/50 complete", step);
        }
    }

    println!("\n--- Analysis Complete ---");
    println!("Phase transition successfully mapped from 256D algebra to physical viscosity.");

    Ok(())
}
