//! Debug test to isolate GPU initialization issue

use lbm_3d::solver::LbmSolver3D;
use lbm_3d_cuda::LbmSolver3DCuda;

#[test]
fn test_debug_initialization() {
    let (nx, ny, nz) = (4, 4, 4); // Small grid for debugging
    let tau = 1.0;
    let rho_init = 1.5;
    let u_init = [0.01, 0.0, 0.0];

    // CPU solver
    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, tau);
    cpu_solver.initialize_uniform(rho_init, u_init);

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, tau) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu_solver.initialize_uniform(rho_init, u_init).unwrap();
    gpu_solver.sync_to_host().unwrap();

    // Compare INITIAL state (before any evolution)
    println!("\n=== INITIAL STATE COMPARISON ===");
    println!("First 5 cells:");
    for i in 0..5 {
        println!(
            "Cell {}: CPU rho={:.6}, GPU rho={:.6}, CPU u={:?}, GPU u={:?}",
            i, cpu_solver.rho[i], gpu_solver.rho[i], cpu_solver.u[i], gpu_solver.u[i]
        );
    }

    // Check if initialization is correct
    for i in 0..cpu_solver.rho.len() {
        let rho_err = (cpu_solver.rho[i] - gpu_solver.rho[i]).abs();
        let u_err = ((cpu_solver.u[i][0] - gpu_solver.u[i][0]).powi(2)
            + (cpu_solver.u[i][1] - gpu_solver.u[i][1]).powi(2)
            + (cpu_solver.u[i][2] - gpu_solver.u[i][2]).powi(2))
        .sqrt();

        if rho_err > 1e-10 || u_err > 1e-10 {
            println!(
                "Cell {} BAD: rho_err={:.3e}, u_err={:.3e}",
                i, rho_err, u_err
            );
        }
    }

    // Now try ONE step
    println!("\n=== AFTER ONE STEP ===");
    cpu_solver.evolve(1);
    gpu_solver.evolve(1).unwrap();

    println!("First 5 cells:");
    for i in 0..5 {
        println!(
            "Cell {}: CPU rho={:.6}, GPU rho={:.6}, CPU u={:?}, GPU u={:?}",
            i, cpu_solver.rho[i], gpu_solver.rho[i], cpu_solver.u[i], gpu_solver.u[i]
        );
    }

    // Check mass conservation
    let cpu_mass: f64 = cpu_solver.rho.iter().sum();
    let gpu_mass: f64 = gpu_solver.rho.iter().sum();
    println!(
        "\nMass: CPU={:.6}, GPU={:.6}, err={:.3e}",
        cpu_mass,
        gpu_mass,
        (cpu_mass - gpu_mass).abs()
    );
}
