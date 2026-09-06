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

fn evolving_cpu_fixture() -> (LbmSolver3D, Vec<f64>, Vec<[f64; 3]>) {
    let (nx, ny, nz) = (8, 8, 8);
    let mut cpu = LbmSolver3D::new(nx, ny, nz, 0.8);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let index = z * nx * ny + y * nx + x;
                cpu.rho[index] = 1.0 + 0.03 * (std::f64::consts::TAU * x as f64 / nx as f64).cos();
                cpu.u[index] = [
                    0.02 * (std::f64::consts::TAU * y as f64 / ny as f64).sin(),
                    0.0,
                    0.0,
                ];
            }
        }
    }
    cpu.reinitialize_from_macroscopic();
    let density = cpu.rho.clone();
    let velocity = cpu.u.clone();
    (cpu, density, velocity)
}

#[test]
fn nonuniform_oracle_rejects_frozen_evolution() {
    let (mut cpu, density, _) = evolving_cpu_fixture();
    cpu.evolve(12);
    let change = (cpu
        .rho
        .iter()
        .zip(density)
        .map(|(final_value, initial)| (final_value - initial).powi(2))
        .sum::<f64>()
        / cpu.rho.len() as f64)
        .sqrt();
    assert!(
        change > 0.001,
        "fixture must evolve beyond the FP32 comparison tolerance"
    );
}

#[test]
#[ignore = "requires admitted CUDA hardware; run explicitly with --ignored"]
fn evolving_cpu_gpu_and_captured_steps_remain_consistent_fp32() {
    let (mut cpu_solver, density, velocity) = evolving_cpu_fixture();
    let steps = 12;
    cpu_solver.evolve(steps - 1);
    let observed_density = cpu_solver.rho.clone();
    let observed_velocity = cpu_solver.u.clone();
    cpu_solver.evolve(1);
    for captured in [false, true] {
        let mut gpu_solver =
            LbmSolver3DCuda::new_capture_capable(8, 8, 8, 0.8, Precision::FP32, false)
                .expect("CUDA hardware admission or solver initialization failed");
        gpu_solver
            .initialize_custom(&density, &velocity)
            .expect("GPU fixture initialization");
        let mut counter = lbm_3d_cuda::box_counting_gpu::GpuBoxCounter::new_with_stream(
            gpu_solver.context(),
            gpu_solver.stream().clone(),
        )
        .expect("same-stream box counter");
        counter
            .fractal_dimension_device_auto(gpu_solver.d_rho_bytes(), 8, 8, 8)
            .expect("initial same-stream density measurement");
        if captured {
            gpu_solver.step_n(steps).expect("captured GPU steps");
        } else {
            for _ in 0..steps {
                gpu_solver.step().expect("GPU step");
            }
        }
        gpu_solver.sync_to_host().expect("GPU host sync");
        counter
            .fractal_dimension_device_auto(gpu_solver.d_rho_bytes(), 8, 8, 8)
            .expect("final same-stream density measurement");
        let populations = gpu_solver
            .read_populations_fp32()
            .expect("population oracle");
        let cpu_populations: Vec<f64> = (0..density.len())
            .flat_map(|cell| {
                let solver = &cpu_solver;
                (0..19).map(move |direction| solver.f[lbm_3d::solver::aosoa_idx(cell, direction)])
            })
            .collect();
        let population_error = rms_error_f64_f32(&cpu_populations, &populations);
        let post_stream_density: Vec<f32> = populations
            .chunks_exact(19)
            .map(|cell| cell.iter().sum())
            .collect();
        let post_stream_density_error = rms_error_f64_f32(&cpu_solver.rho, &post_stream_density);
        let rho_err = rms_error_f64_f32(&observed_density, &gpu_solver.rho);
        let u_err = rms_error_velocity(&observed_velocity, &gpu_solver.u);
        println!(
            "captured={captured} population_rms={population_error} post_stream_density_rms={post_stream_density_error} density_rms={rho_err} velocity_rms={u_err}"
        );
        assert!(population_error.is_finite() && population_error < 2e-5);
        assert!(post_stream_density_error.is_finite() && post_stream_density_error < 2e-5);
        assert!(
            rho_err.is_finite() && rho_err < 2e-5,
            "density RMS={rho_err}, captured={captured}"
        );
        assert!(
            u_err.is_finite() && u_err < 2e-5,
            "velocity RMS={u_err}, captured={captured}"
        );
    }
}

#[test]
#[ignore = "requires admitted CUDA hardware; run explicitly with --ignored"]
fn captured_force_replacement_matches_uncaptured_steps() {
    let (_, density, velocity) = evolving_cpu_fixture();
    let mut captured = LbmSolver3DCuda::new_capture_capable(8, 8, 8, 0.8, Precision::FP32, false)
        .expect("CUDA admission");
    let mut direct = LbmSolver3DCuda::new_capture_capable(8, 8, 8, 0.8, Precision::FP32, false)
        .expect("CUDA admission");
    captured.initialize_custom(&density, &velocity).unwrap();
    direct.initialize_custom(&density, &velocity).unwrap();
    captured.step_n(4).unwrap();
    for _ in 0..4 {
        direct.step().unwrap();
    }
    let force = vec![[1e-4, 0.0, 0.0]; density.len()];
    captured.set_force_field(&force).unwrap();
    direct.set_force_field(&force).unwrap();
    captured.step_n(5).unwrap();
    for _ in 0..5 {
        direct.step().unwrap();
    }
    captured.sync_to_host().unwrap();
    direct.sync_to_host().unwrap();
    for (left, right) in captured.u.iter().flatten().zip(direct.u.iter().flatten()) {
        assert!(
            (left - right).abs() < 2e-6,
            "force replacement changed captured evolution"
        );
    }
    // Reinitialize after odd stepping so the cached graph's buffer parity differs.
    captured.initialize_custom(&density, &velocity).unwrap();
    direct.initialize_custom(&density, &velocity).unwrap();
    captured.step_n(4).unwrap();
    for _ in 0..4 {
        direct.step().unwrap();
    }
    captured.sync_to_host().unwrap();
    direct.sync_to_host().unwrap();
    for (left, right) in captured.rho.iter().zip(&direct.rho) {
        assert!(
            (left - right).abs() < 2e-6,
            "reinitialization changed captured evolution"
        );
    }
}
