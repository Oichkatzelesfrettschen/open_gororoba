//! Bounded periodic MRT population conformance and force-density impulse oracle.
use anyhow::{Result, ensure};
use lbm_3d::{
    lattice::D3Q19Lattice,
    solver::{LbmSolver3D, aosoa_idx},
};
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use std::io::Write;

const GRID: usize = 8;
const CELLS: usize = GRID * GRID * GRID;
const STEPS: usize = 12;
const TAU: f64 = 0.8;
// Absolute RMS gate inherited from the evolving FP32 population conformance oracle.
const ENGINEERING_GATE: f64 = 2e-5;

fn fixture(forced: bool) -> (LbmSolver3D, Vec<[f64; 3]>) {
    let mut solver = LbmSolver3D::new_mrt(GRID, GRID, GRID, TAU);
    let mut force = vec![[0.0; 3]; CELLS];
    for (cell, force_cell) in force.iter_mut().enumerate() {
        let phase_x = std::f64::consts::TAU * (cell % GRID) as f64 / GRID as f64;
        let phase_y = std::f64::consts::TAU * ((cell / GRID) % GRID) as f64 / GRID as f64;
        let phase_z = std::f64::consts::TAU * (cell / (GRID * GRID)) as f64 / GRID as f64;
        solver.rho[cell] = 1.0 + 0.03 * phase_x.cos() + 0.01 * phase_z.sin();
        solver.u[cell] = [
            0.02 * phase_y.sin(),
            0.01 * phase_z.cos(),
            0.005 * phase_x.sin(),
        ];
        if forced {
            // Input denotes force density, not acceleration: no multiplication by density.
            *force_cell = [
                2e-4 + 1e-4 * phase_y.cos(),
                8e-5 * phase_z.sin(),
                5e-5 * phase_x.cos(),
            ];
        }
    }
    solver.reinitialize_from_macroscopic();
    if forced {
        solver.set_force_field(force.clone()).unwrap();
    }
    (solver, force)
}
fn cpu_populations(solver: &LbmSolver3D) -> Vec<f64> {
    (0..solver.nx * solver.ny * solver.nz)
        .flat_map(|cell| (0..19).map(move |direction| solver.f[aosoa_idx(cell, direction)]))
        .collect()
}
struct Moments {
    density: Vec<f64>,
    momentum: Vec<f64>,
    mean_momentum: [f64; 3],
    minimum: f64,
}
fn moments(populations: &[f64]) -> Result<Moments> {
    ensure!(
        !populations.is_empty()
            && populations.len().is_multiple_of(19)
            && populations.iter().all(|value| value.is_finite()),
        "finite complete populations required"
    );
    let lattice = D3Q19Lattice::new();
    let cells = populations.len() / 19;
    let mut density = Vec::with_capacity(cells);
    let mut momentum = Vec::with_capacity(3 * cells);
    let mut mean_momentum = [0.0; 3];
    for cell in populations.chunks_exact(19) {
        density.push(cell.iter().sum::<f64>());
        for (axis, total) in mean_momentum.iter_mut().enumerate() {
            let component = cell
                .iter()
                .enumerate()
                .map(|(direction, population)| {
                    population * f64::from(lattice.velocity(direction)[axis])
                })
                .sum::<f64>();
            momentum.push(component);
            *total += component / cells as f64;
        }
    }
    ensure!(
        density
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            && momentum.iter().all(|value| value.is_finite()),
        "finite positive density and finite momentum required"
    );
    Ok(Moments {
        density,
        momentum,
        mean_momentum,
        minimum: populations.iter().copied().fold(f64::INFINITY, f64::min),
    })
}
fn rms(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    (left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        / left.len() as f64)
        .sqrt()
}
fn impulse_error(
    initial: &Moments,
    final_state: &Moments,
    force: &[[f64; 3]],
    steps: usize,
) -> f64 {
    (0..3)
        .map(|axis| {
            let expected = steps as f64 * force.iter().map(|value| value[axis]).sum::<f64>()
                / force.len() as f64;
            (final_state.mean_momentum[axis] - initial.mean_momentum[axis] - expected).abs()
        })
        .fold(0.0, f64::max)
}
#[test]
fn unforced_cpu_fixture_evolves_and_conserves_periodic_moments() -> Result<()> {
    let (mut cpu, force) = fixture(false);
    let initial = moments(&cpu_populations(&cpu))?;
    cpu.evolve(STEPS);
    let final_state = moments(&cpu_populations(&cpu))?;
    ensure!(
        rms(&initial.density, &final_state.density) > 1e-3,
        "frozen-evolution control must discriminate"
    );
    ensure!(
        impulse_error(&initial, &final_state, &force, STEPS) < 1e-12,
        "unforced periodic momentum"
    );
    ensure!(
        (initial.density.iter().sum::<f64>() - final_state.density.iter().sum::<f64>()).abs()
            / (CELLS as f64)
            < 1e-12,
        "unforced periodic mass"
    );
    Ok(())
}
#[test]
fn cpu_mrt_force_density_impulse_is_relaxation_independent() -> Result<()> {
    for (nx, ny, nz) in [(8, 8, 8), (5, 3, 2)] {
        for tau in [0.6, 0.8, 1.1] {
            let mut solver = LbmSolver3D::new_mrt(nx, ny, nz, tau);
            let cells = nx * ny * nz;
            for cell in 0..cells {
                solver.rho[cell] = 1.0 + 0.2 * cell as f64 / cells as f64;
                solver.u[cell] = [0.01, -0.005, 0.002];
            }
            solver.reinitialize_from_macroscopic();
            let force: Vec<_> = (0..cells)
                .map(|cell| [2e-4 + 1e-4 * cell as f64 / cells as f64, -3e-5, 4e-5])
                .collect();
            solver.set_force_field(force.clone())?;
            let initial = moments(&cpu_populations(&solver))?;
            for step in 1..=STEPS {
                solver.phase1_collision()?;
                solver.phase2_streaming()?;
                let state = moments(&cpu_populations(&solver))?;
                ensure!(
                    impulse_error(&initial, &state, &force, step) < 1e-12,
                    "force-density impulse nx={nx} tau={tau} step={step}"
                );
                ensure!(
                    (initial.density.iter().sum::<f64>() - state.density.iter().sum::<f64>()).abs()
                        / (cells as f64)
                        < 1e-12,
                    "forced mass conservation"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires explicitly admitted CUDA hardware; writes a fresh MRT_CONFORMANCE_OUTPUT receipt"]
fn evolving_mrt_direct_and_captured_population_oracles() -> Result<()> {
    let mut report = String::from(
        "protocol=periodic_MRT_D3Q19_grid8_steps12_tau0.8\nengineering_gate=0.00002\nmetrics=absolute_RMS_post_step_population_density_raw_momentum\nimpulse_gate=0.00002_absolute_max_component_mean_momentum\nforce_semantics=force_density_in_lattice_units\nphysical_velocity_half_step=excluded_from_raw_momentum_comparison\ncaptured_path=explicit_step_graph_pair\n",
    );
    report.push_str("forced,step,comparison,population_rms,density_rms,momentum_rms,minimum_population,cpu_impulse_error,gpu_impulse_error,conformance_pass,impulse_pass\n");
    let mut all_pass = true;
    for forced in [false, true] {
        let (mut cpu, force) = fixture(forced);
        let mut direct =
            LbmSolver3DCuda::new_capture_capable(GRID, GRID, GRID, TAU, Precision::FP32, true)?;
        let mut captured =
            LbmSolver3DCuda::new_capture_capable(GRID, GRID, GRID, TAU, Precision::FP32, true)?;
        for gpu in [&mut direct, &mut captured] {
            gpu.initialize_custom(&cpu.rho, &cpu.u)?;
            gpu.set_force_field(&force)?;
        }
        let initial = moments(&cpu_populations(&cpu))?;
        for step in (2..=STEPS).step_by(2) {
            cpu.evolve(2);
            direct.step()?;
            direct.step()?;
            captured.step_graph_pair()?;
            let cpu_values = cpu_populations(&cpu);
            let cpu_state = moments(&cpu_values)?;
            let direct_values: Vec<_> = direct
                .read_populations_fp32()?
                .into_iter()
                .map(f64::from)
                .collect();
            let captured_values: Vec<_> = captured
                .read_populations_fp32()?
                .into_iter()
                .map(f64::from)
                .collect();
            for (name, left_values, right_values) in [
                ("cpu_direct", &cpu_values, &direct_values),
                ("cpu_captured", &cpu_values, &captured_values),
                ("direct_captured", &direct_values, &captured_values),
            ] {
                let left = moments(left_values)?;
                let right = moments(right_values)?;
                let population_error = rms(left_values, right_values);
                let density_error = rms(&left.density, &right.density);
                let momentum_error = rms(&left.momentum, &right.momentum);
                let cpu_impulse = impulse_error(&initial, &cpu_state, &force, step);
                let gpu_impulse = impulse_error(&initial, &right, &force, step);
                let conformance = [population_error, density_error, momentum_error]
                    .iter()
                    .all(|value| value.is_finite() && *value < ENGINEERING_GATE)
                    && left.minimum >= 0.0
                    && right.minimum >= 0.0;
                let impulse = cpu_impulse < ENGINEERING_GATE && gpu_impulse < ENGINEERING_GATE;
                all_pass &= conformance && impulse;
                report.push_str(&format!("{forced},{step},{name},{population_error:.17e},{density_error:.17e},{momentum_error:.17e},{:.17e},{cpu_impulse:.17e},{gpu_impulse:.17e},{conformance},{impulse}\n",left.minimum.min(right.minimum)));
            }
        }
    }
    println!("{report}");
    if let Ok(path) = std::env::var("MRT_CONFORMANCE_OUTPUT") {
        let mut output = std::fs::File::create_new(path)?;
        output.write_all(report.as_bytes())?;
        output.sync_all()?;
    }
    ensure!(
        all_pass,
        "MRT conformance or force-density impulse predicate failed; retain full receipt"
    );
    Ok(())
}
