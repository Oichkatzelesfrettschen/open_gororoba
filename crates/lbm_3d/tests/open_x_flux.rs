//! Directional and analytical oracles for nonperiodic x transport.

use lbm_3d::{
    open_x_boundary::{OpenXBoundary, XOutflow, population_mass},
    solver::{LbmSolver3D, aosoa_idx},
};

fn index(x: usize, y: usize, z: usize) -> usize {
    z * 15 + y * 5 + x
}

#[test]
fn outgoing_packets_leave_without_opposite_face_wrap() {
    for (x, direction) in [(0, 2), (4, 1), (0, 8), (4, 7)] {
        let mut solver = LbmSolver3D::new(5, 3, 2, 0.8);
        solver.f.fill(0.0);
        solver.f[aosoa_idx(index(x, 1, 1), direction)] = 2.5;
        let ledger = OpenXBoundary::new([5, 3, 2])
            .unwrap()
            .stream_and_reconstruct(&mut solver, [0.0; 3], XOutflow::ZeroGradientPopulations)
            .unwrap();
        assert!(solver.f.iter().all(|&population| population == 0.0));
        assert_eq!(ledger.mass_before, 2.5);
        assert_eq!(ledger.mass_after_streaming, 0.0);
        assert_eq!(ledger.face.min_x_outgoing, if x == 0 { 2.5 } else { 0.0 });
        assert_eq!(ledger.face.max_x_outgoing, if x == 4 { 2.5 } else { 0.0 });
        assert_eq!(ledger.face.min_x_incoming + ledger.face.max_x_incoming, 0.0);
        assert_eq!(ledger.total_residual(), 0.0);
    }
}

#[test]
fn interior_packets_advect_with_only_transverse_wrapping() {
    for (direction, destination) in [
        (1, index(3, 2, 1)),
        (7, index(3, 0, 1)),
        (15, index(2, 0, 0)),
    ] {
        let mut solver = LbmSolver3D::new(5, 3, 2, 0.8);
        solver.f.fill(0.0);
        solver.f[aosoa_idx(index(2, 2, 1), direction)] = 3.0;
        let ledger = OpenXBoundary::new([5, 3, 2])
            .unwrap()
            .stream_and_reconstruct(&mut solver, [0.0; 3], XOutflow::ZeroGradientPopulations)
            .unwrap();
        assert_eq!(solver.f[aosoa_idx(destination, direction)], 3.0);
        assert_eq!(population_mass(&solver).unwrap(), 3.0);
        assert_eq!(ledger.face.net_incoming(), 0.0);
    }
}

#[test]
fn nonuniform_faces_match_independent_analytical_sums() {
    let mut solver = LbmSolver3D::new(5, 3, 2, 0.8);
    for cell in 0..30 {
        for direction in 0..19 {
            solver.f[aosoa_idx(cell, direction)] =
                1.0 + cell as f64 / 100.0 + direction as f64 / 1000.0;
        }
    }
    let face_sum = |x, directions: &[usize]| {
        let mut sum = 0.0;
        for z in 0..2 {
            for y in 0..3 {
                for &direction in directions {
                    sum += solver.f[aosoa_idx(index(x, y, z), direction)];
                }
            }
        }
        sum
    };
    let negative_x = [2, 8, 10, 12, 14];
    let positive_x = [1, 7, 9, 11, 13];
    let expected = [
        face_sum(0, &negative_x),
        face_sum(4, &positive_x),
        face_sum(1, &negative_x),
        face_sum(4, &negative_x),
    ];
    let initial_mass = population_mass(&solver).unwrap();
    let ledger = OpenXBoundary::new([5, 3, 2])
        .unwrap()
        .stream_and_reconstruct(&mut solver, [0.0; 3], XOutflow::ZeroGradientPopulations)
        .unwrap();
    for (actual, expected) in [
        ledger.face.min_x_outgoing,
        ledger.face.max_x_outgoing,
        ledger.face.min_x_incoming,
        ledger.face.max_x_incoming,
    ]
    .into_iter()
    .zip(expected)
    {
        assert!((actual - expected).abs() < 1e-12);
    }
    let expected_final = initial_mass + expected[2] + expected[3] - expected[0] - expected[1];
    assert!((population_mass(&solver).unwrap() - expected_final).abs() < initial_mass * 1e-12);
    assert!(ledger.streaming_residual().abs() < initial_mass * 1e-12);
    assert!(ledger.boundary_residual().abs() < initial_mass * 1e-12);
}

#[test]
fn repeated_open_steps_account_for_flux_and_zero_collision_mass_source() {
    let mut solver = LbmSolver3D::new(5, 3, 2, 0.8);
    for cell in 0..30 {
        solver.rho[cell] = 1.0 + (cell % 5) as f64 * 0.01;
        solver.u[cell] = [0.04, 0.0, 0.0];
    }
    solver.reinitialize_from_macroscopic();
    let initial_mass = population_mass(&solver).unwrap();
    let mut cumulative_flux = 0.0;
    let mut boundary = OpenXBoundary::new([5, 3, 2]).unwrap();
    for step in 1..=20 {
        let ledger = boundary
            .stream_and_reconstruct(
                &mut solver,
                [0.04, 0.0, 0.0],
                XOutflow::ZeroGradientPopulations,
            )
            .unwrap();
        solver.phase1_collision().unwrap();
        let final_mass = population_mass(&solver).unwrap();
        cumulative_flux += ledger.face.net_incoming();
        let collision_delta = final_mass - ledger.mass_after_boundary;
        let residual = final_mass - initial_mass - cumulative_flux;
        println!(
            "step={step} mass={final_mass:.17e} net_flux={:.17e} collision_delta={collision_delta:.17e} cumulative_residual={residual:.17e}",
            ledger.face.net_incoming()
        );
        assert!(collision_delta.abs() < initial_mass * 1e-12);
        assert!(residual.abs() < initial_mass * 1e-12);
        assert!(solver.rho.iter().all(|&rho| rho.is_finite() && rho > 0.0));
    }
}

#[test]
fn invalid_inputs_preserve_populations_and_timestep() {
    for velocity in [[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]] {
        let mut solver = LbmSolver3D::new(5, 3, 2, 0.8);
        let before = solver.f.clone();
        assert!(
            OpenXBoundary::new([5, 3, 2])
                .unwrap()
                .stream_and_reconstruct(&mut solver, velocity, XOutflow::ZeroGradientPopulations)
                .is_err()
        );
        assert_eq!(solver.f, before);
        assert_eq!(solver.timestep, 0);
    }
    let mut solver = LbmSolver3D::new(5, 3, 2, 0.8);
    solver.f[aosoa_idx(29, 18)] = f64::NAN;
    let before: Vec<u64> = solver.f.iter().map(|value| value.to_bits()).collect();
    assert!(
        OpenXBoundary::new([5, 3, 2])
            .unwrap()
            .stream_and_reconstruct(&mut solver, [0.0; 3], XOutflow::ZeroGradientPopulations)
            .is_err()
    );
    assert_eq!(
        before,
        solver
            .f
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(solver.timestep, 0);
    assert!(OpenXBoundary::new([1, 3, 2]).is_err());
}
