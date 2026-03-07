use lbm_3d::solver::LbmSolver3D;

#[test]
fn smoke_lbm_solver_initializes_and_conserves_mass() {
    let mut solver = LbmSolver3D::new(8, 8, 8, 0.8);
    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    let mass_before = solver.total_mass();
    solver.evolve(2);
    let mass_after = solver.total_mass();

    assert!(mass_before > 0.0);
    assert!((mass_before - mass_after).abs() < 1e-8);
}
