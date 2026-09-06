//! Paired FP64 force-response instrument on a uniform periodic mesh.

use lbm_3d::solver::LbmSolver3D;

fn response(density: f64, tau: f64, force: f64) -> (f64, f64) {
    let mut solver = LbmSolver3D::new(4, 4, 4, tau);
    solver.initialize_uniform(density, [0.05, 0.0, 0.0]);
    solver.set_force_field(vec![[force, 0.0, 0.0]; 64]).unwrap();
    solver.evolve(1);
    let velocity = solver.u.iter().map(|value| value[0]).sum::<f64>() / 64.0;
    let mass_error = (solver.total_mass() / (density * 64.0) - 1.0).abs();
    (velocity, mass_error)
}

#[test]
fn paired_force_resolution_sweep() {
    println!(
        "force,tau,density,positive_increment,negative_increment,central_response,relative_error,zero_drift,mass_error,resolved"
    );
    for force in [
        1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20,
    ] {
        let mut all_resolved = true;
        for tau in [0.6, 0.8, 1.0] {
            for density in [0.5, 1.0, 2.0] {
                let (zero, zero_mass_error) = response(density, tau, 0.0);
                let (positive, positive_mass_error) = response(density, tau, force);
                let (negative, negative_mass_error) = response(density, tau, -force);
                let central = (positive - negative) / 2.0;
                let expected = force / density;
                let relative_error = (central / expected - 1.0).abs();
                let drift = (zero - 0.05).abs();
                let mass_error = zero_mass_error
                    .max(positive_mass_error)
                    .max(negative_mass_error);
                let resolved = relative_error <= 0.01 && central > 10.0 * drift;
                all_resolved &= resolved;
                println!(
                    "{force:.17e},{tau},{density},{:.17e},{:.17e},{central:.17e},{relative_error:.17e},{drift:.17e},{mass_error:.17e},{resolved}",
                    positive - zero,
                    negative - zero
                );
                assert!(central.is_finite() && relative_error.is_finite() && mass_error < 1e-12);
                if force >= 1e-8 {
                    assert!(
                        resolved,
                        "resolved-force oracle failed: force={force}, tau={tau}, density={density}"
                    );
                }
            }
        }
        println!("all_tau_density_resolved,{force:.17e},{all_resolved}");
    }
}
