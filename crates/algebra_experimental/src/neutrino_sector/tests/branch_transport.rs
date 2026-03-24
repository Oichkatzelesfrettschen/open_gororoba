use super::super::*;
use flavor_lifts::extract_v6_basis;

#[test]
fn test_stable_branch_loop_closes_trivially() {
    let (v6_basis, _sv, _assessors) = extract_v6_basis();
    let report = compute_loop_transport(
        "stable_branch_loop",
        &v6_basis,
        &stable_branch_loop_points(),
    );

    assert_eq!(report.summary.wall_crossings, 0);
    assert!((report.summary.final_align_g12 - 1.0).abs() < 1e-9);
    assert!((report.summary.final_align_g13 - 1.0).abs() < 1e-9);
    assert!((report.summary.final_align_g23 - 1.0).abs() < 1e-9);
    assert!((report.summary.final_align_u_solar - 1.0).abs() < 1e-9);
    assert!((report.summary.final_align_u_atmo - 1.0).abs() < 1e-9);
}

#[test]
fn test_wall_crossing_loop_closes_after_sign_transport() {
    let (v6_basis, _sv, _assessors) = extract_v6_basis();
    let report = compute_loop_transport(
        "wall_crossing_loop",
        &v6_basis,
        &wall_crossing_loop_points(),
    );

    assert_eq!(report.summary.wall_crossings, 2);
    assert!((report.summary.final_align_g12 - 1.0).abs() < 1e-9);
    assert!((report.summary.final_align_g13 - 1.0).abs() < 1e-9);
    assert!((report.summary.final_align_g23 - 1.0).abs() < 1e-9);
    assert!((report.summary.final_align_u_solar - 1.0).abs() < 1e-9);
    assert!((report.summary.final_align_u_atmo - 1.0).abs() < 1e-9);

    let wall_steps = report.steps.iter().filter(|step| step.wall_crossed).count();
    assert_eq!(wall_steps, 2);
}
