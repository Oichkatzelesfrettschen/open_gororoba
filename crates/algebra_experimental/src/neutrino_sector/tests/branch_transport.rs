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

#[test]
fn test_fixed_alpha_nu_strip_stays_on_base_branch() {
    let (v6_basis, _sv, _assessors) = extract_v6_basis();
    let base = compute_gradient_frame(&v6_basis, 3.00, 1.35);
    let report = compute_path_scan(
        "fixed_alpha_nu_1.35",
        &base,
        &v6_basis,
        &fixed_alpha_nu_scan_points(),
    );

    assert_eq!(report.rows.len(), 6);
    assert!(report.rows.iter().all(|row| row.perm_match));
    assert!(
        report
            .rows
            .iter()
            .all(|row| row.align_g23 > 0.9999 && row.align_u_atmo > 0.9999)
    );
}

#[test]
fn test_fixed_alpha_ch_path_crosses_low_alpha_nu_wall() {
    let (v6_basis, _sv, _assessors) = extract_v6_basis();
    let base = compute_gradient_frame(&v6_basis, 3.00, 1.35);
    let report = compute_path_scan(
        "fixed_alpha_ch_3.00",
        &base,
        &v6_basis,
        &fixed_alpha_ch_scan_points(),
    );

    assert_eq!(report.rows.len(), 6);
    assert!(!report.rows[0].perm_match);
    assert!(report.rows[1].perm_match);
    assert!(report.rows[0].align_g23 < 0.0);
    assert!(report.rows[0].align_u_atmo < 0.0);
}

#[test]
fn test_branch_map_counts_match_transport_probe_baseline() {
    let artifacts = default_probe_artifacts();
    assert_eq!(artifacts.branch_map.rows.len(), 66);
    assert_eq!(artifacts.branch_map.same_perm_points, 43);
    assert_eq!(artifacts.branch_map.switched_perm_points, 23);
    assert_eq!(artifacts.branch_walls.count, 27);
}
