use std::f64::consts::FRAC_PI_2;

use gr_core::{kerr_metric_quantities, photon_orbit_radius, shadow_boundary};

#[test]
fn smoke_kerr_metric_quantities_are_finite() {
    let (sigma, delta) = kerr_metric_quantities(6.0, FRAC_PI_2, 0.5);
    assert!(sigma.is_finite() && sigma > 0.0);
    assert!(delta.is_finite() && delta > 0.0);
}

#[test]
fn smoke_shadow_boundary_produces_closed_curve_samples() {
    let (r_pro, r_retro) = photon_orbit_radius(0.5);
    let (alpha, beta) = shadow_boundary(0.5, 16, FRAC_PI_2);

    assert!(
        r_pro < r_retro,
        "prograde orbit should lie inside retrograde orbit"
    );
    assert_eq!(alpha.len(), 32);
    assert_eq!(beta.len(), 32);
    assert!(alpha.iter().all(|value| value.is_finite()));
    assert!(beta.iter().all(|value| value.is_finite()));
}
