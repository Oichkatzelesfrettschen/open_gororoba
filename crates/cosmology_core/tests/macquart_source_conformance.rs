//! Macquart et al. (2020), Eqs. (1)-(2), under constant diffuse fraction.

use cosmology_core::{dm_excess_to_redshift, dm_to_comoving, macquart_dm_cosmic};

// SI mass density and physical path length provide an independent unit route.
// The parsec and proton constants match the declared production convention.
fn electron_column_per_redshift(redshift: f64, omega_m: f64, omega_b: f64, h0: f64) -> f64 {
    let hubble_si = h0 * 1_000.0 / 3.0857e22;
    let critical_density = 3.0 * hubble_si.powi(2) / (8.0 * std::f64::consts::PI * 6.67430e-11);
    let baryon_density = omega_b * critical_density * (1.0 + redshift).powi(3);
    let electron_density = 0.83 * baryon_density / 1.67262192e-27 * (1.0 - 0.25 / 2.0);
    let expansion = (omega_m * (1.0 + redshift).powi(3) + 1.0 - omega_m).sqrt();
    let physical_path = 299_792_458.0 / (hubble_si * expansion * (1.0 + redshift));
    electron_density * physical_path / (1.0 + redshift) / 3.0857e22
}

fn source_dm(redshift: f64, omega_m: f64, omega_b: f64, h0: f64) -> f64 {
    let intervals = 4096;
    let width = redshift / f64::from(intervals);
    let mut weighted_sum = 0.0;
    for index in 0..=intervals {
        let weight = if index == 0 || index == intervals {
            1.0
        } else if index % 2 == 0 {
            2.0
        } else {
            4.0
        };
        weighted_sum +=
            weight * electron_column_per_redshift(f64::from(index) * width, omega_m, omega_b, h0);
    }
    weighted_sum * width / 3.0
}

#[test]
fn mean_dm_matches_independent_si_density_integral() {
    for (omega_m, omega_b, h0) in [(0.3153, 0.0493, 67.36), (0.25, 0.04, 73.0)] {
        for redshift in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0] {
            let expected = source_dm(redshift, omega_m, omega_b, h0);
            let actual = macquart_dm_cosmic(redshift, omega_m, omega_b, h0);
            let relative_error = (actual / expected - 1.0).abs();
            println!(
                "omega_m={omega_m} omega_b={omega_b} h0={h0} z={redshift} source_dm={expected:.12} actual_dm={actual:.12} relative_error={relative_error:.12e}"
            );
            assert!(
                relative_error < 1e-10,
                "z={redshift}: {actual} versus {expected}"
            );
        }
    }
}

#[test]
fn mean_dm_matches_analytic_expansion_limits() {
    let normalization = electron_column_per_redshift(0.0, 0.3153, 0.0493, 67.36);
    for redshift in [0.01_f64, 0.5, 1.0, 3.0] {
        // Matter-only E=(1+z)^(3/2); constant-Hubble E=1.
        for (omega_m, integral) in [
            (1.0, 2.0 * ((1.0 + redshift).sqrt() - 1.0)),
            (0.0, redshift + redshift.powi(2) / 2.0),
        ] {
            let actual = macquart_dm_cosmic(redshift, omega_m, 0.0493, 67.36);
            assert!((actual / (normalization * integral) - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn inverse_recovers_source_generated_cosmic_dm() {
    for redshift in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0] {
        let cosmic_dm = source_dm(redshift, 0.3153, 0.0493, 67.36);
        let recovered = dm_excess_to_redshift(cosmic_dm, 0.3153, 0.0493, 67.36);
        assert!(
            (recovered - redshift).abs() < 1e-8,
            "{recovered} versus {redshift}"
        );
    }
}

#[test]
fn distance_chain_uses_observer_frame_host_and_total_galactic_dm() {
    let redshift = 1.0;
    let host_rest_frame = 100.0;
    let host_observer_frame = host_rest_frame / (1.0 + redshift);
    let galactic_disk = 30.0;
    let galactic_halo = 40.0;
    // Matter-only comoving distance has an analytic antiderivative.
    let cosmic_dm = source_dm(redshift, 1.0, 0.0493, 67.36);
    let observed_dm = cosmic_dm + galactic_disk + galactic_halo + host_observer_frame;
    let actual = dm_to_comoving(
        observed_dm,
        galactic_disk + galactic_halo,
        host_observer_frame,
        1.0,
        0.0493,
        67.36,
    );
    let expected = 299_792.458 / 67.36 * 2.0 * (1.0 - 1.0 / (1.0_f64 + redshift).sqrt());
    assert!((actual / expected - 1.0).abs() < 1e-8);
    let wrong_frame = dm_to_comoving(
        observed_dm,
        galactic_disk + galactic_halo,
        host_rest_frame,
        1.0,
        0.0493,
        67.36,
    );
    assert!((wrong_frame / expected - 1.0).abs() > 0.01);
}
