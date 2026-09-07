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
    let intervals = 65_536;
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
        for redshift in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 20.0, 80.0, 100.0] {
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
    for redshift in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 20.0, 80.0, 99.9] {
        let cosmic_dm = source_dm(redshift, 0.3153, 0.0493, 67.36);
        let recovered = dm_excess_to_redshift(cosmic_dm, 0.3153, 0.0493, 67.36).unwrap();
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
    )
    .unwrap();
    let expected = 299_792.458 / 67.36 * 2.0 * (1.0 - 1.0 / (1.0_f64 + redshift).sqrt());
    assert!((actual / expected - 1.0).abs() < 1e-8);
    let wrong_frame = dm_to_comoving(
        observed_dm,
        galactic_disk + galactic_halo,
        host_rest_frame,
        1.0,
        0.0493,
        67.36,
    )
    .unwrap();
    assert!((wrong_frame / expected - 1.0).abs() > 0.01);
}

#[test]
fn bounded_inverse_matches_analytic_limits_from_tiny_to_ceiling() {
    let normalization = electron_column_per_redshift(0.0, 0.3153, 0.0493, 67.36);
    for redshift in [1e-14_f64, 1e-8, 0.5, 10.0, 20.0, 40.0, 80.0, 99.999] {
        for (omega_m, integral) in [
            (1.0, 2.0 * redshift / ((1.0 + redshift).sqrt() + 1.0)),
            (0.0, redshift + redshift.powi(2) / 2.0),
        ] {
            let expected_dm = normalization * integral;
            let forward = macquart_dm_cosmic(redshift, omega_m, 0.0493, 67.36);
            assert!((forward / expected_dm - 1.0).abs() < 1e-11);
            let recovered = dm_excess_to_redshift(expected_dm, omega_m, 0.0493, 67.36).unwrap();
            assert!(
                (recovered / redshift - 1.0).abs() < 3e-12,
                "z={redshift}, recovered={recovered}"
            );
            let distance = dm_to_comoving(expected_dm, 0.0, 0.0, omega_m, 0.0493, 67.36).unwrap();
            let distance_integral = if omega_m == 0.0 {
                redshift
            } else {
                2.0 * redshift / ((1.0 + redshift).sqrt() * ((1.0 + redshift).sqrt() + 1.0))
            };
            let expected_distance = 299_792.458 / 67.36 * distance_integral;
            assert!((distance / expected_distance - 1.0).abs() < 3e-12);
        }
    }
}

#[test]
fn inverse_rejects_invalid_values_and_unbracketed_dm() {
    use cosmology_core::{DmInversionError, MAX_MACQUART_REDSHIFT};
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(matches!(
            dm_excess_to_redshift(invalid, 0.3, 0.05, 70.0),
            Err(DmInversionError::InvalidDm { .. })
        ));
        for component in 0..3 {
            let mut measures = [100.0, 10.0, 10.0];
            measures[component] = invalid;
            assert!(matches!(
                dm_to_comoving(measures[0], measures[1], measures[2], 0.3, 0.05, 70.0),
                Err(DmInversionError::InvalidDm { .. })
            ));
        }
    }
    for (omega_m, omega_b, h0) in [
        (f64::NAN, 0.05, 70.0),
        (f64::INFINITY, 0.05, 70.0),
        (-0.1, 0.05, 70.0),
        (1.1, 0.05, 70.0),
        (0.3, f64::NAN, 70.0),
        (0.3, f64::INFINITY, 70.0),
        (0.3, 0.0, 70.0),
        (0.3, -0.1, 70.0),
        (0.3, 1.1, 70.0),
        (0.3, 0.05, f64::NAN),
        (0.3, 0.05, f64::INFINITY),
        (0.3, 0.05, 0.0),
        (0.3, 0.05, -1.0),
    ] {
        for cosmic_dm in [0.0, 100.0] {
            assert_eq!(
                dm_excess_to_redshift(cosmic_dm, omega_m, omega_b, h0),
                Err(DmInversionError::InvalidCosmology)
            );
        }
        assert!(macquart_dm_cosmic(0.0, omega_m, omega_b, h0).is_nan());
    }
    assert_eq!(
        dm_to_comoving(10.0, 20.0, 0.0, 0.3, 0.05, 70.0),
        Err(DmInversionError::NegativeResidual)
    );
    assert_eq!(dm_excess_to_redshift(0.0, 0.3, 0.05, 70.0), Ok(0.0));
    assert_eq!(
        dm_to_comoving(30.0, 20.0, 10.0, 0.3, 0.05, f64::MIN_POSITIVE),
        Ok(0.0)
    );
    let maximum_dm = macquart_dm_cosmic(MAX_MACQUART_REDSHIFT, 0.3, 0.05, 70.0);
    assert_eq!(
        dm_excess_to_redshift(maximum_dm, 0.3, 0.05, 70.0),
        Ok(MAX_MACQUART_REDSHIFT)
    );
    for excess in [maximum_dm.next_up(), maximum_dm * 2.0, f64::MAX] {
        assert_eq!(
            dm_excess_to_redshift(excess, 0.3, 0.05, 70.0),
            Err(DmInversionError::OutOfRange { maximum_dm })
        );
    }
    for invalid_redshift in [-1.0, 100.1, f64::MAX, f64::INFINITY, f64::NAN] {
        assert!(macquart_dm_cosmic(invalid_redshift, 0.3, 0.05, 70.0).is_nan());
    }
    assert_eq!(
        dm_excess_to_redshift(1.0, 0.3, 0.05, f64::MAX),
        Err(DmInversionError::NumericalFailure)
    );
}

#[test]
fn public_comoving_distance_matches_expanded_analytic_limits() {
    for redshift in [20.0_f64, 80.0, 100.0] {
        for (omega_m, integral) in [
            (0.0, redshift),
            (1.0, 2.0 * (1.0 - 1.0 / (1.0 + redshift).sqrt())),
        ] {
            let expected = 299_792.458 / 67.36 * integral;
            let actual = cosmology_core::comoving_distance(redshift, omega_m, 67.36);
            assert!((actual / expected - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn positive_cosmic_dm_rejects_distance_underflow() {
    let redshift = dm_excess_to_redshift(1e-300, 0.3, 1e-100, 1e100).unwrap();
    assert!(redshift > 0.0);
    assert_eq!(
        dm_to_comoving(1e-300, 0.0, 0.0, 0.3, 1e-100, 1e100),
        Err(cosmology_core::DmInversionError::NumericalFailure)
    );
    assert_eq!(dm_to_comoving(0.0, 0.0, 0.0, 0.3, 1e-100, 1e100), Ok(0.0));
}
