use qgp_scaling::bdmps_quenching::OutgoingBdmps;
use std::f64::consts::{LN_2, PI, SQRT_2};

// Independent real-function and cosine-product evaluation avoids the production
// positive-series and log-domain quadrature algorithms.
fn reference_shape(coordinate: f64) -> f64 {
    let argument = (0.5 / coordinate).sqrt();
    if coordinate < 0.001 {
        // The omitted correction is below 4e-20 throughout the branch.
        argument - LN_2
    } else if coordinate <= 10.0 {
        0.5 * (argument.cosh().powi(2) - argument.sin().powi(2)).ln()
    } else {
        // The remaining odd-integer sum is bounded by
        // 8/(pi^4*x^2) * [a^-4 + a^-3/6], a=2049.
        (0..1024)
            .map(|index| {
                let odd = f64::from(2 * index + 1);
                0.5 * (4.0 / (PI * PI * odd * odd * coordinate)).powi(2).ln_1p()
            })
            .sum()
    }
}

fn compactified_simpson(intervals: usize, laplace: Option<f64>) -> f64 {
    let alpha = 4.0 / (3.0 * PI);
    let step = 1.0 / intervals as f64;
    let mut sum = alpha * SQRT_2 * laplace.unwrap_or(1.0);
    for index in 1..intervals {
        let parameter = index as f64 * step;
        let complement = 1.0 - parameter;
        let coordinate = (parameter / complement).powi(2);
        let jacobian = 2.0 * parameter / complement.powi(3);
        let response = laplace.map_or(1.0, |argument| {
            -(-argument * coordinate).exp_m1() / coordinate
        });
        let integrand = alpha * reference_shape(coordinate) * response * jacobian;
        sum += if index % 2 == 0 { 2.0 } else { 4.0 } * integrand;
    }
    sum * step / 3.0
}

#[test]
fn independent_compactified_mean_and_laplace() {
    let model = OutgoingBdmps::new(0.5, 4.0 / 3.0).unwrap();
    let normalized = model.laplace(0.0, 128, 1e-10).unwrap();
    assert_eq!(normalized.value, 1.0);
    assert_eq!(normalized.exponent, 0.0);
    let mean_coarse = compactified_simpson(8192, None);
    let mean_fine = compactified_simpson(16384, None);
    println!("mean coarse={mean_coarse:.17e} fine={mean_fine:.17e}");
    assert!((mean_fine - 1.0 / 3.0).abs() < 1e-8);
    assert!((mean_fine - mean_coarse).abs() < 1e-8);
    for argument in [0.2, 1.0, 5.0] {
        let coarse = compactified_simpson(8192, Some(argument));
        let fine = compactified_simpson(16384, Some(argument));
        let actual = model.laplace(argument, 128, 1e-10).unwrap();
        println!(
            "laplace s={argument} reference={fine:.17e} delta={:.17e} actual={actual:?}",
            (fine - coarse).abs()
        );
        assert!((fine - coarse).abs() < 1e-8);
        assert!((actual.exponent - fine).abs() < 1e-8);
        assert!(actual.quadrature_change < 1e-8);
        assert!(actual.tail_bound <= 1e-10);
    }
}

#[test]
fn convolution_grid_obeys_convexity_and_coordinate_identity() {
    let model = OutgoingBdmps::new(0.5, 4.0 / 3.0).unwrap();
    let mut previous_index = [1.0; 3];
    for spectral_index in [3.0, 6.1, 10.0] {
        let mut previous_momentum = 0.0;
        for (index, momentum) in [0.5, 2.0, 10.0].into_iter().enumerate() {
            let estimate = model
                .raa_omega_c(momentum, spectral_index, 256, 64, 1e-10)
                .unwrap();
            println!("convolution u={momentum} n={spectral_index} estimate={estimate:?}");
            assert!(estimate.passes_numerical_gates(1e-8));
            let lower = (1.0 + model.mean_over_omega_c() / momentum).powf(-spectral_index);
            assert!(estimate.value >= lower - 1e-8);
            assert!(estimate.value > previous_momentum);
            assert!(estimate.value < previous_index[index]);
            previous_momentum = estimate.value;
            previous_index[index] = estimate.value;
            let converted = model
                .raa_mean_loss(
                    momentum / model.mean_over_omega_c(),
                    spectral_index,
                    256,
                    64,
                    1e-10,
                )
                .unwrap();
            assert!((converted.value - estimate.value).abs() < 1e-14);
        }
    }
}

fn independent_gamma_convolution(intervals: usize, momentum: f64, delta_loss: Option<f64>) -> f64 {
    let inner_intervals = 4096;
    let inner_step = 1.0 / inner_intervals as f64;
    let alpha = 4.0 / (3.0 * PI);
    let samples: Vec<_> = (1..inner_intervals)
        .map(|index| {
            let parameter = index as f64 * inner_step;
            let complement = 1.0 - parameter;
            let coordinate = (parameter / complement).powi(2);
            let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
            let factor =
                weight * inner_step / 3.0 * alpha * reference_shape(coordinate) * 2.0 * parameter
                    / complement.powi(3)
                    / coordinate;
            (coordinate, factor)
        })
        .collect();
    // At n=3, the discarded Gamma probability above 80 is
    // exp(-80)*(1+80+80^2/2), below 6e-32; suppression is bounded by one.
    let step = 80.0 / intervals as f64;
    let integrand = |parameter: f64| {
        let argument = parameter / momentum;
        let exponent = delta_loss.map_or_else(
            || {
                samples
                    .iter()
                    .map(|(coordinate, factor)| factor * -(-argument * coordinate).exp_m1())
                    .sum::<f64>()
                    + inner_step / 3.0 * alpha * SQRT_2 * argument
            },
            |loss| argument * loss,
        );
        parameter.powi(2) * (-parameter - exponent).exp() / 2.0
    };
    let mut sum = integrand(80.0);
    for index in 1..intervals {
        sum += if index % 2 == 0 { 2.0 } else { 4.0 } * integrand(index as f64 * step);
    }
    sum * step / 3.0
}

#[test]
fn independent_outer_convolution_and_delta_omission_control() {
    let model = OutgoingBdmps::new(0.5, 4.0 / 3.0).unwrap();
    for momentum in [0.5, 2.0, 10.0] {
        let coarse = independent_gamma_convolution(4096, momentum, None);
        let fine = independent_gamma_convolution(8192, momentum, None);
        let actual = model.raa_omega_c(momentum, 3.0, 256, 64, 1e-10).unwrap();
        let delta = independent_gamma_convolution(8192, momentum, Some(1.0 / 3.0));
        let analytic_delta = (1.0 + 1.0 / (3.0 * momentum)).powi(-3);
        println!(
            "independent outer u={momentum} coarse={coarse:.17e} fine={fine:.17e} production={:.17e} delta={delta:.17e} analytic_delta={analytic_delta:.17e}",
            actual.value
        );
        assert!((coarse - fine).abs() < 1e-8);
        assert!((actual.value - fine).abs() < 1e-8);
        assert!((delta - analytic_delta).abs() < 1e-8);
        assert!((actual.value - delta).abs() > 1e-4);
    }
}
