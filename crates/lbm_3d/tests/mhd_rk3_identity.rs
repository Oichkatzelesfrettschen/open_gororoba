use lbm_3d::mhd::{MhdConfig, MhdField, MhdIntegrator, ssp_rk3_amplification_squared};
use std::f64::consts::TAU;

fn multiply(left: [f64; 2], right: [f64; 2]) -> [f64; 2] {
    [
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    ]
}

fn polynomial(real: f64, imaginary: f64) -> [f64; 2] {
    let argument = [real, imaginary];
    let square = multiply(argument, argument);
    let cube = multiply(square, argument);
    [
        1.0 + real + square[0] / 2.0 + cube[0] / 6.0,
        imaginary + square[1] / 2.0 + cube[1] / 6.0,
    ]
}

fn phase(index: usize, shape: [usize; 3], mode: [usize; 3]) -> f64 {
    let [nx, ny, _] = shape;
    let positions = [index % nx, index / nx % ny, index / (nx * ny)];
    positions
        .into_iter()
        .zip(shape)
        .zip(mode)
        .map(|((position, size), frequency)| TAU * position as f64 * frequency as f64 / size as f64)
        .sum()
}

fn initialize(shape: [usize; 3], mode: [usize; 3], config: MhdConfig) -> MhdField {
    let [nx, ny, nz] = shape;
    let mut field = MhdField::new(nx, ny, nz, config);
    for (index, value) in field.bz.iter_mut().enumerate() {
        *value = phase(index, shape, mode).cos();
    }
    field
}

fn advance(field: &mut MhdField, velocity: &[[f64; 3]]) {
    field
        .try_evolve_b_field_with_integrator(velocity, MhdIntegrator::SspRk3)
        .unwrap();
}

#[test]
fn complex_fourier_gain_matches_independent_polynomial() {
    for (shape, mode, eta, velocity) in [
        ([16, 1, 1], [1, 0, 0], 0.0, [0.4, 0.0, 0.0]),
        ([16, 1, 1], [3, 0, 0], 0.1, [0.4, 0.0, 0.0]),
        ([9, 5, 3], [2, 1, 0], 0.1, [0.2, -0.1, 0.0]),
        ([8, 8, 8], [1, 2, 0], 0.1, [0.3, 0.1, 0.0]),
        ([8, 8, 8], [4, 4, 4], 0.1, [0.0; 3]),
    ] {
        let mut field = initialize(
            shape,
            mode,
            MhdConfig {
                eta,
                ..MhdConfig::default()
            },
        );
        let velocities = vec![velocity; field.bz.len()];
        field
            .validate_uniform_transverse_ssp_rk3(&velocities)
            .unwrap();
        let wave: Vec<_> = mode
            .into_iter()
            .zip(shape)
            .map(|(frequency, size)| TAU * frequency as f64 / size as f64)
            .collect();
        let damping = eta
            * 4.0
            * wave
                .iter()
                .map(|value| (value / 2.0).sin().powi(2))
                .sum::<f64>();
        let rotation = -wave
            .iter()
            .zip(velocity)
            .map(|(value, speed)| value.sin() * speed)
            .sum::<f64>();
        let gain = polynomial(-damping, rotation);
        let mut expected_gain = [1.0, 0.0];
        for step in 1..=32 {
            advance(&mut field, &velocities);
            expected_gain = multiply(expected_gain, gain);
            if [1, 2, 8, 32].contains(&step) {
                let error = field
                    .bz
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        let angle = phase(index, shape, mode);
                        (value - (expected_gain[0] * angle.cos() - expected_gain[1] * angle.sin()))
                            .abs()
                    })
                    .fold(0.0_f64, f64::max);
                println!("fourier shape={shape:?} mode={mode:?} step={step} error={error:.17e}");
                assert!(error < 1e-12);
                let mut projections = [0.0; 2];
                let mut norms = [0.0; 2];
                for (index, value) in field.bz.iter().enumerate() {
                    let angle = phase(index, shape, mode);
                    for (component, basis) in [angle.cos(), -angle.sin()].into_iter().enumerate() {
                        projections[component] += value * basis;
                        norms[component] += basis * basis;
                    }
                }
                for component in 0..2 {
                    // A sampled checkerboard has an identically absent sine basis.
                    if norms[component] > 0.5 {
                        assert!(
                            (projections[component] / norms[component] - expected_gain[component])
                                .abs()
                                < 1e-12
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn longitudinal_damping_matches_staged_operator() {
    let shape = [16, 1, 1];
    let mode = [1, 0, 0];
    let cleaning_rate: f64 = 0.7;
    let eta = 0.05;
    let mut field = initialize(
        shape,
        mode,
        MhdConfig {
            eta,
            cleaning_rate,
            ..MhdConfig::default()
        },
    );
    std::mem::swap(&mut field.bx, &mut field.bz);
    let wave = TAU / 16.0;
    let eigenvalue =
        -4.0 * eta * (wave / 2.0).sin().powi(2) - cleaning_rate.powi(2) * wave.sin().powi(2);
    let gain = polynomial(eigenvalue, 0.0)[0];
    let velocity = vec![[0.0; 3]; 16];
    for step in 1..=32 {
        advance(&mut field, &velocity);
        if [1, 2, 8, 32].contains(&step) {
            for index in 0..16 {
                let angle = phase(index, shape, mode);
                assert!((field.bx[index] - gain.powi(step) * angle.cos()).abs() < 1e-12);
                let expected_psi =
                    cleaning_rate.powi(2) * wave.sin() * gain.powi(step) * angle.sin();
                assert!((field.psi[index] - expected_psi).abs() < 1e-12);
            }
            println!("longitudinal step={step} gain={gain:.17e}");
        }
    }
}

fn uniform_error(
    cells: usize,
    timestep: f64,
    steps: usize,
    speed: f64,
    eta: f64,
    continuum: bool,
) -> f64 {
    let shape = [cells, 1, 1];
    let mode = [1, 0, 0];
    let mut field = initialize(
        shape,
        mode,
        MhdConfig {
            eta,
            dt_mhd: timestep,
            ..MhdConfig::default()
        },
    );
    let velocity = vec![[speed, 0.0, 0.0]; cells];
    for _ in 0..steps {
        advance(&mut field, &velocity);
    }
    let wave = TAU / cells as f64;
    let time = timestep * steps as f64;
    let rotation = speed * if continuum { wave } else { wave.sin() } * time;
    let damping =
        eta * if continuum {
            wave * wave
        } else {
            4.0 * (wave / 2.0).sin().powi(2)
        } * time;
    (field
        .bz
        .iter()
        .enumerate()
        .map(|(index, value)| {
            (value - (-damping).exp() * (phase(index, shape, mode) - rotation).cos()).powi(2)
        })
        .sum::<f64>()
        / cells as f64)
        .sqrt()
}

#[test]
fn prescribed_velocity_temporal_and_spatial_orders_discriminate() {
    let mut temporal_errors = Vec::new();
    for steps in [8, 16, 32, 64] {
        let timestep = 3.2 / steps as f64;
        let error = uniform_error(16, timestep, steps, 1.0, 0.03, false);
        println!("temporal steps={steps} dt={timestep:.17e} error={error:.17e}");
        assert!(error > 1e-12);
        temporal_errors.push(error);
    }
    for errors in temporal_errors.windows(2) {
        let ratio = errors[0] / errors[1];
        println!("temporal ratio={ratio:.17e}");
        assert!((7.0..9.0).contains(&ratio));
    }
    let mut spatial_errors = Vec::new();
    for cells in [16, 32, 64] {
        let coarse_time = uniform_error(cells, 0.002, 250, 0.3 * cells as f64, 0.0, true);
        let fine_time = uniform_error(cells, 0.001, 500, 0.3 * cells as f64, 0.0, true);
        let temporal_sensitivity = uniform_error(cells, 0.002, 250, 0.3 * cells as f64, 0.0, false)
            + uniform_error(cells, 0.001, 500, 0.3 * cells as f64, 0.0, false);
        assert!((coarse_time - fine_time).abs() <= temporal_sensitivity + 1e-14);
        println!(
            "spatial cells={cells} error={fine_time:.17e} temporal_sensitivity={temporal_sensitivity:.17e}"
        );
        assert!(temporal_sensitivity < 0.01 * fine_time);
        spatial_errors.push(fine_time);
    }
    for errors in spatial_errors.windows(2) {
        let ratio = errors[0] / errors[1];
        println!("spatial ratio={ratio:.17e}");
        assert!((3.5..4.5).contains(&ratio));
    }
}

#[test]
fn shear_energy_growth_matches_physical_stretching() {
    let mut field = MhdField::new(
        3,
        16,
        1,
        MhdConfig {
            dt_mhd: 0.25,
            ..MhdConfig::default()
        },
    );
    field.by.fill(1.0);
    let wave = TAU / 16.0;
    let speed = 0.2;
    let velocity: Vec<_> = (0..48)
        .map(|index| [speed * (wave * (index / 3) as f64).sin(), 0.0, 0.0])
        .collect();
    assert!(
        field
            .validate_uniform_transverse_ssp_rk3(&velocity)
            .is_err()
    );
    let initial_energy = field.magnetic_energy();
    for _ in 0..16 {
        advance(&mut field, &velocity);
    }
    let mut error = 0.0_f64;
    for index in 0..48 {
        let expected = speed * wave.sin() * 4.0 * (wave * (index / 3) as f64).cos();
        error = error.max((field.bx[index] - expected).abs());
        assert!((field.by[index] - 1.0).abs() < 1e-12);
    }
    println!(
        "shear error={error:.17e} energy_ratio={:.17e}",
        field.magnetic_energy() / initial_energy
    );
    assert!(error < 1e-12);
    assert!(field.magnetic_energy() > initial_energy);
}

#[test]
fn mixed_stability_and_atomicity_controls_reject_failures() {
    for (damping, rotation, stable) in [
        (0.0, 1.0, true),
        (0.0, 2.0, false),
        (2.0, 0.0, true),
        (0.0, 1.5, true),
        (2.0, 1.5, false),
    ] {
        let gain = polynomial(-damping, rotation);
        let expected = gain[0].powi(2) + gain[1].powi(2);
        let observed = ssp_rk3_amplification_squared(damping, rotation).unwrap();
        println!("scalar d={damping} a={rotation} gain_squared={observed:.17e} stable={stable}");
        assert!((observed - expected).abs() < 1e-14);
        assert_eq!(observed <= 1.0, stable);
    }
    let field = MhdField::new(16, 1, 1, MhdConfig::default());
    assert!(
        field
            .validate_uniform_transverse_ssp_rk3(&[[1.0, 0.0, 0.0]; 16])
            .is_ok()
    );
    assert!(
        field
            .validate_uniform_transverse_ssp_rk3(&[[2.0, 0.0, 0.0]; 16])
            .is_err()
    );
    for invalid in [f64::NAN, f64::INFINITY] {
        let mut field = initialize([16, 1, 1], [1, 0, 0], MhdConfig::default());
        let before: Vec<_> = field
            .bx
            .iter()
            .chain(&field.by)
            .chain(&field.bz)
            .chain(&field.psi)
            .map(|value| value.to_bits())
            .collect();
        let mut velocity = vec![[0.1, 0.0, 0.0]; 16];
        velocity[4][1] = invalid;
        assert!(
            field
                .try_evolve_b_field_with_integrator(&velocity, MhdIntegrator::SspRk3)
                .is_err()
        );
        let after: Vec<_> = field
            .bx
            .iter()
            .chain(&field.by)
            .chain(&field.bz)
            .chain(&field.psi)
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(before, after);
    }
}

#[test]
fn explicit_legacy_selection_preserves_default_bits() {
    let config = MhdConfig {
        eta: 0.1,
        cleaning_rate: 0.1,
        ..MhdConfig::default()
    };
    let mut default = initialize([9, 5, 3], [2, 1, 0], config.clone());
    let mut explicit = initialize([9, 5, 3], [2, 1, 0], config);
    let velocity = vec![[0.2, -0.1, 0.0]; default.bx.len()];
    for step in 1..=4 {
        default.try_evolve_b_field(&velocity).unwrap();
        explicit
            .try_evolve_b_field_with_integrator(&velocity, MhdIntegrator::LegacyEuler)
            .unwrap();
        let left = default
            .bx
            .iter()
            .chain(&default.by)
            .chain(&default.bz)
            .chain(&default.psi);
        let right = explicit
            .bx
            .iter()
            .chain(&explicit.by)
            .chain(&explicit.bz)
            .chain(&explicit.psi);
        for (left, right) in left.zip(right) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
        println!("explicit_legacy step={step} component_bits=540");
    }
}

#[test]
fn cyclic_axes_check_every_magnetic_component() {
    let shape = [9, 5, 7];
    for direction in 0..3 {
        let polarization = (direction + 1) % 3;
        let mut mode = [0; 3];
        mode[direction] = 1;
        let mut field = initialize(
            shape,
            mode,
            MhdConfig {
                eta: 0.07,
                ..MhdConfig::default()
            },
        );
        if polarization == 0 {
            std::mem::swap(&mut field.bx, &mut field.bz);
        }
        if polarization == 1 {
            std::mem::swap(&mut field.by, &mut field.bz);
        }
        let mut flow = [0.0; 3];
        flow[direction] = 0.3;
        let velocity = vec![flow; field.bx.len()];
        let wave = TAU / shape[direction] as f64;
        let gain = polynomial(-0.28 * (wave / 2.0).sin().powi(2), -0.3 * wave.sin());
        let mut expected_gain = [1.0, 0.0];
        for step in 1..=32 {
            advance(&mut field, &velocity);
            expected_gain = multiply(expected_gain, gain);
            if [1, 2, 8, 32].contains(&step) {
                let mut error = 0.0_f64;
                for (component, values) in [&field.bx, &field.by, &field.bz].into_iter().enumerate()
                {
                    for (index, value) in values.iter().enumerate() {
                        let angle = phase(index, shape, mode);
                        let expected = if component == polarization {
                            expected_gain[0] * angle.cos() - expected_gain[1] * angle.sin()
                        } else {
                            0.0
                        };
                        error = error.max((value - expected).abs());
                    }
                }
                println!(
                    "cyclic direction={direction} polarization={polarization} step={step} all_component_error={error:.17e}"
                );
                assert!(error < 1e-12);
            }
        }
    }
}

#[test]
fn second_stage_overflow_preserves_committed_fields() {
    let mut field = initialize([16, 1, 1], [1, 0, 0], MhdConfig::default());
    let before: Vec<_> = field
        .bx
        .iter()
        .chain(&field.by)
        .chain(&field.bz)
        .chain(&field.psi)
        .map(|value| value.to_bits())
        .collect();
    // The first cross product and Euler stage are finite; the next product exceeds binary64.
    let velocity = vec![[1e200, 0.0, 0.0]; 16];
    let result = field.try_evolve_b_field_with_integrator(&velocity, MhdIntegrator::SspRk3);
    assert!(result.is_err());
    let after: Vec<_> = field
        .bx
        .iter()
        .chain(&field.by)
        .chain(&field.bz)
        .chain(&field.psi)
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(before, after);
    println!(
        "second_stage_overflow rejected=true preserved_components={}",
        before.len()
    );
}
