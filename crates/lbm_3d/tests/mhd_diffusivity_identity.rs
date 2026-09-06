use lbm_3d::{
    mhd::{MagneticDiffusivity, MhdConfig, MhdField},
    units::{LatticeUnits, UniformCartesianMesh},
};
use std::f64::consts::TAU;

include!("../../../data/output/audit/mhd-magnetic-diffusivity/retained-step.rs");

fn bits(field: &MhdField) -> Vec<u64> {
    field
        .bx
        .iter()
        .chain(&field.by)
        .chain(&field.bz)
        .chain(&field.psi)
        .map(|value| value.to_bits())
        .collect()
}

fn field_copy(field: &MhdField) -> MhdField {
    let mut copy = MhdField::new(field.nx, field.ny, field.nz, MhdConfig::default());
    copy.config = field.config.clone();
    copy.bx.clone_from(&field.bx);
    copy.by.clone_from(&field.by);
    copy.bz.clone_from(&field.bz);
    copy.psi.clone_from(&field.psi);
    copy
}

fn mode_field(shape: [usize; 3], mode: [usize; 3], diffusivity: f64, timestep: f64) -> MhdField {
    let [nx, ny, nz] = shape;
    let mut field = MhdField::new(
        nx,
        ny,
        nz,
        MhdConfig {
            eta: diffusivity,
            dt_mhd: timestep,
            ..MhdConfig::default()
        },
    );
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let phase: f64 = [x, y, z]
                    .into_iter()
                    .zip(mode)
                    .zip(shape)
                    .map(|((position, frequency), size)| {
                        TAU * position as f64 * frequency as f64 / size as f64
                    })
                    .sum();
                let index = z * nx * ny + y * nx + x;
                field.by[index] = phase.cos();
                if mode[1] != 0 {
                    field.bx[index] = -phase.cos();
                }
            }
        }
    }
    field
}

#[test]
fn evolving_fourier_modes_match_amplitude_and_energy() {
    for (shape, mode, diffusivity, timestep) in [
        ([16, 4, 3], [1, 0, 0], 0.1, 0.5),
        ([16, 4, 3], [3, 0, 0], 0.1, 0.5),
        ([8, 8, 8], [1, 1, 0], 0.125, 1.0),
        ([8, 8, 8], [4, 4, 4], 0.125, 1.0),
        ([16, 1, 1], [3, 0, 0], 0.4, 1.0),
        ([16, 4, 3], [1, 0, 0], 0.0, 1.0),
    ] {
        let mut field = mode_field(shape, mode, diffusivity, timestep);
        let initial = field_copy(&field);
        let initial_energy = field.magnetic_energy();
        let zero_velocity = vec![[0.0; 3]; field.bx.len()];
        let eigen_sum: f64 = mode
            .into_iter()
            .zip(shape)
            .map(|(frequency, size)| {
                (std::f64::consts::PI * frequency as f64 / size as f64)
                    .sin()
                    .powi(2)
            })
            .sum();
        let amplification = 1.0 - 4.0 * diffusivity * timestep * eigen_sum;
        for step in 1..=32_i32 {
            field.try_evolve_b_field(&zero_velocity).unwrap();
            if [1, 2, 8, 32].contains(&step) {
                let factor = amplification.powi(step);
                let maximum_error = field
                    .bx
                    .iter()
                    .chain(&field.by)
                    .chain(&field.bz)
                    .zip(initial.bx.iter().chain(&initial.by).chain(&initial.bz))
                    .map(|(actual, start)| (actual - start * factor).abs())
                    .fold(0.0_f64, f64::max);
                let energy_error =
                    (field.magnetic_energy() / initial_energy - amplification.powi(2 * step)).abs();
                println!(
                    "mode shape={shape:?} k={mode:?} eta={diffusivity} dt={timestep} step={step} g={amplification:.17e} amplitude_error={maximum_error:.17e} energy_error={energy_error:.17e}"
                );
                assert!(maximum_error < 1e-12);
                assert!(energy_error < 1e-12);
            }
        }
    }
}

#[test]
fn admitted_states_match_retained_arithmetic_bitwise() {
    for diffusivity in [0.0, 0.01, 0.1] {
        for cleaning in [0.0, 0.1] {
            let mut field = MhdField::new(
                3,
                4,
                5,
                MhdConfig {
                    eta: diffusivity,
                    cleaning_rate: cleaning,
                    ..MhdConfig::default()
                },
            );
            let mut velocity = Vec::new();
            for index in 0..field.bx.len() {
                let phase = index as f64 * 0.13;
                field.bx[index] = phase.sin();
                field.by[index] = (phase * 0.7).cos();
                field.bz[index] = (phase * 1.1).sin();
                field.psi[index] = phase * 0.01;
                velocity.push([0.01 * phase.cos(), 0.02 * phase.sin(), -0.03 * phase.cos()]);
            }
            let mut retained = field_copy(&field);
            for step in 1..=4 {
                retained_step(&mut retained, &velocity);
                field.try_evolve_b_field(&velocity).unwrap();
                assert_eq!(bits(&field), bits(&retained));
                println!(
                    "retained bitwise eta={diffusivity} cleaning={cleaning} step={step} components={}",
                    bits(&field).len()
                );
            }
        }
    }
}

#[test]
fn invalid_updates_preserve_fields_and_retained_failures() {
    for diffusivity in [-0.1, f64::NAN, f64::INFINITY, 0.2] {
        let mut field = mode_field([8, 8, 8], [4, 4, 4], 0.0, 1.0);
        field.config.eta = diffusivity;
        let before = bits(&field);
        let velocity = vec![[0.0; 3]; field.bx.len()];
        assert!(field.try_evolve_b_field(&velocity).is_err());
        assert_eq!(before, bits(&field));
        if diffusivity < 0.0 || diffusivity.is_nan() {
            retained_step(&mut field, &velocity);
            assert_eq!(before, bits(&field));
            println!(
                "retained invalid eta={diffusivity} silently_skips_diffusion=true corrected_rejects=true"
            );
        } else if diffusivity == 0.2 {
            let energy = field.magnetic_energy();
            retained_step(&mut field, &velocity);
            let ratio = field.magnetic_energy() / energy;
            println!(
                "retained unstable diffusion energy_multiplier={ratio:.17e} corrected_rejects=true"
            );
            assert!((ratio - 1.96).abs() < 1e-12);
        }
    }
    let mut field = mode_field([16, 1, 1], [1, 0, 0], 0.1, 1.0);
    for timestep in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        field.config.dt_mhd = timestep;
        let before = bits(&field);
        assert!(field.try_evolve_b_field(&vec![[0.0; 3]; 16]).is_err());
        assert_eq!(before, bits(&field));
    }
    field.config.dt_mhd = 1.0;
    let before = bits(&field);
    assert!(field.try_evolve_b_field(&[[0.0; 3]; 15]).is_err());
    assert_eq!(before, bits(&field));
    field.bx[0] = f64::NAN;
    let before = bits(&field);
    assert!(field.try_evolve_b_field(&[[0.0; 3]; 16]).is_err());
    assert_eq!(before, bits(&field));
    field.bx.fill(1e308);
    field.by.fill(1e308);
    let before = bits(&field);
    assert!(field.try_evolve_b_field(&[[1e308, 0.0, 0.0]; 16]).is_err());
    assert_eq!(before, bits(&field));
}

#[test]
fn si_diffusivity_preserves_the_dimensionless_transport_number() {
    for (spacing, timestep) in [(1e6, 0.125), (2e6, 0.5), (1e6, 0.25)] {
        let mesh = UniformCartesianMesh::new([16, 4, 3], [0.0; 3], spacing).unwrap();
        let units = LatticeUnits::new(&mesh, timestep, 1e-20).unwrap();
        let physical = 4e8;
        let diffusivity = MagneticDiffusivity::from_si(physical, &units).unwrap();
        let expected = physical * timestep / (spacing * spacing);
        assert!((diffusivity.lattice_value() / expected - 1.0).abs() < 1e-14);
        println!(
            "SI eta_m2_s={physical} dx_m={spacing} dt_s={timestep} eta_lattice={:.17e}",
            diffusivity.lattice_value()
        );
    }
}

#[test]
fn admitted_diffusion_does_not_certify_centered_induction_stability() {
    let mut field = mode_field([16, 1, 1], [1, 0, 0], 0.0, 1.0);
    let before = field.magnetic_energy();
    field.try_evolve_b_field(&[[0.1, 0.0, 0.0]; 16]).unwrap();
    let expected = 1.0 + (0.1 * (TAU / 16.0).sin()).powi(2);
    let observed = field.magnetic_energy() / before;
    println!(
        "centered induction eta0 expected_energy_multiplier={expected:.17e} observed={observed:.17e}"
    );
    assert!(observed > 1.0);
    assert!((observed - expected).abs() < 1e-12);
}
