use qgp_scaling::{
    directional_path::{TransverseDensityGrid, path_eccentricity},
    glauber::{SigmaNN, directional_density_grid},
    nucleus::NucleusParams,
};
use std::f64::consts::{FRAC_PI_2, PI};

fn rectangle(
    columns: usize,
    rows: usize,
    scale: f64,
    production: f64,
    medium: f64,
) -> TransverseDensityGrid {
    TransverseDensityGrid::new(
        columns,
        rows,
        -scale,
        -2.0 * scale,
        2.0 * scale / columns as f64,
        4.0 * scale / rows as f64,
        vec![production; columns * rows],
        vec![medium; columns * rows],
    )
    .unwrap()
}

#[test]
fn joint_rectangle_oracle_discriminates_factor_normalization_and_moment_estimators() {
    let mut previous_error: Option<f64> = None;
    for count in [8, 16, 32] {
        let grid = rectangle(count, count, 1.0, 3.0, 7.0);
        let in_plane = grid.directional_moments(0.0).unwrap();
        let out_of_plane = grid.directional_moments(FRAC_PI_2).unwrap();
        let correction = 1.0 / (3.0 * (count * count) as f64);
        assert!((in_plane.length_fm - (4.0 / 3.0 - correction)).abs() < 1e-12);
        assert!((out_of_plane.length_fm - (8.0 / 3.0 - 2.0 * correction)).abs() < 1e-12);
        assert!((in_plane.denominator - 4.0 * 3.0 * 7.0 * 2.0).abs() < 1e-12);
        let eccentricity = path_eccentricity(&in_plane, &out_of_plane).unwrap();
        assert!((eccentricity - 1.0 / 3.0).abs() < 1e-12);
        assert!((eccentricity - 3.0 / 5.0).abs() > 0.25);
        let error = 4.0 / 3.0 - in_plane.length_fm;
        if let Some(previous) = previous_error {
            assert!((previous / error - 4.0_f64).abs() < 1e-8);
        }
        previous_error = Some(error);
    }
}

#[test]
fn density_and_coordinate_scalings_preserve_the_declared_ratios() {
    let baseline = rectangle(16, 24, 1.0, 1.0, 1.0)
        .directional_moments(0.0)
        .unwrap();
    for (scale, production, medium) in [
        (1.0, 2.0, 1.0),
        (1.0, 1.0, 7.0),
        (1.0, 2.0, 7.0),
        (3.0, 1.0, 1.0),
    ] {
        let observed = rectangle(16, 24, scale, production, medium)
            .directional_moments(0.0)
            .unwrap();
        assert!((observed.length_fm - scale * baseline.length_fm).abs() < 1e-12);
        assert!(
            (observed.denominator / baseline.denominator - production * medium * scale.powi(3))
                .abs()
                < 1e-12
        );
        assert!(
            (observed.numerator / baseline.numerator - production * medium * scale.powi(4)).abs()
                < 1e-12
        );
    }
}

#[test]
fn axis_exchange_reverses_eccentricity_and_forward_support_is_required() {
    let original = rectangle(16, 16, 1.0, 1.0, 1.0);
    let exchanged = TransverseDensityGrid::new(
        16,
        16,
        -2.0,
        -1.0,
        0.25,
        0.125,
        vec![1.0; 256],
        vec![1.0; 256],
    )
    .unwrap();
    let original_x = original.directional_moments(0.0).unwrap();
    let original_y = original.directional_moments(FRAC_PI_2).unwrap();
    let exchanged_x = exchanged.directional_moments(0.0).unwrap();
    let exchanged_y = exchanged.directional_moments(FRAC_PI_2).unwrap();
    assert!((original_x.length_fm - exchanged_y.length_fm).abs() < 1e-12);
    assert!((original_y.length_fm - exchanged_x.length_fm).abs() < 1e-12);
    assert!(
        (path_eccentricity(&original_x, &original_y).unwrap()
            + path_eccentricity(&exchanged_x, &exchanged_y).unwrap())
        .abs()
            < 1e-12
    );
    let disjoint =
        TransverseDensityGrid::new(2, 1, 0.0, 0.0, 1.0, 1.0, vec![0.0, 1.0], vec![1.0, 0.0])
            .unwrap();
    assert!(disjoint.directional_moments(0.0).is_err());
    assert!(disjoint.directional_moments(PI).is_ok());
}

#[test]
fn unequal_fields_use_global_moments_and_reflect_with_the_ray() {
    let forward =
        TransverseDensityGrid::new(2, 1, 0.0, 0.0, 1.0, 1.0, vec![1.0, 3.0], vec![2.0, 5.0])
            .unwrap();
    let reverse =
        TransverseDensityGrid::new(2, 1, 0.0, 0.0, 1.0, 1.0, vec![3.0, 1.0], vec![5.0, 2.0])
            .unwrap();
    let observed = forward.directional_moments(0.0).unwrap();
    let reflected = reverse.directional_moments(PI).unwrap();
    assert_eq!(observed.denominator, 13.5);
    assert_eq!(observed.numerator, 7.125);
    assert!((observed.length_fm - 19.0 / 18.0).abs() < 1e-14);
    assert!((observed.length_fm - 0.8125).abs() > 0.2);
    assert!((reflected.length_fm - observed.length_fm).abs() < 1e-14);
}

#[test]
fn cell_traversal_matches_independent_all_cell_ray_intersections() {
    let columns = 4;
    let rows = 3;
    let spacing = [0.7, 1.1];
    let production: Vec<_> = (0..columns * rows)
        .map(|index| (index % 3) as f64)
        .collect();
    let medium: Vec<_> = (0..columns * rows)
        .map(|index| (index + 1) as f64 / 7.0)
        .collect();
    let grid = TransverseDensityGrid::new(
        columns,
        rows,
        0.0,
        0.0,
        spacing[0],
        spacing[1],
        production.clone(),
        medium.clone(),
    )
    .unwrap();
    for angle in [0.37_f64, 1.2, 2.3, -0.8] {
        let direction = [angle.cos(), angle.sin()];
        let mut denominator = 0.0;
        let mut numerator = 0.0;
        for (origin_index, origin_density) in production.iter().enumerate() {
            let origin = [
                ((origin_index % columns) as f64 + 0.5) * spacing[0],
                ((origin_index / columns) as f64 + 0.5) * spacing[1],
            ];
            for (medium_index, medium_density) in medium.iter().enumerate() {
                let indices = [medium_index % columns, medium_index / columns];
                let mut enter = 0.0_f64;
                let mut exit = f64::INFINITY;
                for axis in 0..2 {
                    let first =
                        (indices[axis] as f64 * spacing[axis] - origin[axis]) / direction[axis];
                    let second = ((indices[axis] + 1) as f64 * spacing[axis] - origin[axis])
                        / direction[axis];
                    enter = enter.max(first.min(second));
                    exit = exit.min(first.max(second));
                }
                if exit > enter {
                    let weight = origin_density * medium_density * spacing[0] * spacing[1];
                    denominator += weight * (exit - enter);
                    numerator += weight * (exit.powi(2) - enter.powi(2)) / 2.0;
                }
            }
        }
        let actual = grid.directional_moments(angle).unwrap();
        assert!((actual.denominator - denominator).abs() < 1e-12);
        assert!((actual.numerator - numerator).abs() < 1e-12);
    }
}

#[test]
fn hard_sphere_fixed_impact_grid_sensitivity_is_reported_separately() {
    let nucleus = NucleusParams::pb208();
    let sigma = SigmaNN::lhc_5020();
    println!("impact_parameter_fm,grid,Lx_fm,Ly_fm,path_eccentricity,Dx_per_fm,Nx,Dy_per_fm,Ny");
    for impact_parameter in [0.0, 8.0] {
        let mut results = Vec::new();
        for count in [64, 128, 256] {
            let grid = directional_density_grid(
                impact_parameter,
                &sigma,
                &nucleus,
                &nucleus,
                count,
                count,
            )
            .unwrap();
            let in_plane = grid.directional_moments(0.0).unwrap();
            let out_of_plane = grid.directional_moments(FRAC_PI_2).unwrap();
            let eccentricity = path_eccentricity(&in_plane, &out_of_plane).unwrap();
            println!(
                "{impact_parameter},{count},{:.17},{:.17},{eccentricity:.17},{:.17},{:.17},{:.17},{:.17}",
                in_plane.length_fm,
                out_of_plane.length_fm,
                in_plane.denominator,
                in_plane.numerator,
                out_of_plane.denominator,
                out_of_plane.numerator
            );
            if impact_parameter == 0.0 {
                assert!(eccentricity.abs() < 1e-12);
            }
            results.push((in_plane.length_fm, out_of_plane.length_fm, eccentricity));
        }
        let (coarse, fine) = (results[1], results[2]);
        assert!((fine.0 - coarse.0).abs() / fine.0 < 0.005);
        assert!((fine.1 - coarse.1).abs() / fine.1 < 0.005);
        assert!((fine.2 - coarse.2).abs() < 0.001);
    }
    assert!(directional_density_grid(f64::NAN, &sigma, &nucleus, &nucleus, 8, 8).is_err());
    let separated =
        directional_density_grid(4.0 * nucleus.r_a, &sigma, &nucleus, &nucleus, 16, 16).unwrap();
    assert!(separated.directional_moments(0.0).is_err());
}
